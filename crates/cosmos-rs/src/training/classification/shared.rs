//! Machinery both classification adapters share.
//!
//! The interior chain — its device-resident forward cache, its forward and its
//! backward — plus the anchor-pooling helpers that turn per-anchor geodesic
//! angles into the per-class quantities the losses consume.

use std::str::FromStr;

use ndarray::Array2;

use crate::{
    arrangement::{
        ArrangementError,
        activations::{Activation, ActivationPlacement},
        blocks::BlockConfig,
        compositions::CompositionStrategy,
        orderings::{OrderingContext, OrderingStrategy},
        params::SubspaceAction,
        selections::{SelectionContext, SelectionStrategy},
    },
    compute::device::{Buffer2, ComputeDevice, Device},
    compute::{projector_backward_projected_device, projector_backward_subspace_device},
    objectives::utils::Targets,
    state::cosmos_state::CosmosState,
    training::{BlockGradients, TrainError, backend::TrainingBackend, trace::BlockSelectionTrace},
};

use super::DEFAULT_EPS;

pub(super) struct ProjectorApplicationTraceDevice {
    pub(super) block_idx: usize,
    pub(super) input: Buffer2<f64>,
    pub(super) subspace: Buffer2<f64>,
    pub(super) projected: Buffer2<f64>,
    /// Coordinates handed to `Wᵀ` on the way out of the subspace placement.
    pub(super) activated_subspace: Option<Buffer2<f64>>,
    pub(super) output: Buffer2<f64>,
    pub(super) activation: Activation,
    pub(super) placement: ActivationPlacement,
    pub(super) action: SubspaceAction,
}

impl std::fmt::Debug for ProjectorApplicationTraceDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectorApplicationTraceDevice")
            .field("block_idx", &self.block_idx)
            .field("input_dim", &self.input.dim())
            .field("output_dim", &self.output.dim())
            .finish()
    }
}

pub(super) enum BlockForwardKindDevice {
    Passthrough,
    Series {
        steps: Vec<ProjectorApplicationTraceDevice>,
        residual: bool,
    },
    Parallel {
        branches: Vec<ProjectorApplicationTraceDevice>,
    },
    /// Two-phase composition: parallel mean-pool of `branches` over the block
    /// input, then a series chain of `steps` over the pooled result. Both
    /// phases use the same projector list.
    ParallelSeries {
        branches: Vec<ProjectorApplicationTraceDevice>,
        steps: Vec<ProjectorApplicationTraceDevice>,
    },
}

impl std::fmt::Debug for BlockForwardKindDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Passthrough => write!(f, "Passthrough"),
            Self::Series { steps, residual } => f
                .debug_struct("Series")
                .field("steps", &steps.len())
                .field("residual", residual)
                .finish(),
            Self::Parallel { branches } => f
                .debug_struct("Parallel")
                .field("branches", &branches.len())
                .finish(),
            Self::ParallelSeries { branches, steps } => f
                .debug_struct("ParallelSeries")
                .field("branches", &branches.len())
                .field("steps", &steps.len())
                .finish(),
        }
    }
}

pub(super) struct ManualBlockForwardDevice {
    pub(super) input: Buffer2<f64>,
    pub(super) output: Buffer2<f64>,
    pub(super) selection: BlockSelectionTrace,
    pub(super) kind: BlockForwardKindDevice,
}

impl std::fmt::Debug for ManualBlockForwardDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManualBlockForwardDevice")
            .field("input_dim", &self.input.dim())
            .field("output_dim", &self.output.dim())
            .field("kind", &self.kind)
            .finish()
    }
}

/// Device-resident forward cache of one interior chain.
pub(crate) struct InteriorForwardDevice {
    pub(super) input: Buffer2<f64>,
    pub(crate) output: Buffer2<f64>,
    pub(super) blocks: Vec<ManualBlockForwardDevice>,
    pub(super) global_activation_input: Option<(Activation, Buffer2<f64>)>,
    pub(super) normalized_input: Option<Buffer2<f64>>,
}

impl std::fmt::Debug for InteriorForwardDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteriorForwardDevice")
            .field("input_dim", &self.input.dim())
            .field("output_dim", &self.output.dim())
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

/// Upload a state basis to the device.
pub(crate) fn upload_basis(device: &Device, basis: &Array2<f64>) -> Buffer2<f64> {
    device.upload_2d(basis)
}
pub(super) fn forward_projector_application_device(
    device: &Device,
    inputs: &Buffer2<f64>,
    block_idx: usize,
    basis: &Buffer2<f64>,
    activation: Activation,
    placement: ActivationPlacement,
    action: SubspaceAction,
) -> ProjectorApplicationTraceDevice {
    let c = action.action_coeff();
    let d = action.identity_coeff();

    let subspace = device.matmul(inputs, basis);
    let basis_t = device.transpose(basis);
    let projected_raw = device.matmul(&subspace, &basis_t);

    let (activated_subspace, projected, output) = match placement {
        ActivationPlacement::None => {
            // T(x) = d·x + c·P(x)
            let mut out = device.clone_buffer(&projected_raw);
            device.scale_inplace(&mut out, c);
            device.add_scaled(&mut out, d, inputs);
            (None, projected_raw, out)
        }
        ActivationPlacement::Post => {
            // T(x) = σ(d·x + c·P(x)); the pre-activation value is kept in
            // `projected` for the backward σ' evaluation.
            let mut pre_act = device.clone_buffer(&projected_raw);
            device.scale_inplace(&mut pre_act, c);
            device.add_scaled(&mut pre_act, d, inputs);
            let mut out = device.clone_buffer(&pre_act);
            if activation != Activation::Identity {
                device.apply_activation(&mut out, activation);
            }
            (None, pre_act, out)
        }
        ActivationPlacement::Subspace => {
            // T(x) = d·x + c·φ(xW)Wᵀ
            let mut activated = device.clone_buffer(&subspace);
            if activation != Activation::Identity {
                device.apply_activation(&mut activated, activation);
            }
            let f_sigma = device.matmul(&activated, &basis_t);
            let mut out = device.clone_buffer(&f_sigma);
            device.scale_inplace(&mut out, c);
            device.add_scaled(&mut out, d, inputs);
            (Some(activated), projected_raw, out)
        }
    };

    ProjectorApplicationTraceDevice {
        block_idx,
        input: device.clone_buffer(inputs),
        subspace,
        projected,
        activated_subspace,
        output,
        activation,
        placement,
        action,
    }
}

/// Look up the runtime [`SubspaceAction`] for a projector given its global
/// block index.
///
/// `runtime_actions` is indexed by position in `config.block_indices` — the
/// same layout `ModularBlock::composition_context` uses.
pub(super) fn resolve_projector_action(
    block_idx: usize,
    config: &BlockConfig,
    runtime_actions: &[SubspaceAction],
    fallback: SubspaceAction,
) -> SubspaceAction {
    config
        .block_indices
        .iter()
        .position(|&bi| bi == block_idx)
        .and_then(|pos| runtime_actions.get(pos).copied())
        .unwrap_or(fallback)
}

pub(super) fn forward_supported_block_device(
    inputs: &Buffer2<f64>,
    config: &BlockConfig,
    runtime_actions: &[SubspaceAction],
    state: &CosmosState,
    backend: &mut impl TrainingBackend,
) -> Result<ManualBlockForwardDevice, TrainError> {
    let candidate_indices = config.block_indices.clone();
    let selected_indices = SelectionStrategy::from_str(&config.selection.strategy)?
        .apply(
            &candidate_indices,
            &SelectionContext::default(),
            &config.selection,
        )?
        .0;
    let ordered_indices = OrderingStrategy::from_str(&config.ordering.strategy)?
        .apply(&selected_indices, &OrderingContext, &config.ordering)?
        .0;

    let device = backend.device();

    if ordered_indices.is_empty() {
        return Ok(ManualBlockForwardDevice {
            input: device.clone_buffer(inputs),
            output: device.clone_buffer(inputs),
            selection: BlockSelectionTrace {
                candidate_indices,
                selected_indices,
                ordered_indices,
            },
            kind: BlockForwardKindDevice::Passthrough,
        });
    }

    let activation = Activation::from_str(&config.composition.activation_name)
        .map_err(ArrangementError::from)?;
    let placement = ActivationPlacement::from_str(&config.composition.activation_placement)
        .map_err(ArrangementError::from)?;
    let composition = CompositionStrategy::from_str(&config.composition.strategy)?;
    let fallback_action = SubspaceAction::from_str(&config.composition.subspace_action)?;

    let (kind, output) = match composition {
        CompositionStrategy::Series | CompositionStrategy::ResidualSeries => {
            let mut current = device.clone_buffer(inputs);
            let mut steps = Vec::with_capacity(ordered_indices.len());
            for &block_idx in &ordered_indices {
                let action =
                    resolve_projector_action(block_idx, config, runtime_actions, fallback_action);
                let basis_buf = upload_basis(device, state.get_basis(block_idx)?);
                let trace = forward_projector_application_device(
                    device, &current, block_idx, &basis_buf, activation, placement, action,
                );
                current = device.clone_buffer(&trace.output);
                steps.push(trace);
            }

            let residual = composition == CompositionStrategy::ResidualSeries;
            if residual {
                device.add_assign(&mut current, inputs);
            }
            (BlockForwardKindDevice::Series { steps, residual }, current)
        }
        CompositionStrategy::Parallel => {
            let mut branches = Vec::with_capacity(ordered_indices.len());
            let (rows, cols) = inputs.dim();
            let mut output: Buffer2<f64> = device.zeros_2d(rows, cols);

            for &block_idx in &ordered_indices {
                let action =
                    resolve_projector_action(block_idx, config, runtime_actions, fallback_action);
                let basis_buf = upload_basis(device, state.get_basis(block_idx)?);
                let trace = forward_projector_application_device(
                    device, inputs, block_idx, &basis_buf, activation, placement, action,
                );
                device.add_assign(&mut output, &trace.output);
                branches.push(trace);
            }

            let scale = 1.0 / ordered_indices.len() as f64;
            device.scale_inplace(&mut output, scale);

            (BlockForwardKindDevice::Parallel { branches }, output)
        }
        CompositionStrategy::ParallelSeries => {
            // Pooling stage: parallel mean-pool of every branch over the shared
            // block input — mirrors the host `apply_parallel` exactly.
            let mut branches = Vec::with_capacity(ordered_indices.len());
            let (rows, cols) = inputs.dim();
            let mut step: Buffer2<f64> = device.zeros_2d(rows, cols);

            for &block_idx in &ordered_indices {
                let action =
                    resolve_projector_action(block_idx, config, runtime_actions, fallback_action);
                let basis_buf = upload_basis(device, state.get_basis(block_idx)?);
                let trace = forward_projector_application_device(
                    device, inputs, block_idx, &basis_buf, activation, placement, action,
                );
                device.add_assign(&mut step, &trace.output);
                branches.push(trace);
            }

            let scale = 1.0 / ordered_indices.len() as f64;
            device.scale_inplace(&mut step, scale);

            // Series stage: chain over the pooled step with the same projector
            // list — mirrors the host `apply_series`.
            let mut current = step;
            let mut steps = Vec::with_capacity(ordered_indices.len());
            for &block_idx in &ordered_indices {
                let action =
                    resolve_projector_action(block_idx, config, runtime_actions, fallback_action);
                let basis_buf = upload_basis(device, state.get_basis(block_idx)?);
                let trace = forward_projector_application_device(
                    device, &current, block_idx, &basis_buf, activation, placement, action,
                );
                current = device.clone_buffer(&trace.output);
                steps.push(trace);
            }

            (
                BlockForwardKindDevice::ParallelSeries { branches, steps },
                current,
            )
        }
    };

    Ok(ManualBlockForwardDevice {
        input: device.clone_buffer(inputs),
        output: device.clone_buffer(&output),
        selection: BlockSelectionTrace {
            candidate_indices,
            selected_indices,
            ordered_indices,
        },
        kind,
    })
}

/// Device forward for the supported interior chain.
///
/// Returns `None` when the interior is empty, has an identity global activation
/// and does not normalize its output — the case in which the embeddings equal
/// the projected inputs.
///
/// # Errors
///
/// Returns [`TrainError`] when a block references a missing state block or
/// carries an unparsable strategy name.
pub(crate) fn forward_supported_interior_device(
    block_configs: &[BlockConfig],
    block_actions: &[Vec<SubspaceAction>],
    global_activation: Activation,
    normalize_final: bool,
    state: &CosmosState,
    inputs: &Buffer2<f64>,
    backend: &mut impl TrainingBackend,
) -> Result<Option<InteriorForwardDevice>, TrainError> {
    if block_configs.is_empty() && global_activation == Activation::Identity && !normalize_final {
        return Ok(None);
    }

    let mut current: Buffer2<f64> = backend.device().clone_buffer(inputs);
    let mut blocks = Vec::with_capacity(block_configs.len());

    for (config, actions) in block_configs.iter().zip(block_actions.iter()) {
        let block = forward_supported_block_device(&current, config, actions, state, backend)?;
        current = backend.device().clone_buffer(&block.output);
        blocks.push(block);
    }

    let device = backend.device();
    let mut global_activation_input = None;
    if global_activation != Activation::Identity {
        global_activation_input = Some((global_activation, device.clone_buffer(&current)));
        device.apply_activation(&mut current, global_activation);
    }

    let mut normalized_input = None;
    if normalize_final {
        normalized_input = Some(device.clone_buffer(&current));
        current = device.normalize_rows(&current);
    }

    Ok(Some(InteriorForwardDevice {
        input: device.clone_buffer(inputs),
        output: current,
        blocks,
        global_activation_input,
        normalized_input,
    }))
}

/// Backward through the supported interior chain.
///
/// Accumulates one gradient per interior state block into `gradients` and
/// returns `∂L/∂x` for the stage upstream of the interior.
///
/// # Errors
///
/// Returns [`TrainError`] when a cached block references a missing state block.
pub(crate) fn backward_supported_interior(
    device: &Device,
    state: &CosmosState,
    interior: Option<&InteriorForwardDevice>,
    grad_embeddings: &Buffer2<f64>,
    gradients: &mut BlockGradients,
) -> Result<Buffer2<f64>, TrainError> {
    let Some(interior) = interior else {
        return Ok(device.clone_buffer(grad_embeddings));
    };

    let mut grad_current = device.clone_buffer(grad_embeddings);

    if let Some(normalized_input) = interior.normalized_input.as_ref() {
        grad_current = backward_normalize_rows_device(device, normalized_input, &grad_current);
    }

    if let Some((activation, ref act_input)) = interior.global_activation_input {
        let derivative = device.activation_derivative(act_input, activation);
        device.elementwise_mul_assign(&mut grad_current, &derivative);
    }

    for block in interior.blocks.iter().rev() {
        grad_current = backward_supported_block(device, block, &grad_current, state, gradients)?;
    }

    Ok(grad_current)
}

pub(super) fn backward_supported_block(
    device: &Device,
    block: &ManualBlockForwardDevice,
    grad_output: &Buffer2<f64>,
    state: &CosmosState,
    gradients: &mut BlockGradients,
) -> Result<Buffer2<f64>, TrainError> {
    match &block.kind {
        BlockForwardKindDevice::Passthrough => Ok(device.clone_buffer(grad_output)),
        BlockForwardKindDevice::Series { steps, residual } => {
            let mut grad_current = device.clone_buffer(grad_output);

            for step in steps.iter().rev() {
                let basis_buf = upload_basis(device, state.get_basis(step.block_idx)?);
                let (grad_input, grad_basis) =
                    backward_projector_application(device, step, &grad_current, &basis_buf);
                accumulate_block_gradient(device, gradients, step.block_idx, grad_basis);
                grad_current = grad_input;
            }

            if *residual {
                device.add_assign(&mut grad_current, grad_output);
            }

            Ok(grad_current)
        }
        BlockForwardKindDevice::Parallel { branches } => {
            if branches.is_empty() {
                return Ok(device.clone_buffer(grad_output));
            }

            let branch_scale = 1.0 / branches.len() as f64;
            let mut branch_grad = device.clone_buffer(grad_output);
            device.scale_inplace(&mut branch_grad, branch_scale);
            let (rows, cols) = block.input.dim();
            let mut grad_input: Buffer2<f64> = device.zeros_2d(rows, cols);

            for branch in branches {
                let basis_buf = upload_basis(device, state.get_basis(branch.block_idx)?);
                let (branch_input, branch_basis) =
                    backward_projector_application(device, branch, &branch_grad, &basis_buf);
                accumulate_block_gradient(device, gradients, branch.block_idx, branch_basis);
                device.add_assign(&mut grad_input, &branch_input);
            }

            Ok(grad_input)
        }
        BlockForwardKindDevice::ParallelSeries { branches, steps } => {
            // Series stage backward: chain gradients down to the pooled step.
            let mut grad_step = device.clone_buffer(grad_output);
            for step in steps.iter().rev() {
                let basis_buf = upload_basis(device, state.get_basis(step.block_idx)?);
                let (grad_input, grad_basis) =
                    backward_projector_application(device, step, &grad_step, &basis_buf);
                accumulate_block_gradient(device, gradients, step.block_idx, grad_basis);
                grad_step = grad_input;
            }

            if branches.is_empty() {
                return Ok(grad_step);
            }

            // Pooling stage backward: every branch receives `grad_step / L` and
            // the input gradients accumulate.
            let branch_scale = 1.0 / branches.len() as f64;
            let mut branch_grad = device.clone_buffer(&grad_step);
            device.scale_inplace(&mut branch_grad, branch_scale);
            let (rows, cols) = block.input.dim();
            let mut grad_input: Buffer2<f64> = device.zeros_2d(rows, cols);

            for branch in branches {
                let basis_buf = upload_basis(device, state.get_basis(branch.block_idx)?);
                let (branch_input, branch_basis) =
                    backward_projector_application(device, branch, &branch_grad, &basis_buf);
                accumulate_block_gradient(device, gradients, branch.block_idx, branch_basis);
                device.add_assign(&mut grad_input, &branch_input);
            }

            Ok(grad_input)
        }
    }
}

pub(super) fn backward_projector_application(
    device: &Device,
    trace: &ProjectorApplicationTraceDevice,
    grad_output: &Buffer2<f64>,
    basis_buf: &Buffer2<f64>,
) -> (Buffer2<f64>, Buffer2<f64>) {
    let c = trace.action.action_coeff();
    let d = trace.action.identity_coeff();

    let (irows, icols) = trace.input.dim();
    let (brows, bcols) = basis_buf.dim();
    let mut grad_input_buf: Buffer2<f64> = device.zeros_2d(irows, icols);
    let mut grad_basis_buf: Buffer2<f64> = device.zeros_2d(brows, bcols);

    // T(x) = d·x + c·f_σ(x), so grad_basis = c · f_σ_grad_basis and
    // grad_input = d · upstream_grad + c · f_σ_grad_input. For `Post` the
    // activation wraps the whole operation, T = σ(d·x + c·P(x)), so the
    // upstream gradient is `grad_output ⊙ σ'(pre_activation)`.
    match trace.placement {
        ActivationPlacement::None => {
            projector_backward_projected_device(
                device,
                &trace.input,
                Some(&trace.subspace),
                grad_output,
                basis_buf,
                None,
                &mut grad_input_buf,
                &mut grad_basis_buf,
            );
            device.scale_inplace(&mut grad_basis_buf, c);
            device.scale_inplace(&mut grad_input_buf, c);
            device.add_scaled(&mut grad_input_buf, d, grad_output);
        }
        ActivationPlacement::Post => {
            let derivative = device.activation_derivative(&trace.projected, trace.activation);
            let effective_grad = device.elementwise_mul(grad_output, &derivative);
            projector_backward_projected_device(
                device,
                &trace.input,
                Some(&trace.subspace),
                &effective_grad,
                basis_buf,
                None,
                &mut grad_input_buf,
                &mut grad_basis_buf,
            );
            device.scale_inplace(&mut grad_basis_buf, c);
            device.scale_inplace(&mut grad_input_buf, c);
            device.add_scaled(&mut grad_input_buf, d, &effective_grad);
        }
        ActivationPlacement::Subspace => {
            let activated = trace
                .activated_subspace
                .as_ref()
                .expect("subspace activation trace must cache activated coordinates");
            let derivative = device.activation_derivative(&trace.subspace, trace.activation);
            projector_backward_subspace_device(
                device,
                &trace.input,
                grad_output,
                basis_buf,
                activated,
                &derivative,
                &mut grad_input_buf,
                &mut grad_basis_buf,
            );
            device.scale_inplace(&mut grad_basis_buf, c);
            device.scale_inplace(&mut grad_input_buf, c);
            device.add_scaled(&mut grad_input_buf, d, grad_output);
        }
    }

    (grad_input_buf, grad_basis_buf)
}

pub(super) fn backward_normalize_rows_device(
    device: &Device,
    inputs: &Buffer2<f64>,
    grad_outputs: &Buffer2<f64>,
) -> Buffer2<f64> {
    let inputs_cpu = device.download_2d(inputs);
    let grad_cpu = device.download_2d(grad_outputs);
    let result = backward_normalize_rows(&inputs_cpu, &grad_cpu);
    upload_basis(device, &result)
}

pub(super) fn accumulate_block_gradient(
    device: &Device,
    gradients: &mut BlockGradients,
    block_idx: usize,
    grad_basis: Buffer2<f64>,
) {
    if let Some(existing) = gradients[block_idx].as_mut() {
        device.add_assign(existing, &grad_basis);
    } else {
        gradients[block_idx] = Some(grad_basis);
    }
}

pub(super) fn backward_normalize_rows(
    inputs: &Array2<f64>,
    grad_outputs: &Array2<f64>,
) -> Array2<f64> {
    let mut grad_inputs = Array2::<f64>::zeros(inputs.dim());

    for row_idx in 0..inputs.nrows() {
        let input = inputs.row(row_idx);
        let grad = grad_outputs.row(row_idx);
        let norm_sq = input.iter().map(|value| value * value).sum::<f64>();
        let norm = norm_sq.sqrt();

        if norm <= DEFAULT_EPS {
            continue;
        }

        let dot = input
            .iter()
            .zip(grad.iter())
            .map(|(x, g)| x * g)
            .sum::<f64>();
        for col_idx in 0..inputs.ncols() {
            grad_inputs[[row_idx, col_idx]] =
                grad[col_idx] / norm - input[col_idx] * dot / norm.powi(3);
        }
    }

    grad_inputs
}

/// Per-(sample, class) geodesic angle of the anchor that won the score
/// max-pool.
///
/// The score max-pool winner is also the minimal-angle anchor, because
/// `s = exp(−θ²)` is strictly decreasing in `θ`. `argmax_anchor` must be
/// populated whenever `anchors_per_class > 1`; for the single-anchor path the
/// pooling is the identity.
///
/// # Panics
///
/// Panics when `anchors_per_class > 1` and `argmax_anchor` is `None`.
pub(super) fn pool_class_angles(
    angles: &Array2<f64>,
    argmax_anchor: Option<&Array2<usize>>,
    num_classes: usize,
    anchors_per_class: usize,
) -> Array2<f64> {
    if anchors_per_class == 1 {
        return angles.clone();
    }
    let argmax = argmax_anchor.expect("argmax_anchor populated when anchors_per_class > 1");
    let batch_size = angles.nrows();
    Array2::from_shape_fn((batch_size, num_classes), |(i, c)| {
        angles[[i, c * anchors_per_class + argmax[[i, c]]]]
    })
}

/// Convert per-(sample, anchor) gradients with respect to the geodesic angle
/// `θ` into equivalent gradients with respect to the membership score
/// `s = exp(−θ²)`, so the shared CMR score-backward kernel is reused unchanged
/// for the geodesic hinge.
///
/// The kernel multiplies the incoming gradient by its cosine-chain factor
/// `ds/dc = 2θ·exp(−θ²)/√(1−cos²θ)`, and the desired chain is
/// `dL/dc = (dL/dθ)·(dθ/dc) = −(dL/dθ)/√(1−cos²θ)`. Dividing by
/// `ds/dθ = −2θ·exp(−θ²)` therefore reproduces `dL/dθ` exactly — the
/// `√(1−cos²θ)` factors cancel, including inside the kernel's clamped
/// cosine ≈ 1 branch. The `θ` in the denominator is floored at `eps` to guard
/// the measure-zero `θ → 0` case for inactive classes inside the margin.
pub(crate) fn angle_grads_to_score_grads(
    angle_grads: &Array2<f64>,
    angles: &Array2<f64>,
    eps: f64,
) -> Array2<f64> {
    debug_assert_eq!(angle_grads.dim(), angles.dim());
    let mut out = Array2::<f64>::zeros(angle_grads.dim());
    for i in 0..angle_grads.nrows() {
        for j in 0..angle_grads.ncols() {
            let g = angle_grads[[i, j]];
            if g == 0.0 {
                continue;
            }
            let theta = angles[[i, j]].max(eps);
            out[[i, j]] = -g / (2.0 * theta * (-theta * theta).exp());
        }
    }
    out
}

/// Top-1 accuracy in percentage points, or `None` for multi-label targets.
pub(crate) fn single_label_accuracy(scores: &Array2<f64>, targets: &Targets) -> Option<f64> {
    let Targets::SingleLabel(labels) = targets else {
        return None;
    };

    if labels.is_empty() {
        return Some(0.0);
    }

    let correct = scores
        .outer_iter()
        .zip(labels.iter())
        .filter(|(row, label)| argmax_row(row) == **label)
        .count();

    Some(correct as f64 * 100.0 / labels.len() as f64)
}

pub(super) fn argmax_row(row: &ndarray::ArrayView1<'_, f64>) -> usize {
    let mut best_idx = 0;
    let mut best_value = f64::NEG_INFINITY;

    for (idx, value) in row.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }

    best_idx
}
