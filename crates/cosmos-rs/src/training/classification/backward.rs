//! Backward pass of the flat classification adapter, and the batch trace.
//!
//! The head gradient is written straight into the block-aligned
//! [`BlockGradients`] vector; the interior and input-projector stages then walk
//! the cached forward buffers back to the input basis.

use std::time::Instant;

use ndarray::Array2;

use crate::{
    compute::{
        cmr_multi_anchor_backward_device,
        device::{Buffer1, Buffer2, ComputeDevice, Device},
        projector_basis_gradient_device,
    },
    models::CosmosModel,
    objectives::cmr_losses::geodesic_cmr_loss_grad,
    training::{
        BlockGradients, TrainError,
        trace::{BatchTrace, BlockTrace, CmrTrace, InputProjectionTrace, InteriorTrace},
    },
};

use super::{
    ClassificationLoss, FlatClassificationAdapterConfig, FlatClassificationBatch,
    forward::ForwardBatch,
    loss::direct_membership_score_grad,
    shared::{
        InteriorForwardDevice, angle_grads_to_score_grads, backward_supported_interior,
        upload_basis,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct BackwardTimingBreakdown {
    pub(super) anchor_sec: f64,
    pub(super) interior_sec: f64,
    pub(super) input_projector_sec: f64,
}

pub(super) fn backward(
    device: &Device,
    model: &CosmosModel,
    batch: &FlatClassificationBatch,
    forward: &ForwardBatch,
    config: &FlatClassificationAdapterConfig,
) -> Result<(BlockGradients, BackwardTimingBreakdown), TrainError> {
    let state = model.get_state();
    let input_basis = state.get_basis(forward.input_block_idx)?;

    let mut gradients: BlockGradients = vec![None; state.num_blocks()];
    let head_start = Instant::now();
    let grad_embeddings_buf = cmr_head_backward(
        device,
        model,
        batch,
        forward,
        config,
        &forward.embeddings_buf,
        &mut gradients,
    )?;
    let backward_anchor_sec = head_start.elapsed().as_secs_f64();

    let interior_start = Instant::now();
    let grad_projected_inputs_buf = backward_supported_interior(
        device,
        state,
        forward.interior_device.as_ref(),
        &grad_embeddings_buf,
        &mut gradients,
    )?;
    let backward_interior_sec = interior_start.elapsed().as_secs_f64();

    // The input projector applies the full map `x W_0 W_0ᵀ`, whose basis
    // gradient is the two-term projector sum.
    let input_projector_start = Instant::now();
    let input_basis_buf = upload_basis(device, input_basis);
    let mut input_gradient_buf: Buffer2<f64> =
        device.zeros_2d(input_basis.nrows(), input_basis.ncols());
    projector_basis_gradient_device(
        device,
        &forward.inputs_buf,
        None,
        &grad_projected_inputs_buf,
        &input_basis_buf,
        &mut input_gradient_buf,
    );
    let backward_input_projector_sec = input_projector_start.elapsed().as_secs_f64();
    gradients[forward.input_block_idx] = Some(input_gradient_buf);

    Ok((
        gradients,
        BackwardTimingBreakdown {
            anchor_sec: backward_anchor_sec,
            interior_sec: backward_interior_sec,
            input_projector_sec: backward_input_projector_sec,
        },
    ))
}

/// Backward through the geodesic CMR anchor head.
///
/// Writes every anchor gradient into `gradients` and returns `∂L/∂z`.
pub(super) fn cmr_head_backward(
    device: &Device,
    model: &CosmosModel,
    batch: &FlatClassificationBatch,
    forward: &ForwardBatch,
    config: &FlatClassificationAdapterConfig,
    embeddings_buf: &Buffer2<f64>,
    gradients: &mut BlockGradients,
) -> Result<Buffer2<f64>, TrainError> {
    let eps = config.eps;
    let state = model.get_state();

    let embedding_norms_buf: Buffer1<f64> = device.row_norms(embeddings_buf);

    let anchor_bufs: Vec<Buffer2<f64>> = forward
        .anchor_indices
        .iter()
        .map(|&idx| Ok(upload_basis(device, state.get_basis(idx)?)))
        .collect::<Result<Vec<_>, TrainError>>()?;
    let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();

    // For the direct-membership loss these are gradients with respect to the
    // scores `s = exp(−θ²)`; for the geodesic hinge they are gradients with
    // respect to the pooled angles `θ`, converted to score gradients after the
    // anchor fan-out below. The CMR backward kernel expects one column per
    // anchor, so multi-anchor models fan the per-class gradient out to the
    // winning anchor (max-pool subgradient); losing anchors receive zero.
    let score_grads_per_class = match config.loss {
        ClassificationLoss::DirectMembership => {
            direct_membership_score_grad(&forward.scores, &batch.targets)?
        }
        ClassificationLoss::GeodesicHinge { margin } => {
            let class_distances = forward.class_distances.as_ref().ok_or_else(|| {
                TrainError::task_adapter(
                    "geodesic hinge backward requires class distances from forward",
                )
            })?;
            // `dloss_dtheta` carries the `1 / (batch · num_classes)`
            // normalization — the exact gradient of the loss forward reports.
            geodesic_cmr_loss_grad(class_distances, &batch.targets, margin)?.dloss_dtheta
        }
    };
    let score_grads_f64 = if forward.anchors_per_class == 1 {
        score_grads_per_class
    } else {
        let batch_size = score_grads_per_class.nrows();
        let num_classes = score_grads_per_class.ncols();
        let total_anchors = num_classes * forward.anchors_per_class;
        let argmax = forward
            .argmax_anchor
            .as_ref()
            .expect("argmax_anchor populated when anchors_per_class > 1");
        let mut per_anchor = Array2::<f64>::zeros((batch_size, total_anchors));
        for sample_idx in 0..batch_size {
            for class_idx in 0..num_classes {
                let winner = argmax[[sample_idx, class_idx]];
                let col = class_idx * forward.anchors_per_class + winner;
                per_anchor[[sample_idx, col]] = score_grads_per_class[[sample_idx, class_idx]];
            }
        }
        per_anchor
    };
    // For the angle-supervised loss, `score_grads_f64` holds ∂L/∂θ per
    // (sample, anchor); re-expressing it as an equivalent ∂L/∂s lets the shared
    // CMR score-backward kernel be reused unchanged.
    let score_grads_f64 = if config.loss.supervises_angles() {
        let angles = forward.angles.as_ref().ok_or_else(|| {
            TrainError::task_adapter(
                "angle-supervised backward requires per-anchor angles from forward",
            )
        })?;
        angle_grads_to_score_grads(&score_grads_f64, angles, eps)
    } else {
        score_grads_f64
    };
    let score_grads_buf = upload_basis(device, &score_grads_f64);

    let mut grad_embeddings_buf: Buffer2<f64> =
        device.zeros_2d(embeddings_buf.dim().0, embeddings_buf.dim().1);
    let mut grad_anchor_bufs: Vec<Buffer2<f64>> = anchor_bufs
        .iter()
        .map(|a| device.zeros_2d(a.nrows(), a.ncols()))
        .collect();
    cmr_multi_anchor_backward_device(
        device,
        embeddings_buf,
        &embedding_norms_buf,
        &anchor_refs,
        &score_grads_buf,
        eps,
        &mut grad_embeddings_buf,
        &mut grad_anchor_bufs,
    );

    for (i, &block_idx) in forward.anchor_indices.iter().enumerate() {
        let grad = std::mem::replace(&mut grad_anchor_bufs[i], device.zeros_2d(0, 0));
        gradients[block_idx] = Some(grad);
    }

    Ok(grad_embeddings_buf)
}

pub(super) fn build_trace(
    batch: &FlatClassificationBatch,
    forward: &ForwardBatch,
    collect_trace: bool,
) -> Option<BatchTrace> {
    if !collect_trace {
        return None;
    }

    let cpu = crate::compute::device::cpu::CpuDevice;
    Some(BatchTrace {
        input_projection: Some(InputProjectionTrace {
            block_idx: forward.input_block_idx,
            input: batch.inputs.clone(),
            output: forward.projected_inputs.clone(),
        }),
        interior: build_interior_trace_from_device(&cpu, &forward.interior_device),
        cmr: Some(CmrTrace {
            embeddings: cpu.download_2d(&forward.embeddings_buf),
            scores: forward.scores.clone(),
            winning_anchor_indices: None,
        }),
        scalar_loss: Some(forward.loss),
    })
}

pub(super) fn build_interior_trace_from_device(
    device: &impl ComputeDevice,
    interior: &Option<InteriorForwardDevice>,
) -> Option<InteriorTrace> {
    let interior = interior.as_ref()?;

    Some(InteriorTrace {
        input: Some(device.download_2d(&interior.input)),
        output: Some(device.download_2d(&interior.output)),
        blocks: interior
            .blocks
            .iter()
            .map(|block| BlockTrace {
                selection: block.selection.clone(),
                input: device.download_2d(&block.input),
                output: device.download_2d(&block.output),
            })
            .collect(),
    })
}
