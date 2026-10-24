//! Runtime pipeline wrapper for arrangement primitives.
//!
//! [`ModularBlock`] bundles selection, ordering, composition and optional
//! aggregation into a single batch-major `forward()` call.

use std::str::FromStr;

use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::{
    arrangement::{
        ArrangementError,
        activations::{Activation, ActivationPlacement, project_with_activation},
        aggregations::{AggregationContext, AggregationInfo, AggregationStrategy},
        compositions::{CompositionContext, CompositionStrategy, FlowNorm},
        orderings::{OrderingContext, OrderingInfo, OrderingStrategy},
        params::{
            AggregationParams, CompositionParams, OrderingParams, ParallelMode, SelectionParams,
            SubspaceAction,
        },
        selections::{SelectionContext, SelectionInfo, SelectionStrategy},
        utils::{get_w_blocks, projection_energy},
    },
    state::cosmos_state::CosmosState,
};

/// Runtime inputs shared across a single block execution.
///
/// The execution context carries mutable training/runtime state that is owned
/// by a higher-level caller such as `CosmosInterior`, while the block itself
/// remains focused on routing logic and caches.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext {
    /// Whether routing should use training-time behavior.
    pub training: bool,

    /// Current epoch index.
    pub epoch: u64,

    /// Current batch index.
    pub batch_idx: u64,
}

impl ExecutionContext {
    /// Create an execution context with explicit runtime values.
    #[must_use]
    pub fn new(training: bool, epoch: u64, batch_idx: u64) -> Self {
        Self {
            training,
            epoch,
            batch_idx,
        }
    }

    /// Create a standalone context for direct block usage.
    #[must_use]
    pub fn standalone() -> Self {
        Self::new(true, 0, 0)
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::standalone()
    }
}

/// Immutable configuration for one modular routing block.
///
/// The config stores parameter objects rather than parsed strategy enums so it
/// stays close to the higher-level builder/spec surface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockConfig {
    /// Candidate block ids visible to this routing block.
    pub block_indices: Vec<usize>,

    /// Selection primitive configuration.
    pub selection: SelectionParams,

    /// Ordering primitive configuration.
    pub ordering: OrderingParams,

    /// Composition primitive configuration.
    pub composition: CompositionParams,

    /// Optional aggregation primitive for parallel-style composition.
    pub aggregation: Option<AggregationParams>,
}

impl BlockConfig {
    /// Create a block config over the provided candidate indices.
    pub fn new(block_indices: Vec<usize>) -> Self {
        Self {
            block_indices,
            ..Default::default()
        }
    }
}

/// Runtime arrangement block composed from the arrangement primitives.
///
/// The block owns parsed strategy enums plus mutable diagnostics/caches used by
/// repeated `forward()` calls.
#[derive(Debug)]
pub struct ModularBlock {
    config: BlockConfig,
    selection_strategy: SelectionStrategy,
    ordering_strategy: OrderingStrategy,
    composition_strategy: CompositionStrategy,
    aggregation_strategy: Option<AggregationStrategy>,
    activation: Activation,
    activation_placement: ActivationPlacement,
    actions: Vec<SubspaceAction>,
    last_energies: Option<Array1<f64>>,
    last_selection: Option<SelectionInfo>,
    last_ordering: Option<OrderingInfo>,
    last_aggregation: Option<AggregationInfo>,
    last_selected_indices: Vec<usize>,
    last_ordered_indices: Vec<usize>,
}

impl ModularBlock {
    /// Construct a new modular block and eagerly parse its strategy names.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::UnknownStrategy`] when any strategy,
    /// activation or subspace-action name in `config` is not recognized.
    pub fn new(config: BlockConfig) -> Result<Self, ArrangementError> {
        let selection_strategy = SelectionStrategy::from_str(&config.selection.strategy)?;
        let ordering_strategy = OrderingStrategy::from_str(&config.ordering.strategy)?;
        let composition_strategy = CompositionStrategy::from_str(&config.composition.strategy)?;
        let aggregation_strategy = match config.aggregation.as_ref() {
            Some(params) => Some(AggregationStrategy::from_str(&params.strategy)?),
            None => None,
        };
        let activation = Activation::from_str(&config.composition.activation_name)?;
        let activation_placement =
            ActivationPlacement::from_str(&config.composition.activation_placement)?;
        let action = SubspaceAction::from_str(&config.composition.subspace_action)?;
        let num_indices = config.block_indices.len();
        let actions = vec![action; num_indices];

        Ok(Self {
            config,
            selection_strategy,
            ordering_strategy,
            composition_strategy,
            aggregation_strategy,
            activation,
            activation_placement,
            actions,
            last_energies: None,
            last_selection: None,
            last_ordering: None,
            last_aggregation: None,
            last_selected_indices: Vec::new(),
            last_ordered_indices: Vec::new(),
        })
    }

    /// Borrow the immutable block configuration.
    pub fn config(&self) -> &BlockConfig {
        &self.config
    }

    /// Last per-candidate mean projection energies aligned with `block_indices`.
    pub fn last_energies(&self) -> Option<&Array1<f64>> {
        self.last_energies.as_ref()
    }

    /// Selection telemetry from the last `forward()` call.
    pub fn last_selection(&self) -> Option<&SelectionInfo> {
        self.last_selection.as_ref()
    }

    /// Ordering telemetry from the last `forward()` call.
    pub fn last_ordering(&self) -> Option<&OrderingInfo> {
        self.last_ordering.as_ref()
    }

    /// Aggregation telemetry from the last `forward()` call.
    pub fn last_aggregation(&self) -> Option<&AggregationInfo> {
        self.last_aggregation.as_ref()
    }

    /// Last selected block ids after the selection stage.
    pub fn last_selected_indices(&self) -> &[usize] {
        &self.last_selected_indices
    }

    /// Last ordered block ids after the ordering stage.
    pub fn last_ordered_indices(&self) -> &[usize] {
        &self.last_ordered_indices
    }

    /// Per-block subspace actions, one per candidate block index.
    #[must_use]
    pub fn actions(&self) -> &[SubspaceAction] {
        &self.actions
    }

    /// Replace the action for a single candidate block at `local_idx`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::ShapeMismatch`] when `local_idx` is out of
    /// range for the configured candidate blocks.
    pub fn set_action_at(
        &mut self,
        local_idx: usize,
        action: SubspaceAction,
    ) -> Result<(), ArrangementError> {
        if local_idx >= self.actions.len() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("local_idx < {}", self.actions.len()),
                got: format!("local_idx = {local_idx}"),
            });
        }
        self.actions[local_idx] = action;
        Ok(())
    }

    /// Replace all actions at once.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::ShapeMismatch`] when `actions` does not
    /// have one entry per candidate block.
    pub fn set_all_actions(
        &mut self,
        actions: Vec<SubspaceAction>,
    ) -> Result<(), ArrangementError> {
        if actions.len() != self.config.block_indices.len() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!(
                    "actions.len() == block_indices.len() ({})",
                    self.config.block_indices.len()
                ),
                got: format!("actions.len() = {}", actions.len()),
            });
        }
        self.actions = actions;
        Ok(())
    }

    /// Execute the routing pipeline over a batch-major input.
    ///
    /// The convenience wrapper uses a standalone execution context with
    /// training enabled and `(epoch, batch_idx) = (0, 0)`.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::forward_with_context`].
    pub fn forward(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        let ctx = ExecutionContext::standalone();
        self.forward_with_context(x, state, &ctx)
    }

    /// Execute the routing pipeline over a batch-major input with an explicit
    /// runtime context.
    ///
    /// Pipeline:
    ///
    /// 1. fetch candidate projectors from [`CosmosState`]
    /// 2. compute mean projection energies
    /// 3. apply selection
    /// 4. apply ordering
    /// 5. compose ordered projectors
    /// 6. aggregate parallel branches when needed
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::ShapeMismatch`] when a candidate block's
    /// ambient dimension does not match `x`, and propagates routing-primitive
    /// failures.
    pub fn forward_with_context(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
        ctx: &ExecutionContext,
    ) -> Result<Array2<f64>, ArrangementError> {
        self.forward_with_context_impl(x, state, ctx, None)
    }

    /// Execute the routing pipeline while recording per-projector flow norms.
    ///
    /// The output is identical to [`ModularBlock::forward_with_context`];
    /// the returned records come from
    /// [`CompositionStrategy::apply_with_flow`] over the composed projector
    /// chain. Branches that bypass the compositions layer return an empty
    /// record vector:
    ///
    /// - empty candidate/selected/ordered sets (identity passthrough)
    /// - `Parallel`, which routes through the aggregation pipeline rather
    ///   than `CompositionStrategy::apply`
    ///
    /// # Errors
    ///
    /// Same failure modes as [`ModularBlock::forward_with_context`].
    pub fn forward_with_flow(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
        ctx: &ExecutionContext,
    ) -> Result<(Array2<f64>, Vec<FlowNorm>), ArrangementError> {
        let mut records = Vec::new();
        let output = self.forward_with_context_impl(x, state, ctx, Some(&mut records))?;
        Ok((output, records))
    }

    /// Shared routing pipeline behind `forward_with_context` and
    /// `forward_with_flow`. When `flow` is present, compositions dispatched
    /// through `CompositionStrategy` record per-projector flow norms into it;
    /// the computed output is identical either way.
    fn forward_with_context_impl(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
        ctx: &ExecutionContext,
        flow: Option<&mut Vec<FlowNorm>>,
    ) -> Result<Array2<f64>, ArrangementError> {
        self.clear_last_forward_state();

        let candidate_indices = self.config.block_indices.clone();
        if candidate_indices.is_empty() {
            return Ok(x.clone());
        }

        let candidate_blocks = get_w_blocks(state, &candidate_indices)?;
        validate_projector_rows(&candidate_blocks, x, &candidate_indices)?;

        let energies = mean_projection_energies(x, &candidate_blocks);
        self.last_energies = Some(energies);

        let selection_ctx = SelectionContext {
            training: ctx.training,
        };
        let (selected_indices, selection_info) = self.selection_strategy.apply(
            &candidate_indices,
            &selection_ctx,
            &self.config.selection,
        )?;

        self.last_selected_indices = selected_indices.clone();
        self.last_selection = Some(selection_info);

        if selected_indices.is_empty() {
            return Ok(x.clone());
        }

        let (ordered_indices, ordering_info) = self.ordering_strategy.apply(
            &selected_indices,
            &OrderingContext,
            &self.config.ordering,
        )?;

        self.last_ordered_indices = ordered_indices.clone();
        self.last_ordering = Some(ordering_info);

        if ordered_indices.is_empty() {
            return Ok(x.clone());
        }

        let ordered_blocks = get_w_blocks(state, &ordered_indices)?;
        validate_projector_rows(&ordered_blocks, x, &ordered_indices)?;

        match self.composition_strategy {
            CompositionStrategy::Series | CompositionStrategy::ResidualSeries => {
                self.last_aggregation = None;
                self.dispatch_composition(&ordered_blocks, x, &ordered_indices, flow)
            }
            // Parallel branches route through the aggregation pipeline
            // rather than `CompositionStrategy::apply`, so no per-projector
            // flow norms are recorded for them.
            CompositionStrategy::Parallel => self.apply_parallel_composition(x, &ordered_blocks),
            // ParallelSeries does not participate in the per-block
            // aggregation/diagnostics graph that
            // `apply_parallel_composition` builds — it goes through the
            // compositions layer's direct dispatch instead.
            CompositionStrategy::ParallelSeries => {
                self.last_aggregation = None;
                self.dispatch_composition(&ordered_blocks, x, &ordered_indices, flow)
            }
        }
    }

    /// Dispatch a composition through the compositions layer, recording flow
    /// norms when a recorder is supplied. Both paths compute the identical
    /// output.
    fn dispatch_composition(
        &self,
        ordered_blocks: &[Array2<f64>],
        x: &Array2<f64>,
        ordered_indices: &[usize],
        flow: Option<&mut Vec<FlowNorm>>,
    ) -> Result<Array2<f64>, ArrangementError> {
        let ctx = self.composition_context(ordered_indices);
        match flow {
            Some(records) => {
                let (output, step_records) =
                    self.composition_strategy
                        .apply_with_flow(ordered_blocks, x, &ctx)?;
                records.extend(step_records);
                Ok(output)
            }
            None => self.composition_strategy.apply(ordered_blocks, x, &ctx),
        }
    }

    /// Convenience wrapper for single-sample inputs.
    ///
    /// This adapts a rank-1 input into the batch-major API used by the
    /// arrangement runtime.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::forward`].
    pub fn forward_one(
        &mut self,
        x: &Array1<f64>,
        state: &CosmosState,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }

    /// Execute the routing pipeline over a single sample with an explicit
    /// runtime context.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::forward_with_context`].
    pub fn forward_one_with_context(
        &mut self,
        x: &Array1<f64>,
        state: &CosmosState,
        ctx: &ExecutionContext,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward_with_context(&batch, state, ctx)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }

    fn composition_context(&self, ordered_indices: &[usize]) -> CompositionContext {
        // Map ordered indices to their position in block_indices, then pick
        // the corresponding action. Falls back to default if not found.
        let default_action = SubspaceAction::default();
        let actions: Vec<SubspaceAction> = ordered_indices
            .iter()
            .map(|&idx| {
                self.config
                    .block_indices
                    .iter()
                    .position(|&bi| bi == idx)
                    .and_then(|pos| self.actions.get(pos).copied())
                    .unwrap_or(default_action)
            })
            .collect();

        CompositionContext {
            activation: self.activation,
            placement: self.activation_placement,
            actions,
            block_indices: ordered_indices.to_vec(),
        }
    }

    fn apply_parallel_composition(
        &mut self,
        x: &Array2<f64>,
        ordered_blocks: &[Array2<f64>],
    ) -> Result<Array2<f64>, ArrangementError> {
        if self.config.composition.parallel_mode == ParallelMode::List {
            return Err(ArrangementError::InvalidParameter {
                param: "parallel_mode",
                reason: "ParallelMode::List is not supported by ModularBlock::forward() yet"
                    .to_string(),
            });
        }

        let tensor = build_parallel_tensor(
            ordered_blocks,
            x,
            self.activation,
            self.activation_placement,
        );
        let (output, aggregation_info) = self.aggregate_parallel_outputs(&tensor)?;

        self.last_aggregation = Some(aggregation_info);

        Ok(output)
    }

    fn aggregate_parallel_outputs(
        &self,
        tensor: &Array3<f64>,
    ) -> Result<(Array2<f64>, AggregationInfo), ArrangementError> {
        let (strategy, params) = match (&self.aggregation_strategy, &self.config.aggregation) {
            (Some(strategy), Some(params)) => (*strategy, params.clone()),
            _ => (AggregationStrategy::Mean, AggregationParams::default()),
        };

        strategy.apply(tensor, &AggregationContext, &params)
    }

    fn clear_last_forward_state(&mut self) {
        self.last_energies = None;
        self.last_selection = None;
        self.last_ordering = None;
        self.last_aggregation = None;
        self.last_selected_indices.clear();
        self.last_ordered_indices.clear();
    }
}

fn validate_projector_rows(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    block_indices: &[usize],
) -> Result<(), ArrangementError> {
    let ambient_dim = x.ncols();

    for (pos, projector) in projectors.iter().enumerate() {
        if projector.nrows() != ambient_dim {
            let block_label = block_indices
                .get(pos)
                .map(|idx| format!("block {idx}"))
                .unwrap_or_else(|| format!("position {pos}"));

            return Err(ArrangementError::ShapeMismatch {
                expected: format!("{block_label} with ambient dimension {ambient_dim}"),
                got: format!("({}, {})", projector.nrows(), projector.ncols()),
            });
        }
    }

    Ok(())
}

fn mean_projection_energies(x: &Array2<f64>, projectors: &[Array2<f64>]) -> Array1<f64> {
    if projectors.is_empty() {
        return Array1::zeros(0);
    }

    // Check if all projectors have the same k for stacked GEMM
    let k0 = projectors[0].ncols();
    let uniform_k = projectors.iter().all(|w| w.ncols() == k0);

    if uniform_k && projectors.len() > 1 {
        // Stacked GEMM: one large matmul instead of L separate ones
        let n = projectors[0].nrows();
        let l = projectors.len();
        let mut w_all = Array2::<f64>::zeros((n, l * k0));
        for (i, w) in projectors.iter().enumerate() {
            w_all
                .slice_mut(ndarray::s![.., i * k0..(i + 1) * k0])
                .assign(w);
        }
        let z = x.dot(&w_all); // batch × (L*k) — single GEMM
        let batch = z.nrows() as f64;
        Array1::from_iter((0..l).map(|i| {
            let block = z.slice(ndarray::s![.., i * k0..(i + 1) * k0]);
            block.iter().map(|v| v * v).sum::<f64>() / batch
        }))
    } else {
        // Fallback: per-block projection energy
        Array1::from_iter(projectors.iter().map(|projector| {
            let energies = projection_energy(x, projector);
            if energies.is_empty() {
                0.0
            } else {
                energies.sum() / energies.len() as f64
            }
        }))
    }
}

fn build_parallel_tensor(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    activation: Activation,
    placement: ActivationPlacement,
) -> Array3<f64> {
    let (batch, dim) = x.dim();
    let mut tensor = Array3::<f64>::zeros((batch, projectors.len(), dim));

    for (branch_idx, projector) in projectors.iter().enumerate() {
        let projected = project_with_activation(projector, x, activation, placement);
        tensor.slice_mut(s![.., branch_idx, ..]).assign(&projected);
    }

    tensor
}

#[cfg(test)]
mod tests {
    use std::f64::consts::SQRT_2;

    use ndarray::{Array2, array};

    use super::*;
    use crate::arrangement::{params::ParallelMode, selections::SelectionDetails};

    const EPSILON: f64 = 1e-10;

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn x_axis() -> Array2<f64> {
        array![[1.0], [0.0]]
    }

    fn y_axis() -> Array2<f64> {
        array![[0.0], [1.0]]
    }

    fn diagonal_xy() -> Array2<f64> {
        let inv_sqrt2 = 1.0 / SQRT_2;
        array![[inv_sqrt2], [inv_sqrt2]]
    }

    fn build_state(blocks: &[Array2<f64>]) -> CosmosState {
        let mut state = CosmosState::new(&[]).unwrap();
        for block in blocks {
            state.register_external_block(block).unwrap();
        }
        state
    }

    #[test]
    fn new_parses_all_strategies() {
        let config = BlockConfig {
            block_indices: vec![0, 1],
            selection: SelectionParams::default(),
            ordering: OrderingParams::default(),
            composition: CompositionParams::new("parallel", ParallelMode::Stack, "relu", "post"),
            aggregation: Some(AggregationParams::default()),
        };

        let block = ModularBlock::new(config).unwrap();

        assert_eq!(block.config().block_indices, vec![0, 1]);
    }

    #[test]
    fn new_rejects_an_unknown_strategy() {
        for (label, config) in [
            (
                "selection",
                BlockConfig {
                    selection: SelectionParams {
                        strategy: "roulette".to_string(),
                    },
                    ..Default::default()
                },
            ),
            (
                "ordering",
                BlockConfig {
                    ordering: OrderingParams {
                        strategy: "spiral".to_string(),
                    },
                    ..Default::default()
                },
            ),
            (
                "composition",
                BlockConfig {
                    composition: CompositionParams::new(
                        "cascade",
                        ParallelMode::Stack,
                        "none",
                        "none",
                    ),
                    ..Default::default()
                },
            ),
            (
                "aggregation",
                BlockConfig {
                    aggregation: Some(AggregationParams {
                        strategy: "harmonic".to_string(),
                    }),
                    ..Default::default()
                },
            ),
        ] {
            let err = ModularBlock::new(config).unwrap_err();
            assert!(
                matches!(err, ArrangementError::UnknownStrategy { .. }),
                "{label}: expected an unknown-strategy rejection, got {err:?}"
            );
        }
    }

    #[test]
    fn forward_with_empty_block_indices_returns_input_and_no_diagnostics() {
        let state = build_state(&[x_axis()]);
        let mut block = ModularBlock::new(BlockConfig::default()).unwrap();
        let x = array![[1.0, 2.0]];

        let out = block.forward(&x, &state).unwrap();

        assert_array2_close(&out, &x);
        assert!(block.last_energies().is_none());
        assert!(block.last_selection().is_none());
        assert!(block.last_ordering().is_none());
        assert!(block.last_aggregation().is_none());
        assert!(block.last_selected_indices().is_empty());
        assert!(block.last_ordered_indices().is_empty());
    }

    #[test]
    fn forward_selects_every_candidate_block() {
        let state = build_state(&[x_axis(), y_axis(), diagonal_xy()]);
        let mut block = ModularBlock::new(BlockConfig::new(vec![0, 1, 2])).unwrap();

        let _ = block.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_eq!(block.last_selected_indices(), &[0, 1, 2]);
        assert_eq!(block.last_ordered_indices(), &[0, 1, 2]);
        assert_eq!(
            block.last_selection().unwrap().details,
            SelectionDetails::All
        );
        assert_eq!(block.last_energies().unwrap().len(), 3);
    }

    #[test]
    fn forward_series_applies_projectors_in_candidate_order() {
        let state = build_state(&[x_axis(), diagonal_xy()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        })
        .unwrap();

        let out = block.forward(&array![[2.0, 1.0]], &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 1.0]]);
    }

    #[test]
    fn forward_one_with_context_matches_batch_forward() {
        let state = build_state(&[x_axis(), diagonal_xy()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        })
        .unwrap();
        let x = array![2.0, 1.0];
        let ctx = ExecutionContext::new(false, 3, 7);

        let single = block.forward_one_with_context(&x, &state, &ctx).unwrap();
        let batch = block
            .forward_with_context(&x.clone().insert_axis(Axis(0)), &state, &ctx)
            .unwrap();

        assert_array2_close(&single.insert_axis(Axis(0)), &batch);
    }

    #[test]
    fn forward_parallel_mean_averages_branch_projections() {
        let state = build_state(&[x_axis(), y_axis()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("parallel", ParallelMode::Stack, "none", "none"),
            aggregation: Some(AggregationParams::default()),
            ..Default::default()
        })
        .unwrap();

        let out = block.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_array2_close(&out, &array![[0.5, 1.0]]);
        assert!(block.last_aggregation().is_some());
    }

    #[test]
    fn forward_rejects_projector_ambient_dimension_mismatch() {
        let state = build_state(&[array![[1.0], [0.0], [0.0]]]);
        let mut block = ModularBlock::new(BlockConfig::new(vec![0])).unwrap();

        let err = block.forward(&array![[1.0, 2.0]], &state).unwrap_err();

        match err {
            ArrangementError::ShapeMismatch { expected, .. } => {
                assert!(expected.contains("block 0"));
                assert!(expected.contains("ambient dimension 2"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn forward_with_flow_matches_forward_and_reports_series_steps() {
        let state = build_state(&[x_axis(), diagonal_xy()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        })
        .unwrap();
        let ctx = ExecutionContext::default();
        let x = array![[2.0, 1.0], [-1.0, 3.0]];

        let plain = block.forward_with_context(&x, &state, &ctx).unwrap();
        let (flowed, records) = block.forward_with_flow(&x, &state, &ctx).unwrap();

        assert_eq!(plain, flowed);
        assert_eq!(records.len(), 2);
        for record in &records {
            assert!(record.mean_in_norm.is_finite());
            assert!(record.mean_out_norm.is_finite());
        }
    }

    #[test]
    fn forward_with_flow_parallel_aggregation_returns_empty_records() {
        let state = build_state(&[x_axis(), y_axis()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("parallel", ParallelMode::Stack, "none", "none"),
            aggregation: Some(AggregationParams::default()),
            ..Default::default()
        })
        .unwrap();
        let ctx = ExecutionContext::default();
        let x = array![[1.0, 2.0]];

        let plain = block.forward_with_context(&x, &state, &ctx).unwrap();
        let (flowed, records) = block.forward_with_flow(&x, &state, &ctx).unwrap();

        assert_eq!(plain, flowed);
        assert!(records.is_empty());
    }

    #[test]
    fn parallel_list_mode_is_rejected_for_now() {
        let state = build_state(&[x_axis(), y_axis()]);
        let mut block = ModularBlock::new(BlockConfig {
            block_indices: vec![0, 1],
            composition: CompositionParams::new("parallel", ParallelMode::List, "none", "none"),
            ..Default::default()
        })
        .unwrap();

        let err = block.forward(&array![[1.0, 2.0]], &state).unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => {
                assert_eq!(param, "parallel_mode");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
