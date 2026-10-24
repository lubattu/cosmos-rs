//! Runtime dispatch over sequences of modular arrangement blocks.
//!
//! [`CosmosInterior`] applies a list of blocks sequentially, then optionally
//! applies a global activation and row-wise output normalization.

use ndarray::{Array1, Array2, Axis};

use crate::{
    arrangement::{
        ArrangementError,
        activations::Activation,
        blocks::{BlockConfig, ExecutionContext, ModularBlock},
        builder::InteriorSpec,
        compositions::{FlowNorm, mean_row_norm},
        params::{AggregationParams, CompositionParams, ParallelMode},
        utils::normalize_rank1,
    },
    state::cosmos_state::CosmosState,
};

/// Runtime container that dispatches a sequence of [`ModularBlock`]s.
///
/// The current implementation is batch-major and intentionally thin: it
/// coordinates block execution plus final activation/normalization, while the
/// detailed routing logic remains inside each block.
#[derive(Debug)]
pub struct CosmosInterior {
    blocks: Vec<ModularBlock>,
    global_activation: Activation,
    normalize_final: bool,
    training: bool,
    epoch: u64,
    batch_idx: u64,
}

impl CosmosInterior {
    /// Construct an interior with default output settings.
    #[must_use]
    pub fn new(blocks: Vec<ModularBlock>) -> Self {
        Self::with_options(blocks, Activation::Identity, false)
    }

    /// Construct an interior with explicit output settings.
    #[must_use]
    pub fn with_options(
        blocks: Vec<ModularBlock>,
        global_activation: Activation,
        normalize_final: bool,
    ) -> Self {
        Self {
            blocks,
            global_activation,
            normalize_final,
            training: true,
            epoch: 0,
            batch_idx: 0,
        }
    }

    /// Build an interior from block configs using default output settings.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn from_block_configs(block_configs: Vec<BlockConfig>) -> Result<Self, ArrangementError> {
        Self::from_block_configs_with_options(block_configs, Activation::Identity, false)
    }

    /// Build an interior from block configs with explicit output settings.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn from_block_configs_with_options(
        block_configs: Vec<BlockConfig>,
        global_activation: Activation,
        normalize_final: bool,
    ) -> Result<Self, ArrangementError> {
        let blocks: Result<Vec<_>, _> = block_configs.into_iter().map(ModularBlock::new).collect();

        Ok(Self::with_options(
            blocks?,
            global_activation,
            normalize_final,
        ))
    }

    /// Build an interior from an immutable builder spec.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn from_spec(spec: &InteriorSpec) -> Result<Self, ArrangementError> {
        let blocks: Result<Vec<_>, _> = spec.blocks.iter().map(|block| block.build()).collect();

        Ok(Self::with_options(
            blocks?,
            spec.global_activation,
            spec.normalize_final,
        ))
    }

    /// Convenience constructor for a single series block.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn simple_series(block_indices: Vec<usize>) -> Result<Self, ArrangementError> {
        let config = BlockConfig {
            block_indices,
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        };

        Self::from_block_configs(vec![config])
    }

    /// Convenience constructor for a single mean-pooled parallel block.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn parallel_pool(block_indices: Vec<usize>) -> Result<Self, ArrangementError> {
        let config = BlockConfig {
            block_indices,
            composition: CompositionParams::new("parallel", ParallelMode::Stack, "none", "none"),
            aggregation: Some(AggregationParams::default()),
            ..Default::default()
        };

        Self::from_block_configs(vec![config])
    }

    /// Convenience constructor for a single residual series block.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn residual_chain(block_indices: Vec<usize>) -> Result<Self, ArrangementError> {
        let config = BlockConfig {
            block_indices,
            composition: CompositionParams::new(
                "residual_series",
                ParallelMode::Stack,
                "none",
                "none",
            ),
            ..Default::default()
        };

        Self::from_block_configs(vec![config])
    }

    /// Borrow the managed blocks.
    #[must_use]
    pub fn blocks(&self) -> &[ModularBlock] {
        &self.blocks
    }

    /// Mutably borrow the managed blocks.
    pub fn blocks_mut(&mut self) -> &mut [ModularBlock] {
        &mut self.blocks
    }

    /// Number of runtime blocks in this interior.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Global post-block activation.
    #[must_use]
    pub fn global_activation(&self) -> Activation {
        self.global_activation
    }

    /// Whether the final output is row-wise normalized.
    #[must_use]
    pub fn normalize_final(&self) -> bool {
        self.normalize_final
    }

    /// Whether the interior is in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Current epoch index.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Current batch index.
    #[must_use]
    pub fn batch_idx(&self) -> u64 {
        self.batch_idx
    }

    /// Set training or evaluation mode for the full interior.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Update epoch and batch counters for the full interior.
    pub fn set_epoch_batch(&mut self, epoch: u64, batch_idx: u64) {
        self.epoch = epoch;
        self.batch_idx = batch_idx;
    }

    /// Reset the batch index at the start of a new epoch.
    pub fn on_epoch_start(&mut self, epoch: u64) {
        self.set_epoch_batch(epoch, 0);
    }

    /// Apply all blocks sequentially to a batch-major input.
    ///
    /// # Errors
    ///
    /// Propagates the failures of
    /// [`ModularBlock::forward_with_context`].
    pub fn forward(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        let mut current = x.clone();
        let ctx = ExecutionContext::new(self.training, self.epoch, self.batch_idx);

        for block in &mut self.blocks {
            current = block.forward_with_context(&current, state, &ctx)?;
        }

        if self.global_activation != Activation::Identity {
            current = self.global_activation.apply_array2(&current);
        }

        if self.normalize_final {
            current = normalize_rows(&current);
        }

        Ok(current)
    }

    /// Apply all blocks sequentially while recording per-projector flow norms.
    ///
    /// The output is identical to [`CosmosInterior::forward`]. The records
    /// concatenate each block's [`ModularBlock::forward_with_flow`] entries
    /// in execution order; the global activation and optional row
    /// normalization apply to the final output only and are not recorded as
    /// per-block steps.
    ///
    /// # Errors
    ///
    /// Same failure modes as [`CosmosInterior::forward`].
    pub fn forward_with_flow(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<(Array2<f64>, Vec<FlowNorm>), ArrangementError> {
        let (output, records, _pre_normalize_norm) = self.forward_with_flow_detailed(x, state)?;
        Ok((output, records))
    }

    /// Flow-instrumented forward that also reports the batch-mean row norm
    /// of the output after the global activation but before the optional
    /// row normalization.
    ///
    /// Row normalization forces every row to unit norm, which would erase
    /// the norm-dilution signal that flow diagnostics measure; callers that
    /// build an overall input/output ratio therefore need the
    /// pre-normalization norm.
    pub(crate) fn forward_with_flow_detailed(
        &mut self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<(Array2<f64>, Vec<FlowNorm>, f64), ArrangementError> {
        let mut current = x.clone();
        let mut records = Vec::new();
        let ctx = ExecutionContext::new(self.training, self.epoch, self.batch_idx);

        for block in &mut self.blocks {
            let (output, block_records) = block.forward_with_flow(&current, state, &ctx)?;
            records.extend(block_records);
            current = output;
        }

        if self.global_activation != Activation::Identity {
            current = self.global_activation.apply_array2(&current);
        }

        let pre_normalize_norm = mean_row_norm(&current);

        if self.normalize_final {
            current = normalize_rows(&current);
        }

        Ok((current, records, pre_normalize_norm))
    }

    /// Apply all blocks to a single sample.
    ///
    /// # Errors
    ///
    /// Same failure modes as [`CosmosInterior::forward`].
    pub fn forward_one(
        &mut self,
        x: &Array1<f64>,
        state: &CosmosState,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

fn normalize_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut normalized = Array2::<f64>::zeros(x.dim());

    for (row_idx, row) in x.outer_iter().enumerate() {
        let row_normalized = normalize_rank1(&row.to_owned());
        normalized.row_mut(row_idx).assign(&row_normalized);
    }

    normalized
}

#[cfg(test)]
mod tests {
    use std::f64::consts::SQRT_2;

    use ndarray::{Array2, array};

    use super::*;
    use crate::arrangement::{blocks::ModularBlock, params::ParallelMode};

    const EPSILON: f64 = 1e-10;

    fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

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

    fn make_block(config: BlockConfig) -> ModularBlock {
        ModularBlock::new(config).unwrap()
    }

    #[test]
    fn forward_applies_blocks_then_global_activation() {
        let state = build_state(&[y_axis()]);
        let block = make_block(BlockConfig {
            block_indices: vec![0],
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        });
        let mut interior = CosmosInterior::with_options(vec![block], Activation::Relu, false);

        let x = array![[-1.0, -2.0]];
        let out = interior.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[0.0, 0.0]]);
    }

    #[test]
    fn forward_normalizes_final_rows() {
        let mut interior = CosmosInterior::with_options(Vec::new(), Activation::Identity, true);
        let state = CosmosState::new(&[]).unwrap();
        let x = array![[3.0, 4.0], [0.0, 0.0]];

        let out = interior.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[0.6, 0.8], [0.0, 0.0]]);
    }

    #[test]
    fn forward_one_matches_batch_forward() {
        let state = build_state(&[x_axis()]);
        let block = make_block(BlockConfig {
            block_indices: vec![0],
            composition: CompositionParams::new("series", ParallelMode::Stack, "none", "none"),
            ..Default::default()
        });
        let mut interior = CosmosInterior::new(vec![block]);
        let x = array![1.0, 2.0];

        let single = interior.forward_one(&x, &state).unwrap();
        let batch = interior
            .forward(&x.clone().insert_axis(Axis(0)), &state)
            .unwrap();

        assert_array1_close(&single, &batch.index_axis(Axis(0), 0).to_owned());
    }

    #[test]
    fn set_training_changes_runtime_behavior() {
        let state = build_state(&[x_axis(), y_axis(), diagonal_xy()]);
        let block = make_block(BlockConfig::new(vec![0, 1, 2]));
        let mut interior = CosmosInterior::new(vec![block]);

        interior.set_training(false);
        let _ = interior.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert!(!interior.is_training());
        assert_eq!(interior.blocks()[0].last_selected_indices().len(), 3);
    }

    #[test]
    fn set_epoch_batch_updates_runtime_state() {
        let mut interior = CosmosInterior::new(vec![make_block(BlockConfig::new(vec![0]))]);

        interior.set_epoch_batch(3, 7);

        assert_eq!(interior.epoch(), 3);
        assert_eq!(interior.batch_idx(), 7);
    }

    #[test]
    fn forward_with_flow_matches_forward_and_concatenates_block_records() {
        let state = build_state(&[x_axis(), diagonal_xy()]);
        let make = || {
            make_block(BlockConfig {
                block_indices: vec![0, 1],
                composition: CompositionParams::new(
                    "residual_series",
                    ParallelMode::Stack,
                    "none",
                    "none",
                ),
                ..Default::default()
            })
        };
        let x = array![[1.0, 2.0], [3.0, -1.0]];

        let mut interior =
            CosmosInterior::with_options(vec![make(), make()], Activation::Relu, true);
        let plain = interior.forward(&x, &state).unwrap();
        let (flowed, records) = interior.forward_with_flow(&x, &state).unwrap();

        assert_eq!(plain, flowed);
        // Two residual-series blocks over two projectors each.
        assert_eq!(records.len(), 4);
    }

    #[test]
    fn simple_series_constructor_runs() {
        let state = build_state(&[x_axis()]);
        let mut interior = CosmosInterior::simple_series(vec![0]).unwrap();

        let out = interior.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_eq!(interior.num_blocks(), 1);
        assert_array2_close(&out, &array![[1.0, 0.0]]);
    }

    #[test]
    fn parallel_pool_constructor_averages_parallel_outputs() {
        let state = build_state(&[x_axis(), y_axis()]);
        let mut interior = CosmosInterior::parallel_pool(vec![0, 1]).unwrap();

        let out = interior.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_array2_close(&out, &array![[0.5, 1.0]]);
    }

    #[test]
    fn residual_chain_constructor_adds_skip_connection() {
        let state = build_state(&[x_axis()]);
        let mut interior = CosmosInterior::residual_chain(vec![0]).unwrap();

        let out = interior.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_array2_close(&out, &array![[2.0, 2.0]]);
    }
}
