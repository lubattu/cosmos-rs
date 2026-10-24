//! Serializable arrangement specs and fluent builders.
//!
//! This module provides two complementary layers:
//!
//! - [`BuilderBlockConfig`] and [`InteriorSpec`]: immutable, serializable
//!   specifications for arrangement runtime objects
//! - [`BlockBuilder`] and [`InteriorBuilder`]: fluent helpers for constructing
//!   those specs and the corresponding runtime values

use serde::{Deserialize, Serialize};

use crate::arrangement::{
    ArrangementError,
    activations::{Activation, ActivationPlacement},
    blocks::{BlockConfig, ModularBlock},
    interior::CosmosInterior,
    params::{
        AggregationParams, CompositionParams, OrderingParams, ParallelMode, SelectionParams,
        SubspaceAction,
    },
};

/// Serializable specification for one modular routing block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuilderBlockConfig {
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

impl BuilderBlockConfig {
    /// Create a block spec over the provided candidate indices.
    #[must_use]
    pub fn new(block_indices: Vec<usize>) -> Self {
        Self {
            block_indices,
            ..Default::default()
        }
    }

    /// Convert this builder spec into the runtime block config.
    #[must_use]
    pub fn to_block_config(&self) -> BlockConfig {
        BlockConfig {
            block_indices: self.block_indices.clone(),
            selection: self.selection.clone(),
            ordering: self.ordering.clone(),
            composition: self.composition.clone(),
            aggregation: self.aggregation.clone(),
        }
    }

    /// Build a runtime block from this specification.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn build(&self) -> Result<ModularBlock, ArrangementError> {
        ModularBlock::new(self.to_block_config())
    }
}

impl From<BlockConfig> for BuilderBlockConfig {
    fn from(config: BlockConfig) -> Self {
        Self {
            block_indices: config.block_indices,
            selection: config.selection,
            ordering: config.ordering,
            composition: config.composition,
            aggregation: config.aggregation,
        }
    }
}

/// Immutable specification for a full arrangement interior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteriorSpec {
    /// Ordered list of block specifications.
    pub blocks: Vec<BuilderBlockConfig>,

    /// Global activation applied after all blocks run.
    pub global_activation: Activation,

    /// Whether the final output should be row-wise normalized.
    pub normalize_final: bool,
}

impl InteriorSpec {
    /// Create a new interior spec from block specifications.
    #[must_use]
    pub fn new(blocks: Vec<BuilderBlockConfig>) -> Self {
        Self {
            blocks,
            ..Default::default()
        }
    }

    /// Borrow the configured block specs.
    #[must_use]
    pub fn blocks(&self) -> &[BuilderBlockConfig] {
        &self.blocks
    }

    /// Build the runtime interior represented by this spec.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn build(&self) -> Result<CosmosInterior, ArrangementError> {
        CosmosInterior::from_spec(self)
    }
}

impl Default for InteriorSpec {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            global_activation: Activation::Identity,
            normalize_final: false,
        }
    }
}

/// Fluent builder for one modular block.
#[derive(Debug, Clone, Default)]
pub struct BlockBuilder {
    config: BuilderBlockConfig,
}

impl BlockBuilder {
    /// Start a new block builder with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a block builder from an existing spec.
    #[must_use]
    pub fn from_config(config: BuilderBlockConfig) -> Self {
        Self { config }
    }

    /// Set the candidate block indices.
    #[must_use]
    pub fn with_indices(mut self, block_indices: Vec<usize>) -> Self {
        self.config.block_indices = block_indices;
        self
    }

    /// Replace the selection parameters wholesale.
    #[must_use]
    pub fn select(mut self, selection: SelectionParams) -> Self {
        self.config.selection = selection;
        self
    }

    /// Select all candidate blocks.
    #[must_use]
    pub fn select_all(mut self) -> Self {
        self.config.selection = SelectionParams::default();
        self
    }

    /// Replace the ordering parameters wholesale.
    #[must_use]
    pub fn order(mut self, ordering: OrderingParams) -> Self {
        self.config.ordering = ordering;
        self
    }

    /// Keep the candidate order unchanged.
    #[must_use]
    pub fn order_fixed(mut self) -> Self {
        self.config.ordering = OrderingParams::default();
        self
    }

    /// Replace the composition parameters wholesale.
    #[must_use]
    pub fn compose(mut self, composition: CompositionParams) -> Self {
        self.config.composition = composition;
        self
    }

    /// Configure sequential projector composition.
    #[must_use]
    pub fn compose_series(mut self) -> Self {
        self.config.composition =
            CompositionParams::new("series", ParallelMode::Stack, "none", "none");
        self
    }

    /// Configure parallel projector composition.
    #[must_use]
    pub fn compose_parallel(mut self) -> Self {
        self.config.composition =
            CompositionParams::new("parallel", ParallelMode::Stack, "none", "none");
        self
    }

    /// Configure residual sequential projector composition.
    #[must_use]
    pub fn compose_residual_series(mut self) -> Self {
        self.config.composition =
            CompositionParams::new("residual_series", ParallelMode::Stack, "none", "none");
        self
    }

    /// Configure two-phase parallel-then-series projector composition.
    #[must_use]
    pub fn compose_parallel_series(mut self) -> Self {
        self.config.composition =
            CompositionParams::new("parallel_series", ParallelMode::Stack, "none", "none");
        self
    }

    /// Configure activation placement within the current composition parameters.
    #[must_use]
    pub fn activate(mut self, activation: Activation, placement: ActivationPlacement) -> Self {
        self.config.composition.activation_name = activation.to_string();
        self.config.composition.activation_placement = placement.to_string();
        self
    }

    /// Configure the gauge-invariant subspace action.
    #[must_use]
    pub fn subspace_action(mut self, action: SubspaceAction) -> Self {
        self.config.composition.subspace_action = action.to_string();
        self
    }

    /// Replace the aggregation parameters wholesale.
    #[must_use]
    pub fn aggregate(mut self, aggregation: AggregationParams) -> Self {
        self.config.aggregation = Some(aggregation);
        self
    }

    /// Configure arithmetic-mean aggregation.
    #[must_use]
    pub fn aggregate_mean(mut self) -> Self {
        self.config.aggregation = Some(AggregationParams::default());
        self
    }

    /// Remove aggregation from the block configuration.
    #[must_use]
    pub fn without_aggregation(mut self) -> Self {
        self.config.aggregation = None;
        self
    }

    /// Finish the builder and return the immutable block spec.
    #[must_use]
    pub fn build_config(self) -> BuilderBlockConfig {
        self.config
    }

    /// Build the runtime block directly.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn build(self) -> Result<ModularBlock, ArrangementError> {
        self.build_config().build()
    }
}

/// Fluent builder for a full arrangement interior.
#[derive(Debug, Clone, Default)]
pub struct InteriorBuilder {
    spec: InteriorSpec,
}

impl InteriorBuilder {
    /// Start a new interior builder with default output settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start an interior builder from an existing immutable spec.
    #[must_use]
    pub fn from_spec(spec: InteriorSpec) -> Self {
        Self { spec }
    }

    /// Add an immutable block spec.
    #[must_use]
    pub fn add_block(mut self, block: BuilderBlockConfig) -> Self {
        self.spec.blocks.push(block);
        self
    }

    /// Add a block from a fluent block builder.
    #[must_use]
    pub fn add_block_builder(self, builder: BlockBuilder) -> Self {
        self.add_block(builder.build_config())
    }

    /// Add a series block.
    #[must_use]
    pub fn add_series_block(self, block_indices: Vec<usize>) -> Self {
        self.add_block_builder(
            BlockBuilder::new()
                .with_indices(block_indices)
                .compose_series(),
        )
    }

    /// Add a parallel block with mean aggregation.
    #[must_use]
    pub fn add_parallel_block(self, block_indices: Vec<usize>) -> Self {
        self.add_block_builder(
            BlockBuilder::new()
                .with_indices(block_indices)
                .compose_parallel()
                .aggregate_mean(),
        )
    }

    /// Add a residual series block.
    #[must_use]
    pub fn add_residual_block(self, block_indices: Vec<usize>) -> Self {
        self.add_block_builder(
            BlockBuilder::new()
                .with_indices(block_indices)
                .compose_residual_series(),
        )
    }

    /// Set the final global activation.
    #[must_use]
    pub fn with_global_activation(mut self, global_activation: Activation) -> Self {
        self.spec.global_activation = global_activation;
        self
    }

    /// Set whether the final output should be row-wise normalized.
    #[must_use]
    pub fn with_normalize_final(mut self, normalize_final: bool) -> Self {
        self.spec.normalize_final = normalize_final;
        self
    }

    /// Finish the builder and return the immutable interior spec.
    #[must_use]
    pub fn build_spec(self) -> InteriorSpec {
        self.spec
    }

    /// Build the runtime interior directly.
    ///
    /// # Errors
    ///
    /// Propagates the failures of [`ModularBlock::new`].
    pub fn build(self) -> Result<CosmosInterior, ArrangementError> {
        self.spec.build()
    }

    /// Build a one-block series interior.
    #[must_use]
    pub fn simple_series(block_indices: Vec<usize>) -> Self {
        Self::new().add_series_block(block_indices)
    }

    /// Build a one-block parallel-pooling interior.
    #[must_use]
    pub fn parallel_pool(block_indices: Vec<usize>) -> Self {
        Self::new().add_parallel_block(block_indices)
    }

    /// Build a one-block residual-chain interior.
    #[must_use]
    pub fn residual_chain(block_indices: Vec<usize>) -> Self {
        Self::new().add_residual_block(block_indices)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::state::cosmos_state::CosmosState;

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

    fn build_state(blocks: &[Array2<f64>]) -> CosmosState {
        let mut state = CosmosState::new(&[]).unwrap();
        for block in blocks {
            state.register_external_block(block).unwrap();
        }
        state
    }

    #[test]
    fn builder_block_config_builds_runtime_block() {
        let spec = BuilderBlockConfig {
            block_indices: vec![0, 1],
            ..Default::default()
        };

        let block = spec.build().unwrap();

        assert_eq!(block.config().block_indices, vec![0, 1]);
    }

    #[test]
    fn block_builder_fluent_configuration_updates_expected_fields() {
        let spec = BlockBuilder::new()
            .with_indices(vec![1, 3, 5])
            .select_all()
            .order_fixed()
            .compose_parallel()
            .activate(Activation::Relu, ActivationPlacement::Post)
            .subspace_action(SubspaceAction::reflection())
            .aggregate_mean()
            .build_config();

        assert_eq!(spec.block_indices, vec![1, 3, 5]);
        assert_eq!(spec.selection.strategy, "all");
        assert_eq!(spec.ordering.strategy, "fixed");
        assert_eq!(spec.composition.strategy, "parallel");
        assert_eq!(spec.composition.activation_name, "relu");
        assert_eq!(spec.composition.activation_placement, "post");
        assert_eq!(spec.composition.subspace_action, "reflection");
        assert_eq!(spec.aggregation.as_ref().unwrap().strategy, "mean");
    }

    #[test]
    fn block_builder_rejects_an_unknown_strategy() {
        let err = BlockBuilder::new()
            .with_indices(vec![0])
            .select(SelectionParams {
                strategy: "roulette".to_string(),
            })
            .build()
            .unwrap_err();

        assert!(matches!(
            err,
            ArrangementError::UnknownStrategy {
                primitive: "selection",
                ..
            }
        ));
    }

    #[test]
    fn block_builder_build_produces_runtime_block() {
        let block = BlockBuilder::new()
            .with_indices(vec![0])
            .compose_series()
            .build()
            .unwrap();

        assert_eq!(block.config().block_indices, vec![0]);
    }

    #[test]
    fn interior_spec_builds_runtime_interior() {
        let spec = InteriorSpec {
            blocks: vec![
                BlockBuilder::new()
                    .with_indices(vec![0])
                    .compose_series()
                    .build_config(),
            ],
            global_activation: Activation::Relu,
            normalize_final: true,
        };

        let interior = spec.build().unwrap();

        assert_eq!(interior.num_blocks(), 1);
        assert_eq!(interior.global_activation(), Activation::Relu);
        assert!(interior.normalize_final());
    }

    #[test]
    fn interior_builder_build_runs_forward() {
        let state = build_state(&[x_axis()]);
        let mut interior = InteriorBuilder::simple_series(vec![0]).build().unwrap();

        let out = interior.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 0.0]]);
    }

    #[test]
    fn interior_builder_parallel_pool_matches_runtime_constructor() {
        let state = build_state(&[x_axis(), y_axis()]);
        let mut built = InteriorBuilder::parallel_pool(vec![0, 1]).build().unwrap();
        let mut manual = CosmosInterior::parallel_pool(vec![0, 1]).unwrap();

        let built_out = built.forward(&array![[1.0, 2.0]], &state).unwrap();
        let manual_out = manual.forward(&array![[1.0, 2.0]], &state).unwrap();

        assert_array2_close(&built_out, &manual_out);
    }

    #[test]
    fn builder_block_config_serde_roundtrip() {
        let spec = BuilderBlockConfig {
            block_indices: vec![2, 4],
            aggregation: Some(AggregationParams::default()),
            ..Default::default()
        };

        let json = serde_json::to_string(&spec).unwrap();
        let decoded: BuilderBlockConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, spec);
    }

    #[test]
    fn interior_spec_deserialization_rejects_unknown_fields() {
        let json = r#"{
            "blocks": [],
            "unexpected_field": null,
            "global_activation": "Identity",
            "normalize_final": false
        }"#;

        let err = serde_json::from_str::<InteriorSpec>(json).unwrap_err();

        assert!(
            err.to_string().contains("unexpected_field"),
            "expected the unknown key to be named, got: {err}"
        );
    }

    #[test]
    fn builder_block_config_deserialization_rejects_unknown_fields() {
        let json = r#"{
            "block_indices": [0],
            "selection": { "strategy": "all" },
            "ordering": { "strategy": "fixed" },
            "composition": {
                "strategy": "series",
                "parallel_mode": "Stack",
                "activation_name": "none",
                "activation_placement": "none",
                "subspace_action": "projection"
            },
            "aggregation": null,
            "not_a_parameter": 42
        }"#;

        let err = serde_json::from_str::<BuilderBlockConfig>(json).unwrap_err();

        assert!(
            err.to_string().contains("not_a_parameter"),
            "expected the unknown key to be named, got: {err}"
        );
    }

    #[test]
    fn interior_spec_serde_roundtrip() {
        let spec = InteriorBuilder::new()
            .add_parallel_block(vec![0, 1])
            .with_global_activation(Activation::Tanh)
            .with_normalize_final(true)
            .build_spec();

        let json = serde_json::to_string(&spec).unwrap();
        let decoded: InteriorSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, spec);
    }
}
