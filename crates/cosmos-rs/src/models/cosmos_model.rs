//! Flat-input COSMOS classification model.
//!
//! [`CosmosModel`] is the default classification composition:
//!
//! `input -> FlatInputProjector -> CosmosInterior -> geodesic CMR scoring`
//!
//! All learnable parameters remain inside
//! [`CosmosState`]. The arrangement
//! modules are parameter-free and are composed here into a runnable model.

use std::{collections::HashMap, ops::Range};

use ndarray::Array2;

use crate::{
    arrangement::{
        activations::{Activation, ActivationPlacement},
        blocks::BlockConfig,
        builder::InteriorSpec,
        compositions::{FlowNorm, mean_row_norm},
        input::flat_input::FlatInputProjector,
        params::SubspaceAction,
    },
    objectives::cmr_losses::RegionMetrics,
    state::cosmos_state::CosmosState,
};

use super::{
    base::{Model, ModelError},
    common::{
        InteriorRuntimeConfig, InteriorTopology, ModelCore, ModelHead, block_parameter_count,
        validate_positive,
    },
};

/// Immutable constructor configuration for [`CosmosModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct CosmosModelConfig {
    /// Ambient feature dimension of the flat input.
    pub input_dim: usize,
    /// Subspace dimension of the input projector block.
    pub input_subspace_dim: usize,
    /// Subspace dimension of each interior projector block.
    pub interior_subspace_dim: usize,
    /// Number of projector blocks available to the interior runtime.
    pub num_interior_blocks: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Subspace dimension of each CMR anchor block.
    pub cmr_k: usize,
    /// Number of anchors assigned to each class.
    pub n_anchors_per_class: usize,
    /// Interior shortcut used when `interior_spec` is absent.
    pub interior_topology: InteriorTopology,
    /// Optional explicit interior specification overriding `interior_topology`.
    pub interior_spec: Option<InteriorSpec>,
    /// Activation used inside interior composition presets.
    pub activation: Activation,
    /// Placement used for the interior activation.
    pub activation_placement: ActivationPlacement,
    /// Whether interior blocks project or reflect through subspaces.
    pub action: SubspaceAction,
    /// Emission head terminating the model.
    pub head: ModelHead,
}

impl CosmosModelConfig {
    /// Create a config with project-wide defaults for a flat classifier.
    pub fn new(input_dim: usize, num_classes: usize) -> Self {
        Self {
            input_dim,
            input_subspace_dim: 128,
            interior_subspace_dim: 64,
            num_interior_blocks: 3,
            num_classes,
            cmr_k: 32,
            n_anchors_per_class: 1,
            interior_topology: InteriorTopology::Series,
            interior_spec: None,
            activation: Activation::Identity,
            activation_placement: ActivationPlacement::None,
            action: SubspaceAction::default(),
            head: ModelHead::Cmr,
        }
    }

    /// Override the input subspace dimension.
    pub fn with_input_subspace_dim(mut self, input_subspace_dim: usize) -> Self {
        self.input_subspace_dim = input_subspace_dim;
        self
    }

    /// Override the interior subspace dimension.
    pub fn with_interior_subspace_dim(mut self, interior_subspace_dim: usize) -> Self {
        self.interior_subspace_dim = interior_subspace_dim;
        self
    }

    /// Override the number of interior blocks.
    pub fn with_num_interior_blocks(mut self, num_interior_blocks: usize) -> Self {
        self.num_interior_blocks = num_interior_blocks;
        self
    }

    /// Override the CMR anchor subspace dimension.
    pub fn with_cmr_k(mut self, cmr_k: usize) -> Self {
        self.cmr_k = cmr_k;
        self
    }

    /// Override the number of anchors per class.
    pub fn with_anchors_per_class(mut self, n_anchors_per_class: usize) -> Self {
        self.n_anchors_per_class = n_anchors_per_class;
        self
    }

    /// Use a topology shortcut when no explicit spec is present.
    pub fn with_interior_topology(mut self, interior_topology: InteriorTopology) -> Self {
        self.interior_topology = interior_topology;
        self
    }

    /// Supply a full interior specification.
    pub fn with_interior_spec(mut self, interior_spec: InteriorSpec) -> Self {
        self.interior_spec = Some(interior_spec);
        self
    }

    /// Override the activation and placement used by topology presets.
    pub fn with_activation(
        mut self,
        activation: Activation,
        activation_placement: ActivationPlacement,
    ) -> Self {
        self.activation = activation;
        self.activation_placement = activation_placement;
        self
    }

    /// Override the subspace action for interior blocks.
    pub fn with_subspace_action(mut self, action: SubspaceAction) -> Self {
        self.action = action;
        self
    }

    fn validate(&self) -> Result<(), ModelError> {
        validate_positive("input_dim", self.input_dim)?;
        validate_positive("input_subspace_dim", self.input_subspace_dim)?;
        validate_positive("num_classes", self.num_classes)?;
        validate_positive("cmr_k", self.cmr_k)?;
        validate_positive("n_anchors_per_class", self.n_anchors_per_class)?;

        if self.num_interior_blocks > 0 {
            validate_positive("interior_subspace_dim", self.interior_subspace_dim)?;
        }

        // Every subspace dimension must fit inside the ambient input
        // dimension — Gr(k, n) is only defined for k ≤ n.
        if self.input_subspace_dim > self.input_dim {
            return Err(ModelError::InvalidParameter {
                param: "input_subspace_dim",
                reason: format!(
                    "must be ≤ input_dim, got input_subspace_dim={} > input_dim={}",
                    self.input_subspace_dim, self.input_dim
                ),
            });
        }
        // `cmr_k` sizes the anchor blocks the CMR head owns.
        if self.cmr_k > self.input_subspace_dim {
            return Err(ModelError::InvalidParameter {
                param: "cmr_k",
                reason: format!(
                    "must be ≤ input_subspace_dim, got cmr_k={} > input_subspace_dim={}",
                    self.cmr_k, self.input_subspace_dim
                ),
            });
        }
        if self.num_interior_blocks > 0 && self.interior_subspace_dim > self.input_dim {
            return Err(ModelError::InvalidParameter {
                param: "interior_subspace_dim",
                reason: format!(
                    "must be ≤ input_dim, got interior_subspace_dim={} > input_dim={}",
                    self.interior_subspace_dim, self.input_dim
                ),
            });
        }

        Ok(())
    }
}

/// Flat-input COSMOS classification model.
#[derive(Debug)]
pub struct CosmosModel {
    core: ModelCore,
    input_projector: FlatInputProjector,
    input_dim: usize,
    input_subspace_dim: usize,
    interior_subspace_dim: usize,
    input_block_idx: usize,
}

impl CosmosModel {
    /// Build a flat-input COSMOS model from a typed config.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when a configured dimension is zero
    /// or exceeds the ambient input dimension, and [`ModelError::State`] when the
    /// resulting block layout cannot be allocated.
    pub fn new(config: CosmosModelConfig) -> Result<Self, ModelError> {
        config.validate()?;

        let input_block_idx = 0;

        let head_block_count = config.num_classes * config.n_anchors_per_class;
        let mut projector_shapes =
            Vec::with_capacity(1 + config.num_interior_blocks + head_block_count);
        projector_shapes.push((config.input_dim, config.input_subspace_dim));
        for _ in 0..config.num_interior_blocks {
            projector_shapes.push((config.input_dim, config.interior_subspace_dim));
        }
        for _ in 0..head_block_count {
            projector_shapes.push((config.input_dim, config.cmr_k));
        }

        let state = CosmosState::new(&projector_shapes)?;
        let interior_config = InteriorRuntimeConfig {
            num_interior_blocks: config.num_interior_blocks,
            interior_topology: config.interior_topology,
            interior_spec: config.interior_spec.clone(),
            activation: config.activation,
            activation_placement: config.activation_placement,
            action: config.action,
        };
        let core = ModelCore::new_with_head(
            state,
            config.num_classes,
            config.cmr_k,
            config.n_anchors_per_class,
            1,
            &interior_config,
            config.head,
        )?;

        Ok(Self {
            core,
            input_projector: FlatInputProjector::new(input_block_idx),
            input_dim: config.input_dim,
            input_subspace_dim: config.input_subspace_dim,
            interior_subspace_dim: config.interior_subspace_dim,
            input_block_idx,
        })
    }

    /// Map a raw batch into the space the interior chain runs in.
    ///
    /// The input projector re-expands into the ambient input
    /// (`x W_0 W_0^T`, width `input_dim`), which is the space the interior
    /// blocks and the CMR anchors are defined on.
    fn forward_input_head(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        self.input_projector
            .forward(x, self.core.state())
            .map_err(ModelError::from)
    }

    /// Convenience wrapper over the trait-based batch forward path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, when the interior runtime is already borrowed, or when the
    /// anchor layout is inconsistent.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        <Self as Model>::forward(self, x)
    }

    /// Convenience wrapper returning scores and region metrics.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, when the interior runtime is already borrowed, or when the
    /// anchor layout is inconsistent.
    pub fn forward_with_metrics(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
        <Self as Model>::forward_with_metrics(self, x)
    }

    /// Convenience wrapper over the trait-based embedding extraction path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, or when the interior runtime is already borrowed.
    pub fn get_embeddings(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        <Self as Model>::get_embeddings(self, x)
    }

    /// Flow diagnostics along the interior data path.
    ///
    /// Traverses the same stages as [`CosmosModel::get_embeddings`]: the
    /// input projection produces the interior input `x_int`, then the
    /// interior runs with per-projector flow recording. Returns the
    /// per-projector [`FlowNorm`] records plus an overall ratio
    ///
    /// `mean row norm of the interior output / mean row norm of x_int`
    ///
    /// where the output norm is measured after the interior's global
    /// activation but before any final row normalization — normalized rows
    /// all have unit norm, which would erase the dilution signal this ratio
    /// tracks (the per-block records already capture the chain).
    ///
    /// Interior mutability flows through the runtime's interior cell, so the
    /// method takes `&self` like the regular forward path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, or when the interior runtime is already borrowed.
    pub fn interior_flow_norms(&self, x: &Array2<f64>) -> Result<(Vec<FlowNorm>, f64), ModelError> {
        let x_int = self.forward_input_head(x)?;
        let input_norm = mean_row_norm(&x_int);
        let (_output, records, output_norm) = self.core.forward_interior_with_flow(&x_int)?;

        let overall_ratio = if input_norm > 0.0 {
            output_norm / input_norm
        } else {
            0.0
        };

        Ok((records, overall_ratio))
    }

    /// Human-readable model summary.
    pub fn summary(&self) -> String {
        self.render_summary()
    }

    /// Parameter counts split by major model component.
    pub fn count_parameters(&self) -> HashMap<String, usize> {
        <Self as Model>::count_parameters(self)
    }

    /// Borrow the unified COSMOS state.
    pub fn get_state(&self) -> &CosmosState {
        self.core.state()
    }

    /// Mutably borrow the unified COSMOS state.
    pub fn get_state_mut(&mut self) -> &mut CosmosState {
        self.core.state_mut()
    }

    /// Index of the input projector block.
    pub fn input_block_idx(&self) -> usize {
        self.input_block_idx
    }

    /// Interior block index range.
    pub fn interior_block_range(&self) -> Range<usize> {
        self.core.interior_block_range()
    }

    /// Every interior subspace action `(alpha, beta)`, in interior block order.
    ///
    /// Runtime controllers may rewrite these during training, so they belong
    /// to the state a forward pass reads rather than to the constructor
    /// config.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn interior_actions(&self) -> Result<Vec<SubspaceAction>, ModelError> {
        self.core.interior_actions()
    }

    /// Replace every interior subspace action, in the order
    /// [`CosmosModel::interior_actions`] reports them.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] when the count differs from what
    /// [`CosmosModel::interior_actions`] reports, or
    /// [`ModelError::RuntimeStateConflict`] when the interior runtime is already
    /// borrowed.
    pub fn set_interior_actions(&self, actions: &[SubspaceAction]) -> Result<(), ModelError> {
        self.core.set_interior_actions(actions)
    }

    /// CMR anchor block index range.
    pub fn cmr_block_range(&self) -> Range<usize> {
        self.core.cmr_block_range()
    }

    /// Emission head terminating the model.
    pub fn head(&self) -> ModelHead {
        self.core.head()
    }

    /// Every state block owned by the emission head.
    pub fn head_block_range(&self) -> Range<usize> {
        self.core.head_block_range()
    }

    /// Configured interior shortcut, if the runtime was not built from an explicit spec.
    pub fn interior_topology(&self) -> Option<InteriorTopology> {
        self.core.interior_topology()
    }

    /// Recompose the interior under a different topology preset.
    ///
    /// The interior topology is a gradient-free arrangement over
    /// topology-independent learned state: the interior projector blocks
    /// `V_j`, the input basis `W_0`, and the CMR anchors have identical
    /// shapes under every preset, so switching the composition is a
    /// well-defined operation on the same [`CosmosState`]. This makes
    /// mid-training swaps (curriculum, quench, hysteresis protocols) a
    /// state-preserving change of the arrangement alone.
    ///
    /// Errors when the model was built from an explicit interior spec.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when the model was built from an
    /// explicit interior spec, [`ModelError::RuntimeStateConflict`] when the
    /// interior runtime is already borrowed, and [`ModelError::Arrangement`] when
    /// the new topology fails to recompose the interior blocks.
    pub fn set_interior_topology(&mut self, topology: InteriorTopology) -> Result<(), ModelError> {
        self.core.set_interior_topology(topology)
    }

    /// Interior subspace dimension `k` of every interior projector block.
    pub fn interior_subspace_dim(&self) -> usize {
        self.interior_subspace_dim
    }

    /// Whether the model uses a custom interior specification.
    pub fn uses_custom_interior_spec(&self) -> bool {
        self.core.uses_custom_interior_spec()
    }

    /// Activation configured for the default interior shortcut.
    pub fn interior_activation(&self) -> Activation {
        self.core.interior_activation()
    }

    /// Activation placement configured for the default interior shortcut.
    pub fn interior_activation_placement(&self) -> ActivationPlacement {
        self.core.interior_activation_placement()
    }

    /// Number of interior state blocks.
    pub fn num_interior_state_blocks(&self) -> usize {
        self.core.num_interior_state_blocks()
    }

    /// Global indices of interior state blocks.
    pub fn interior_state_block_indices(&self) -> Vec<usize> {
        self.core.interior_block_indices()
    }

    /// Snapshot the block configuration of each interior modular block.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub(crate) fn interior_block_configs(&self) -> Result<Vec<BlockConfig>, ModelError> {
        self.core.interior_block_configs()
    }

    /// Runtime per-projector subspace actions for each interior block.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn interior_block_actions(&self) -> Result<Vec<Vec<SubspaceAction>>, ModelError> {
        self.core.interior_block_actions()
    }

    /// Global activation the interior runtime applies after composition.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub(crate) fn interior_global_activation(&self) -> Result<Activation, ModelError> {
        self.core.interior_global_activation()
    }

    /// Whether the interior normalizes its final output row-wise.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub(crate) fn interior_normalize_final(&self) -> Result<bool, ModelError> {
        self.core.interior_normalize_final()
    }

    /// Mirror the interior training flag.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn is_training(&self) -> Result<bool, ModelError> {
        self.core.is_training()
    }

    /// Forward training mode changes to the interior runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn set_training(&self, training: bool) -> Result<(), ModelError> {
        self.core.set_training(training)
    }

    /// Forward epoch/batch updates to the interior runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError> {
        self.core.set_epoch_batch(epoch, batch_idx)
    }

    /// Notify the interior runtime that a new epoch has started.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError> {
        self.core.on_epoch_start(epoch)
    }

    fn render_summary(&self) -> String {
        let counts = <Self as Model>::count_parameters(self);
        let topology_label = if self.uses_custom_interior_spec() {
            "custom_spec".to_string()
        } else {
            self.interior_topology()
                .unwrap_or(InteriorTopology::Series)
                .to_string()
        };

        let head_desc = match self.head() {
            ModelHead::Cmr if self.core.anchors_per_class() == 1 => format!(
                "  CMR: {} x Gr({}, {})",
                self.core.num_classes(),
                self.core.cmr_k(),
                self.input_dim
            ),
            ModelHead::Cmr => format!(
                "  CMR: {} x {} anchors x Gr({}, {})",
                self.core.num_classes(),
                self.core.anchors_per_class(),
                self.core.cmr_k(),
                self.input_dim
            ),
        };

        [
            "CosmosModel (Flat Input)".to_string(),
            format!("  Ambient dimension: {}", self.input_dim),
            format!(
                "  Input: Gr({}, {})",
                self.input_subspace_dim, self.input_dim
            ),
            format!(
                "  Interior: {} blocks, topology {}, subspace dim {}",
                self.core.num_interior_blocks(),
                topology_label,
                self.interior_subspace_dim
            ),
            head_desc,
            format!("  Total blocks: {}", self.core.total_blocks()),
            format!("  Parameters: {} (all in CosmosState)", counts["total"]),
        ]
        .join("\n")
    }
}

impl Model for CosmosModel {
    type Input = Array2<f64>;

    fn state(&self) -> &CosmosState {
        self.core.state()
    }

    fn state_mut(&mut self) -> &mut CosmosState {
        self.core.state_mut()
    }

    fn cmr_block_indices(&self) -> &[usize] {
        self.core.cmr_block_indices()
    }

    fn num_classes(&self) -> usize {
        self.core.num_classes()
    }

    fn anchors_per_class(&self) -> usize {
        self.core.anchors_per_class()
    }

    fn get_embeddings(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        let projected = self.forward_input_head(x)?;
        self.core.forward_interior(&projected)
    }

    /// Per-class scores, higher meaning closer: the Gaussian-angular CMR
    /// membership `s = exp(-theta^2)`.
    fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        let embeddings = self.get_embeddings(x)?;
        self.compute_cmr_scores(&embeddings)
    }

    /// CMR scores with per-class region metrics.
    fn forward_with_metrics(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
        let embeddings = self.get_embeddings(x)?;
        self.compute_cmr_scores_with_metrics(&embeddings)
    }

    fn summary(&self) -> String {
        self.render_summary()
    }

    fn count_parameters(&self) -> HashMap<String, usize> {
        let block_shapes = self.core.state().block_shapes();
        let input_block = block_parameter_count(block_shapes[self.input_block_idx]);
        let interior_blocks = self
            .interior_block_range()
            .map(|idx| block_parameter_count(block_shapes[idx]))
            .sum();
        let cmr_blocks = self
            .cmr_block_range()
            .map(|idx| block_parameter_count(block_shapes[idx]))
            .sum();
        let head_blocks: usize = self
            .head_block_range()
            .map(|idx| block_parameter_count(block_shapes[idx]))
            .sum();
        let state_total = input_block + interior_blocks + head_blocks;

        HashMap::from([
            ("input_block".to_string(), input_block),
            ("interior_blocks".to_string(), interior_blocks),
            ("cmr_blocks".to_string(), cmr_blocks),
            ("head_blocks".to_string(), head_blocks),
            ("state_total".to_string(), state_total),
            ("total".to_string(), state_total),
        ])
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array, s};

    use crate::arrangement::builder::InteriorBuilder;

    use super::*;

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

    fn set_global_blocks(model: &mut CosmosModel, blocks: &[Array2<f64>]) {
        let block_shapes = model.core.state().block_shapes().to_vec();
        assert_eq!(blocks.len(), block_shapes.len());

        let total_n: usize = block_shapes.iter().map(|(n, _)| n).sum();
        let total_k: usize = block_shapes.iter().map(|(_, k)| k).sum();
        let mut global = Array2::<f64>::zeros((total_n, total_k));

        let mut row_offset = 0;
        let mut col_offset = 0;
        for ((n, k), block) in block_shapes.iter().copied().zip(blocks.iter()) {
            assert_eq!(block.dim(), (n, k));
            global
                .slice_mut(s![row_offset..row_offset + n, col_offset..col_offset + k])
                .assign(block);
            row_offset += n;
            col_offset += k;
        }

        model.core.state_mut().update_from_global_basis(&global);
    }

    #[test]
    fn config_new_sets_expected_defaults() {
        let config = CosmosModelConfig::new(784, 10);

        assert_eq!(config.input_dim, 784);
        assert_eq!(config.num_classes, 10);
        assert_eq!(config.input_subspace_dim, 128);
        assert_eq!(config.interior_subspace_dim, 64);
        assert_eq!(config.num_interior_blocks, 3);
        assert_eq!(config.cmr_k, 32);
        assert_eq!(config.n_anchors_per_class, 1);
        assert_eq!(config.interior_topology, InteriorTopology::Series);
        assert!(config.interior_spec.is_none());
        assert_eq!(config.activation, Activation::Identity);
        assert_eq!(config.activation_placement, ActivationPlacement::None);
    }

    #[test]
    fn new_rejects_zero_num_classes() {
        let err = CosmosModel::new(CosmosModelConfig::new(4, 0)).unwrap_err();

        match err {
            ModelError::InvalidParameter { param, .. } => assert_eq!(param, "num_classes"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn embeddings_match_input_projection_when_no_interior_blocks() {
        let model = CosmosModel::new(
            CosmosModelConfig::new(2, 1)
                .with_input_subspace_dim(1)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();

        let embeddings = model
            .get_embeddings(&array![[1.0, 2.0], [-3.0, 4.0]])
            .unwrap();

        assert_array2_close(&embeddings, &array![[1.0, 0.0], [-3.0, 0.0]]);
    }

    #[test]
    fn forward_matches_known_two_class_axis_scores() {
        let mut model = CosmosModel::new(
            CosmosModelConfig::new(2, 2)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();
        let off_axis = (-(std::f64::consts::FRAC_PI_2.powi(2))).exp();

        set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), y_axis()]);

        let (scores, metrics) = model
            .forward_with_metrics(&array![[1.0, 0.0], [0.0, 1.0]])
            .unwrap();

        assert_array2_close(&scores, &array![[1.0, off_axis], [off_axis, 1.0]]);
        assert_eq!(metrics.len(), 2);
        assert_array2_close(
            &metrics[0].principal_angles,
            &array![[0.0], [std::f64::consts::FRAC_PI_2]],
        );
        assert_array2_close(
            &metrics[1].principal_angles,
            &array![[std::f64::consts::FRAC_PI_2], [0.0]],
        );
    }

    #[test]
    fn explicit_interior_spec_overrides_topology_shortcut() {
        let spec = InteriorBuilder::new()
            .with_normalize_final(true)
            .build_spec();
        let model = CosmosModel::new(
            CosmosModelConfig::new(2, 1)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(1)
                .with_interior_subspace_dim(2)
                .with_cmr_k(1)
                .with_interior_topology(InteriorTopology::Residual)
                .with_interior_spec(spec),
        )
        .unwrap();

        let embeddings = model.get_embeddings(&array![[3.0, 4.0]]).unwrap();
        let interior = model.core.borrow_interior().unwrap();

        assert_eq!(interior.num_blocks(), 0);
        assert!(interior.normalize_final());
        assert_array2_close(&embeddings, &array![[0.6, 0.8]]);
        assert!(model.uses_custom_interior_spec());
        assert_eq!(model.interior_topology(), None);
    }

    #[test]
    fn interior_flow_norms_track_projection_dilution() {
        let mut model = CosmosModel::new(
            CosmosModelConfig::new(2, 1)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(1)
                .with_interior_subspace_dim(1)
                .with_cmr_k(1),
        )
        .unwrap();
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let diagonal = array![[inv_sqrt2], [inv_sqrt2]];
        set_global_blocks(&mut model, &[Array2::eye(2), diagonal, x_axis()]);

        let (records, overall_ratio) = model.interior_flow_norms(&array![[1.0, 0.0]]).unwrap();

        // Identity input projection keeps ||x_int|| = 1; the interior
        // projects onto the diagonal, leaving norm 1/sqrt(2).
        assert_eq!(records.len(), 1);
        assert!((records[0].mean_in_norm - 1.0).abs() < EPSILON);
        assert!((records[0].mean_out_norm - inv_sqrt2).abs() < EPSILON);
        assert!((overall_ratio - inv_sqrt2).abs() < EPSILON);
    }

    #[test]
    fn parallel_series_topology_builds_and_reports_two_phase_flow() {
        assert_eq!(
            InteriorTopology::ParallelSeries.to_string(),
            "parallel_series"
        );
        assert_eq!(
            "parallel-series".parse::<InteriorTopology>().unwrap(),
            InteriorTopology::ParallelSeries
        );
        assert_eq!(
            "parallel_series".parse::<InteriorTopology>().unwrap(),
            InteriorTopology::ParallelSeries
        );

        let model = CosmosModel::new(
            CosmosModelConfig::new(2, 2)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(2)
                .with_interior_subspace_dim(1)
                .with_cmr_k(1)
                .with_interior_topology(InteriorTopology::ParallelSeries),
        )
        .unwrap();
        let x = array![[3.0, 4.0], [-1.0, 2.0]];

        let scores = model.forward(&x).unwrap();
        assert_eq!(scores.dim(), (2, 2));

        // Two-phase composition over two projectors: parallel records
        // followed by series records — 2 * L entries.
        let (records, overall_ratio) = model.interior_flow_norms(&x).unwrap();
        assert_eq!(records.len(), 4);
        assert!(overall_ratio.is_finite());
    }

    /// Swapping the interior topology mid-life leaves every learned block
    /// bit-identical and recomposes the interior: the swapped model must
    /// reproduce a model built from scratch under the new topology with the
    /// same state, while differing from its own pre-swap composition.
    #[test]
    fn set_interior_topology_preserves_state_and_recomposes_interior() {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let blocks = [
            Array2::eye(2),
            x_axis(),
            array![[inv_sqrt2], [inv_sqrt2]],
            x_axis(),
            y_axis(),
        ];
        let make_model = |topology: InteriorTopology| {
            let mut model = CosmosModel::new(
                CosmosModelConfig::new(2, 2)
                    .with_input_subspace_dim(2)
                    .with_num_interior_blocks(2)
                    .with_interior_subspace_dim(1)
                    .with_cmr_k(1)
                    .with_interior_topology(topology),
            )
            .unwrap();
            set_global_blocks(&mut model, &blocks);
            model
        };

        let mut model = make_model(InteriorTopology::Series);
        let x = array![[3.0, 4.0], [-1.0, 2.0]];
        let series_embeddings = model.get_embeddings(&x).unwrap();
        let series_scores = model.forward(&x).unwrap();

        let state_before = model.get_state().assemble_global_basis();
        let blocks_before = (0..model.get_state().num_blocks())
            .map(|idx| model.get_state().get_basis(idx).unwrap().clone())
            .collect::<Vec<_>>();

        model
            .set_interior_topology(InteriorTopology::Parallel)
            .unwrap();

        assert_eq!(model.interior_topology(), Some(InteriorTopology::Parallel));
        assert!(model.summary().contains("topology parallel"));

        // State is untouched, bit for bit: bases, interior blocks, anchors.
        assert_eq!(model.get_state().assemble_global_basis(), state_before);
        for (idx, expected) in blocks_before.iter().enumerate() {
            assert_eq!(model.get_state().get_basis(idx).unwrap(), expected);
        }

        let parallel_embeddings = model.get_embeddings(&x).unwrap();
        let parallel_scores = model.forward(&x).unwrap();
        assert_eq!(parallel_embeddings.dim(), series_embeddings.dim());
        assert_eq!(parallel_scores.dim(), series_scores.dim());

        // The composition really changed: series is P_2(P_1(x)), parallel is
        // the mean of the two projections.
        assert!(
            parallel_embeddings
                .iter()
                .zip(series_embeddings.iter())
                .any(|(after, before)| (after - before).abs() > 1e-6),
            "series and parallel compositions must differ for L >= 2"
        );

        // ... and it composes exactly as a from-scratch parallel model does.
        let reference = make_model(InteriorTopology::Parallel);
        assert_array2_close(&parallel_embeddings, &reference.get_embeddings(&x).unwrap());
        assert_array2_close(&parallel_scores, &reference.forward(&x).unwrap());
    }

    /// The topology swap carries the interior runtime settings across:
    /// training flag, epoch/batch counters, and the per-projector subspace
    /// actions.
    #[test]
    fn set_interior_topology_carries_runtime_settings() {
        let mut model = CosmosModel::new(
            CosmosModelConfig::new(4, 2)
                .with_input_subspace_dim(3)
                .with_num_interior_blocks(2)
                .with_interior_subspace_dim(2)
                .with_cmr_k(1),
        )
        .unwrap();

        model.set_training(false).unwrap();
        model.set_epoch_batch(5, 9).unwrap();
        model
            .set_interior_actions(&[SubspaceAction::reflection(), SubspaceAction::projection()])
            .unwrap();
        let actions_before = model.interior_block_actions().unwrap();

        model
            .set_interior_topology(InteriorTopology::Residual)
            .unwrap();

        let interior = model.core.borrow_interior().unwrap();
        assert!(!interior.is_training());
        assert_eq!(interior.epoch(), 5);
        assert_eq!(interior.batch_idx(), 9);
        drop(interior);
        assert_eq!(model.interior_block_actions().unwrap(), actions_before);
    }

    /// Topology presets have no meaning for an explicitly specified interior,
    /// so the swap is rejected instead of silently discarding the spec.
    #[test]
    fn set_interior_topology_rejects_custom_interior_spec() {
        let spec = InteriorBuilder::new()
            .with_normalize_final(true)
            .build_spec();
        let mut model = CosmosModel::new(
            CosmosModelConfig::new(2, 2)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(2)
                .with_interior_subspace_dim(1)
                .with_cmr_k(1)
                .with_interior_spec(spec),
        )
        .unwrap();

        let err = model
            .set_interior_topology(InteriorTopology::Parallel)
            .unwrap_err();

        match err {
            ModelError::InvalidParameter { param, .. } => {
                assert_eq!(param, "interior_topology")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn count_parameters_breaks_down_state_components() {
        let model = CosmosModel::new(
            CosmosModelConfig::new(5, 4)
                .with_input_subspace_dim(2)
                .with_interior_subspace_dim(3)
                .with_num_interior_blocks(2)
                .with_cmr_k(1)
                .with_anchors_per_class(2),
        )
        .unwrap();

        let counts = model.count_parameters();

        assert_eq!(counts["input_block"], 10);
        assert_eq!(counts["interior_blocks"], 30);
        assert_eq!(counts["cmr_blocks"], 40);
        assert_eq!(counts["state_total"], 80);
        assert_eq!(counts["total"], 80);
    }

    #[test]
    fn runtime_helpers_proxy_to_interior() {
        let model = CosmosModel::new(
            CosmosModelConfig::new(2, 1)
                .with_input_subspace_dim(1)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();

        model.set_training(false).unwrap();
        model.set_epoch_batch(3, 7).unwrap();
        model.on_epoch_start(4).unwrap();

        let interior = model.core.borrow_interior().unwrap();
        assert!(!interior.is_training());
        assert_eq!(interior.epoch(), 4);
        assert_eq!(interior.batch_idx(), 0);
    }

    #[test]
    fn summary_mentions_topology_and_block_count() {
        let model = CosmosModel::new(
            CosmosModelConfig::new(8, 3)
                .with_input_subspace_dim(4)
                .with_interior_subspace_dim(2)
                .with_num_interior_blocks(2)
                .with_cmr_k(1)
                .with_interior_topology(InteriorTopology::Parallel),
        )
        .unwrap();

        let summary = model.summary();

        assert!(summary.contains("CosmosModel (Flat Input)"));
        assert!(summary.contains("topology parallel"));
        assert!(summary.contains("Total blocks: 6"));
    }
}
