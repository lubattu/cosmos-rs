//! Patchwise COSMOS classification model for image-like inputs.
//!
//! [`CosmosModelPatchwise`] composes:
//!
//! `image -> Patchify -> PatchProjector -> PatchAggregator -> CosmosInterior -> CMR`

use std::{collections::HashMap, ops::Range};

use ndarray::{Array2, Array4};

use crate::{
    arrangement::{
        activations::{Activation, ActivationPlacement},
        builder::InteriorSpec,
        input::{PatchAggregationMode, PatchAggregator, PatchProjector, Patchify},
        params::SubspaceAction,
    },
    objectives::cmr_losses::RegionMetrics,
    state::cosmos_state::CosmosState,
};

use super::{
    base::{Model, ModelError},
    common::{
        InteriorRuntimeConfig, InteriorTopology, ModelCore, PatchGeometry, block_parameter_count,
        validate_positive,
    },
};

/// Immutable constructor configuration for [`CosmosModelPatchwise`].
#[derive(Debug, Clone, PartialEq)]
pub struct PatchwiseModelConfig {
    /// Input image shape `(channels, height, width)`.
    pub image_shape: (usize, usize, usize),
    /// Size of each square patch.
    pub patch_size: usize,
    /// Shared subspace dimension used for each patch projection.
    pub patch_subspace_dim: usize,
    /// Output ambient dimension after patch aggregation.
    pub agg_subspace_dim: usize,
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
    /// Aggregation mode before the final patchwise projection.
    pub agg_mode: PatchAggregationMode,
    /// Interior shortcut used when `interior_spec` is absent.
    pub interior_topology: InteriorTopology,
    /// Optional explicit interior specification overriding `interior_topology`.
    pub interior_spec: Option<InteriorSpec>,
    /// Activation used inside interior composition presets.
    pub activation: Activation,
    /// Placement used for the interior activation.
    pub activation_placement: ActivationPlacement,
}

impl PatchwiseModelConfig {
    /// Create a config with project-wide defaults for a patchwise classifier.
    pub fn new(image_shape: (usize, usize, usize), num_classes: usize) -> Self {
        Self {
            image_shape,
            patch_size: 4,
            patch_subspace_dim: 16,
            agg_subspace_dim: 256,
            interior_subspace_dim: 128,
            num_interior_blocks: 3,
            num_classes,
            cmr_k: 32,
            n_anchors_per_class: 1,
            agg_mode: PatchAggregationMode::Concat,
            interior_topology: InteriorTopology::Series,
            interior_spec: None,
            activation: Activation::Identity,
            activation_placement: ActivationPlacement::None,
        }
    }

    /// Override the patch size.
    pub fn with_patch_size(mut self, patch_size: usize) -> Self {
        self.patch_size = patch_size;
        self
    }

    /// Override the patch projector subspace dimension.
    pub fn with_patch_subspace_dim(mut self, patch_subspace_dim: usize) -> Self {
        self.patch_subspace_dim = patch_subspace_dim;
        self
    }

    /// Override the post-aggregation ambient dimension.
    pub fn with_agg_subspace_dim(mut self, agg_subspace_dim: usize) -> Self {
        self.agg_subspace_dim = agg_subspace_dim;
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

    /// Override the patch aggregation mode.
    pub fn with_agg_mode(mut self, agg_mode: PatchAggregationMode) -> Self {
        self.agg_mode = agg_mode;
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

    fn validate(&self, geometry: &PatchGeometry) -> Result<(), ModelError> {
        let _ = geometry;
        validate_positive("patch_subspace_dim", self.patch_subspace_dim)?;
        validate_positive("agg_subspace_dim", self.agg_subspace_dim)?;
        validate_positive("num_classes", self.num_classes)?;
        validate_positive("cmr_k", self.cmr_k)?;
        validate_positive("n_anchors_per_class", self.n_anchors_per_class)?;

        if self.num_interior_blocks > 0 {
            validate_positive("interior_subspace_dim", self.interior_subspace_dim)?;
        }

        Ok(())
    }
}

/// Patchwise COSMOS classification model for image-like inputs.
#[derive(Debug)]
pub struct CosmosModelPatchwise {
    core: ModelCore,
    patchify: Patchify,
    patch_projector: PatchProjector,
    aggregator: PatchAggregator,
    geometry: PatchGeometry,
    patch_subspace_dim: usize,
    agg_input_dim: usize,
    agg_subspace_dim: usize,
    interior_subspace_dim: usize,
    patch_block_idx: usize,
    agg_block_idx: usize,
}

impl CosmosModelPatchwise {
    /// Build a patchwise COSMOS model from a typed config.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when a configured dimension is zero,
    /// when the image shape is not divisible by the patch size, or when a subspace
    /// dimension exceeds its ambient dimension, and [`ModelError::State`] when the
    /// interior state fails to build from the resulting block layout.
    pub fn new(config: PatchwiseModelConfig) -> Result<Self, ModelError> {
        let geometry = PatchGeometry::new(config.image_shape, config.patch_size)?;
        config.validate(&geometry)?;

        let agg_input_dim = match config.agg_mode {
            PatchAggregationMode::Concat => geometry
                .num_patches()
                .checked_mul(config.patch_subspace_dim)
                .ok_or(ModelError::InvalidParameter {
                    param: "agg_input_dim",
                    reason: format!(
                        "num_patches * patch_subspace_dim overflows usize: {} * {}",
                        geometry.num_patches(),
                        config.patch_subspace_dim
                    ),
                })?,
            PatchAggregationMode::Mean | PatchAggregationMode::Sum => config.patch_subspace_dim,
        };
        let patch_block_idx = 0;
        let agg_block_idx = 1;
        let total_cmr_blocks = config
            .num_classes
            .checked_mul(config.n_anchors_per_class)
            .ok_or(ModelError::InvalidCmrConfiguration {
                field: "n_anchors_per_class",
                reason: format!(
                    "num_classes * n_anchors_per_class overflows usize: {} * {}",
                    config.num_classes, config.n_anchors_per_class
                ),
            })?;

        let mut projector_shapes =
            Vec::with_capacity(2 + config.num_interior_blocks + total_cmr_blocks);
        projector_shapes.push((geometry.patch_dim(), config.patch_subspace_dim));
        projector_shapes.push((agg_input_dim, config.agg_subspace_dim));
        for _ in 0..config.num_interior_blocks {
            projector_shapes.push((config.agg_subspace_dim, config.interior_subspace_dim));
        }
        for _ in 0..total_cmr_blocks {
            projector_shapes.push((config.agg_subspace_dim, config.cmr_k));
        }

        let state = CosmosState::new(&projector_shapes)?;
        let interior_config = InteriorRuntimeConfig {
            num_interior_blocks: config.num_interior_blocks,
            interior_topology: config.interior_topology,
            interior_spec: config.interior_spec.clone(),
            activation: config.activation,
            activation_placement: config.activation_placement,
            action: SubspaceAction::default(),
        };
        let core = ModelCore::new(
            state,
            config.num_classes,
            config.cmr_k,
            config.n_anchors_per_class,
            2,
            &interior_config,
        )?;

        Ok(Self {
            core,
            patchify: Patchify::new(config.patch_size, true)?,
            patch_projector: PatchProjector::new(patch_block_idx),
            aggregator: PatchAggregator::new(agg_block_idx, config.agg_mode),
            geometry,
            patch_subspace_dim: config.patch_subspace_dim,
            agg_input_dim,
            agg_subspace_dim: config.agg_subspace_dim,
            interior_subspace_dim: config.interior_subspace_dim,
            patch_block_idx,
            agg_block_idx,
        })
    }

    /// Convenience wrapper over the trait-based batch forward path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, when the interior runtime is already borrowed, or when the
    /// anchor layout is inconsistent.
    pub fn forward(&self, x: &Array4<f64>) -> Result<Array2<f64>, ModelError> {
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
        x: &Array4<f64>,
    ) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
        <Self as Model>::forward_with_metrics(self, x)
    }

    /// Convenience wrapper over the trait-based embedding extraction path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, or when the interior runtime is already borrowed.
    pub fn get_embeddings(&self, x: &Array4<f64>) -> Result<Array2<f64>, ModelError> {
        <Self as Model>::get_embeddings(self, x)
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

    /// Input image shape `(channels, height, width)`.
    pub fn image_shape(&self) -> (usize, usize, usize) {
        self.geometry.image_shape()
    }

    /// Configured patch size.
    pub fn patch_size(&self) -> usize {
        self.geometry.patch_size()
    }

    /// Number of extracted patches per image.
    pub fn num_patches(&self) -> usize {
        self.geometry.num_patches()
    }

    /// Flattened dimension of each patch before projection.
    pub fn patch_dim(&self) -> usize {
        self.geometry.patch_dim()
    }

    /// Input dimension of the aggregation projector.
    pub fn agg_input_dim(&self) -> usize {
        self.agg_input_dim
    }

    /// Index of the patch projector block.
    pub fn patch_block_idx(&self) -> usize {
        self.patch_block_idx
    }

    /// Index of the aggregation projector block.
    pub fn agg_block_idx(&self) -> usize {
        self.agg_block_idx
    }

    /// Patch aggregation mode.
    pub fn agg_mode(&self) -> PatchAggregationMode {
        self.aggregator.mode()
    }

    /// Interior block index range.
    pub fn interior_block_range(&self) -> Range<usize> {
        self.core.interior_block_range()
    }

    /// CMR anchor block index range.
    pub fn cmr_block_range(&self) -> Range<usize> {
        self.core.cmr_block_range()
    }

    /// Configured interior shortcut, if the runtime was not built from an explicit spec.
    pub fn interior_topology(&self) -> Option<InteriorTopology> {
        self.core.interior_topology()
    }

    /// Whether the model uses a custom interior specification.
    pub fn uses_custom_interior_spec(&self) -> bool {
        self.core.uses_custom_interior_spec()
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
    pub(crate) fn interior_block_configs(
        &self,
    ) -> Result<Vec<crate::arrangement::blocks::BlockConfig>, ModelError> {
        self.core.interior_block_configs()
    }

    /// Runtime per-projector subspace actions for each interior block.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::RuntimeStateConflict`] when the interior runtime is
    /// already borrowed.
    pub(crate) fn interior_block_actions(&self) -> Result<Vec<Vec<SubspaceAction>>, ModelError> {
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

    fn render_summary(&self) -> String {
        let counts = <Self as Model>::count_parameters(self);
        let topology_label = if self.uses_custom_interior_spec() {
            "custom_spec".to_string()
        } else {
            self.interior_topology()
                .unwrap_or(InteriorTopology::Series)
                .to_string()
        };
        let cmr_desc = if self.core.anchors_per_class() == 1 {
            format!(
                "{} x Gr({}, {})",
                self.core.num_classes(),
                self.core.cmr_k(),
                self.agg_subspace_dim
            )
        } else {
            format!(
                "{} x {} anchors x Gr({}, {})",
                self.core.num_classes(),
                self.core.anchors_per_class(),
                self.core.cmr_k(),
                self.agg_subspace_dim
            )
        };

        [
            "CosmosModelPatchwise".to_string(),
            format!(
                "  Input: {:?} -> {} patches of {} dims",
                self.image_shape(),
                self.num_patches(),
                self.patch_dim()
            ),
            format!(
                "  Patch projector: Gr({}, {}) [shared]",
                self.patch_subspace_dim,
                self.patch_dim()
            ),
            format!(
                "  Aggregation: {} ({}) -> Gr({}, {})",
                self.agg_mode(),
                self.agg_input_dim,
                self.agg_subspace_dim,
                self.agg_input_dim
            ),
            format!(
                "  Interior: {} blocks, topology {}, subspace dim {}",
                self.core.num_interior_blocks(),
                topology_label,
                self.interior_subspace_dim
            ),
            format!("  CMR: {cmr_desc}"),
            format!("  Total blocks: {}", self.core.total_blocks()),
            format!("  Parameters: {} (all in CosmosState)", counts["total"]),
        ]
        .join("\n")
    }
}

impl Model for CosmosModelPatchwise {
    type Input = Array4<f64>;

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

    fn get_embeddings(&self, x: &Array4<f64>) -> Result<Array2<f64>, ModelError> {
        self.geometry.validate_input(x)?;

        let patches = self.patchify.forward_flat(x)?;
        let z_patches = self.patch_projector.forward(&patches, self.core.state())?;
        let z_agg = self.aggregator.forward(&z_patches, self.core.state())?;
        self.core.forward_interior(&z_agg)
    }

    fn summary(&self) -> String {
        self.render_summary()
    }

    fn count_parameters(&self) -> HashMap<String, usize> {
        let block_shapes = self.core.state().block_shapes();
        let patch_block = block_parameter_count(block_shapes[self.patch_block_idx]);
        let aggregation_block = block_parameter_count(block_shapes[self.agg_block_idx]);
        let interior_blocks = self
            .interior_block_range()
            .map(|idx| block_parameter_count(block_shapes[idx]))
            .sum();
        let cmr_blocks = self
            .cmr_block_range()
            .map(|idx| block_parameter_count(block_shapes[idx]))
            .sum();
        let state_total = patch_block + aggregation_block + interior_blocks + cmr_blocks;

        HashMap::from([
            ("patch_block".to_string(), patch_block),
            ("aggregation_block".to_string(), aggregation_block),
            ("interior_blocks".to_string(), interior_blocks),
            ("cmr_blocks".to_string(), cmr_blocks),
            ("state_total".to_string(), state_total),
            ("total".to_string(), state_total),
        ])
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array4, array, s};

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

    fn set_global_blocks(model: &mut CosmosModelPatchwise, blocks: &[Array2<f64>]) {
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

    fn two_sample_images() -> Array4<f64> {
        array![[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]],]
    }

    #[test]
    fn config_new_sets_expected_defaults() {
        let config = PatchwiseModelConfig::new((3, 32, 32), 10);

        assert_eq!(config.image_shape, (3, 32, 32));
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.patch_subspace_dim, 16);
        assert_eq!(config.agg_subspace_dim, 256);
        assert_eq!(config.interior_subspace_dim, 128);
        assert_eq!(config.num_interior_blocks, 3);
        assert_eq!(config.num_classes, 10);
        assert_eq!(config.cmr_k, 32);
        assert_eq!(config.n_anchors_per_class, 1);
        assert_eq!(config.agg_mode, PatchAggregationMode::Concat);
        assert_eq!(config.interior_topology, InteriorTopology::Series);
        assert!(config.interior_spec.is_none());
    }

    #[test]
    fn forward_matches_known_two_class_axis_scores() {
        let mut model = CosmosModelPatchwise::new(
            PatchwiseModelConfig::new((1, 2, 2), 2)
                .with_patch_size(1)
                .with_patch_subspace_dim(1)
                .with_agg_subspace_dim(2)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();
        let off_axis = (-(std::f64::consts::FRAC_PI_2.powi(2))).exp();

        set_global_blocks(
            &mut model,
            &[
                array![[1.0]],
                array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                array![[1.0], [0.0]],
                array![[0.0], [1.0]],
            ],
        );

        let (scores, metrics) = model.forward_with_metrics(&two_sample_images()).unwrap();

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
    fn input_shape_mismatch_is_rejected_before_patch_pipeline() {
        let model = CosmosModelPatchwise::new(
            PatchwiseModelConfig::new((1, 2, 2), 2)
                .with_patch_size(1)
                .with_patch_subspace_dim(1)
                .with_agg_subspace_dim(2)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();

        let err = model
            .forward(&array![[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
            .unwrap_err();

        match err {
            ModelError::ShapeMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn count_parameters_breaks_down_patchwise_components() {
        let model = CosmosModelPatchwise::new(
            PatchwiseModelConfig::new((3, 8, 8), 4)
                .with_patch_size(4)
                .with_patch_subspace_dim(2)
                .with_agg_subspace_dim(5)
                .with_interior_subspace_dim(3)
                .with_num_interior_blocks(2)
                .with_cmr_k(1)
                .with_anchors_per_class(2),
        )
        .unwrap();

        let counts = model.count_parameters();

        assert_eq!(counts["patch_block"], 96);
        assert_eq!(counts["aggregation_block"], 40);
        assert_eq!(counts["interior_blocks"], 30);
        assert_eq!(counts["cmr_blocks"], 40);
        assert_eq!(counts["state_total"], 206);
        assert_eq!(counts["total"], 206);
    }

    #[test]
    fn runtime_helpers_proxy_to_interior() {
        let model = CosmosModelPatchwise::new(
            PatchwiseModelConfig::new((1, 2, 2), 1)
                .with_patch_size(1)
                .with_patch_subspace_dim(1)
                .with_agg_subspace_dim(2)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();

        model.set_training(false).unwrap();
        model.set_epoch_batch(5, 9).unwrap();
        model.on_epoch_start(6).unwrap();

        let interior = model.core.borrow_interior().unwrap();
        assert!(!interior.is_training());
        assert_eq!(interior.epoch(), 6);
        assert_eq!(interior.batch_idx(), 0);
    }

    #[test]
    fn summary_mentions_aggregation_mode_and_anchor_count() {
        let model = CosmosModelPatchwise::new(
            PatchwiseModelConfig::new((3, 32, 32), 10)
                .with_anchors_per_class(3)
                .with_agg_mode(PatchAggregationMode::Mean)
                .with_num_interior_blocks(1)
                .with_interior_subspace_dim(16)
                .with_agg_subspace_dim(16)
                .with_cmr_k(8),
        )
        .unwrap();

        let summary = model.summary();

        assert!(summary.contains("CosmosModelPatchwise"));
        assert!(summary.contains("Aggregation: mean"));
        assert!(summary.contains("10 x 3 anchors x Gr(8, 16)"));
        assert_eq!(model.cmr_block_range().len(), 30);
    }
}
