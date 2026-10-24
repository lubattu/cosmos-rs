//! Building blocks shared by every model variant: interior topology, layout
//! (how interior, CMR and head blocks map onto state indices) and the core
//! state container that owns them.

use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    ops::Range,
    str::FromStr,
};

use ndarray::{Array2, Array4};

use crate::{
    arrangement::{
        activations::{Activation, ActivationPlacement},
        blocks::BlockConfig,
        builder::{BlockBuilder, InteriorBuilder, InteriorSpec},
        compositions::FlowNorm,
        interior::CosmosInterior,
        params::SubspaceAction,
    },
    state::cosmos_state::CosmosState,
};

use super::base::ModelError;

/// High-level interior shortcut used when no explicit [`InteriorSpec`] is provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InteriorTopology {
    /// Compose interior blocks sequentially.
    #[default]
    Series,
    /// Compose interior blocks in parallel and mean-pool the result.
    Parallel,
    /// Compose interior blocks with a residual series pipeline.
    Residual,
    /// Compose interior blocks with a two-phase parallel-then-series pipeline.
    ParallelSeries,
}

impl fmt::Display for InteriorTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Series => write!(f, "series"),
            Self::Parallel => write!(f, "parallel"),
            Self::Residual => write!(f, "residual"),
            Self::ParallelSeries => write!(f, "parallel_series"),
        }
    }
}

impl FromStr for InteriorTopology {
    type Err = ModelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "series" => Ok(Self::Series),
            "parallel" => Ok(Self::Parallel),
            "residual" => Ok(Self::Residual),
            "parallel_series" | "parallel-series" => Ok(Self::ParallelSeries),
            _ => Err(ModelError::InvalidParameter {
                param: "interior_topology",
                reason: format!(
                    "expected one of [series, parallel, residual, parallel_series], got {s}"
                ),
            }),
        }
    }
}

/// Shared interior build configuration for concrete model variants.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InteriorRuntimeConfig {
    pub num_interior_blocks: usize,
    pub interior_topology: InteriorTopology,
    pub interior_spec: Option<InteriorSpec>,
    pub activation: Activation,
    pub activation_placement: ActivationPlacement,
    pub action: SubspaceAction,
}

impl InteriorRuntimeConfig {
    fn build(&self, layout: &ModelLayout) -> Result<CosmosInterior, ModelError> {
        if let Some(spec) = self.interior_spec.as_ref() {
            return Ok(CosmosInterior::from_spec(spec)?);
        }

        let interior_indices = layout.interior_block_range().collect::<Vec<_>>();
        if interior_indices.is_empty() {
            return Ok(CosmosInterior::new(Vec::new()));
        }

        let block_builder = match self.interior_topology {
            InteriorTopology::Series => BlockBuilder::new()
                .with_indices(interior_indices)
                .compose_series(),
            InteriorTopology::Parallel => BlockBuilder::new()
                .with_indices(interior_indices)
                .compose_parallel()
                .aggregate_mean(),
            InteriorTopology::Residual => BlockBuilder::new()
                .with_indices(interior_indices)
                .compose_residual_series(),
            InteriorTopology::ParallelSeries => BlockBuilder::new()
                .with_indices(interior_indices)
                .compose_parallel_series(),
        }
        .activate(self.activation, self.activation_placement)
        .subspace_action(self.action);

        Ok(InteriorBuilder::new()
            .add_block_builder(block_builder)
            .build()?)
    }
}

/// Emission head terminating a classification model.
///
/// The head owns the state blocks that sit after the interior chain: it
/// fixes both their shapes and how the model turns an embedding into a
/// per-class quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelHead {
    /// Geodesic CMR anchors — `num_classes * n_anchors_per_class` blocks of
    /// shape `(d, cmr_k)` in class-major order, scored by the Gaussian-angular
    /// membership `s = exp(-theta^2)`.
    #[default]
    Cmr,
}

impl fmt::Display for ModelHead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cmr => write!(f, "cmr"),
        }
    }
}

/// Shared block layout for model variants that terminate in an emission head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelLayout {
    interior_block_start: usize,
    interior_block_end: usize,
    head_block_start: usize,
    head_block_end: usize,
    /// CMR anchor indices in class-major order.
    cmr_block_indices: Vec<usize>,
}

impl ModelLayout {
    pub fn new(
        prefix_block_count: usize,
        num_interior_blocks: usize,
        num_classes: usize,
        n_anchors_per_class: usize,
    ) -> Result<Self, ModelError> {
        let (interior_block_start, interior_block_end) =
            Self::interior_bounds(prefix_block_count, num_interior_blocks)?;
        let total_cmr_blocks = num_classes.checked_mul(n_anchors_per_class).ok_or(
            ModelError::InvalidCmrConfiguration {
                field: "n_anchors_per_class",
                reason: format!(
                    "num_classes * n_anchors_per_class overflows usize: {num_classes} * {n_anchors_per_class}"
                ),
            },
        )?;
        let head_block_start = interior_block_end;
        let head_block_end = head_block_start.checked_add(total_cmr_blocks).ok_or(
            ModelError::InvalidCmrConfiguration {
                field: "cmr_block_end",
                reason: format!(
                    "cmr_block_start + total_cmr_blocks overflows usize: {head_block_start} + {total_cmr_blocks}"
                ),
            },
        )?;

        Ok(Self {
            interior_block_start,
            interior_block_end,
            head_block_start,
            head_block_end,
            cmr_block_indices: (head_block_start..head_block_end).collect(),
        })
    }

    fn interior_bounds(
        prefix_block_count: usize,
        num_interior_blocks: usize,
    ) -> Result<(usize, usize), ModelError> {
        let interior_block_start = prefix_block_count;
        let interior_block_end = interior_block_start.checked_add(num_interior_blocks).ok_or(
            ModelError::InvalidCmrConfiguration {
                field: "interior_block_end",
                reason: format!(
                    "prefix_block_count + num_interior_blocks overflows usize: {prefix_block_count} + {num_interior_blocks}"
                ),
            },
        )?;
        Ok((interior_block_start, interior_block_end))
    }

    pub fn interior_block_range(&self) -> Range<usize> {
        self.interior_block_start..self.interior_block_end
    }

    /// Every state block owned by the emission head.
    pub fn head_block_range(&self) -> Range<usize> {
        self.head_block_start..self.head_block_end
    }

    pub fn cmr_block_range(&self) -> Range<usize> {
        if self.cmr_block_indices.is_empty() {
            self.head_block_start..self.head_block_start
        } else {
            self.head_block_start..self.head_block_end
        }
    }

    pub fn cmr_block_indices(&self) -> &[usize] {
        &self.cmr_block_indices
    }
}

/// Shared state/interior runtime used by concrete model variants.
#[derive(Debug)]
pub(crate) struct ModelCore {
    state: CosmosState,
    interior: RefCell<CosmosInterior>,
    num_classes: usize,
    cmr_k: usize,
    n_anchors_per_class: usize,
    head: ModelHead,
    interior_config: InteriorRuntimeConfig,
    layout: ModelLayout,
}

impl ModelCore {
    pub fn new(
        state: CosmosState,
        num_classes: usize,
        cmr_k: usize,
        n_anchors_per_class: usize,
        prefix_block_count: usize,
        interior_config: &InteriorRuntimeConfig,
    ) -> Result<Self, ModelError> {
        Self::new_with_head(
            state,
            num_classes,
            cmr_k,
            n_anchors_per_class,
            prefix_block_count,
            interior_config,
            ModelHead::Cmr,
        )
    }

    /// Build a core whose head blocks follow `head`'s layout.
    pub fn new_with_head(
        state: CosmosState,
        num_classes: usize,
        cmr_k: usize,
        n_anchors_per_class: usize,
        prefix_block_count: usize,
        interior_config: &InteriorRuntimeConfig,
        head: ModelHead,
    ) -> Result<Self, ModelError> {
        let layout = match head {
            ModelHead::Cmr => ModelLayout::new(
                prefix_block_count,
                interior_config.num_interior_blocks,
                num_classes,
                n_anchors_per_class,
            )?,
        };
        let interior = interior_config.build(&layout)?;

        Ok(Self {
            state,
            interior: RefCell::new(interior),
            num_classes,
            cmr_k,
            n_anchors_per_class,
            head,
            interior_config: interior_config.clone(),
            layout,
        })
    }

    pub fn state(&self) -> &CosmosState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut CosmosState {
        &mut self.state
    }

    pub fn cmr_block_indices(&self) -> &[usize] {
        self.layout.cmr_block_indices()
    }

    /// Emission head terminating the model.
    pub fn head(&self) -> ModelHead {
        self.head
    }

    /// Every state block owned by the emission head.
    pub fn head_block_range(&self) -> Range<usize> {
        self.layout.head_block_range()
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn cmr_k(&self) -> usize {
        self.cmr_k
    }

    pub fn anchors_per_class(&self) -> usize {
        self.n_anchors_per_class
    }

    pub fn num_interior_blocks(&self) -> usize {
        self.layout.interior_block_range().len()
    }

    pub fn interior_block_range(&self) -> Range<usize> {
        self.layout.interior_block_range()
    }

    pub fn cmr_block_range(&self) -> Range<usize> {
        self.layout.cmr_block_range()
    }

    pub fn interior_topology(&self) -> Option<InteriorTopology> {
        if self.uses_custom_interior_spec() {
            None
        } else {
            Some(self.interior_config.interior_topology)
        }
    }

    pub fn uses_custom_interior_spec(&self) -> bool {
        self.interior_config.interior_spec.is_some()
    }

    pub fn interior_activation(&self) -> Activation {
        self.interior_config.activation
    }

    pub fn interior_activation_placement(&self) -> ActivationPlacement {
        self.interior_config.activation_placement
    }

    /// Rebuild the interior composition under a different topology preset.
    ///
    /// The topology is a gradient-free arrangement over topology-independent
    /// learned state: every preset composes the same interior projector
    /// indices with the same block shapes, so only the composition runtime is
    /// rebuilt. [`CosmosState`] — bases, anchors, and everything the evolution
    /// law owns — is untouched. Runtime interior settings carry over: the
    /// training flag, the epoch/batch counters, and the per-projector subspace
    /// actions.
    ///
    /// Returns an error when the interior was built from an explicit
    /// [`InteriorSpec`], where topology presets have no meaning.
    pub fn set_interior_topology(&mut self, topology: InteriorTopology) -> Result<(), ModelError> {
        if self.uses_custom_interior_spec() {
            return Err(ModelError::InvalidParameter {
                param: "interior_topology",
                reason: "interior was built from an explicit interior spec; topology presets do not apply"
                    .to_string(),
            });
        }
        if self.interior_config.interior_topology == topology {
            return Ok(());
        }

        let (training, epoch, batch_idx, actions) = {
            let interior = self.borrow_interior()?;
            let actions = interior
                .blocks()
                .iter()
                .flat_map(|block| block.actions().to_vec())
                .collect::<Vec<_>>();
            (
                interior.is_training(),
                interior.epoch(),
                interior.batch_idx(),
                actions,
            )
        };

        self.interior_config.interior_topology = topology;
        let rebuilt = self.interior_config.build(&self.layout)?;

        let interior = self.interior.get_mut();
        *interior = rebuilt;
        interior.set_training(training);
        interior.set_epoch_batch(epoch, batch_idx);

        let mut action_offset = 0;
        for block in interior.blocks_mut() {
            let n = block.config().block_indices.len();
            if action_offset + n > actions.len() {
                break;
            }
            block
                .set_all_actions(actions[action_offset..action_offset + n].to_vec())
                .map_err(|_e| ModelError::RuntimeStateConflict {
                    component: "interior",
                    reason: "failed to carry subspace actions across a topology swap",
                })?;
            action_offset += n;
        }

        Ok(())
    }

    pub fn interior_block_configs(&self) -> Result<Vec<BlockConfig>, ModelError> {
        Ok(self
            .borrow_interior()?
            .blocks()
            .iter()
            .map(|block| block.config().clone())
            .collect())
    }

    pub fn interior_block_actions(&self) -> Result<Vec<Vec<SubspaceAction>>, ModelError> {
        Ok(self
            .borrow_interior()?
            .blocks()
            .iter()
            .map(|block| block.actions().to_vec())
            .collect())
    }

    pub fn interior_global_activation(&self) -> Result<Activation, ModelError> {
        Ok(self.borrow_interior()?.global_activation())
    }

    pub fn interior_normalize_final(&self) -> Result<bool, ModelError> {
        Ok(self.borrow_interior()?.normalize_final())
    }

    pub fn total_blocks(&self) -> usize {
        self.state.num_blocks()
    }

    pub fn is_training(&self) -> Result<bool, ModelError> {
        Ok(self.borrow_interior()?.is_training())
    }

    pub fn set_training(&self, training: bool) -> Result<(), ModelError> {
        self.borrow_interior_mut()?.set_training(training);
        Ok(())
    }

    pub fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError> {
        self.borrow_interior_mut()?
            .set_epoch_batch(epoch, batch_idx);
        Ok(())
    }

    pub fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError> {
        self.borrow_interior_mut()?.on_epoch_start(epoch);
        Ok(())
    }

    pub fn forward_interior(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        let mut interior = self.borrow_interior_mut()?;
        Ok(interior.forward(x, &self.state)?)
    }

    /// Flow-instrumented interior forward.
    ///
    /// Returns the interior output, the per-projector [`FlowNorm`] records,
    /// and the batch-mean row norm of the output measured after the global
    /// activation but before the optional final row normalization (row
    /// normalization would erase the norm-dilution signal).
    pub(crate) fn forward_interior_with_flow(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<FlowNorm>, f64), ModelError> {
        let mut interior = self.borrow_interior_mut()?;
        Ok(interior.forward_with_flow_detailed(x, &self.state)?)
    }

    /// Number of interior state blocks (projector indices inside the interior).
    pub fn num_interior_state_blocks(&self) -> usize {
        self.layout.interior_block_range().len()
    }

    /// Global indices of interior state blocks.
    pub fn interior_block_indices(&self) -> Vec<usize> {
        self.layout.interior_block_range().collect()
    }

    /// Every interior subspace action, concatenated across the runtime's
    /// modular blocks in block order.
    ///
    /// The actions are the `(alpha, beta)` pair each projector index is
    /// applied through, so they are part of what a forward pass reads and
    /// what a checkpoint must therefore carry.
    pub fn interior_actions(&self) -> Result<Vec<SubspaceAction>, ModelError> {
        let interior = self.borrow_interior()?;
        Ok(interior
            .blocks()
            .iter()
            .flat_map(|block| block.actions().iter().copied())
            .collect())
    }

    /// Replace every interior subspace action, in the order
    /// [`ModelCore::interior_actions`] reports them.
    ///
    /// The count must match exactly; a differing count is a shape error
    /// rather than a partial assignment.
    pub fn set_interior_actions(&self, actions: &[SubspaceAction]) -> Result<(), ModelError> {
        let mut interior = self.borrow_interior_mut()?;
        let expected: usize = interior
            .blocks()
            .iter()
            .map(|block| block.actions().len())
            .sum();
        if actions.len() != expected {
            return Err(ModelError::ShapeMismatch {
                expected: format!("{expected} interior subspace actions"),
                got: format!("{}", actions.len()),
            });
        }

        let mut offset = 0;
        for block in interior.blocks_mut() {
            let width = block.actions().len();
            let slice = actions[offset..offset + width].to_vec();
            block
                .set_all_actions(slice)
                .map_err(|_e| ModelError::RuntimeStateConflict {
                    component: "interior",
                    reason: "failed to set subspace actions on modular block",
                })?;
            offset += width;
        }

        Ok(())
    }

    pub fn borrow_interior(&self) -> Result<Ref<'_, CosmosInterior>, ModelError> {
        self.interior
            .try_borrow()
            .map_err(|_| ModelError::RuntimeStateConflict {
                component: "interior",
                reason: "already mutably borrowed",
            })
    }

    pub fn borrow_interior_mut(&self) -> Result<RefMut<'_, CosmosInterior>, ModelError> {
        self.interior
            .try_borrow_mut()
            .map_err(|_| ModelError::RuntimeStateConflict {
                component: "interior",
                reason: "already borrowed",
            })
    }
}

/// Derived geometry for image-like inputs used by patch-based models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PatchGeometry {
    image_shape: (usize, usize, usize),
    patch_size: usize,
    num_patches: usize,
    patch_dim: usize,
}

impl PatchGeometry {
    pub fn new(image_shape: (usize, usize, usize), patch_size: usize) -> Result<Self, ModelError> {
        let (channels, height, width) = image_shape;
        if channels == 0 || height == 0 || width == 0 {
            return Err(ModelError::InvalidParameter {
                param: "image_shape",
                reason: "channels, height, and width must be greater than zero".to_string(),
            });
        }

        validate_positive("patch_size", patch_size)?;

        if height % patch_size != 0 || width % patch_size != 0 {
            return Err(ModelError::InvalidParameter {
                param: "patch_size",
                reason: format!(
                    "image spatial dimensions ({height}, {width}) must be divisible by patch_size {patch_size}"
                ),
            });
        }

        let patch_area =
            patch_size
                .checked_mul(patch_size)
                .ok_or(ModelError::InvalidParameter {
                    param: "patch_size",
                    reason: format!("patch_size * patch_size overflows usize: {patch_size}^2"),
                })?;
        let patch_rows = height / patch_size;
        let patch_cols = width / patch_size;
        let num_patches =
            patch_rows
                .checked_mul(patch_cols)
                .ok_or(ModelError::InvalidParameter {
                    param: "image_shape",
                    reason: format!(
                        "patch_rows * patch_cols overflows usize: {patch_rows} * {patch_cols}"
                    ),
                })?;
        let patch_dim = channels
            .checked_mul(patch_area)
            .ok_or(ModelError::InvalidParameter {
                param: "image_shape",
                reason: format!("channels * patch_area overflows usize: {channels} * {patch_area}"),
            })?;

        Ok(Self {
            image_shape,
            patch_size,
            num_patches,
            patch_dim,
        })
    }

    pub fn image_shape(&self) -> (usize, usize, usize) {
        self.image_shape
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    pub fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    pub fn validate_input(&self, x: &Array4<f64>) -> Result<(), ModelError> {
        let (batch, channels, height, width) = x.dim();
        let expected = self.image_shape;

        if (channels, height, width) != expected {
            return Err(ModelError::ShapeMismatch {
                expected: format!(
                    "image batch with shape (batch, {}, {}, {})",
                    expected.0, expected.1, expected.2
                ),
                got: format!("({}, {}, {}, {})", batch, channels, height, width),
            });
        }

        Ok(())
    }
}

pub(crate) fn validate_positive(param: &'static str, value: usize) -> Result<(), ModelError> {
    if value == 0 {
        return Err(ModelError::InvalidParameter {
            param,
            reason: "must be greater than zero".to_string(),
        });
    }

    Ok(())
}

pub(crate) fn block_parameter_count(shape: (usize, usize)) -> usize {
    shape.0 * shape.1
}
