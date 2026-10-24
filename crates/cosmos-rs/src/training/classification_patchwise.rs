//! Patch-input classification adapter.
//!
//! Images arrive as `(batch, channels, height, width)`. The adapter patchifies
//! them, projects every patch through the shared patch block, pools spatially,
//! aggregates into one embedding per image and hands that to the same interior
//! and CMR head the flat adapter uses.

use std::time::Instant;

use ndarray::{Array2, Array3, Array4, Axis};

use crate::{
    arrangement::input::PatchAggregationMode,
    compute::{
        cmr_multi_anchor_backward_device,
        device::{Buffer2, ComputeDevice, Device},
    },
    models::{CosmosModelPatchwise, Model},
    objectives::{
        Reduction,
        cmr_losses::{direct_cmr_loss, geodesic_cmr_hinge_loss, geodesic_cmr_loss_grad},
        utils::Targets,
    },
    training::classification::{
        ClassificationLoss, InteriorForwardDevice, angle_grads_to_score_grads,
        backward_supported_interior, direct_membership_score_grad,
        forward_supported_interior_device, single_label_accuracy, upload_basis,
    },
};

use super::{
    BlockGradients, TrainError,
    backend::TrainingBackend,
    task_adapter::{
        BatchMetrics, BatchTimingProfile, EvalBatchOutput, TaskAdapter, TrainBatchOutput,
    },
};

/// Batch payload for patchwise supervised classification.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchwiseClassificationBatch {
    /// Image batch in shape `(batch, channels, height, width)`.
    pub inputs: Array4<f64>,
    /// Single-label or multi-label supervision targets.
    pub targets: Targets,
}

impl PatchwiseClassificationBatch {
    /// Construct a new patchwise batch.
    pub fn new(inputs: Array4<f64>, targets: Targets) -> Self {
        Self { inputs, targets }
    }

    /// Convenience constructor for single-label classification.
    pub fn single_label(inputs: Array4<f64>, labels: Vec<usize>) -> Self {
        Self {
            inputs,
            targets: Targets::SingleLabel(labels),
        }
    }

    /// Number of samples in this batch.
    pub fn samples(&self) -> usize {
        self.inputs.shape()[0]
    }
}

/// Configuration for the patchwise classification adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchwiseClassificationAdapterConfig {
    /// Numerical epsilon for stabilized norm-based derivatives.
    pub eps: f64,
    /// Supervised loss. Default: [`ClassificationLoss::DirectMembership`].
    pub loss: ClassificationLoss,
}

impl Default for PatchwiseClassificationAdapterConfig {
    fn default() -> Self {
        Self {
            eps: 1e-8,
            loss: ClassificationLoss::default(),
        }
    }
}

impl PatchwiseClassificationAdapterConfig {
    fn validate(&self) -> Result<(), TrainError> {
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(TrainError::InvalidConfig {
                field: "eps",
                reason: format!("must be finite and > 0, got {}", self.eps),
            });
        }
        if let ClassificationLoss::GeodesicHinge { margin } = self.loss {
            if !margin.is_finite() || margin < 0.0 {
                return Err(TrainError::InvalidConfig {
                    field: "loss.margin",
                    reason: format!("must be finite and >= 0, got {margin}"),
                });
            }
        }
        Ok(())
    }
}

/// Manual adapter for patch-input classification.
///
/// The patch projector and the aggregator apply the coordinate map `y = x W`,
/// not the full projector `y = x W Wᵀ` the flat adapter's input head uses, so
/// their basis gradient is the plain matmul `dL/dW = xᵀ (dL/dy)`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PatchwiseClassificationAdapter {
    config: PatchwiseClassificationAdapterConfig,
}

impl PatchwiseClassificationAdapter {
    /// Create a new adapter.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidConfig`] when `eps` or the loss margin is
    /// out of range.
    pub fn new(config: PatchwiseClassificationAdapterConfig) -> Result<Self, TrainError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Borrow the adapter configuration.
    pub fn config(&self) -> &PatchwiseClassificationAdapterConfig {
        &self.config
    }

    /// Forward pass with every intermediate the backward needs retained.
    ///
    /// Mirrors `CosmosModelPatchwise::get_embeddings` but keeps the flattened
    /// patches, the aggregated tensor and the pre-anchor embeddings resident on
    /// the device so the backward computes per-block gradients without
    /// recomputing or re-uploading.
    fn forward_with_cache<B: TrainingBackend>(
        &self,
        model: &CosmosModelPatchwise,
        batch: &PatchwiseClassificationBatch,
        backend: &mut B,
    ) -> Result<PatchwiseForwardCache, TrainError> {
        use crate::arrangement::input::{PatchAggregator, Patchify};

        let state = Model::state(model);
        let patchify = Patchify::new(model.patch_size(), true)
            .map_err(|e| TrainError::task_adapter(format!("patchify init failed: {e:?}")))?;
        let patches = patchify
            .forward_flat(&batch.inputs)
            .map_err(|e| TrainError::task_adapter(format!("patchify forward failed: {e:?}")))?;

        // Patch projector: `z_patches = patches @ W_patch`. Pooling every
        // (sample, patch) pair into one `(B·NP, patch_dim)` matrix lets a single
        // matmul cover the whole batch — the same pooling the backward uses for
        // the shared-projector gradient.
        let w_patch = state
            .get_basis(model.patch_block_idx())
            .map_err(|e| TrainError::task_adapter(format!("read W_patch failed: {e:?}")))?;
        let (b_size, num_patches, _patch_dim) = patches.dim();
        let k_patch = w_patch.ncols();
        let patches_flat = flatten_batched_patches(&patches);
        let device = backend.device();
        let patches_flat_buf = upload_basis(device, &patches_flat);
        let w_patch_buf = upload_basis(device, w_patch);
        let z_patches_flat_buf = device.matmul(&patches_flat_buf, &w_patch_buf);
        let z_patches_flat = device.download_2d(&z_patches_flat_buf);
        let z_patches = unflatten_batched_patches(&z_patches_flat, b_size, num_patches, k_patch);

        // Aggregator: `agg_flat = aggregate(z_patches, mode)` is host-side data
        // movement; `z_agg = agg_flat @ W_agg` runs on the device.
        let aggregator = PatchAggregator::new(model.agg_block_idx(), model.agg_mode());
        let agg_flat = aggregator.aggregate(&z_patches);
        let w_agg = state
            .get_basis(model.agg_block_idx())
            .map_err(|e| TrainError::task_adapter(format!("read W_agg failed: {e:?}")))?;
        if agg_flat.ncols() != w_agg.nrows() {
            return Err(TrainError::task_adapter(format!(
                "agg_flat columns ({}) do not match W_agg rows ({})",
                agg_flat.ncols(),
                w_agg.nrows()
            )));
        }
        let agg_flat_buf = upload_basis(device, &agg_flat);
        let w_agg_buf = upload_basis(device, w_agg);
        let z_agg_buf = device.matmul(&agg_flat_buf, &w_agg_buf);

        // Interior chain — invoked only when the model has interior blocks, a
        // non-trivial global activation or a final normalization. Otherwise the
        // embeddings equal `z_agg`.
        let interior_block_configs = model.interior_block_configs().map_err(|e| {
            TrainError::task_adapter(format!("read interior_block_configs failed: {e:?}"))
        })?;
        let interior_block_actions = model.interior_block_actions().map_err(|e| {
            TrainError::task_adapter(format!("read interior_block_actions failed: {e:?}"))
        })?;
        let interior_global_activation = model.interior_global_activation().map_err(|e| {
            TrainError::task_adapter(format!("read interior_global_activation failed: {e:?}"))
        })?;
        let interior_normalize_final = model.interior_normalize_final().map_err(|e| {
            TrainError::task_adapter(format!("read interior_normalize_final failed: {e:?}"))
        })?;

        let interior = forward_supported_interior_device(
            &interior_block_configs,
            &interior_block_actions,
            interior_global_activation,
            interior_normalize_final,
            state,
            &z_agg_buf,
            backend,
        )?;
        let device = backend.device();
        let embeddings_buf: Buffer2<f64> = match interior.as_ref() {
            Some(int) => device.clone_buffer(&int.output),
            None => device.clone_buffer(&z_agg_buf),
        };

        // Per-class Gaussian-in-geodesic-angle scores, max-pooled over the
        // anchors of each class exactly as the flat adapter does.
        let num_classes = Model::num_classes(model);
        let anchors_per_class = Model::anchors_per_class(model);
        let cmr_indices = Model::cmr_block_indices(model);
        let mut anchor_bufs: Vec<Buffer2<f64>> = Vec::with_capacity(cmr_indices.len());
        for &idx in cmr_indices {
            let anchor = state.get_basis(idx).map_err(|e| {
                TrainError::task_adapter(format!("read anchor {idx} failed: {e:?}"))
            })?;
            anchor_bufs.push(upload_basis(device, anchor));
        }
        let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();
        let (scores_buf, angles_buf) =
            device.compute_geodesic_scores(&embeddings_buf, &anchor_refs, self.config.eps);
        let raw_scores = device.download_2d(&scores_buf);
        let angles = device.download_2d(&angles_buf);

        let mut scores = Array2::<f64>::zeros((b_size, num_classes));
        let mut argmax_anchor = Array2::<usize>::zeros((b_size, num_classes));
        for sample_idx in 0..b_size {
            for class_idx in 0..num_classes {
                let mut class_score = f64::NEG_INFINITY;
                let mut best_anchor_local = 0usize;
                for anchor_local in 0..anchors_per_class {
                    let anchor_col = class_idx * anchors_per_class + anchor_local;
                    let s = raw_scores[[sample_idx, anchor_col]];
                    if s > class_score {
                        class_score = s;
                        best_anchor_local = anchor_local;
                    }
                }
                scores[[sample_idx, class_idx]] = class_score;
                argmax_anchor[[sample_idx, class_idx]] = best_anchor_local;
            }
        }

        Ok(PatchwiseForwardCache {
            z_patches_dim: (b_size, num_patches, k_patch),
            scores,
            angles,
            argmax_anchor,
            patches_flat_buf,
            agg_flat_buf,
            embeddings_buf,
            interior_device: interior,
        })
    }

    fn validate_model_support(&self, model: &CosmosModelPatchwise) -> Result<(), TrainError> {
        if Model::anchors_per_class(model) == 0 {
            return Err(TrainError::task_adapter(
                "PatchwiseClassificationAdapter: anchors_per_class must be >= 1",
            ));
        }
        Ok(())
    }

    fn compute_metrics(&self, scores: &Array2<f64>, targets: &Targets, loss: f64) -> BatchMetrics {
        let accuracy = single_label_accuracy(scores, targets);
        BatchMetrics::new(scores.nrows(), loss, accuracy)
    }

    /// Per-class geodesic distance: the max-pool winner's angle.
    ///
    /// The score winner is also the minimal-angle anchor because
    /// `s = exp(−θ²)` is strictly decreasing in `θ`, so indexing the cached
    /// angles by the score argmax yields the per-class distance.
    fn class_distances(
        cache: &PatchwiseForwardCache,
        num_classes: usize,
        anchors_per_class: usize,
    ) -> Array2<f64> {
        let batch_size = cache.angles.nrows();
        if anchors_per_class == 1 {
            return cache.angles.clone();
        }
        Array2::from_shape_fn((batch_size, num_classes), |(i, c)| {
            cache.angles[[i, c * anchors_per_class + cache.argmax_anchor[[i, c]]]]
        })
    }

    /// Supervised loss on a completed forward.
    ///
    /// For the geodesic hinge the reported value is
    /// `hinge(Reduction::Mean) / num_classes`, matching the reference
    /// normalization by `batch · num_classes` and the analytic gradient the
    /// backward uses. Also returns the per-class distances the hinge backward
    /// consumes, `None` for the direct-membership loss.
    fn compute_batch_loss(
        &self,
        cache: &PatchwiseForwardCache,
        targets: &Targets,
        num_classes: usize,
        anchors_per_class: usize,
    ) -> Result<(f64, Option<Array2<f64>>), TrainError> {
        match self.config.loss {
            ClassificationLoss::DirectMembership => {
                let loss = direct_cmr_loss(&cache.scores, targets, Reduction::Mean)?[0];
                Ok((loss, None))
            }
            ClassificationLoss::GeodesicHinge { margin } => {
                let class_distances = Self::class_distances(cache, num_classes, anchors_per_class);
                let loss =
                    geodesic_cmr_hinge_loss(&class_distances, targets, margin, Reduction::Mean)?[0]
                        / num_classes as f64;
                Ok((loss, Some(class_distances)))
            }
        }
    }

    /// Backward pass over the device-resident forward buffers.
    ///
    /// Returns the block gradients plus the anchor, interior and input-projector
    /// timings.
    fn backward(
        &self,
        device: &Device,
        model: &CosmosModelPatchwise,
        batch: &PatchwiseClassificationBatch,
        cache: &PatchwiseForwardCache,
        class_distances: Option<&Array2<f64>>,
    ) -> Result<(BlockGradients, f64, f64, f64), TrainError> {
        let state = Model::state(model);
        let num_classes = Model::num_classes(model);
        let anchors_per_class = Model::anchors_per_class(model);
        let n_blocks = state.num_blocks();
        let mut gradients: BlockGradients = vec![None; n_blocks];

        // Per-class loss gradients. For the direct-membership loss these are
        // gradients with respect to the scores `s = exp(−θ²)`; for the geodesic
        // hinge they are gradients with respect to the pooled angles `θ`,
        // converted to score gradients after the anchor fan-out below.
        let per_class_grads = match self.config.loss {
            ClassificationLoss::DirectMembership => {
                direct_membership_score_grad(&cache.scores, &batch.targets)?
            }
            ClassificationLoss::GeodesicHinge { margin } => {
                let class_distances = class_distances.ok_or_else(|| {
                    TrainError::task_adapter(
                        "geodesic hinge backward requires class distances from forward",
                    )
                })?;
                // `dloss_dtheta` carries the `1 / (batch · num_classes)`
                // normalization — the exact gradient of the reported loss.
                geodesic_cmr_loss_grad(class_distances, &batch.targets, margin)?.dloss_dtheta
            }
        };

        // Route each per-class gradient to the single anchor that won the
        // max-pool in the forward (subgradient of `max`); losing anchors
        // receive zero.
        let total_anchors = num_classes * anchors_per_class;
        let mut grads_per_anchor = Array2::<f64>::zeros((per_class_grads.nrows(), total_anchors));
        if anchors_per_class == 1 {
            grads_per_anchor.assign(&per_class_grads);
        } else {
            for sample_idx in 0..per_class_grads.nrows() {
                for class_idx in 0..num_classes {
                    let winner = cache.argmax_anchor[[sample_idx, class_idx]];
                    let anchor_col = class_idx * anchors_per_class + winner;
                    grads_per_anchor[[sample_idx, anchor_col]] =
                        per_class_grads[[sample_idx, class_idx]];
                }
            }
        }
        // For the angle-supervised loss, `grads_per_anchor` holds ∂L/∂θ per
        // (sample, anchor); re-expressing it as an equivalent ∂L/∂s lets the
        // shared CMR score-backward kernel be reused unchanged.
        let score_grads_per_anchor = if self.config.loss.supervises_angles() {
            angle_grads_to_score_grads(&grads_per_anchor, &cache.angles, self.config.eps)
        } else {
            grads_per_anchor
        };
        let score_grads_buf = upload_basis(device, &score_grads_per_anchor);

        let anchor_start = Instant::now();
        let cmr_indices = Model::cmr_block_indices(model);
        let mut anchor_bufs: Vec<Buffer2<f64>> = Vec::with_capacity(cmr_indices.len());
        for &idx in cmr_indices {
            let anchor = state.get_basis(idx).map_err(|e| {
                TrainError::task_adapter(format!("read anchor {idx} failed: {e:?}"))
            })?;
            anchor_bufs.push(upload_basis(device, anchor));
        }
        let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();
        let embedding_norms_buf = device.row_norms(&cache.embeddings_buf);
        let (emb_rows, emb_cols) = cache.embeddings_buf.dim();
        let mut grad_embeddings_buf: Buffer2<f64> = device.zeros_2d(emb_rows, emb_cols);
        let mut grad_anchor_bufs: Vec<Buffer2<f64>> = anchor_bufs
            .iter()
            .map(|a| device.zeros_2d(a.nrows(), a.ncols()))
            .collect();
        cmr_multi_anchor_backward_device(
            device,
            &cache.embeddings_buf,
            &embedding_norms_buf,
            &anchor_refs,
            &score_grads_buf,
            self.config.eps,
            &mut grad_embeddings_buf,
            &mut grad_anchor_bufs,
        );
        for (i, &idx) in cmr_indices.iter().enumerate() {
            let grad = std::mem::replace(&mut grad_anchor_bufs[i], device.zeros_2d(0, 0));
            gradients[idx] = Some(grad);
        }
        let backward_anchor_sec = anchor_start.elapsed().as_secs_f64();

        // Interior backward: walks the gradient back through the cached chain,
        // accumulating per-block gradients along the way. A no-op clone when
        // the interior is absent.
        let interior_start = Instant::now();
        let grad_z_agg_buf = backward_supported_interior(
            device,
            state,
            cache.interior_device.as_ref(),
            &grad_embeddings_buf,
            &mut gradients,
        )?;
        let backward_interior_sec = interior_start.elapsed().as_secs_f64();

        // Aggregator backward. `z_agg = agg_flat @ W_agg` is a coordinate map,
        // so `dL/dW_agg = agg_flatᵀ (dL/dz_agg)` and
        // `dL/dagg_flat = (dL/dz_agg) W_aggᵀ`.
        let agg_start = Instant::now();
        let w_agg_basis = state
            .get_basis(model.agg_block_idx())
            .map_err(|e| TrainError::task_adapter(format!("read W_agg failed: {e:?}")))?;
        let w_agg_buf = upload_basis(device, w_agg_basis);
        let agg_flat_t = device.transpose(&cache.agg_flat_buf);
        let grad_w_agg_buf = device.matmul(&agg_flat_t, &grad_z_agg_buf);
        let w_agg_t = device.transpose(&w_agg_buf);
        let grad_agg_flat_buf = device.matmul(&grad_z_agg_buf, &w_agg_t);
        let grad_agg_flat = device.download_2d(&grad_agg_flat_buf);
        gradients[model.agg_block_idx()] = Some(grad_w_agg_buf);

        // Aggregate inverse: reshape or broadcast back to
        // `(batch, num_patches, k_patch)`.
        let (b_size, num_patches, k_patch) = cache.z_patches_dim;
        let grad_z_patches = aggregate_inverse(
            &grad_agg_flat,
            model.agg_mode(),
            b_size,
            num_patches,
            k_patch,
        );

        // Patch projector backward, the same coordinate-map rule as the
        // aggregator over the pooled `(B·NP, ·)` patch matrix.
        let grad_z_patches_flat = flatten_batched_patches(&grad_z_patches);
        let grad_z_patches_flat_buf = upload_basis(device, &grad_z_patches_flat);
        let patches_flat_t = device.transpose(&cache.patches_flat_buf);
        let grad_w_patch_buf = device.matmul(&patches_flat_t, &grad_z_patches_flat_buf);
        gradients[model.patch_block_idx()] = Some(grad_w_patch_buf);
        let backward_input_projector_sec = agg_start.elapsed().as_secs_f64();

        Ok((
            gradients,
            backward_anchor_sec,
            backward_interior_sec,
            backward_input_projector_sec,
        ))
    }
}

/// Forward-pass cache used by the patchwise backward path.
///
/// The host side keeps only the small tensors the loss and the gradient fan-out
/// need; the large intermediates stay resident on the device.
#[derive(Debug)]
struct PatchwiseForwardCache {
    /// Dimensions of the projected-patch tensor `(batch, num_patches, k_patch)`;
    /// drives the aggregate-inverse reshape in the backward.
    z_patches_dim: (usize, usize, usize),
    /// Per-class scores `s_c = exp(−θ_c²)` in shape `(batch, num_classes)`.
    scores: Array2<f64>,
    /// Per-(sample, anchor) geodesic angles `θ` in shape
    /// `(batch, num_classes · anchors_per_class)`.
    angles: Array2<f64>,
    /// For each `(sample, class)`, the local anchor index in
    /// `0..anchors_per_class` that won the max-pool in the forward.
    argmax_anchor: Array2<usize>,
    patches_flat_buf: Buffer2<f64>,
    agg_flat_buf: Buffer2<f64>,
    embeddings_buf: Buffer2<f64>,
    interior_device: Option<InteriorForwardDevice>,
}

/// Invert the patch aggregator to produce the per-patch gradient.
fn aggregate_inverse(
    grad_agg_flat: &Array2<f64>,
    mode: PatchAggregationMode,
    batch_size: usize,
    num_patches: usize,
    k_patch: usize,
) -> Array3<f64> {
    if batch_size == 0 || num_patches == 0 || k_patch == 0 {
        return Array3::zeros((batch_size, num_patches, k_patch));
    }

    match mode {
        PatchAggregationMode::Concat => {
            let grad = grad_agg_flat.slice(ndarray::s![..batch_size, ..num_patches * k_patch]);
            grad.as_standard_layout()
                .into_shape_with_order((batch_size, num_patches, k_patch))
                .expect("standard-layout aggregate dimensions preserve the element count")
                .into_owned()
        }
        PatchAggregationMode::Mean => {
            let scale = 1.0 / (num_patches as f64);
            grad_agg_flat
                .slice(ndarray::s![..batch_size, ..k_patch])
                .insert_axis(Axis(1))
                .broadcast((batch_size, num_patches, k_patch))
                .expect("aggregate gradient dimensions match the patch layout")
                .mapv(|value| scale * value)
        }
        PatchAggregationMode::Sum => grad_agg_flat
            .slice(ndarray::s![..batch_size, ..k_patch])
            .insert_axis(Axis(1))
            .broadcast((batch_size, num_patches, k_patch))
            .expect("aggregate gradient dimensions match the patch layout")
            .to_owned(),
    }
}

/// Pack a 3D array `(B, NP, k)` into a 2D array `(B·NP, k)` in row-major order.
fn flatten_batched_patches(a: &Array3<f64>) -> Array2<f64> {
    let (b, np, k) = a.dim();
    a.as_standard_layout()
        .into_shape_with_order((b * np, k))
        .expect("standard-layout patch dimensions preserve the element count")
        .into_owned()
}

/// Inverse of [`flatten_batched_patches`].
fn unflatten_batched_patches(a: &Array2<f64>, b: usize, np: usize, k: usize) -> Array3<f64> {
    if b == 0 || np == 0 || k == 0 {
        return Array3::zeros((b, np, k));
    }

    a.slice(ndarray::s![..b * np, ..k])
        .as_standard_layout()
        .into_shape_with_order((b, np, k))
        .expect("standard-layout patch dimensions preserve the element count")
        .into_owned()
}

impl<B: TrainingBackend> TaskAdapter<CosmosModelPatchwise, B> for PatchwiseClassificationAdapter {
    type Batch = PatchwiseClassificationBatch;

    fn train_batch(
        &mut self,
        model: &CosmosModelPatchwise,
        batch: &Self::Batch,
        backend: &mut B,
        _collect_trace: bool,
    ) -> Result<TrainBatchOutput, TrainError> {
        self.validate_model_support(model)?;

        let forward_start = Instant::now();
        let cache = self.forward_with_cache(model, batch, backend)?;
        let forward_sec = forward_start.elapsed().as_secs_f64();

        let num_classes = Model::num_classes(model);
        let anchors_per_class = Model::anchors_per_class(model);
        let (loss, class_distances) =
            self.compute_batch_loss(&cache, &batch.targets, num_classes, anchors_per_class)?;
        let metrics = self.compute_metrics(&cache.scores, &batch.targets, loss);

        let backward_start = Instant::now();
        let device = backend.device();
        let (gradients, backward_anchor_sec, backward_interior_sec, backward_input_projector_sec) =
            self.backward(device, model, batch, &cache, class_distances.as_ref())?;
        let backward_sec = backward_start.elapsed().as_secs_f64();

        Ok(TrainBatchOutput {
            metrics,
            gradients,
            trace: None,
            timing: BatchTimingProfile {
                forward_sec,
                backward_sec,
                backward_anchor_sec,
                backward_interior_sec,
                backward_input_projector_sec,
                trace_sec: 0.0,
            },
        })
    }

    fn eval_batch(
        &mut self,
        model: &CosmosModelPatchwise,
        batch: &Self::Batch,
        backend: &mut B,
        _collect_trace: bool,
    ) -> Result<EvalBatchOutput, TrainError> {
        self.validate_model_support(model)?;

        let cache = self.forward_with_cache(model, batch, backend)?;
        let (loss, _class_distances) = self.compute_batch_loss(
            &cache,
            &batch.targets,
            Model::num_classes(model),
            Model::anchors_per_class(model),
        )?;
        let metrics = self.compute_metrics(&cache.scores, &batch.targets, loss);
        Ok(EvalBatchOutput {
            metrics,
            trace: None,
            timing: BatchTimingProfile::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::CpuBackend;
    use super::*;
    use crate::arrangement::input::PatchAggregationMode;
    use crate::models::patchwise_model::PatchwiseModelConfig;

    #[test]
    fn aggregate_inverse_matches_scalar_reference_for_all_modes() {
        let (batch, num_patches, k) = (5, 7, 16);
        for mode in [
            PatchAggregationMode::Concat,
            PatchAggregationMode::Mean,
            PatchAggregationMode::Sum,
        ] {
            let columns = if mode == PatchAggregationMode::Concat {
                num_patches * k
            } else {
                k
            };
            let grad_agg_flat = Array2::from_shape_fn((batch, columns), |(b, f)| {
                ((b * columns + f) as f64 - 200.0) / 16.0
            });
            let actual = aggregate_inverse(&grad_agg_flat, mode, batch, num_patches, k);
            let mut expected = Array3::<f64>::zeros((batch, num_patches, k));
            match mode {
                PatchAggregationMode::Concat => {
                    for b in 0..batch {
                        let mut flat_idx = 0usize;
                        for p in 0..num_patches {
                            for f in 0..k {
                                expected[(b, p, f)] = grad_agg_flat[(b, flat_idx)];
                                flat_idx += 1;
                            }
                        }
                    }
                    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                        assert_eq!(a.to_bits(), e.to_bits(), "bit mismatch at {index}");
                    }
                }
                PatchAggregationMode::Mean => {
                    let scale = 1.0 / num_patches as f64;
                    for b in 0..batch {
                        for p in 0..num_patches {
                            for f in 0..k {
                                expected[(b, p, f)] = scale * grad_agg_flat[(b, f)];
                            }
                        }
                    }
                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!((a - e).abs() <= 1e-12, "expected {e}, got {a}");
                    }
                }
                PatchAggregationMode::Sum => {
                    for b in 0..batch {
                        for p in 0..num_patches {
                            for f in 0..k {
                                expected[(b, p, f)] = grad_agg_flat[(b, f)];
                            }
                        }
                    }
                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!((a - e).abs() <= 1e-12, "expected {e}, got {a}");
                    }
                }
            }
        }
    }

    #[test]
    fn flatten_batched_patches_matches_scalar_reference() {
        let (batch, num_patches, k) = (5, 7, 16);
        let input = Array3::from_shape_fn((batch, num_patches, k), |(b, p, f)| {
            ((b * num_patches + p) * k + f) as f64 / 16.0
        });
        let actual = flatten_batched_patches(&input);
        let mut expected = Array2::<f64>::zeros((batch * num_patches, k));
        for b in 0..batch {
            for p in 0..num_patches {
                for f in 0..k {
                    expected[(b * num_patches + p, f)] = input[(b, p, f)];
                }
            }
        }
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "bit mismatch at {index}");
        }
    }

    #[test]
    fn unflatten_batched_patches_matches_scalar_reference() {
        let (batch, num_patches, k) = (5, 7, 16);
        let input = Array2::from_shape_fn((batch * num_patches, k), |(bp, f)| {
            (bp * k + f) as f64 / 16.0
        });
        let actual = unflatten_batched_patches(&input, batch, num_patches, k);
        let mut expected = Array3::<f64>::zeros((batch, num_patches, k));
        for b in 0..batch {
            for p in 0..num_patches {
                for f in 0..k {
                    expected[(b, p, f)] = input[(b * num_patches + p, f)];
                }
            }
        }
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "bit mismatch at {index}");
        }
    }

    /// The eval forward produces a well-shaped score matrix whose entries lie
    /// in `(0, 1]` for every class (Gaussian-in-angle scores).
    #[test]
    fn forward_produces_valid_scores() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(2)
            .with_cmr_k(2)
            .with_anchors_per_class(1)
            .with_agg_mode(PatchAggregationMode::Mean);
        let model = CosmosModelPatchwise::new(cfg).expect("model builds");
        let inputs = Array4::<f64>::from_elem((3, 1, 4, 4), 0.5);
        let batch = PatchwiseClassificationBatch::single_label(inputs, vec![0, 1, 0]);
        let mut adapter = PatchwiseClassificationAdapter::default();
        let mut backend = CpuBackend::default();
        let out = adapter
            .eval_batch(&model, &batch, &mut backend, false)
            .expect("eval forward should succeed");
        // Sanity: metrics.samples matches batch size, accuracy is
        // computable (Some), and not NaN.
        assert_eq!(out.metrics.samples, 3);
        let acc = out.metrics.accuracy.expect("accuracy reported");
        assert!(
            (0.0..=100.0).contains(&acc),
            "accuracy in [0,100], got {}",
            acc
        );
    }

    /// `train_batch` on a no-interior model produces correctly shaped gradients
    /// that match a central-difference check against the analytic loss.
    ///
    /// NB: keep `cmr_k < agg_subspace_dim`. When the two are equal and
    /// the anchors are orthonormal (default seed via QR), every CMR
    /// anchor is a square orthogonal matrix, so cosine ≡ 1 and the
    /// score derivative hits its limit branch — perturbations toward
    /// cosine > 1 get clamped while perturbations toward cosine < 1
    /// move freely, breaking the central-difference symmetry that FD
    /// relies on. With `cmr_k < agg_subspace_dim` the projection is
    /// strictly lower-rank and cosine ∈ (0, 1) generically.
    #[test]
    fn train_batch_finite_difference_check() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(3) // > cmr_k so cosine isn't degenerate-1
            .with_num_interior_blocks(0)
            .with_cmr_k(1)
            .with_anchors_per_class(1)
            .with_agg_mode(PatchAggregationMode::Concat);

        let mut model = CosmosModelPatchwise::new(cfg).expect("model builds");

        // Randomise the state with deterministic seed for reproducibility.
        // We orthonormalise via QR to keep each block on the Stiefel manifold.
        seed_state_with_qr(model.get_state_mut(), 0xC051_2A57);

        // Inputs with non-trivial structure so the gradients are non-zero.
        let mut inputs = Array4::<f64>::zeros((2, 1, 4, 4));
        let mut seed = 1u64;
        for v in inputs.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as i32 as f64 / (1u64 << 31) as f64) * 0.3;
        }
        let batch = PatchwiseClassificationBatch::single_label(inputs.clone(), vec![0, 1]);

        let mut adapter = PatchwiseClassificationAdapter::default();
        let mut backend = CpuBackend::default();

        let out = adapter
            .train_batch(&model, &batch, &mut backend, false)
            .expect("train_batch should succeed");

        // Shape/presence checks (snapshot block info before any mutation).
        let block_shapes: Vec<(usize, usize)> = model.get_state().block_shapes().to_vec();
        let n_state_blocks = block_shapes.len();
        assert_eq!(
            out.gradients.len(),
            n_state_blocks,
            "BlockGradients must be sized to state.num_blocks()"
        );
        assert!(
            out.gradients[model.patch_block_idx()].is_some(),
            "patch projector gradient missing"
        );
        assert!(
            out.gradients[model.agg_block_idx()].is_some(),
            "aggregator gradient missing"
        );
        let cmr_indices: Vec<usize> = Model::cmr_block_indices(&model).to_vec();
        for &idx in &cmr_indices {
            assert!(
                out.gradients[idx].is_some(),
                "CMR anchor gradient missing at idx={idx}"
            );
        }
        for (idx, g) in out.gradients.iter().enumerate() {
            if let Some(g) = g {
                assert_eq!(g.dim(), block_shapes[idx], "shape mismatch at block {idx}");
            }
        }

        // Snapshot every block's basis up front so we can restore after
        // each perturbation without holding a long-lived borrow.
        let initial_bases: Vec<Array2<f64>> = (0..n_state_blocks)
            .map(|i| {
                model
                    .get_state()
                    .get_basis(i)
                    .expect("basis present")
                    .clone()
            })
            .collect();

        // Pre-download analytic gradients to ndarray.
        let device = backend.device();
        let analytic_grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| device.download_2d(b)))
            .collect();

        // Finite-difference check. Central differences, h = 1e-5.
        // Tolerance budget: FD truncation O(h²) ≈ 1e-10 plus f64
        // cancellation noise scaling like ‖loss‖·1e-16 / h ≈ 1e-11.
        // Allow 5e-5 absolute or 5e-3 relative, whichever larger, for
        // the geodesic/arccos non-linearity which has bounded but
        // non-trivial second derivatives.
        let h = 1e-5_f64;
        let abs_tol = 5e-5_f64;
        let rel_tol = 5e-3_f64;

        let blocks_to_check: Vec<usize> = (0..n_state_blocks)
            .filter(|&i| analytic_grads[i].is_some())
            .collect();
        for block_idx in blocks_to_check {
            let g_analytic = analytic_grads[block_idx].as_ref().unwrap();
            let base = &initial_bases[block_idx];
            let (rows, cols) = base.dim();

            for r in 0..rows {
                for c in 0..cols {
                    // +h
                    let mut perturbed = base.clone();
                    perturbed[(r, c)] += h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_plus = loss_at(&adapter, &model, &batch);

                    // -h
                    let mut perturbed = base.clone();
                    perturbed[(r, c)] -= h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_minus = loss_at(&adapter, &model, &batch);

                    // Restore to baseline.
                    write_block(model.get_state_mut(), block_idx, base, &initial_bases);

                    let g_fd = (loss_plus - loss_minus) / (2.0 * h);
                    let g_an = g_analytic[(r, c)];
                    let err = (g_fd - g_an).abs();
                    let rel = err / g_an.abs().max(1e-12);
                    assert!(
                        err < abs_tol || rel < rel_tol,
                        "FD mismatch at block {block_idx} ({r},{c}): analytic={g_an:.6e}, fd={g_fd:.6e}, abs_err={err:.3e}, rel_err={rel:.3e}"
                    );
                }
            }
        }
    }

    /// Loss under the adapter's configured `ClassificationLoss` — the
    /// exact value `train_batch` reports, so FD sweeps differentiate
    /// the same scalar the analytic backward implements.
    fn loss_at(
        adapter: &PatchwiseClassificationAdapter,
        model: &CosmosModelPatchwise,
        batch: &PatchwiseClassificationBatch,
    ) -> f64 {
        let mut backend = CpuBackend::default();
        let cache = adapter
            .forward_with_cache(model, batch, &mut backend)
            .expect("forward_with_cache");
        adapter
            .compute_batch_loss(
                &cache,
                &batch.targets,
                model.num_classes(),
                model.anchors_per_class(),
            )
            .expect("compute_batch_loss")
            .0
    }

    /// Finite-difference check with a non-trivial interior chain: the gradient
    /// flowing back through `backward_supported_interior` (shared with the flat
    /// path) matches the actual loss gradient at every block.
    ///
    /// Tolerance budget is the same as the no-interior case but
    /// scaled up slightly (`abs_tol = 1e-4`, `rel_tol = 1e-2`) because
    /// the interior pass introduces extra numerical noise via the
    /// internal projector / activation chain.
    #[test]
    fn train_batch_finite_difference_check_with_interior() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(3)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(2)
            .with_cmr_k(1)
            .with_anchors_per_class(1)
            .with_agg_mode(PatchAggregationMode::Concat);

        let mut model = CosmosModelPatchwise::new(cfg).expect("model builds");
        seed_state_with_qr(model.get_state_mut(), 0xC051_2A57);

        // Random inputs, deterministic seed.
        let mut inputs = Array4::<f64>::zeros((2, 1, 4, 4));
        let mut seed = 1u64;
        for v in inputs.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as i32 as f64 / (1u64 << 31) as f64) * 0.3;
        }
        let batch = PatchwiseClassificationBatch::single_label(inputs, vec![0, 1]);

        let mut adapter = PatchwiseClassificationAdapter::default();
        let mut backend = CpuBackend::default();
        let out = adapter
            .train_batch(&model, &batch, &mut backend, false)
            .expect("train_batch with interior");

        // Snapshot baseline bases.
        let block_shapes: Vec<(usize, usize)> = model.get_state().block_shapes().to_vec();
        let n_state_blocks = block_shapes.len();
        let initial_bases: Vec<Array2<f64>> = (0..n_state_blocks)
            .map(|i| {
                model
                    .get_state()
                    .get_basis(i)
                    .expect("basis present")
                    .clone()
            })
            .collect();

        let device = backend.device();
        let analytic_grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| device.download_2d(b)))
            .collect();

        let h = 1e-5_f64;
        let abs_tol = 1e-4_f64;
        let rel_tol = 1e-2_f64;

        let blocks_to_check: Vec<usize> = (0..n_state_blocks)
            .filter(|&i| analytic_grads[i].is_some())
            .collect();
        for block_idx in blocks_to_check {
            let g_analytic = analytic_grads[block_idx].as_ref().unwrap();
            let base = &initial_bases[block_idx];
            let (rows, cols) = base.dim();

            for r in 0..rows {
                for c in 0..cols {
                    let mut perturbed = base.clone();
                    perturbed[(r, c)] += h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_plus = loss_at(&adapter, &model, &batch);

                    let mut perturbed = base.clone();
                    perturbed[(r, c)] -= h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_minus = loss_at(&adapter, &model, &batch);

                    write_block(model.get_state_mut(), block_idx, base, &initial_bases);

                    let g_fd = (loss_plus - loss_minus) / (2.0 * h);
                    let g_an = g_analytic[(r, c)];
                    let err = (g_fd - g_an).abs();
                    let rel = err / g_an.abs().max(1e-12);
                    assert!(
                        err < abs_tol || rel < rel_tol,
                        "interior-FD mismatch at block {block_idx} ({r},{c}): analytic={g_an:.6e}, fd={g_fd:.6e}, abs_err={err:.3e}, rel_err={rel:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    fn patchwise_hinge_config_rejects_negative_margin() {
        let result = PatchwiseClassificationAdapter::new(PatchwiseClassificationAdapterConfig {
            loss: ClassificationLoss::GeodesicHinge { margin: -0.1 },
            ..Default::default()
        });
        assert!(result.is_err());
    }

    /// Finite-difference check of the full backward in geodesic-hinge
    /// mode, WITH a non-trivial interior chain. Exercises the hinge
    /// per-class gradient, the anchor fan-out, and the θ-grad →
    /// score-grad conversion feeding the shared CMR kernel. The margin
    /// is chosen away from every baseline angle so ±h perturbations
    /// cannot cross the hinge kink (asserted below). Tolerance budget
    /// matches the interior direct-membership FD test.
    #[test]
    fn patchwise_hinge_gradient_finite_difference() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(3)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(2)
            .with_cmr_k(1)
            .with_anchors_per_class(1)
            .with_agg_mode(PatchAggregationMode::Concat);

        let mut model = CosmosModelPatchwise::new(cfg).expect("model builds");
        seed_state_with_qr(model.get_state_mut(), 0xC051_2A57);

        // Random inputs, deterministic seed.
        let mut inputs = Array4::<f64>::zeros((2, 1, 4, 4));
        let mut seed = 1u64;
        for v in inputs.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as i32 as f64 / (1u64 << 31) as f64) * 0.3;
        }
        let batch = PatchwiseClassificationBatch::single_label(inputs, vec![0, 1]);

        let margin = 0.6;
        let mut adapter =
            PatchwiseClassificationAdapter::new(PatchwiseClassificationAdapterConfig {
                loss: ClassificationLoss::GeodesicHinge { margin },
                ..Default::default()
            })
            .expect("hinge adapter builds");
        let mut backend = CpuBackend::default();

        // Precondition: every baseline angle is far from the hinge kink
        // at θ = margin, so ±h perturbations cannot cross it and the
        // central difference stays smooth.
        {
            let cache = adapter
                .forward_with_cache(&model, &batch, &mut backend)
                .expect("baseline forward");
            for d in cache.angles.iter() {
                assert!(
                    (d - margin).abs() > 1e-2,
                    "baseline angle {d} too close to margin {margin} for a clean FD check"
                );
            }
        }

        let out = adapter
            .train_batch(&model, &batch, &mut backend, false)
            .expect("train_batch in hinge mode");

        // Snapshot baseline bases.
        let block_shapes: Vec<(usize, usize)> = model.get_state().block_shapes().to_vec();
        let n_state_blocks = block_shapes.len();
        let initial_bases: Vec<Array2<f64>> = (0..n_state_blocks)
            .map(|i| {
                model
                    .get_state()
                    .get_basis(i)
                    .expect("basis present")
                    .clone()
            })
            .collect();

        let device = backend.device();
        let analytic_grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| device.download_2d(b)))
            .collect();

        let h = 1e-5_f64;
        let abs_tol = 1e-4_f64;
        let rel_tol = 1e-2_f64;

        let blocks_to_check: Vec<usize> = (0..n_state_blocks)
            .filter(|&i| analytic_grads[i].is_some())
            .collect();
        assert!(
            !blocks_to_check.is_empty(),
            "hinge backward produced no gradients"
        );
        for block_idx in blocks_to_check {
            let g_analytic = analytic_grads[block_idx].as_ref().unwrap();
            let base = &initial_bases[block_idx];
            let (rows, cols) = base.dim();

            for r in 0..rows {
                for c in 0..cols {
                    let mut perturbed = base.clone();
                    perturbed[(r, c)] += h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_plus = loss_at(&adapter, &model, &batch);

                    let mut perturbed = base.clone();
                    perturbed[(r, c)] -= h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_minus = loss_at(&adapter, &model, &batch);

                    write_block(model.get_state_mut(), block_idx, base, &initial_bases);

                    let g_fd = (loss_plus - loss_minus) / (2.0 * h);
                    let g_an = g_analytic[(r, c)];
                    let err = (g_fd - g_an).abs();
                    let rel = err / g_an.abs().max(1e-12);
                    assert!(
                        err < abs_tol || rel < rel_tol,
                        "hinge-FD mismatch at block {block_idx} ({r},{c}): analytic={g_an:.6e}, fd={g_fd:.6e}, abs_err={err:.3e}, rel_err={rel:.3e}"
                    );
                }
            }
        }
    }

    /// Finite-difference check with `anchors_per_class > 1`.
    /// Validates that the max-pool subgradient correctly routes only
    /// to the winning anchor per (sample, class). Tolerance budget is
    /// loosened slightly because the max-pool boundary is a kink: an
    /// anchor entry whose perturbation flips the argmax produces a
    /// discontinuous FD value. We mitigate by initialising anchors
    /// such that one wins each (sample, class) by a comfortable
    /// margin (different QR seeds), then check that the FD gradient
    /// matches at points where the argmax is stable.
    ///
    /// Tolerance: `abs_tol=2e-4` or `rel_tol=2e-2`. Looser than the
    /// single-anchor case because the per-class max-pool introduces a
    /// boundary near `cos θ_winner ≈ cos θ_runner_up`. The seed
    /// (0xC051_2A57 with two different QR draws) yields well-separated
    /// margins on this tiny model in practice.
    #[test]
    fn train_batch_finite_difference_check_multi_anchor() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(3)
            .with_num_interior_blocks(0)
            .with_cmr_k(1)
            .with_anchors_per_class(2)
            .with_agg_mode(PatchAggregationMode::Concat);

        let mut model = CosmosModelPatchwise::new(cfg).expect("model builds");
        seed_state_with_qr(model.get_state_mut(), 0xC051_2A57);

        // Random inputs, deterministic seed.
        let mut inputs = Array4::<f64>::zeros((2, 1, 4, 4));
        let mut seed = 1u64;
        for v in inputs.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as i32 as f64 / (1u64 << 31) as f64) * 0.3;
        }
        let batch = PatchwiseClassificationBatch::single_label(inputs, vec![0, 1]);

        let mut adapter = PatchwiseClassificationAdapter::default();
        let mut backend = CpuBackend::default();
        let out = adapter
            .train_batch(&model, &batch, &mut backend, false)
            .expect("train_batch with multi-anchor");

        // Shape and presence checks.
        let block_shapes: Vec<(usize, usize)> = model.get_state().block_shapes().to_vec();
        let n_state_blocks = block_shapes.len();
        assert!(out.gradients[model.patch_block_idx()].is_some());
        assert!(out.gradients[model.agg_block_idx()].is_some());
        let cmr_indices: Vec<usize> = Model::cmr_block_indices(&model).to_vec();
        // Every CMR anchor must get a (possibly zero) gradient — the
        // kernel always allocates one per anchor and writes to it.
        for &idx in &cmr_indices {
            assert!(
                out.gradients[idx].is_some(),
                "CMR anchor gradient missing at idx={idx}"
            );
        }

        let initial_bases: Vec<Array2<f64>> = (0..n_state_blocks)
            .map(|i| model.get_state().get_basis(i).expect("basis").clone())
            .collect();

        let device = backend.device();
        let analytic_grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| device.download_2d(b)))
            .collect();

        let h = 1e-5_f64;
        let abs_tol = 2e-4_f64;
        let rel_tol = 2e-2_f64;

        let blocks_to_check: Vec<usize> = (0..n_state_blocks)
            .filter(|&i| analytic_grads[i].is_some())
            .collect();
        for block_idx in blocks_to_check {
            let g_analytic = analytic_grads[block_idx].as_ref().unwrap();
            let base = &initial_bases[block_idx];
            let (rows, cols) = base.dim();

            for r in 0..rows {
                for c in 0..cols {
                    let mut perturbed = base.clone();
                    perturbed[(r, c)] += h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_plus = loss_at(&adapter, &model, &batch);

                    let mut perturbed = base.clone();
                    perturbed[(r, c)] -= h;
                    write_block(model.get_state_mut(), block_idx, &perturbed, &initial_bases);
                    let loss_minus = loss_at(&adapter, &model, &batch);

                    write_block(model.get_state_mut(), block_idx, base, &initial_bases);

                    let g_fd = (loss_plus - loss_minus) / (2.0 * h);
                    let g_an = g_analytic[(r, c)];
                    let err = (g_fd - g_an).abs();
                    let rel = err / g_an.abs().max(1e-12);
                    assert!(
                        err < abs_tol || rel < rel_tol,
                        "multi-anchor FD mismatch at block {block_idx} ({r},{c}): analytic={g_an:.6e}, fd={g_fd:.6e}, abs_err={err:.3e}, rel_err={rel:.3e}"
                    );
                }
            }
        }
    }

    /// `train_batch` with a non-trivial interior chain runs end to end and
    /// produces gradients on every block (patch, aggregator, interior, CMR).
    /// Shape sanity only — finite-difference validation lives in
    /// `train_batch_finite_difference_check_with_interior`.
    #[test]
    fn train_batch_with_interior_blocks_produces_gradients() {
        let cfg = PatchwiseModelConfig::new((1, 4, 4), 2)
            .with_patch_size(2)
            .with_patch_subspace_dim(2)
            .with_agg_subspace_dim(3)
            .with_num_interior_blocks(2)
            .with_interior_subspace_dim(2)
            .with_cmr_k(1)
            .with_anchors_per_class(1)
            .with_agg_mode(PatchAggregationMode::Concat);
        let mut model = CosmosModelPatchwise::new(cfg).expect("model builds");
        seed_state_with_qr(model.get_state_mut(), 0xC051_2A57);

        let inputs = Array4::<f64>::from_elem((2, 1, 4, 4), 0.5);
        let batch = PatchwiseClassificationBatch::single_label(inputs, vec![0, 1]);
        let mut adapter = PatchwiseClassificationAdapter::default();
        let mut backend = CpuBackend::default();
        let out = adapter
            .train_batch(&model, &batch, &mut backend, false)
            .expect("train_batch should succeed with an interior chain");

        // Patch projector, aggregator, every interior block, and every
        // CMR anchor must receive a non-zero gradient.
        assert!(out.gradients[model.patch_block_idx()].is_some());
        assert!(out.gradients[model.agg_block_idx()].is_some());
        for idx in model.interior_block_range() {
            assert!(
                out.gradients[idx].is_some(),
                "interior block gradient missing at idx={idx}"
            );
        }
        for &idx in Model::cmr_block_indices(&model) {
            assert!(
                out.gradients[idx].is_some(),
                "CMR anchor gradient missing at idx={idx}"
            );
        }
    }

    // ─── helpers ──────────────────────────────────────────────────────

    /// Seed every block in the state with a deterministic
    /// orthonormalised matrix derived from a PCG-style PRNG.
    fn seed_state_with_qr(state: &mut crate::state::cosmos_state::CosmosState, seed: u64) {
        use crate::geometry::qr::{QrConfig, qr_orthonormalize};
        let shapes = state.block_shapes().to_vec();
        let mut rng_state = seed;
        let mut next_normal = || -> f64 {
            // Box-Muller atop two PCG samples — simple, deterministic, good enough.
            let mut u = || -> f64 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng_state >> 33) as u32) as f64 / (1u64 << 32) as f64
            };
            let r = (-2.0 * u().max(1e-300).ln()).sqrt();
            r * (2.0 * std::f64::consts::PI * u()).cos()
        };

        // Build a global basis with the right block-diagonal shape.
        let total_n: usize = shapes.iter().map(|(n, _)| n).sum();
        let total_k: usize = shapes.iter().map(|(_, k)| k).sum();
        let mut global = Array2::<f64>::zeros((total_n, total_k));
        let mut row_off = 0usize;
        let mut col_off = 0usize;
        for &(n, k) in &shapes {
            let mut raw = Array2::<f64>::zeros((n, k));
            for v in raw.iter_mut() {
                *v = next_normal();
            }
            let ortho = qr_orthonormalize(&raw, &QrConfig::default());
            for i in 0..n {
                for j in 0..k {
                    global[(row_off + i, col_off + j)] = ortho[(i, j)];
                }
            }
            row_off += n;
            col_off += k;
        }
        state.update_from_global_basis(&global);
    }

    /// Overwrite a single state block with the provided basis. Pulls
    /// non-target blocks from the pre-captured `initial_bases` slice
    /// to avoid a long-lived immutable borrow on the state itself.
    fn write_block(
        state: &mut crate::state::cosmos_state::CosmosState,
        block_idx: usize,
        new_basis: &Array2<f64>,
        initial_bases: &[Array2<f64>],
    ) {
        let shapes = state.block_shapes().to_vec();
        let total_n: usize = shapes.iter().map(|(n, _)| n).sum();
        let total_k: usize = shapes.iter().map(|(_, k)| k).sum();
        let mut global = Array2::<f64>::zeros((total_n, total_k));
        let mut row_off = 0usize;
        let mut col_off = 0usize;
        for (idx, &(n, k)) in shapes.iter().enumerate() {
            let src = if idx == block_idx {
                new_basis
            } else {
                &initial_bases[idx]
            };
            for i in 0..n {
                for j in 0..k {
                    global[(row_off + i, col_off + j)] = src[(i, j)];
                }
            }
            row_off += n;
            col_off += k;
        }
        state.update_from_global_basis(&global);
    }
}
