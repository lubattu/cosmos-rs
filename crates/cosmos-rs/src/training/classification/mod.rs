//! Flat classification adapter.
//!
//! Supervises a [`CosmosModel`] on batch-major feature vectors: the input
//! projector lifts the batch, the supported interior chain transforms it and
//! the CMR head scores it against the subspace anchors of each class. Both
//! directions are hand-written; the forward caches every buffer the backward
//! needs.

use std::time::Instant;

use ndarray::Array2;

use crate::{
    compute::device::{Buffer2, ComputeDevice},
    models::{CosmosModel, Model},
    objectives::utils::Targets,
};

use super::{
    TrainError,
    backend::{CpuBackend, TrainingBackend},
    task_adapter::{
        BatchMetrics, BatchTimingProfile, EvalBatchOutput, TaskAdapter, TrainBatchOutput,
    },
};

pub(crate) mod backward;
pub(crate) mod forward;
pub(crate) mod loss;
pub(crate) mod shared;

use backward::{backward, build_trace};
use forward::forward_batch;
pub(crate) use loss::direct_membership_score_grad;
use loss::validate_model_support;
pub(crate) use shared::{
    InteriorForwardDevice, angle_grads_to_score_grads, backward_supported_interior,
    forward_supported_interior_device, single_label_accuracy, upload_basis,
};

#[cfg(test)]
mod tests;

/// Numerical floor shared by the norm-stabilized derivatives and the score
/// kernel.
pub(crate) const DEFAULT_EPS: f64 = 1e-8;

/// Batch payload for flat supervised classification.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatClassificationBatch {
    /// Batch-major flat inputs.
    pub inputs: Array2<f64>,
    /// Single-label or multi-label supervision targets.
    pub targets: Targets,
}

impl FlatClassificationBatch {
    /// Construct a new batch.
    pub fn new(inputs: Array2<f64>, targets: Targets) -> Self {
        Self { inputs, targets }
    }

    /// Convenience constructor for single-label classification.
    pub fn single_label(inputs: Array2<f64>, labels: Vec<usize>) -> Self {
        Self {
            inputs,
            targets: Targets::SingleLabel(labels),
        }
    }

    fn samples(&self) -> usize {
        self.inputs.nrows()
    }
}

/// Supervised objective driving the classification adapters.
///
/// Both variants consume the Gaussian-angular geometry produced by
/// `compute_geodesic_scores` — the per-(sample, anchor) angle `θ` and the
/// membership score `s = exp(−θ²)`; they differ in which of the two is
/// supervised.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ClassificationLoss {
    /// Softmax cross-entropy over CMR membership scores.
    #[default]
    DirectMembership,
    /// The paper's canonical loss: the Grassmann angular hinge, with `margin`
    /// in radians. The certified runs use `margin = PI / 5`.
    GeodesicHinge {
        /// Angular margin in radians.
        margin: f64,
    },
}

impl ClassificationLoss {
    /// Whether the loss is supervised on the per-class geodesic angles rather
    /// than on the membership scores.
    ///
    /// Angle-supervised losses need the raw `θ` matrix from the score kernel in
    /// both directions of the pass.
    pub(crate) fn supervises_angles(self) -> bool {
        !matches!(self, Self::DirectMembership)
    }
}

/// Configuration for the flat classification adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatClassificationAdapterConfig {
    /// Numerical epsilon used in stabilized norm-based derivatives.
    pub eps: f64,
    /// Supervised loss. Default: [`ClassificationLoss::DirectMembership`].
    pub loss: ClassificationLoss,
}

impl Default for FlatClassificationAdapterConfig {
    fn default() -> Self {
        Self {
            eps: DEFAULT_EPS,
            loss: ClassificationLoss::default(),
        }
    }
}

impl FlatClassificationAdapterConfig {
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

/// Manual adapter for flat classification with deterministic interiors.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FlatClassificationAdapter {
    config: FlatClassificationAdapterConfig,
}

impl FlatClassificationAdapter {
    /// Create a new adapter.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidConfig`] when `eps` or the loss margin is
    /// out of range.
    pub fn new(config: FlatClassificationAdapterConfig) -> Result<Self, TrainError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Borrow the adapter configuration.
    pub fn config(&self) -> &FlatClassificationAdapterConfig {
        &self.config
    }

    /// Per-(sample, class) geodesic angles `θ` for a raw input batch.
    ///
    /// Runs the same embedding and geodesic-score stages as the evaluation
    /// forward — input projection, supported interior, and
    /// `compute_geodesic_scores`. The traversal reads model state only and
    /// leaves the model's runtime caches untouched. For models with multiple
    /// anchors per class, each entry is the minimal angle over that class's
    /// anchors — the same winner the evaluation score max-pool selects, because
    /// `s = exp(−θ²)` is strictly decreasing in `θ`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the model arrangement is unsupported or the
    /// input width does not match the input projector.
    pub fn class_angles(
        &self,
        model: &CosmosModel,
        inputs: &Array2<f64>,
    ) -> Result<Array2<f64>, TrainError> {
        validate_model_support(model)?;

        let state = model.get_state();
        let input_basis = state.get_basis(model.input_block_idx())?;
        if inputs.ncols() != input_basis.nrows() {
            return Err(TrainError::task_adapter(format!(
                "input dimension mismatch: expected {}, got {}",
                input_basis.nrows(),
                inputs.ncols()
            )));
        }

        let mut backend = CpuBackend::default();
        let proj_buf: Buffer2<f64> = {
            let device = backend.device();
            let inputs_buf = device.upload_2d(inputs);
            let input_basis_buf = device.upload_2d(input_basis);
            device.project_rows(&inputs_buf, &input_basis_buf)
        };

        let interior_device = forward_supported_interior_device(
            &model.interior_block_configs()?,
            &model.interior_block_actions()?,
            model.interior_global_activation()?,
            model.interior_normalize_final()?,
            state,
            &proj_buf,
            &mut backend,
        )?;
        let device = backend.device();
        let embeddings_buf: Buffer2<f64> = match interior_device {
            Some(ref int_dev) => device.clone_buffer(&int_dev.output),
            None => device.clone_buffer(&proj_buf),
        };

        let anchor_bufs: Vec<Buffer2<f64>> = Model::cmr_block_indices(model)
            .iter()
            .map(|&idx| Ok(device.upload_2d(state.get_basis(idx)?)))
            .collect::<Result<Vec<_>, TrainError>>()?;
        let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();
        let (_scores_buf, angles_buf) =
            device.compute_geodesic_scores(&embeddings_buf, &anchor_refs, self.config.eps);
        let raw_angles = device.download_2d(&angles_buf);

        let num_classes = Model::num_classes(model);
        let anchors_per_class = Model::anchors_per_class(model);
        if anchors_per_class == 1 {
            return Ok(raw_angles);
        }

        let batch_size = raw_angles.nrows();
        Ok(Array2::from_shape_fn(
            (batch_size, num_classes),
            |(i, c)| {
                (0..anchors_per_class)
                    .map(|a| raw_angles[[i, c * anchors_per_class + a]])
                    .fold(f64::INFINITY, f64::min)
            },
        ))
    }

    /// Mean geodesic CMR margin over a single-label batch.
    ///
    /// For each sample, `margin = θ_closest_wrong − θ_correct` over the
    /// per-class angles from [`FlatClassificationAdapter::class_angles`];
    /// positive values indicate correct geodesic separation.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] for an empty batch, a label count that does not
    /// match the batch, fewer than two classes, or an out-of-range label.
    pub fn cmr_margin(
        &self,
        model: &CosmosModel,
        inputs: &Array2<f64>,
        labels: &[usize],
    ) -> Result<f64, TrainError> {
        let angles = self.class_angles(model, inputs)?;
        let (batch_size, num_classes) = angles.dim();

        if batch_size == 0 {
            return Err(TrainError::task_adapter(
                "cmr_margin requires a non-empty batch",
            ));
        }
        if labels.len() != batch_size {
            return Err(TrainError::task_adapter(format!(
                "cmr_margin label count mismatch: expected {batch_size}, got {}",
                labels.len()
            )));
        }
        if num_classes < 2 {
            return Err(TrainError::task_adapter(
                "cmr_margin requires at least two classes",
            ));
        }

        let mut total = 0.0;
        for (sample_idx, &label) in labels.iter().enumerate() {
            if label >= num_classes {
                return Err(TrainError::task_adapter(format!(
                    "cmr_margin label {label} out of range for {num_classes} classes"
                )));
            }

            let theta_correct = angles[[sample_idx, label]];
            let theta_closest_wrong = (0..num_classes)
                .filter(|&class_idx| class_idx != label)
                .map(|class_idx| angles[[sample_idx, class_idx]])
                .fold(f64::INFINITY, f64::min);
            total += theta_closest_wrong - theta_correct;
        }

        Ok(total / batch_size as f64)
    }
}

impl<B> TaskAdapter<CosmosModel, B> for FlatClassificationAdapter
where
    B: TrainingBackend,
{
    type Batch = FlatClassificationBatch;

    fn train_batch(
        &mut self,
        model: &CosmosModel,
        batch: &Self::Batch,
        backend: &mut B,
        collect_trace: bool,
    ) -> Result<TrainBatchOutput, TrainError> {
        let forward_start = Instant::now();
        let forward = forward_batch(model, batch, &self.config, backend)?;
        let forward_sec = forward_start.elapsed().as_secs_f64();
        let backward_start = Instant::now();
        let (gradients, backward_timing) =
            backward(backend.device(), model, batch, &forward, &self.config)?;
        let backward_sec = backward_start.elapsed().as_secs_f64();
        let trace_start = Instant::now();
        let trace = build_trace(batch, &forward, collect_trace);
        let trace_sec = trace_start.elapsed().as_secs_f64();

        Ok(TrainBatchOutput {
            metrics: BatchMetrics::new(forward.samples, forward.loss, forward.accuracy),
            gradients,
            trace,
            timing: BatchTimingProfile {
                forward_sec,
                backward_sec,
                backward_anchor_sec: backward_timing.anchor_sec,
                backward_interior_sec: backward_timing.interior_sec,
                backward_input_projector_sec: backward_timing.input_projector_sec,
                trace_sec,
            },
        })
    }

    fn eval_batch(
        &mut self,
        model: &CosmosModel,
        batch: &Self::Batch,
        backend: &mut B,
        collect_trace: bool,
    ) -> Result<EvalBatchOutput, TrainError> {
        let forward_start = Instant::now();
        let forward = forward_batch(model, batch, &self.config, backend)?;
        let forward_sec = forward_start.elapsed().as_secs_f64();
        let trace_start = Instant::now();
        let trace = build_trace(batch, &forward, collect_trace);
        let trace_sec = trace_start.elapsed().as_secs_f64();

        Ok(EvalBatchOutput {
            metrics: BatchMetrics::new(forward.samples, forward.loss, forward.accuracy),
            trace,
            timing: BatchTimingProfile {
                forward_sec,
                backward_sec: 0.0,
                backward_anchor_sec: 0.0,
                backward_interior_sec: 0.0,
                backward_input_projector_sec: 0.0,
                trace_sec,
            },
        })
    }
}
