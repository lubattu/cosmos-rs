//! Task-specific training adapters.
//!
//! Adapters own the task semantics that the generic evolution governor should
//! not know:
//! how to run a batch, how to compute task loss, and how to emit block-aligned
//! gradients for [`crate::rules::GaugeDynamics`].

use crate::metrics::TrainingPhase;

use super::{BlockGradients, TrainError, backend::TrainingBackend, trace::BatchTrace};

/// Lightweight runtime timings captured for one adapter batch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BatchTimingProfile {
    /// Time spent in the adapter forward path.
    pub forward_sec: f64,
    /// Time spent in the adapter backward path.
    pub backward_sec: f64,
    /// Time spent in anchor-score backward kernels.
    pub backward_anchor_sec: f64,
    /// Time spent in interior backward propagation.
    pub backward_interior_sec: f64,
    /// Time spent in the input-projector basis gradient.
    pub backward_input_projector_sec: f64,
    /// Time spent materializing optional traces.
    pub trace_sec: f64,
}

impl BatchTimingProfile {
    /// Total measured adapter time for this batch.
    pub fn total_sec(&self) -> f64 {
        self.forward_sec + self.backward_sec + self.trace_sec
    }
}

/// Batch-level metrics returned by training and evaluation adapters.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchMetrics {
    /// Number of samples represented by the batch.
    pub samples: usize,
    /// Aggregate loss reported to the governor.
    pub loss: f64,
    /// Optional task accuracy in percentage points.
    pub accuracy: Option<f64>,
}

impl BatchMetrics {
    /// Construct batch metrics.
    pub fn new(samples: usize, loss: f64, accuracy: Option<f64>) -> Self {
        Self {
            samples,
            loss,
            accuracy,
        }
    }
}

/// Output of one training batch.
#[derive(Debug)]
pub struct TrainBatchOutput {
    /// Weighted batch metrics.
    pub metrics: BatchMetrics,
    /// Per-block gradients aligned with the model state.
    pub gradients: BlockGradients,
    /// Optional retained forward trace.
    pub trace: Option<BatchTrace>,
    /// Runtime timings captured for this batch.
    pub timing: BatchTimingProfile,
}

/// Output of one evaluation batch.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalBatchOutput {
    /// Weighted batch metrics.
    pub metrics: BatchMetrics,
    /// Optional retained forward trace.
    pub trace: Option<BatchTrace>,
    /// Runtime timings captured for this batch.
    pub timing: BatchTimingProfile,
}

/// Task-facing batch interface consumed by
/// [`crate::training::trainer::EvolutionGovernor`].
pub trait TaskAdapter<M, B>
where
    B: TrainingBackend,
{
    /// Batch type owned by the caller.
    type Batch;

    /// Logical training phase associated with this adapter.
    fn training_phase(&self) -> TrainingPhase {
        TrainingPhase::Standard
    }

    /// Execute one training batch and emit explicit block gradients.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the batch is malformed for this adapter or
    /// when the model, arrangement or objective layers fail on it.
    fn train_batch(
        &mut self,
        model: &M,
        batch: &Self::Batch,
        backend: &mut B,
        collect_trace: bool,
    ) -> Result<TrainBatchOutput, TrainError>;

    /// Execute one evaluation batch without gradients.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the batch is malformed for this adapter or
    /// when the model, arrangement or objective layers fail on it.
    fn eval_batch(
        &mut self,
        model: &M,
        batch: &Self::Batch,
        backend: &mut B,
        collect_trace: bool,
    ) -> Result<EvalBatchOutput, TrainError>;
}
