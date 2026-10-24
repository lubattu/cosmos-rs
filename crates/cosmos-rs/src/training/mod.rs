//! Training orchestration for COSMOS models.
//!
//! - [`backend`]: the numerical kernels a task adapter calls.
//! - [`classification`] and [`classification_patchwise`]: the supervised
//!   classification adapters for flat and image inputs.
//! - [`task_adapter`]: the per-task forward, loss and gradient interface.
//! - [`trace`]: forward caches retained for the manual backward pass.
//! - [`trainer`]: epoch and batch orchestration around a model, an adapter and
//!   the gauge dynamics.
//!
//! Backward passes are hand-written, not autograd: an adapter returns one
//! Euclidean gradient per state block and [`crate::rules::GaugeDynamics`] turns
//! those into a coherent move on the manifold.

pub mod backend;
pub mod classification;
pub mod classification_patchwise;
pub mod task_adapter;
pub mod trace;
pub mod trainer;

use thiserror::Error;

use crate::{
    arrangement::ArrangementError, compute::device::Buffer2, models::ModelError,
    objectives::ObjectiveError, rules::RulesError, state::cosmos_state::CosmosStateError,
};

/// Per-block gradient payload produced by task adapters.
///
/// Each entry is `Some(buf)` for a block that received a gradient and `None` for
/// a block that was inactive. The index space is the state's block index space.
pub type BlockGradients = Vec<Option<Buffer2<f64>>>;

/// Errors produced by the training module.
#[derive(Debug, Error)]
pub enum TrainError {
    /// Wrap model and runtime failures.
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Wrap arrangement failures surfaced by task adapters.
    #[error("Arrangement error: {0}")]
    Arrangement(#[from] ArrangementError),

    /// Wrap objective failures surfaced by task adapters.
    #[error("Objective error: {0}")]
    Objective(#[from] ObjectiveError),

    /// Wrap rule and evolution-law failures.
    #[error("Rules error: {0}")]
    Rules(#[from] RulesError),

    /// Wrap state-layer failures.
    #[error("State error: {0}")]
    State(#[from] CosmosStateError),

    /// Invalid training configuration.
    #[error("Invalid training configuration for `{field}`: {reason}")]
    InvalidConfig {
        /// Name of the offending configuration field.
        field: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// Training or validation data is missing.
    #[error("Empty `{split}` split: at least one batch is required")]
    EmptyDataset {
        /// Which split was empty.
        split: &'static str,
    },

    /// Task-adapter level validation failure.
    #[error("Task adapter error: {message}")]
    TaskAdapter {
        /// What the adapter rejected.
        message: String,
    },
}

impl TrainError {
    /// Convenience constructor for adapter-originated validation failures.
    pub fn task_adapter(message: impl Into<String>) -> Self {
        Self::TaskAdapter {
            message: message.into(),
        }
    }
}

pub use backend::{CpuBackend, DeviceBackend, TrainingBackend};
pub use classification::{
    ClassificationLoss, FlatClassificationAdapter, FlatClassificationAdapterConfig,
    FlatClassificationBatch,
};
pub use classification_patchwise::{
    PatchwiseClassificationAdapter, PatchwiseClassificationAdapterConfig,
    PatchwiseClassificationBatch,
};
pub use task_adapter::{
    BatchMetrics, BatchTimingProfile, EvalBatchOutput, TaskAdapter, TrainBatchOutput,
};
pub use trace::{
    BatchTrace, BlockSelectionTrace, BlockTrace, CmrTrace, InputProjectionTrace, InteriorTrace,
};
pub use trainer::{
    AdapterEpochProfile, EpochObserver, EpochRuntimeProfile, EpochTraceCapture, EvolutionConfig,
    EvolutionEpochProfile, EvolutionGovernor, NoopEpochObserver, TrainingArtifacts, TrainingModel,
};
