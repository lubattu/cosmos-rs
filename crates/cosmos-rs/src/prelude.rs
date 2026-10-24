//! Curated re-exports for day-to-day use.
//!
//! ```
//! use cosmos_rs::prelude::*;
//! ```
//!
//! Everything here is reachable by its full module path as well; the prelude
//! only spares the caller the paths.

pub use crate::arrangement::params::SubspaceAction;
pub use crate::checkpoint::{Checkpoint, CheckpointError, restore_state, store_state};
pub use crate::compute::{Activation, ActivationPlacement, ComputeDevice, Device};
pub use crate::metrics::{BaseExperimentResult, EpochMetrics};
pub use crate::models::{
    CosmosModel, CosmosModelConfig, CosmosModelPatchwise, InteriorTopology, Model, ModelError,
    PatchwiseModelConfig, count_state_parameters,
};
pub use crate::objectives::utils::Targets;
pub use crate::rules::gauge_dynamics::{
    GaugeBlockUpdateMethod, GaugeDynamics, GaugeDynamicsConfig,
};
pub use crate::state::cosmos_state::{CosmosState, CosmosStateError};
pub use crate::training::{
    ClassificationLoss, DeviceBackend, EvolutionConfig, EvolutionGovernor,
    FlatClassificationAdapter, FlatClassificationAdapterConfig, FlatClassificationBatch,
    PatchwiseClassificationAdapter, PatchwiseClassificationAdapterConfig,
    PatchwiseClassificationBatch, TaskAdapter, TrainError, TrainingArtifacts,
};
