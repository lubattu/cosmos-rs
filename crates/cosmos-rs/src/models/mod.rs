//! Model composition over State, Arrangement and Objectives.
//!
//! A model wires an input projector, an interior of modular blocks and a CMR
//! head onto the block layout of one
//! [`CosmosState`](crate::state::cosmos_state::CosmosState). Two variants ship:
//! [`CosmosModel`] for flat feature vectors and [`CosmosModelPatchwise`] for
//! images. Both implement [`Model`], the shared contract.

mod common;

pub mod base;
pub mod cosmos_model;
pub mod patchwise_model;

pub use base::{
    Model, ModelError, compute_cmr_scores_device, compute_cmr_scores_from_state,
    compute_cmr_scores_with_metrics_from_state, count_state_parameters,
};
pub use common::{InteriorTopology, ModelHead};
pub use cosmos_model::{CosmosModel, CosmosModelConfig};
pub use patchwise_model::{CosmosModelPatchwise, PatchwiseModelConfig};
