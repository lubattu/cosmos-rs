//! Arrangement: the A of the COSMOS tuple `(S, A, O, R)`.
//!
//! The arrangement decides how the state's projector blocks are selected,
//! ordered, composed and aggregated on the way from input to head. It owns no
//! trainable tensor of its own — SAOR contract 2, "the Arrangement is
//! gradient-free". Everything it reads comes from
//! [`CosmosState`](crate::state::cosmos_state::CosmosState); everything it
//! produces is an activation, never a parameter.
//!
//! - [`params`]: typed parameter structs for the routing primitives, plus
//!   [`params::SubspaceAction`], the projection/reflection action `alpha I + beta P`.
//! - [`selections`], [`orderings`], [`compositions`], [`aggregations`]: the four
//!   routing primitives, in that pipeline order.
//! - [`activations`]: nonlinearities and the projection helpers that place them.
//! - [`blocks`]: [`blocks::ModularBlock`], the Select → Order → Compose →
//!   Aggregate runtime.
//! - [`interior`]: sequential dispatch over several modular blocks.
//! - [`builder`]: a serializable [`builder::InteriorSpec`] and the fluent
//!   builders that produce one.
//! - [`input`]: flat and patchwise input projectors.
//! - [`heads`]: the geometric CMR head family.
//! - [`utils`]: block access, alignment matrices and projection energies.

pub mod activations;
pub mod aggregations;
pub mod blocks;
pub mod builder;
pub mod compositions;
pub mod heads;
pub mod input;
pub mod interior;
pub mod orderings;
pub mod params;
pub mod selections;
pub mod utils;

use thiserror::Error;

/// Errors that can occur in the arrangement module.
#[derive(Debug, Error)]
pub enum ArrangementError {
    /// Unknown strategy name for a primitive.
    #[error("Unknown {primitive} strategy: \"{name}\"")]
    UnknownStrategy {
        /// Routing primitive being configured, for example `"selection"`.
        primitive: &'static str,
        /// The unrecognized strategy name, as supplied.
        name: String,
    },

    /// A required context field was `None` when a strategy needed it.
    #[error("{strategy} requires `{field}` in context, but it was None")]
    MissingContext {
        /// Strategy that needed the missing field.
        strategy: &'static str,
        /// Name of the context field that was `None`.
        field: &'static str,
    },

    /// Shape mismatch in input arrays.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Shape the operation required.
        expected: String,
        /// Shape that was supplied.
        got: String,
    },

    /// Parameter validation failure.
    #[error("Invalid parameter `{param}`: {reason}")]
    InvalidParameter {
        /// Name of the rejected parameter.
        param: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// Not enough elements for the requested operation.
    #[error("{operation}: need at least {needed}, got {got}")]
    InsufficientElements {
        /// Operation that required the elements.
        operation: &'static str,
        /// Minimum number of elements required.
        needed: usize,
        /// Number of elements supplied.
        got: usize,
    },
}
