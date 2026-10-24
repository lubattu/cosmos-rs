//! Geometric utilities for Grassmann manifolds.
//!
//! This module is part of the foundational geometry utilities in COSMOS,
//! supporting S (State), O (Objectives), and R (Rules) components.
//!
//! Provides fundamental geometric operations on the Grassmann manifold Gr(k, n):
//!   - Distance metrics (geodesic, chordal, projection)
//!   - Principal angles between subspaces
//!   - Exponential and logarithmic maps
//!   - Geodesic interpolation
//!   - Retractions (QR, SVD)

mod distances;
mod interpolation;
mod maps;
mod retractions;

pub use distances::*;
pub use interpolation::*;
pub use maps::*;
pub use retractions::*;

use thiserror::Error;

/// Errors that can occur in Grassmann manifold operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GrassmannError {
    /// The two subspace bases have incompatible ambient or subspace dimensions.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension required by the operation.
        expected: usize,
        /// Dimension actually supplied.
        got: usize,
    },

    /// The underlying singular value decomposition did not converge.
    #[error("SVD failed")]
    SvdFailed,
}
