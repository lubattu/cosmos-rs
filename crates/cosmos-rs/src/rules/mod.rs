//! Rules module — the R component of the COSMOS tuple (S, A, O, R).
//!
//! This module implements gauge-theoretic update rules that evolve the
//! orthonormal bases in [`CosmosState`](crate::state::cosmos_state::CosmosState)
//! via orthogonal conjugation.
//!
//! The core update for each block is:
//! ```text
//! P = W W^T                           (projector)
//! G_tangent = grad - W (W^T grad)     (tangent projection)
//! G_P = G_tangent W^T + W G_tangent^T (lift to projector space)
//! C = [G_P, P]                        (commutator)
//! A = antisym(C)                      (anti-symmetric connection)
//! U = exp(-lr * A)                    (orthogonal transform)
//! W <- U W                            (preserves Stiefel structure)
//! ```

pub mod gauge_dynamics;

use thiserror::Error;

use crate::geometry::grassmann::GrassmannError;
use crate::geometry::matrix_exp::MatrixExpError;
use crate::state::cosmos_state::CosmosStateError;

// ─── Constants ────────────────────────────────────────────────────

/// Default learning rate for gauge dynamics.
pub const DEFAULT_LR: f64 = 0.01;

/// Default momentum coefficient.
pub const DEFAULT_MOMENTUM: f64 = 0.0;

/// Default retraction interval (re-orthonormalize every N steps). 0 disables.
pub const DEFAULT_RETRACTION_INTERVAL: usize = 100;

/// Default EMA beta for commutator target tracking.
pub const DEFAULT_COMM_TARGET_BETA: f64 = 0.9;

/// Default bootstrap steps for commutator target EMA.
pub const DEFAULT_COMM_TARGET_BOOTSTRAP_STEPS: usize = 8;

/// Default power-iteration steps for spectral norm approximation.
pub const DEFAULT_SPECTRAL_NORM_POWER_ITERS: usize = 5;

/// Default deadband for geometric error updates.
pub const DEFAULT_COMM_GEOM_DEADBAND: f64 = 0.0;

/// Small epsilon for numerical stability.
pub const EPS: f64 = 1e-8;

// ─── Errors ───────────────────────────────────────────────────────

/// Errors that can occur in the rules module.
#[derive(Debug, Error)]
pub enum RulesError {
    /// Learning rate must be non-negative.
    #[error("Invalid learning rate: {0} (must be > 0)")]
    InvalidLearningRate(f64),

    /// Momentum must be in [0, 1).
    #[error("Invalid momentum: {0} (must be in [0, 1))")]
    InvalidMomentum(f64),

    /// Momentum max norm must be positive.
    #[error("Invalid momentum max norm: {0} (must be > 0)")]
    InvalidMomentumMaxNorm(f64),

    /// Gradient count doesn't match block count.
    #[error("Block count mismatch: expected {expected}, got {got}")]
    BlockCountMismatch {
        /// Number of blocks the state holds.
        expected: usize,
        /// Number of gradients supplied to the step call.
        got: usize,
    },

    /// Forwarded from matrix exponential computation.
    #[error("Matrix exponential error: {0}")]
    MatrixExp(#[from] MatrixExpError),

    /// Forwarded from Grassmann geometry operations.
    #[error("Grassmann geometry error: {0}")]
    Grassmann(#[from] GrassmannError),

    /// Forwarded from state updates.
    #[error("State update error: {0}")]
    State(#[from] CosmosStateError),

    /// Grassmann block updates require block-local evolution.
    #[error("Grassmann block updates require use_block_updates = true")]
    GrassmannRequiresBlockUpdates,
}

// ─── Re-exports ───────────────────────────────────────────────────

pub use gauge_dynamics::{
    GaugeBlockUpdateMethod, GaugeDynamics, GaugeDynamicsConfig, GaugeDynamicsDiagnostics,
    GaugeDynamicsStepProfile, GaugeDynamicsStepReport,
};

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lr_is_positive() {
        const { assert!(DEFAULT_LR > 0.0) };
    }

    #[test]
    fn default_momentum_is_zero() {
        assert_eq!(DEFAULT_MOMENTUM, 0.0);
    }

    #[test]
    fn default_retraction_interval_is_100() {
        assert_eq!(DEFAULT_RETRACTION_INTERVAL, 100);
    }

    #[test]
    fn default_comm_target_beta_in_range() {
        const { assert!(DEFAULT_COMM_TARGET_BETA > 0.0 && DEFAULT_COMM_TARGET_BETA < 1.0) };
    }

    #[test]
    fn error_display_messages() {
        let e = RulesError::InvalidLearningRate(-0.5);
        assert!(e.to_string().contains("-0.5"));

        let e = RulesError::BlockCountMismatch {
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains("4"));
        assert!(e.to_string().contains("3"));
    }
}
