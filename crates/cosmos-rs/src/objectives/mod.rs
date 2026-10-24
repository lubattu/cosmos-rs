//! Objectives: the O of the COSMOS tuple `(S, A, O, R)`.
//!
//! Every function here evaluates a configuration and returns a value. None of
//! them mutates a [`CosmosState`](crate::state::cosmos_state::CosmosState) or any
//! other input — SAOR contract 3, "objectives evaluate; they do not act".
//! Gradients are returned as owned arrays for the caller to apply; applying them
//! is the business of [`crate::rules`].
//!
//! - [`cmr_losses`]: the Class-Membership-Region family — direct membership,
//!   geodesic membership, and the geodesic angular hinge that is the paper's
//!   canonical loss.
//! - [`utils`]: target encoding.
//! - [`constants`]: named numerical defaults shared across the losses.

pub mod cmr_losses;
pub mod constants;
pub mod utils;

use ndarray::Array1;
use thiserror::Error;

// Re-export key types for convenience
pub use constants::*;
pub use utils::{Targets, targets_to_probabilities};

/// Errors that can occur during objective computation.
#[derive(Debug, Error)]
pub enum ObjectiveError {
    /// Batch sizes of predictions and targets don't match.
    #[error("Batch size mismatch: predictions have {predictions}, targets have {targets}")]
    BatchSizeMismatch {
        /// Number of predictions in the batch.
        predictions: usize,
        /// Number of targets in the batch.
        targets: usize,
    },

    /// Target shape is incompatible with num_classes.
    #[error("Multi-label targets shape mismatch: expected (batch, {expected}), got (batch, {got})")]
    TargetShapeMismatch {
        /// Expected target width, equal to `num_classes`.
        expected: usize,
        /// Target width actually observed.
        got: usize,
    },

    /// Margin must be non-negative.
    #[error("Margin must be >= 0, got {0}")]
    InvalidMargin(f64),

    /// Region metrics are required but missing.
    #[error("Region metrics required for geodesic distance mode but not provided")]
    MissingRegionMetrics,

    /// Region metrics count doesn't match num_classes.
    #[error("Expected {expected} region metrics, got {got}")]
    MetricsCountMismatch {
        /// Expected number of region metrics, equal to `num_classes`.
        expected: usize,
        /// Number of region metrics actually provided.
        got: usize,
    },
}

/// Reduction method for aggregating per-sample losses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Average loss over the batch.
    Mean,
    /// Sum loss over the batch.
    Sum,
    /// Return per-sample losses without aggregation.
    None,
}

impl Reduction {
    /// Apply this reduction to a per-sample loss vector.
    ///
    /// # Arguments
    ///
    /// * `losses` - Per-sample losses of shape `(batch,)`.
    ///
    /// # Returns
    ///
    /// - `Mean`: scalar = `losses.mean()`
    /// - `Sum`: scalar = `losses.sum()`
    /// - `None`: returns `losses` unchanged
    pub fn apply(&self, losses: &Array1<f64>) -> Array1<f64> {
        match self {
            Reduction::Mean => {
                let n = losses.len() as f64;
                if n == 0.0 {
                    return Array1::zeros(1);
                }
                Array1::from_elem(1, losses.sum() / n)
            }
            Reduction::Sum => Array1::from_elem(1, losses.sum()),
            Reduction::None => losses.clone(),
        }
    }
}

impl std::fmt::Display for Reduction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reduction::Mean => write!(f, "mean"),
            Reduction::Sum => write!(f, "sum"),
            Reduction::None => write!(f, "none"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Reduction::apply
    // ========================================================================

    #[test]
    fn reduction_mean_basic() {
        let losses = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let result = Reduction::Mean.apply(&losses);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn reduction_sum_basic() {
        let losses = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = Reduction::Sum.apply(&losses);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn reduction_none_preserves_all() {
        let losses = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = Reduction::None.apply(&losses);
        assert_eq!(result.len(), 3);
        assert_eq!(result, losses);
    }

    #[test]
    fn reduction_mean_empty_returns_zero() {
        let losses = Array1::<f64>::zeros(0);
        let result = Reduction::Mean.apply(&losses);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn reduction_sum_single_element() {
        let losses = Array1::from_vec(vec![42.0]);
        let result = Reduction::Sum.apply(&losses);
        assert!((result[0] - 42.0).abs() < 1e-15);
    }

    // ========================================================================
    // Reduction Display
    // ========================================================================

    #[test]
    fn reduction_display() {
        assert_eq!(format!("{}", Reduction::Mean), "mean");
        assert_eq!(format!("{}", Reduction::Sum), "sum");
        assert_eq!(format!("{}", Reduction::None), "none");
    }

    // ========================================================================
    // ObjectiveError Display
    // ========================================================================

    #[test]
    fn error_batch_mismatch_message() {
        let err = ObjectiveError::BatchSizeMismatch {
            predictions: 10,
            targets: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_invalid_margin_message() {
        let err = ObjectiveError::InvalidMargin(-0.5);
        assert!(format!("{err}").contains("-0.5"));
    }

    #[test]
    fn error_missing_region_metrics_message() {
        let err = ObjectiveError::MissingRegionMetrics;
        assert!(format!("{err}").contains("Region metrics"));
    }
}
