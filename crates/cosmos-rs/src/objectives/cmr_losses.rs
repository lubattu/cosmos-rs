//! CMR (Class Manifold Region) loss functions.
//!
//! These losses implement the Objectives (O) component for Class Manifold
//! Regions, where classes correspond to regions on the Grassmann manifold
//! rather than fixed Euclidean prototypes.
//!
//! # Loss modes
//!
//! - [`CmrMode::DirectMembership`]: supervised loss on membership scores
//!   (cross-entropy for single-label, squared error for multi-label).
//! - [`CmrMode::GeodesicDistance`]: hinge-style loss on Grassmann geodesic
//!   distances from principal angles.
//!
//! # Examples
//! ```
//! use ndarray::array;
//! use cosmos_rs::objectives::Reduction;
//! use cosmos_rs::objectives::utils::Targets;
//! use cosmos_rs::objectives::cmr_losses::{cmr_region_loss, CmrMode};
//!
//! let scores = array![[0.9, 0.1], [0.2, 0.8]]; // (2 samples, 2 classes)
//! let targets = Targets::SingleLabel(vec![0, 1]);
//! let loss = cmr_region_loss(&scores, &targets, CmrMode::DirectMembership,
//!                            0.0, Reduction::Mean, None).unwrap();
//! ```

use ndarray::{Array1, Array2, Axis};

use super::constants::DEFAULT_MARGIN;
use super::utils::{Targets, targets_to_probabilities};
use super::{ObjectiveError, Reduction};

/// Pre-computed region metrics for one class from a CMR head.
///
/// Each entry holds the principal angles between the predicted subspace
/// and the class prototype subspace for all samples in the batch.
#[derive(Debug, Clone)]
pub struct RegionMetrics {
    /// Principal angles θ between predicted and prototype subspaces.
    /// Shape: `(batch, k)` where k is the subspace dimension.
    pub principal_angles: Array2<f64>,
}

/// Loss computation mode for CMR losses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmrMode {
    /// Supervised loss on membership scores.
    /// Single-label: cross-entropy (log-softmax + NLL).
    /// Multi-label: squared error.
    DirectMembership,
    /// Hinge-style loss on Grassmann geodesic distances.
    /// Requires `region_metrics` to be provided.
    GeodesicDistance,
}

// ============================================================================
// Public API
// ============================================================================

/// Compute geometric CMR loss with configurable modes.
///
/// This is the main entry point for CMR losses. The `mode` parameter
/// selects between membership-based and geodesic distance-based loss.
///
/// # Arguments
///
/// * `scores` - Membership scores, shape `(batch, num_classes)`.
/// * `targets` - Class labels (single-label or multi-label).
/// * `mode` - Loss computation mode.
/// * `margin` - Separation margin for inactive classes (radians).
///   Only used with [`CmrMode::GeodesicDistance`].
/// * `reduction` - How to aggregate per-sample losses.
/// * `region_metrics`- Per-class region metrics from a CMR head.
///   Required for [`CmrMode::GeodesicDistance`], ignored for
///   [`CmrMode::DirectMembership`].
///
/// # Errors
///
/// - [`ObjectiveError::BatchSizeMismatch`] if batch dims don't match.
/// - [`ObjectiveError::MissingRegionMetrics`] if geodesic mode but no metrics.
/// - [`ObjectiveError::InvalidMargin`] if margin < 0.
pub fn cmr_region_loss(
    scores: &Array2<f64>,
    targets: &Targets,
    mode: CmrMode,
    margin: f64,
    reduction: Reduction,
    region_metrics: Option<&[RegionMetrics]>,
) -> Result<Array1<f64>, ObjectiveError> {
    match mode {
        CmrMode::DirectMembership => direct_membership_loss(scores, targets, reduction),
        CmrMode::GeodesicDistance => {
            let metrics = region_metrics.ok_or(ObjectiveError::MissingRegionMetrics)?;
            geodesic_distance_loss(scores, targets, metrics, margin, reduction)
        }
    }
}

/// Convenience: geodesic CMR loss from scores + pre-computed metrics.
///
/// Equivalent to [`cmr_region_loss`] with [`CmrMode::GeodesicDistance`]
/// and default margin [`DEFAULT_MARGIN`] (π/4).
///
/// # Arguments
///
/// * `scores` - Membership scores, shape `(batch, num_classes)`.
/// * `region_metrics` - Per-class metrics with principal angles.
/// * `targets` - Class labels.
/// * `reduction` - Reduction method.
///
/// # Errors
///
/// - [`ObjectiveError::MetricsCountMismatch`] if `region_metrics` has the wrong length.
/// - [`ObjectiveError::TargetShapeMismatch`] for malformed multi-label targets.
/// - [`ObjectiveError::BatchSizeMismatch`] if a metric's row count disagrees with `scores`.
pub fn geodesic_cmr_loss(
    scores: &Array2<f64>,
    region_metrics: &[RegionMetrics],
    targets: &Targets,
    reduction: Reduction,
) -> Result<Array1<f64>, ObjectiveError> {
    geodesic_distance_loss(scores, targets, region_metrics, DEFAULT_MARGIN, reduction)
}

/// Convenience: direct membership loss on scores.
///
/// Equivalent to [`cmr_region_loss`] with [`CmrMode::DirectMembership`].
///
/// # Errors
///
/// - [`ObjectiveError::BatchSizeMismatch`] if `scores` and `targets` disagree on batch size.
/// - [`ObjectiveError::TargetShapeMismatch`] for malformed multi-label targets.
pub fn direct_cmr_loss(
    scores: &Array2<f64>,
    targets: &Targets,
    reduction: Reduction,
) -> Result<Array1<f64>, ObjectiveError> {
    direct_membership_loss(scores, targets, reduction)
}

// ============================================================================
// Private: Direct Membership Loss
// ============================================================================

/// Cross-entropy (single-label) or squared error (multi-label) on scores.
fn direct_membership_loss(
    scores: &Array2<f64>,
    targets: &Targets,
    reduction: Reduction,
) -> Result<Array1<f64>, ObjectiveError> {
    let batch_size = scores.nrows();
    let num_classes = scores.ncols();

    let per_sample = match targets {
        Targets::SingleLabel(indices) => {
            // Cross-entropy: -log_softmax(scores)[target_class]
            if indices.len() != batch_size {
                return Err(ObjectiveError::BatchSizeMismatch {
                    predictions: batch_size,
                    targets: indices.len(),
                });
            }
            let log_probs = log_softmax(scores);
            let mut losses = Array1::zeros(batch_size);
            for (i, &class_idx) in indices.iter().enumerate() {
                losses[i] = -log_probs[[i, class_idx]];
            }
            losses
        }
        Targets::MultiLabel(target_probs) => {
            // Squared error: mean over classes of (score - target)²
            if target_probs.nrows() != batch_size {
                return Err(ObjectiveError::BatchSizeMismatch {
                    predictions: batch_size,
                    targets: target_probs.nrows(),
                });
            }
            if target_probs.ncols() != num_classes {
                return Err(ObjectiveError::TargetShapeMismatch {
                    expected: num_classes,
                    got: target_probs.ncols(),
                });
            }
            let diff = scores - target_probs;
            let sq = &diff * &diff;
            // Per-sample: sum of squared errors across classes
            sq.sum_axis(Axis(1))
        }
    };

    Ok(reduction.apply(&per_sample))
}

// ============================================================================
// Private: Geodesic Distance Loss
// ============================================================================

/// Hinge-style loss on Grassmann geodesic distances from principal angles.
///
/// Active classes are pulled (minimize d²), inactive classes are pushed
/// away with a hinge margin: max(0, margin - d)².
fn geodesic_distance_loss(
    scores: &Array2<f64>,
    targets: &Targets,
    region_metrics: &[RegionMetrics],
    margin: f64,
    reduction: Reduction,
) -> Result<Array1<f64>, ObjectiveError> {
    if margin < 0.0 {
        return Err(ObjectiveError::InvalidMargin(margin));
    }

    let batch_size = scores.nrows();
    let num_classes = scores.ncols();

    // Validate region metrics count
    if region_metrics.len() != num_classes {
        return Err(ObjectiveError::MetricsCountMismatch {
            expected: num_classes,
            got: region_metrics.len(),
        });
    }

    // Convert targets to probability matrix for active/inactive masking
    let target_probs = targets_to_probabilities(targets, num_classes)?;

    // Validate batch sizes in metrics
    for rm in region_metrics {
        if rm.principal_angles.nrows() != batch_size {
            return Err(ObjectiveError::BatchSizeMismatch {
                predictions: batch_size,
                targets: rm.principal_angles.nrows(),
            });
        }
    }

    // Compute geodesic distance per (sample, class):
    //   d(sample, class) = ||θ||₂ = sqrt(Σ θᵢ²)
    // where θ are the principal angles for that (sample, class) pair.
    let mut distances = Array2::zeros((batch_size, num_classes));
    for (c, rm) in region_metrics.iter().enumerate() {
        for i in 0..batch_size {
            let angles = rm.principal_angles.row(i);
            let d_sq: f64 = angles.dot(&angles);
            distances[[i, c]] = d_sq.sqrt();
        }
    }

    // Per-sample loss: active pulls + inactive pushes
    let mut per_sample = Array1::zeros(batch_size);
    for i in 0..batch_size {
        let mut loss = 0.0;
        for c in 0..num_classes {
            let d = distances[[i, c]];
            let is_active = target_probs[[i, c]] > 0.5;
            if is_active {
                // Pull: minimize d²
                loss += d * d;
            } else {
                // Push: max(0, margin - d)²
                let gap = (margin - d).max(0.0);
                loss += gap * gap;
            }
        }
        per_sample[i] = loss;
    }

    Ok(reduction.apply(&per_sample))
}

// ============================================================================
// Geodesic hinge — direct-distance interface (additive)
// ============================================================================
//
// The functions below take per-(sample, class) geodesic distances `d`
// already aggregated from principal angles, sparing callers the
// `RegionMetrics` plumbing when they have `d` available cheaply (e.g.,
// a backend that returns distances directly, or downstream layers
// that have already pre-aggregated).
//
// These complement — they do not replace — [`geodesic_cmr_loss`] and
// [`cmr_region_loss`] above, which start from `region_metrics` and
// compute `d = ‖θ‖₂` internally. The two paths give numerically
// identical results when the inputs match.

/// Closed-form gradient of the geodesic margin-hinge loss with respect
/// to per-(sample, class) geodesic distance.
///
/// Returned by [`geodesic_cmr_loss_grad`]. The `1 / (B · C)` mean
/// reduction is already folded in, where `B = batch` and
/// `C = num_classes`.
#[derive(Debug, Clone)]
pub struct GeodesicHingeGrad {
    /// Partial derivative `∂L / ∂d_{b,c}` of the geodesic margin-hinge
    /// loss with respect to the per-(sample, class) distance, with
    /// `1 / (B · C)` reduction baked in. Shape `(batch, num_classes)`.
    pub dloss_dtheta: Array2<f64>,
}

/// Geodesic margin-hinge loss starting from pre-aggregated per-class
/// distances.
///
/// Equivalent to [`geodesic_cmr_loss`] / [`cmr_region_loss`] with mode
/// [`CmrMode::GeodesicDistance`], but accepts the per-(sample, class)
/// geodesic distance `d` directly instead of recomputing it from
/// principal angles. Useful when the upstream layer already has `d`
/// on hand.
///
/// For each sample `b` and class `c`:
/// - **Active** (`target_prob > 0.5`): pull term `d²`.
/// - **Inactive**: push term `max(0, m − d)²`.
///
/// # Arguments
///
/// * `distances` — geodesic distances `d`, shape `(batch, num_classes)`.
/// * `targets` — class labels (single-label or multi-label).
/// * `margin` — hinge margin in radians; must be `≥ 0`.
/// * `reduction` — aggregation across the batch.
///
/// # Errors
///
/// - [`ObjectiveError::InvalidMargin`] if `margin < 0`.
/// - [`ObjectiveError::BatchSizeMismatch`] if `targets` batch size
///   disagrees with `distances`.
/// - [`ObjectiveError::TargetShapeMismatch`] for multi-label `targets`
///   whose column count differs from `num_classes`.
pub fn geodesic_cmr_hinge_loss(
    distances: &Array2<f64>,
    targets: &Targets,
    margin: f64,
    reduction: Reduction,
) -> Result<Array1<f64>, ObjectiveError> {
    if margin < 0.0 {
        return Err(ObjectiveError::InvalidMargin(margin));
    }

    let batch = distances.nrows();
    let num_classes = distances.ncols();

    let target_probs = targets_to_probabilities(targets, num_classes)?;

    if target_probs.nrows() != batch {
        return Err(ObjectiveError::BatchSizeMismatch {
            predictions: batch,
            targets: target_probs.nrows(),
        });
    }

    let per_sample = Array1::from_shape_fn(batch, |i| {
        let mut loss = 0.0;
        for c in 0..num_classes {
            let d = distances[[i, c]];
            if target_probs[[i, c]] > 0.5 {
                loss += d * d;
            } else {
                let gap = margin - d;
                if gap > 0.0 {
                    loss += gap * gap;
                }
            }
        }
        loss
    });

    Ok(reduction.apply(&per_sample))
}

/// Closed-form gradient of the geodesic margin-hinge loss.
///
/// Returns `∂L / ∂d_{b,c}` with `1 / (B · C)` mean reduction:
///
/// - **Active**: `2d / (B · C)`
/// - **Inactive**, `d < m`: `−2 (m − d) / (B · C)`
/// - **Inactive**, `d ≥ m`: `0`
///
/// The reduction matches palace's convention and is validated by
/// finite-difference tests (`test_geodesic_cmr_loss_grad_finite_diff`).
///
/// # Arguments
///
/// * `distances` — geodesic distances `d`, shape `(batch, num_classes)`.
/// * `targets` — class labels (single-label or multi-label).
/// * `margin` — hinge margin in radians; must be `≥ 0`.
///
/// # Errors
///
/// - [`ObjectiveError::InvalidMargin`] if `margin < 0`.
/// - [`ObjectiveError::BatchSizeMismatch`] if `targets` batch size
///   disagrees with `distances`.
/// - [`ObjectiveError::TargetShapeMismatch`] for multi-label `targets`
///   whose column count differs from `num_classes`.
pub fn geodesic_cmr_loss_grad(
    distances: &Array2<f64>,
    targets: &Targets,
    margin: f64,
) -> Result<GeodesicHingeGrad, ObjectiveError> {
    if margin < 0.0 {
        return Err(ObjectiveError::InvalidMargin(margin));
    }

    let batch = distances.nrows();
    let num_classes = distances.ncols();

    let target_probs = targets_to_probabilities(targets, num_classes)?;

    if target_probs.nrows() != batch {
        return Err(ObjectiveError::BatchSizeMismatch {
            predictions: batch,
            targets: target_probs.nrows(),
        });
    }

    let norm = 1.0 / (batch as f64 * num_classes as f64);
    let mut dloss_dtheta = Array2::zeros((batch, num_classes));

    for i in 0..batch {
        for c in 0..num_classes {
            let d = distances[[i, c]];
            if target_probs[[i, c]] > 0.5 {
                // Active: d(d²)/dd = 2d.
                dloss_dtheta[[i, c]] = 2.0 * d * norm;
            } else {
                // Inactive: d(max(0, m − d)²)/dd = −2 (m − d) if d < m.
                let gap = margin - d;
                if gap > 0.0 {
                    dloss_dtheta[[i, c]] = -2.0 * gap * norm;
                }
                // else: zero (already initialised).
            }
        }
    }

    Ok(GeodesicHingeGrad { dloss_dtheta })
}

// ============================================================================
// Numeric helpers
// ============================================================================

/// Numerically stable log-softmax over columns (classes) for each row (sample).
///
/// log_softmax(x_i) = x_i - max(x) - log(Σ exp(x_j - max(x)))
///
/// Returns array of same shape as input.
fn log_softmax(scores: &Array2<f64>) -> Array2<f64> {
    let batch_size = scores.nrows();
    let num_classes = scores.ncols();
    let mut result = Array2::zeros((batch_size, num_classes));

    for i in 0..batch_size {
        let row = scores.row(i);

        // Find max for numerical stability
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute log(Σ exp(x_j - max))
        let log_sum_exp: f64 = row.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();

        // log_softmax = x_i - max - log_sum_exp
        for j in 0..num_classes {
            result[[i, j]] = scores[[i, j]] - max_val - log_sum_exp;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOL: f64 = 1e-10;

    // ========================================================================
    // log_softmax
    // ========================================================================

    #[test]
    fn log_softmax_sums_to_one_in_probability_space() {
        let scores = array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]];
        let log_probs = log_softmax(&scores);
        // exp(log_softmax) should sum to 1 per row
        for i in 0..2 {
            let row_sum: f64 = log_probs.row(i).iter().map(|&x| x.exp()).sum();
            assert!((row_sum - 1.0).abs() < TOL, "row {i}: sum = {row_sum}");
        }
    }

    #[test]
    fn log_softmax_uniform_scores() {
        let scores = array![[1.0, 1.0, 1.0]];
        let log_probs = log_softmax(&scores);
        let expected = -(3.0_f64).ln(); // log(1/3)
        for j in 0..3 {
            assert!((log_probs[[0, j]] - expected).abs() < TOL);
        }
    }

    #[test]
    fn log_softmax_large_values_numerically_stable() {
        let scores = array![[1000.0, 1001.0, 999.0]];
        let log_probs = log_softmax(&scores);
        // Should not overflow — all values should be finite
        for j in 0..3 {
            assert!(
                log_probs[[0, j]].is_finite(),
                "j={j}: {}",
                log_probs[[0, j]]
            );
        }
        // Max score should have highest log-prob
        assert!(log_probs[[0, 1]] > log_probs[[0, 0]]);
        assert!(log_probs[[0, 0]] > log_probs[[0, 2]]);
    }

    #[test]
    fn log_softmax_values_are_non_positive() {
        let scores = array![[5.0, 3.0, 1.0]];
        let log_probs = log_softmax(&scores);
        for j in 0..3 {
            assert!(log_probs[[0, j]] <= 0.0 + TOL);
        }
    }

    // ========================================================================
    // direct_membership_loss — single-label (cross-entropy)
    // ========================================================================

    #[test]
    fn direct_loss_single_label_perfect_scores() {
        // High score for correct class → low loss
        let scores = array![[10.0, 0.0], [0.0, 10.0]];
        let targets = Targets::SingleLabel(vec![0, 1]);
        let loss = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap();
        assert!(loss[0] < 0.001); // near zero
    }

    #[test]
    fn direct_loss_single_label_wrong_scores() {
        // High score for wrong class → high loss
        let scores = array![[0.0, 10.0], [10.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0, 1]);
        let loss = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap();
        assert!(loss[0] > 5.0); // large loss
    }

    #[test]
    fn direct_loss_single_label_uniform_scores() {
        // Uniform scores → cross-entropy = log(num_classes)
        let scores = array![[0.0, 0.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let loss = direct_cmr_loss(&scores, &targets, Reduction::None).unwrap();
        let expected = (3.0_f64).ln();
        assert!((loss[0] - expected).abs() < TOL);
    }

    #[test]
    fn direct_loss_single_label_batch_mismatch() {
        let scores = array![[1.0, 2.0]];
        let targets = Targets::SingleLabel(vec![0, 1]); // 2 targets, 1 score
        let err = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap_err();
        assert!(format!("{err}").contains("mismatch"));
    }

    // ========================================================================
    // direct_membership_loss — multi-label (squared error)
    // ========================================================================

    #[test]
    fn direct_loss_multi_label_perfect() {
        let scores = array![[1.0, 0.0], [0.0, 1.0]];
        let target_probs = array![[1.0, 0.0], [0.0, 1.0]];
        let targets = Targets::MultiLabel(target_probs);
        let loss = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap();
        assert!(loss[0].abs() < TOL);
    }

    #[test]
    fn direct_loss_multi_label_known_value() {
        // diff = [[0.1, -0.2]], sq = [[0.01, 0.04]], sum per sample = 0.05
        let scores = array![[0.6, 0.3]];
        let target_probs = array![[0.5, 0.5]];
        let targets = Targets::MultiLabel(target_probs);
        let loss = direct_cmr_loss(&scores, &targets, Reduction::None).unwrap();
        assert!((loss[0] - 0.05).abs() < TOL);
    }

    #[test]
    fn direct_loss_multi_label_shape_mismatch() {
        let scores = array![[1.0, 2.0, 3.0]];
        let target_probs = array![[1.0, 0.0]]; // 2 cols vs 3 cols
        let targets = Targets::MultiLabel(target_probs);
        let err = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap_err();
        assert!(format!("{err}").contains("expected"));
    }

    // ========================================================================
    // geodesic_distance_loss
    // ========================================================================

    fn make_metrics(angles: &[Array2<f64>]) -> Vec<RegionMetrics> {
        angles
            .iter()
            .map(|a| RegionMetrics {
                principal_angles: a.clone(),
            })
            .collect()
    }

    #[test]
    fn geodesic_loss_zero_distance_active_class() {
        // Active class with zero angles → zero loss
        let scores = array![[1.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let angles_class0 = array![[0.0, 0.0]]; // zero distance
        let angles_class1 = array![[0.5, 0.5]]; // some distance
        let metrics = make_metrics(&[angles_class0, angles_class1]);
        let loss = geodesic_cmr_loss(&scores, &metrics, &targets, Reduction::None).unwrap();
        // Active pull: 0², Inactive push: max(0, margin - d)²
        let d1 = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt(); // ||[0.5, 0.5]||
        let margin = DEFAULT_MARGIN;
        let expected = (margin - d1).max(0.0).powi(2);
        assert!((loss[0] - expected).abs() < TOL);
    }

    #[test]
    fn geodesic_loss_large_distance_active_class() {
        // Active class far away → large pull loss
        let scores = array![[1.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let angles_class0 = array![[1.0, 1.0]]; // d = sqrt(2)
        let angles_class1 = array![[0.0, 0.0]]; // d = 0
        let metrics = make_metrics(&[angles_class0, angles_class1]);
        let loss = geodesic_cmr_loss(&scores, &metrics, &targets, Reduction::None).unwrap();
        // Active pull: d² = 2.0, Inactive push: max(0, margin - 0)² = margin²
        let expected = 2.0 + DEFAULT_MARGIN.powi(2);
        assert!((loss[0] - expected).abs() < TOL);
    }

    #[test]
    fn geodesic_loss_inactive_beyond_margin_is_zero() {
        // Inactive class with distance > margin → no push penalty
        let scores = array![[1.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let angles_class0 = array![[0.0]]; // active, d = 0
        let angles_class1 = array![[2.0]]; // inactive, d = 2 > margin
        let metrics = make_metrics(&[angles_class0, angles_class1]);
        let loss = geodesic_cmr_loss(&scores, &metrics, &targets, Reduction::None).unwrap();
        // Active: 0, Inactive: max(0, π/4 - 2)² = 0 (d > margin)
        assert!(loss[0].abs() < TOL);
    }

    #[test]
    fn geodesic_loss_missing_metrics_error() {
        let scores = array![[1.0, 2.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let err = cmr_region_loss(
            &scores,
            &targets,
            CmrMode::GeodesicDistance,
            0.5,
            Reduction::Mean,
            None,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("Region metrics"));
    }

    #[test]
    fn geodesic_loss_metrics_count_mismatch() {
        let scores = array![[1.0, 2.0, 3.0]]; // 3 classes
        let targets = Targets::SingleLabel(vec![0]);
        let metrics = make_metrics(&[array![[0.1]]]); // only 1 metric
        let err = geodesic_cmr_loss(&scores, &metrics, &targets, Reduction::Mean).unwrap_err();
        assert!(format!("{err}").contains("Expected"));
    }

    #[test]
    fn geodesic_loss_negative_margin_error() {
        let scores = array![[1.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let metrics = make_metrics(&[array![[0.1]]]);
        let err = cmr_region_loss(
            &scores,
            &targets,
            CmrMode::GeodesicDistance,
            -0.1,
            Reduction::Mean,
            Some(&metrics),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("Margin"));
    }

    // ========================================================================
    // cmr_region_loss dispatch
    // ========================================================================

    #[test]
    fn cmr_dispatch_direct_membership() {
        let scores = array![[10.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let loss = cmr_region_loss(
            &scores,
            &targets,
            CmrMode::DirectMembership,
            0.0,
            Reduction::Mean,
            None,
        )
        .unwrap();
        assert!(loss[0] < 0.001);
    }

    #[test]
    fn cmr_dispatch_geodesic_distance() {
        let scores = array![[1.0, 0.0]];
        let targets = Targets::SingleLabel(vec![0]);
        let metrics = make_metrics(&[array![[0.0]], array![[1.0]]]);
        let loss = cmr_region_loss(
            &scores,
            &targets,
            CmrMode::GeodesicDistance,
            0.5,
            Reduction::Mean,
            Some(&metrics),
        )
        .unwrap();
        assert!(loss[0].is_finite());
    }

    // ========================================================================
    // Reduction integration
    // ========================================================================

    #[test]
    fn direct_loss_reduction_consistency() {
        let scores = array![[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]];
        let targets = Targets::SingleLabel(vec![0, 1, 0]);

        let none = direct_cmr_loss(&scores, &targets, Reduction::None).unwrap();
        let mean = direct_cmr_loss(&scores, &targets, Reduction::Mean).unwrap();
        let sum = direct_cmr_loss(&scores, &targets, Reduction::Sum).unwrap();

        assert_eq!(none.len(), 3);
        assert!((sum[0] - none.sum()).abs() < TOL);
        assert!((mean[0] - none.sum() / 3.0).abs() < TOL);
    }

    // ========================================================================
    // Geodesic hinge — direct-distance interface
    // ========================================================================

    /// Active class with zero distance → zero pull loss.
    #[test]
    fn geodesic_hinge_zero_distance_active() {
        let distances = Array2::from_shape_vec((1, 2), vec![0.0, 1.5]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::None).unwrap();
        // Class 0 active, d=0 → pull = 0. Class 1 inactive, d=1.5 > 1.0 → push = 0.
        assert!((result[0] - 0.0).abs() < TOL);
    }

    /// Active class far away → pull loss = d².
    #[test]
    fn geodesic_hinge_large_distance_active() {
        let distances = Array2::from_shape_vec((1, 1), vec![1.414]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::None).unwrap();
        assert!((result[0] - 1.414 * 1.414).abs() < 1e-6);
    }

    /// Inactive class beyond margin → zero push loss.
    #[test]
    fn geodesic_hinge_inactive_beyond_margin() {
        let distances = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, 0.5, Reduction::None).unwrap();
        assert!((result[0] - 0.0).abs() < TOL);
    }

    /// Inactive class within margin → positive push loss `(m − d)²`.
    #[test]
    fn geodesic_hinge_inactive_within_margin() {
        let distances = Array2::from_shape_vec((1, 2), vec![0.0, 0.3]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::None).unwrap();
        let expected = (1.0 - 0.3) * (1.0 - 0.3);
        assert!((result[0] - expected).abs() < TOL);
    }

    /// Negative margin must error out.
    #[test]
    fn geodesic_hinge_negative_margin_errors() {
        let distances = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, -1.0, Reduction::Mean);
        assert!(result.is_err());
    }

    /// Batch-size mismatch between distances and targets must error out.
    #[test]
    fn geodesic_hinge_batch_mismatch_errors() {
        let distances = Array2::from_shape_vec((2, 2), vec![0.0; 4]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::Mean);
        assert!(result.is_err());
    }

    /// Reduction::Mean divides by B (samples) — same convention as the
    /// existing `geodesic_distance_loss` path.
    #[test]
    fn geodesic_hinge_reduction_mean_matches_average() {
        let distances = Array2::from_shape_vec((2, 3), vec![0.0, 0.5, 0.8, 0.1, 0.0, 0.6]).unwrap();
        let targets = Targets::SingleLabel(vec![0, 1]);
        let none = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::None).unwrap();
        let mean = geodesic_cmr_hinge_loss(&distances, &targets, 1.0, Reduction::Mean).unwrap();
        let expected = (none[0] + none[1]) / 2.0;
        assert!((mean[0] - expected).abs() < TOL);
    }

    /// `geodesic_cmr_hinge_loss` and the existing `RegionMetrics`-based
    /// `geodesic_cmr_loss` must agree when the inputs match. We pick
    /// `k = 1` so that `||θ||₂ = θ` and the two interfaces can be
    /// compared cell-by-cell.
    #[test]
    fn geodesic_hinge_matches_region_metrics_path() {
        let distances = Array2::from_shape_vec((2, 3), vec![0.2, 0.4, 0.8, 0.1, 0.5, 0.7]).unwrap();
        let targets = Targets::SingleLabel(vec![0, 1]);
        let margin = 0.6;

        let direct =
            geodesic_cmr_hinge_loss(&distances, &targets, margin, Reduction::None).unwrap();

        // Build equivalent RegionMetrics with k=1 so ||θ|| == θ.
        let scores = Array2::zeros((2, 3)); // unused by the geodesic path
        let metrics: Vec<RegionMetrics> = (0..3)
            .map(|c| RegionMetrics {
                principal_angles: distances.column(c).to_owned().insert_axis(Axis(1)),
            })
            .collect();

        let via_metrics = cmr_region_loss(
            &scores,
            &targets,
            CmrMode::GeodesicDistance,
            margin,
            Reduction::None,
            Some(&metrics),
        )
        .unwrap();

        for i in 0..2 {
            assert!(
                (direct[i] - via_metrics[i]).abs() < TOL,
                "direct={} vs via_metrics={} at i={}",
                direct[i],
                via_metrics[i],
                i
            );
        }
    }

    // ---- analytic gradient ----

    /// Single-sample, single-active-class: `∂L/∂d = 2d / (B·C)`.
    #[test]
    fn geodesic_hinge_grad_active_class() {
        let distances = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let grad = geodesic_cmr_loss_grad(&distances, &targets, 1.0).unwrap();
        // 2 * 0.5 / (1 * 1) = 1.0.
        assert!((grad.dloss_dtheta[[0, 0]] - 1.0).abs() < TOL);
    }

    /// Active vs inactive-within-margin gradient signs and magnitudes.
    #[test]
    fn geodesic_hinge_grad_active_and_inactive_within_margin() {
        let distances = Array2::from_shape_vec((1, 2), vec![0.3, 0.4]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let margin = 1.0;
        let grad = geodesic_cmr_loss_grad(&distances, &targets, margin).unwrap();
        let norm = 1.0 / (1.0 * 2.0); // 1 / (B * C).
        // Active: 2 * 0.3 * norm.
        assert!((grad.dloss_dtheta[[0, 0]] - 2.0 * 0.3 * norm).abs() < TOL);
        // Inactive within margin: -2 * (1.0 - 0.4) * norm.
        assert!((grad.dloss_dtheta[[0, 1]] - (-2.0 * 0.6 * norm)).abs() < TOL);
    }

    /// Inactive beyond margin → zero gradient.
    #[test]
    fn geodesic_hinge_grad_inactive_beyond_margin() {
        let distances = Array2::from_shape_vec((1, 2), vec![0.3, 1.5]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let grad = geodesic_cmr_loss_grad(&distances, &targets, 1.0).unwrap();
        assert!((grad.dloss_dtheta[[0, 1]] - 0.0).abs() < TOL);
    }

    /// Verify `1 / (B · C)` normalisation explicitly on a 2 × 3 batch.
    #[test]
    fn geodesic_hinge_grad_bxc_normalization() {
        let distances = Array2::from_shape_vec((2, 3), vec![0.5; 6]).unwrap();
        let targets = Targets::SingleLabel(vec![0, 1]);
        let margin = 1.0;
        let grad = geodesic_cmr_loss_grad(&distances, &targets, margin).unwrap();
        let norm = 1.0 / (2.0 * 3.0);
        // Active for (0, 0).
        assert!((grad.dloss_dtheta[[0, 0]] - 2.0 * 0.5 * norm).abs() < TOL);
        // Active for (1, 1).
        assert!((grad.dloss_dtheta[[1, 1]] - 2.0 * 0.5 * norm).abs() < TOL);
        // Inactive within margin for (0, 1).
        assert!((grad.dloss_dtheta[[0, 1]] - (-2.0 * 0.5 * norm)).abs() < TOL);
    }

    /// Finite-difference cross-check of the analytic gradient.
    ///
    /// Compares against `Reduction::Mean` of `geodesic_cmr_hinge_loss`,
    /// matching palace's convention: `Reduction::Mean` divides by `B`
    /// only, while the analytic gradient carries `1 / (B · C)`, so the
    /// numerical gradient equals the analytic one times `C`.
    #[test]
    fn geodesic_hinge_grad_finite_diff() {
        let batch = 4;
        let num_classes = 3;
        let dist_data = vec![0.3, 0.8, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.3, 0.5, 0.4, 0.7];
        let distances = Array2::from_shape_vec((batch, num_classes), dist_data).unwrap();
        let targets = Targets::SingleLabel(vec![0, 1, 2, 0]);
        let margin = 0.785; // ≈ π / 4.

        let grad = geodesic_cmr_loss_grad(&distances, &targets, margin).unwrap();

        let eps = 1e-6;
        for i in 0..batch {
            for c in 0..num_classes {
                let mut d_plus = distances.clone();
                let mut d_minus = distances.clone();
                d_plus[[i, c]] += eps;
                d_minus[[i, c]] -= eps;

                let l_plus =
                    geodesic_cmr_hinge_loss(&d_plus, &targets, margin, Reduction::Mean).unwrap()[0];
                let l_minus = geodesic_cmr_hinge_loss(&d_minus, &targets, margin, Reduction::Mean)
                    .unwrap()[0];

                let numerical = (l_plus - l_minus) / (2.0 * eps);
                let analytical = grad.dloss_dtheta[[i, c]] * num_classes as f64;

                let abs_err = (analytical - numerical).abs();
                let rel_err = abs_err / (numerical.abs().max(1e-10));
                assert!(
                    abs_err < 1e-5 || rel_err < 1e-3,
                    "grad mismatch at [{i},{c}]: analytical={analytical:.8}, numerical={numerical:.8}, abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
                );
            }
        }
    }

    /// Negative margin in the gradient path must error out.
    #[test]
    fn geodesic_hinge_grad_negative_margin_errors() {
        let distances = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let targets = Targets::SingleLabel(vec![0]);
        let result = geodesic_cmr_loss_grad(&distances, &targets, -1.0);
        assert!(result.is_err());
    }
}
