//! Shared utilities for COSMOS objectives.
//!
//! Provides target encoding helpers used across loss functions.

use ndarray::Array2;

use super::ObjectiveError;

/// Target labels for objectives computation.
///
/// # Variants
///
/// - `SingleLabel`: class indices, one per sample. Shape: `(batch,)`.
/// - `MultiLabel`: probability/indicator matrix. Shape: `(batch, num_classes)`.
#[derive(Debug, Clone, PartialEq)]
pub enum Targets {
    /// Class index per sample (single-label classification).
    SingleLabel(Vec<usize>),
    /// Probability or indicator matrix (multi-label classification).
    MultiLabel(Array2<f64>),
}

/// Convert targets to a probability matrix of shape `(batch, num_classes)`.
///
/// - `SingleLabel`: creates one-hot encoding
/// - `MultiLabel`: validates shape and return as-is
///
/// # Arguments
///
/// * `targets` - Single-label indices or multi-label probabilities.
/// * `num_classes` - Expected number of classes.
///
/// # Errors
///
/// Returns [`ObjectiveError::TargetShapeMismatch`] if multi-label targets
/// have wrong number of columns.
///
/// # Examples
///
/// ```
/// use cosmos_rs::objectives::utils::{Targets, targets_to_probabilities};
///
/// // Single-label: [0, 2, 1] with 3 classes -> one-hot
/// let targets = Targets::SingleLabel(vec![0, 2, 1]);
/// let probs = targets_to_probabilities(&targets, 3).unwrap();
/// assert_eq!(probs.shape(), &[3, 3]);
/// assert_eq!(probs[[0, 0]], 1.0); // class 0 for sample 0
/// assert_eq!(probs[[1, 2]], 1.0); // class 2 for sample 1
/// ```
pub fn targets_to_probabilities(
    targets: &Targets,
    num_classes: usize,
) -> Result<Array2<f64>, ObjectiveError> {
    match targets {
        Targets::SingleLabel(indices) => {
            let batch_size = indices.len();
            let mut probs = Array2::zeros((batch_size, num_classes));
            for (i, &class_idx) in indices.iter().enumerate() {
                probs[[i, class_idx]] = 1.0;
            }
            Ok(probs)
        }
        Targets::MultiLabel(probs) => {
            let cols = probs.ncols();
            if cols != num_classes {
                return Err(ObjectiveError::TargetShapeMismatch {
                    expected: num_classes,
                    got: cols,
                });
            }
            Ok(probs.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ========================================================================
    // targets_to_probabilities — SingleLabel
    // ========================================================================

    #[test]
    fn single_label_one_hot_basic() {
        let targets = Targets::SingleLabel(vec![0, 2, 1]);
        let probs = targets_to_probabilities(&targets, 3).unwrap();
        assert_eq!(probs.shape(), &[3, 3]);
        assert_eq!(probs[[0, 0]], 1.0);
        assert_eq!(probs[[0, 1]], 0.0);
        assert_eq!(probs[[1, 2]], 1.0);
        assert_eq!(probs[[2, 1]], 1.0);
    }

    #[test]
    fn single_label_row_sums_are_one() {
        let targets = Targets::SingleLabel(vec![0, 1, 2, 3]);
        let probs = targets_to_probabilities(&targets, 5).unwrap();
        for i in 0..4 {
            let row_sum: f64 = probs.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn single_label_single_sample() {
        let targets = Targets::SingleLabel(vec![0]);
        let probs = targets_to_probabilities(&targets, 2).unwrap();
        assert_eq!(probs, array![[1.0, 0.0]]);
    }

    // ========================================================================
    // targets_to_probabilities — MultiLabel
    // ========================================================================

    #[test]
    fn multi_label_passthrough() {
        let matrix = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let targets = Targets::MultiLabel(matrix.clone());
        let probs = targets_to_probabilities(&targets, 3).unwrap();
        assert_eq!(probs, matrix);
    }

    #[test]
    fn multi_label_shape_mismatch() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let targets = Targets::MultiLabel(matrix);
        let err = targets_to_probabilities(&targets, 5).unwrap_err();
        assert!(format!("{err}").contains("expected"));
    }

    #[test]
    fn multi_label_exact_shape_ok() {
        let matrix = array![[0.5, 0.5]];
        let targets = Targets::MultiLabel(matrix);
        let probs = targets_to_probabilities(&targets, 2).unwrap();
        assert_eq!(probs.shape(), &[1, 2]);
    }
}
