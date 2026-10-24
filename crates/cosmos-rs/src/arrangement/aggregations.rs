//! Aggregation strategies for combining parallel projector outputs.
//!
//! Aggregation reduces a routed tensor of shape `(batch, num_projectors, dim)`
//! into a final tensor of shape `(batch, dim)` after selection and parallel
//! composition steps.
//!
//! # Key types
//!
//! - [`AggregationStrategy`]: built-in aggregation modes
//! - [`AggregationContext`]: optional runtime inputs
//! - [`AggregationInfo`]: typed telemetry describing the applied aggregation
//!
//! # Built-in strategies
//!
//! - `Mean`: uniform average over all projector branches

use std::fmt;
use std::str::FromStr;

use ndarray::{Array2, Array3, Axis};

use super::{ArrangementError, params::AggregationParams};

/// Aggregation strategy for combining multi-projector outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationStrategy {
    /// Arithmetic mean over the projector axis.
    Mean,
}

impl fmt::Display for AggregationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
        }
    }
}

impl FromStr for AggregationStrategy {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "aggregation",
                name: s.to_string(),
            }),
        }
    }
}

/// Runtime context for aggregation operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AggregationContext;

/// Typed telemetry returned alongside aggregation results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregationInfo {
    /// Strategy that produced the output.
    pub strategy: AggregationStrategy,

    /// Whether the requested strategy fell back to a simpler behavior.
    pub fallback: bool,

    /// Strategy-specific metadata.
    pub details: AggregationDetails,
}

/// Strategy-specific aggregation metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregationDetails {
    /// No extra metadata for this strategy.
    None,
}

impl AggregationStrategy {
    /// Aggregate a tensor along the projector axis.
    ///
    /// `tensor` must have shape `(batch, num_projectors, dim)`.
    ///
    /// # Returns
    ///
    /// Returns the aggregated tensor of shape `(batch, dim)` plus
    /// [`AggregationInfo`] describing the chosen strategy.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InsufficientElements`] if
    /// `num_projectors == 0`.
    pub fn apply(
        &self,
        tensor: &Array3<f64>,
        _ctx: &AggregationContext,
        _params: &AggregationParams,
    ) -> Result<(Array2<f64>, AggregationInfo), ArrangementError> {
        let (_, num_projectors, _) = validate_projector_axis(tensor, "aggregation")?;

        match self {
            Self::Mean => Ok((
                aggregate_mean(tensor, num_projectors),
                AggregationInfo {
                    strategy: *self,
                    fallback: false,
                    details: AggregationDetails::None,
                },
            )),
        }
    }
}

/// One-shot convenience helper that dispatches to
/// [`AggregationStrategy::apply`] with default context and parameters.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use cosmos_rs::arrangement::aggregations::{
///     robust_aggregate, AggregationStrategy,
/// };
///
/// let tensor = array![
///     [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
/// ];
///
/// let out = robust_aggregate(&tensor, AggregationStrategy::Mean).unwrap();
/// assert_eq!(out, array![[3.0, 30.0]]);
/// ```
///
/// # Errors
///
/// Propagates the same validation errors as [`AggregationStrategy::apply`].
pub fn robust_aggregate(
    tensor: &Array3<f64>,
    strategy: AggregationStrategy,
) -> Result<Array2<f64>, ArrangementError> {
    let params = AggregationParams {
        strategy: strategy.to_string(),
    };

    let (result, _) = strategy.apply(tensor, &AggregationContext, &params)?;
    Ok(result)
}

fn validate_projector_axis(
    tensor: &Array3<f64>,
    operation: &'static str,
) -> Result<(usize, usize, usize), ArrangementError> {
    let dims = tensor.dim();
    if dims.1 == 0 {
        return Err(ArrangementError::InsufficientElements {
            operation,
            needed: 1,
            got: 0,
        });
    }
    Ok(dims)
}

fn aggregate_mean(tensor: &Array3<f64>, num_projectors: usize) -> Array2<f64> {
    tensor
        .sum_axis(Axis(1))
        .mapv(|value| value / num_projectors as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, col), value) in actual.indexed_iter() {
            let expected_value = expected[(row, col)];
            let diff = (*value - expected_value).abs();
            assert!(
                diff <= EPSILON,
                "entry ({row}, {col}): expected {expected_value}, got {value}, diff {diff} > {EPSILON}"
            );
        }
    }

    fn base_tensor() -> Array3<f64> {
        array![[[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]]
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(AggregationStrategy::Mean.to_string(), "mean");
    }

    #[test]
    fn test_strategy_from_str_valid() {
        assert_eq!(
            "mean".parse::<AggregationStrategy>().unwrap(),
            AggregationStrategy::Mean
        );
        assert_eq!(
            "MEAN".parse::<AggregationStrategy>().unwrap(),
            AggregationStrategy::Mean
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        assert!("harmonic".parse::<AggregationStrategy>().is_err());
    }

    #[test]
    fn test_strategy_display_roundtrip() {
        let strategy = AggregationStrategy::Mean;
        let encoded = strategy.to_string();
        assert_eq!(encoded.parse::<AggregationStrategy>().unwrap(), strategy);
    }

    #[test]
    fn test_mean_strategy() {
        let tensor = base_tensor();
        let params = AggregationParams::default();

        let (out, info) = AggregationStrategy::Mean
            .apply(&tensor, &AggregationContext, &params)
            .unwrap();

        assert_array2_close(&out, &array![[3.0, 30.0]]);
        assert_eq!(info.strategy, AggregationStrategy::Mean);
        assert!(!info.fallback);
        assert_eq!(info.details, AggregationDetails::None);
    }

    #[test]
    fn test_apply_rejects_empty_projector_axis() {
        let tensor = Array3::<f64>::zeros((2, 0, 3));
        let params = AggregationParams::default();

        let err = AggregationStrategy::Mean
            .apply(&tensor, &AggregationContext, &params)
            .unwrap_err();

        match err {
            ArrangementError::InsufficientElements {
                operation,
                needed,
                got,
            } => {
                assert_eq!(operation, "aggregation");
                assert_eq!(needed, 1);
                assert_eq!(got, 0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_robust_aggregate_mean() {
        let tensor = base_tensor();
        let out = robust_aggregate(&tensor, AggregationStrategy::Mean).unwrap();

        assert_array2_close(&out, &array![[3.0, 30.0]]);
    }
}
