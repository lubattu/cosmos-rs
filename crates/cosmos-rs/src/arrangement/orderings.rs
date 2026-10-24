//! Ordering strategies for reordering projector branches before composition.
//!
//! Ordering consumes a slice of projector identifiers and returns a permutation
//! of those identifiers.
//!
//! # Key types
//!
//! - [`OrderingStrategy`]: built-in ordering modes
//! - [`OrderingContext`]: runtime ordering signals
//! - [`OrderingInfo`]: typed telemetry describing the applied ordering
//!
//! # Built-in strategies
//!
//! - `Fixed`: preserve the incoming order.

use std::fmt;
use std::str::FromStr;

use super::{ArrangementError, params::OrderingParams};

/// Ordering strategy for reordering projector indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingStrategy {
    /// Keep the original order.
    Fixed,
}

impl fmt::Display for OrderingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed => write!(f, "fixed"),
        }
    }
}

impl FromStr for OrderingStrategy {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fixed" => Ok(Self::Fixed),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "ordering",
                name: s.to_string(),
            }),
        }
    }
}

/// Runtime context for ordering operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OrderingContext;

/// Typed telemetry returned alongside an ordering result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderingInfo {
    /// Strategy that produced the ordering.
    pub strategy: OrderingStrategy,

    /// Number of candidate indices before ordering.
    pub input_count: usize,

    /// Whether the requested strategy fell back to a simpler behavior.
    pub fallback: bool,

    /// Strategy-specific metadata.
    pub details: OrderingDetails,
}

/// Strategy-specific ordering metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderingDetails {
    /// Metadata for [`OrderingStrategy::Fixed`].
    Fixed,
}

impl OrderingStrategy {
    /// Reorder projector indices.
    ///
    /// # Arguments
    ///
    /// - `indices`: candidate projector identifiers
    /// - `ctx`: runtime ordering context
    /// - `params`: caller-constructed ordering parameters
    ///
    /// # Returns
    ///
    /// Returns `(ordered_indices, info)`.
    ///
    /// # Errors
    ///
    /// The `Fixed` strategy is infallible; the result type is kept so that the
    /// block runtime treats every routing primitive uniformly.
    pub fn apply(
        &self,
        indices: &[usize],
        _ctx: &OrderingContext,
        _params: &OrderingParams,
    ) -> Result<(Vec<usize>, OrderingInfo), ArrangementError> {
        let input_count = indices.len();

        match self {
            Self::Fixed => Ok((
                indices.to_vec(),
                OrderingInfo {
                    strategy: *self,
                    input_count,
                    fallback: false,
                    details: OrderingDetails::Fixed,
                },
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_display() {
        assert_eq!(OrderingStrategy::Fixed.to_string(), "fixed");
    }

    #[test]
    fn test_strategy_from_str_valid() {
        assert_eq!(
            "fixed".parse::<OrderingStrategy>().unwrap(),
            OrderingStrategy::Fixed
        );
        assert_eq!(
            "FIXED".parse::<OrderingStrategy>().unwrap(),
            OrderingStrategy::Fixed
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        assert!("spiral".parse::<OrderingStrategy>().is_err());
    }

    #[test]
    fn test_strategy_display_roundtrip() {
        let strategy = OrderingStrategy::Fixed;
        let encoded = strategy.to_string();
        assert_eq!(encoded.parse::<OrderingStrategy>().unwrap(), strategy);
    }

    #[test]
    fn test_fixed_preserves_input_order() {
        let indices = vec![3, 0, 9, 1];
        let (ordered, info) = OrderingStrategy::Fixed
            .apply(&indices, &OrderingContext, &OrderingParams::default())
            .unwrap();

        assert_eq!(ordered, indices);
        assert_eq!(info.strategy, OrderingStrategy::Fixed);
        assert_eq!(info.input_count, 4);
        assert!(!info.fallback);
        assert_eq!(info.details, OrderingDetails::Fixed);
    }

    #[test]
    fn test_apply_on_empty_indices_returns_empty_ordering() {
        let (ordered, info) = OrderingStrategy::Fixed
            .apply(&[], &OrderingContext, &OrderingParams::default())
            .unwrap();

        assert!(ordered.is_empty());
        assert_eq!(info.input_count, 0);
    }
}
