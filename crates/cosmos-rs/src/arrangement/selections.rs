//! Selection strategies for choosing which projector branches remain active.
//!
//! Selection operates over a list of projector identifiers and returns the
//! subset that the rest of the pipeline will route through.
//!
//! # Key types
//!
//! - [`SelectionStrategy`]: built-in selection modes
//! - [`SelectionContext`]: runtime signals such as the training mode
//! - [`SelectionInfo`]: typed telemetry describing what was selected
//!
//! # Built-in strategies
//!
//! - `All`: keep every projector.

use std::fmt;
use std::str::FromStr;

use super::{ArrangementError, params::SelectionParams};

/// Selection strategy for choosing active projector indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Keep all available projectors.
    All,
}

impl fmt::Display for SelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, "all"),
        }
    }
}

impl FromStr for SelectionStrategy {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "selection",
                name: s.to_string(),
            }),
        }
    }
}

/// Runtime context for selection operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionContext {
    /// Whether the caller is in training mode.
    pub training: bool,
}

impl Default for SelectionContext {
    fn default() -> Self {
        Self { training: true }
    }
}

/// Typed telemetry returned alongside a selection result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionInfo {
    /// Strategy that produced the selection.
    pub strategy: SelectionStrategy,

    /// Number of candidate indices before selection.
    pub input_count: usize,

    /// Number of indices retained after selection.
    pub output_count: usize,

    /// Strategy-specific metadata.
    pub details: SelectionDetails,
}

/// Strategy-specific selection metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionDetails {
    /// Metadata for [`SelectionStrategy::All`].
    All,
}

impl SelectionStrategy {
    /// Select a subset of projector indices.
    ///
    /// # Arguments
    ///
    /// - `indices`: candidate projector identifiers
    /// - `ctx`: runtime selection context
    /// - `params`: caller-constructed selection parameters
    ///
    /// # Returns
    ///
    /// Returns `(selected_indices, info)`.
    ///
    /// # Errors
    ///
    /// The `All` strategy is infallible; the result type is kept so that the
    /// block runtime treats every routing primitive uniformly.
    pub fn apply(
        &self,
        indices: &[usize],
        _ctx: &SelectionContext,
        _params: &SelectionParams,
    ) -> Result<(Vec<usize>, SelectionInfo), ArrangementError> {
        let input_count = indices.len();

        match self {
            Self::All => Ok((
                indices.to_vec(),
                SelectionInfo {
                    strategy: self.clone(),
                    input_count,
                    output_count: input_count,
                    details: SelectionDetails::All,
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
        assert_eq!(SelectionStrategy::All.to_string(), "all");
    }

    #[test]
    fn test_strategy_from_str_valid() {
        assert_eq!(
            "all".parse::<SelectionStrategy>().unwrap(),
            SelectionStrategy::All
        );
        assert_eq!(
            "ALL".parse::<SelectionStrategy>().unwrap(),
            SelectionStrategy::All
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        assert!("roulette".parse::<SelectionStrategy>().is_err());
    }

    #[test]
    fn test_strategy_display_roundtrip() {
        let strategy = SelectionStrategy::All;
        let encoded = strategy.to_string();
        assert_eq!(encoded.parse::<SelectionStrategy>().unwrap(), strategy);
    }

    #[test]
    fn test_context_default() {
        assert!(SelectionContext::default().training);
    }

    #[test]
    fn test_all_strategy_returns_input_unchanged() {
        let indices = vec![4, 1, 7];
        let params = SelectionParams::default();
        let ctx = SelectionContext::default();

        let (selected, info) = SelectionStrategy::All
            .apply(&indices, &ctx, &params)
            .unwrap();

        assert_eq!(selected, indices);
        assert_eq!(info.strategy, SelectionStrategy::All);
        assert_eq!(info.input_count, 3);
        assert_eq!(info.output_count, 3);
        assert_eq!(info.details, SelectionDetails::All);
    }

    #[test]
    fn test_apply_on_empty_indices_returns_empty_selection() {
        let indices = Vec::new();
        let params = SelectionParams::default();
        let ctx = SelectionContext::default();

        let (selected, info) = SelectionStrategy::All
            .apply(&indices, &ctx, &params)
            .unwrap();

        assert!(selected.is_empty());
        assert_eq!(info.input_count, 0);
        assert_eq!(info.output_count, 0);
    }
}
