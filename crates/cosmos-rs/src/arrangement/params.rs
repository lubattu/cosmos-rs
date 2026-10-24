//! Typed parameter structs and enums for arrangement primitives.
//!
//! This module keeps runtime configuration separate from the concrete strategy
//! enums implemented in sibling modules. Strategy names remain as strings here
//! to avoid circular dependencies, while the mode enums are validated eagerly.
//!
//! # Key types
//!
//! - [`SelectionParams`]: configuration for block selection
//! - [`OrderingParams`]: configuration for block ordering
//! - [`CompositionParams`]: configuration for projection composition
//! - [`AggregationParams`]: configuration for branch aggregation
//! - [`SubspaceAction`]: the gauge-invariant linear action `alpha I + beta P`

use std::fmt;
use std::str::FromStr;

use crate::arrangement::ArrangementError;
use serde::{Deserialize, Serialize};

/// How to handle parallel branches in composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParallelMode {
    /// Stack outputs into a single tensor and average.
    Stack,
    /// Keep outputs as a list for downstream processing.
    List,
}

/// Gauge-invariant linear action applied by each block.
///
/// Parameterizes the complete family of gauge-invariant linear operations:
///
/// ```text
/// T_{α,β}(x) = α · x_V + β · x_V⊥
/// ```
///
/// where `x_V = x W Wᵀ` (subspace component) and `x_V⊥ = x − x_V` (complement).
///
/// Equivalently: `T(x) = β·x + (α−β)·f_σ(x)` where `f_σ` is the placement-dependent
/// projection function.
///
/// Named constructors cover the classical cases:
/// - [`SubspaceAction::projection()`]: `(α=1, β=0)` — contractive projection.
/// - [`SubspaceAction::reflection()`]: `(α=1, β=−1)` — norm-preserving isometry.
/// - [`SubspaceAction::identity()`]: `(α=1, β=1)` — pass-through.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubspaceAction {
    /// Scaling of the subspace component x_V.
    pub alpha: f64,
    /// Scaling of the orthogonal complement x_V⊥.
    pub beta: f64,
}

const ACTION_EPS: f64 = 1e-12;

impl SubspaceAction {
    /// Identity operation: output equals input.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Orthogonal projection onto the subspace spanned by W.
    #[must_use]
    pub fn projection() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Householder reflection through the subspace spanned by W.
    #[must_use]
    pub fn reflection() -> Self {
        Self {
            alpha: 1.0,
            beta: -1.0,
        }
    }

    /// Complement projection: keeps only the orthogonal complement.
    #[must_use]
    pub fn complement() -> Self {
        Self {
            alpha: 0.0,
            beta: 1.0,
        }
    }

    /// The coefficient applied to the projection/activation output `f_σ(x)`.
    ///
    /// In the formula `T(x) = d·x + c·f_σ(x)`, this returns `c = α − β`.
    #[inline]
    #[must_use]
    pub fn action_coeff(&self) -> f64 {
        self.alpha - self.beta
    }

    /// The coefficient applied to the identity (pass-through) component.
    ///
    /// In the formula `T(x) = d·x + c·f_σ(x)`, this returns `d = β`.
    #[inline]
    #[must_use]
    pub fn identity_coeff(&self) -> f64 {
        self.beta
    }

    /// True if this is the standard orthogonal projection `(α=1, β=0)`.
    #[must_use]
    pub fn is_projection(&self) -> bool {
        (self.alpha - 1.0).abs() < ACTION_EPS && self.beta.abs() < ACTION_EPS
    }

    /// True if this is the Householder reflection `(α=1, β=−1)`.
    #[must_use]
    pub fn is_reflection(&self) -> bool {
        (self.alpha - 1.0).abs() < ACTION_EPS && (self.beta + 1.0).abs() < ACTION_EPS
    }

    /// True if this is the identity `(α=1, β=1)`.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.alpha - 1.0).abs() < ACTION_EPS && (self.beta - 1.0).abs() < ACTION_EPS
    }
}

impl Default for SubspaceAction {
    fn default() -> Self {
        Self::projection()
    }
}

impl fmt::Display for SubspaceAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_projection() {
            write!(f, "projection")
        } else if self.is_reflection() {
            write!(f, "reflection")
        } else if self.is_identity() {
            write!(f, "identity")
        } else {
            write!(f, "action({},{})", self.alpha, self.beta)
        }
    }
}

impl FromStr for SubspaceAction {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "projection" => Ok(Self::projection()),
            "reflection" => Ok(Self::reflection()),
            "identity" => Ok(Self::identity()),
            "complement" => Ok(Self::complement()),
            _ if s.starts_with("action(") && s.ends_with(')') => {
                let inner = &s[7..s.len() - 1];
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() != 2 {
                    return Err(ArrangementError::UnknownStrategy {
                        primitive: "subspace_action",
                        name: s,
                    });
                }
                let alpha: f64 =
                    parts[0]
                        .trim()
                        .parse()
                        .map_err(|_| ArrangementError::UnknownStrategy {
                            primitive: "subspace_action",
                            name: s.clone(),
                        })?;
                let beta: f64 =
                    parts[1]
                        .trim()
                        .parse()
                        .map_err(|_| ArrangementError::UnknownStrategy {
                            primitive: "subspace_action",
                            name: s.clone(),
                        })?;
                Ok(Self { alpha, beta })
            }
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "subspace_action",
                name: s,
            }),
        }
    }
}

/// Parameters for selection primitives — which projectors to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionParams {
    /// Selection strategy name. The only accepted value is `"all"`.
    pub strategy: String,
}

/// Parameters for ordering primitives — in what order to apply projectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderingParams {
    /// Ordering strategy name. The only accepted value is `"fixed"`.
    pub strategy: String,
}

/// Parameters for composition primitives — how projectors are combined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionParams {
    /// Composition strategy name: `"series"`, `"parallel"`,
    /// `"residual_series"` or `"parallel_series"`.
    pub strategy: String,
    /// How parallel branches leave the composition stage.
    pub parallel_mode: ParallelMode,
    /// Activation name, resolved by [`crate::arrangement::activations`].
    pub activation_name: String,
    /// Where the activation is applied: `"none"`, `"post"` or `"subspace"`.
    pub activation_placement: String,
    /// Subspace action name, parsed by [`SubspaceAction::from_str`].
    pub subspace_action: String,
}

/// Parameters for aggregation primitives — how to combine parallel outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregationParams {
    /// Aggregation strategy name. The only accepted value is `"mean"`.
    pub strategy: String,
}

// ─── Display & FromStr for ParallelMode ───────────────────────────────────────

impl fmt::Display for ParallelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParallelMode::Stack => write!(f, "stack"),
            ParallelMode::List => write!(f, "list"),
        }
    }
}

impl FromStr for ParallelMode {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "stack" => Ok(Self::Stack),
            "list" => Ok(Self::List),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "parallel_mode",
                name: s.to_string(),
            }),
        }
    }
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

impl Default for SelectionParams {
    fn default() -> Self {
        Self {
            strategy: "all".to_string(),
        }
    }
}

impl Default for OrderingParams {
    fn default() -> Self {
        Self {
            strategy: "fixed".to_string(),
        }
    }
}

impl CompositionParams {
    /// Create new composition parameters.
    ///
    /// The subspace action defaults to `"projection"`; set
    /// [`CompositionParams::subspace_action`] to change it.
    pub fn new(
        strategy: impl Into<String>,
        parallel_mode: ParallelMode,
        activation_name: impl Into<String>,
        activation_placement: impl Into<String>,
    ) -> Self {
        Self {
            strategy: strategy.into(),
            parallel_mode,
            activation_name: activation_name.into(),
            activation_placement: activation_placement.into(),
            subspace_action: "projection".to_string(),
        }
    }
}

impl Default for CompositionParams {
    fn default() -> Self {
        Self {
            strategy: "series".to_string(),
            parallel_mode: ParallelMode::Stack,
            activation_name: "none".to_string(),
            activation_placement: "none".to_string(),
            subspace_action: "projection".to_string(),
        }
    }
}

impl Default for AggregationParams {
    fn default() -> Self {
        Self {
            strategy: "mean".to_string(),
        }
    }
}

// ─── Factory Functions ────────────────────────────────────────────────────────

/// Default selection: use all projectors, no filtering.
#[must_use]
pub fn default_selection_params() -> SelectionParams {
    SelectionParams::default()
}

/// Default ordering: fixed (identity) order.
#[must_use]
pub fn default_ordering_params() -> OrderingParams {
    OrderingParams::default()
}

/// Default composition: series (sequential) application.
#[must_use]
pub fn default_composition_params() -> CompositionParams {
    CompositionParams::default()
}

/// Residual series: sequential with skip connection.
#[must_use]
pub fn residual_series_params() -> CompositionParams {
    CompositionParams::new("residual_series", ParallelMode::Stack, "none", "none")
}

/// Default aggregation: arithmetic mean.
#[must_use]
pub fn default_aggregation_params() -> AggregationParams {
    AggregationParams::default()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_mode_display() {
        assert_eq!(ParallelMode::Stack.to_string(), "stack");
        assert_eq!(ParallelMode::List.to_string(), "list");
    }

    #[test]
    fn test_parallel_mode_from_str() {
        assert_eq!(
            "stack".parse::<ParallelMode>().unwrap(),
            ParallelMode::Stack
        );
        assert_eq!("LIST".parse::<ParallelMode>().unwrap(), ParallelMode::List);
    }

    #[test]
    fn test_parallel_mode_from_str_invalid() {
        assert!("concatenate".parse::<ParallelMode>().is_err());
    }

    #[test]
    fn test_enum_display_roundtrip() {
        for mode in [ParallelMode::Stack, ParallelMode::List] {
            let s = mode.to_string();
            assert_eq!(s.parse::<ParallelMode>().unwrap(), mode);
        }
    }

    #[test]
    fn test_selection_params_default() {
        assert_eq!(SelectionParams::default().strategy, "all");
    }

    #[test]
    fn test_ordering_params_default() {
        assert_eq!(OrderingParams::default().strategy, "fixed");
    }

    #[test]
    fn test_composition_params_default() {
        let p = CompositionParams::default();
        assert_eq!(p.strategy, "series");
        assert_eq!(p.parallel_mode, ParallelMode::Stack);
        assert_eq!(p.activation_name, "none");
        assert_eq!(p.activation_placement, "none");
        assert_eq!(p.subspace_action, "projection");
    }

    #[test]
    fn test_aggregation_params_default() {
        assert_eq!(AggregationParams::default().strategy, "mean");
    }

    #[test]
    fn test_composition_params_new() {
        let p = CompositionParams::new("parallel", ParallelMode::List, "relu", "post");
        assert_eq!(p.strategy, "parallel");
        assert_eq!(p.parallel_mode, ParallelMode::List);
        assert_eq!(p.activation_name, "relu");
        assert_eq!(p.activation_placement, "post");
    }

    #[test]
    fn test_subspace_action_named_constructors() {
        assert!(SubspaceAction::projection().is_projection());
        assert!(SubspaceAction::reflection().is_reflection());
        assert!(SubspaceAction::identity().is_identity());
        assert_eq!(SubspaceAction::default(), SubspaceAction::projection());
    }

    #[test]
    fn test_subspace_action_coefficients() {
        let a = SubspaceAction::reflection();
        assert_eq!(a.action_coeff(), 2.0);
        assert_eq!(a.identity_coeff(), -1.0);
    }

    #[test]
    fn test_subspace_action_display_roundtrip() {
        for action in [
            SubspaceAction::projection(),
            SubspaceAction::reflection(),
            SubspaceAction::identity(),
            SubspaceAction {
                alpha: 0.5,
                beta: 0.25,
            },
        ] {
            let s = action.to_string();
            assert_eq!(s.parse::<SubspaceAction>().unwrap(), action);
        }
    }

    #[test]
    fn test_subspace_action_from_str_rejects_unknown() {
        assert!(matches!(
            "t=0.3".parse::<SubspaceAction>(),
            Err(ArrangementError::UnknownStrategy { .. })
        ));
        assert!(matches!(
            "nonsense".parse::<SubspaceAction>(),
            Err(ArrangementError::UnknownStrategy { .. })
        ));
    }

    #[test]
    fn test_default_selection_factory() {
        assert_eq!(default_selection_params().strategy, "all");
    }

    #[test]
    fn test_default_ordering_factory() {
        assert_eq!(default_ordering_params().strategy, "fixed");
    }

    #[test]
    fn test_default_composition_factory() {
        assert_eq!(default_composition_params().strategy, "series");
    }

    #[test]
    fn test_residual_series_factory() {
        assert_eq!(residual_series_params().strategy, "residual_series");
    }

    #[test]
    fn test_default_aggregation_factory() {
        assert_eq!(default_aggregation_params().strategy, "mean");
    }

    #[test]
    fn test_selection_params_serde_roundtrip() {
        let p = SelectionParams::default();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<SelectionParams>(&json).unwrap(), p);
    }

    #[test]
    fn params_deserialization_rejects_unknown_fields() {
        let err =
            serde_json::from_str::<SelectionParams>(r#"{"strategy":"all","unexpected_field":3}"#)
                .unwrap_err();
        assert!(
            err.to_string().contains("unexpected_field"),
            "expected the unknown key to be named, got: {err}"
        );

        let err =
            serde_json::from_str::<OrderingParams>(r#"{"strategy":"fixed","unexpected_field":3}"#)
                .unwrap_err();
        assert!(
            err.to_string().contains("unexpected_field"),
            "expected the unknown key to be named, got: {err}"
        );

        let err = serde_json::from_str::<AggregationParams>(
            r#"{"strategy":"mean","not_a_parameter":0.1}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("not_a_parameter"),
            "expected the unknown key to be named, got: {err}"
        );
    }

    #[test]
    fn test_ordering_params_serde_roundtrip() {
        let p = OrderingParams::default();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<OrderingParams>(&json).unwrap(), p);
    }

    #[test]
    fn test_composition_params_serde_roundtrip() {
        let p = CompositionParams::new("parallel_series", ParallelMode::List, "gelu", "subspace");
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<CompositionParams>(&json).unwrap(), p);
    }

    #[test]
    fn test_aggregation_params_serde_roundtrip() {
        let p = AggregationParams::default();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<AggregationParams>(&json).unwrap(), p);
    }
}
