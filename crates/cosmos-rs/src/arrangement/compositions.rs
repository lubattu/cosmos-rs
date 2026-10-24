//! Composition strategies for combining multiple projector applications.
//!
//! Compositions consume a list of projector matrices `W_i` with shapes
//! `(n, k_i)` plus batch-major input `x` with shape `(batch, n)`, and return a
//! composed output in the same ambient coordinates `(batch, n)`.
//!
//! # Key types
//!
//! - [`CompositionStrategy`]: built-in ways to combine projector applications
//! - [`CompositionContext`]: activation and placement settings shared across
//!   all projectors in the composition
//!
//! # Built-in strategies
//!
//! - `Series`: sequential projection `P_L(...P_2(P_1(x)))`
//! - `Parallel`: arithmetic mean of independent branch projections
//! - `ResidualSeries`: skip connection `x + series(x)`
//! - `ParallelSeries`: two-phase `step = parallel(x); y = series(step)`

use std::fmt;
use std::str::FromStr;

use ndarray::Array2;

use super::{
    ArrangementError,
    activations::{Activation, ActivationPlacement, apply_subspace_action},
    params::SubspaceAction,
};

/// Batch-mean input/output row norms for one projector step of a
/// composition, recorded by [`CompositionStrategy::apply_with_flow`].
///
/// `mean_in_norm` is the mean over batch rows of the L2 norm of the data
/// entering the step; `mean_out_norm` is the same statistic for the step's
/// output. The ratio `mean_out_norm / mean_in_norm` measures how much signal
/// energy the step preserves (1.0 = isometric, < 1.0 = contractive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowNorm {
    /// Mean row-wise L2 norm of the step input.
    pub mean_in_norm: f64,

    /// Mean row-wise L2 norm of the step output.
    pub mean_out_norm: f64,
}

/// Mean over batch rows of the row-wise L2 norm. Empty input returns zero.
pub(crate) fn mean_row_norm(x: &Array2<f64>) -> f64 {
    let rows = x.nrows();
    if rows == 0 {
        return 0.0;
    }

    let total: f64 = x
        .outer_iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
        .sum();
    total / rows as f64
}

/// Composition strategy for combining multiple projector outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositionStrategy {
    /// Sequential projection: `P_L(...P_2(P_1(x)))`.
    Series,

    /// Independent projections averaged: `mean(P_i(x))`.
    Parallel,

    /// Series composition with a skip connection: `x + series(x)`.
    ResidualSeries,

    /// Two-phase composition: `step = mean(P_i(x))`, then `y = series(step)`
    /// using the same projector list. Useful for letting a parallel pool
    /// of branches feed a deeper sequential refinement chain.
    ParallelSeries,
}

impl fmt::Display for CompositionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Series => write!(f, "series"),
            Self::Parallel => write!(f, "parallel"),
            Self::ResidualSeries => write!(f, "residual_series"),
            Self::ParallelSeries => write!(f, "parallel_series"),
        }
    }
}

impl FromStr for CompositionStrategy {
    type Err = ArrangementError;

    /// Parse a composition-strategy name.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "series" => Ok(Self::Series),
            "parallel" => Ok(Self::Parallel),
            "residual_series" | "residualseries" => Ok(Self::ResidualSeries),
            "parallel_series" | "parallelseries" => Ok(Self::ParallelSeries),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "composition",
                name: s.to_string(),
            }),
        }
    }
}

/// Runtime context shared by all projector applications in a composition.
///
/// Each projector can have its own [`SubspaceAction`] via the `actions` vector.
/// When all projectors share the same action, use [`CompositionContext::uniform`]
/// to build the context.
/// `block_indices` is carried for alignment with higher-level arrangement code
/// that may want to associate the projector list with specific block ids.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionContext {
    /// Activation applied at each projector step.
    pub activation: Activation,

    /// Where the activation is applied relative to the Grassmann projection.
    pub placement: ActivationPlacement,

    /// Per-projector gauge-invariant subspace actions. Length must equal the
    /// number of projectors passed to [`CompositionStrategy::apply`].
    pub actions: Vec<SubspaceAction>,

    /// Optional bookkeeping of block identifiers associated with `projectors`.
    pub block_indices: Vec<usize>,
}

impl CompositionContext {
    /// Build a context where every projector shares the same action.
    pub fn uniform(action: SubspaceAction, num_projectors: usize) -> Self {
        Self {
            actions: vec![action; num_projectors],
            ..Default::default()
        }
    }
}

impl Default for CompositionContext {
    fn default() -> Self {
        Self {
            activation: Activation::Identity,
            placement: ActivationPlacement::None,
            actions: Vec::new(),
            block_indices: Vec::new(),
        }
    }
}

impl CompositionStrategy {
    /// Compose projectors applied to input data.
    ///
    /// # Arguments
    ///
    /// - `projectors`: list of projector matrices, each with shape `(n, k_i)`
    /// - `x`: input data with shape `(batch, n)`
    /// - `ctx`: activation configuration shared across the composition
    ///
    /// # Returns
    ///
    /// Returns the composed output with shape `(batch, n)`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::ShapeMismatch`] if any projector row count
    /// differs from `x.ncols()`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use cosmos_rs::arrangement::compositions::{CompositionContext, CompositionStrategy};
    ///
    /// let projectors = vec![array![[1.0], [0.0]], array![[0.0], [1.0]]];
    /// let x = array![[2.0, -3.0]];
    /// let y = CompositionStrategy::Parallel
    ///     .apply(&projectors, &x, &CompositionContext::default())
    ///     .unwrap();
    ///
    /// assert_eq!(y, array![[1.0, -1.5]]);
    /// ```
    pub fn apply(
        &self,
        projectors: &[Array2<f64>],
        x: &Array2<f64>,
        ctx: &CompositionContext,
    ) -> Result<Array2<f64>, ArrangementError> {
        validate_projectors(projectors, x)?;

        if !ctx.actions.is_empty() && ctx.actions.len() != projectors.len() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("actions.len() == projectors.len() ({})", projectors.len()),
                got: format!("actions.len() = {}", ctx.actions.len()),
            });
        }

        match self {
            Self::Series => apply_series(projectors, x, ctx),
            Self::Parallel => apply_parallel(projectors, x, ctx),
            Self::ResidualSeries => {
                let series_output = apply_series(projectors, x, ctx)?;
                let mut output = x.clone();
                output += &series_output;
                Ok(output)
            }
            Self::ParallelSeries => apply_parallel_series(projectors, x, ctx),
        }
    }

    /// Compose projectors while recording per-step flow norms.
    ///
    /// The output array is identical to [`CompositionStrategy::apply`]; the
    /// second element holds one [`FlowNorm`] per projector step, in projector
    /// order. Per strategy:
    ///
    /// - `Series`: step `i` records the running chain value before and after
    ///   projector `i`.
    /// - `Parallel`: every branch reads the shared input `x`; the recorded
    ///   output is that branch's projection before mean-pooling.
    /// - `ResidualSeries`: records come from the wrapped series composition;
    ///   the skip connection `x + f(x)` is applied after the chain and is not
    ///   folded into per-step records.
    /// - `ParallelSeries`: phase-1 parallel records followed by phase-2
    ///   series records — the vector holds `2 * projectors.len()` entries.
    ///
    /// # Errors
    ///
    /// Same failure modes as [`CompositionStrategy::apply`].
    pub fn apply_with_flow(
        &self,
        projectors: &[Array2<f64>],
        x: &Array2<f64>,
        ctx: &CompositionContext,
    ) -> Result<(Array2<f64>, Vec<FlowNorm>), ArrangementError> {
        validate_projectors(projectors, x)?;

        if !ctx.actions.is_empty() && ctx.actions.len() != projectors.len() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("actions.len() == projectors.len() ({})", projectors.len()),
                got: format!("actions.len() = {}", ctx.actions.len()),
            });
        }

        let mut records = Vec::new();

        let output = match self {
            Self::Series => apply_series_recorded(projectors, x, ctx, Some(&mut records))?,
            Self::Parallel => apply_parallel_recorded(projectors, x, ctx, Some(&mut records))?,
            Self::ResidualSeries => {
                let series_output = apply_series_recorded(projectors, x, ctx, Some(&mut records))?;
                let mut output = x.clone();
                output += &series_output;
                output
            }
            Self::ParallelSeries => {
                let step = apply_parallel_recorded(projectors, x, ctx, Some(&mut records))?;
                apply_series_recorded(projectors, &step, ctx, Some(&mut records))?
            }
        };

        Ok((output, records))
    }
}

fn validate_projectors(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
) -> Result<(), ArrangementError> {
    let ambient_dim = x.ncols();

    for (idx, projector) in projectors.iter().enumerate() {
        if projector.nrows() != ambient_dim {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("projector[{idx}] with ambient dimension {ambient_dim}"),
                got: format!("({}, {})", projector.nrows(), projector.ncols()),
            });
        }
    }

    Ok(())
}

fn apply_series(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    ctx: &CompositionContext,
) -> Result<Array2<f64>, ArrangementError> {
    apply_series_recorded(projectors, x, ctx, None)
}

/// Series chain with optional per-step [`FlowNorm`] recording.
///
/// The computation is identical to the unrecorded path; when `records` is
/// present, each step appends the batch-mean row norms of its input and
/// output.
fn apply_series_recorded(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    ctx: &CompositionContext,
    mut records: Option<&mut Vec<FlowNorm>>,
) -> Result<Array2<f64>, ArrangementError> {
    let mut current = x.clone();
    let default_action = SubspaceAction::default();

    for (idx, projector) in projectors.iter().enumerate() {
        let action = ctx.actions.get(idx).copied().unwrap_or(default_action);
        let output =
            apply_subspace_action(projector, &current, ctx.activation, ctx.placement, action);
        if let Some(recs) = records.as_deref_mut() {
            recs.push(FlowNorm {
                mean_in_norm: mean_row_norm(&current),
                mean_out_norm: mean_row_norm(&output),
            });
        }
        current = output;
    }

    Ok(current)
}

fn apply_parallel(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    ctx: &CompositionContext,
) -> Result<Array2<f64>, ArrangementError> {
    apply_parallel_recorded(projectors, x, ctx, None)
}

/// Parallel mean-pool with optional per-branch [`FlowNorm`] recording.
///
/// Every branch reads the shared input `x`; the recorded output is that
/// branch's projection before mean-pooling.
fn apply_parallel_recorded(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    ctx: &CompositionContext,
    mut records: Option<&mut Vec<FlowNorm>>,
) -> Result<Array2<f64>, ArrangementError> {
    if projectors.is_empty() {
        return Ok(x.clone());
    }

    let shared_in_norm = records.is_some().then(|| mean_row_norm(x));
    let mut acc = Array2::<f64>::zeros(x.dim());
    let default_action = SubspaceAction::default();

    for (idx, projector) in projectors.iter().enumerate() {
        let action = ctx.actions.get(idx).copied().unwrap_or(default_action);
        let output = apply_subspace_action(projector, x, ctx.activation, ctx.placement, action);
        if let Some(recs) = records.as_deref_mut() {
            recs.push(FlowNorm {
                mean_in_norm: shared_in_norm.unwrap_or_default(),
                mean_out_norm: mean_row_norm(&output),
            });
        }
        acc += &output;
    }

    let scale = 1.0 / projectors.len() as f64;
    Ok(acc.mapv(|value| value * scale))
}

/// Two-phase composition: parallel mean fed into a series chain.
///
/// `step = mean(P_i(x))`; then `result = P_L(... P_2(P_1(step)))`.
/// Both phases share the same projector list.
fn apply_parallel_series(
    projectors: &[Array2<f64>],
    x: &Array2<f64>,
    ctx: &CompositionContext,
) -> Result<Array2<f64>, ArrangementError> {
    let step = apply_parallel(projectors, x, ctx)?;
    apply_series(projectors, &step, ctx)
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

    fn axis_projectors() -> Vec<Array2<f64>> {
        vec![array![[1.0], [0.0]], array![[0.0], [1.0]]]
    }

    fn mixed_sign_basis() -> Array2<f64> {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        array![[inv_sqrt2], [-inv_sqrt2]]
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(CompositionStrategy::Series.to_string(), "series");
        assert_eq!(CompositionStrategy::Parallel.to_string(), "parallel");
        assert_eq!(
            CompositionStrategy::ResidualSeries.to_string(),
            "residual_series"
        );
        assert_eq!(
            CompositionStrategy::ParallelSeries.to_string(),
            "parallel_series"
        );
    }

    #[test]
    fn test_strategy_from_str_valid() {
        assert_eq!(
            "series".parse::<CompositionStrategy>().unwrap(),
            CompositionStrategy::Series
        );
        assert_eq!(
            "parallel".parse::<CompositionStrategy>().unwrap(),
            CompositionStrategy::Parallel
        );
        assert_eq!(
            "residual_series".parse::<CompositionStrategy>().unwrap(),
            CompositionStrategy::ResidualSeries
        );
        assert_eq!(
            "RESIDUALSERIES".parse::<CompositionStrategy>().unwrap(),
            CompositionStrategy::ResidualSeries
        );
        assert_eq!(
            "parallel_series".parse::<CompositionStrategy>().unwrap(),
            CompositionStrategy::ParallelSeries
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        assert!("cascade".parse::<CompositionStrategy>().is_err());
    }

    #[test]
    fn test_strategy_display_roundtrip() {
        for strategy in [
            CompositionStrategy::Series,
            CompositionStrategy::Parallel,
            CompositionStrategy::ResidualSeries,
            CompositionStrategy::ParallelSeries,
        ] {
            let encoded = strategy.to_string();
            assert_eq!(encoded.parse::<CompositionStrategy>().unwrap(), strategy);
        }
    }

    #[test]
    fn test_context_default() {
        let ctx = CompositionContext::default();
        assert_eq!(ctx.activation, Activation::Identity);
        assert_eq!(ctx.placement, ActivationPlacement::None);
        assert!(ctx.block_indices.is_empty());
    }

    #[test]
    fn test_series_identity_applies_projectors_sequentially() {
        let projectors = axis_projectors();
        let x = array![[2.0, -3.0], [4.0, 5.0]];

        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &CompositionContext::default())
            .unwrap();

        assert_array2_close(&out, &array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_series_post_activation_uses_project_with_activation() {
        let projectors = vec![mixed_sign_basis()];
        let x = array![[1.0, -1.0], [-1.0, 1.0]];
        let ctx = CompositionContext {
            activation: Activation::Relu,
            placement: ActivationPlacement::Post,
            block_indices: vec![0],
            ..Default::default()
        };

        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();

        assert_array2_close(&out, &array![[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn test_series_subspace_activation_uses_project_with_activation() {
        let projectors = vec![mixed_sign_basis()];
        let x = array![[1.0, -1.0], [-1.0, 1.0]];
        let ctx = CompositionContext {
            activation: Activation::Relu,
            placement: ActivationPlacement::Subspace,
            block_indices: vec![0],
            ..Default::default()
        };

        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();

        assert_array2_close(&out, &array![[1.0, -1.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_parallel_averages_independent_projections() {
        let projectors = axis_projectors();
        let x = array![[2.0, -3.0], [4.0, 5.0]];

        let out = CompositionStrategy::Parallel
            .apply(&projectors, &x, &CompositionContext::default())
            .unwrap();

        assert_array2_close(&out, &array![[1.0, -1.5], [2.0, 2.5]]);
    }

    #[test]
    fn test_parallel_empty_projectors_returns_input() {
        let x = array![[2.0, -3.0], [4.0, 5.0]];

        let out = CompositionStrategy::Parallel
            .apply(&[], &x, &CompositionContext::default())
            .unwrap();

        assert_array2_close(&out, &x);
    }

    #[test]
    fn test_residual_series_adds_skip_connection() {
        let projectors = vec![array![[1.0], [0.0]]];
        let x = array![[2.0, -3.0], [4.0, 5.0]];

        let out = CompositionStrategy::ResidualSeries
            .apply(&projectors, &x, &CompositionContext::default())
            .unwrap();

        assert_array2_close(&out, &array![[4.0, -3.0], [8.0, 5.0]]);
    }

    #[test]
    fn test_single_projector_series_matches_parallel() {
        let projectors = vec![mixed_sign_basis()];
        let x = array![[1.0, -1.0], [-1.0, 1.0]];
        let ctx = CompositionContext {
            activation: Activation::Relu,
            placement: ActivationPlacement::Post,
            block_indices: vec![3],
            ..Default::default()
        };

        let series = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();
        let parallel = CompositionStrategy::Parallel
            .apply(&projectors, &x, &ctx)
            .unwrap();

        assert_array2_close(&series, &parallel);
    }

    #[test]
    fn test_apply_rejects_mismatched_projector_shape() {
        let projectors = vec![array![[1.0], [0.0], [0.0]]];
        let x = array![[2.0, -3.0]];

        let err = CompositionStrategy::Series
            .apply(&projectors, &x, &CompositionContext::default())
            .unwrap_err();

        match err {
            ArrangementError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, "projector[0] with ambient dimension 2");
                assert_eq!(got, "(3, 1)");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ── Reflection composition tests ──────────────────────────────────

    fn reflection_ctx(num_projectors: usize) -> CompositionContext {
        CompositionContext {
            actions: vec![SubspaceAction::reflection(); num_projectors],
            ..Default::default()
        }
    }

    fn row_norm(x: &Array2<f64>, row: usize) -> f64 {
        x.row(row).iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn reflection_series_preserves_norm() {
        // Series of reflections through orthogonal subspaces: each is an isometry,
        // so the composition is an isometry
        let projectors = axis_projectors(); // e1, e2 in R^2
        let x = array![[3.0, -4.0], [1.0, 2.0]];

        let ctx = reflection_ctx(projectors.len());
        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();

        for i in 0..x.nrows() {
            let in_norm = row_norm(&x, i);
            let out_norm = row_norm(&out, i);
            assert!(
                (out_norm - in_norm).abs() < 1e-10,
                "row {i}: input norm {in_norm}, output norm {out_norm}"
            );
        }
    }

    #[test]
    fn reflection_series_single_is_2p_minus_x() {
        let projectors = vec![array![[1.0], [0.0]]];
        let x = array![[2.0, -3.0], [4.0, 5.0]];

        let ctx = reflection_ctx(projectors.len());
        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();

        // R(x) = 2*proj_e1(x) - x = 2*[x1,0] - [x1,x2] = [x1, -x2]
        assert_array2_close(&out, &array![[2.0, 3.0], [4.0, -5.0]]);
    }

    #[test]
    fn reflection_parallel_averages_reflections() {
        let projectors = axis_projectors();
        let x = array![[2.0, -3.0]];

        let ctx = reflection_ctx(projectors.len());
        let out = CompositionStrategy::Parallel
            .apply(&projectors, &x, &ctx)
            .unwrap();

        // R_e1(x) = [2, 3], R_e2(x) = [-2, -3]
        // mean = [0, 0]
        assert_array2_close(&out, &array![[0.0, 0.0]]);
    }

    #[test]
    fn reflection_residual_series_well_conditioned() {
        // The key test: x + R(x) should have both terms at similar scale
        let projectors = vec![array![[1.0], [0.0]]];
        let x = array![[2.0, -3.0]];

        let ctx = reflection_ctx(projectors.len());
        let out = CompositionStrategy::ResidualSeries
            .apply(&projectors, &x, &ctx)
            .unwrap();

        // x + R(x) = [2,-3] + [2,3] = [4, 0]
        assert_array2_close(&out, &array![[4.0, 0.0]]);
    }

    #[test]
    fn reflection_series_many_blocks_preserves_norm() {
        // Even with 6 blocks, norm is preserved (unlike projection which loses ~97%)
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let projectors = vec![
            array![[1.0], [0.0]],
            array![[0.0], [1.0]],
            array![[inv_sqrt2], [inv_sqrt2]],
            array![[inv_sqrt2], [-inv_sqrt2]],
            array![[1.0], [0.0]],
            array![[0.0], [1.0]],
        ];
        let x = array![[3.0, 4.0]];
        let input_norm = row_norm(&x, 0);

        let ctx = reflection_ctx(projectors.len());
        let out = CompositionStrategy::Series
            .apply(&projectors, &x, &ctx)
            .unwrap();

        let output_norm = row_norm(&out, 0);
        assert!(
            (output_norm - input_norm).abs() < 1e-10,
            "6-block series: input norm {input_norm}, output norm {output_norm}"
        );
    }

    // -----------------------------------------------------------------------
    // ParallelSeries
    // -----------------------------------------------------------------------

    /// `parallel_series` should equal `series ∘ parallel` applied with
    /// the same projector list.
    #[test]
    fn parallel_series_matches_manual_pipeline() {
        let projectors = axis_projectors();
        let x = array![[3.0, 4.0]];
        let ctx = CompositionContext {
            actions: vec![SubspaceAction::projection(); projectors.len()],
            ..Default::default()
        };

        let direct = CompositionStrategy::ParallelSeries
            .apply(&projectors, &x, &ctx)
            .unwrap();

        let step = CompositionStrategy::Parallel
            .apply(&projectors, &x, &ctx)
            .unwrap();
        let manual = CompositionStrategy::Series
            .apply(&projectors, &step, &ctx)
            .unwrap();

        assert_array2_close(&direct, &manual);
    }

    /// Empty projector list passes through `parallel_series` unchanged.
    #[test]
    fn parallel_series_empty_projectors_is_identity() {
        let projectors: Vec<Array2<f64>> = Vec::new();
        let x = array![[1.0, 2.0]];
        let ctx = CompositionContext::default();
        let out = CompositionStrategy::ParallelSeries
            .apply(&projectors, &x, &ctx)
            .unwrap();
        assert_array2_close(&out, &x);
    }

    /// Display + FromStr roundtrip for `parallel_series`.
    #[test]
    fn parallel_series_display_fromstr() {
        assert_eq!(
            format!("{}", CompositionStrategy::ParallelSeries),
            "parallel_series"
        );
        assert_eq!(
            CompositionStrategy::from_str("parallel_series").unwrap(),
            CompositionStrategy::ParallelSeries
        );
        // Unknown names must error.
        assert!(CompositionStrategy::from_str("parallel_series:2").is_err());
    }

    // -----------------------------------------------------------------------
    // Flow-instrumented composition (apply_with_flow)
    // -----------------------------------------------------------------------

    fn flow_projectors() -> Vec<Array2<f64>> {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        vec![
            array![[1.0], [0.0]],
            array![[0.0], [1.0]],
            array![[inv_sqrt2], [inv_sqrt2]],
        ]
    }

    fn flow_input() -> Array2<f64> {
        array![[3.0, -4.0], [1.0, 2.0], [-0.5, 0.25]]
    }

    fn all_strategies() -> Vec<CompositionStrategy> {
        vec![
            CompositionStrategy::Series,
            CompositionStrategy::Parallel,
            CompositionStrategy::ResidualSeries,
            CompositionStrategy::ParallelSeries,
        ]
    }

    /// `apply_with_flow` must return the exact (bit-equal) output of
    /// `apply` for every strategy, both with the default context and with
    /// an activated reflection context.
    #[test]
    fn apply_with_flow_output_bit_equals_apply_for_all_strategies() {
        let projectors = flow_projectors();
        let x = flow_input();
        let contexts = [
            CompositionContext::uniform(SubspaceAction::projection(), projectors.len()),
            CompositionContext {
                activation: Activation::Relu,
                placement: ActivationPlacement::Post,
                actions: vec![SubspaceAction::reflection(); projectors.len()],
                ..Default::default()
            },
        ];

        for ctx in &contexts {
            for strategy in all_strategies() {
                let plain = strategy.apply(&projectors, &x, ctx).unwrap();
                let (flowed, _records) = strategy.apply_with_flow(&projectors, &x, ctx).unwrap();
                assert_eq!(
                    plain, flowed,
                    "strategy {strategy} output diverged between apply and apply_with_flow"
                );
            }
        }
    }

    /// Record counts per strategy: series/parallel/residual record one
    /// entry per projector, and parallel_series records two phases.
    #[test]
    fn apply_with_flow_record_counts_match_strategy_shape() {
        let projectors = flow_projectors();
        let x = flow_input();
        let ctx = CompositionContext::uniform(SubspaceAction::projection(), projectors.len());
        let n = projectors.len();

        let expected = [
            (CompositionStrategy::Series, n),
            (CompositionStrategy::Parallel, n),
            (CompositionStrategy::ResidualSeries, n),
            (CompositionStrategy::ParallelSeries, 2 * n),
        ];

        for (strategy, count) in expected {
            let (_, records) = strategy.apply_with_flow(&projectors, &x, &ctx).unwrap();
            assert_eq!(
                records.len(),
                count,
                "strategy {strategy} recorded {} entries, expected {count}",
                records.len()
            );
        }
    }

    /// Pure projection is non-expansive, so every series step must have
    /// `mean_out_norm <= mean_in_norm` (up to numerical noise).
    #[test]
    fn apply_with_flow_series_projection_is_non_expansive() {
        let projectors = flow_projectors();
        let x = flow_input();
        let ctx = CompositionContext::uniform(SubspaceAction::projection(), projectors.len());

        let (_, records) = CompositionStrategy::Series
            .apply_with_flow(&projectors, &x, &ctx)
            .unwrap();

        assert_eq!(records.len(), projectors.len());
        for (idx, record) in records.iter().enumerate() {
            assert!(
                record.mean_out_norm <= record.mean_in_norm + 1e-12,
                "step {idx}: projection expanded norms ({} > {})",
                record.mean_out_norm,
                record.mean_in_norm
            );
        }
    }

    /// The residual skip is applied after the chain: the output equals
    /// `x + series(x)` while the records match the bare series records.
    #[test]
    fn apply_with_flow_residual_series_records_exclude_skip() {
        let projectors = flow_projectors();
        let x = flow_input();
        let ctx = CompositionContext::uniform(SubspaceAction::projection(), projectors.len());

        let (residual_out, residual_records) = CompositionStrategy::ResidualSeries
            .apply_with_flow(&projectors, &x, &ctx)
            .unwrap();
        let (series_out, series_records) = CompositionStrategy::Series
            .apply_with_flow(&projectors, &x, &ctx)
            .unwrap();

        let mut expected = x.clone();
        expected += &series_out;
        assert_eq!(residual_out, expected);
        assert_eq!(residual_records, series_records);
    }
}
