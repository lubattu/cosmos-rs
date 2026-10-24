//! Geometric output heads over state-owned projector blocks.
//!
//! The functional API operates on batch-major arrays and avoids materializing
//! ambient projector matrices. The struct wrappers provide a thin reusable
//! surface for classification heads without taking ownership of
//! [`CosmosState`].

use std::{fmt, str::FromStr};

use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    arrangement::{ArrangementError, utils::get_w_blocks},
    state::cosmos_state::CosmosState,
};

const DEFAULT_EPS: f64 = 1e-8;

/// Score family used by the geometric logit heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeometricScoreMode {
    /// Quadratic energy `||x W||^2`.
    Quadratic,
    /// Rank-1 lift distance `1 + k - 2 ||u W||^2`.
    Rank1Distance,
}

impl fmt::Display for GeometricScoreMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quadratic => write!(f, "quadratic"),
            Self::Rank1Distance => write!(f, "rank1_distance"),
        }
    }
}

impl FromStr for GeometricScoreMode {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "quadratic" => Ok(Self::Quadratic),
            "rank1_distance" | "rank1" | "distance" => Ok(Self::Rank1Distance),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "geometric_score_mode",
                name: s.to_string(),
            }),
        }
    }
}

/// Compute quadratic energy `||x W||^2` for one projector over a batch.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn quadratic_energy(x: &Array2<f64>, w: &Array2<f64>) -> Result<Array1<f64>, ArrangementError> {
    quadratic_energy_view(x, w.view())
}

/// Compute quadratic energies for multiple projectors.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn quadratic_energies_batched(
    x: &Array2<f64>,
    w_list: &[Array2<f64>],
) -> Result<Array2<f64>, ArrangementError> {
    if w_list.is_empty() {
        return Ok(empty_logits(x.nrows(), 0));
    }

    let mut energies = Array2::<f64>::zeros((x.nrows(), w_list.len()));
    for (col_idx, w) in w_list.iter().enumerate() {
        let values = quadratic_energy(x, w)?;
        energies.column_mut(col_idx).assign(&values);
    }

    Ok(energies)
}

/// Compute `log(e_pos + eps) - log(e_neg + eps)` for one sensor pair.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn differential_log_energy_ratio(
    x: &Array2<f64>,
    w_pos: &Array2<f64>,
    w_neg: &Array2<f64>,
    eps: f64,
) -> Result<Array1<f64>, ArrangementError> {
    validate_eps(eps)?;

    let e_pos = quadratic_energy(x, w_pos)?.mapv(|value| value.max(eps).ln());
    let e_neg = quadratic_energy(x, w_neg)?.mapv(|value| value.max(eps).ln());

    Ok(&e_pos - &e_neg)
}

/// Compute log energy ratios for multiple positive/negative sensor pairs.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn differential_log_energy_ratio_batched(
    x: &Array2<f64>,
    w_pos_list: &[Array2<f64>],
    w_neg_list: &[Array2<f64>],
    eps: f64,
) -> Result<Array2<f64>, ArrangementError> {
    validate_eps(eps)?;

    if w_pos_list.len() != w_neg_list.len() {
        return Err(ArrangementError::InvalidParameter {
            param: "w_lists",
            reason: format!(
                "positive and negative projector lists must have the same length, got {} and {}",
                w_pos_list.len(),
                w_neg_list.len()
            ),
        });
    }

    if w_pos_list.is_empty() {
        return Ok(empty_logits(x.nrows(), 0));
    }

    let e_pos = quadratic_energies_batched(x, w_pos_list)?.mapv(|value| value.max(eps).ln());
    let e_neg = quadratic_energies_batched(x, w_neg_list)?.mapv(|value| value.max(eps).ln());

    Ok(&e_pos - &e_neg)
}

/// Compute the rank-1 lift distance `1 + k - 2 ||u W||^2`.
///
/// `u` is obtained by row-wise normalization with denominator `||x|| + eps`.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn rank1_lift_distance(
    x: &Array2<f64>,
    w: &Array2<f64>,
    eps: f64,
) -> Result<Array1<f64>, ArrangementError> {
    rank1_lift_distance_view(x, w.view(), eps)
}

/// Compute rank-1 lift distances for multiple projectors.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn rank1_distances_batched(
    x: &Array2<f64>,
    w_list: &[Array2<f64>],
    eps: f64,
) -> Result<Array2<f64>, ArrangementError> {
    validate_eps(eps)?;

    if w_list.is_empty() {
        return Ok(empty_logits(x.nrows(), 0));
    }

    let mut distances = Array2::<f64>::zeros((x.nrows(), w_list.len()));
    for (col_idx, w) in w_list.iter().enumerate() {
        let values = rank1_lift_distance(x, w, eps)?;
        distances.column_mut(col_idx).assign(&values);
    }

    Ok(distances)
}

/// Compute quadratic energies from state-owned blocks.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn quadratic_energies_from_blocks(
    x: &Array2<f64>,
    state: &CosmosState,
    block_indices: Option<&[usize]>,
) -> Result<Array2<f64>, ArrangementError> {
    let block_indices = resolved_block_indices(state, block_indices);
    let w_list = get_w_blocks(state, &block_indices)?;
    quadratic_energies_batched(x, &w_list)
}

/// Compute rank-1 distances from state-owned blocks.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn rank1_distances_from_blocks(
    x: &Array2<f64>,
    state: &CosmosState,
    block_indices: Option<&[usize]>,
    eps: f64,
) -> Result<Array2<f64>, ArrangementError> {
    let block_indices = resolved_block_indices(state, block_indices);
    let w_list = get_w_blocks(state, &block_indices)?;
    rank1_distances_batched(x, &w_list, eps)
}

/// Compute quadratic energies against anchor subspaces.
///
/// `anchor_w` must have shape `(num_anchors, ambient_dim, subspace_dim)`.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn quadratic_energies_from_anchors(
    x: &Array2<f64>,
    anchor_w: &Array3<f64>,
) -> Result<Array2<f64>, ArrangementError> {
    let (num_anchors, _, _) = anchor_w.dim();
    if num_anchors == 0 {
        return Ok(empty_logits(x.nrows(), 0));
    }

    let mut energies = Array2::<f64>::zeros((x.nrows(), num_anchors));
    for anchor_idx in 0..num_anchors {
        let anchor = anchor_w.index_axis(Axis(0), anchor_idx);
        let values = quadratic_energy_view(x, anchor)?;
        energies.column_mut(anchor_idx).assign(&values);
    }

    Ok(energies)
}

/// Compute rank-1 lift distances against anchor subspaces.
///
/// # Errors
///
/// Returns an [`ArrangementError`] when the operand shapes do not line up or a
/// parameter fails validation.
pub fn rank1_distances_from_anchors(
    x: &Array2<f64>,
    anchor_w: &Array3<f64>,
    eps: f64,
) -> Result<Array2<f64>, ArrangementError> {
    validate_eps(eps)?;

    let (num_anchors, _, _) = anchor_w.dim();
    if num_anchors == 0 {
        return Ok(empty_logits(x.nrows(), 0));
    }

    let mut distances = Array2::<f64>::zeros((x.nrows(), num_anchors));
    for anchor_idx in 0..num_anchors {
        let anchor = anchor_w.index_axis(Axis(0), anchor_idx);
        let values = rank1_lift_distance_view(x, anchor, eps)?;
        distances.column_mut(anchor_idx).assign(&values);
    }

    Ok(distances)
}

/// Parameter-free classification head over state-owned blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricLogitHead {
    block_indices: Vec<usize>,
    mode: GeometricScoreMode,
    negate_distances: bool,
    eps: f64,
}

impl GeometricLogitHead {
    /// Create a logit head over the provided block indices.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn new(
        block_indices: Vec<usize>,
        mode: GeometricScoreMode,
        negate_distances: bool,
        eps: f64,
    ) -> Result<Self, ArrangementError> {
        validate_eps(eps)?;

        Ok(Self {
            block_indices,
            mode,
            negate_distances,
            eps,
        })
    }

    /// Create a logit head spanning all current state blocks.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn all_blocks(
        state: &CosmosState,
        mode: GeometricScoreMode,
        negate_distances: bool,
        eps: f64,
    ) -> Result<Self, ArrangementError> {
        Self::new(
            (0..state.num_blocks()).collect(),
            mode,
            negate_distances,
            eps,
        )
    }

    /// Selected block indices in class order.
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }

    /// Scoring family used by this head.
    pub fn mode(&self) -> GeometricScoreMode {
        self.mode
    }

    /// Whether rank-1 distances are negated before being returned as logits.
    pub fn negate_distances(&self) -> bool {
        self.negate_distances
    }

    /// Numerical stability constant used by rank-1 normalization.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Number of output classes.
    pub fn num_classes(&self) -> usize {
        self.block_indices.len()
    }

    /// Compute logits for a batch-major input.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(
        &self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        match self.mode {
            GeometricScoreMode::Quadratic => {
                quadratic_energies_from_blocks(x, state, Some(&self.block_indices))
            }
            GeometricScoreMode::Rank1Distance => {
                let distances =
                    rank1_distances_from_blocks(x, state, Some(&self.block_indices), self.eps)?;
                if self.negate_distances {
                    Ok(distances.mapv(|value| -value))
                } else {
                    Ok(distances)
                }
            }
        }
    }

    /// Compute logits for one input sample.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_one(
        &self,
        x: &Array1<f64>,
        state: &CosmosState,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

/// Parameter-free classification head over a fixed anchor tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorGeometricLogitHead {
    anchor_w: Array3<f64>,
    mode: GeometricScoreMode,
    negate_distances: bool,
    eps: f64,
}

impl AnchorGeometricLogitHead {
    /// Create a logit head over a fixed set of anchor subspaces.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn new(
        anchor_w: Array3<f64>,
        mode: GeometricScoreMode,
        negate_distances: bool,
        eps: f64,
    ) -> Result<Self, ArrangementError> {
        validate_eps(eps)?;

        Ok(Self {
            anchor_w,
            mode,
            negate_distances,
            eps,
        })
    }

    /// Borrow the anchor tensor of shape `(num_anchors, ambient_dim, subspace_dim)`.
    pub fn anchors(&self) -> &Array3<f64> {
        &self.anchor_w
    }

    /// Scoring family used by this head.
    pub fn mode(&self) -> GeometricScoreMode {
        self.mode
    }

    /// Whether rank-1 distances are negated before being returned as logits.
    pub fn negate_distances(&self) -> bool {
        self.negate_distances
    }

    /// Numerical stability constant used by rank-1 normalization.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Number of anchor classes.
    pub fn num_classes(&self) -> usize {
        self.anchor_w.dim().0
    }

    /// Ambient input dimension expected by the anchors.
    pub fn input_dim(&self) -> usize {
        self.anchor_w.dim().1
    }

    /// Anchor subspace dimension.
    pub fn subspace_dim(&self) -> usize {
        self.anchor_w.dim().2
    }

    /// Compute logits for a batch-major input.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>, ArrangementError> {
        match self.mode {
            GeometricScoreMode::Quadratic => quadratic_energies_from_anchors(x, &self.anchor_w),
            GeometricScoreMode::Rank1Distance => {
                let distances = rank1_distances_from_anchors(x, &self.anchor_w, self.eps)?;
                if self.negate_distances {
                    Ok(distances.mapv(|value| -value))
                } else {
                    Ok(distances)
                }
            }
        }
    }

    /// Compute logits for one sample.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_one(&self, x: &Array1<f64>) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward(&batch)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

fn empty_logits(batch: usize, cols: usize) -> Array2<f64> {
    Array2::zeros((batch, cols))
}

fn quadratic_energy_view(
    x: &Array2<f64>,
    w: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, ArrangementError> {
    ensure_input_dim(x, w.nrows(), "quadratic_energy")?;

    let coords = x.dot(&w);
    Ok(Array1::from_iter(coords.outer_iter().map(|row| {
        row.iter().map(|value| value * value).sum::<f64>()
    })))
}

fn rank1_lift_distance_view(
    x: &Array2<f64>,
    w: ArrayView2<'_, f64>,
    eps: f64,
) -> Result<Array1<f64>, ArrangementError> {
    validate_eps(eps)?;
    ensure_input_dim(x, w.nrows(), "rank1_lift_distance")?;

    let normalized = normalize_rows_with_eps(x, eps);
    let coords = normalized.dot(&w);
    let k = w.ncols() as f64;

    Ok(Array1::from_iter(coords.outer_iter().map(|row| {
        let alignment = row.iter().map(|value| value * value).sum::<f64>();
        1.0 + k - 2.0 * alignment
    })))
}

fn resolved_block_indices(state: &CosmosState, block_indices: Option<&[usize]>) -> Vec<usize> {
    block_indices
        .map(|indices| indices.to_vec())
        .unwrap_or_else(|| (0..state.num_blocks()).collect())
}

fn ensure_input_dim(
    x: &Array2<f64>,
    expected: usize,
    context: &'static str,
) -> Result<(), ArrangementError> {
    if x.ncols() != expected {
        return Err(ArrangementError::ShapeMismatch {
            expected: format!("{context} input with {expected} columns"),
            got: format!("({}, {})", x.nrows(), x.ncols()),
        });
    }

    Ok(())
}

fn normalize_rows_with_eps(x: &Array2<f64>, eps: f64) -> Array2<f64> {
    let mut normalized = Array2::<f64>::zeros(x.dim());

    for (row_idx, row) in x.outer_iter().enumerate() {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        let denom = norm + eps;

        for (col_idx, value) in row.iter().enumerate() {
            normalized[(row_idx, col_idx)] = *value / denom;
        }
    }

    normalized
}

fn validate_eps(eps: f64) -> Result<(), ArrangementError> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(ArrangementError::InvalidParameter {
            param: "eps",
            reason: format!("must be finite and > 0, got {eps}"),
        });
    }

    Ok(())
}

impl Default for GeometricLogitHead {
    fn default() -> Self {
        Self {
            block_indices: Vec::new(),
            mode: GeometricScoreMode::Quadratic,
            negate_distances: true,
            eps: DEFAULT_EPS,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*;

    const EPSILON: f64 = 1e-7;

    fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn x_axis() -> Array2<f64> {
        array![[1.0], [0.0]]
    }

    fn y_axis() -> Array2<f64> {
        array![[0.0], [1.0]]
    }

    fn diagonal_xy() -> Array2<f64> {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        array![[inv_sqrt2], [inv_sqrt2]]
    }

    fn build_state(blocks: &[Array2<f64>]) -> CosmosState {
        let mut state = CosmosState::new(&[]).unwrap();
        for block in blocks {
            state.register_external_block(block).unwrap();
        }
        state
    }

    #[test]
    fn score_mode_display_roundtrip() {
        for mode in [
            GeometricScoreMode::Quadratic,
            GeometricScoreMode::Rank1Distance,
        ] {
            let encoded = mode.to_string();
            assert_eq!(encoded.parse::<GeometricScoreMode>().unwrap(), mode);
        }
    }

    #[test]
    fn score_mode_from_str_rejects_unknown_value() {
        assert!("cosine".parse::<GeometricScoreMode>().is_err());
    }

    #[test]
    fn quadratic_energy_matches_known_axis_projection() {
        let x = array![[3.0, 4.0], [0.0, 2.0]];
        let out = quadratic_energy(&x, &x_axis()).unwrap();

        assert_array1_close(&out, &array![9.0, 0.0]);
    }

    #[test]
    fn quadratic_energy_rejects_mismatched_input_dimension() {
        let x = array![[1.0, 2.0, 3.0]];
        let err = quadratic_energy(&x, &x_axis()).unwrap_err();

        match err {
            ArrangementError::ShapeMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn quadratic_energies_batched_preserves_projector_order() {
        let x = array![[3.0, 4.0]];
        let w_list = vec![y_axis(), x_axis()];

        let out = quadratic_energies_batched(&x, &w_list).unwrap();

        assert_array2_close(&out, &array![[16.0, 9.0]]);
    }

    #[test]
    fn differential_log_energy_ratio_matches_known_value() {
        let x = array![[2.0, 1.0]];
        let out = differential_log_energy_ratio(&x, &x_axis(), &y_axis(), DEFAULT_EPS).unwrap();

        assert!((out[0] - 4.0_f64.ln()).abs() < EPSILON);
    }

    #[test]
    fn differential_log_energy_ratio_batched_rejects_length_mismatch() {
        let x = array![[1.0, 2.0]];
        let err = differential_log_energy_ratio_batched(
            &x,
            &[x_axis(), y_axis()],
            &[x_axis()],
            DEFAULT_EPS,
        )
        .unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => assert_eq!(param, "w_lists"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rank1_lift_distance_matches_known_axis_distances() {
        let x = array![[3.0, 4.0], [0.0, 2.0]];
        let out = rank1_lift_distance(&x, &x_axis(), DEFAULT_EPS).unwrap();

        assert_array1_close(&out, &array![1.28, 2.0]);
    }

    #[test]
    fn rank1_distances_batched_support_mixed_subspace_dims() {
        let x = array![[1.0, 0.0]];
        let w_list = vec![x_axis(), array![[1.0, 0.0], [0.0, 1.0]]];

        let out = rank1_distances_batched(&x, &w_list, DEFAULT_EPS).unwrap();

        assert_array2_close(&out, &array![[0.0, 1.0]]);
    }

    #[test]
    fn quadratic_energies_from_blocks_uses_requested_order() {
        let state = build_state(&[x_axis(), y_axis(), diagonal_xy()]);
        let x = array![[3.0, 4.0]];

        let out = quadratic_energies_from_blocks(&x, &state, Some(&[1, 0])).unwrap();

        assert_array2_close(&out, &array![[16.0, 9.0]]);
    }

    #[test]
    fn rank1_distances_from_blocks_uses_all_blocks_when_indices_are_none() {
        let state = build_state(&[x_axis(), y_axis()]);
        let x = array![[1.0, 0.0]];

        let out = rank1_distances_from_blocks(&x, &state, None, DEFAULT_EPS).unwrap();

        assert_array2_close(&out, &array![[0.0, 2.0]]);
    }

    #[test]
    fn quadratic_energies_from_anchors_matches_expected_values() {
        let anchors = array![[[1.0], [0.0]], [[0.0], [1.0]]];
        let x = array![[1.0, 2.0]];

        let out = quadratic_energies_from_anchors(&x, &anchors).unwrap();

        assert_array2_close(&out, &array![[1.0, 4.0]]);
    }

    #[test]
    fn rank1_distances_from_anchors_matches_expected_values() {
        let anchors = array![[[1.0], [0.0]], [[0.0], [1.0]]];
        let x = array![[1.0, 0.0]];

        let out = rank1_distances_from_anchors(&x, &anchors, DEFAULT_EPS).unwrap();

        assert_array2_close(&out, &array![[0.0, 2.0]]);
    }

    #[test]
    fn geometric_logit_head_quadratic_matches_functional_api() {
        let state = build_state(&[x_axis(), y_axis()]);
        let head =
            GeometricLogitHead::new(vec![0, 1], GeometricScoreMode::Quadratic, true, DEFAULT_EPS)
                .unwrap();
        let x = array![[1.0, 2.0]];

        let out = head.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 4.0]]);
    }

    #[test]
    fn geometric_logit_head_rank1_can_negate_distances() {
        let state = build_state(&[x_axis(), y_axis()]);
        let head = GeometricLogitHead::new(
            vec![0, 1],
            GeometricScoreMode::Rank1Distance,
            true,
            DEFAULT_EPS,
        )
        .unwrap();
        let x = array![[1.0, 0.0]];

        let out = head.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[0.0, -2.0]]);
    }

    #[test]
    fn geometric_logit_head_forward_one_matches_batch_forward() {
        let state = build_state(&[x_axis(), y_axis()]);
        let head =
            GeometricLogitHead::new(vec![0, 1], GeometricScoreMode::Quadratic, true, DEFAULT_EPS)
                .unwrap();
        let x = array![1.0, 2.0];

        let single = head.forward_one(&x, &state).unwrap();
        let batch = head
            .forward(&x.clone().insert_axis(Axis(0)), &state)
            .unwrap();

        assert_array1_close(&single, &batch.index_axis(Axis(0), 0).to_owned());
    }

    #[test]
    fn anchor_geometric_logit_head_matches_functional_api() {
        let anchors = array![[[1.0], [0.0]], [[0.0], [1.0]]];
        let head = AnchorGeometricLogitHead::new(
            anchors.clone(),
            GeometricScoreMode::Quadratic,
            true,
            DEFAULT_EPS,
        )
        .unwrap();
        let x = array![[1.0, 2.0]];

        let out = head.forward(&x).unwrap();

        assert_array2_close(
            &out,
            &quadratic_energies_from_anchors(&x, &anchors).unwrap(),
        );
    }

    #[test]
    fn anchor_geometric_logit_head_forward_one_matches_batch_forward() {
        let anchors = array![[[1.0], [0.0]], [[0.0], [1.0]]];
        let head = AnchorGeometricLogitHead::new(
            anchors,
            GeometricScoreMode::Rank1Distance,
            true,
            DEFAULT_EPS,
        )
        .unwrap();
        let x = array![1.0, 0.0];

        let single = head.forward_one(&x).unwrap();
        let batch = head.forward(&x.clone().insert_axis(Axis(0))).unwrap();

        assert_array1_close(&single, &batch.index_axis(Axis(0), 0).to_owned());
    }

    #[test]
    fn head_constructors_reject_invalid_eps() {
        assert!(
            GeometricLogitHead::new(vec![0], GeometricScoreMode::Quadratic, true, 0.0).is_err()
        );
        assert!(
            AnchorGeometricLogitHead::new(
                Array3::zeros((1, 2, 1)),
                GeometricScoreMode::Quadratic,
                true,
                f64::NAN
            )
            .is_err()
        );
    }
}
