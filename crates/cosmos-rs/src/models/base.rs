//! Shared classification model infrastructure for COSMOS.
//!
//! This module holds the contract shared by the classification variants of
//! COSMOS together with the geodesic CMR scoring they terminate in. The
//! scoring lives here rather than in the arrangement because it is a property
//! of the emission head, not of the parameter-free routing layer.
//!
//! # Key items
//!
//! - [`Model`] — trait for models that produce embeddings and score them
//!   against CMR anchor blocks stored in [`CosmosState`].
//! - [`ModelError`] — typed model-layer errors wrapping state and arrangement
//!   failures.
//! - [`compute_cmr_scores_with_metrics_from_state`] — shared CMR scoring helper
//!   returning scores plus [`RegionMetrics`].

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

use crate::{
    arrangement::ArrangementError,
    compute::device::{Buffer2, ComputeDevice, Device},
    objectives::cmr_losses::RegionMetrics,
    state::cosmos_state::{CosmosState, CosmosStateError},
};

const DEFAULT_NORM_EPS: f64 = 1e-8;

/// Errors that can occur while building or executing COSMOS models.
#[derive(Debug, Error)]
pub enum ModelError {
    /// Wrap arrangement-layer failures.
    #[error("Arrangement error: {0}")]
    Arrangement(#[from] ArrangementError),

    /// Wrap state-layer failures.
    #[error("State error: {0}")]
    State(#[from] CosmosStateError),

    /// The model's CMR configuration is internally inconsistent.
    #[error("Invalid CMR configuration for `{field}`: {reason}")]
    InvalidCmrConfiguration {
        /// Name of the offending configuration field.
        field: &'static str,
        /// Why the value is rejected.
        reason: String,
    },

    /// A model parameter failed validation.
    #[error("Invalid parameter `{param}`: {reason}")]
    InvalidParameter {
        /// Name of the offending parameter.
        param: &'static str,
        /// Why the value is rejected.
        reason: String,
    },

    /// Input embeddings have a shape incompatible with the configured anchors.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Shape the operation requires.
        expected: String,
        /// Shape that was supplied.
        got: String,
    },

    /// A runtime-owned mutable component was already borrowed.
    #[error("Runtime state conflict in `{component}`: {reason}")]
    RuntimeStateConflict {
        /// Runtime component that could not be borrowed.
        component: &'static str,
        /// How the borrow conflicted.
        reason: &'static str,
    },
}

/// Contract shared by every COSMOS model.
///
/// A model composes an [`Arrangement`](crate::arrangement) over a
/// [`CosmosState`] and terminates in a CMR head. It owns no parameter of its
/// own: every learnable quantity it exposes is a block of the state it
/// borrows.
///
/// The trait provides default implementations for:
///
/// - `forward()` and `forward_with_metrics()`
/// - CMR anchor lookup and grouping
/// - geodesic-style CMR scoring over state-owned anchor blocks
/// - default parameter counting over the state container
pub trait Model {
    /// Raw input type accepted by this model variant.
    type Input;

    /// Borrow the unified COSMOS state.
    fn state(&self) -> &CosmosState;

    /// Mutably borrow the unified COSMOS state.
    fn state_mut(&mut self) -> &mut CosmosState;

    /// Return the block indices used as CMR anchors in class-major order.
    ///
    /// The layout is:
    ///
    /// - classes are contiguous
    /// - each class contributes `anchors_per_class()` anchor blocks
    fn cmr_block_indices(&self) -> &[usize];

    /// Number of output classes.
    fn num_classes(&self) -> usize;

    /// Number of anchors assigned to each class.
    fn anchors_per_class(&self) -> usize {
        1
    }

    /// Extract batch-major embeddings before CMR scoring.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, or when the interior runtime is already borrowed.
    fn get_embeddings(&self, x: &Self::Input) -> Result<Array2<f64>, ModelError>;

    /// Return a human-readable model summary.
    fn summary(&self) -> String;

    /// Run the full embedding + CMR scoring pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, when the interior runtime is already borrowed, or when the
    /// anchor layout is inconsistent.
    fn forward(&self, x: &Self::Input) -> Result<Array2<f64>, ModelError> {
        let embeddings = self.get_embeddings(x)?;
        self.compute_cmr_scores(&embeddings)
    }

    /// Run the full embedding + CMR scoring pipeline and retain region metrics.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the input shape does not match the configured
    /// input dimension, when the interior runtime is already borrowed, or when the
    /// anchor layout is inconsistent.
    fn forward_with_metrics(
        &self,
        x: &Self::Input,
    ) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
        let embeddings = self.get_embeddings(x)?;
        self.compute_cmr_scores_with_metrics(&embeddings)
    }

    /// Borrow the configured CMR anchors in class-major order.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout does
    /// not match `num_classes * anchors_per_class`, or [`ModelError::State`] when an
    /// anchor block index is out of range.
    fn get_cmr_anchors(&self) -> Result<Vec<&Array2<f64>>, ModelError> {
        validate_cmr_layout(
            self.cmr_block_indices(),
            self.num_classes(),
            self.anchors_per_class(),
        )?;

        self.cmr_block_indices()
            .iter()
            .map(|&block_idx| Ok(self.state().get_basis(block_idx)?))
            .collect()
    }

    /// Borrow the configured CMR anchors grouped by class.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout does
    /// not match `num_classes * anchors_per_class`, or [`ModelError::State`] when an
    /// anchor block index is out of range.
    fn get_cmr_anchors_grouped(&self) -> Result<Vec<Vec<&Array2<f64>>>, ModelError> {
        validate_cmr_layout(
            self.cmr_block_indices(),
            self.num_classes(),
            self.anchors_per_class(),
        )?;

        if self.num_classes() == 0 {
            return Ok(Vec::new());
        }

        let anchors = self.get_cmr_anchors()?;
        let anchors_per_class = self.anchors_per_class();

        Ok(anchors
            .chunks(anchors_per_class)
            .map(|chunk| chunk.to_vec())
            .collect())
    }

    /// Score pre-computed embeddings against the model's configured anchors.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout is
    /// inconsistent, [`ModelError::ShapeMismatch`] when the embeddings do not span
    /// the anchors' ambient dimension, or [`ModelError::State`] when a configured
    /// anchor block index does not exist in the state.
    fn compute_cmr_scores(&self, embeddings: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        compute_cmr_scores_from_state(
            embeddings,
            self.state(),
            self.cmr_block_indices(),
            self.num_classes(),
            self.anchors_per_class(),
        )
    }

    /// Score pre-computed embeddings and return per-class region metrics.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout is
    /// inconsistent, [`ModelError::ShapeMismatch`] when the embeddings do not span
    /// the anchors' ambient dimension, or [`ModelError::State`] when a configured
    /// anchor block index does not exist in the state.
    fn compute_cmr_scores_with_metrics(
        &self,
        embeddings: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
        compute_cmr_scores_with_metrics_from_state(
            embeddings,
            self.state(),
            self.cmr_block_indices(),
            self.num_classes(),
            self.anchors_per_class(),
        )
    }

    /// Count parameters by component.
    ///
    /// The default implementation counts only the unified state container.
    /// Concrete models can override this to break totals down by sub-pipeline.
    fn count_parameters(&self) -> HashMap<String, usize> {
        let state_total = count_state_parameters(self.state());

        HashMap::from([
            ("state_total".to_string(), state_total),
            ("total".to_string(), state_total),
        ])
    }
}

/// Count the number of stored scalar parameters in a [`CosmosState`].
pub fn count_state_parameters(state: &CosmosState) -> usize {
    state
        .block_shapes()
        .iter()
        .map(|(ambient_dim, subspace_dim)| ambient_dim * subspace_dim)
        .sum()
}

/// Compute CMR scores from state-owned anchor blocks.
///
/// # Errors
///
/// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout is
/// inconsistent, [`ModelError::ShapeMismatch`] when the embeddings do not span
/// the anchors' ambient dimension, or [`ModelError::State`] when a configured
/// anchor block index does not exist in the state.
pub fn compute_cmr_scores_from_state(
    embeddings: &Array2<f64>,
    state: &CosmosState,
    cmr_block_indices: &[usize],
    num_classes: usize,
    anchors_per_class: usize,
) -> Result<Array2<f64>, ModelError> {
    Ok(compute_cmr_scores_with_metrics_from_state(
        embeddings,
        state,
        cmr_block_indices,
        num_classes,
        anchors_per_class,
    )?
    .0)
}

/// Compute CMR scores and region metrics from state-owned anchor blocks.
///
/// The score for one embedding/anchor pair is:
///
/// `score = exp(-theta^2)`
///
/// where `theta` is the principal angle between the embedding vector and the
/// anchor subspace. When multiple anchors are assigned to a class, the maximum
/// score wins and its corresponding angle is retained in the output metrics.
///
/// # Errors
///
/// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout is
/// inconsistent, [`ModelError::ShapeMismatch`] when the embeddings do not span
/// the anchors' ambient dimension, or [`ModelError::State`] when a configured
/// anchor block index does not exist in the state.
pub fn compute_cmr_scores_with_metrics_from_state(
    embeddings: &Array2<f64>,
    state: &CosmosState,
    cmr_block_indices: &[usize],
    num_classes: usize,
    anchors_per_class: usize,
) -> Result<(Array2<f64>, Vec<RegionMetrics>), ModelError> {
    validate_cmr_layout(cmr_block_indices, num_classes, anchors_per_class)?;

    let batch_size = embeddings.nrows();
    if num_classes == 0 {
        return Ok((Array2::zeros((batch_size, 0)), Vec::new()));
    }

    let mut scores = Array2::<f64>::zeros((batch_size, num_classes));
    let mut best_angles = Array2::<f64>::zeros((batch_size, num_classes));

    for class_idx in 0..num_classes {
        let start = class_idx * anchors_per_class;
        let end = start + anchors_per_class;

        let mut best_class_scores = Array1::from_elem(batch_size, f64::NEG_INFINITY);
        let mut best_class_angles = Array1::<f64>::zeros(batch_size);

        for &block_idx in &cmr_block_indices[start..end] {
            let anchor = state.get_basis(block_idx)?;
            let (anchor_scores, anchor_angles) = compute_anchor_scores(embeddings, anchor)?;

            for row_idx in 0..batch_size {
                if anchor_scores[row_idx] > best_class_scores[row_idx] {
                    best_class_scores[row_idx] = anchor_scores[row_idx];
                    best_class_angles[row_idx] = anchor_angles[row_idx];
                }
            }
        }

        scores.column_mut(class_idx).assign(&best_class_scores);
        best_angles.column_mut(class_idx).assign(&best_class_angles);
    }

    let region_metrics = (0..num_classes)
        .map(|class_idx| RegionMetrics {
            principal_angles: best_angles
                .column(class_idx)
                .to_owned()
                .insert_axis(Axis(1)),
        })
        .collect();

    Ok((scores, region_metrics))
}

fn validate_cmr_layout(
    cmr_block_indices: &[usize],
    num_classes: usize,
    anchors_per_class: usize,
) -> Result<(), ModelError> {
    if anchors_per_class == 0 {
        return Err(ModelError::InvalidCmrConfiguration {
            field: "anchors_per_class",
            reason: "must be greater than zero".to_string(),
        });
    }

    let expected = num_classes
        .checked_mul(anchors_per_class)
        .ok_or_else(|| ModelError::InvalidCmrConfiguration {
            field: "cmr_block_indices",
            reason: format!(
                "num_classes * anchors_per_class overflows usize: {num_classes} * {anchors_per_class}"
            ),
        })?;

    if cmr_block_indices.len() != expected {
        return Err(ModelError::InvalidCmrConfiguration {
            field: "cmr_block_indices",
            reason: format!(
                "expected {expected} anchor indices for {num_classes} classes with {anchors_per_class} anchors each, got {}",
                cmr_block_indices.len()
            ),
        });
    }

    Ok(())
}

fn compute_anchor_scores(
    embeddings: &Array2<f64>,
    anchor: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), ModelError> {
    if embeddings.ncols() != anchor.nrows() {
        return Err(ModelError::ShapeMismatch {
            expected: format!("batch-major embeddings with {} columns", anchor.nrows()),
            got: format!("({}, {})", embeddings.nrows(), embeddings.ncols()),
        });
    }

    let coords = embeddings.dot(anchor);
    let projected_norms = Array1::from_iter(
        coords
            .outer_iter()
            .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt()),
    );
    let embedding_norms = Array1::from_iter(embeddings.outer_iter().map(|row| {
        row.iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(DEFAULT_NORM_EPS)
    }));

    let angles = Array1::from_iter(projected_norms.iter().zip(embedding_norms.iter()).map(
        |(projected_norm, embedding_norm)| {
            let cosine = (*projected_norm / *embedding_norm).clamp(0.0, 1.0);
            cosine.acos()
        },
    ));
    let scores = angles.mapv(|angle| (-(angle * angle)).exp());

    Ok((scores, angles))
}

/// Device-resident CMR scoring. Returns `(scores_cpu, scores_buf)`.
///
/// Uploads anchor bases from state, scores on device, downloads the small
/// scores matrix. For `anchors_per_class == 1` (the training fast path),
/// the result is direct — no per-class max reduction needed.
///
/// # Errors
///
/// Returns [`ModelError::InvalidCmrConfiguration`] when the anchor layout is
/// inconsistent, [`ModelError::ShapeMismatch`] when the embeddings do not span
/// the anchors' ambient dimension, or [`ModelError::State`] when a configured
/// anchor block index does not exist in the state.
pub fn compute_cmr_scores_device(
    device: &Device,
    embeddings: &Buffer2<f64>,
    state: &CosmosState,
    cmr_block_indices: &[usize],
    num_classes: usize,
    anchors_per_class: usize,
) -> Result<(Array2<f64>, Buffer2<f64>), ModelError> {
    validate_cmr_layout(cmr_block_indices, num_classes, anchors_per_class)?;

    let batch_size = embeddings.dim().0;
    if num_classes == 0 {
        let z = Array2::zeros((batch_size, 0));
        let zb = device.upload_2d(&z);
        return Ok((z, zb));
    }

    // Upload anchors (small: D × k each)
    let anchor_bufs: Vec<Buffer2<f64>> = cmr_block_indices
        .iter()
        .map(|&idx| {
            let basis = state.get_basis(idx).map_err(ModelError::from)?;
            Ok(device.upload_2d(basis))
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();

    let (all_scores_buf, _all_angles_buf) =
        device.compute_geodesic_scores(embeddings, &anchor_refs, DEFAULT_NORM_EPS);

    if anchors_per_class == 1 {
        // Each anchor IS one class — scores are already (N × num_classes)
        let scores_cpu = device.download_2d(&all_scores_buf);
        Ok((scores_cpu, all_scores_buf))
    } else {
        // Per-class max reduction on CPU (rare path — training always uses 1)
        let all_scores = device.download_2d(&all_scores_buf);
        let mut scores = Array2::<f64>::zeros((batch_size, num_classes));
        for class_idx in 0..num_classes {
            let start = class_idx * anchors_per_class;
            let end = start + anchors_per_class;
            for i in 0..batch_size {
                let best = (start..end)
                    .map(|j| all_scores[[i, j]])
                    .fold(f64::NEG_INFINITY, f64::max);
                scores[[i, class_idx]] = best;
            }
        }
        let scores_buf = device.upload_2d(&scores);
        Ok((scores, scores_buf))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    use ndarray::{Array1, array};

    use super::*;

    const EPSILON: f64 = 1e-10;

    #[derive(Debug)]
    struct DummyModel {
        state: CosmosState,
        cmr_block_indices: Vec<usize>,
        num_classes: usize,
        anchors_per_class: usize,
    }

    impl Model for DummyModel {
        type Input = Array2<f64>;

        fn state(&self) -> &CosmosState {
            &self.state
        }

        fn state_mut(&mut self) -> &mut CosmosState {
            &mut self.state
        }

        fn cmr_block_indices(&self) -> &[usize] {
            &self.cmr_block_indices
        }

        fn num_classes(&self) -> usize {
            self.num_classes
        }

        fn anchors_per_class(&self) -> usize {
            self.anchors_per_class
        }

        fn get_embeddings(&self, x: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
            Ok(x.clone())
        }

        fn summary(&self) -> String {
            "DummyModel".to_string()
        }
    }

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
    fn get_cmr_anchors_returns_configured_order() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis(), diagonal_xy()]),
            cmr_block_indices: vec![2, 0],
            num_classes: 2,
            anchors_per_class: 1,
        };

        let anchors = model.get_cmr_anchors().unwrap();

        assert_eq!(anchors.len(), 2);
        assert_array2_close(anchors[0], &diagonal_xy());
        assert_array2_close(anchors[1], &x_axis());
    }

    #[test]
    fn get_cmr_anchors_grouped_respects_class_layout() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis(), diagonal_xy(), x_axis()]),
            cmr_block_indices: vec![0, 2, 1, 3],
            num_classes: 2,
            anchors_per_class: 2,
        };

        let grouped = model.get_cmr_anchors_grouped().unwrap();

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].len(), 2);
        assert_eq!(grouped[1].len(), 2);
        assert_array2_close(grouped[0][0], &x_axis());
        assert_array2_close(grouped[0][1], &diagonal_xy());
        assert_array2_close(grouped[1][0], &y_axis());
        assert_array2_close(grouped[1][1], &x_axis());
    }

    #[test]
    fn compute_cmr_scores_matches_known_axis_alignment() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis()]),
            cmr_block_indices: vec![0, 1],
            num_classes: 2,
            anchors_per_class: 1,
        };
        let embeddings = array![[1.0, 0.0], [0.0, 1.0]];
        let off_axis = (-(FRAC_PI_2 * FRAC_PI_2)).exp();

        let scores = model.compute_cmr_scores(&embeddings).unwrap();

        assert_array2_close(&scores, &array![[1.0, off_axis], [off_axis, 1.0]]);
    }

    #[test]
    fn compute_cmr_scores_with_metrics_tracks_best_angles() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis()]),
            cmr_block_indices: vec![0, 1],
            num_classes: 2,
            anchors_per_class: 1,
        };
        let embeddings = array![[1.0, 0.0], [1.0, 1.0]];

        let (_scores, metrics) = model.compute_cmr_scores_with_metrics(&embeddings).unwrap();

        assert_eq!(metrics.len(), 2);
        assert_array2_close(&metrics[0].principal_angles, &array![[0.0], [FRAC_PI_4]]);
        assert_array2_close(
            &metrics[1].principal_angles,
            &array![[FRAC_PI_2], [FRAC_PI_4]],
        );
    }

    #[test]
    fn multi_anchor_cmr_uses_closest_anchor_per_class() {
        let model = DummyModel {
            state: build_state(&[diagonal_xy(), x_axis(), y_axis(), diagonal_xy()]),
            cmr_block_indices: vec![0, 1, 2, 3],
            num_classes: 2,
            anchors_per_class: 2,
        };
        let embeddings = array![[1.0, 0.0]];

        let (scores, metrics) = model.compute_cmr_scores_with_metrics(&embeddings).unwrap();

        assert!((scores[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((scores[[0, 1]] - (-(FRAC_PI_4 * FRAC_PI_4)).exp()).abs() < EPSILON);
        assert!((metrics[0].principal_angles[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((metrics[1].principal_angles[[0, 0]] - FRAC_PI_4).abs() < EPSILON);
    }

    #[test]
    fn forward_with_metrics_uses_default_pipeline() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis()]),
            cmr_block_indices: vec![0, 1],
            num_classes: 2,
            anchors_per_class: 1,
        };
        let embeddings = array![[1.0, 0.0], [1.0, 1.0]];

        let direct = model.compute_cmr_scores_with_metrics(&embeddings).unwrap();
        let forwarded = model.forward_with_metrics(&embeddings).unwrap();

        assert_array2_close(&forwarded.0, &direct.0);
        assert_eq!(forwarded.1.len(), direct.1.len());
        for (lhs, rhs) in forwarded.1.iter().zip(direct.1.iter()) {
            assert_array2_close(&lhs.principal_angles, &rhs.principal_angles);
        }
    }

    #[test]
    fn count_parameters_defaults_to_state_total() {
        let model = DummyModel {
            state: build_state(&[x_axis(), y_axis(), diagonal_xy()]),
            cmr_block_indices: vec![0, 1, 2],
            num_classes: 3,
            anchors_per_class: 1,
        };

        let counts = model.count_parameters();

        assert_eq!(counts["state_total"], 6);
        assert_eq!(counts["total"], 6);
    }

    #[test]
    fn count_state_parameters_sums_block_shapes() {
        let state = build_state(&[x_axis(), array![[1.0, 0.0], [0.0, 1.0]]]);

        assert_eq!(count_state_parameters(&state), 6);
    }

    #[test]
    fn compute_scores_rejects_invalid_cmr_layout() {
        let state = build_state(&[x_axis(), y_axis()]);
        let embeddings = array![[1.0, 0.0]];

        let err = compute_cmr_scores_from_state(&embeddings, &state, &[0], 2, 1).unwrap_err();

        match err {
            ModelError::InvalidCmrConfiguration { field, .. } => {
                assert_eq!(field, "cmr_block_indices");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn grouped_anchors_reject_zero_anchors_per_class() {
        let model = DummyModel {
            state: build_state(&[x_axis()]),
            cmr_block_indices: Vec::new(),
            num_classes: 0,
            anchors_per_class: 0,
        };

        let err = model.get_cmr_anchors_grouped().unwrap_err();

        match err {
            ModelError::InvalidCmrConfiguration { field, .. } => {
                assert_eq!(field, "anchors_per_class");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn compute_scores_rejects_embedding_dimension_mismatch() {
        let state = build_state(&[x_axis()]);
        let embeddings = array![[1.0, 0.0, 0.0]];

        let err = compute_cmr_scores_from_state(&embeddings, &state, &[0], 1, 1).unwrap_err();

        match err {
            ModelError::ShapeMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn compute_scores_on_zero_vector_fall_back_to_right_angle() {
        let state = build_state(&[x_axis()]);
        let embeddings = array![[0.0, 0.0]];
        let expected = (-(FRAC_PI_2 * FRAC_PI_2)).exp();

        let scores = compute_cmr_scores_from_state(&embeddings, &state, &[0], 1, 1).unwrap();

        assert_array1_close(&scores.column(0).to_owned(), &array![expected]);
    }

    #[test]
    fn device_scoring_matches_cpu_scoring() {
        use crate::compute::device::{ComputeDevice, Device};

        let state = build_state(&[
            x_axis(),      // block 0
            y_axis(),      // block 1
            diagonal_xy(), // block 2
        ]);

        let embeddings = array![[1.0, 0.0], [0.707, 0.707], [0.0, 1.0], [0.5, 0.866],];
        let cmr_indices = vec![1, 2];

        let (cpu_scores, _) =
            compute_cmr_scores_with_metrics_from_state(&embeddings, &state, &cmr_indices, 2, 1)
                .unwrap();

        let device = Device::default();
        let emb_buf = device.upload_2d(&embeddings);
        let (device_scores, _) =
            compute_cmr_scores_device(&device, &emb_buf, &state, &cmr_indices, 2, 1).unwrap();

        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (cpu_scores[[i, j]] - device_scores[[i, j]]).abs() < 1e-12,
                    "mismatch at [{}, {}]: cpu={} device={}",
                    i,
                    j,
                    cpu_scores[[i, j]],
                    device_scores[[i, j]]
                );
            }
        }
    }
}
