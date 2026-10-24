//! Minimal geometry diagnostics needed for the E1 paper experiment.
//!
//! This module intentionally implements only the notebook-critical subset of
//! the Python diagnostics surface:
//!
//! - `snapshot_initial_blocks`
//! - `compute_block_geometry`
//!
//! The returned metrics match the first experiment's invariant tracking needs:
//! per-block orthogonality error, geodesic drift from initialization, and the
//! most recent commutator norms cached by [`GaugeDynamics`].

use ndarray::Array2;
use thiserror::Error;

use crate::{
    geometry::grassmann::{GrassmannError, principal_angles},
    models::Model,
    rules::GaugeDynamics,
    state::cosmos_state::{CosmosState, CosmosStateError},
};

/// Minimal geometric diagnostics used by the first paper experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockGeometryDiagnostics {
    /// Per-block orthogonality error `||W_i^T W_i - I||_F`.
    pub orth_error: Vec<f64>,
    /// Per-block Grassmann geodesic drift from initialization `||theta_i||_2`.
    pub drift: Vec<f64>,
    /// Per-block commutator norms cached by the most recent evolution step.
    pub comm_norm: Vec<f64>,
}

/// Errors returned by the minimal geometry diagnostics helpers.
#[derive(Debug, Error)]
pub enum GeometryDiagnosticsError {
    /// Wrap state-layer access failures.
    #[error("State error: {0}")]
    State(#[from] CosmosStateError),

    /// Wrap Grassmann-geometry failures during drift computation.
    #[error("Grassmann error: {0}")]
    Grassmann(#[from] GrassmannError),

    /// The initial snapshot length does not match the current state.
    #[error("Initial block count mismatch: expected {expected}, got {got}")]
    InitialBlockCountMismatch {
        /// Number of blocks in the current state.
        expected: usize,
        /// Number of blocks in the initial snapshot.
        got: usize,
    },

    /// The evolution-law cache is configured for a different block count.
    #[error("Evolution-law block count mismatch: expected {expected}, got {got}")]
    EvolutionLawBlockCountMismatch {
        /// Number of blocks in the current state.
        expected: usize,
        /// Number of blocks the evolution law is configured for.
        got: usize,
    },

    /// One of the cached initial blocks has the wrong shape.
    #[error(
        "Initial block {block} shape mismatch: expected ({expected_rows}, {expected_cols}), got ({got_rows}, {got_cols})"
    )]
    InitialBlockShapeMismatch {
        /// Index of the offending block.
        block: usize,
        /// Expected row count, taken from the current state.
        expected_rows: usize,
        /// Expected column count, taken from the current state.
        expected_cols: usize,
        /// Row count found in the initial snapshot.
        got_rows: usize,
        /// Column count found in the initial snapshot.
        got_cols: usize,
    },
}

/// Clone the current model blocks for later drift measurement.
pub fn snapshot_initial_blocks<M: Model>(model: &M) -> Vec<Array2<f64>> {
    snapshot_initial_blocks_from_state(model.state())
}

/// Clone the current state blocks for later drift measurement.
///
/// # Panics
///
/// Never panics: every index iterated comes from `state.block_shapes()`,
/// so each `get_basis` lookup succeeds.
pub fn snapshot_initial_blocks_from_state(state: &CosmosState) -> Vec<Array2<f64>> {
    state
        .block_shapes()
        .iter()
        .enumerate()
        .map(|(block_idx, _)| {
            state
                .get_basis(block_idx)
                .expect("state block indices derived from block_shapes must be valid")
                .clone()
        })
        .collect()
}

/// Compute the minimal block-geometry diagnostics used by the E1 experiment.
///
/// # Errors
///
/// Returns an error when `init_blocks` or `evolution_law` disagree with
/// the model's current block count or shapes, or when the underlying
/// Grassmann-geometry computation fails.
pub fn compute_block_geometry<M: Model>(
    model: &M,
    init_blocks: &[Array2<f64>],
    evolution_law: &GaugeDynamics,
) -> Result<BlockGeometryDiagnostics, GeometryDiagnosticsError> {
    compute_block_geometry_from_state(model.state(), init_blocks, evolution_law)
}

/// Compute minimal block-geometry diagnostics directly from a state snapshot.
///
/// # Errors
///
/// Returns an error when `init_blocks` or `evolution_law` disagree with
/// the state's current block count or shapes, or when the underlying
/// Grassmann-geometry computation fails.
pub fn compute_block_geometry_from_state(
    state: &CosmosState,
    init_blocks: &[Array2<f64>],
    evolution_law: &GaugeDynamics,
) -> Result<BlockGeometryDiagnostics, GeometryDiagnosticsError> {
    let num_blocks = state.num_blocks();
    if init_blocks.len() != num_blocks {
        return Err(GeometryDiagnosticsError::InitialBlockCountMismatch {
            expected: num_blocks,
            got: init_blocks.len(),
        });
    }
    if evolution_law.num_blocks() != num_blocks {
        return Err(GeometryDiagnosticsError::EvolutionLawBlockCountMismatch {
            expected: num_blocks,
            got: evolution_law.num_blocks(),
        });
    }

    let orth_error = state.orthogonality_errors();
    let mut drift = Vec::with_capacity(num_blocks);

    for (block_idx, init_block) in init_blocks.iter().enumerate() {
        let current = state.get_basis(block_idx)?;
        let expected = current.dim();
        let got = init_block.dim();

        if got != expected {
            return Err(GeometryDiagnosticsError::InitialBlockShapeMismatch {
                block: block_idx,
                expected_rows: expected.0,
                expected_cols: expected.1,
                got_rows: got.0,
                got_cols: got.1,
            });
        }

        let angles = principal_angles(init_block, current)?;
        let block_drift = angles.iter().map(|theta| theta * theta).sum::<f64>().sqrt();
        drift.push(block_drift);
    }

    let comm_norm = evolution_law.get_diagnostics(state).commutator_norms;

    Ok(BlockGeometryDiagnostics {
        orth_error,
        drift,
        comm_norm,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use ndarray::{Array2, array};

    use super::*;
    use crate::{
        models::ModelError,
        rules::{GaugeDynamicsConfig, RulesError},
    };

    const EPSILON: f64 = 1e-10;

    #[derive(Debug)]
    struct DummyModel {
        state: CosmosState,
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
            &[]
        }

        fn num_classes(&self) -> usize {
            0
        }

        fn get_embeddings(&self, x: &Self::Input) -> Result<Array2<f64>, ModelError> {
            Ok(x.clone())
        }

        fn summary(&self) -> String {
            "DummyModel".to_string()
        }
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() < EPSILON,
            "expected {rhs}, got {lhs} (diff = {})",
            (lhs - rhs).abs()
        );
    }

    fn rotation_2d(angle: f64) -> Array2<f64> {
        let (sin, cos) = angle.sin_cos();
        array![[cos, -sin], [sin, cos]]
    }

    fn gradient_4x2() -> Array2<f64> {
        array![[0.2, -0.1], [0.3, 0.4], [-0.5, 0.7], [0.1, -0.2],]
    }

    fn make_model(shapes: &[(usize, usize)]) -> DummyModel {
        DummyModel {
            state: CosmosState::new(shapes).unwrap(),
        }
    }

    fn make_evolution_law(num_blocks: usize) -> Result<GaugeDynamics, RulesError> {
        GaugeDynamics::new(num_blocks, GaugeDynamicsConfig::default())
    }

    #[test]
    fn snapshot_initial_blocks_clones_model_state() {
        let mut model = make_model(&[(2, 1)]);
        let snapshot = snapshot_initial_blocks(&model);

        model.state.apply_conjugation(&rotation_2d(FRAC_PI_2));

        assert_eq!(snapshot.len(), 1);
        assert_close(snapshot[0][[0, 0]], 1.0);
        assert_close(snapshot[0][[1, 0]], 0.0);
        assert_close(model.state().get_basis(0).unwrap()[[0, 0]], 0.0);
        assert_close(model.state().get_basis(0).unwrap()[[1, 0]], 1.0);
    }

    #[test]
    fn snapshot_initial_blocks_from_state_matches_model_wrapper() {
        let model = make_model(&[(4, 2), (3, 1)]);
        let via_model = snapshot_initial_blocks(&model);
        let via_state = snapshot_initial_blocks_from_state(model.state());

        assert_eq!(via_model, via_state);
    }

    #[test]
    fn compute_block_geometry_returns_zero_drift_for_fresh_state() {
        let model = make_model(&[(4, 2), (4, 2)]);
        let init_blocks = snapshot_initial_blocks(&model);
        let evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();

        let diagnostics = compute_block_geometry(&model, &init_blocks, &evolution_law).unwrap();

        assert_eq!(diagnostics.orth_error.len(), 2);
        assert_eq!(diagnostics.drift.len(), 2);
        assert_eq!(diagnostics.comm_norm.len(), 2);
        assert!(diagnostics.orth_error.iter().all(|err| *err < 1e-14));
        assert!(diagnostics.drift.iter().all(|err| *err < 1e-14));
        assert!(diagnostics.comm_norm.iter().all(|norm| *norm == 0.0));
    }

    #[test]
    fn compute_block_geometry_tracks_geodesic_drift_after_rotation() {
        let mut model = make_model(&[(2, 1)]);
        let init_blocks = snapshot_initial_blocks(&model);
        let evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();

        model.state.apply_conjugation(&rotation_2d(FRAC_PI_2));

        let diagnostics = compute_block_geometry(&model, &init_blocks, &evolution_law).unwrap();

        assert_close(diagnostics.orth_error[0], 0.0);
        assert_close(diagnostics.drift[0], FRAC_PI_2);
    }

    #[test]
    fn compute_block_geometry_uses_evolution_law_commutator_cache() {
        let mut state = CosmosState::new(&[(4, 2)]).unwrap();
        let init_blocks = snapshot_initial_blocks_from_state(&state);
        let mut evolution_law = make_evolution_law(state.num_blocks()).unwrap();
        let gradient = gradient_4x2();
        let gradients = vec![Some(&gradient)];

        evolution_law.step(&mut state, &gradients).unwrap();

        let diagnostics =
            compute_block_geometry_from_state(&state, &init_blocks, &evolution_law).unwrap();
        let expected = evolution_law.get_diagnostics(&state).commutator_norms;

        assert_eq!(diagnostics.comm_norm.len(), 1);
        assert_close(diagnostics.comm_norm[0], expected[0]);
        assert!(diagnostics.comm_norm[0] > 0.0);
    }

    #[test]
    fn compute_block_geometry_rejects_initial_block_count_mismatch() {
        let state = CosmosState::new(&[(2, 1)]).unwrap();
        let evolution_law = make_evolution_law(state.num_blocks()).unwrap();

        let err = compute_block_geometry_from_state(&state, &[], &evolution_law).unwrap_err();

        match err {
            GeometryDiagnosticsError::InitialBlockCountMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn compute_block_geometry_rejects_evolution_law_block_count_mismatch() {
        let state = CosmosState::new(&[(2, 1)]).unwrap();
        let init_blocks = snapshot_initial_blocks_from_state(&state);
        let evolution_law = make_evolution_law(2).unwrap();

        let err =
            compute_block_geometry_from_state(&state, &init_blocks, &evolution_law).unwrap_err();

        match err {
            GeometryDiagnosticsError::EvolutionLawBlockCountMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn compute_block_geometry_rejects_initial_block_shape_mismatch() {
        let state = CosmosState::new(&[(2, 1)]).unwrap();
        let evolution_law = make_evolution_law(state.num_blocks()).unwrap();
        let wrong_shape = array![[1.0, 0.0], [0.0, 1.0]];

        let err =
            compute_block_geometry_from_state(&state, &[wrong_shape], &evolution_law).unwrap_err();

        match err {
            GeometryDiagnosticsError::InitialBlockShapeMismatch {
                block,
                expected_rows,
                expected_cols,
                got_rows,
                got_cols,
            } => {
                assert_eq!(block, 0);
                assert_eq!(expected_rows, 2);
                assert_eq!(expected_cols, 1);
                assert_eq!(got_rows, 2);
                assert_eq!(got_cols, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
