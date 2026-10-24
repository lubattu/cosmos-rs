//! Ambient-drift contrast curve.
//!
//! A deterministic simulation with no dataset and no model: an orthonormal
//! frame is transported by repeated orthogonal updates generated first by a
//! dense antisymmetric matrix and then by the block-diagonal restriction of
//! the same matrix, and the distance from the starting subspace is recorded
//! at every step. The two curves are the contrast panel of the invariant
//! figure.
//!
//! The transported frame is `(ambient_dim, subspace_dim)`, so orthonormality
//! is checked as `U^T U = I_k` on its columns, not the square `U U^T = I_n`
//! form `matrix_exp::orthogonality_error` assumes.

use cosmos_rs::geometry::grassmann::projection_distance;
use cosmos_rs::geometry::matrix_exp::{
    OrthoMethod, matrix_exp_antisym, orthogonality_error_columns,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::exp_bench::generator;

/// Simulation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct DriftConfig {
    /// Ambient dimension.
    pub ambient_dim: usize,
    /// Subspace width.
    pub subspace_dim: usize,
    /// Number of diagonal blocks in the restricted generator.
    pub blocks: usize,
    /// Steps to transport.
    pub steps: usize,
    /// Generator scaling per step.
    pub step_size: f64,
}

/// The two curves and their orthogonality traces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftResults {
    /// Step indices, one-based.
    pub step: Vec<usize>,
    /// Distance from the starting subspace under the dense generator.
    pub ambient_distance: Vec<f64>,
    /// Distance from the starting subspace under the block generator.
    pub block_distance: Vec<f64>,
    /// Orthogonality error under the dense generator.
    pub ambient_orth_error: Vec<f64>,
    /// Orthogonality error under the block generator.
    pub block_orth_error: Vec<f64>,
}

fn block_diagonal(a: &Array2<f64>, blocks: usize) -> Array2<f64> {
    let n = a.nrows();
    let width = n / blocks;
    Array2::from_shape_fn((n, n), |(row, col)| {
        if row / width == col / width {
            a[[row, col]]
        } else {
            0.0
        }
    })
}

fn transport(
    generator_matrix: &Array2<f64>,
    start: &Array2<f64>,
    steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let update = matrix_exp_antisym(generator_matrix, OrthoMethod::Pade)
        .expect("square antisymmetric generator");
    let mut current = start.clone();
    let mut distance = Vec::with_capacity(steps);
    let mut error = Vec::with_capacity(steps);
    for _ in 0..steps {
        current = update.dot(&current);
        distance.push(projection_distance(start, &current).expect("equal shapes"));
        error.push(orthogonality_error_columns(&current));
    }
    (distance, error)
}

/// Run the simulation.
pub fn drift_curve(config: &DriftConfig) -> DriftResults {
    let a = generator(config.ambient_dim, 7, 1.0) * config.step_size;
    let restricted = block_diagonal(&a, config.blocks);
    let start = Array2::from_shape_fn((config.ambient_dim, config.subspace_dim), |(row, col)| {
        if row == col { 1.0 } else { 0.0 }
    });

    let (ambient_distance, ambient_orth_error) = transport(&a, &start, config.steps);
    let (block_distance, block_orth_error) = transport(&restricted, &start, config.steps);

    DriftResults {
        step: (1..=config.steps).collect(),
        ambient_distance,
        block_distance,
        ambient_orth_error,
        block_orth_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> DriftConfig {
        DriftConfig {
            ambient_dim: 16,
            subspace_dim: 4,
            blocks: 4,
            steps: 20,
            step_size: 0.05,
        }
    }

    #[test]
    fn curves_have_the_requested_length() {
        let results = drift_curve(&small());
        assert_eq!(results.step.len(), 20);
        assert_eq!(results.ambient_distance.len(), 20);
        assert_eq!(results.block_distance.len(), 20);
    }

    #[test]
    fn transport_preserves_orthonormality() {
        let results = drift_curve(&small());
        assert!(
            results
                .ambient_orth_error
                .iter()
                .all(|error| *error < 1e-11)
        );
        assert!(results.block_orth_error.iter().all(|error| *error < 1e-11));
    }

    #[test]
    fn simulation_is_deterministic() {
        assert_eq!(drift_curve(&small()), drift_curve(&small()));
    }

    #[test]
    fn dense_generator_moves_the_subspace_further() {
        let results = drift_curve(&small());
        let ambient = results.ambient_distance.last().copied().expect("last");
        let block = results.block_distance.last().copied().expect("last");
        assert!(ambient > block);
    }
}
