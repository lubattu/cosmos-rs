//! Shared utilities for arrangement primitives.
//!
//! This module centralizes helper logic reused by multiple arrangement
//! submodules: block accessors over [`CosmosState`] and geometric
//! similarity/energy measurements.
//!
//! # Key groups
//!
//! - block access helpers: [`get_w_block`], [`get_w_blocks`], [`all_same_k`]
//! - geometric measurements: [`compute_alignment_matrix`],
//!   [`projection_energy`], [`normalize_rank1`]

use ndarray::{Array1, Array2, ArrayView2};

use crate::state::cosmos_state::CosmosState;

use super::ArrangementError;

const NORM_EPS: f64 = 1e-12;

/// Extract the `block_idx` orthonormal basis from a [`CosmosState`].
///
/// Returns a read-only view into the state's internal block storage.
///
/// # Errors
///
/// Returns [`ArrangementError::InvalidParameter`] if `block_idx` is out of
/// range for the provided state.
pub fn get_w_block(
    state: &CosmosState,
    block_idx: usize,
) -> Result<ArrayView2<'_, f64>, ArrangementError> {
    state
        .get_block_by_index(block_idx)
        .map(|block| block.view())
        .map_err(|_| ArrangementError::InvalidParameter {
            param: "block_idx",
            reason: format!(
                "index {block_idx} is out of range for {} blocks",
                state.num_blocks()
            ),
        })
}

/// Extract multiple orthonormal bases from a [`CosmosState`].
///
/// The returned vector preserves the order of `indices`.
///
/// # Errors
///
/// Returns [`ArrangementError::InvalidParameter`] if any requested index is out
/// of range.
pub fn get_w_blocks(
    state: &CosmosState,
    indices: &[usize],
) -> Result<Vec<Array2<f64>>, ArrangementError> {
    indices
        .iter()
        .map(|&idx| get_w_block(state, idx).map(|block| block.to_owned()))
        .collect()
}

/// Check whether all requested blocks share the same subspace dimension `k`.
///
/// Returns `true` for empty or single-element selections. Invalid indices
/// return `false`.
pub fn all_same_k(state: &CosmosState, indices: &[usize]) -> bool {
    let Some((&first_idx, rest)) = indices.split_first() else {
        return true;
    };

    let Ok(first) = state.get_block_by_index(first_idx) else {
        return false;
    };
    let k0 = first.ncols();

    rest.iter().all(|&idx| {
        state
            .get_block_by_index(idx)
            .map(|block| block.ncols() == k0)
            .unwrap_or(false)
    })
}

/// Similarity between two orthonormal bases:
/// `sim(W_i, W_j) = ||W_i^T W_j||_F / sqrt(min(k_i, k_j))`.
///
/// Returns `0.0` when the ambient dimensions differ.
fn pairwise_similarity(w_i: &Array2<f64>, w_j: &Array2<f64>) -> f64 {
    if w_i.nrows() != w_j.nrows() || w_i.ncols() == 0 || w_j.ncols() == 0 {
        return 0.0;
    }

    let cross = w_i.t().dot(w_j);
    let fro_sq: f64 = cross.iter().map(|value| value * value).sum();
    let k_min = w_i.ncols().min(w_j.ncols()) as f64;

    (fro_sq.sqrt() / k_min.sqrt()).clamp(0.0, 1.0)
}

/// Compute the full pairwise alignment matrix for a set of blocks.
///
/// Returns an `(L, L)` symmetric matrix where entry `[i, j]` equals
/// `||W_i^T W_j||_F / sqrt(min(k_i, k_j))` for `blocks[i]` and `blocks[j]`.
/// Diagonal entries are set to `1.0`.
pub fn compute_alignment_matrix(blocks: &[Array2<f64>]) -> Array2<f64> {
    let num_blocks = blocks.len();
    let mut alignment = Array2::zeros((num_blocks, num_blocks));

    for i in 0..num_blocks {
        alignment[(i, i)] = 1.0;
        for j in (i + 1)..num_blocks {
            let similarity = pairwise_similarity(&blocks[i], &blocks[j]);
            alignment[(i, j)] = similarity;
            alignment[(j, i)] = similarity;
        }
    }

    alignment
}

/// Normalize a vector to unit norm.
///
/// Returns the zero vector unchanged when the norm is smaller than `1e-12`.
pub fn normalize_rank1(v: &Array1<f64>) -> Array1<f64> {
    let norm_sq: f64 = v.iter().map(|value| value * value).sum();
    let norm = norm_sq.sqrt();

    if norm < NORM_EPS {
        Array1::zeros(v.len())
    } else {
        v.mapv(|value| value / norm)
    }
}

/// Compute projection energy `E(x) = ||x W||^2` per sample.
///
/// For orthonormal W (W^T W = I), this equals `||x W W^T||^2` but avoids
/// the second matmul by exploiting `||P x||^2 = ||x W||^2` where P = W W^T.
pub fn projection_energy(x: &Array2<f64>, w: &Array2<f64>) -> Array1<f64> {
    let coords = x.dot(w); // batch × k — single GEMM
    Array1::from_iter(
        coords
            .outer_iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f64>()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    use ndarray::array;

    use crate::state::cosmos_state::CosmosState;

    const EPSILON: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= EPSILON,
            "expected {expected}, got {actual}, diff {diff} > {EPSILON}"
        );
    }

    fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (*a - *e).abs();
            assert!(
                diff <= EPSILON,
                "index {idx}: expected {e}, got {a}, diff {diff} > {EPSILON}"
            );
        }
    }

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

    fn basis_e1() -> Array2<f64> {
        array![[1.0], [0.0], [0.0]]
    }

    fn basis_e2() -> Array2<f64> {
        array![[0.0], [1.0], [0.0]]
    }

    fn basis_e12() -> Array2<f64> {
        array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    }

    fn basis_e13() -> Array2<f64> {
        array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    }

    #[test]
    fn test_get_w_block_valid_index() {
        let state = CosmosState::new(&[(4, 2), (3, 1)]).unwrap();

        let block = get_w_block(&state, 0).unwrap();
        assert_eq!(block.dim(), (4, 2));
    }

    #[test]
    fn test_get_w_block_invalid_index() {
        let state = CosmosState::new(&[(4, 2)]).unwrap();
        let err = get_w_block(&state, 1).unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => {
                assert_eq!(param, "block_idx");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_get_w_blocks_preserves_order() {
        let state = CosmosState::new(&[(4, 2), (3, 1), (5, 2)]).unwrap();
        let blocks = get_w_blocks(&state, &[2, 0]).unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].dim(), (5, 2));
        assert_eq!(blocks[1].dim(), (4, 2));
    }

    #[test]
    fn test_all_same_k_true_for_uniform_selection() {
        let state = CosmosState::new(&[(4, 2), (5, 2), (3, 2)]).unwrap();
        assert!(all_same_k(&state, &[0, 1, 2]));
    }

    #[test]
    fn test_all_same_k_false_for_mixed_selection() {
        let state = CosmosState::new(&[(4, 2), (5, 1), (3, 2)]).unwrap();
        assert!(!all_same_k(&state, &[0, 1, 2]));
    }

    #[test]
    fn test_alignment_matrix_identical_bases_are_fully_similar() {
        let w = basis_e12();
        let alignment = compute_alignment_matrix(&[w.clone(), w]);
        assert_close(alignment[(0, 1)], 1.0);
    }

    #[test]
    fn test_alignment_matrix_orthogonal_bases_are_dissimilar() {
        let alignment = compute_alignment_matrix(&[basis_e1(), basis_e2()]);
        assert_close(alignment[(0, 1)], 0.0);
    }

    #[test]
    fn test_alignment_matrix_partial_overlap() {
        let alignment = compute_alignment_matrix(&[basis_e12(), basis_e13()]);
        assert_close(alignment[(0, 1)], 1.0 / SQRT_2);
    }

    #[test]
    fn test_compute_alignment_matrix_is_symmetric_with_unit_diagonal() {
        let blocks = vec![basis_e1(), basis_e2(), basis_e12()];
        let alignment = compute_alignment_matrix(&blocks);

        assert_eq!(alignment.dim(), (3, 3));
        assert_close(alignment[(0, 0)], 1.0);
        assert_close(alignment[(1, 1)], 1.0);
        assert_close(alignment[(2, 2)], 1.0);
        assert_close(alignment[(0, 1)], alignment[(1, 0)]);
        assert_close(alignment[(0, 2)], alignment[(2, 0)]);
        assert_close(alignment[(1, 2)], alignment[(2, 1)]);
    }

    #[test]
    fn test_compute_alignment_matrix_matches_manual_pairwise_values() {
        let blocks = vec![basis_e1(), basis_e2(), basis_e12()];
        let alignment = compute_alignment_matrix(&blocks);
        let expected = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],];

        assert_array2_close(&alignment, &expected);
    }

    #[test]
    fn test_compute_alignment_matrix_handles_mixed_k() {
        let blocks = vec![basis_e1(), basis_e12(), basis_e13()];
        let alignment = compute_alignment_matrix(&blocks);

        assert_close(alignment[(0, 1)], 1.0);
        assert_close(alignment[(0, 2)], 1.0);
        assert_close(alignment[(1, 2)], 1.0 / SQRT_2);
    }

    #[test]
    fn test_normalize_rank1_nonzero_vector() {
        let v = array![3.0, 4.0];
        let normalized = normalize_rank1(&v);
        let expected = array![0.6, 0.8];

        assert_array1_close(&normalized, &expected);
    }

    #[test]
    fn test_normalize_rank1_zero_vector() {
        let v = array![0.0, 0.0, 0.0];
        let normalized = normalize_rank1(&v);
        let expected = array![0.0, 0.0, 0.0];

        assert_array1_close(&normalized, &expected);
    }

    #[test]
    fn test_projection_energy_axis_aligned_basis() {
        let x = array![[3.0, 4.0], [-1.0, 2.0]];
        let w = array![[1.0], [0.0]];
        let energies = projection_energy(&x, &w);
        let expected = array![9.0, 1.0];

        assert_array1_close(&energies, &expected);
    }

    #[test]
    fn test_projection_energy_zero_input_is_zero() {
        let x = array![[0.0, 0.0], [0.0, 0.0]];
        let w = array![[1.0], [0.0]];
        let expected = array![0.0, 0.0];

        assert_array1_close(&projection_energy(&x, &w), &expected);
    }

    #[test]
    fn test_projection_energy_matches_projected_norm() {
        let inv_sqrt2 = 1.0 / SQRT_2;
        let x = array![[1.0, 1.0], [1.0, -1.0]];
        let w = array![[inv_sqrt2], [inv_sqrt2]];
        let expected = array![2.0, 0.0];

        assert_array1_close(&projection_energy(&x, &w), &expected);
    }
}
