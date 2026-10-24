//! Deterministic block initialization.
//!
//! Every entry of every block basis is a pure function of
//! `(seed, block index, row, column)`. The draws themselves now live in the
//! library as [`cosmos_rs::state::keyed_init`]; this module re-exports them
//! and keeps the pinned-value tests that specify the reproduction contract.
//! Changing the generator, the order in which draws are consumed, or the
//! orthonormalization changes every number the driver reports.
//
// Premises:
// - The reference accuracies in the smoke reference table assume the bases
//   this module produces. A change here invalidates that table.
// - The state's own `seed_orthonormal` convenience initializer draws from a
//   different distribution and is never used by this driver;
//   `seed_orthonormal_gaussian` is the library form of these draws.

use cosmos_rs::prelude::{CosmosState, CosmosStateError};

pub use cosmos_rs::state::keyed_init::{block_entry, gaussian_block, orthonormal_block};

/// Replace every block of `state` with its keyed orthonormal basis.
///
/// Delegates to [`CosmosState::seed_orthonormal_gaussian`], the library
/// constructor that produces the article's bases.
///
/// # Errors
///
/// Returns [`CosmosStateError`] when a basis is rejected by the state,
/// which cannot happen while each block is drawn at its own registered
/// shape.
pub fn seed_state(state: &mut CosmosState, seed: u64) -> Result<(), CosmosStateError> {
    state.seed_orthonormal_gaussian(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const SEED: u64 = 319_635_789;

    fn column_orthonormality_error(basis: &Array2<f64>) -> f64 {
        let k = basis.ncols();
        let gram = basis.t().dot(basis);
        let eye = Array2::<f64>::eye(k);
        (gram - eye).mapv(|x| x.powi(2)).sum().sqrt()
    }

    #[test]
    fn entries_match_pinned_values() {
        let cases = [
            (0usize, 0usize, 0usize, 1.840_235_749_600_384_5_f64),
            (0, 0, 1, 0.088_302_041_516_372_66),
            (0, 1, 0, 1.885_535_264_336_16),
            (3, 2, 5, 1.730_555_465_850_431_2),
        ];
        for (block, row, col, expected) in cases {
            let got = block_entry(SEED, block, row, col);
            assert!(
                (got - expected).abs() < 1e-12,
                "block {block} row {row} col {col}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn entry_is_independent_of_block_shape() {
        let small = gaussian_block(SEED, 2, 4, 3);
        let large = gaussian_block(SEED, 2, 16, 9);
        for row in 0..4 {
            for col in 0..3 {
                assert_eq!(small[[row, col]], large[[row, col]]);
            }
        }
    }

    #[test]
    fn different_blocks_draw_different_matrices() {
        let first = gaussian_block(SEED, 0, 8, 4);
        let second = gaussian_block(SEED, 1, 8, 4);
        assert_ne!(first, second);
    }

    #[test]
    fn different_seeds_draw_different_matrices() {
        let first = gaussian_block(SEED, 0, 8, 4);
        let second = gaussian_block(SEED + 1, 0, 8, 4);
        assert_ne!(first, second);
    }

    #[test]
    fn orthonormal_block_is_orthonormal() {
        let basis = orthonormal_block(SEED, 0, 784, 32);
        assert_eq!(basis.dim(), (784, 32));
        assert!(column_orthonormality_error(&basis) < 1e-13);

        let small = orthonormal_block(SEED, 1, 6, 3);
        assert_eq!(small.dim(), (6, 3));
        assert!(column_orthonormality_error(&small) < 1e-13);
    }

    #[test]
    fn seed_state_fills_every_block_reproducibly() {
        let shapes = [(784usize, 32usize), (32, 16), (16, 8)];
        let mut first = CosmosState::new(&shapes).expect("state");
        let mut second = CosmosState::new(&shapes).expect("state");
        seed_state(&mut first, SEED).expect("seed");
        seed_state(&mut second, SEED).expect("seed");
        for block in 0..shapes.len() {
            assert_eq!(
                first.get_basis(block).expect("basis"),
                second.get_basis(block).expect("basis")
            );
        }
        assert!(first.total_orthogonality_error() < 1e-12);
    }
}
