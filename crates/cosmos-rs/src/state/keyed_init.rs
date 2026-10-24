//! Position-keyed Gaussian initialization.
//!
//! Every entry of every block basis is a pure function of
//! `(seed, block index, row, column)`. These are the draws behind the
//! article's reproduction runs: changing the generator, the order in which
//! draws are consumed, or the orthonormalization changes every number those
//! runs report, so this module is covered by pinned-value tests and is the
//! single place any of that may be specified.
//!
//! [`crate::state::cosmos_state::CosmosState::seed_orthonormal`] draws from a
//! different, sequential generator keyed on the block index alone; use
//! [`crate::state::cosmos_state::CosmosState::seed_orthonormal_gaussian`] to
//! start from the same bases the article's runs use.

use ndarray::Array2;

use crate::geometry::qr::{QrConfig, qr_orthonormalize};

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

fn splitmix64_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(GOLDEN);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
    z ^= z >> 31;
    (z as f64) / (u64::MAX as f64)
}

/// One standard-normal entry, keyed on its position.
///
/// The key is `seed ^ (block_idx << 32) ^ (row << 16) ^ col ^ GOLDEN`. A
/// single splitmix64 stream is then advanced twice from that key: the
/// first draw is the radial variate, clamped away from both ends of the
/// unit interval so the logarithm is finite, the second is the angular
/// variate, and the two are combined by the cosine branch of the
/// Box-Muller transform. The key schedule, the order of the two draws, the
/// clamp and the choice of branch are all part of the reproduction
/// contract; changing any of them changes every basis and every reported
/// number. Because the key carries the row and the column, an entry does
/// not depend on the shape of the block it sits in, so a block of a given
/// shape is reproducible in isolation.
pub fn block_entry(seed: u64, block_idx: usize, row: usize, col: usize) -> f64 {
    let mut state =
        seed ^ ((block_idx as u64) << 32) ^ ((row as u64) << 16) ^ (col as u64) ^ GOLDEN;
    let u1 = splitmix64_unit(&mut state).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let u2 = splitmix64_unit(&mut state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A `rows` by `cols` standard-normal matrix keyed on `(seed, block_idx)`.
pub fn gaussian_block(seed: u64, block_idx: usize, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        block_entry(seed, block_idx, row, col)
    })
}

/// An orthonormal `rows` by `cols` basis keyed on `(seed, block_idx)`.
pub fn orthonormal_block(seed: u64, block_idx: usize, rows: usize, cols: usize) -> Array2<f64> {
    qr_orthonormalize(
        &gaussian_block(seed, block_idx, rows, cols),
        &QrConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 319_635_789;

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
        let gram = basis.t().dot(&basis);
        let eye = Array2::<f64>::eye(32);
        let error = (gram - eye).mapv(|x| x.powi(2)).sum().sqrt();
        assert!(error < 1e-13);
    }
}
