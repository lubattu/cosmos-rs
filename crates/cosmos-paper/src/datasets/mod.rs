//! Dataset acquisition, decoding, splitting and batching.
//!
//! Premises:
//! - The per-epoch shuffle seed is `base_seed + epoch * 10_000`.
//! - The split seed is independent of the run seed, so every arm and every
//!   seed of a comparison sees the identical partition.

pub mod cifar10;
pub mod download;
pub mod fashion_mnist;
pub mod idx;

use std::fmt;
use std::io;
use std::path::Path;
use std::str::FromStr;

use cosmos_rs::prelude::{FlatClassificationBatch, PatchwiseClassificationBatch};
use ndarray::{Array2, Array4};

use download::FetchError;
use idx::IdxError;

/// A dataset already normalized into matrices the adapters accept.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training inputs, one row per sample.
    pub train_inputs: Array2<f64>,
    /// Training labels.
    pub train_labels: Vec<usize>,
    /// Evaluation inputs, one row per sample.
    pub test_inputs: Array2<f64>,
    /// Evaluation labels.
    pub test_labels: Vec<usize>,
}

/// Datasets the driver can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetName {
    /// Fashion-MNIST, 28 by 28 grayscale.
    FashionMnist,
    /// CIFAR-10, 32 by 32 colour.
    Cifar10,
}

impl DatasetName {
    /// Ambient width of one flattened sample.
    pub fn input_dim(self) -> usize {
        match self {
            Self::FashionMnist => 784,
            Self::Cifar10 => 3072,
        }
    }

    /// Image shape as `(channels, height, width)`.
    pub fn image_shape(self) -> (usize, usize, usize) {
        match self {
            Self::FashionMnist => (1, 28, 28),
            Self::Cifar10 => (3, 32, 32),
        }
    }
}

impl fmt::Display for DatasetName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FashionMnist => write!(f, "fashion-mnist"),
            Self::Cifar10 => write!(f, "cifar10"),
        }
    }
}

impl FromStr for DatasetName {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fashion-mnist" => Ok(Self::FashionMnist),
            "cifar10" => Ok(Self::Cifar10),
            other => Err(format!(
                "unknown dataset {other}; expected fashion-mnist or cifar10"
            )),
        }
    }
}

/// Reason a dataset could not be made available.
#[derive(Debug)]
pub enum DatasetError {
    /// An archive could not be obtained or verified.
    Fetch(FetchError),
    /// A payload could not be decoded.
    Idx(IdxError),
    /// A payload decoded to an unexpected shape.
    Shape {
        /// What was being read.
        what: String,
        /// Shape that was expected.
        expected: String,
        /// Shape that was found.
        got: String,
    },
    /// A filesystem operation failed.
    Io(io::Error),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fetch(err) => write!(f, "{err}"),
            Self::Idx(err) => write!(f, "{err}"),
            Self::Shape {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected}, got {got}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<FetchError> for DatasetError {
    fn from(value: FetchError) -> Self {
        Self::Fetch(value)
    }
}

impl From<IdxError> for DatasetError {
    fn from(value: IdxError) -> Self {
        Self::Idx(value)
    }
}

impl From<io::Error> for DatasetError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub(crate) fn expect_dims(
    got: &[usize],
    expected: &[usize],
    what: &str,
) -> Result<(), DatasetError> {
    if got == expected {
        Ok(())
    } else {
        Err(DatasetError::Shape {
            what: what.to_string(),
            expected: format!("{expected:?}"),
            got: format!("{got:?}"),
        })
    }
}

/// Load a dataset from `data_dir`, fetching it when permitted.
///
/// # Errors
///
/// Returns [`DatasetError`] when the data is absent and `allow_download` is
/// false, when a digest does not match, or when a payload does not decode.
pub fn load(
    name: DatasetName,
    data_dir: &Path,
    allow_download: bool,
) -> Result<Dataset, DatasetError> {
    match name {
        DatasetName::FashionMnist => fashion_mnist::load(data_dir, allow_download),
        DatasetName::Cifar10 => cifar10::load(data_dir, allow_download),
    }
}

/// Fetch and verify a dataset without decoding it.
///
/// # Errors
///
/// Returns [`DatasetError`] when an archive cannot be obtained or verified.
pub fn fetch(name: DatasetName, data_dir: &Path) -> Result<(), DatasetError> {
    match name {
        DatasetName::FashionMnist => {
            fashion_mnist::load(data_dir, true)?;
        }
        DatasetName::Cifar10 => {
            cifar10::load(data_dir, true)?;
        }
    }
    Ok(())
}

/// Shuffle seed for `epoch`, one-based.
pub fn epoch_seed(base: u64, epoch: u32) -> u64 {
    base.wrapping_add(u64::from(epoch).wrapping_mul(10_000))
}

const SHUFFLE_SALT: u64 = 0x9e37_79b9_7f4a_7c15;
const SHUFFLE_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

fn next_state(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(SHUFFLE_MULTIPLIER).wrapping_add(1);
    *state
}

/// A deterministic permutation of `0..len`.
///
/// The permutation is produced by a descending Fisher-Yates pass — index
/// `len - 1` down to `1`, each swapped with `state % (index + 1)` — driven
/// by a 64-bit linear congruential generator whose state starts at
/// `seed ^ SHUFFLE_SALT` and advances **before** each draw. Every constant
/// and the direction of the pass are part of the reproduction contract:
/// changing any of them changes every batch order in every run.
pub fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    if len < 2 {
        return order;
    }
    let mut state = seed ^ SHUFFLE_SALT;
    for index in (1..len).rev() {
        let pick = (next_state(&mut state) % (index as u64 + 1)) as usize;
        order.swap(index, pick);
    }
    order
}

/// Split `0..len` into a training order and a validation order.
///
/// The whole range is shuffled with `split_seed` and the trailing
/// `val_size` indices become the validation split.
pub fn split_indices(len: usize, val_size: usize, split_seed: u64) -> (Vec<usize>, Vec<usize>) {
    let order = shuffled_indices(len, split_seed);
    let cut = len.saturating_sub(val_size);
    let train = order[..cut].to_vec();
    let val = order[cut..].to_vec();
    (train, val)
}

/// Build flat batches from `order`.
///
/// When `drop_last` is true the trailing partial batch is discarded.
pub fn flat_batches(
    inputs: &Array2<f64>,
    labels: &[usize],
    order: &[usize],
    batch_size: usize,
    drop_last: bool,
) -> Vec<FlatClassificationBatch> {
    let width = inputs.ncols();
    let mut batches = Vec::new();
    for chunk in order.chunks(batch_size) {
        if drop_last && chunk.len() < batch_size {
            break;
        }
        let rows = chunk.len();
        let batch_inputs =
            Array2::from_shape_fn((rows, width), |(row, col)| inputs[[chunk[row], col]]);
        let batch_labels: Vec<usize> = chunk.iter().map(|&index| labels[index]).collect();
        batches.push(FlatClassificationBatch::single_label(
            batch_inputs,
            batch_labels,
        ));
    }
    batches
}

/// Build image batches from `order`, reshaping each flat row in place.
///
/// When `drop_last` is true the trailing partial batch is discarded.
pub fn patch_batches(
    inputs: &Array2<f64>,
    labels: &[usize],
    order: &[usize],
    batch_size: usize,
    drop_last: bool,
    image_shape: (usize, usize, usize),
) -> Vec<PatchwiseClassificationBatch> {
    let (channels, height, width) = image_shape;
    let mut batches = Vec::new();
    for chunk in order.chunks(batch_size) {
        if drop_last && chunk.len() < batch_size {
            break;
        }
        let rows = chunk.len();
        let batch_inputs =
            Array4::from_shape_fn((rows, channels, height, width), |(row, c, y, x)| {
                inputs[[chunk[row], (c * height + y) * width + x]]
            });
        let batch_labels: Vec<usize> = chunk.iter().map(|&index| labels[index]).collect();
        batches.push(PatchwiseClassificationBatch::single_label(
            batch_inputs,
            batch_labels,
        ));
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_seed_advances_by_ten_thousand_from_a_one_based_epoch() {
        assert_eq!(epoch_seed(319_635_789, 1), 319_645_789);
        assert_eq!(epoch_seed(319_635_789, 7), 319_705_789);
        assert_eq!(epoch_seed(319_635_789, 0), 319_635_789);
    }

    #[test]
    fn shuffle_is_a_permutation_and_is_deterministic() {
        let first = shuffled_indices(1000, 12345);
        let second = shuffled_indices(1000, 12345);
        assert_eq!(first, second);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<usize>>());
        assert_ne!(first, (0..1000).collect::<Vec<usize>>());
    }

    #[test]
    fn shuffle_matches_its_pinned_permutations() {
        assert_eq!(shuffled_indices(4, 0), vec![1, 0, 3, 2]);
        assert_eq!(shuffled_indices(8, 12_345), vec![2, 7, 4, 1, 6, 3, 0, 5]);
        assert_eq!(
            shuffled_indices(8, epoch_seed(319_635_789, 1)),
            vec![0, 2, 5, 4, 6, 3, 7, 1]
        );
        assert_eq!(
            shuffled_indices(10, 12_345),
            vec![8, 6, 4, 7, 5, 1, 2, 3, 0, 9]
        );
    }

    #[test]
    fn split_sizes_and_disjointness_hold() {
        let (train, val) = split_indices(60_000, 5_000, 12_345);
        assert_eq!(train.len(), 55_000);
        assert_eq!(val.len(), 5_000);
        let mut all = train.clone();
        all.extend_from_slice(&val);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 60_000);
    }

    #[test]
    fn split_is_independent_of_the_run_seed() {
        let first = split_indices(60_000, 5_000, 12_345);
        let second = split_indices(60_000, 5_000, 12_345);
        assert_eq!(first, second);
        let other = split_indices(60_000, 5_000, 999);
        assert_ne!(first, other);
    }

    #[test]
    fn cifar_split_sizes_hold() {
        let (train, val) = split_indices(50_000, 5_000, 12_345);
        assert_eq!(train.len(), 45_000);
        assert_eq!(val.len(), 5_000);
    }

    #[test]
    fn drop_last_discards_the_partial_batch() {
        let inputs = Array2::from_shape_fn((10, 3), |(row, col)| (row * 3 + col) as f64);
        let labels: Vec<usize> = (0..10).map(|index| index % 2).collect();
        let order: Vec<usize> = (0..10).collect();
        let kept = flat_batches(&inputs, &labels, &order, 4, true);
        assert_eq!(kept.len(), 2);
        let all = flat_batches(&inputs, &labels, &order, 4, false);
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].inputs.nrows(), 2);
    }

    #[test]
    fn patch_batches_reshape_channel_major_rows() {
        let inputs = Array2::from_shape_fn((1, 12), |(_, col)| col as f64);
        let labels = vec![0usize];
        let order = vec![0usize];
        let batches = patch_batches(&inputs, &labels, &order, 1, false, (3, 2, 2));
        assert_eq!(batches[0].inputs.dim(), (1, 3, 2, 2));
        assert_eq!(batches[0].inputs[[0, 0, 0, 0]], 0.0);
        assert_eq!(batches[0].inputs[[0, 1, 0, 0]], 4.0);
        assert_eq!(batches[0].inputs[[0, 2, 1, 1]], 11.0);
    }
}
