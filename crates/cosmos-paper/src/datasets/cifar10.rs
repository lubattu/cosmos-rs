//! CIFAR-10 loader and per-channel normalization.
//!
//! Premises:
//! - The binary distribution stores one label byte followed by 3072 pixel
//!   bytes per record, in channel-major planes, so the flat 3072-column
//!   layout reshapes to `(3, 32, 32)` with no permutation.
//! - Normalization is per channel: `(pixel / 255 - mean[c]) / std[c]`.

use std::fs;
use std::path::Path;

use ndarray::Array2;

use super::download::{Archive, ensure_archive, untar_gz};
use super::{Dataset, DatasetError};

const RECORD_BYTES: usize = 3073;
const RECORDS_PER_FILE: usize = 10_000;
const FILE_BYTES: usize = RECORD_BYTES * RECORDS_PER_FILE;
const CHANNEL_MEANS: [f64; 3] = [0.4914, 0.4822, 0.4465];
const CHANNEL_STDS: [f64; 3] = [0.2470, 0.2435, 0.2616];

const ARCHIVE: Archive = Archive {
    file_name: "cifar-10-binary.tar.gz",
    url: "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    size: 170_052_171,
    sha256: "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
};

/// The single archive this dataset is built from.
pub const ARCHIVES: [Archive; 1] = [ARCHIVE];

const TRAIN_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const TEST_FILE: &str = "test_batch.bin";

fn decode_records(bytes: &[u8]) -> (Array2<f64>, Vec<usize>) {
    let records = bytes.len() / RECORD_BYTES;
    let mut labels = Vec::with_capacity(records);
    let inputs = Array2::from_shape_fn((records, 3072), |(row, col)| {
        let base = row * RECORD_BYTES + 1;
        let channel = col / 1024;
        let value = f64::from(bytes[base + col]) / 255.0;
        (value - CHANNEL_MEANS[channel]) / CHANNEL_STDS[channel]
    });
    for row in 0..records {
        labels.push(bytes[row * RECORD_BYTES] as usize);
    }
    (inputs, labels)
}

/// Load the dataset, fetching the archive when permitted.
///
/// # Errors
///
/// Returns [`DatasetError`] when the archive is absent and fetching is
/// disabled, when its digest does not match, or when an extracted batch
/// file does not have the expected size.
pub fn load(data_dir: &Path, allow_download: bool) -> Result<Dataset, DatasetError> {
    let dir = data_dir.join("cifar10");
    let hint = "cargo run -p cosmos-paper -- fetch cifar10";
    let batches = dir.join("cifar-10-batches-bin");

    if !batches.join(TEST_FILE).exists() {
        let path = ensure_archive(&ARCHIVE, &dir, allow_download, hint)?;
        untar_gz(&path, &dir)?;
    }

    let mut train_bytes = Vec::with_capacity(FILE_BYTES * TRAIN_FILES.len());
    for name in TRAIN_FILES {
        let path = batches.join(name);
        let bytes = fs::read(&path)?;
        if bytes.len() != FILE_BYTES {
            return Err(DatasetError::Shape {
                what: name.to_string(),
                expected: format!("{FILE_BYTES} bytes"),
                got: format!("{} bytes", bytes.len()),
            });
        }
        train_bytes.extend_from_slice(&bytes);
    }

    let test_bytes = fs::read(batches.join(TEST_FILE))?;
    if test_bytes.len() != FILE_BYTES {
        return Err(DatasetError::Shape {
            what: TEST_FILE.to_string(),
            expected: format!("{FILE_BYTES} bytes"),
            got: format!("{} bytes", test_bytes.len()),
        });
    }

    let (train_inputs, train_labels) = decode_records(&train_bytes);
    let (test_inputs, test_labels) = decode_records(&test_bytes);

    Ok(Dataset {
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_label_and_channel_layout() {
        let mut bytes = vec![0u8; RECORD_BYTES * 2];
        bytes[0] = 7;
        bytes[RECORD_BYTES] = 3;
        bytes[1 + 1024] = 255;
        let (inputs, labels) = decode_records(&bytes);
        assert_eq!(labels, vec![7, 3]);
        assert_eq!(inputs.dim(), (2, 3072));
        let expected = (1.0 - CHANNEL_MEANS[1]) / CHANNEL_STDS[1];
        assert!((inputs[[0, 1024]] - expected).abs() < 1e-12);
        let zero_green = (0.0 - CHANNEL_MEANS[1]) / CHANNEL_STDS[1];
        assert!((inputs[[0, 1025]] - zero_green).abs() < 1e-12);
    }
}
