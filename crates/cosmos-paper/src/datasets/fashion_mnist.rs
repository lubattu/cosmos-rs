//! Fashion-MNIST loader and normalization.
//!
//! Premises:
//! - Images are 28 by 28 unsigned bytes, decoded row-major into a flat
//!   784-column matrix.
//! - Normalization is `(pixel / 255 - 0.2860) / 0.3530`, applied per pixel.

use std::path::Path;

use ndarray::Array2;

use super::download::{Archive, ensure_archive, gunzip};
use super::{Dataset, DatasetError};

const BASE_URL: &str = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com";
const PIXEL_MEAN: f64 = 0.2860;
const PIXEL_STD: f64 = 0.3530;

const TRAIN_IMAGES: Archive = Archive {
    file_name: "train-images-idx3-ubyte.gz",
    url: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    size: 26_421_880,
    sha256: "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
};
const TRAIN_LABELS: Archive = Archive {
    file_name: "train-labels-idx1-ubyte.gz",
    url: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    size: 29_515,
    sha256: "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
};
const TEST_IMAGES: Archive = Archive {
    file_name: "t10k-images-idx3-ubyte.gz",
    url: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    size: 4_422_102,
    sha256: "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
};
const TEST_LABELS: Archive = Archive {
    file_name: "t10k-labels-idx1-ubyte.gz",
    url: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    size: 5_148,
    sha256: "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
};

/// The four archives this dataset is built from.
pub const ARCHIVES: [Archive; 4] = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS];

/// Source host the archives are fetched from.
pub const SOURCE: &str = BASE_URL;

/// Normalize one image split into a `(samples, 784)` matrix.
pub fn normalize_images(raw: &[u8], samples: usize) -> Array2<f64> {
    Array2::from_shape_fn((samples, 784), |(row, col)| {
        let value = f64::from(raw[row * 784 + col]) / 255.0;
        (value - PIXEL_MEAN) / PIXEL_STD
    })
}

/// Load the dataset, fetching the archives when permitted.
///
/// # Errors
///
/// Returns [`DatasetError`] when an archive is absent and fetching is
/// disabled, when a digest does not match, or when a payload does not
/// decode to the expected shape.
pub fn load(data_dir: &Path, allow_download: bool) -> Result<Dataset, DatasetError> {
    let dir = data_dir.join("fashion-mnist");
    let hint = "cargo run -p cosmos-paper -- fetch fashion-mnist";

    let mut payloads = Vec::with_capacity(4);
    for archive in ARCHIVES {
        let path = ensure_archive(&archive, &dir, allow_download, hint)?;
        payloads.push(gunzip(&path)?);
    }

    let train_images = super::idx::decode(&payloads[0], 3)?;
    let train_labels = super::idx::decode(&payloads[1], 1)?;
    let test_images = super::idx::decode(&payloads[2], 3)?;
    let test_labels = super::idx::decode(&payloads[3], 1)?;

    super::expect_dims(&train_images.dims, &[60_000, 28, 28], "train images")?;
    super::expect_dims(&train_labels.dims, &[60_000], "train labels")?;
    super::expect_dims(&test_images.dims, &[10_000, 28, 28], "test images")?;
    super::expect_dims(&test_labels.dims, &[10_000], "test labels")?;

    Ok(Dataset {
        train_inputs: normalize_images(&train_images.data, 60_000),
        train_labels: train_labels.data.iter().map(|&b| b as usize).collect(),
        test_inputs: normalize_images(&test_images.data, 10_000),
        test_labels: test_labels.data.iter().map(|&b| b as usize).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_maps_the_extremes() {
        let raw = vec![0u8; 784];
        let normalized = normalize_images(&raw, 1);
        assert!((normalized[[0, 0]] - (-PIXEL_MEAN / PIXEL_STD)).abs() < 1e-12);

        let raw = vec![255u8; 784];
        let normalized = normalize_images(&raw, 1);
        assert!((normalized[[0, 0]] - ((1.0 - PIXEL_MEAN) / PIXEL_STD)).abs() < 1e-12);
    }

    #[test]
    fn normalization_is_row_major() {
        let mut raw = vec![0u8; 1568];
        raw[784] = 255;
        let normalized = normalize_images(&raw, 2);
        assert!(normalized[[1, 0]] > normalized[[0, 0]]);
    }
}
