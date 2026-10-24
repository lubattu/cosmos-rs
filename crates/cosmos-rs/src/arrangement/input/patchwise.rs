//! Patchwise input projection over image-like tensors.
//!
//! This module implements four parameter-free building blocks:
//!
//! - [`Patchify`]: extract non-overlapping square patches
//! - [`PatchProjector`]: apply one shared state block to every patch
//! - [`SpatialPooling`]: aggregate patches over a 2D patch grid
//! - [`PatchAggregator`]: reduce a patch set into a final projected embedding

use std::fmt;
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use serde::{Deserialize, Serialize};

use crate::{
    arrangement::{ArrangementError, utils::get_w_block},
    state::cosmos_state::CosmosState,
};

/// Output variant produced by [`Patchify::forward`].
#[derive(Debug, Clone, PartialEq)]
pub enum PatchifyOutput {
    /// Flattened patches with shape `(batch, num_patches, channels * p * p)`.
    Flattened(Array3<f64>),
    /// Channel-preserving patches with shape `(batch, num_patches, channels, p * p)`.
    Channelwise(Array4<f64>),
}

/// Spatial pooling mode over a patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolingMode {
    /// Average the patch grid.
    Mean,
    /// Sum the patch grid.
    Sum,
    /// Concatenate the patch grid without reducing it.
    Concat,
}

impl fmt::Display for PoolingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Sum => write!(f, "sum"),
            Self::Concat => write!(f, "concat"),
        }
    }
}

impl FromStr for PoolingMode {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "concat" => Ok(Self::Concat),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "pooling_mode",
                name: s.to_string(),
            }),
        }
    }
}

/// Patch aggregation mode before the final projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatchAggregationMode {
    /// Concatenate the patch embeddings.
    Concat,
    /// Average the patch embeddings.
    Mean,
    /// Sum the patch embeddings.
    Sum,
}

impl fmt::Display for PatchAggregationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concat => write!(f, "concat"),
            Self::Mean => write!(f, "mean"),
            Self::Sum => write!(f, "sum"),
        }
    }
}

impl FromStr for PatchAggregationMode {
    type Err = ArrangementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "concat" => Ok(Self::Concat),
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            _ => Err(ArrangementError::UnknownStrategy {
                primitive: "patch_aggregation_mode",
                name: s.to_string(),
            }),
        }
    }
}

/// Extract non-overlapping square patches from image-like inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Patchify {
    patch_size: usize,
    flatten_channels: bool,
}

impl Patchify {
    /// Create a new patch extractor.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn new(patch_size: usize, flatten_channels: bool) -> Result<Self, ArrangementError> {
        if patch_size == 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "patch_size",
                reason: "must be at least 1".to_string(),
            });
        }

        Ok(Self {
            patch_size,
            flatten_channels,
        })
    }

    /// Return the configured patch size.
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Whether `forward()` returns flattened patches.
    pub fn flatten_channels(&self) -> bool {
        self.flatten_channels
    }

    /// Return the patch-grid layout `(patch_rows, patch_cols)` for an input.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn spatial_layout_for(&self, x: &Array4<f64>) -> Result<(usize, usize), ArrangementError> {
        let (_, _, height, width) = x.dim();
        self.validate_spatial_dims(height, width)?;
        Ok((height / self.patch_size, width / self.patch_size))
    }

    /// Return the total number of patches extracted from an input.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn num_patches_for(&self, x: &Array4<f64>) -> Result<usize, ArrangementError> {
        let (patch_rows, patch_cols) = self.spatial_layout_for(x)?;
        Ok(patch_rows * patch_cols)
    }

    /// Extract flattened patches of shape `(batch, num_patches, channels * p * p)`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_flat(&self, x: &Array4<f64>) -> Result<Array3<f64>, ArrangementError> {
        let (batch, channels, height, width) = x.dim();
        let (patch_rows, patch_cols) = self.spatial_layout_for(x)?;
        let patch_area = self.patch_size * self.patch_size;
        let num_patches = patch_rows * patch_cols;
        let patch_dim = channels * patch_area;
        let tiled = x
            .to_shape((
                batch,
                channels,
                patch_rows,
                self.patch_size,
                patch_cols,
                self.patch_size,
            ))
            .map_err(|_| ArrangementError::ShapeMismatch {
                expected: format!(
                    "input reshape ({batch}, {channels}, {patch_rows}, {}, {patch_cols}, {})",
                    self.patch_size, self.patch_size
                ),
                got: format!("{:?}", x.dim()),
            })?;
        let patch_major = tiled.permuted_axes([0, 2, 4, 1, 3, 5]);
        let patches = patch_major
            .as_standard_layout()
            .into_shape_with_order((batch, num_patches, patch_dim))
            .map_err(|_| ArrangementError::ShapeMismatch {
                expected: format!("flattened patches ({batch}, {num_patches}, {patch_dim})"),
                got: format!("{:?}", patch_major.dim()),
            })?
            .into_owned();

        debug_assert_eq!(
            num_patches,
            (height / self.patch_size) * (width / self.patch_size)
        );

        Ok(patches)
    }

    /// Extract channel-preserving patches of shape `(batch, num_patches, channels, p * p)`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_channelwise(&self, x: &Array4<f64>) -> Result<Array4<f64>, ArrangementError> {
        let (batch, channels, _, _) = x.dim();
        let (patch_rows, patch_cols) = self.spatial_layout_for(x)?;
        let patch_area = self.patch_size * self.patch_size;
        let num_patches = patch_rows * patch_cols;
        let mut patches = Array4::<f64>::zeros((batch, num_patches, channels, patch_area));

        for batch_idx in 0..batch {
            for patch_row in 0..patch_rows {
                for patch_col in 0..patch_cols {
                    let patch_idx = patch_row * patch_cols + patch_col;
                    let patch = x.slice(s![
                        batch_idx,
                        ..,
                        patch_row * self.patch_size..(patch_row + 1) * self.patch_size,
                        patch_col * self.patch_size..(patch_col + 1) * self.patch_size
                    ]);

                    for channel_idx in 0..channels {
                        let channel_patch = patch.slice(s![channel_idx, .., ..]);
                        for (flat_idx, value) in channel_patch.iter().enumerate() {
                            patches[(batch_idx, patch_idx, channel_idx, flat_idx)] = *value;
                        }
                    }
                }
            }
        }

        Ok(patches)
    }

    /// Extract patches according to `flatten_channels`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(&self, x: &Array4<f64>) -> Result<PatchifyOutput, ArrangementError> {
        if self.flatten_channels {
            Ok(PatchifyOutput::Flattened(self.forward_flat(x)?))
        } else {
            Ok(PatchifyOutput::Channelwise(self.forward_channelwise(x)?))
        }
    }

    fn validate_spatial_dims(&self, height: usize, width: usize) -> Result<(), ArrangementError> {
        if height == 0 || width == 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "input_shape",
                reason: "height and width must be positive".to_string(),
            });
        }

        if height % self.patch_size != 0 || width % self.patch_size != 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "patch_size",
                reason: format!(
                    "input spatial dimensions ({height}, {width}) must be divisible by patch_size {}",
                    self.patch_size
                ),
            });
        }

        Ok(())
    }
}

/// Project every patch through one shared state block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchProjector {
    block_idx: usize,
}

impl PatchProjector {
    /// Create a patch projector bound to one state block.
    pub fn new(block_idx: usize) -> Self {
        Self { block_idx }
    }

    /// Return the bound block index.
    pub fn block_idx(&self) -> usize {
        self.block_idx
    }

    /// Expected patch dimension for this projector.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InvalidParameter`] with `param: "block_idx"`
    /// when the configured block index is out of range for `state`.
    pub fn input_dim(&self, state: &CosmosState) -> Result<usize, ArrangementError> {
        Ok(get_w_block(state, self.block_idx)?.nrows())
    }

    /// Output subspace dimension for this projector.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InvalidParameter`] with `param: "block_idx"`
    /// when the configured block index is out of range for `state`.
    pub fn subspace_dim(&self, state: &CosmosState) -> Result<usize, ArrangementError> {
        Ok(get_w_block(state, self.block_idx)?.ncols())
    }

    /// Project a batch of flattened patches with shape `(batch, num_patches, n_patch)`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(
        &self,
        patches: &Array3<f64>,
        state: &CosmosState,
    ) -> Result<Array3<f64>, ArrangementError> {
        let w = get_w_block(state, self.block_idx)?;
        let (batch, num_patches, patch_dim) = patches.dim();

        if patch_dim != w.nrows() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("patch vectors with {} columns", w.nrows()),
                got: format!("({}, {}, {})", batch, num_patches, patch_dim),
            });
        }

        let mut projected = Array3::<f64>::zeros((batch, num_patches, w.ncols()));
        for batch_idx in 0..batch {
            let batch_patches = patches.slice(s![batch_idx, .., ..]);
            let batch_projected = batch_patches.dot(&w);
            projected
                .slice_mut(s![batch_idx, .., ..])
                .assign(&batch_projected);
        }

        Ok(projected)
    }

    /// Project one sample of patches with shape `(num_patches, n_patch)`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_one(
        &self,
        patches: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        let batch = patches.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

/// Deterministic pooling over a 2D patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpatialPooling {
    pool_factor: usize,
    mode: PoolingMode,
}

impl SpatialPooling {
    /// Create a new spatial pooling operator.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn new(pool_factor: usize, mode: PoolingMode) -> Result<Self, ArrangementError> {
        if pool_factor == 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "pool_factor",
                reason: "must be at least 1".to_string(),
            });
        }

        Ok(Self { pool_factor, mode })
    }

    /// Return the pooling factor.
    pub fn pool_factor(&self) -> usize {
        self.pool_factor
    }

    /// Return the pooling mode.
    pub fn mode(&self) -> PoolingMode {
        self.mode
    }

    /// Compute the pooled spatial layout.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn pooled_layout(
        &self,
        spatial_layout: (usize, usize),
    ) -> Result<(usize, usize), ArrangementError> {
        let (height, width) = spatial_layout;
        if height == 0 || width == 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "spatial_layout",
                reason: "height and width must be positive".to_string(),
            });
        }

        if height % self.pool_factor != 0 || width % self.pool_factor != 0 {
            return Err(ArrangementError::InvalidParameter {
                param: "pool_factor",
                reason: format!(
                    "spatial layout ({height}, {width}) must be divisible by pool_factor {}",
                    self.pool_factor
                ),
            });
        }

        Ok((height / self.pool_factor, width / self.pool_factor))
    }

    /// Pool patch features of shape `(batch, num_patches, k)`.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(
        &self,
        z_patches: &Array3<f64>,
        spatial_layout: (usize, usize),
    ) -> Result<Array3<f64>, ArrangementError> {
        let (batch, num_patches, k) = z_patches.dim();
        let (height, width) = spatial_layout;
        if height * width != num_patches {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("spatial layout with {} total patches", num_patches),
                got: format!("({height}, {width})"),
            });
        }

        let (pooled_height, pooled_width) = self.pooled_layout(spatial_layout)?;
        let pooled_patches = pooled_height * pooled_width;
        let output_dim = match self.mode {
            PoolingMode::Mean | PoolingMode::Sum => k,
            PoolingMode::Concat => self.pool_factor * self.pool_factor * k,
        };
        let mut pooled = Array3::<f64>::zeros((batch, pooled_patches, output_dim));

        for batch_idx in 0..batch {
            for pooled_row in 0..pooled_height {
                for pooled_col in 0..pooled_width {
                    let pooled_idx = pooled_row * pooled_width + pooled_col;
                    match self.mode {
                        PoolingMode::Mean | PoolingMode::Sum => {
                            for feature_idx in 0..k {
                                let mut acc = 0.0;
                                for delta_row in 0..self.pool_factor {
                                    for delta_col in 0..self.pool_factor {
                                        let patch_row = pooled_row * self.pool_factor + delta_row;
                                        let patch_col = pooled_col * self.pool_factor + delta_col;
                                        let patch_idx = patch_row * width + patch_col;
                                        acc += z_patches[(batch_idx, patch_idx, feature_idx)];
                                    }
                                }
                                pooled[(batch_idx, pooled_idx, feature_idx)] =
                                    if self.mode == PoolingMode::Mean {
                                        acc / (self.pool_factor * self.pool_factor) as f64
                                    } else {
                                        acc
                                    };
                            }
                        }
                        PoolingMode::Concat => {
                            let mut concat_idx = 0usize;
                            for delta_row in 0..self.pool_factor {
                                for delta_col in 0..self.pool_factor {
                                    let patch_row = pooled_row * self.pool_factor + delta_row;
                                    let patch_col = pooled_col * self.pool_factor + delta_col;
                                    let patch_idx = patch_row * width + patch_col;
                                    for feature_idx in 0..k {
                                        pooled[(batch_idx, pooled_idx, concat_idx)] =
                                            z_patches[(batch_idx, patch_idx, feature_idx)];
                                        concat_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(pooled)
    }
}

/// Aggregate a patch set and project it with a state block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchAggregator {
    block_idx: usize,
    mode: PatchAggregationMode,
}

impl PatchAggregator {
    /// Create a new patch aggregator.
    pub fn new(block_idx: usize, mode: PatchAggregationMode) -> Self {
        Self { block_idx, mode }
    }

    /// Return the aggregation block index.
    pub fn block_idx(&self) -> usize {
        self.block_idx
    }

    /// Return the aggregation mode.
    pub fn mode(&self) -> PatchAggregationMode {
        self.mode
    }

    /// Aggregate patches before the final projection.
    ///
    /// # Panics
    ///
    /// Panics if the reshape from `(batch, num_patches, k)` to
    /// `(batch, num_patches * k)` fails, which cannot happen for a
    /// standard-layout input.
    pub fn aggregate(&self, z_patches: &Array3<f64>) -> Array2<f64> {
        let (batch, num_patches, k) = z_patches.dim();
        match self.mode {
            PatchAggregationMode::Concat => z_patches
                .as_standard_layout()
                .into_shape_with_order((batch, num_patches * k))
                .expect("standard-layout patch dimensions preserve the element count")
                .into_owned(),
            PatchAggregationMode::Mean => {
                let mut aggregated = Array2::<f64>::zeros((batch, k));
                for batch_idx in 0..batch {
                    for feature_idx in 0..k {
                        let mut acc = 0.0;
                        for patch_idx in 0..num_patches {
                            acc += z_patches[(batch_idx, patch_idx, feature_idx)];
                        }
                        aggregated[(batch_idx, feature_idx)] = acc / num_patches as f64;
                    }
                }
                aggregated
            }
            PatchAggregationMode::Sum => {
                let mut aggregated = Array2::<f64>::zeros((batch, k));
                for batch_idx in 0..batch {
                    for feature_idx in 0..k {
                        let mut acc = 0.0;
                        for patch_idx in 0..num_patches {
                            acc += z_patches[(batch_idx, patch_idx, feature_idx)];
                        }
                        aggregated[(batch_idx, feature_idx)] = acc;
                    }
                }
                aggregated
            }
        }
    }

    /// Aggregate and project a batch of patch embeddings.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(
        &self,
        z_patches: &Array3<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        let flat = self.aggregate(z_patches);
        let w = get_w_block(state, self.block_idx)?;

        if flat.ncols() != w.nrows() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("aggregated patches with {} columns", w.nrows()),
                got: format!("({}, {})", flat.nrows(), flat.ncols()),
            });
        }

        Ok(flat.dot(&w))
    }

    /// Aggregate and project one sample of patch embeddings.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_one(
        &self,
        z_patches: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = z_patches.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, Array4, array};

    use super::*;

    const EPSILON: f64 = 1e-10;

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

    fn assert_array3_close(actual: &Array3<f64>, expected: &Array3<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn assert_array2_bit_exact(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "bit mismatch at flat index {index}"
            );
        }
    }

    fn assert_array3_bit_exact(actual: &Array3<f64>, expected: &Array3<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "bit mismatch at flat index {index}"
            );
        }
    }

    fn build_state(blocks: &[Array2<f64>]) -> CosmosState {
        let mut state = CosmosState::new(&[]).unwrap();
        for block in blocks {
            state.register_external_block(block).unwrap();
        }
        state
    }

    fn patch_input() -> Array4<f64> {
        array![[[[1.0, 2.0], [3.0, 4.0]]]]
    }

    fn two_by_two_patch_grid() -> Array3<f64> {
        array![[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],]]
    }

    #[test]
    fn patchify_flat_extracts_non_overlapping_patches() {
        let patchify = Patchify::new(1, true).unwrap();
        let patches = patchify.forward_flat(&patch_input()).unwrap();
        let expected = array![[[1.0], [2.0], [3.0], [4.0]]];

        assert_array3_close(&patches, &expected);
    }

    #[test]
    fn patchify_flat_matches_scalar_reference_for_realistic_layouts() {
        for &(batch, channels, height, width, patch_size) in &[(3, 3, 8, 8, 4), (2, 1, 28, 28, 4)] {
            let x = Array4::from_shape_fn((batch, channels, height, width), |(b, c, i, j)| {
                let index = ((b * channels + c) * height + i) * width + j;
                (index as f64 - 1024.0) / 8.0
            });
            let patchify = Patchify::new(patch_size, true).unwrap();
            let actual = patchify.forward_flat(&x).unwrap();

            let patch_rows = height / patch_size;
            let patch_cols = width / patch_size;
            let patch_dim = channels * patch_size * patch_size;
            let mut expected = Array3::<f64>::zeros((batch, patch_rows * patch_cols, patch_dim));
            for batch_idx in 0..batch {
                for patch_row in 0..patch_rows {
                    for patch_col in 0..patch_cols {
                        let patch_idx = patch_row * patch_cols + patch_col;
                        let patch = x.slice(ndarray::s![
                            batch_idx,
                            ..,
                            patch_row * patch_size..(patch_row + 1) * patch_size,
                            patch_col * patch_size..(patch_col + 1) * patch_size
                        ]);
                        for (flat_idx, value) in patch.iter().enumerate() {
                            expected[(batch_idx, patch_idx, flat_idx)] = *value;
                        }
                    }
                }
            }

            assert_array3_bit_exact(&actual, &expected);
        }
    }

    #[test]
    fn patchify_channelwise_preserves_channel_axis() {
        let patchify = Patchify::new(1, false).unwrap();
        let output = patchify.forward(&patch_input()).unwrap();

        match output {
            PatchifyOutput::Channelwise(patches) => {
                assert_eq!(patches.dim(), (1, 4, 1, 1));
                assert!((patches[(0, 2, 0, 0)] - 3.0).abs() < EPSILON);
            }
            other => panic!("unexpected output: {other:?}"),
        }
    }

    #[test]
    fn patchify_rejects_non_divisible_spatial_dimensions() {
        let patchify = Patchify::new(2, true).unwrap();
        let x = Array4::<f64>::zeros((1, 1, 3, 2));

        let err = patchify.forward(&x).unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => {
                assert_eq!(param, "patch_size");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn patch_projector_projects_each_patch() {
        let state = build_state(&[array![[1.0], [0.0]]]);
        let projector = PatchProjector::new(0);
        let patches = array![[[1.0, 5.0], [2.0, 6.0]]];
        let expected = array![[[1.0], [2.0]]];

        let actual = projector.forward(&patches, &state).unwrap();

        assert_array3_close(&actual, &expected);
    }

    #[test]
    fn patch_projector_rejects_wrong_patch_dimension() {
        let state = build_state(&[array![[1.0], [0.0]]]);
        let projector = PatchProjector::new(0);
        let err = projector
            .forward(&Array3::<f64>::zeros((1, 2, 3)), &state)
            .unwrap_err();

        match err {
            ArrangementError::ShapeMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn spatial_pooling_mean_reduces_each_window() {
        let pooling = SpatialPooling::new(2, PoolingMode::Mean).unwrap();
        let pooled = pooling.forward(&two_by_two_patch_grid(), (2, 2)).unwrap();
        let expected = array![[[2.5, 25.0]]];

        assert_array3_close(&pooled, &expected);
    }

    #[test]
    fn spatial_pooling_concat_flattens_window_in_row_major_order() {
        let pooling = SpatialPooling::new(2, PoolingMode::Concat).unwrap();
        let pooled = pooling.forward(&two_by_two_patch_grid(), (2, 2)).unwrap();
        let expected = array![[[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]]];

        assert_array3_close(&pooled, &expected);
    }

    #[test]
    fn spatial_pooling_rejects_incompatible_layout() {
        let pooling = SpatialPooling::new(2, PoolingMode::Mean).unwrap();
        let err = pooling
            .forward(&two_by_two_patch_grid(), (1, 4))
            .unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => {
                assert_eq!(param, "pool_factor");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn patch_aggregator_concat_projects_flattened_patch_grid() {
        let state = build_state(&[array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]);
        let aggregator = PatchAggregator::new(0, PatchAggregationMode::Concat);
        let z_patches = array![[[1.0, 2.0], [3.0, 4.0]]];

        let out = aggregator.forward(&z_patches, &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 2.0]]);
    }

    #[test]
    fn patch_aggregator_mean_reduces_patch_axis() {
        let state = build_state(&[array![[1.0], [0.0]]]);
        let aggregator = PatchAggregator::new(0, PatchAggregationMode::Mean);
        let z_patches = array![[[1.0, 10.0], [3.0, 30.0]]];

        let out = aggregator.forward(&z_patches, &state).unwrap();

        assert_array2_close(&out, &array![[2.0]]);
    }

    #[test]
    fn patch_aggregator_matches_scalar_reference_for_all_modes() {
        let (batch, num_patches, k) = (5, 7, 16);
        let z_patches = Array3::from_shape_fn((batch, num_patches, k), |(b, p, f)| {
            let index = (b * num_patches + p) * k + f;
            (index as f64 - 200.0) / 16.0
        });

        for mode in [
            PatchAggregationMode::Concat,
            PatchAggregationMode::Mean,
            PatchAggregationMode::Sum,
        ] {
            let actual = PatchAggregator::new(0, mode).aggregate(&z_patches);
            let mut expected = Array2::<f64>::zeros(match mode {
                PatchAggregationMode::Concat => (batch, num_patches * k),
                PatchAggregationMode::Mean | PatchAggregationMode::Sum => (batch, k),
            });
            match mode {
                PatchAggregationMode::Concat => {
                    for batch_idx in 0..batch {
                        let mut flat_idx = 0usize;
                        for patch_idx in 0..num_patches {
                            for feature_idx in 0..k {
                                expected[(batch_idx, flat_idx)] =
                                    z_patches[(batch_idx, patch_idx, feature_idx)];
                                flat_idx += 1;
                            }
                        }
                    }
                    assert_array2_bit_exact(&actual, &expected);
                }
                PatchAggregationMode::Mean | PatchAggregationMode::Sum => {
                    for batch_idx in 0..batch {
                        for feature_idx in 0..k {
                            let mut acc = 0.0;
                            for patch_idx in 0..num_patches {
                                acc += z_patches[(batch_idx, patch_idx, feature_idx)];
                            }
                            expected[(batch_idx, feature_idx)] =
                                if mode == PatchAggregationMode::Mean {
                                    acc / num_patches as f64
                                } else {
                                    acc
                                };
                        }
                    }
                    assert_eq!(actual.dim(), expected.dim());
                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!((a - e).abs() <= 1e-12, "expected {e}, got {a}");
                    }
                }
            }
        }
    }

    #[test]
    fn pooling_and_aggregation_mode_roundtrip() {
        assert_eq!("mean".parse::<PoolingMode>().unwrap(), PoolingMode::Mean);
        assert_eq!(
            "concat".parse::<PatchAggregationMode>().unwrap(),
            PatchAggregationMode::Concat
        );
        assert_eq!(PoolingMode::Concat.to_string(), "concat");
        assert_eq!(PatchAggregationMode::Sum.to_string(), "sum");
    }
}
