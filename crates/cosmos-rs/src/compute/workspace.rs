//! Reusable CPU workspaces for COSMOS kernels.

use ndarray::{Array1, Array2};

use super::Scalar;

/// Reusable buffers for the CMR anchor backward kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct CmrAnchorBackwardWorkspace<S: Scalar = f64> {
    pub(crate) coords: Array2<S>,
    pub(crate) projected: Array2<S>,
    pub(crate) projected_norms: Array1<S>,
    pub(crate) denom_scale: Array1<S>,
    pub(crate) correction: Array1<S>,
    pub(crate) scaled_coords: Array2<S>,
}

impl<S: Scalar> Default for CmrAnchorBackwardWorkspace<S> {
    fn default() -> Self {
        Self {
            coords: Array2::zeros((0, 0)),
            projected: Array2::zeros((0, 0)),
            projected_norms: Array1::zeros(0),
            denom_scale: Array1::zeros(0),
            correction: Array1::zeros(0),
            scaled_coords: Array2::zeros((0, 0)),
        }
    }
}

impl<S: Scalar> CmrAnchorBackwardWorkspace<S> {
    /// Resize all buffers to match the current batch and anchor shapes.
    pub fn resize(&mut self, batch_size: usize, embedding_dim: usize, anchor_dim: usize) {
        if self.coords.dim() != (batch_size, anchor_dim) {
            self.coords = Array2::zeros((batch_size, anchor_dim));
        }
        if self.projected.dim() != (batch_size, embedding_dim) {
            self.projected = Array2::zeros((batch_size, embedding_dim));
        }
        if self.projected_norms.len() != batch_size {
            self.projected_norms = Array1::zeros(batch_size);
        }
        if self.denom_scale.len() != batch_size {
            self.denom_scale = Array1::zeros(batch_size);
        }
        if self.correction.len() != batch_size {
            self.correction = Array1::zeros(batch_size);
        }
        if self.scaled_coords.dim() != (batch_size, anchor_dim) {
            self.scaled_coords = Array2::zeros((batch_size, anchor_dim));
        }
    }
}

/// Reusable buffers for the fused multi-anchor CMR backward kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct CmrMultiAnchorBackwardWorkspace<S: Scalar = f64> {
    pub(crate) stacked_anchors: Array2<S>,
    pub(crate) stacked_anchor_t: Array2<S>,
    pub(crate) coords: Array2<S>,
    pub(crate) scaled_coords: Array2<S>,
    pub(crate) projected: Array2<S>,
    pub(crate) correction: Array1<S>,
}

impl<S: Scalar> Default for CmrMultiAnchorBackwardWorkspace<S> {
    fn default() -> Self {
        Self {
            stacked_anchors: Array2::zeros((0, 0)),
            stacked_anchor_t: Array2::zeros((0, 0)),
            coords: Array2::zeros((0, 0)),
            scaled_coords: Array2::zeros((0, 0)),
            projected: Array2::zeros((0, 0)),
            correction: Array1::zeros(0),
        }
    }
}

impl<S: Scalar> CmrMultiAnchorBackwardWorkspace<S> {
    /// Resize all buffers to match the current batch and full anchor stack.
    pub fn resize(
        &mut self,
        batch_size: usize,
        embedding_dim: usize,
        anchor_dim: usize,
        num_anchors: usize,
    ) {
        let total_anchor_dim = anchor_dim * num_anchors;
        if self.stacked_anchors.dim() != (embedding_dim, total_anchor_dim) {
            self.stacked_anchors = Array2::zeros((embedding_dim, total_anchor_dim));
        }
        if self.stacked_anchor_t.dim() != (total_anchor_dim, embedding_dim) {
            self.stacked_anchor_t = Array2::zeros((total_anchor_dim, embedding_dim));
        }
        if self.coords.dim() != (batch_size, total_anchor_dim) {
            self.coords = Array2::zeros((batch_size, total_anchor_dim));
        }
        if self.scaled_coords.dim() != (batch_size, total_anchor_dim) {
            self.scaled_coords = Array2::zeros((batch_size, total_anchor_dim));
        }
        if self.projected.dim() != (batch_size, embedding_dim) {
            self.projected = Array2::zeros((batch_size, embedding_dim));
        }
        if self.correction.len() != batch_size {
            self.correction = Array1::zeros(batch_size);
        }
    }
}

/// Reusable buffers for projector forward and backward kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectorKernelWorkspace<S: Scalar = f64> {
    pub(crate) row_buffer: Array2<S>,
    pub(crate) left_coords: Array2<S>,
    pub(crate) right_coords: Array2<S>,
}

impl<S: Scalar> Default for ProjectorKernelWorkspace<S> {
    fn default() -> Self {
        Self {
            row_buffer: Array2::zeros((0, 0)),
            left_coords: Array2::zeros((0, 0)),
            right_coords: Array2::zeros((0, 0)),
        }
    }
}

impl<S: Scalar> ProjectorKernelWorkspace<S> {
    /// Resize all buffers to match the current projector application.
    pub fn resize(&mut self, batch_size: usize, row_dim: usize, subspace_dim: usize) {
        if self.row_buffer.dim() != (batch_size, row_dim) {
            self.row_buffer = Array2::zeros((batch_size, row_dim));
        }
        if self.left_coords.dim() != (batch_size, subspace_dim) {
            self.left_coords = Array2::zeros((batch_size, subspace_dim));
        }
        if self.right_coords.dim() != (batch_size, subspace_dim) {
            self.right_coords = Array2::zeros((batch_size, subspace_dim));
        }
    }
}
