//! Internal compute layer for COSMOS kernels.
//!
//! This module hosts repository-local numerical kernels that the higher-level
//! arrangement, model, and training layers can reuse without centering the
//! architecture around an external backend framework.
//!
//! Kernels are generic over [`Scalar`], which the released crate implements for
//! `f64` alone: COSMOS runs every path in double precision.

use std::iter::Sum;

use ndarray::LinalgScalar;
use num_traits::Float;

pub mod activation;
pub mod device;
pub mod kernels;
pub mod workspace;

/// Scalar type shared by all COSMOS compute kernels.
///
/// `f64` is the only implementation. COSMOS mandates double precision on every
/// geometry-critical path — QR, SVD, retraction, matrix exponential — and the
/// paper protocol runs the whole forward and backward in `f64` as well. The
/// trait is kept as the type seam a mixed-precision backend would widen.
pub trait Scalar:
    LinalgScalar + Float + Sum + std::fmt::Debug + std::fmt::Display + Send + Sync + 'static
{
    /// Upcast to `f64` for geometry boundary crossings.
    fn to_f64(self) -> f64;
    /// Downcast from `f64`.
    fn from_f64(value: f64) -> Self;
}

impl Scalar for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
}

pub use activation::{Activation, ActivationPlacement};
pub use device::cpu::CpuDevice;
pub use device::{Buffer1, Buffer2, ComputeDevice, Device, DeviceKind};
pub use kernels::cmr::{
    cmr_anchor_backward, cmr_multi_anchor_backward, cmr_multi_anchor_backward_device,
};
pub use kernels::projector::{
    project_rows, projector_backward_projected, projector_backward_projected_device,
    projector_backward_subspace, projector_backward_subspace_device, projector_basis_gradient,
    projector_basis_gradient_device,
};
pub use workspace::{
    CmrAnchorBackwardWorkspace, CmrMultiAnchorBackwardWorkspace, ProjectorKernelWorkspace,
};
