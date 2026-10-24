//! Numerical geometry underlying COSMOS.
//!
//! Every routine here is `f64` and CPU-only. The submodules are independent of
//! the rest of the crate and form a self-contained Grassmann toolkit:
//!
//! - [`grassmann`]: principal angles, distances, exponential and logarithmic
//!   maps, parallel transport, geodesic interpolation, retractions.
//! - [`matrix_exp`]: antisymmetric matrix exponentials, the Cayley transform,
//!   retraction onto the orthogonal group, and orthogonality error.
//! - [`qr`]: modified Gram-Schmidt QR and orthonormalization of a basis. With
//!   the `lapack` feature this dispatches to `ndarray-linalg`'s Householder QR.
//! - [`svd`]: singular value decomposition. With the `lapack` feature this
//!   dispatches to `ndarray-linalg`; by default it uses the in-crate
//!   one-sided Jacobi implementation.
//! - [`spectrum`]: symmetric and skew-symmetric eigenvalue problems.
//! - [`lie_groups`]: geodesic interpolation on `SO(n)`.

pub mod grassmann;
pub mod lie_groups;
pub mod matrix_exp;
pub mod qr;
pub mod spectrum;
pub mod svd;
