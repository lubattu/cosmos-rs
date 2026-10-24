//! Device abstraction for COSMOS compute kernels.
//!
//! [`Buffer2`] and [`Buffer1`] are the storage handles the kernels operate on;
//! [`ComputeDevice`] is the trait a backend implements and [`Device`] is the
//! concrete handle threaded through models, adapters and the trainer.
//!
//! The released crate ships one backend, [`CpuDevice`](cpu::CpuDevice), and one
//! scalar type, `f64`: the paper protocol is CPU double precision throughout.
//! The trait and the [`Device`] enum are kept as the seam a second backend would
//! implement.

pub mod cpu;

use ndarray::{Array1, Array2};

use super::Scalar;

use crate::geometry::grassmann::GrassmannError;

// ---------------------------------------------------------------------------
// DeviceKind
// ---------------------------------------------------------------------------

/// Identifies which device family a buffer or device belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// Host memory, operated on by `ndarray`.
    Cpu,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "cpu"),
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer2 — 2-D storage
// ---------------------------------------------------------------------------

/// Handle to a 2-D matrix held by a compute device.
///
/// On the CPU backend this wraps an [`Array2<S>`] with zero overhead.
#[derive(Debug, Clone)]
pub struct Buffer2<S: Scalar> {
    inner: Array2<S>,
}

impl<S: Scalar> Buffer2<S> {
    /// Wrap a host-side `Array2` as a buffer (zero-copy).
    pub fn from_array(array: Array2<S>) -> Self {
        Self { inner: array }
    }

    /// Shape `(rows, cols)`.
    pub fn dim(&self) -> (usize, usize) {
        self.inner.dim()
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.dim().0
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.dim().1
    }

    /// Whether the matrix holds no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Device kind this buffer resides on.
    pub fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    /// Borrow the underlying `Array2`.
    pub fn as_array(&self) -> &Array2<S> {
        &self.inner
    }

    /// Mutably borrow the underlying `Array2`.
    pub fn as_array_mut(&mut self) -> &mut Array2<S> {
        &mut self.inner
    }

    /// Consume the buffer and return the underlying `Array2`.
    pub fn into_array(self) -> Array2<S> {
        self.inner
    }
}

impl<S: Scalar> From<Array2<S>> for Buffer2<S> {
    fn from(array: Array2<S>) -> Self {
        Self::from_array(array)
    }
}

// ---------------------------------------------------------------------------
// Buffer1 — 1-D storage
// ---------------------------------------------------------------------------

/// Handle to a 1-D vector held by a compute device.
#[derive(Debug, Clone)]
pub struct Buffer1<S: Scalar> {
    inner: Array1<S>,
}

impl<S: Scalar> Buffer1<S> {
    /// Wrap a host-side `Array1` as a buffer (zero-copy).
    pub fn from_array(array: Array1<S>) -> Self {
        Self { inner: array }
    }

    /// Length of the vector.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Device kind this buffer resides on.
    pub fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    /// Borrow the underlying `Array1`.
    pub fn as_array(&self) -> &Array1<S> {
        &self.inner
    }

    /// Mutably borrow the underlying `Array1`.
    pub fn as_array_mut(&mut self) -> &mut Array1<S> {
        &mut self.inner
    }

    /// Consume the buffer and return the underlying `Array1`.
    pub fn into_array(self) -> Array1<S> {
        self.inner
    }
}

impl<S: Scalar> From<Array1<S>> for Buffer1<S> {
    fn from(array: Array1<S>) -> Self {
        Self::from_array(array)
    }
}

// ---------------------------------------------------------------------------
// ComputeDevice trait
// ---------------------------------------------------------------------------

/// Backend-agnostic compute interface for COSMOS kernels.
///
/// A `ComputeDevice` manages buffer allocation, host↔device transfer, and the
/// core numerical operations needed by the training loop. The CPU
/// implementation delegates to `ndarray`.
pub trait ComputeDevice: Send + Sync + std::fmt::Debug {
    /// Which device family this is.
    fn kind(&self) -> DeviceKind;

    // -- allocation --------------------------------------------------------

    /// Allocate a zero-filled 2-D buffer.
    fn zeros_2d<S: Scalar>(&self, rows: usize, cols: usize) -> Buffer2<S>;

    /// Allocate a zero-filled 1-D buffer.
    fn zeros_1d<S: Scalar>(&self, len: usize) -> Buffer1<S>;

    // -- transfer ----------------------------------------------------------

    /// Upload a host `Array2` to device memory.
    fn upload_2d<S: Scalar>(&self, data: &Array2<S>) -> Buffer2<S>;

    /// Download a device buffer to a host `Array2`.
    fn download_2d<S: Scalar>(&self, buf: &Buffer2<S>) -> Array2<S>;

    /// Upload a host `Array1` to device memory.
    fn upload_1d<S: Scalar>(&self, data: &Array1<S>) -> Buffer1<S>;

    /// Download a device buffer to a host `Array1`.
    fn download_1d<S: Scalar>(&self, buf: &Buffer1<S>) -> Array1<S>;

    // -- BLAS-like ---------------------------------------------------------

    /// General matrix multiply: `C = alpha * A @ B + beta * C`.
    fn gemm<S: Scalar>(
        &self,
        alpha: S,
        a: &Buffer2<S>,
        b: &Buffer2<S>,
        beta: S,
        c: &mut Buffer2<S>,
    );

    /// Matrix product `A @ B`, returning a new buffer.
    fn matmul<S: Scalar>(&self, a: &Buffer2<S>, b: &Buffer2<S>) -> Buffer2<S> {
        let (m, _) = a.dim();
        let (_, n) = b.dim();
        let mut c = self.zeros_2d(m, n);
        self.gemm(S::one(), a, b, S::zero(), &mut c);
        c
    }

    /// Owned transpose.
    fn transpose<S: Scalar>(&self, a: &Buffer2<S>) -> Buffer2<S>;

    // -- projections -------------------------------------------------------

    /// Project rows: `X @ W @ W^T`.
    fn project_rows<S: Scalar>(&self, x: &Buffer2<S>, basis: &Buffer2<S>) -> Buffer2<S> {
        let coords = self.matmul(x, basis);
        let basis_t = self.transpose(basis);
        self.matmul(&coords, &basis_t)
    }

    // -- reductions --------------------------------------------------------

    /// Per-row squared norms: for each row `r`, compute `sum(r_i^2)`.
    fn row_norms_sq<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S>;

    /// Sum all elements of a 1-D buffer.
    fn sum_1d<S: Scalar>(&self, x: &Buffer1<S>) -> S;

    /// Mean of a 1-D buffer. Returns zero for empty buffers.
    fn mean_1d<S: Scalar>(&self, x: &Buffer1<S>) -> S {
        if x.is_empty() {
            S::zero()
        } else {
            self.sum_1d(x) / S::from_f64(x.len() as f64)
        }
    }

    // -- element-wise operations -----------------------------------------------

    /// Per-row L2 norms: `sqrt(sum(r_i^2))` for each row.
    fn row_norms<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S>;

    /// In-place element-wise multiply: `a *= b`.
    fn elementwise_mul_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>);

    /// Element-wise multiply returning a new buffer: `a * b`.
    fn elementwise_mul<S: Scalar>(&self, a: &Buffer2<S>, b: &Buffer2<S>) -> Buffer2<S> {
        let mut out = self.clone_buffer(a);
        self.elementwise_mul_assign(&mut out, b);
        out
    }

    /// In-place accumulate: `a += b`.
    fn add_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>);

    /// In-place scaled accumulate: `a += alpha * b`.
    fn add_scaled<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S, b: &Buffer2<S>);

    /// In-place scalar multiply: `a *= alpha`.
    fn scale_inplace<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S);

    /// Clone a buffer on the same device.
    fn clone_buffer<S: Scalar>(&self, src: &Buffer2<S>) -> Buffer2<S>;

    /// Multiply each row of `mat` element-wise by the corresponding scalar in `col`.
    ///
    /// `mat` has shape `(rows, cols)`, `col` has length `rows`.
    /// Result: `out[i, j] = mat[i, j] * col[i]`.
    fn broadcast_col_mul<S: Scalar>(&self, mat: &Buffer2<S>, col: &Buffer1<S>) -> Buffer2<S>;

    /// Subtract from `a` the element-wise product `b * col` (broadcast per row).
    ///
    /// `a[i, j] -= b[i, j] * col[i]`.
    fn sub_broadcast_col_mul<S: Scalar>(
        &self,
        a: &mut Buffer2<S>,
        b: &Buffer2<S>,
        col: &Buffer1<S>,
    );

    /// Assign into a column slice of `dst`: `dst[:, start..start+src.ncols()] = src`.
    fn assign_cols<S: Scalar>(&self, dst: &mut Buffer2<S>, start: usize, src: &Buffer2<S>);

    /// Extract a column slice: `src[:, start..start+ncols]`.
    fn slice_cols<S: Scalar>(&self, src: &Buffer2<S>, start: usize, ncols: usize) -> Buffer2<S>;

    // -- normalization ---------------------------------------------------------

    /// Element-wise reciprocal of a 1-D buffer: `out[i] = 1 / x[i]`.
    fn reciprocal_1d<S: Scalar>(&self, x: &Buffer1<S>) -> Buffer1<S>;

    /// Row-normalize a matrix: divide each row by its L2 norm.
    ///
    /// Returns the normalized matrix. Rows with zero norm are left as-is.
    fn normalize_rows<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer2<S> {
        let norms = self.row_norms(x);
        let inv_norms = self.reciprocal_1d(&norms);
        self.broadcast_col_mul(x, &inv_norms)
    }

    // -- activations -----------------------------------------------------------

    /// Apply an activation function element-wise in-place.
    ///
    /// Operates on `f64`, the crate's only scalar type; every transcendental
    /// function (GELU, SiLU, etc.) is evaluated at full double precision.
    fn apply_activation<S: Scalar>(
        &self,
        buf: &mut Buffer2<S>,
        activation: crate::compute::activation::Activation,
    );

    /// Compute the element-wise derivative of an activation at the given input.
    ///
    /// Returns a new buffer where each element is `activation.derivative(input[i,j])`.
    fn activation_derivative<S: Scalar>(
        &self,
        input: &Buffer2<S>,
        activation: crate::compute::activation::Activation,
    ) -> Buffer2<S>;

    // -- geodesic scoring ------------------------------------------------------

    /// Compute geodesic CMR scores for embeddings against a set of anchors.
    ///
    /// For each anchor `j` (shape D × k_j) and each sample `i`:
    ///   - `coords = embedding_i @ anchor_j`
    ///   - `proj_norm = ||coords||`
    ///   - `embed_norm = max(||embedding_i||, eps)`
    ///   - `angle = arccos(clamp(proj_norm / embed_norm, 0, 1))`
    ///   - `score = exp(-angle²)`
    ///
    /// Returns `(scores, angles)` both with shape `(N, num_anchors)`.
    fn compute_geodesic_scores<S: Scalar>(
        &self,
        embeddings: &Buffer2<S>,
        anchors: &[&Buffer2<S>],
        eps: S,
    ) -> (Buffer2<S>, Buffer2<S>);

    // -- geometry-aware compute paths --------------------------------------
    //
    // The methods below are f64-only — they target the geometry-critical
    // paths where COSMOS mandates f64. Each comes with a host default that
    // round-trips through the canonical `geometry::*` implementation;
    // backends with dedicated kernels override the default selectively.

    /// Grassmann exponential map at `w1` applied to tangent `h`.
    ///
    /// Default: download, call
    /// [`crate::geometry::grassmann::exponential_map`] with the
    /// default step `t = 1`, upload.
    ///
    /// # Errors
    ///
    /// Returns [`GrassmannError`] when the underlying exponential map rejects
    /// the basis or the tangent.
    fn grassmann_exp(
        &self,
        w1: &Buffer2<f64>,
        h: &Buffer2<f64>,
    ) -> Result<Buffer2<f64>, GrassmannError> {
        let w1h = self.download_2d(w1);
        let hh = self.download_2d(h);
        let cfg = crate::geometry::grassmann::ExponentialMapConfig::default();
        let r = crate::geometry::grassmann::exponential_map(&w1h, &hh, &cfg)?;
        Ok(self.upload_2d(&r))
    }
}

// ---------------------------------------------------------------------------
// Device — enum dispatch (avoids dyn + generic incompatibility)
// ---------------------------------------------------------------------------

/// Concrete device handle used throughout COSMOS.
///
/// Dispatches to the backend at runtime via match. On CPU the overhead is a
/// single branch per kernel call — negligible compared to the O(n²k) matrix
/// work inside each kernel. The enum is the declared seam where a second
/// backend would add its variant.
#[derive(Debug, Clone)]
pub enum Device {
    /// CPU backend backed by `ndarray`.
    Cpu(cpu::CpuDevice),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu(cpu::CpuDevice)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu(_) => write!(f, "cpu"),
        }
    }
}

impl ComputeDevice for Device {
    fn kind(&self) -> DeviceKind {
        match self {
            Device::Cpu(d) => d.kind(),
        }
    }

    fn zeros_2d<S: Scalar>(&self, rows: usize, cols: usize) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.zeros_2d(rows, cols),
        }
    }

    fn zeros_1d<S: Scalar>(&self, len: usize) -> Buffer1<S> {
        match self {
            Device::Cpu(d) => d.zeros_1d(len),
        }
    }

    fn upload_2d<S: Scalar>(&self, data: &Array2<S>) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.upload_2d(data),
        }
    }

    fn download_2d<S: Scalar>(&self, buf: &Buffer2<S>) -> Array2<S> {
        match self {
            Device::Cpu(d) => d.download_2d(buf),
        }
    }

    fn upload_1d<S: Scalar>(&self, data: &Array1<S>) -> Buffer1<S> {
        match self {
            Device::Cpu(d) => d.upload_1d(data),
        }
    }

    fn download_1d<S: Scalar>(&self, buf: &Buffer1<S>) -> Array1<S> {
        match self {
            Device::Cpu(d) => d.download_1d(buf),
        }
    }

    fn gemm<S: Scalar>(
        &self,
        alpha: S,
        a: &Buffer2<S>,
        b: &Buffer2<S>,
        beta: S,
        c: &mut Buffer2<S>,
    ) {
        match self {
            Device::Cpu(d) => d.gemm(alpha, a, b, beta, c),
        }
    }

    fn transpose<S: Scalar>(&self, a: &Buffer2<S>) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.transpose(a),
        }
    }

    fn row_norms_sq<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S> {
        match self {
            Device::Cpu(d) => d.row_norms_sq(x),
        }
    }

    fn sum_1d<S: Scalar>(&self, x: &Buffer1<S>) -> S {
        match self {
            Device::Cpu(d) => d.sum_1d(x),
        }
    }

    fn row_norms<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S> {
        match self {
            Device::Cpu(d) => d.row_norms(x),
        }
    }

    fn elementwise_mul_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>) {
        match self {
            Device::Cpu(d) => d.elementwise_mul_assign(a, b),
        }
    }

    fn add_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>) {
        match self {
            Device::Cpu(d) => d.add_assign(a, b),
        }
    }

    fn add_scaled<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S, b: &Buffer2<S>) {
        match self {
            Device::Cpu(d) => d.add_scaled(a, alpha, b),
        }
    }

    fn scale_inplace<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S) {
        match self {
            Device::Cpu(d) => d.scale_inplace(a, alpha),
        }
    }

    fn clone_buffer<S: Scalar>(&self, src: &Buffer2<S>) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.clone_buffer(src),
        }
    }

    fn broadcast_col_mul<S: Scalar>(&self, mat: &Buffer2<S>, col: &Buffer1<S>) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.broadcast_col_mul(mat, col),
        }
    }

    fn sub_broadcast_col_mul<S: Scalar>(
        &self,
        a: &mut Buffer2<S>,
        b: &Buffer2<S>,
        col: &Buffer1<S>,
    ) {
        match self {
            Device::Cpu(d) => d.sub_broadcast_col_mul(a, b, col),
        }
    }

    fn assign_cols<S: Scalar>(&self, dst: &mut Buffer2<S>, start: usize, src: &Buffer2<S>) {
        match self {
            Device::Cpu(d) => d.assign_cols(dst, start, src),
        }
    }

    fn slice_cols<S: Scalar>(&self, src: &Buffer2<S>, start: usize, ncols: usize) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.slice_cols(src, start, ncols),
        }
    }

    fn reciprocal_1d<S: Scalar>(&self, x: &Buffer1<S>) -> Buffer1<S> {
        match self {
            Device::Cpu(d) => d.reciprocal_1d(x),
        }
    }

    fn apply_activation<S: Scalar>(
        &self,
        buf: &mut Buffer2<S>,
        activation: crate::compute::activation::Activation,
    ) {
        match self {
            Device::Cpu(d) => d.apply_activation(buf, activation),
        }
    }

    fn activation_derivative<S: Scalar>(
        &self,
        input: &Buffer2<S>,
        activation: crate::compute::activation::Activation,
    ) -> Buffer2<S> {
        match self {
            Device::Cpu(d) => d.activation_derivative(input, activation),
        }
    }

    fn compute_geodesic_scores<S: Scalar>(
        &self,
        embeddings: &Buffer2<S>,
        anchors: &[&Buffer2<S>],
        eps: S,
    ) -> (Buffer2<S>, Buffer2<S>) {
        match self {
            Device::Cpu(d) => d.compute_geodesic_scores(embeddings, anchors, eps),
        }
    }

    // -- geometry-aware overrides — forward to the variant's own
    //    implementation so a backend override actually fires.

    fn grassmann_exp(
        &self,
        w1: &Buffer2<f64>,
        h: &Buffer2<f64>,
    ) -> Result<Buffer2<f64>, GrassmannError> {
        match self {
            Device::Cpu(d) => d.grassmann_exp(w1, h),
        }
    }
}

// ===========================================================================
// Tests for the geometry-aware device methods
// ===========================================================================

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::compute::device::cpu::CpuDevice;
    use ndarray::Array2;

    /// Deterministic orthonormal basis from QR of a seeded matrix.
    fn synth_basis(n: usize, k: usize, salt: usize) -> Array2<f64> {
        use crate::geometry::qr::{QrConfig, qr_orthonormalize};
        let mut w = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                w[[i, j]] = ((i * 7 + j * 13 + 3 + salt * 11) % 17) as f64 / 17.0
                    + if i == j { 0.5 } else { 0.0 };
            }
        }
        qr_orthonormalize(&w, &QrConfig::default())
    }

    /// `grassmann_exp` default matches host `exponential_map`.
    #[test]
    fn cpu_grassmann_exp_matches_host() {
        let dev = CpuDevice;
        let w1 = synth_basis(6, 2, 0);
        // Horizontal tangent: H − W1·(W1ᵀ·H).
        let h_raw = synth_basis(6, 2, 5);
        let m = w1.t().dot(&h_raw);
        let h = &h_raw - &w1.dot(&m);

        let w1b = dev.upload_2d(&w1);
        let hb = dev.upload_2d(&h);
        let r_dev = dev.grassmann_exp(&w1b, &hb).unwrap();
        let r = dev.download_2d(&r_dev);

        let cfg = crate::geometry::grassmann::ExponentialMapConfig::default();
        let r_ref = crate::geometry::grassmann::exponential_map(&w1, &h, &cfg).unwrap();
        let diff = (&r - &r_ref).mapv(f64::abs).sum();
        assert!(diff < 1e-12);
    }
}
