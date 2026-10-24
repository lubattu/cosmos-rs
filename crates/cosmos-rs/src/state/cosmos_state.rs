//! Unified global state for gauge-theoretic learning on the Grassmannian.
//!
//! This module implements the **S (State)** component of the COSMOS tuple `(S, A, O, R)`.
//!
//! The state centralizes all learnable Grassmann parameters as orthonormal
//! bases `W_i ∈ R^{n×k}` satisfying `W_i^T W_i = I` (Stiefel manifold
//! representatives). Associated projectors `P_i = W_i W_i^T` define
//! points on the Grassmann manifold `Gr(k, n)`.
//!
//! # Key types
//!
//! - [`CosmosState`] — the central state container holding all blocks.
//! - [`CosmosStateError`] — error enum for validation failures.
//! - [`CosmosStateDiagnostics`] — numerical health report.
//!
//! # Geometric invariants (Contract S)
//!
//! 1. **Orthonormality:** `W_i^T W_i = I_k` for every block.
//! 2. **Projector validity:** `P_i = W_i W_i^T` is symmetric, idempotent,
//!    with `trace(P_i) = k`.
//! 3. **Global coherence:** `P* = diag(P_1, ..., P_L)` is updated jointly
//!    via orthogonal conjugation.

use std::fmt;

use ndarray::{Array2, s};
use thiserror::Error;

use crate::compute::{Buffer2, ComputeDevice, Device};
use crate::geometry::matrix_exp::RetractionMethod;
use crate::geometry::qr::{QrConfig, qr_orthonormalize};
use crate::geometry::svd::{SvdConfig, svd};

/// Errors that can occur in [`CosmosState`].
#[derive(Debug, Error)]
pub enum CosmosStateError {
    /// A block's `(n, k)` shape violates `k ≤ n`.
    #[error("Block {block}: k ({k}) must be ≤ n ({n}) for Gr(k, n)")]
    InvalidShape {
        /// Index of the offending block.
        block: usize,
        /// Row dimension `n` of the block.
        n: usize,
        /// Column dimension `k` of the block.
        k: usize,
    },

    /// A block index was outside `[0, num_blocks)`.
    #[error("Block index {index} out of range [0, {num_blocks})")]
    BlockIndexOutOfRange {
        /// Index that was requested.
        index: usize,
        /// Number of blocks currently registered.
        num_blocks: usize,
    },

    /// A basis supplied for registration has `k > n`.
    #[error("W must be 2D with k ≤ n, got shape ({rows}, {cols})")]
    InvalidBasisShape {
        /// Row count of the supplied basis.
        rows: usize,
        /// Column count of the supplied basis.
        cols: usize,
    },

    /// A matrix's shape does not match the expected dimensions.
    #[error(
        "Shape mismatch: expected ({expected_rows}, {expected_cols}), got ({got_rows}, {got_cols})"
    )]
    ShapeMismatch {
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
        /// Row count that was actually supplied.
        got_rows: usize,
        /// Column count that was actually supplied.
        got_cols: usize,
    },
}

/// One deterministic pseudo-random value in `[-1, 1]`, advancing `state` in place.
///
/// SplitMix64 mixing step followed by a uniform-to-`[-1, 1]` rescale.
fn seeded_normal(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// CosmosState diagnostic information about numerical health.
#[derive(Debug)]
pub struct CosmosStateDiagnostics {
    /// Per-block orthogonality error `||W_i^T W_i - I||_F`.
    pub orthogonality_errors: Vec<f64>,
    /// RMS orthogonality error across all blocks.
    pub total_orth_error: f64,
    /// Exponential moving average of the commutator target per block.
    pub comm_target_ema: Vec<f64>,
    /// Block shapes `(n, k)`, each interpreted as `Gr(k, n)`.
    pub block_shapes: Vec<(usize, usize)>,
    /// Number of blocks in the state.
    pub num_blocks: usize,
}

/// Unified global state container for gauge-theoretic learning on the Grassmannian.
///
/// This is the S (State) component in the COSMOS tuple (S, A, O, R).
///
/// `CosmosState` centralizes all Grassmann learnable parameters as orthonormal bases
/// `W_i ∈ R^{n×k}` such that `W_i^T W_i = I`.
///
/// Each block defines a projector
///
/// `P_i = W_i W_i^T`
///
/// which represents a point on the Grassmann manifold `Gr(k, n)`.
///
/// # Geometric invariants
///
/// These properties are expected to hold throughout training:
///
/// 1. **Orthonormality:** `W_i^T W_i = I_k`
/// 2. **Projector validity:** `P_i = W_i W_i^T` satisfies:
///     - symmetry: `P_i^T = P_i`
///     - idempotence: `P_i^2 = P_i`
///     - rank: `trace(P_i) = k`
/// 3. **Global coherence:** the global state is organized as
///    `P* = diag(P_1, ..., P_L)`
///
/// # Notes
///
/// This type stores the bases `W_i` rather than materialized projectors,
/// since projectors can be reconstructed on demand.
///
/// Global updates may be applied by orthogonal conjugation while preserving
/// the geometric structure of the state.
#[derive(Debug)]
pub struct CosmosState {
    /// Orthonormal bases W_i ∈ R^{n_i × k_i}, one per block.
    ///
    /// Stored as device-resident buffers so an accelerated backend can keep the
    /// bases off-host without changing the public API.
    w_blocks: Vec<Buffer2<f64>>,

    /// Compute device that owns the buffers.
    device: Device,

    /// Block shapes `(n, k)`, each interpreted as `Gr(k, n)`.
    block_shapes: Vec<(usize, usize)>,

    /// (row_start, row_end, col_start, col_end) per block
    block_positions: Vec<(usize, usize, usize, usize)>,

    /// Total row dimension `Σ n_i`.
    total_n: usize,

    /// Total column dimension `Σ k_i`.
    total_k: usize,

    /// Exponential moving average of the commutator target per block.
    comm_target_ema: Vec<f64>,
}

impl CosmosState {
    // ─── Constructor ──────────────────────────────────────────────

    /// Create a new `CosmosState` with one orthonormal block per shape.
    ///
    /// Each block `W_i` is initialized as the first `k_i` columns of `I_{n_i}`.
    ///
    /// # Arguments
    ///
    /// * `projector_shapes` - List of `(n, k)` pairs, one per block.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::InvalidShape`] if any block has `k > n`.
    /// Create a new `CosmosState` on the default CPU device.
    ///
    /// Equivalent to `CosmosState::with_device(projector_shapes, Device::default())`.
    pub fn new(projector_shapes: &[(usize, usize)]) -> Result<Self, CosmosStateError> {
        Self::with_device(projector_shapes, Device::default())
    }

    /// Create a new `CosmosState` with a specific compute device.
    ///
    /// Each block `W_i` is initialized as the first `k_i` columns of `I_{n_i}`
    /// and uploaded to the given device.
    ///
    /// # Arguments
    ///
    /// * `projector_shapes` - List of `(n, k)` pairs, one per block.
    /// * `device` - Compute device that will own the block buffers.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::InvalidShape`] if any block has `k > n`.
    pub fn with_device(
        projector_shapes: &[(usize, usize)],
        device: Device,
    ) -> Result<Self, CosmosStateError> {
        for (i, &(n, k)) in projector_shapes.iter().enumerate() {
            if k > n {
                return Err(CosmosStateError::InvalidShape { block: i, n, k });
            }
        }

        let w_blocks: Vec<Buffer2<f64>> = projector_shapes
            .iter()
            .map(|&(n, k)| device.upload_2d(&Self::init_orthonormal(n, k)))
            .collect();

        let block_shapes: Vec<(usize, usize)> = projector_shapes.to_vec();
        let block_positions = Self::compute_block_positions(&block_shapes);
        let total_n = block_shapes.iter().map(|&(n, _)| n).sum();
        let total_k = block_shapes.iter().map(|&(_, k)| k).sum();
        let num_blocks = block_shapes.len();

        Ok(Self {
            w_blocks,
            device,
            block_shapes,
            block_positions,
            total_n,
            total_k,
            comm_target_ema: vec![0.0; num_blocks],
        })
    }

    // ─── Getters ──────────────────────────────────────────────────

    /// Return a reference to the orthonormal basis `W_i` for block `block_idx`.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::BlockIndexOutOfRange`] if the index is invalid.
    pub fn get_basis(&self, block_idx: usize) -> Result<&Array2<f64>, CosmosStateError> {
        if block_idx >= self.w_blocks.len() {
            return Err(CosmosStateError::BlockIndexOutOfRange {
                index: block_idx,
                num_blocks: self.w_blocks.len(),
            });
        }
        Ok(self.w_blocks[block_idx].as_array())
    }

    /// Compute the projector `P_i = W_i W_i^T` for block `block_idx`.
    ///
    /// Returns a new owned matrix of shape `(n_i, n_i)`.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::BlockIndexOutOfRange`] if the index is invalid.
    pub fn get_projector(&self, block_idx: usize) -> Result<Array2<f64>, CosmosStateError> {
        let w = self.get_basis(block_idx)?;
        Ok(w.dot(&w.t()))
    }

    /// Number of blocks in the state.
    pub fn num_blocks(&self) -> usize {
        self.w_blocks.len()
    }

    /// Block shapes as `(n, k)` pairs.
    pub fn block_shapes(&self) -> &[(usize, usize)] {
        &self.block_shapes
    }

    /// Total row dimension `Σ n_i`.
    pub fn total_n(&self) -> usize {
        self.total_n
    }

    /// Total column dimension `Σ k_i`.
    pub fn total_k(&self) -> usize {
        self.total_k
    }

    /// Return a reference to a block basis by index (alias for `get_basis`).
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::BlockIndexOutOfRange`] if the index is invalid.
    pub fn get_block_by_index(&self, idx: usize) -> Result<&Array2<f64>, CosmosStateError> {
        self.get_basis(idx)
    }

    /// Return a reference to the device-resident buffer for block `block_idx`.
    ///
    /// This is the device-aware counterpart of [`get_basis`](Self::get_basis).
    /// On CPU, `buf.as_array()` gives back the same `&Array2<f64>`.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::BlockIndexOutOfRange`] if the index is invalid.
    pub fn get_basis_buffer(&self, block_idx: usize) -> Result<&Buffer2<f64>, CosmosStateError> {
        if block_idx >= self.w_blocks.len() {
            return Err(CosmosStateError::BlockIndexOutOfRange {
                index: block_idx,
                num_blocks: self.w_blocks.len(),
            });
        }
        Ok(&self.w_blocks[block_idx])
    }

    /// The compute device that owns this state's buffers.
    pub fn device(&self) -> &Device {
        &self.device
    }

    // ─── Private helpers ──────────────────────────────────────────

    /// Compute `(row_start, row_end, col_start, col_end)` for each block.
    fn compute_block_positions(
        block_shapes: &[(usize, usize)],
    ) -> Vec<(usize, usize, usize, usize)> {
        let mut positions = Vec::with_capacity(block_shapes.len());
        let mut row_offset = 0;
        let mut col_offset = 0;
        for &(n, k) in block_shapes {
            positions.push((row_offset, row_offset + n, col_offset, col_offset + k));
            row_offset += n;
            col_offset += k;
        }
        positions
    }

    /// Initialize an orthonormal basis as the first `k` columns of `I_n`.
    fn init_orthonormal(n: usize, k: usize) -> Array2<f64> {
        let mut w = Array2::zeros((n, k));
        for j in 0..k {
            w[[j, j]] = 1.0;
        }
        w
    }

    /// Retract a single block back to the Stiefel manifold.
    fn orthonormalize_one(&mut self, i: usize, method: RetractionMethod) {
        let w = self.w_blocks[i].as_array();
        let retracted = match method {
            RetractionMethod::Qr => qr_orthonormalize(w, &QrConfig::default()),
            RetractionMethod::Svd | RetractionMethod::Polar => {
                let result = svd(w, &SvdConfig::default());
                result.u.dot(&result.vt)
            }
        };
        self.w_blocks[i] = Buffer2::from_array(retracted);
    }

    // ─── Assembly ─────────────────────────────────────────────────

    /// Assemble the block-diagonal global basis `W* = diag(W_1, ..., W_L)`.
    ///
    /// Returns a matrix of shape `(Σ n_i, Σ k_i)`.
    pub fn assemble_global_basis(&self) -> Array2<f64> {
        let mut w_star = Array2::zeros((self.total_n, self.total_k));
        for (i, buf) in self.w_blocks.iter().enumerate() {
            let (rs, re, cs, ce) = self.block_positions[i];
            w_star.slice_mut(s![rs..re, cs..ce]).assign(buf.as_array());
        }
        w_star
    }

    /// Assemble the block-diagonal global projector `P* = diag(P_1, ..., P_L)`.
    ///
    /// Each block `P_i = W_i W_i^T` is placed on the diagonal.
    /// Returns a matrix of shape `(Σ n_i, Σ n_i)`.
    pub fn assemble_global_projector(&self) -> Array2<f64> {
        let mut p_star = Array2::zeros((self.total_n, self.total_n));
        let mut row_offset = 0;
        for buf in &self.w_blocks {
            let w = buf.as_array();
            let n = w.nrows();
            let p = w.dot(&w.t());
            p_star
                .slice_mut(s![row_offset..row_offset + n, row_offset..row_offset + n])
                .assign(&p);
            row_offset += n;
        }
        p_star
    }

    // ─── Mutation ─────────────────────────────────────────────────

    /// Update individual blocks from a global basis matrix `W*`.
    ///
    /// This is the inverse of [`CosmosState::assemble_global_basis`]: it extracts each
    /// block slice and writes it into the corresponding `w_blocks[i]`.
    pub fn update_from_global_basis(&mut self, w_star: &Array2<f64>) {
        for (i, &(rs, re, cs, ce)) in self.block_positions.iter().enumerate() {
            self.w_blocks[i] = Buffer2::from_array(w_star.slice(s![rs..re, cs..ce]).to_owned());
        }
    }

    /// Retract all blocks back to the Stiefel manifold.
    ///
    /// After updates that may violate orthonormality, this enforces
    /// `W_i^T W_i = I` via QR or SVD-based retraction.
    pub fn orthonormalize_all(&mut self, method: RetractionMethod) {
        for i in 0..self.w_blocks.len() {
            self.orthonormalize_one(i, method);
        }
    }

    /// Retract specific blocks back to the Stiefel manifold.
    ///
    /// Blocks not in `block_indices` are left untouched.
    /// Out-of-range indices are silently ignored.
    pub fn orthonormalize_blocks(&mut self, block_indices: &[usize], method: RetractionMethod) {
        for &i in block_indices {
            if i < self.w_blocks.len() {
                self.orthonormalize_one(i, method);
            }
        }
    }

    // ─── Conjugation ──────────────────────────────────────────────

    /// Apply a global conjugation: `W*_new = U @ W*`.
    ///
    /// This implies `P*_new = U P* U^T` for the global projector.
    /// `U` must be orthogonal with shape `(Σ n_i, Σ n_i)`.
    pub fn apply_conjugation(&mut self, u: &Array2<f64>) {
        let w_star = self.assemble_global_basis();
        let w_star_new = u.dot(&w_star);
        self.update_from_global_basis(&w_star_new);
    }

    /// Apply separate conjugations per block: `W_i_new = U_i @ W_i`.
    ///
    /// Each `U_i` must be orthogonal with shape `(n_i, n_i)`.
    pub fn apply_block_conjugations(&mut self, u_blocks: &[Array2<f64>]) {
        for (i, u) in u_blocks.iter().enumerate() {
            self.w_blocks[i] = Buffer2::from_array(u.dot(self.w_blocks[i].as_array()));
        }
    }

    /// Apply one left action to a single block: `W_i_new = U @ W_i`.
    ///
    /// `u` must have shape `(n_i, n_i)` for the selected block.
    pub fn apply_block_left_action(&mut self, block_idx: usize, u: &Array2<f64>) {
        self.w_blocks[block_idx] = Buffer2::from_array(u.dot(self.w_blocks[block_idx].as_array()));
    }

    /// Replace a single block basis with an externally computed basis.
    ///
    /// The new basis must match the registered `(n_i, k_i)` shape for the block.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::BlockIndexOutOfRange`] if the index is invalid.
    /// Returns [`CosmosStateError::ShapeMismatch`] if `basis` does not match `(n_i, k_i)`.
    pub fn set_block_basis(
        &mut self,
        block_idx: usize,
        basis: Array2<f64>,
    ) -> Result<(), CosmosStateError> {
        if block_idx >= self.w_blocks.len() {
            return Err(CosmosStateError::BlockIndexOutOfRange {
                index: block_idx,
                num_blocks: self.w_blocks.len(),
            });
        }

        let (expected_rows, expected_cols) = self.block_shapes[block_idx];
        let (got_rows, got_cols) = basis.dim();
        if (expected_rows, expected_cols) != (got_rows, got_cols) {
            return Err(CosmosStateError::ShapeMismatch {
                expected_rows,
                expected_cols,
                got_rows,
                got_cols,
            });
        }

        self.w_blocks[block_idx] = Buffer2::from_array(basis);
        Ok(())
    }

    /// Replace every block basis with a deterministic, orthonormalized random
    /// basis, seeded from `seed`.
    ///
    /// The default basis for a freshly constructed state is the first `k`
    /// columns of the identity for each block; training with plain gauge
    /// descent can stall near chance from that starting point. This method
    /// draws each block from a small deterministic generator keyed on
    /// `seed ^ block_idx`, orthonormalizes it with [`qr_orthonormalize`], and
    /// installs it via [`Self::set_block_basis`]. The same seed always
    /// reproduces bit-identical bases.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError`] if a block basis is rejected, which does
    /// not happen in practice since every matrix is drawn at the block's own
    /// registered shape.
    pub fn seed_orthonormal(&mut self, seed: u64) -> Result<(), CosmosStateError> {
        let qr = QrConfig::default();
        for block in 0..self.w_blocks.len() {
            let (n, k) = self.block_shapes[block];
            let mut rng = seed ^ (block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let raw = Array2::from_shape_fn((n, k), |_| seeded_normal(&mut rng));
            self.set_block_basis(block, qr_orthonormalize(&raw, &qr))?;
        }
        Ok(())
    }

    /// Replace every block basis with the position-keyed Gaussian orthonormal
    /// basis the article's reproduction runs start from.
    ///
    /// Unlike [`Self::seed_orthonormal`], which advances a sequential
    /// generator keyed on the block index alone, every entry here is a pure
    /// function of `(seed, block index, row, column)`: a standard-normal draw
    /// from [`crate::state::keyed_init::block_entry`], orthonormalized per
    /// block with [`qr_orthonormalize`]. The same seed always reproduces
    /// bit-identical bases, and a block's entries do not depend on the shapes
    /// of the other blocks.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError`] if a block basis is rejected, which does
    /// not happen in practice since every matrix is drawn at the block's own
    /// registered shape.
    pub fn seed_orthonormal_gaussian(&mut self, seed: u64) -> Result<(), CosmosStateError> {
        for block in 0..self.w_blocks.len() {
            let (n, k) = self.block_shapes[block];
            let basis = crate::state::keyed_init::orthonormal_block(seed, block, n, k);
            self.set_block_basis(block, basis)?;
        }
        Ok(())
    }

    // ─── Diagnostics ──────────────────────────────────────────────

    /// Compute per-block orthogonality error `||W_i^T W_i - I||_F`.
    ///
    /// Returns a `Vec<f64>` with one Frobenius norm per block.
    pub fn orthogonality_errors(&self) -> Vec<f64> {
        self.w_blocks
            .iter()
            .map(|buf| {
                let w = buf.as_array();
                let k = w.ncols();
                let gram = w.t().dot(w);
                let eye: Array2<f64> = Array2::eye(k);
                let diff = &gram - &eye;
                diff.mapv(|x| x.powi(2)).sum().sqrt()
            })
            .collect()
    }

    /// Compute RMS orthogonality error across all blocks.
    ///
    /// Returns `sqrt(mean(errors²))`, or `0.0` if there are no blocks.
    pub fn total_orthogonality_error(&self) -> f64 {
        let errors = self.orthogonality_errors();
        if errors.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = errors.iter().map(|e| e * e).sum();
        (sum_sq / errors.len() as f64).sqrt()
    }

    /// Return a full diagnostic report of numerical health.
    pub fn get_diagnostics(&self) -> CosmosStateDiagnostics {
        CosmosStateDiagnostics {
            orthogonality_errors: self.orthogonality_errors(),
            total_orth_error: self.total_orthogonality_error(),
            comm_target_ema: self.comm_target_ema.clone(),
            block_shapes: self.block_shapes.clone(),
            num_blocks: self.num_blocks(),
        }
    }

    // ─── External block registration ──────────────────────────────

    /// Register an external orthonormal basis as a new block.
    ///
    /// The basis is cloned and re-orthonormalized via QR.
    /// Returns the index of the newly registered block.
    ///
    /// # Arguments
    ///
    /// * `w` - Matrix of shape `(n, k)` with `k ≤ n`.
    ///
    /// # Errors
    ///
    /// Returns [`CosmosStateError::InvalidBasisShape`] if `k > n`.
    pub fn register_external_block(&mut self, w: &Array2<f64>) -> Result<usize, CosmosStateError> {
        let (n, k) = (w.nrows(), w.ncols());
        if k > n {
            return Err(CosmosStateError::InvalidBasisShape { rows: n, cols: k });
        }

        // Re-orthonormalize via QR
        let q = qr_orthonormalize(w, &QrConfig::default());

        self.w_blocks.push(Buffer2::from_array(q));
        self.block_shapes.push((n, k));
        self.comm_target_ema.push(0.0);
        self.total_n += n;
        self.total_k += k;

        // Recompute block positions for all blocks
        self.block_positions = Self::compute_block_positions(&self.block_shapes);

        Ok(self.w_blocks.len() - 1)
    }
}

impl fmt::Display for CosmosState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shapes: Vec<String> = self
            .block_shapes
            .iter()
            .map(|&(n, k)| format!("Gr({k},{n})"))
            .collect();
        write!(f, "CosmosState(blocks=[{}])", shapes.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    // ─── Helper: build a rotation matrix ──────────────────────────

    /// Build a 2D rotation matrix of angle `theta`.
    fn rot2(theta: f64) -> Array2<f64> {
        let (s, c) = theta.sin_cos();
        ndarray::array![[c, -s], [s, c]]
    }

    /// Build a block-diagonal rotation from individual angles.
    fn block_diag_rotation(sizes: &[usize], angles: &[f64]) -> Array2<f64> {
        let total: usize = sizes.iter().sum();
        let mut u = Array2::eye(total);
        let mut offset = 0;
        for (&n, &angle) in sizes.iter().zip(angles) {
            let r = rot2(angle);
            // Apply 2x2 rotation to the first two rows/cols of each block
            for i in 0..2 {
                for j in 0..2 {
                    u[[offset + i, offset + j]] = r[[i, j]];
                }
            }
            offset += n;
        }
        u
    }

    /// Frobenius norm helper.
    fn frobenius(a: &Array2<f64>) -> f64 {
        a.mapv(|x| x.powi(2)).sum().sqrt()
    }

    // ═══════════════════════════════════════════════════════════════
    //  Constructor tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn new_creates_correct_num_blocks() {
        let state = CosmosState::new(&[(8, 4), (6, 3), (10, 5)]).unwrap();
        assert_eq!(state.num_blocks(), 3);
    }

    #[test]
    fn new_stores_correct_shapes() {
        let shapes = [(8, 4), (6, 3)];
        let state = CosmosState::new(&shapes).unwrap();
        assert_eq!(state.block_shapes(), &shapes);
    }

    #[test]
    fn new_computes_correct_totals() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        assert_eq!(state.total_n(), 14);
        assert_eq!(state.total_k(), 7);
    }

    #[test]
    fn new_initializes_orthonormal_bases() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        for i in 0..state.num_blocks() {
            let w = state.get_basis(i).unwrap();
            let gram = w.t().dot(w);
            let eye: Array2<f64> = Array2::eye(w.ncols());
            assert!(frobenius(&(&gram - &eye)) < 1e-12);
        }
    }

    #[test]
    fn new_empty_is_valid() {
        let state = CosmosState::new(&[]).unwrap();
        assert_eq!(state.num_blocks(), 0);
        assert_eq!(state.total_n(), 0);
        assert_eq!(state.total_k(), 0);
    }

    #[test]
    fn new_k_equals_n_is_valid() {
        let state = CosmosState::new(&[(4, 4)]).unwrap();
        let w = state.get_basis(0).unwrap();
        assert_eq!(w.dim(), (4, 4));
    }

    #[test]
    fn new_rejects_k_greater_than_n() {
        let result = CosmosState::new(&[(4, 8)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            CosmosStateError::InvalidShape { block, n, k } => {
                assert_eq!(block, 0);
                assert_eq!(n, 4);
                assert_eq!(k, 8);
            }
            other => panic!("Expected InvalidShape, got {:?}", other),
        }
    }

    #[test]
    fn new_rejects_second_block_invalid() {
        let result = CosmosState::new(&[(8, 4), (3, 5)]);
        match result.unwrap_err() {
            CosmosStateError::InvalidShape { block, .. } => assert_eq!(block, 1),
            other => panic!("Expected InvalidShape, got {:?}", other),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Getter tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn get_basis_returns_correct_shape() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        assert_eq!(state.get_basis(0).unwrap().dim(), (8, 4));
        assert_eq!(state.get_basis(1).unwrap().dim(), (6, 3));
    }

    #[test]
    fn get_basis_out_of_range() {
        let state = CosmosState::new(&[(8, 4)]).unwrap();
        assert!(state.get_basis(1).is_err());
    }

    #[test]
    fn get_projector_is_symmetric() {
        let state = CosmosState::new(&[(8, 4)]).unwrap();
        let p = state.get_projector(0).unwrap();
        let diff = &p - &p.t();
        assert!(frobenius(&diff) < 1e-12);
    }

    #[test]
    fn get_projector_is_idempotent() {
        let state = CosmosState::new(&[(8, 4)]).unwrap();
        let p = state.get_projector(0).unwrap();
        let p2 = p.dot(&p);
        assert!(frobenius(&(&p - &p2)) < 1e-12);
    }

    #[test]
    fn get_projector_has_correct_rank() {
        let state = CosmosState::new(&[(8, 4)]).unwrap();
        let p = state.get_projector(0).unwrap();
        let trace: f64 = (0..p.nrows()).map(|i| p[[i, i]]).sum();
        assert!((trace - 4.0).abs() < 1e-12);
    }

    #[test]
    fn get_block_by_index_is_alias() {
        let state = CosmosState::new(&[(8, 4)]).unwrap();
        let a = state.get_basis(0).unwrap();
        let b = state.get_block_by_index(0).unwrap();
        assert_eq!(a, b);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Assembly tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn global_basis_has_correct_shape() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let w_star = state.assemble_global_basis();
        assert_eq!(w_star.dim(), (14, 7));
    }

    #[test]
    fn global_basis_is_block_diagonal() {
        let state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let w_star = state.assemble_global_basis();
        // Off-diagonal blocks should be zero
        let off_diag_top_right = w_star.slice(s![0..4, 2..4]);
        let off_diag_bottom_left = w_star.slice(s![4..8, 0..2]);
        assert!(frobenius(&off_diag_top_right.to_owned()) < 1e-15);
        assert!(frobenius(&off_diag_bottom_left.to_owned()) < 1e-15);
    }

    #[test]
    fn global_projector_has_correct_shape() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let p_star = state.assemble_global_projector();
        assert_eq!(p_star.dim(), (14, 14));
    }

    #[test]
    fn global_projector_is_idempotent() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let p = state.assemble_global_projector();
        let p2 = p.dot(&p);
        assert!(frobenius(&(&p - &p2)) < 1e-12);
    }

    #[test]
    fn global_projector_has_correct_trace() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let p = state.assemble_global_projector();
        let trace: f64 = (0..p.nrows()).map(|i| p[[i, i]]).sum();
        assert!((trace - 7.0).abs() < 1e-12); // 4 + 3
    }

    #[test]
    fn global_basis_empty_state() {
        let state = CosmosState::new(&[]).unwrap();
        let w = state.assemble_global_basis();
        assert_eq!(w.dim(), (0, 0));
    }

    // ═══════════════════════════════════════════════════════════════
    //  Mutation tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn update_from_global_basis_roundtrip() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let w_star = state.assemble_global_basis();
        let mut state2 = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        state2.update_from_global_basis(&w_star);
        for i in 0..2 {
            let a = state.get_basis(i).unwrap();
            let b = state2.get_basis(i).unwrap();
            assert!(frobenius(&(a - b)) < 1e-15);
        }
    }

    #[test]
    fn orthonormalize_all_restores_orthogonality() {
        let mut state = CosmosState::new(&[(8, 4)]).unwrap();
        // Perturb the basis
        state.w_blocks[0].as_array_mut()[[0, 0]] += 0.1;
        state.w_blocks[0].as_array_mut()[[1, 1]] += 0.05;
        let err_before = state.orthogonality_errors()[0];
        assert!(err_before > 1e-3);
        state.orthonormalize_all(RetractionMethod::Qr);
        let err_after = state.orthogonality_errors()[0];
        assert!(err_after < 1e-12);
    }

    #[test]
    fn orthonormalize_blocks_only_touches_selected() {
        let mut state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        // Perturb both blocks
        state.w_blocks[0].as_array_mut()[[0, 0]] += 0.1;
        state.w_blocks[1].as_array_mut()[[0, 0]] += 0.1;
        // Only fix block 0
        state.orthonormalize_blocks(&[0], RetractionMethod::Qr);
        assert!(state.orthogonality_errors()[0] < 1e-12);
        assert!(state.orthogonality_errors()[1] > 1e-3);
    }

    #[test]
    fn orthonormalize_svd_works() {
        let mut state = CosmosState::new(&[(8, 4)]).unwrap();
        state.w_blocks[0].as_array_mut()[[0, 0]] += 0.1;
        state.orthonormalize_all(RetractionMethod::Svd);
        assert!(state.orthogonality_errors()[0] < 1e-10);
    }

    #[test]
    fn orthonormalize_preserves_shape() {
        let mut state = CosmosState::new(&[(8, 4)]).unwrap();
        state.w_blocks[0].as_array_mut()[[0, 0]] += 0.1;
        state.orthonormalize_all(RetractionMethod::Qr);
        assert_eq!(state.get_basis(0).unwrap().dim(), (8, 4));
    }

    // ═══════════════════════════════════════════════════════════════
    //  Conjugation tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn apply_conjugation_preserves_orthogonality() {
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let u = block_diag_rotation(&[4, 4], &[0.3, 0.7]);
        state.apply_conjugation(&u);
        let errors = state.orthogonality_errors();
        for e in &errors {
            assert!(*e < 1e-10);
        }
    }

    #[test]
    fn apply_conjugation_identity_is_noop() {
        let state_orig = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let w_before = state_orig.assemble_global_basis();
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let eye = Array2::eye(8);
        state.apply_conjugation(&eye);
        let w_after = state.assemble_global_basis();
        assert!(frobenius(&(&w_before - &w_after)) < 1e-14);
    }

    #[test]
    fn apply_conjugation_changes_projector() {
        let mut state = CosmosState::new(&[(4, 2)]).unwrap();
        let p_before = state.get_projector(0).unwrap();
        // Rotation in the (0,2) plane — mixes a subspace column with an
        // out-of-subspace dimension, so the projector must change.
        let angle = 0.5_f64;
        let (s, c) = angle.sin_cos();
        let mut u4 = Array2::eye(4);
        u4[[0, 0]] = c;
        u4[[0, 2]] = -s;
        u4[[2, 0]] = s;
        u4[[2, 2]] = c;
        state.apply_conjugation(&u4);
        let p_after = state.get_projector(0).unwrap();
        assert!(frobenius(&(&p_before - &p_after)) > 0.01);
    }

    #[test]
    fn apply_block_conjugations_preserves_orthogonality() {
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let mut u0 = Array2::eye(4);
        let r = rot2(0.3);
        for i in 0..2 {
            for j in 0..2 {
                u0[[i, j]] = r[[i, j]];
            }
        }
        let u1 = Array2::eye(4);
        state.apply_block_conjugations(&[u0, u1]);
        for e in &state.orthogonality_errors() {
            assert!(*e < 1e-10);
        }
    }

    #[test]
    fn apply_block_left_action_updates_only_selected_block() {
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let before_selected = state.get_basis(0).unwrap().clone();
        let before_other = state.get_basis(1).unwrap().clone();
        let mut u = Array2::eye(4);
        let r = rot2(0.3);
        for i in 0..2 {
            for j in 0..2 {
                u[[i, j]] = r[[i, j]];
            }
        }

        state.apply_block_left_action(0, &u);

        assert!(frobenius(&(state.get_basis(0).unwrap() - &before_selected)) > 0.01);
        assert_eq!(state.get_basis(1).unwrap(), &before_other);
        for e in &state.orthogonality_errors() {
            assert!(*e < 1e-10);
        }
    }

    #[test]
    fn set_block_basis_replaces_only_selected_block() {
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let before_other = state.get_basis(1).unwrap().clone();
        let replacement = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],];

        state.set_block_basis(0, replacement.clone()).unwrap();

        assert_eq!(state.get_basis(0).unwrap(), &replacement);
        assert_eq!(state.get_basis(1).unwrap(), &before_other);
        for e in &state.orthogonality_errors() {
            assert!(*e < 1e-10);
        }
    }

    #[test]
    fn set_block_basis_validates_shape() {
        let mut state = CosmosState::new(&[(4, 2)]).unwrap();
        let err = state.set_block_basis(0, Array2::zeros((5, 2))).unwrap_err();
        assert!(matches!(
            err,
            CosmosStateError::ShapeMismatch {
                expected_rows: 4,
                expected_cols: 2,
                got_rows: 5,
                got_cols: 2,
            }
        ));
    }

    #[test]
    fn seed_orthonormal_produces_orthonormal_bases() {
        let mut state = CosmosState::new(&[(8, 4), (6, 3), (10, 5)]).unwrap();
        state.seed_orthonormal(42).unwrap();

        for block in 0..state.num_blocks() {
            let w = state.get_basis(block).unwrap();
            let k = w.ncols();
            let gram = w.t().dot(w);
            let error = frobenius(&(&gram - &Array2::<f64>::eye(k)));
            assert!(
                error <= 1e-13,
                "block {block}: orthogonality error {error:e}"
            );
        }
    }

    #[test]
    fn seed_orthonormal_is_deterministic_for_a_fixed_seed() {
        let mut a = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let mut b = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        a.seed_orthonormal(319_635_789).unwrap();
        b.seed_orthonormal(319_635_789).unwrap();

        for block in 0..a.num_blocks() {
            let wa = a.get_basis(block).unwrap();
            let wb = b.get_basis(block).unwrap();
            for (x, y) in wa.iter().zip(wb.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "block {block} diverged");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Diagnostics tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn fresh_state_has_zero_errors() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let errors = state.orthogonality_errors();
        assert_eq!(errors.len(), 2);
        for e in &errors {
            assert!(*e < 1e-14);
        }
    }

    #[test]
    fn total_error_is_rms() {
        let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        state.w_blocks[0].as_array_mut()[[0, 0]] += 0.1;
        let errors = state.orthogonality_errors();
        let expected = ((errors[0].powi(2) + errors[1].powi(2)) / 2.0).sqrt();
        let total = state.total_orthogonality_error();
        assert!((total - expected).abs() < 1e-14);
    }

    #[test]
    fn empty_state_total_error_is_zero() {
        let state = CosmosState::new(&[]).unwrap();
        assert_eq!(state.total_orthogonality_error(), 0.0);
    }

    #[test]
    fn diagnostics_returns_correct_fields() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let diag = state.get_diagnostics();
        assert_eq!(diag.num_blocks, 2);
        assert_eq!(diag.block_shapes, vec![(8, 4), (6, 3)]);
        assert_eq!(diag.orthogonality_errors.len(), 2);
        assert_eq!(diag.comm_target_ema, vec![0.0, 0.0]);
    }

    // ═══════════════════════════════════════════════════════════════
    //  External block registration tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn register_block_returns_correct_index() {
        let mut state = CosmosState::new(&[(8, 4)]).unwrap();
        let ext = Array2::eye(6).slice(s![.., ..3]).to_owned();
        let idx = state.register_external_block(&ext).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(state.num_blocks(), 2);
    }

    #[test]
    fn register_block_updates_totals() {
        let mut state = CosmosState::new(&[(8, 4)]).unwrap();
        let ext = Array2::eye(6).slice(s![.., ..3]).to_owned();
        state.register_external_block(&ext).unwrap();
        assert_eq!(state.total_n(), 14);
        assert_eq!(state.total_k(), 7);
        assert_eq!(state.block_shapes(), &[(8, 4), (6, 3)]);
    }

    #[test]
    fn register_block_orthonormalizes() {
        let mut state = CosmosState::new(&[]).unwrap();
        // Create a non-orthonormal matrix
        let mut w = Array2::eye(4).slice(s![.., ..2]).to_owned();
        w[[0, 0]] = 2.0; // break orthonormality
        state.register_external_block(&w).unwrap();
        let errors = state.orthogonality_errors();
        assert!(errors[0] < 1e-12);
    }

    #[test]
    fn register_block_rejects_k_gt_n() {
        let mut state = CosmosState::new(&[]).unwrap();
        let w = Array2::zeros((2, 5));
        assert!(state.register_external_block(&w).is_err());
    }

    #[test]
    fn register_block_updates_global_assembly() {
        let mut state = CosmosState::new(&[(4, 2)]).unwrap();
        let ext = Array2::eye(4).slice(s![.., ..2]).to_owned();
        state.register_external_block(&ext).unwrap();
        let w_star = state.assemble_global_basis();
        assert_eq!(w_star.dim(), (8, 4));
    }

    // ═══════════════════════════════════════════════════════════════
    //  Display test
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn display_format() {
        let state = CosmosState::new(&[(8, 4), (6, 3)]).unwrap();
        let s = format!("{}", state);
        assert_eq!(s, "CosmosState(blocks=[Gr(4,8), Gr(3,6)])");
    }
}
