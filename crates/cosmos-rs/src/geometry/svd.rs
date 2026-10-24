//! Singular Value Decomposition.
//!
//! With the `lapack` feature the decomposition is delegated to
//! `ndarray-linalg`; by default it is computed in pure Rust by the one-sided
//! Jacobi (Hestenes) method, with no LAPACK dependency.
//!
//! ## One-sided Jacobi
//!
//! For a tall matrix $A \in \mathbb{R}^{m \times n}$ ($m \geq n$) the method
//! works on a copy $W \leftarrow A$ and accumulates the right factor in
//! $V \leftarrow I_n$. A sweep visits every column pair $(p, q)$ and applies
//! the plane rotation that makes those two columns of $W$ orthogonal:
//!
//! $$\zeta = \frac{\|w_q\|^2 - \|w_p\|^2}{2 \, w_p^\top w_q}, \qquad
//!   t = \frac{\operatorname{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}},
//!   \qquad c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = c\,t.$$
//!
//! The same rotation is applied to $V$. A pair is skipped when
//! $|w_p^\top w_q| \leq \varepsilon \, \|w_p\| \, \|w_q\|$, and the iteration
//! stops as soon as a whole sweep rotates nothing, which is orthogonality to
//! machine precision.
//!
//! At convergence the columns of $W$ are mutually orthogonal, their norms are
//! the singular values, and $U$ is $W$ with each column divided by its norm.
//! Since $U$ is read off an explicitly orthogonalized $W$ rather than
//! accumulated triplet by triplet, its orthogonality does not degrade with the
//! number of columns, and a degenerate spectrum (repeated singular values) is
//! handled as well as a well-separated one.
//!
//! Wide matrices ($m < n$) are handled by transposing, decomposing, and
//! swapping the factors back.
use ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicU8, Ordering};

/// Result of a Singular Value Decomposition.
///
/// For a matrix A ∈ ℝ^(m×n), the SVD is:
///
/// $$A = U \Sigma V^\top$$
///
/// where:
/// - U ∈ ℝ^(m×k) has orthonormal columns (left singular vectors)
/// - Σ = diag(σ₁, ..., σ_k) with σ₁ ≥ σ₂ ≥ ... ≥ σ_k ≥ 0
/// - Vt ∈ ℝ^(k×n) has orthonormal rows (right singular vectors, transposed)
/// - k = min(m, n)
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors, orthonormal columns. On the pure-Rust path a
    /// column with a zero singular value is itself the zero vector rather
    /// than an arbitrary orthonormal completion.
    pub u: Array2<f64>,
    /// Singular values in descending order.
    pub s: Array1<f64>,
    /// Right singular vectors transposed, orthonormal rows.
    pub vt: Array2<f64>,
}

/// LAPACK driver used for a dense thin SVD.
///
/// The enum exists regardless of the `lapack` feature so that configuration
/// carrying a driver name compiles unchanged; on the pure-Rust path the choice
/// is inert and the one-sided Jacobi method is used either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapackSvdDriver {
    /// Full `dgesvd` factorisation (`jobu = jobvt = 'A'`) sliced to the thin
    /// `k = min(m, n)` factors. This is the driver the reference run artifacts
    /// are computed with.
    Gesvd,
    /// Divide-and-conquer `dgesdd` with `jobz = 'S'`, which allocates only the
    /// leading `k` singular vectors and is faster on tall-and-skinny inputs.
    /// A `LapackComputationalFailure` falls back to `dgesvd`.
    Gesdd,
}

const DRIVER_GESVD: u8 = 0;
const DRIVER_GESDD: u8 = 1;

static LAPACK_SVD_DRIVER: AtomicU8 = AtomicU8::new(DRIVER_GESVD);

/// Select the LAPACK driver every later [`SvdConfig::default`] will carry.
///
/// Intended to be called once at process start, before any decomposition runs,
/// so that a whole process is decomposed by a single driver.
pub fn set_lapack_svd_driver(driver: LapackSvdDriver) {
    let value = match driver {
        LapackSvdDriver::Gesvd => DRIVER_GESVD,
        LapackSvdDriver::Gesdd => DRIVER_GESDD,
    };
    LAPACK_SVD_DRIVER.store(value, Ordering::Relaxed);
}

/// The process-wide LAPACK SVD driver. [`LapackSvdDriver::Gesvd`] by default.
pub fn lapack_svd_driver() -> LapackSvdDriver {
    match LAPACK_SVD_DRIVER.load(Ordering::Relaxed) {
        DRIVER_GESDD => LapackSvdDriver::Gesdd,
        _ => LapackSvdDriver::Gesvd,
    }
}

/// Configuration for a dense SVD.
///
/// `max_iter` is ignored by the `lapack` path; `lapack_driver` is ignored by
/// the pure-Rust path.
#[derive(Debug, Clone, Copy)]
pub struct SvdConfig {
    /// Upper bound on the number of Jacobi sweeps, itself capped at 60.
    pub max_iter: usize,
    /// LAPACK driver to decompose with.
    pub lapack_driver: LapackSvdDriver,
}

impl Default for SvdConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            lapack_driver: lapack_svd_driver(),
        }
    }
}

/// Full LAPACK factorisation sliced to the thin `k = min(m, n)` factors.
#[cfg(feature = "lapack")]
fn full_svd_thin(a: &Array2<f64>) -> SvdResult {
    use ndarray_linalg::SVD as SVDTrait;
    let k = a.nrows().min(a.ncols());
    let (u_opt, s, vt_opt) = a.svd(true, true).expect("LAPACK SVD failed");
    let u_full = u_opt.expect("LAPACK SVD did not return U");
    let vt_full = vt_opt.expect("LAPACK SVD did not return Vt");
    let u = u_full.slice(ndarray::s![.., ..k]).to_owned();
    let vt = vt_full.slice(ndarray::s![..k, ..]).to_owned();
    SvdResult { u, s, vt }
}

/// Compute the thin SVD using LAPACK (via ndarray-linalg).
///
/// U is `(m, k)`, Vt is `(k, n)` and `k = min(m, n)`, whichever driver
/// `config.lapack_driver` selects.
///
/// [`LapackSvdDriver::Gesvd`] runs the full `dgesvd` factorisation
/// (`jobu = jobvt = 'A'`) and slices it to those shapes; the full `m×m` left
/// factor is built and all but `k` of its columns discarded.
///
/// [`LapackSvdDriver::Gesdd`] runs the divide-and-conquer routine `dgesdd`
/// with `jobz = 'S'`, which computes and allocates only the leading `k`
/// singular vectors — this matters for the tall-and-skinny matrices the
/// Grassmann maps work with. `dgesdd` can fail to converge on ill-conditioned
/// inputs (`LapackComputationalFailure`); when that happens the function falls
/// back to the slower but more robust `dgesvd` path, so callers always see
/// `(m, k)` / `(k, n)` factors.
///
/// `config.max_iter` is accepted for API compatibility with the pure-Rust
/// fallback but is ignored.
#[cfg(feature = "lapack")]
pub fn svd(a: &Array2<f64>, config: &SvdConfig) -> SvdResult {
    use ndarray_linalg::{JobSvd, SVDDC as SVDDCTrait};
    match config.lapack_driver {
        LapackSvdDriver::Gesvd => full_svd_thin(a),
        LapackSvdDriver::Gesdd => match a.svddc(JobSvd::Some) {
            Ok((u_opt, s, vt_opt)) => {
                let u = u_opt.expect("LAPACK SVD did not return U");
                let vt = vt_opt.expect("LAPACK SVD did not return Vt");
                SvdResult { u, s, vt }
            }
            Err(_) => full_svd_thin(a),
        },
    }
}

/// Compute the thin SVD of a matrix by one-sided Jacobi rotations.
///
/// Decomposes A = U Σ Vᵀ with `k = min(m, n)`: U is `(m, k)` with orthonormal
/// columns, σ₁ ≥ … ≥ σ_k ≥ 0, and Vt is `(k, n)` with orthonormal rows. A
/// column of U whose singular value is exactly zero is left zero, so a
/// rank-deficient or zero input yields the expected singular values without a
/// spurious direction.
///
/// `config.max_iter` bounds the number of sweeps (itself capped at 60); a
/// sweep that rotates nothing means the columns are orthogonal to machine
/// precision and the iteration stops there.
///
/// # Arguments
///
/// * `a` - Input matrix, shape (m, n)
///
/// # Returns
///
/// `SvdResult` containing U, s, and Vt matrices.
#[cfg(not(feature = "lapack"))]
pub fn svd(a: &Array2<f64>, config: &SvdConfig) -> SvdResult {
    let (m, n) = a.dim();

    if m < n {
        let wide = svd(&a.t().to_owned(), config);
        return SvdResult {
            u: wide.vt.t().to_owned(),
            s: wide.s,
            vt: wide.u.t().to_owned(),
        };
    }

    let k = n;
    let mut w = a.clone();
    let mut v = Array2::<f64>::eye(n);
    let max_sweeps = config.max_iter.clamp(1, 60);

    for _ in 0..max_sweeps {
        let mut rotated = false;

        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    alpha += w[[i, p]] * w[[i, p]];
                    beta += w[[i, q]] * w[[i, q]];
                    gamma += w[[i, p]] * w[[i, q]];
                }

                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m {
                    let wp = w[[i, p]];
                    let wq = w[[i, q]];
                    w[[i, p]] = c * wp - s * wq;
                    w[[i, q]] = s * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[[i, p]];
                    let vq = v[[i, q]];
                    v[[i, p]] = c * vp - s * vq;
                    v[[i, q]] = s * vp + c * vq;
                }
            }
        }

        if !rotated {
            break;
        }
    }

    let mut order: Vec<(f64, usize)> = (0..k)
        .map(|j| {
            let mut squared = 0.0;
            for i in 0..m {
                squared += w[[i, j]] * w[[i, j]];
            }
            (squared.sqrt(), j)
        })
        .collect();
    order.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut u = Array2::<f64>::zeros((m, k));
    let mut s = Array1::<f64>::zeros(k);
    let mut vt = Array2::<f64>::zeros((k, n));

    for (j, &(sigma, src)) in order.iter().enumerate() {
        s[j] = sigma;
        if sigma > 0.0 {
            for i in 0..m {
                u[[i, j]] = w[[i, src]] / sigma;
            }
        }
        for i in 0..n {
            vt[[j, i]] = v[[i, src]];
        }
    }

    SvdResult { u, s, vt }
}

/// Compute only the singular values of a matrix (LAPACK path).
///
/// Always uses `dgesvd`, so the singular values are identical under both
/// settings of [`SvdConfig::lapack_driver`].
#[cfg(feature = "lapack")]
pub fn svdvals(a: &Array2<f64>, _config: &SvdConfig) -> Array1<f64> {
    use ndarray_linalg::SVD as SVDTrait;
    let (_, s, _) = a.svd(false, false).expect("LAPACK SVD failed");
    s
}

/// Compute only the singular values of a matrix.
///
/// Convenience wrapper around [`svd`] that discards U and Vt.
///
/// This is what `grassmann::principal_angles` needs internally.
///
/// # Arguments
///
/// * `a` - Input matrix, shape (m, n)
///
/// # Returns
///
/// Singular values in descending order, shape (min(m, n),)
#[cfg(not(feature = "lapack"))]
pub fn svdvals(a: &Array2<f64>, config: &SvdConfig) -> Array1<f64> {
    svd(a, config).s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ——————————————————————————————————————————————
    // svd
    // ——————————————————————————————————————————————

    // —— Test 3: SVD of identity matrix ——
    #[test]
    fn svd_identity_matrix() {
        let a = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = svd(&a, &SvdConfig::default());

        // All singular values = 1
        for &s in result.s.iter() {
            assert!((s - 1.0).abs() < 1e-6, "Expected σ=1, got {s}");
        }

        // U and Vt should be 3×3
        assert_eq!(result.u.dim(), (3, 3));
        assert_eq!(result.vt.dim(), (3, 3));
    }

    // —— Test 4: SVD of known diagonal matrix ——
    #[test]
    fn svd_diagonal_matrix() {
        let a = array![[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];

        let result = svd(&a, &SvdConfig::default());

        // σ should be [5, 3, 1] (descending order)
        assert!(
            (result.s[0] - 5.0).abs() < 1e-6,
            "σ₁ expected 5, got {}",
            result.s[0]
        );
        assert!(
            (result.s[1] - 3.0).abs() < 1e-6,
            "σ₂ expected 3, got {}",
            result.s[1]
        );
        assert!(
            (result.s[2] - 1.0).abs() < 1e-6,
            "σ₃ expected 1, got {}",
            result.s[2]
        );
    }

    // —— Test 5: Fundamental property — reconstruction A = U Σ Vt ——
    #[test]
    fn svd_reconstruction() {
        // Arbitrary matrix (non-symmetric, non-diagonal)
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let result = svd(&a, &SvdConfig::default());

        // Reconstruct: A_reconstructed = U × diag(s) × Vt
        let (m, n) = a.dim();
        let k = result.s.len();

        // Build S as diagonal k×k matrix
        let s_diag = Array2::from_shape_fn((k, k), |(i, j)| if i == j { result.s[i] } else { 0.0 });

        // A_reconstructed = U @ S_diag @ Vt
        let a_reconstructed = result.u.dot(&s_diag).dot(&result.vt);

        // Compare element-wise
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (a[[i, j]] - a_reconstructed[[i, j]]).abs() < 1e-6,
                    "Mismatch at ({i},{j}): original={}, reconstructed={}",
                    a[[i, j]],
                    a_reconstructed[[i, j]]
                );
            }
        }
    }

    // —— Test 6: Singular values in descending order ——
    #[test]
    fn svd_singular_values_descending() {
        let a = array![[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0]];

        let result = svd(&a, &SvdConfig::default());

        // Each σᵢ should be ≥ σᵢ₊₁
        for i in 0..result.s.len() - 1 {
            assert!(
                result.s[i] >= result.s[i + 1] - 1e-10,
                "Not descending: σ[{}]={} < σ[{}]={}",
                i,
                result.s[i],
                i + 1,
                result.s[i + 1]
            );
        }
    }

    // —— Test 7: U columns are orthonormal (U^T U ≈ I) ——
    #[test]
    fn svd_u_columns_orthonormal() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let result = svd(&a, &SvdConfig::default());

        // U^T U should be ≈ identity
        let utu = result.u.t().dot(&result.u);
        let k = utu.nrows();

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[[i, j]] - expected).abs() < 1e-6,
                    "U^T U[{i},{j}] = {}, expected {expected}",
                    utu[[i, j]]
                );
            }
        }
    }

    // —— Test 8: Vt rows are orthonormal (Vt Vt^T ≈ I) ——
    #[test]
    fn svd_vt_rows_orthonormal() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let result = svd(&a, &SvdConfig::default());

        // Vt @ Vt^T should be ≈ identity
        let vvt = result.vt.dot(&result.vt.t());
        let k = vvt.nrows();

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vvt[[i, j]] - expected).abs() < 1e-6,
                    "Vt Vt^T[{i},{j}] = {}, expected {expected}",
                    vvt[[i, j]]
                );
            }
        }
    }

    // —— Test 9: Tall matrix (m > n) — correct shapes ——
    #[test]
    fn svd_rectangular_tall() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]; // 4×2

        let result = svd(&a, &SvdConfig::default());

        // k = min(4, 2) = 2
        assert_eq!(result.u.dim(), (4, 2), "U shape");
        assert_eq!(result.s.len(), 2, "s length");
        assert_eq!(result.vt.dim(), (2, 2), "Vt shape");
    }

    // —— Test 10: Wide matrix (m < n) — correct shapes ——
    #[test]
    fn svd_rectangular_wide() {
        let a = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]; // 2×4

        let result = svd(&a, &SvdConfig::default());

        // k = min(2, 4) = 2
        assert_eq!(result.u.dim(), (2, 2), "U shape");
        assert_eq!(result.s.len(), 2, "s length");
        assert_eq!(result.vt.dim(), (2, 4), "Vt shape");
    }

    // —— Test 11: Matriz zero ——
    #[test]
    fn svd_zero_matrix() {
        let a = array![[0.0, 0.0], [0.0, 0.0]];

        let result = svd(&a, &SvdConfig::default());

        for &s in result.s.iter() {
            assert!(s.abs() < 1e-6, "Expected σ=0, got {s}");
        }
    }

    // —— Test 12: Rank-1 matrix — only σ₁ > 0 ——
    #[test]
    fn svd_rank_one_matrix() {
        // Rank 1: all rows are multiples of [1, 2]
        let a = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];

        let result = svd(&a, &SvdConfig::default());

        assert!(result.s[0] > 1e-6, "σ₁ should be > 0, got {}", result.s[0]);
        assert!(
            result.s[1].abs() < 1e-6,
            "σ₂ should be ≈ 0, got {}",
            result.s[1]
        );
    }

    // ——————————————————————————————————————————————
    // svdvals
    // ——————————————————————————————————————————————

    // —— Test 13: svdvals returns same values as svd.s ——
    #[test]
    fn svdvals_matches_svd() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config = SvdConfig::default();
        let from_svd = svd(&a, &config).s;
        let from_svdvals = svdvals(&a, &config);

        for (a, b) in from_svd.iter().zip(from_svdvals.iter()) {
            assert!((a - b).abs() < 1e-10, "svdvals mismatch: {a} vs {b}");
        }
    }

    // ——————————————————————————————————————————————
    // SvdConfig
    // ——————————————————————————————————————————————

    // —— Test 14: Config defaults ——
    #[test]
    fn svd_config_default() {
        let config = SvdConfig::default();
        assert_eq!(config.max_iter, 1000);
    }

    // —— Test 15: default config carries the process-wide driver ——
    #[test]
    fn svd_config_default_tracks_process_wide_driver() {
        assert_eq!(SvdConfig::default().lapack_driver, lapack_svd_driver());
    }

    // ——————————————————————————————————————————————
    // Pure-Rust one-sided Jacobi precision
    // ——————————————————————————————————————————————

    /// Precision floor of the pure-Rust fallback on the shapes the Grassmann
    /// maps actually use.
    ///
    /// One-sided Jacobi re-orthogonalizes the left factor every sweep, so
    /// `‖UᵀU − I‖_F` holds at machine precision on these shapes; these bounds
    /// pin that guarantee down.
    #[cfg(not(feature = "lapack"))]
    mod fallback_precision {
        use super::*;

        /// Deterministic linear-congruential matrix with entries in `[-1, 1)`.
        fn pseudo_random_matrix(m: usize, n: usize, seed: u64) -> Array2<f64> {
            let mut state = seed;
            Array2::from_shape_fn((m, n), |_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
        }

        fn frobenius(a: &Array2<f64>) -> f64 {
            a.iter().map(|x| x * x).sum::<f64>().sqrt()
        }

        #[test]
        fn u_columns_stay_orthonormal_on_a_tall_matrix() {
            let a = pseudo_random_matrix(784, 32, 0x1234_5678_9ABC_DEF0);

            let r = svd(&a, &SvdConfig::default());
            let err = frobenius(&(&r.u.t().dot(&r.u) - &Array2::<f64>::eye(32)));

            assert!(err <= 1e-14, "‖UᵀU − I‖_F = {err} on (784, 32)");
        }

        #[test]
        fn reconstruction_is_exact_to_machine_precision() {
            let a = pseudo_random_matrix(32, 8, 0x0BAD_BEEF_0BAD_BEEF);

            let r = svd(&a, &SvdConfig::default());
            let s_diag = Array2::from_shape_fn((8, 8), |(i, j)| if i == j { r.s[i] } else { 0.0 });
            let err = frobenius(&(&r.u.dot(&s_diag).dot(&r.vt) - &a));

            assert!(
                err <= 1e-12 * frobenius(&a),
                "‖UΣVᵀ − A‖_F = {err} on (32, 8)"
            );
        }
    }

    // ——————————————————————————————————————————————
    // Divide-and-conquer thin SVD vs. full SVD
    // ——————————————————————————————————————————————

    /// Pins the divide-and-conquer thin factorisation (`dgesdd`, `jobz = 'S'`)
    /// to the full factorisation (`dgesvd`, `jobu = jobvt = 'A'`) sliced to its
    /// leading `k` columns/rows.
    ///
    /// The two LAPACK drivers are free to disagree on the *sign* of each
    /// singular-vector pair `(u_i, v_i)`, so the factors themselves are not
    /// comparable. What is well defined — and what every consumer of
    /// [`SvdResult`] actually uses — is the gauge-invariant product
    /// `U f(Σ) Vᵀ`. These tests therefore compare reconstructions and
    /// singular values, never `U` or `Vt` element-wise.
    #[cfg(feature = "lapack")]
    mod thin_vs_full {
        use super::*;
        use ndarray_linalg::SVD as SVDTrait;

        /// Deterministic xorshift64 matrix generator with entries in `[-1, 1)`.
        ///
        /// Avoids a `rand` dependency while keeping the cases reproducible.
        fn pseudo_random_matrix(m: usize, n: usize, seed: u64) -> Array2<f64> {
            let mut state = seed | 1;
            Array2::from_shape_fn((m, n), |_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Top 53 bits -> uniform in [0, 1), then rescaled to [-1, 1).
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
        }

        /// Full LAPACK SVD followed by slicing to `k = min(m, n)`.
        fn full_svd_then_slice(a: &Array2<f64>) -> SvdResult {
            let (m, n) = a.dim();
            let k = m.min(n);
            let (u_opt, s, vt_opt) = a.svd(true, true).expect("LAPACK SVD failed");
            let u = u_opt.unwrap().slice(ndarray::s![.., ..k]).to_owned();
            let vt = vt_opt.unwrap().slice(ndarray::s![..k, ..]).to_owned();
            SvdResult { u, s, vt }
        }

        /// Configuration pinned to one driver, leaving the process-wide
        /// selector untouched.
        fn driver(driver: LapackSvdDriver) -> SvdConfig {
            SvdConfig {
                lapack_driver: driver,
                ..SvdConfig::default()
            }
        }

        /// Rebuild `U diag(s) Vt`.
        fn reconstruct(r: &SvdResult) -> Array2<f64> {
            let k = r.s.len();
            let s_diag = Array2::from_shape_fn((k, k), |(i, j)| if i == j { r.s[i] } else { 0.0 });
            r.u.dot(&s_diag).dot(&r.vt)
        }

        #[test]
        fn thin_svd_matches_full_svd_reconstruction_and_singular_values() {
            const TOL: f64 = 1e-12;

            for (m, n, seed) in [
                (8_usize, 3_usize, 0x1234_5678_u64),
                (784, 16, 0x0BAD_BEEF),
                (200, 64, 0x00C0_FFEE),
            ] {
                let a = pseudo_random_matrix(m, n, seed);
                let k = m.min(n);

                let thin = svd(&a, &driver(LapackSvdDriver::Gesdd));
                let full = full_svd_then_slice(&a);

                // Shapes must stay byte-compatible for downstream consumers.
                assert_eq!(thin.u.dim(), (m, k), "U shape for ({m},{n})");
                assert_eq!(thin.s.len(), k, "s length for ({m},{n})");
                assert_eq!(thin.vt.dim(), (k, n), "Vt shape for ({m},{n})");
                assert_eq!(thin.u.dim(), full.u.dim(), "U shape vs full ({m},{n})");
                assert_eq!(thin.s.len(), full.s.len(), "s length vs full ({m},{n})");
                assert_eq!(thin.vt.dim(), full.vt.dim(), "Vt shape vs full ({m},{n})");

                // Singular values are gauge-independent: they must agree.
                for i in 0..k {
                    let diff = (thin.s[i] - full.s[i]).abs();
                    assert!(
                        diff < TOL,
                        "({m},{n}) sigma[{i}] mismatch: thin={} full={} diff={diff}",
                        thin.s[i],
                        full.s[i]
                    );
                }

                // Reconstructions are gauge-independent: they must agree with
                // each other and with the original matrix.
                let rec_thin = reconstruct(&thin);
                let rec_full = reconstruct(&full);
                for i in 0..m {
                    for j in 0..n {
                        let d_paths = (rec_thin[[i, j]] - rec_full[[i, j]]).abs();
                        assert!(
                            d_paths < TOL,
                            "({m},{n}) reconstruction mismatch at ({i},{j}): \
                             thin={} full={} diff={d_paths}",
                            rec_thin[[i, j]],
                            rec_full[[i, j]]
                        );
                        let d_orig = (rec_thin[[i, j]] - a[[i, j]]).abs();
                        assert!(
                            d_orig < TOL,
                            "({m},{n}) thin reconstruction lost A at ({i},{j}): \
                             got={} want={} diff={d_orig}",
                            rec_thin[[i, j]],
                            a[[i, j]]
                        );
                    }
                }
            }
        }

        #[test]
        fn explicit_gesvd_driver_matches_full_svd_bit_for_bit() {
            for (m, n, seed) in [
                (784_usize, 16_usize, 0x0BAD_BEEF_u64),
                (200, 64, 0x00C0_FFEE),
                (8, 3, 0x1234_5678),
                (3, 8, 0x8765_4321),
            ] {
                let a = pseudo_random_matrix(m, n, seed);

                let selected = svd(&a, &driver(LapackSvdDriver::Gesvd));
                let full = full_svd_then_slice(&a);

                assert_eq!(selected.u, full.u, "({m},{n}) U");
                assert_eq!(selected.s, full.s, "({m},{n}) s");
                assert_eq!(selected.vt, full.vt, "({m},{n}) Vt");
            }
        }

        #[test]
        fn gesdd_driver_yields_orthonormal_factors_and_reconstructs() {
            for (m, n, seed) in [
                (784_usize, 16_usize, 0x7777_8888_u64),
                (200, 64, 0x9999_AAAA),
                (8, 3, 0xBBBB_CCCC),
                (3, 8, 0xDDDD_EEEE),
            ] {
                let a = pseudo_random_matrix(m, n, seed);
                let k = m.min(n);
                let r = svd(&a, &driver(LapackSvdDriver::Gesdd));

                assert_eq!(r.u.dim(), (m, k), "({m},{n}) U shape");
                assert_eq!(r.vt.dim(), (k, n), "({m},{n}) Vt shape");

                let utu = r.u.t().dot(&r.u);
                let vvt = r.vt.dot(&r.vt.t());
                for i in 0..k {
                    for j in 0..k {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (utu[[i, j]] - expected).abs() < 1e-12,
                            "({m},{n}) U^T U[{i},{j}] = {}",
                            utu[[i, j]]
                        );
                        assert!(
                            (vvt[[i, j]] - expected).abs() < 1e-12,
                            "({m},{n}) Vt Vt^T[{i},{j}] = {}",
                            vvt[[i, j]]
                        );
                    }
                }

                let rec = reconstruct(&r);
                for i in 0..m {
                    for j in 0..n {
                        let diff = (rec[[i, j]] - a[[i, j]]).abs();
                        assert!(
                            diff < 1e-10,
                            "({m},{n}) reconstruction at ({i},{j}) off by {diff}"
                        );
                    }
                }
            }
        }

        #[test]
        fn thin_svd_factors_are_orthonormal() {
            for (m, n, seed) in [
                (8_usize, 3_usize, 0x1111_2222_u64),
                (784, 16, 0x3333_4444),
                (200, 64, 0x5555_6666),
            ] {
                let a = pseudo_random_matrix(m, n, seed);
                let k = m.min(n);
                let r = svd(&a, &driver(LapackSvdDriver::Gesdd));

                let utu = r.u.t().dot(&r.u);
                let vvt = r.vt.dot(&r.vt.t());
                for i in 0..k {
                    for j in 0..k {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (utu[[i, j]] - expected).abs() < 1e-13,
                            "({m},{n}) U^T U[{i},{j}] = {}",
                            utu[[i, j]]
                        );
                        assert!(
                            (vvt[[i, j]] - expected).abs() < 1e-13,
                            "({m},{n}) Vt Vt^T[{i},{j}] = {}",
                            vvt[[i, j]]
                        );
                    }
                }
            }
        }
    }
}
