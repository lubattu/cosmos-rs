//! QR decomposition via Modified Gram-Schmidt.
//!
//! Computes the thin QR factorization A = QR where:
//!     - Q ∈ ℝ^(m×k) has orthonormal columns
//!     - R ∈ ℝ^(k×p) is upper triangular
//!     - k = min(m, n), p = n
//!
//! Uses the Modified Gram-Schmidt (MGS) algorithm, which is more
//! numerically stable than Classical Gram-Schmidt.
//!
//! Provides:
//!     - [`qr`] — full QR decomposition returning Q and R
//!     - [`qr_orthonormalize`] — convenience wrapper returning only Q

use ndarray::Array2;

/// Configuration for QR decomposition,
///
/// # Fields
///
/// * `tol` - Tolerance for detecting rank-deficient columns.
///   If a column norm falls below this value, it is treated as zero.
#[derive(Debug, Clone, Copy)]
pub struct QrConfig {
    /// Tolerance below which a column norm is treated as zero.
    pub tol: f64,
}

impl Default for QrConfig {
    fn default() -> Self {
        Self { tol: 1e-10 }
    }
}

/// Result of a QR decomposition.
///
/// For a matrix A ∈ ℝ^(m×n), the thin QR decomposition is:
///
/// A = Q R
///
/// where:
/// - Q ∈ ℝ^(m×k) has orthonormal columns, k = min(m, n)
/// - R ∈ ℝ^(k×n) is upper triangular
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Matrix with orthonormal columns.
    pub q: Array2<f64>,
    /// Upper-triangular factor.
    pub r: Array2<f64>,
}

/// Compute the thin QR decomposition.
///
/// When the `lapack` feature is enabled, uses LAPACK's highly optimized
/// Householder QR via `ndarray-linalg`. Otherwise, falls back to the
/// Modified Gram-Schmidt algorithm.
///
/// For A of shape (m, n):
/// - If m >= n: Q is (m, n), R is (n, n)
/// - If m < n: Q is (m, m), R is (m, n)
///
/// # Arguments
///
/// * `a` - Input matrix, shape (m, n)
/// * `config` - Configuration with tolerance for rank detection
///
/// # Returns
///
/// `QrResult` containing Q and R matrices.
pub fn qr(a: &Array2<f64>, config: &QrConfig) -> QrResult {
    #[cfg(feature = "lapack")]
    {
        qr_lapack(a, config)
    }
    #[cfg(not(feature = "lapack"))]
    {
        qr_gram_schmidt(a, config)
    }
}

/// LAPACK-backed thin QR decomposition.
///
/// Uses Householder reflections via `ndarray-linalg`. The result is
/// sign-normalized so that R has non-negative diagonal entries.
#[cfg(feature = "lapack")]
fn qr_lapack(a: &Array2<f64>, config: &QrConfig) -> QrResult {
    use ndarray_linalg::QR as QRTrait;

    let (q, r) = a.qr().expect("LAPACK QR decomposition failed");
    let mut q = q;
    let mut r = r;
    let k = r.nrows().min(r.ncols());

    // Normalize signs so R has non-negative diagonal (matching Gram-Schmidt convention).
    // If R[i,i] < 0, negate Q column i and R row i.
    for i in 0..k {
        if r[[i, i]] < 0.0 {
            for row in 0..q.nrows() {
                q[[row, i]] = -q[[row, i]];
            }
            for col in 0..r.ncols() {
                r[[i, col]] = -r[[i, col]];
            }
        }
    }

    // Apply tolerance: zero out R diagonal entries below tol (rank deficiency)
    for i in 0..k {
        if r[[i, i]].abs() < config.tol {
            r[[i, i]] = 0.0;
        }
    }

    QrResult { q, r }
}

/// Modified Gram-Schmidt QR decomposition (fallback when LAPACK is unavailable).
#[cfg(not(feature = "lapack"))]
fn qr_gram_schmidt(a: &Array2<f64>, config: &QrConfig) -> QrResult {
    let (m, n) = a.dim();
    let k = m.min(n);

    let mut q = Array2::<f64>::zeros((m, k));
    let mut r = Array2::<f64>::zeros((k, n));

    // Copy all columns of A into a working buffer
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let col: Vec<f64> = (0..m).map(|i| a[[i, j]]).collect();
        cols.push(col);
    }

    // Modified Gram-Schmidt
    for j in 0..k {
        let norm: f64 = cols[j].iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm < config.tol {
            r[[j, j]] = 0.0;
            continue;
        }

        r[[j, j]] = norm;
        for i in 0..m {
            q[[i, j]] = cols[j][i] / norm;
        }

        for p in (j + 1)..n {
            let dot: f64 = (0..m).map(|i| q[[i, j]] * cols[p][i]).sum();
            r[[j, p]] = dot;

            for i in 0..m {
                cols[p][i] -= dot * q[[i, j]];
            }
        }
    }

    QrResult { q, r }
}

/// Compute only the orthonormal factor Q from QR decomposition.
///
/// Convenience wrapper around [`qr`] that discards R.
/// Useful for re-orthonormalization of nearly-orthonormal matrices.
///
/// # Arguments
///
/// * `a` - Input matrix, shape (m, n)
/// * `config` - Configuration with tolerance for rank detection
///
/// # Returns
///
/// Q matrix with orthonormal columns, shape (m, min(m, n)).
pub fn qr_orthonormalize(a: &Array2<f64>, config: &QrConfig) -> Array2<f64> {
    qr(a, config).q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ——————————————————————————————————————————————
    // QrConfig
    // ——————————————————————————————————————————————

    // — Test 1: Config defaults —
    #[test]
    fn qr_config_default() {
        let config = QrConfig::default();
        assert!((config.tol - 1e-10).abs() < 1e-15);
    }

    // ——————————————————————————————————————————————
    // Core QR properties
    // ——————————————————————————————————————————————

    // — Test 2: Identity matrix —
    #[test]
    fn qr_identity_matrix() {
        let a = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = qr(&a, &QrConfig::default());

        // Q should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.q[[i, j]] - expected).abs() < 1e-10,
                    "Q[{i},{j}] = {}, expected {expected}",
                    result.q[[i, j]]
                );
            }
        }

        // R should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.r[[i, j]] - expected).abs() < 1e-10,
                    "R[{i},{j}] = {}, expected {expected}",
                    result.r[[i, j]]
                );
            }
        }
    }

    // — Test 3: Diagonal matrix —
    #[test]
    fn qr_diagonal_matrix() {
        let a = array![[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
        let result = qr(&a, &QrConfig::default());

        // R diagonal should be the original values (all positive)
        assert!((result.r[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((result.r[[1, 1]] - 3.0).abs() < 1e-10);
        assert!((result.r[[2, 2]] - 2.0).abs() < 1e-10);

        // Q should be identity (columns already orthonormal after scaling)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.q[[i, j]] - expected).abs() < 1e-10,
                    "Q[{i},{j}] = {}, expected {expected}",
                    result.q[[i, j]]
                );
            }
        }
    }

    // — Test 4: Reconstruction A = Q * R —
    #[test]
    fn qr_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = qr(&a, &QrConfig::default());
        let reconstructed = result.q.dot(&result.r);
        let (m, n) = a.dim();

        for i in 0..m {
            for j in 0..n {
                assert!(
                    (a[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({i},{j}): original={}, reconstructed={}",
                    a[[i, j]],
                    reconstructed[[i, j]]
                );
            }
        }
    }

    // — Test 5: Q columns orthonormal (Q^T Q ≈ I) —
    #[test]
    fn qr_q_columns_orthonormal() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = qr(&a, &QrConfig::default());
        let qtq = result.q.t().dot(&result.q);
        let k = qtq.nrows();

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q[{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }

    // — Test 6: R is upper triangular —
    #[test]
    fn qr_r_is_upper_triangular() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = qr(&a, &QrConfig::default());
        let (rows, cols) = result.r.dim();

        for i in 0..rows {
            for j in 0..cols {
                if i > j {
                    assert!(
                        result.r[[i, j]].abs() < 1e-10,
                        "R[{i},{j}] = {} should be 0 (below diagonal)",
                        result.r[[i, j]]
                    );
                }
            }
        }
    }

    // — Test 7: R diagonal non-negative —
    #[test]
    fn qr_r_diagonal_non_negative() {
        let a = array![[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0]];
        let result = qr(&a, &QrConfig::default());
        let k = result.r.nrows().min(result.r.ncols());

        for i in 0..k {
            assert!(
                result.r[[i, i]] >= -1e-10,
                "R[{i},{i}] = {} should be non-negative",
                result.r[[i, i]]
            );
        }
    }

    // ——————————————————————————————————————————————
    // Shape handling
    // ——————————————————————————————————————————————

    // — Test 8: Rectangular tall (m > n) —
    #[test]
    fn qr_rectangular_tall() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]; // 4x2
        let result = qr(&a, &QrConfig::default());

        assert_eq!(result.q.dim(), (4, 2), "Q shape");
        assert_eq!(result.r.dim(), (2, 2), "R shape");
    }

    // — Test 9: Rectangular wide (m < n) —
    #[test]
    fn qr_rectangular_wide() {
        let a = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]; // 2x4
        let result = qr(&a, &QrConfig::default());

        assert_eq!(result.q.dim(), (2, 2), "Q shape");
        assert_eq!(result.r.dim(), (2, 4), "R shape");
    }

    // — Test 10: Square matrix —
    #[test]
    fn qr_square_matrix() {
        let a = array![[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0]];
        let result = qr(&a, &QrConfig::default());

        assert_eq!(result.q.dim(), (3, 3), "Q shape");
        assert_eq!(result.r.dim(), (3, 3), "R shape");
    }

    // ——————————————————————————————————————————————
    // Edge cases
    // ——————————————————————————————————————————————

    // — Test 11: Rank-deficient matrix —
    #[test]
    fn qr_rank_deficient_matrix() {
        // Column 2 = 2 * Column 1 -> linearly dependent
        let a = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let result = qr(&a, &QrConfig::default());

        // R[0,0] should be non-zero (first column is fine)
        assert!(result.r[[0, 0]] > 1e-6, "R[0,0] should be > 0");

        // After orthogonalizing col2 against col1, residual should be ~0
        // So R[1,1] should be ~0
        assert!(
            result.r[[1, 1]].abs() < 1e-6,
            "R[1,1] = {} should be ~0 for rank-deficient matrix",
            result.r[[1, 1]]
        );
    }

    // — Test 12: Zero matrix —
    #[test]
    fn qr_zero_matrix() {
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let result = qr(&a, &QrConfig::default());

        for i in 0..result.r.nrows().min(result.r.ncols()) {
            assert!(
                result.r[[i, i]].abs() < 1e-10,
                "R[{i},{i}] = {} should be 0",
                result.r[[i, i]]
            );
        }
    }

    // — Test 13: Single column —
    #[test]
    fn qr_single_column() {
        let a = array![[3.0], [4.0]]; // 2x1, norm = 5
        let result = qr(&a, &QrConfig::default());

        assert_eq!(result.q.dim(), (2, 1), "Q shape");
        assert_eq!(result.r.dim(), (1, 1), "R shape");

        // R[0,0] should be the norm
        assert!((result.r[[0, 0]] - 5.0).abs() < 1e-10);

        // Q should be the normalized vector
        assert!((result.q[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((result.q[[1, 0]] - 0.8).abs() < 1e-10);
    }

    // — Test 14: Already orthonormal input —
    #[test]
    fn qr_already_orthonormal() {
        // Columns of a rotation matrix
        let s = f64::sqrt(0.5);
        let a = array![[s, s], [-s, s]];
        let result = qr(&a, &QrConfig::default());

        // Q should approximately equal input
        let (m, n) = a.dim();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (result.q[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Q[{i},{j}] = {}, expected {}",
                    result.q[[i, j]],
                    a[[i, j]]
                );
            }
        }

        // R should be approximately identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.r[[i, j]] - expected).abs() < 1e-10,
                    "R[{i},{j}] = {}, expected {expected}",
                    result.r[[i, j]]
                );
            }
        }
    }

    // ——————————————————————————————————————————————
    // Numerical properties
    // ——————————————————————————————————————————————

    // — Test 15: Orthonormality for ill-conditioned input —
    #[test]
    fn qr_orthonormality_ill_conditioned() {
        // Nearly parallel columns
        let a = array![[1.0, 1.0 + 1e-8], [0.0, 1e-8], [0.0, 0.0]];
        let result = qr(&a, &QrConfig::default());
        let qtq = result.q.t().dot(&result.q);
        let k = qtq.nrows();

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-6,
                    "Q^T Q[{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }

    // — Test 16: Reconstruction for different tall matrix —
    #[test]
    fn qr_reconstruction_tall_arbitrary() {
        let a = array![
            [2.0, -1.0, 3.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [3.0, 2.0, 0.0]
        ]; // 4x3
        let result = qr(&a, &QrConfig::default());
        let reconstructed = result.q.dot(&result.r);
        let (m, n) = a.dim();

        for i in 0..m {
            for j in 0..n {
                assert!(
                    (a[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({i},{j}): original={}, reconstructed={}",
                    a[[i, j]],
                    reconstructed[[i, j]]
                );
            }
        }
    }

    // ——————————————————————————————————————————————
    // qr_orthonormalize
    // ——————————————————————————————————————————————

    // — Test 17: Returns same Q as qr() —
    #[test]
    fn qr_orthonormalize_returns_q() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = QrConfig::default();
        let from_qr = qr(&a, &config).q;
        let from_ortho = qr_orthonormalize(&a, &config);

        assert_eq!(from_qr, from_ortho);
    }

    // — Test 18: Produces orthonormal columns —
    #[test]
    fn qr_orthonormalize_produces_orthonormal() {
        let a = array![[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let q = qr_orthonormalize(&a, &QrConfig::default());
        let qtq = q.t().dot(&q);
        let k = qtq.nrows();

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q[{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }
}
