//! Spectral helpers for skew-symmetric and subspace geometry.
//!
//! - [`skew_eigenvalues`] returns the non-negative pair-singular values
//!   $\beta_j$ of a real skew-symmetric matrix $A$. The eigenvalues of $A$
//!   are $\pm i\beta_j$ (with one extra zero if $A$ has odd dimension),
//!   and the eigenvalues of the Hermitian matrix $iA$ are $\pm\beta_j$.
//! - [`principal_angles_full`] returns both $\sigma = \cos\theta$ and
//!   $\theta$ for two subspaces, extending
//!   [`grassmann::principal_angles`](crate::geometry::grassmann::principal_angles)
//!   which returns $\theta$ only.
//! - `eigh` (LAPACK only) returns eigenvalues and eigenvectors of a
//!   real symmetric matrix in ascending order.
//!
//! Skew-eigenvalues and `principal_angles_full` work on both LAPACK and
//! pure-Rust SVD backends; `eigh` requires the `lapack` feature.

use ndarray::{Array1, Array2};

use crate::geometry::{
    grassmann::GrassmannError,
    svd::{SvdConfig, svdvals},
};

/// Pair-singular values $\beta_j \geq 0$ of a real skew-symmetric matrix
/// $A \in \mathbb{R}^{n\times n}$.
///
/// # Mathematical setup
///
/// For $A^\top = -A$:
/// - Eigenvalues of $A$: $\pm i\beta_j$ for $j = 1, \dots, \lfloor n/2\rfloor$,
///   plus one extra zero if $n$ is odd.
/// - Eigenvalues of $iA$ (Hermitian): $\pm\beta_j$, plus one extra zero if odd.
/// - Singular values of $A$: $\beta_j$, each appearing with multiplicity 2,
///   plus one zero if $n$ is odd.
///
/// This function returns the unique $\beta_j$ in non-increasing order
/// — a vector of length $\lfloor n/2 \rfloor$.
///
/// # Inputs
///
/// `a` is assumed skew-symmetric within numerical tolerance. Callers that
/// are uncertain should antisymmetrize via `(A - A^T) / 2` beforehand;
/// `skew_eigenvalues` does not validate the input.
///
/// # Returns
///
/// `Array1<f64>` of length $\lfloor n/2 \rfloor$, in non-increasing order.
pub fn skew_eigenvalues(a: &Array2<f64>) -> Array1<f64> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols(), "skew_eigenvalues requires a square matrix");
    let svs = svdvals(a, &SvdConfig::default());
    let n_pairs = n / 2;
    let mut out = Array1::<f64>::zeros(n_pairs);
    for j in 0..n_pairs {
        // Pair-singular values: take every other SV from a sorted-descending
        // list. svdvals returns σ_1 ≥ σ_2 ≥ … so index 2j picks one element
        // from each conjugate pair.
        out[j] = svs[2 * j];
    }
    out
}

/// Principal angles between two orthonormal bases, returning both the
/// cosines $\sigma_i = \cos\theta_i$ and the angles $\theta_i$.
///
/// $\sigma_i$ are the singular values of $W_1^\top W_2$, clamped to
/// $[0, 1]$ to absorb floating-point drift, and $\theta_i = \arccos\sigma_i \in [0, \pi/2]$.
///
/// # Returns
///
/// A pair `(sigma, theta)`, each of length $\min(k_1, k_2)$, in
/// non-decreasing order of $\theta$ (equivalently non-increasing order
/// of $\sigma$).
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if the ambient dimensions
/// of `w1` and `w2` differ.
pub fn principal_angles_full(
    w1: &Array2<f64>,
    w2: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), GrassmannError> {
    if w1.nrows() != w2.nrows() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w1.nrows(),
            got: w2.nrows(),
        });
    }
    let gram = w1.t().dot(w2);
    let cosines = svdvals(&gram, &SvdConfig::default()).mapv(|s| s.clamp(0.0, 1.0));
    let angles = cosines.mapv(|s| s.acos());
    Ok((cosines, angles))
}

/// Real symmetric eigendecomposition (LAPACK path).
///
/// Returns eigenvalues in ascending order and the corresponding
/// eigenvectors as columns. Uses the upper-triangular part of `a`.
///
/// # Panics
///
/// Panics if the LAPACK call fails (typically only on shape errors or
/// non-finite entries).
#[cfg(feature = "lapack")]
pub fn eigh(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    use ndarray_linalg::{Eigh, UPLO};
    a.eigh(UPLO::Upper).expect("LAPACK eigh failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOL: f64 = 1e-10;

    // ─── skew_eigenvalues ──────────────────────────────────────────

    #[test]
    fn skew_eigenvalues_2x2_block() {
        // A = [[0, b], [-b, 0]]  has β = b (single pair).
        let a = array![[0.0, 3.0], [-3.0, 0.0]];
        let beta = skew_eigenvalues(&a);
        assert_eq!(beta.len(), 1);
        assert!((beta[0] - 3.0).abs() < TOL);
    }

    #[test]
    fn skew_eigenvalues_block_diag_4x4() {
        // Two 2x2 antisymmetric blocks with β1 = 5, β2 = 2.
        let a = array![
            [0.0, 5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0, 0.0],
        ];
        let beta = skew_eigenvalues(&a);
        assert_eq!(beta.len(), 2);
        // Returned in non-increasing order
        assert!((beta[0] - 5.0).abs() < TOL);
        assert!((beta[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn skew_eigenvalues_zero_matrix() {
        let a: Array2<f64> = Array2::zeros((4, 4));
        let beta = skew_eigenvalues(&a);
        assert_eq!(beta.len(), 2);
        for &b in beta.iter() {
            assert!(b.abs() < TOL);
        }
    }

    #[test]
    fn skew_eigenvalues_count_for_odd_n() {
        // A 3x3 antisymmetric has eigenvalues {±iβ, 0}, so β has length 1.
        let a = array![[0.0, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]];
        let beta = skew_eigenvalues(&a);
        assert_eq!(beta.len(), 1);
        // Frobenius norm is √(2(1+4+9)) = √28; pair-singular value is the
        // unique positive eigenvalue magnitude.
        // For a 3x3 antisymmetric the (only) β = ||a||_F / √2.
        let f_norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let expected = f_norm / std::f64::consts::SQRT_2;
        assert!(
            (beta[0] - expected).abs() < 1e-9,
            "beta[0] = {}, expected = {}",
            beta[0],
            expected
        );
    }

    // ─── principal_angles_full ─────────────────────────────────────

    #[test]
    fn principal_angles_full_identity() {
        // Same basis: σ_i = 1, θ_i = 0.
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let (sigma, theta) = principal_angles_full(&w, &w).unwrap();
        assert_eq!(sigma.len(), 2);
        for &s in sigma.iter() {
            assert!((s - 1.0).abs() < TOL);
        }
        for &t in theta.iter() {
            assert!(t.abs() < TOL);
        }
    }

    #[test]
    fn principal_angles_full_orthogonal() {
        // Orthogonal bases: σ_i = 0, θ_i = π/2.
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[0.0], [1.0], [0.0]];
        let (sigma, theta) = principal_angles_full(&w1, &w2).unwrap();
        assert_eq!(sigma.len(), 1);
        assert!(sigma[0].abs() < TOL);
        assert!((theta[0] - std::f64::consts::FRAC_PI_2).abs() < TOL);
    }

    #[test]
    fn principal_angles_full_consistent_with_arccos() {
        // For random (well-conditioned) bases, σ = cos θ exactly.
        // We construct W1 and W2 with a known angle.
        let cos_a = 0.6;
        let sin_a = 0.8;
        let w1 = array![[1.0], [0.0]];
        let w2 = array![[cos_a], [sin_a]];
        let (sigma, theta) = principal_angles_full(&w1, &w2).unwrap();
        assert_eq!(sigma.len(), 1);
        assert!((sigma[0] - cos_a).abs() < TOL);
        assert!((theta[0] - cos_a.acos()).abs() < TOL);
    }

    #[test]
    fn principal_angles_full_dimension_mismatch_errors() {
        let w1 = Array2::<f64>::zeros((4, 2));
        let w2 = Array2::<f64>::zeros((3, 2));
        let err = principal_angles_full(&w1, &w2).unwrap_err();
        assert!(matches!(
            err,
            GrassmannError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    // ─── eigh ──────────────────────────────────────────────────────

    #[cfg(feature = "lapack")]
    #[test]
    fn eigh_diagonal_matrix() {
        // A = diag(2, 5, 1) → eigenvalues ascending: 1, 2, 5.
        let a = array![[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]];
        let (vals, _vecs) = eigh(&a);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < TOL);
        assert!((vals[1] - 2.0).abs() < TOL);
        assert!((vals[2] - 5.0).abs() < TOL);
    }

    #[cfg(feature = "lapack")]
    #[test]
    fn eigh_recovers_negative_squared_skew() {
        // For A skew-symmetric, -A^2 is symmetric PSD with eigenvalues β_j^2
        // each with multiplicity 2 (plus a 0 if n odd).
        let a = array![
            [0.0, 5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0, 0.0],
        ];
        let neg_a2 = (-1.0) * a.dot(&a);
        let (vals, _) = eigh(&neg_a2);
        // Sorted ascending: 4, 4, 25, 25.
        assert!((vals[0] - 4.0).abs() < TOL);
        assert!((vals[1] - 4.0).abs() < TOL);
        assert!((vals[2] - 25.0).abs() < TOL);
        assert!((vals[3] - 25.0).abs() < TOL);
        // Cross-check against skew_eigenvalues: β = (5, 2), β² = (25, 4).
        let beta = skew_eigenvalues(&a);
        let mut beta_sq = beta.mapv(|b| b * b).to_vec();
        beta_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((beta_sq[0] - 4.0).abs() < TOL);
        assert!((beta_sq[1] - 25.0).abs() < TOL);
    }
}
