//! Retractions: project back onto the Grassmann manifold.
//!
//! After perturbation (e.g., gradient step in ambient space), these functions
//! map the result back to a valid point on Gr(k, n):
//!   - [`retract_qr`] — via QR decomposition (fast, first-order)
//!   - [`retract_svd`] — via SVD (more stable, second-order)
//!   - [`retract_polar_gram`] — via `(W^T W)^(-1/2)` on the compact Gram matrix

use crate::geometry::qr::{QrConfig, qr_orthonormalize};
use crate::geometry::svd::{SvdConfig, svd};
use ndarray::Array2;

/// Retract a matrix back to Gr(k, n) using QR decomposition.
///
/// Given a matrix that has drifted from the Stiefel manifold (i.e., its columns
/// are no longer orthonormal), compute the closest point on St(k, n) via QR.
///
/// This is used after gradient updates to maintain the constraint W^T W = I.
///
/// # Mathematical Properties
/// - Projects to nearest point on Stiefel manifold
/// - Preserves column space (span)
/// - O(nk²) complexity
///
/// # Arguments
///
/// * `w_p` - Perturbed matrix, shape (n, k).
///
/// # Returns:
///
/// Orthonormalized matrix on St(k, n), shape (n, k).
///
/// # Note
/// QR decomposition can flip sign of columns. Use retract_svd if sign consistency is important.
pub fn retract_qr(w_p: &Array2<f64>) -> Array2<f64> {
    qr_orthonormalize(w_p, &QrConfig::default())
}

/// Retract a matrix back to Gr(k, n) using SVD (polar decomposition).
///
/// Given a matrix that has drifted from the Stiefel manifold, compute the
/// closest point via polar decomposition (SVD). This preserves orientation
/// better than QR and is the true metric projection.
///
/// # Mathematical Properties
/// - Polar factor: Q = U @ Vt where U, _, Vt = svd(W)
/// - Minimizes ||W- Q||_F over all orthonormal Q (Procrustes solution)
/// - Preserves sign/orientation
/// - O(nk²) complexity
///
/// # Arguments
///
/// * `w_p` - Perturbed matrix, shape (n, k).
///
/// # Returns
///
/// Orthonormalized matrix on St(k, n), shape (n, k).
///
/// # Note
/// More expensive than retract_qr but preserves orientation and is
/// the true orthogonal Procrustes solution.
pub fn retract_svd(w_p: &Array2<f64>) -> Array2<f64> {
    let svd_res = svd(w_p, &SvdConfig::default());
    svd_res.u.dot(&svd_res.vt)
}

/// Retract a tall-skinny matrix via the compact polar factor.
///
/// Computes `Q = W (W^T W)^(-1/2)` using only the `k x k` Gram matrix.
/// This is often more stable than MGS on tall-skinny matrices while avoiding
/// the full SVD of the ambient `n x k` matrix.
pub fn retract_polar_gram(w_p: &Array2<f64>) -> Array2<f64> {
    let gram_raw = w_p.t().dot(w_p);
    let gram = 0.5 * (&gram_raw + &gram_raw.t());
    let svd_res = svd(&gram, &SvdConfig::default());
    let k = svd_res.s.len();
    let inv_sqrt = Array2::from_shape_fn((k, k), |(i, j)| {
        if i == j {
            let sigma = svd_res.s[i].max(1e-12);
            sigma.sqrt().recip()
        } else {
            0.0
        }
    });
    let gram_inv_sqrt = svd_res.u.dot(&inv_sqrt).dot(&svd_res.vt);
    w_p.dot(&gram_inv_sqrt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Frobenius norm helper.
    fn frob_norm(a: &Array2<f64>) -> f64 {
        a.mapv(|x| x.powi(2)).sum().sqrt()
    }

    // ========================================================
    // retract_qr tests
    // ========================================================

    // — Test 1: Already orthonormal → unchanged —
    #[test]
    fn retract_qr_orthonormal_input_unchanged() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let q = retract_qr(&w);

        let diff = frob_norm(&(&w - &q));
        assert!(
            diff < 1e-10,
            "Orthonormal input should be unchanged, got diff {diff}"
        );
    }

    // — Test 2: Output has orthonormal columns —
    #[test]
    fn retract_qr_output_is_orthonormal() {
        let w = array![[1.1, 0.2], [0.3, 0.9], [0.1, 0.4]];
        let q = retract_qr(&w);

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

    // — Test 3: Preserves column space (span) —
    #[test]
    fn retract_qr_preserves_column_space() {
        // Perturbed version of identity columns
        let w = array![[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]];
        let q = retract_qr(&w);

        // Column space should still be span{e1, e2}
        // So q should have zeros in the third row
        assert!(q[[2, 0]].abs() < 1e-10);
        assert!(q[[2, 1]].abs() < 1e-10);
    }

    // — Test 4: Shape is preserved —
    #[test]
    fn retract_qr_preserves_shape() {
        let w = array![[1.0, 0.5], [0.3, 1.0], [0.2, 0.1], [0.0, 0.4]]; // 4x2
        let q = retract_qr(&w);
        assert_eq!(q.dim(), (4, 2));
    }

    // — Test 5: Single column —
    #[test]
    fn retract_qr_single_column() {
        let w = array![[3.0], [4.0]]; // norm = 5
        let q = retract_qr(&w);

        assert!((q[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((q[[1, 0]] - 0.8).abs() < 1e-10);
    }

    // ========================================================
    // retract_svd tests
    // ========================================================

    // — Test 6: Already orthonormal → unchanged —
    #[test]
    fn retract_svd_orthonormal_input_unchanged() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let q = retract_svd(&w);

        let diff = frob_norm(&(&w - &q));
        assert!(
            diff < 1e-6,
            "Orthonormal input should be unchanged, got diff {diff}"
        );
    }

    // — Test 7: Output has orthonormal columns —
    #[test]
    fn retract_svd_output_is_orthonormal() {
        let w = array![[1.1, 0.2], [0.3, 0.9], [0.1, 0.4]];
        let q = retract_svd(&w);

        let qtq = q.t().dot(&q);
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

    // — Test 8: SVD retraction is true Procrustes (minimizes ||W - Q||_F) —
    #[test]
    fn retract_svd_is_closer_than_qr() {
        let w = array![[1.2, 0.3], [0.1, 1.1], [0.4, 0.2]];

        let q_svd = retract_svd(&w);
        let q_qr = retract_qr(&w);

        let dist_svd = frob_norm(&(&w - &q_svd));
        let dist_qr = frob_norm(&(&w - &q_qr));

        // SVD retraction should be at least as close (it's the optimal solution)
        assert!(
            dist_svd <= dist_qr + 1e-10,
            "SVD should be closer: dist_svd={dist_svd}, dist_qr={dist_qr}"
        );
    }

    // — Test 9: Shape is preserved —
    #[test]
    fn retract_svd_preserves_shape() {
        let w = array![[1.0, 0.5], [0.3, 1.0], [0.2, 0.1], [0.0, 0.4]]; // 4x2
        let q = retract_svd(&w);
        assert_eq!(q.dim(), (4, 2));
    }

    #[test]
    fn retract_polar_gram_output_is_orthonormal_for_tall_skinny_matrix() {
        let w = Array2::from_shape_fn((128, 16), |(row, col)| {
            (((row * 37 + col * 19 + 11) % 101) as f64 / 101.0) - 0.5
        });
        let q = retract_polar_gram(&w);
        let qtq = q.t().dot(&q);
        let k = qtq.nrows();
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-5,
                    "Q^T Q[{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }

    // — Test 10: Both retractions agree on orthonormal input —
    #[test]
    fn both_retractions_agree_on_orthonormal() {
        let s = f64::sqrt(0.5);
        let w = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let q_qr = retract_qr(&w);
        let q_svd = retract_svd(&w);

        // Both should return (approximately) the same thing for orthonormal input
        let diff = frob_norm(&(&q_qr - &q_svd));
        assert!(
            diff < 1e-6,
            "Both retractions should agree on orthonormal input, got diff {diff}"
        );
    }
}
