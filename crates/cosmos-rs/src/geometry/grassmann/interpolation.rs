//! Geodesic interpolation between subspaces on Gr(k, n).
//!
//! Manifold-aware interpolation:
//!   - [`geodesic_interpolate`] — Riemannian analog of linear interpolation

use ndarray::Array2;

use crate::geometry::grassmann::{
    ExponentialMapConfig, GrassmannError, exponential_map, logarithmic_map,
};

/// Interpolate along geodesic between two points on Gr(k, n).
///
/// Given two subspaces W_start and W_end, compute the point at fraction t
/// along the geodesic connecting them. This is the manifold-aware analog
/// of linear interpolation.
///
/// # Mathematical Foundation
/// The geodesic from W_start to W_end is parameterized by:
/// - γ(t) = Exp_{W_start}(t · Log_{W_start}(W_end))
///
/// For t=0: γ(0) = W_start
/// For t=1: γ(1) = W_end
/// For t=0.5: γ(0.5) = midpoint on the geodesic
///
/// # Arguments
///
/// * `w_start` - Starting point on Gr(k, n), shape (n, k) with orthonormal columns.
/// * `w_end` - Ending point on Gr(k, n), shape (n, k) with orthonormal columns.
/// * `t` - Interpolation parameter in [0, 1]. Default 0.5 (midpoint).
///
/// # Returns
///
/// Interpolated point on Gr(k, n), shape (n, k) with orthonormal columns.
///
/// # Note
/// For identical subspaces, returns W_start unchanged.
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if `w_start` and `w_end`
/// have a different number of rows.
pub fn geodesic_interpolate(
    w_start: &Array2<f64>,
    w_end: &Array2<f64>,
    t: f64,
) -> Result<Array2<f64>, GrassmannError> {
    if w_start.nrows() != w_end.nrows() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w_start.nrows(),
            got: w_end.nrows(),
        });
    }

    // Edge cases
    if t <= 0.0 {
        return Ok(w_start.clone());
    }

    if t >= 1.0 {
        return Ok(w_end.clone());
    }

    // Compute tangent vector via Log map
    let tangent_vec = logarithmic_map(w_start, w_end)?;

    // Move along geodesic via Exp map
    exponential_map(w_start, &tangent_vec, &ExponentialMapConfig { t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::grassmann::distances::geodesic_distance;
    use ndarray::array;

    /// Frobenius norm helper.
    fn frob_norm(a: &Array2<f64>) -> f64 {
        a.mapv(|x| x.powi(2)).sum().sqrt()
    }

    // — Test 1: t=0 returns W_start —
    #[test]
    fn interpolate_t0_returns_start() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let result = geodesic_interpolate(&w1, &w2, 0.0).unwrap();
        let diff = frob_norm(&(&result - &w1));
        assert!(diff < 1e-10, "t=0 should return W_start, got diff {diff}");
    }

    // — Test 2: t=1 returns W_end —
    #[test]
    fn interpolate_t1_returns_end() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let result = geodesic_interpolate(&w1, &w2, 1.0).unwrap();
        let diff = frob_norm(&(&result - &w2));
        assert!(diff < 1e-10, "t=1 should return W_end, got diff {diff}");
    }

    // — Test 3: t=0.5 gives midpoint (equidistant from both) —
    #[test]
    fn interpolate_midpoint_is_equidistant() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [0.0, s], [s, 0.0], [0.0, s]];

        let mid = geodesic_interpolate(&w1, &w2, 0.5).unwrap();

        let d1 = geodesic_distance(&w1, &mid).unwrap();
        let d2 = geodesic_distance(&mid, &w2).unwrap();

        assert!(
            (d1 - d2).abs() < 1e-6,
            "Midpoint should be equidistant: d1={d1}, d2={d2}"
        );
    }

    // — Test 4: Distances scale linearly with t —
    #[test]
    fn interpolate_distances_scale_linearly() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [0.0, s], [s, 0.0], [0.0, s]];

        let d_total = geodesic_distance(&w1, &w2).unwrap();

        let w_quarter = geodesic_interpolate(&w1, &w2, 0.25).unwrap();
        let d_quarter = geodesic_distance(&w1, &w_quarter).unwrap();

        assert!(
            (d_quarter - 0.25 * d_total).abs() < 1e-6,
            "d(W1, γ(0.25)) should be 0.25 * d_total: {} vs {}",
            d_quarter,
            0.25 * d_total
        );
    }

    // — Test 5: Output has orthonormal columns —
    #[test]
    fn interpolate_output_is_orthonormal() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [0.0, s], [s, 0.0], [0.0, s]];

        let mid = geodesic_interpolate(&w1, &w2, 0.5).unwrap();

        let qtq = mid.t().dot(&mid);
        let k = qtq.nrows();
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-6,
                    "W^T W[{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }

    // — Test 6: Identical subspaces → returns same point —
    #[test]
    fn interpolate_identical_subspaces() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let result = geodesic_interpolate(&w, &w, 0.5).unwrap();
        let d = geodesic_distance(&w, &result).unwrap();
        assert!(
            d < 1e-10,
            "Same subspaces should give same point, got distance {d}"
        );
    }

    // — Test 7: Dimension mismatch → Err —
    #[test]
    fn interpolate_dimension_mismatch() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = geodesic_interpolate(&w1, &w2, 0.5);
        assert!(result.is_err());
    }

    // — Test 8: Interpolation lies on geodesic (triangle inequality tight) —
    #[test]
    fn interpolate_lies_on_geodesic() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [0.0, s], [s, 0.0], [0.0, s]];

        let d_total = geodesic_distance(&w1, &w2).unwrap();
        let w_mid = geodesic_interpolate(&w1, &w2, 0.5).unwrap();

        let d1 = geodesic_distance(&w1, &w_mid).unwrap();
        let d2 = geodesic_distance(&w_mid, &w2).unwrap();

        // On geodesic: d(W1, mid) + d(mid, W2) = d(W1, W2) exactly
        assert!(
            (d1 + d2 - d_total).abs() < 1e-6,
            "Should lie on geodesic: d1 + d2 = {} vs d_total = {}",
            d1 + d2,
            d_total
        );
    }

    // — Test 9: 1D interpolation —
    #[test]
    fn interpolate_1d() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[0.0], [1.0], [0.0]];

        let d_total = geodesic_distance(&w1, &w2).unwrap();
        let w_third = geodesic_interpolate(&w1, &w2, 1.0 / 3.0).unwrap();
        let d_third = geodesic_distance(&w1, &w_third).unwrap();

        assert!(
            (d_third - d_total / 3.0).abs() < 1e-6,
            "Expected d_total/3 = {}, got {d_third}",
            d_total / 3.0
        );
    }

    // — Test 10: Negative t clamps to W_start —
    #[test]
    fn interpolate_negative_t_returns_start() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let w2 = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let result = geodesic_interpolate(&w1, &w2, -0.5).unwrap();
        let diff = frob_norm(&(&result - &w1));
        assert!(diff < 1e-10, "t<0 should return W_start");
    }

    // — Test 11: t > 1 clamps to W_end —
    #[test]
    fn interpolate_t_above_1_returns_end() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let w2 = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let result = geodesic_interpolate(&w1, &w2, 1.5).unwrap();
        let diff = frob_norm(&(&result - &w2));
        assert!(diff < 1e-10, "t>1 should return W_end");
    }
}
