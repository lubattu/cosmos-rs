//! Distance metrics and principal angles on the Grassmann manifold.
//!
//! Functions for measuring distances between subspaces:
//!   - [`principal_angles`] — fundamental building block
//!   - [`geodesic_distance`] — intrinsic Riemannian distance
//!   - [`chordal_distance`] — extrinsic (projector-based) distance
//!   - [`projection_distance`] — Frobenius norm of projector difference
//!
use super::GrassmannError;
use crate::geometry::svd::{SvdConfig, svdvals};
use ndarray::{Array1, Array2};

/// Compute principal angles between two subspaces.
///
/// Given two orthonormal bases W1 ∈ ℝ^(n×k1) and W2 ∈ ℝ^(n×k2),
/// compute the principal angles θ_i ∈ [0, π/2] between their column spaces.
///
/// The principal angles are defined via the singular values of W1^T W2:
///
/// $$\cos(\theta_i) = \sigma_i(W_1^\top W_2)$$
///
/// # Mathematical Properties
///
/// - 0 ≤ θ₁ ≤ θ₂ ≤ ... ≤ θ_min(k1,k2) ≤ π/2
/// - θ = 0 iff subspaces share a common direction
/// - θ = π/2 iff subspaces are orthogonal in that direction
///
/// # Arguments
///
/// * `w1` - First orthonormal basis, shape (n, k1)
/// * `w2` - Second orthonormal basis, shape (n, k2)
///
/// # Returns
///
/// Principal angles in radians, shape (min(k1, k2),), sorted ascending.
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if `w1` and `w2` have
/// different numbers of rows (ambient dimension).
pub fn principal_angles(w1: &Array2<f64>, w2: &Array2<f64>) -> Result<Array1<f64>, GrassmannError> {
    if w1.nrows() != w2.nrows() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w1.nrows(),
            got: w2.nrows(),
        });
    }
    // Gram matrix: G = W1^T @ W2
    let gram = w1.t().dot(w2);

    // Singular values of G give cosines of principal angles
    let singular_values = svdvals(&gram, &SvdConfig::default());

    // Clamp to [0, 1] and compute arccos -> principal angles
    let angles = singular_values.mapv(|s| s.clamp(0.0, 1.0).acos());

    Ok(angles)
}

/// Compute geodesic distance between two points on Gr(k, n).
///
/// The geodesic distance is the length of the shortest path on the manifold
/// connecting the two subspaces. It is computed from principal angles:
///
/// d_geo(W1, W2) = ||θ||_2 = sqrt(Σ θ_i²)
///
/// where θ_i are the principal angles between span(W1) and span(W2).
///
/// This is the intrinsic Riemannian distance on Gr(k, n) and respects
/// the manifold geometry. It is invariant to the choice of orthonormal
/// basis representing each subspace.
///
/// # Mathematical Properties
/// - d_geo(W1, W2) ≥ 0 (non-negative)
/// - d_geo(W1, W2) = 0 iff span(W1) = span(W2) (identity)
/// - d_geo(W1, W2) = d_geo(W2, W1) (symmetric)
/// - d_geo(W1, W3) ≤ d_geo(W1, W2) + d_geo(W2, W3) (triangle inequality)
///
/// # Arguments
///
/// * `w1` - First orthonormal basis, shape (n, k1)
/// * `w2` - Second orthonormal basis, shape (n, k2)
///
/// # Returns
///
/// Geodesic distance (scalar).
///
/// # Errors
/// Returns [`GrassmannError::DimensionMismatch`] if `w1` and `w2` have
/// different numbers of rows (ambient dimensions).
pub fn geodesic_distance(w1: &Array2<f64>, w2: &Array2<f64>) -> Result<f64, GrassmannError> {
    let angles = principal_angles(w1, w2)?;

    let distance = angles
        .iter()
        .map(|&theta| theta.powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(distance)
}

/// Compute chordal distance between two points on Gr(k, n)
///
/// The chordal distance measures the Frobenius norm difference between
/// projection matrices:
///
/// d_chord(w1, w2) = ||P1 - P2||_F / sqrt(2)
///
/// where P1 = w1 w1^T and P2 = w2 w2^T
///
/// This is an extrinsic distance (embeds Gr(k,n) in R^(n×n)) but is
/// computationally cheaper than geodesic distance. It relates to
/// principal angles via:
///
/// d_chord² = (1/2) Σ sin²(θ_i)
///
/// # Mathematical Properties
/// - 0 ≤ d_chord ≤ sqrt(k) (bounded)
/// - d_chord = 0 iff span(w1) = span(w2)
/// - d_chord = 0 iff span(w1) = span(w2)
/// - Faster to compute than geodesic distance
///
/// # Arguments
///
/// * `w1` - First orthonormal basis, shape (n, k1)
/// * `w2` - Second orthonormal basis, shape (n, k2)
///
/// # Returns
///
/// Chordal distance (scalar)
///
/// # Errors
/// Returns [`GrassmannError::DimensionMismatch`] if `w1` and `w2` have
/// different numbers of rows (ambient dimensions).
pub fn chordal_distance(w1: &Array2<f64>, w2: &Array2<f64>) -> Result<f64, GrassmannError> {
    let angles = principal_angles(w1, w2)?;

    let distance = angles
        .iter()
        .map(|&theta| theta.sin().powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(distance)
}

/// Compute projection F-norm distance between subspaces.
///
/// This is the unnormalized Frobenius norm of the difference between
/// projection matrices:
///
/// d_proj(w1, w2) = ||P1 - P2||_F = ||w1 w1^T - w2 w2^T||_F
///
/// Unlike chordal distance, this is NOT normalizaded. It relates to
/// principal angles via:
///
/// d_proj² = 2k - 2 tr(W1^T W2 W2^T W1) = 2 Σ sin²(θ_i)
///
/// This distance is useful for:
/// - Loss functions (differentiable)
/// - Direct optimization objectives
/// - Gradient-based methods
///
/// # Arguments
/// * `w1` - First orthonormal basis, shape (n, k1)
/// * `w2` - Second orthonormal basis, shape (n, k2)
///
/// # Returns
///
/// Projection Distance (scalar)
///
/// # Errors
/// Returns [`GrassmannError::DimensionMismatch`] if `w1` and `w2` have
/// different numbers of rows (ambient dimensions).
pub fn projection_distance(w1: &Array2<f64>, w2: &Array2<f64>) -> Result<f64, GrassmannError> {
    if w1.nrows() != w2.nrows() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w1.nrows(),
            got: w2.nrows(),
        });
    }

    // Compute projection matrices
    let p1 = w1.dot(&w1.t());
    let p2 = w2.dot(&w2.t());

    // Frobenius norm of difference
    let diff = p1 - p2;

    let distance = diff.mapv(|x| x.powi(2)).sum().sqrt();

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::FRAC_PI_2;

    // —— Test 1: 3x2 identity matrix ——
    #[test]
    fn identical_subspaces_give_zero_angles() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let angles = principal_angles(&w, &w).unwrap();
        // .unwrap() here is safe — we know the dimensions match

        for &angle in angles.iter() {
            // Each angle should be ~0
            assert!((angle).abs() < 1e-6, "Expected ~0, got {angle}");
        }
    }

    // —— Test 2: orthogonal 1D subspaces give angle π/2 ——
    #[test]
    fn orthogonal_subspaces_give_pi_over_2() {
        let w1_1d = array![[1.0], [0.0], [0.0]]; // span{e1}

        let w2_1d = array![[0.0], [0.0], [1.0]]; // span{e3}

        let angles = principal_angles(&w1_1d, &w2_1d).unwrap();

        assert!(
            (angles[0] - FRAC_PI_2).abs() < 1e-6,
            "Expected π/2, got {}",
            angles[0]
        );
    }

    // —— Test 3: incompatible dimensions → Err ——
    #[test]
    fn dimension_mismatch_returns_error() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = principal_angles(&w1, &w2);

        assert_eq!(
            result,
            Err(GrassmannError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    // —— Test 4: known 45° angle ——
    #[test]
    fn known_45_degree_angle() {
        use std::f64::consts::FRAC_PI_4;

        // W1 = span{e1}
        let w1 = array![[1.0], [0.0]];

        // W2 = span{(1/√2, 1/√2)} — e1 rotated 45°
        let s = (0.5_f64).sqrt();
        let w2 = array![[s], [s]];

        let angles = principal_angles(&w1, &w2).unwrap();

        assert!(
            (angles[0] - FRAC_PI_4).abs() < 1e-6,
            "Expected π/4, got {}",
            angles[0]
        );
    }

    // —— Test 5: all angles lie in [0, π/2] ——
    #[test]
    fn angles_are_in_valid_range() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let angles = principal_angles(&w1, &w2).unwrap();

        for &angle in angles.iter() {
            assert!(angle >= -1e-10, "Angle below 0: {angle}");
            assert!(angle <= FRAC_PI_2 + 1e-10, "Angle above π/2: {angle}");
        }
    }

    // —— Test 6: symmetry — angles(W1, W2) == angles(W2, W1) ——
    #[test]
    fn principal_angles_symmetry() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let angles_ab = principal_angles(&w1, &w2).unwrap();
        let angles_ba = principal_angles(&w2, &w1).unwrap();

        assert_eq!(angles_ab.len(), angles_ba.len());
        for i in 0..angles_ab.len() {
            assert!(
                (angles_ab[i] - angles_ba[i]).abs() < 1e-6,
                "Asymmetry at index {i}: {} vs {}",
                angles_ab[i],
                angles_ba[i]
            );
        }
    }

    // —— Test 7: 1D subspaces → returns exactly 1 angle ——
    #[test]
    fn single_dimension_subspaces() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[0.0], [1.0], [0.0]];

        let angles = principal_angles(&w1, &w2).unwrap();

        assert_eq!(angles.len(), 1, "Expected 1 angle, got {}", angles.len());
    }

    // —— Test 8: k1 ≠ k2 → returns min(k1, k2) angles ——
    #[test]
    fn different_subspace_dimensions() {
        // W1: 4×3 (3 columns)
        let w1 = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ];

        // W2: 4×1 (1 column)
        let w2 = array![[1.0], [0.0], [0.0], [0.0]];

        let angles = principal_angles(&w1, &w2).unwrap();

        // min(3, 1) = 1 angle
        assert_eq!(
            angles.len(),
            1,
            "Expected min(3,1)=1 angle, got {}",
            angles.len()
        );

        // W2 is span{e1}, which is contained in W1 → angle = 0
        assert!(angles[0].abs() < 1e-6, "Expected ~0, got {}", angles[0]);
    }

    // ========================================================
    // geodesic_distance tests
    // ========================================================

    // —— Test 9: identical subspaces → geodesic distance = 0 ——
    #[test]
    fn geodesic_identical_subspaces_give_zero() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let d = geodesic_distance(&w, &w).unwrap();
        assert!(d.abs() < 1e-6, "Expected ~0, got {d}");
    }

    // —— Test 10: orthogonal 1D subspaces → distance = π/2 ——
    #[test]
    fn geodesic_orthogonal_1d_gives_pi_over_2() {
        let w1 = array![[1.0], [0.0], [0.0]]; // span{e1}
        let w2 = array![[0.0], [0.0], [1.0]]; // span{e3}

        let d = geodesic_distance(&w1, &w2).unwrap();
        assert!((d - FRAC_PI_2).abs() < 1e-6, "Expected π/2, got {d}");
    }

    // —— Test 11: 45° angle → geodesic distance = π/4 ——
    #[test]
    fn geodesic_known_45_degree() {
        use std::f64::consts::FRAC_PI_4;

        let w1 = array![[1.0], [0.0]];
        let s = (0.5_f64).sqrt();
        let w2 = array![[s], [s]];

        let d = geodesic_distance(&w1, &w2).unwrap();
        assert!((d - FRAC_PI_4).abs() < 1e-6, "Expected π/4, got {d}");
    }

    // —— Test 12: symmetry — d_geo(W1, W2) == d_geo(W2, W1) ——
    #[test]
    fn geodesic_symmetry() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let d_ab = geodesic_distance(&w1, &w2).unwrap();
        let d_ba = geodesic_distance(&w2, &w1).unwrap();

        assert!((d_ab - d_ba).abs() < 1e-6, "Asymmetry: {d_ab} vs {d_ba}");
    }

    // —— Test 13: incompatible dimensions → Err ——
    #[test]
    fn geodesic_dimension_mismatch_returns_error() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = geodesic_distance(&w1, &w2);
        assert!(result.is_err());
    }

    // —— Test 14: geodesic distance is non-negative ——
    #[test]
    fn geodesic_is_non_negative() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let d = geodesic_distance(&w1, &w2).unwrap();
        assert!(d >= 0.0, "Distance should be non-negative, got {d}");
    }

    // —— Test 15: 2D subspaces with multiple angles → d = ||(θ1, θ2)||_2 ——
    #[test]
    fn geodesic_multidimensional_known_value() {
        // W1 = span{e1, e2}, W2 = span{e3, e4} → both angles = π/2
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let w2 = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let d = geodesic_distance(&w1, &w2).unwrap();
        // d = sqrt((π/2)² + (π/2)²) = (π/2) * sqrt(2)
        let expected = FRAC_PI_2 * 2.0_f64.sqrt();
        assert!(
            (d - expected).abs() < 1e-6,
            "Expected (π/2)√2 ≈ {expected}, got {d}"
        );
    }

    // ========================================================
    // chordal_distance tests
    // ========================================================

    // —— Test 16: identical subspaces → chordal distance = 0 ——
    #[test]
    fn chordal_identical_subspaces_give_zero() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let d = chordal_distance(&w, &w).unwrap();
        assert!(d.abs() < 1e-6, "Expected ~0, got {d}");
    }

    // —— Test 17: orthogonal 1D subspaces → distance = sin(π/2) = 1 ——
    #[test]
    fn chordal_orthogonal_1d_gives_one() {
        let w1 = array![[1.0], [0.0], [0.0]]; // span{e1}
        let w2 = array![[0.0], [0.0], [1.0]]; // span{e3}

        let d = chordal_distance(&w1, &w2).unwrap();
        assert!((d - 1.0).abs() < 1e-6, "Expected 1.0, got {d}");
    }

    // —— Test 18: 45° angle → chordal distance = sin(π/4) = 1/√2 ——
    #[test]
    fn chordal_known_45_degree() {
        let w1 = array![[1.0], [0.0]];
        let s = (0.5_f64).sqrt();
        let w2 = array![[s], [s]];

        let d = chordal_distance(&w1, &w2).unwrap();
        let expected = (std::f64::consts::FRAC_PI_4).sin(); // sin(π/4) = 1/√2
        assert!(
            (d - expected).abs() < 1e-6,
            "Expected sin(π/4) ≈ {expected}, got {d}"
        );
    }

    // —— Test 19: Simetria — d_chord(W1, W2) == d_chord(W2, W1) ——
    #[test]
    fn chordal_symmetry() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let d_ab = chordal_distance(&w1, &w2).unwrap();
        let d_ba = chordal_distance(&w2, &w1).unwrap();

        assert!((d_ab - d_ba).abs() < 1e-6, "Asymmetry: {d_ab} vs {d_ba}");
    }

    // —— Test 20: incompatible dimensions → Err ——
    #[test]
    fn chordal_dimension_mismatch_returns_error() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = chordal_distance(&w1, &w2);
        assert!(result.is_err());
    }

    // —— Test 21: d_chord ≤ sqrt(k) — upper bound ——
    #[test]
    fn chordal_bounded_by_sqrt_k() {
        // Fully orthogonal 2D subspaces → maximal d_chord
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let w2 = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let d = chordal_distance(&w1, &w2).unwrap();
        let k = w1.ncols() as f64;
        assert!(d <= k.sqrt() + 1e-6, "Expected d ≤ sqrt({k}), got {d}");
    }

    // —— Test 22: orthogonal 2D subspaces → d_chord = sqrt(2) ——
    #[test]
    fn chordal_orthogonal_2d_gives_sqrt2() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let w2 = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let d = chordal_distance(&w1, &w2).unwrap();
        // 2 angles at π/2 → sqrt(sin²(π/2) + sin²(π/2)) = sqrt(2)
        let expected = 2.0_f64.sqrt();
        assert!(
            (d - expected).abs() < 1e-6,
            "Expected √2 ≈ {expected}, got {d}"
        );
    }

    // ========================================================
    // projection_distance tests
    // ========================================================

    // —— Test 23: identical subspaces → projection distance = 0 ——
    #[test]
    fn projection_identical_subspaces_give_zero() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let d = projection_distance(&w, &w).unwrap();
        assert!(d.abs() < 1e-6, "Expected ~0, got {d}");
    }

    // —— Test 24: orthogonal 1D subspaces → ||P1 - P2||_F = sqrt(2) ——
    #[test]
    fn projection_orthogonal_1d_gives_sqrt2() {
        let w1 = array![[1.0], [0.0], [0.0]]; // span{e1}
        let w2 = array![[0.0], [0.0], [1.0]]; // span{e3}

        // P1 = diag(1,0,0), P2 = diag(0,0,1)
        // P1 - P2 = diag(1, 0, -1) → ||.||_F = sqrt(1+0+1) = sqrt(2)
        let d = projection_distance(&w1, &w2).unwrap();
        let expected = 2.0_f64.sqrt();
        assert!(
            (d - expected).abs() < 1e-6,
            "Expected √2 ≈ {expected}, got {d}"
        );
    }

    // —— Test 25: symmetry — d_proj(W1, W2) == d_proj(W2, W1) ——
    #[test]
    fn projection_symmetry() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s]];

        let d_ab = projection_distance(&w1, &w2).unwrap();
        let d_ba = projection_distance(&w2, &w1).unwrap();

        assert!((d_ab - d_ba).abs() < 1e-6, "Asymmetry: {d_ab} vs {d_ba}");
    }

    // —— Test 26: incompatible dimensions → Err ——
    #[test]
    fn projection_dimension_mismatch_returns_error() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = projection_distance(&w1, &w2);
        assert_eq!(
            result,
            Err(GrassmannError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    // —— Test 27: relation d_proj = sqrt(2) * d_chordal ——
    #[test]
    fn projection_equals_sqrt2_times_chordal() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let d_proj = projection_distance(&w1, &w2).unwrap();
        let d_chord = chordal_distance(&w1, &w2).unwrap();

        let expected = 2.0_f64.sqrt() * d_chord;
        assert!(
            (d_proj - expected).abs() < 1e-6,
            "Expected d_proj = √2 · d_chord = {expected}, got {d_proj}"
        );
    }

    // —— Test 28: projection distance is non-negative ——
    #[test]
    fn projection_is_non_negative() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let d = projection_distance(&w1, &w2).unwrap();
        assert!(d >= 0.0, "Distance should be non-negative, got {d}");
    }
}
