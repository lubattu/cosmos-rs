//! Geometric operations on Lie groups.
//!
//! This module is part of the foundational geometry utilities in COSMOS,
//! supporting S (State), O (Objectives), and R (Rules) components.
//!
//! This module provides geodesic operations on Lie groups such as SO(k).
//! These are distinct from Grassmann manifold operations (see `grassmann` module).
//!
//! The core operation is geodesic interpolation on SO(k):
//!
//! ```text
//! G(θ) = G_A · exp(θ · log(G_Aᵀ · G_B))
//! ```
//!
//! The matrix logarithm is computed via inverse scaling & squaring
//! (Denman-Beavers square roots + Taylor series), and the matrix
//! exponential reuses `matrix_exp_antisym` from the `matrix_exp` module.

use ndarray::Array2;
#[cfg(not(feature = "lapack"))]
use ndarray::s;

use crate::geometry::matrix_exp::{
    MatrixExpError, OrthoMethod, antisymmetrize, matrix_exp_antisym,
};
use crate::geometry::qr::{QrConfig, qr_orthonormalize};

/// Errors for Lie group operations.
#[derive(Debug, Clone)]
pub enum LieGroupError {
    /// Matrices are not square (k × k).
    NotSquare {
        /// Number of rows the input matrix has.
        rows: usize,
        /// Number of columns the input matrix has.
        cols: usize,
    },
    /// G_A and G_B have different shapes.
    ShapeMismatch {
        /// Shape of `G_A`.
        shape_a: (usize, usize),
        /// Shape of `G_B`.
        shape_b: (usize, usize),
    },
    /// Matrix exponential failed.
    MatrixExp(MatrixExpError),
}

// ── Private helpers ──────────────────────────────────────

#[cfg(feature = "lapack")]
fn solve_matrix_system(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    use ndarray_linalg::Solve;
    let m = b.ncols();
    let mut result = Array2::<f64>::zeros(b.dim());
    for col in 0..m {
        let b_col = b.column(col).to_owned();
        match a.solve(&b_col) {
            Ok(x) => result.column_mut(col).assign(&x),
            Err(_) => return None,
        }
    }
    Some(result)
}

#[cfg(not(feature = "lapack"))]
fn solve_matrix_system(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let m = b.ncols();
    let mut aug = Array2::<f64>::zeros((n, n + m));
    aug.slice_mut(s![.., ..n]).assign(a);
    aug.slice_mut(s![.., n..]).assign(b);

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != col {
            for j in 0..(n + m) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(n + m) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut x = Array2::<f64>::zeros((n, m));
    for k in 0..m {
        for row in (0..n).rev() {
            let mut sum = aug[[row, n + k]];
            for j in (row + 1)..n {
                sum -= aug[[row, j]] * x[[j, k]];
            }
            x[[row, k]] = sum / aug[[row, row]];
        }
    }
    Some(x)
}

/// Matrix square root via Denman-Beavers iteration.
///
/// Uses the coupled iteration:
/// ```text
/// Y₀ = R,  Z₀ = I
/// Y_{k+1} = (Y_k + Z_k⁻¹) / 2
/// Z_{k+1} = (Z_k + Y_k⁻¹) / 2
/// ```
/// which converges quadratically to Y → R^{1/2}.
///
/// # Arguments
///
/// * `r` - A rotation matrix of shape (n, n). Must be non-singular.
///
/// # Returns
///
/// Matrix S of shape (n, n) such that S·S ≈ R.
fn matrix_sqrt_db(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let eye = Array2::<f64>::eye(n);

    let mut y = r.clone();
    let mut z = eye.clone();

    for _ in 0..32 {
        // z_inv = Z⁻¹,  y_inv = Y⁻¹
        let z_inv = solve_matrix_system(&z, &eye).unwrap_or_else(|| eye.clone());
        let y_inv = solve_matrix_system(&y, &eye).unwrap_or_else(|| eye.clone());

        let y_new = (&y + &z_inv) * 0.5;
        let z_new = (&z + &y_inv) * 0.5;

        // Converge check: ||Y_new - Y||_F < ε
        let diff = &y_new - &y;
        let norm_sq: f64 = diff.iter().map(|x| x * x).sum();

        y = y_new;
        z = z_new;

        if norm_sq < 1e-28 {
            // 1e-14 squared
            break;
        }
    }
    y
}

/// Matrix logarithm for R ∈ SO(k) via inverse scaling & squaring.
///
/// Algorithm:
/// 1. Take repeated square roots (Denman-Beavers) until ‖R_s - I‖_F < 0.5
/// 2. Compute log(I + X) via Taylor series (16 terms), where X = R_s - I
/// 3. Un-scale by multiplying by 2^s
/// 4. Force anti-symmetry via (A - Aᵀ)/2
///
/// # Arguments
///
/// * `r` - A rotation matrix in SO(k) of shape (k, k).
///
/// # Returns
///
/// Anti-symmetric matrix A ∈ so(k) such that exp(A) ≈ R.
fn matrix_log_rotation(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let eye = Array2::<f64>::eye(n);

    // Step 1: Repeated square roots until ||R_s - I|| < 0.5
    let mut r_s = r.clone();
    let mut s: u32 = 0;
    loop {
        let diff = &r_s - &eye;
        let norm: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 0.5 || s >= 50 {
            break;
        }
        r_s = matrix_sqrt_db(&r_s);
        s += 1;
    }

    // Step 2: X = R_s - I (small now)
    let x = &r_s - &eye;

    // Step 3: Taylor series  log(I + X) = X - X²/2 + X³/3 - ...
    let mut log_r = Array2::<f64>::zeros((n, n));
    let mut x_power = x.clone(); //X^1
    for k in 1..=16_usize {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        log_r = log_r + &x_power * (sign / k as f64);
        x_power = x_power.dot(&x); // X^(k+1)
    }

    // Step 4: Un-scale — multiply by 2^s
    log_r *= 2.0_f64.powi(s as i32);

    // Step 5: Force anti-symmetry (clean up numerical drift)
    antisymmetrize(&log_r)
}

#[cfg(feature = "lapack")]
fn determinant(a: &Array2<f64>) -> f64 {
    use ndarray_linalg::Determinant;
    a.det().unwrap_or(0.0)
}

#[cfg(not(feature = "lapack"))]
fn determinant(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut m = a.clone();
    let mut sign = 1.0_f64;

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[[col, col]].abs();
        for row in (col + 1)..n {
            let val = m[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return 0.0;
        }

        if max_row != col {
            for j in 0..n {
                let tmp = m[[col, j]];
                m[[col, j]] = m[[max_row, j]];
                m[[max_row, j]] = tmp;
            }
            sign = -sign;
        }

        let pivot = m[[col, col]];
        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for j in (col + 1)..n {
                m[[row, j]] -= factor * m[[col, j]];
            }
        }
    }

    let mut det = sign;
    for i in 0..n {
        det *= m[[i, i]];
    }
    det
}

// ── Public API ───────────────────────────────────────────

/// Geodesically interpolate between two SO(k) matrices.
///
/// Computes the SO(k) geodesic interpolation:
/// ```text
/// G(θ) = G_A · exp(θ · log(G_Aᵀ · G_B))
/// ```
/// then re-orthonormalizes via QR and enforces det = +1 (SO(k), not O(k)).
///
/// # Arguments
///
/// * `g_a` - Start rotation matrix of shape (k, k), must be in SO(k).
/// * `g_b` - End rotation matrix of shape (k, k), must be in SO(k).
/// * `theta` - Interpolation parameter. At θ=0 returns `g_a`, at θ=1 returns `g_b`.
///
/// # Returns
///
/// Interpolated rotation matrix G(θ) ∈ SO(k) of shape (k, k).
///
/// # Errors
///
/// * [`LieGroupError::NotSquare`] — if `g_a` is not square.
/// * [`LieGroupError::ShapeMismatch`] — if `g_a` and `g_b` have different shapes.
/// * [`LieGroupError::MatrixExp`] — if the internal matrix exponential fails.
pub fn so_geodesic_interpolation(
    g_a: &Array2<f64>,
    g_b: &Array2<f64>,
    theta: f64,
) -> Result<Array2<f64>, LieGroupError> {
    let (ra, ca) = (g_a.nrows(), g_a.ncols());
    let (rb, cb) = (g_b.nrows(), g_b.ncols());

    // Validation
    if ra != ca {
        return Err(LieGroupError::NotSquare { rows: ra, cols: ca });
    }
    if (ra, ca) != (rb, cb) {
        return Err(LieGroupError::ShapeMismatch {
            shape_a: (ra, ca),
            shape_b: (rb, cb),
        });
    }

    // Early returns
    if theta == 0.0 {
        return Ok(g_a.clone());
    }
    if theta == 1.0 {
        return Ok(g_b.clone());
    }

    let n = ra;

    // 1. Relative rotation: R = G_A^T · G_B
    let r = g_a.t().dot(g_b);

    // 2. Matrix logarithm: A = log(R),  A ∈ so(k)
    let a = matrix_log_rotation(&r);

    // 3. Scaled exponential: R_θ = exp(θ · A)
    let r_theta =
        matrix_exp_antisym(&(&a * theta), OrthoMethod::Taylor).map_err(LieGroupError::MatrixExp)?;

    // 4. Map forward: G_θ = G_A · R_θ
    let g_theta = g_a.dot(&r_theta);

    // 5. Re-orthonormalize via QR
    let mut g_theta = qr_orthonormalize(&g_theta, &QrConfig::default());

    // 6. Enforce det = +1 (SO(k), not O(k))
    let det = determinant(&g_theta);
    if det < 0.0 {
        // Flip sign of first column
        for i in 0..n {
            g_theta[[i, 0]] = -g_theta[[i, 0]];
        }
    }

    Ok(g_theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ── Test helpers ─────────────────────────────────────

    /// Build a 2D rotation matrix for angle α (radians).
    fn rot2(alpha: f64) -> Array2<f64> {
        let (s, c) = alpha.sin_cos();
        Array2::from_shape_vec((2, 2), vec![c, -s, s, c]).unwrap()
    }

    /// Build a 3D rotation about the Z axis by angle α.
    fn rot3_z(alpha: f64) -> Array2<f64> {
        let (s, c) = alpha.sin_cos();
        Array2::from_shape_vec((3, 3), vec![c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]).unwrap()
    }

    /// Build a 3D rotation about the X axis by angle α.
    fn rot3_x(alpha: f64) -> Array2<f64> {
        let (s, c) = alpha.sin_cos();
        Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]).unwrap()
    }

    /// Assert that U is orthogonal: ‖UᵀU - I‖_F < tol.
    fn assert_orthogonal(u: &Array2<f64>, tol: f64, label: &str) {
        let n = u.nrows();
        let eye = Array2::<f64>::eye(n);
        let err = &u.t().dot(u) - &eye;
        let norm: f64 = err.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < tol,
            "{label}: orthogonality error {norm:.2e} >= {tol:.2e}"
        );
    }

    /// Assert that det(U) ≈ +1 (i.e. U ∈ SO(k)).
    fn assert_so(u: &Array2<f64>, tol: f64, label: &str) {
        assert_orthogonal(u, tol, label);
        let det = determinant(u);
        assert!(
            (det - 1.0).abs() < tol,
            "{label}: det = {det:.6}, expected +1"
        );
    }

    /// Frobenius distance ‖A - B‖_F.
    fn fro_dist(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let diff = a - b;
        diff.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    // ── solve_matrix_system ──────────────────────────────

    #[test]
    fn solve_identity_system() {
        let eye = Array2::<f64>::eye(3);
        let b = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let x = solve_matrix_system(&eye, &b).unwrap();
        assert!(fro_dist(&x, &b) < 1e-12);
    }

    #[test]
    fn solve_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 5.0, 3.0]).unwrap();
        let eye = Array2::<f64>::eye(2);
        let a_inv = solve_matrix_system(&a, &eye).unwrap();
        let product = a.dot(&a_inv);
        assert!(fro_dist(&product, &eye) < 1e-12);
    }

    #[test]
    fn solve_singular_returns_none() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let eye = Array2::<f64>::eye(2);
        assert!(solve_matrix_system(&a, &eye).is_none());
    }

    // ── matrix_sqrt_db ───────────────────────────────────

    #[test]
    fn sqrt_of_identity_is_identity() {
        let eye = Array2::<f64>::eye(4);
        let s = matrix_sqrt_db(&eye);
        assert!(fro_dist(&s, &eye) < 1e-12);
    }

    #[test]
    fn sqrt_squared_recovers_original() {
        let r = rot3_z(0.8);
        let s = matrix_sqrt_db(&r);
        let s_sq = s.dot(&s);
        assert!(
            fro_dist(&s_sq, &r) < 1e-10,
            "S² ≠ R, dist = {}",
            fro_dist(&s_sq, &r)
        );
    }

    #[test]
    fn sqrt_of_rotation_is_half_angle() {
        // rot2(0.6) → sqrt should be ≈ rot2(0.3)
        let r = rot2(0.6);
        let s = matrix_sqrt_db(&r);
        let expected = rot2(0.3);
        assert!(fro_dist(&s, &expected) < 1e-10);
    }

    // ── matrix_log_rotation ──────────────────────────────

    #[test]
    fn log_of_identity_is_zero() {
        let eye = Array2::<f64>::eye(3);
        let a = matrix_log_rotation(&eye);
        let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1e-12, "log(I) should be zero, got ‖A‖ = {norm:.2e}");
    }

    #[test]
    fn log_result_is_antisymmetric() {
        let r = rot3_z(0.7).dot(&rot3_x(0.4));
        let a = matrix_log_rotation(&r);
        let sym_err = &a + &a.t();
        let norm: f64 = sym_err.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 1e-10,
            "log(R) not anti-symmetric, ‖A + Aᵀ‖ = {norm:.2e}"
        );
    }

    #[test]
    fn exp_of_log_recovers_rotation() {
        let r = rot3_z(0.9);
        let a = matrix_log_rotation(&r);
        let r_recovered = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        assert!(
            fro_dist(&r_recovered, &r) < 1e-8,
            "exp(log(R)) ≠ R, dist = {}",
            fro_dist(&r_recovered, &r)
        );
    }

    #[test]
    fn log_2x2_matches_known_generator() {
        // log(rot2(α)) should be [[0, -α], [α, 0]]
        let alpha = 0.5;
        let r = rot2(alpha);
        let a = matrix_log_rotation(&r);
        assert!((a[[0, 1]] - (-alpha)).abs() < 1e-10, "a[0,1] should be -α");
        assert!((a[[1, 0]] - alpha).abs() < 1e-10, "a[1,0] should be α");
        assert!(a[[0, 0]].abs() < 1e-12);
        assert!(a[[1, 1]].abs() < 1e-12);
    }

    #[test]
    fn log_large_angle_still_works() {
        // Almost π rotation (large angle, stresses the scaling)
        let r = rot2(2.8);
        let a = matrix_log_rotation(&r);
        let r_recovered = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        assert!(
            fro_dist(&r_recovered, &r) < 1e-6,
            "exp(log(R)) ≠ R for large angle, dist = {}",
            fro_dist(&r_recovered, &r)
        );
    }

    // ── determinant ──────────────────────────────────────

    #[test]
    fn det_of_identity_is_one() {
        let eye = Array2::<f64>::eye(4);
        assert!((determinant(&eye) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn det_of_rotation_is_one() {
        let r = rot3_z(1.2).dot(&rot3_x(0.7));
        assert!(
            (determinant(&r) - 1.0).abs() < 1e-10,
            "det(rotation) should be 1"
        );
    }

    #[test]
    fn det_known_2x2() {
        // det([[a, b], [c, d]]) = ad - bc
        let a = Array2::from_shape_vec((2, 2), vec![3.0, 8.0, 4.0, 6.0]).unwrap();
        let expected = 3.0 * 6.0 - 8.0 * 4.0; // -14
        assert!((determinant(&a) - expected).abs() < 1e-10);
    }

    #[test]
    fn det_of_singular_is_zero() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(determinant(&a).abs() < 1e-12);
    }

    #[test]
    fn det_sign_flip_on_column_negate() {
        // Negating one column flips the sign of det
        let r = rot3_z(0.5);
        let det_r = determinant(&r);
        let mut r_flipped = r.clone();
        for i in 0..3 {
            r_flipped[[i, 0]] = -r_flipped[[i, 0]];
        }
        let det_flipped = determinant(&r_flipped);
        assert!(
            (det_flipped + det_r).abs() < 1e-10,
            "column negate should flip det sign"
        );
    }

    // ── so_geodesic_interpolation ────────────────────────

    #[test]
    fn interp_theta_0_returns_g_a() {
        let g_a = rot3_z(0.3).dot(&rot3_x(0.5));
        let g_b = rot3_z(1.1).dot(&rot3_x(0.2));
        let result = so_geodesic_interpolation(&g_a, &g_b, 0.0).unwrap();
        assert!(fro_dist(&result, &g_a) < 1e-14);
    }

    #[test]
    fn interp_theta_1_returns_g_b() {
        let g_a = rot3_z(0.3).dot(&rot3_x(0.5));
        let g_b = rot3_z(1.1).dot(&rot3_x(0.2));
        let result = so_geodesic_interpolation(&g_a, &g_b, 1.0).unwrap();
        assert!(fro_dist(&result, &g_b) < 1e-14);
    }

    #[test]
    fn interp_result_is_in_so_k() {
        let g_a = rot3_z(0.4);
        let g_b = rot3_z(1.2).dot(&rot3_x(0.6));
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let g_t = so_geodesic_interpolation(&g_a, &g_b, t).unwrap();
            assert_so(&g_t, 1e-8, &format!("G(θ={t})"));
        }
    }

    #[test]
    fn interp_2x2_midpoint_is_half_angle() {
        // G_A = I, G_B = rot2(α) → G(0.5) ≈ rot2(α/2)
        let alpha = 0.8;
        let g_a = Array2::<f64>::eye(2);
        let g_b = rot2(alpha);
        let g_mid = so_geodesic_interpolation(&g_a, &g_b, 0.5).unwrap();
        let expected = rot2(alpha / 2.0);
        assert!(
            fro_dist(&g_mid, &expected) < 1e-6,
            "midpoint should be half-angle rotation, dist = {}",
            fro_dist(&g_mid, &expected)
        );
    }

    #[test]
    fn interp_identical_returns_same() {
        let g = rot3_z(0.7).dot(&rot3_x(0.3));
        let result = so_geodesic_interpolation(&g, &g, 0.5).unwrap();
        assert!(
            fro_dist(&result, &g) < 1e-8,
            "interpolation of identical matrices failed"
        );
    }

    #[test]
    fn interp_from_identity() {
        // G_A = I, G_B = R → G(θ) traces the geodesic from I to R
        let g_a = Array2::<f64>::eye(3);
        let g_b = rot3_z(1.0);
        let g_t = so_geodesic_interpolation(&g_a, &g_b, 0.3).unwrap();
        assert_so(&g_t, 1e-8, "G(0.3) from identity");
    }

    #[test]
    fn interp_monotonic_distance() {
        // As θ ↑ from 0 to 1, distance from G_A should increase monotonically
        let g_a = Array2::<f64>::eye(3);
        let g_b = rot3_z(1.0).dot(&rot3_x(0.5));
        let thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut prev_dist = 0.0_f64;
        for &t in &thetas {
            let g_t = so_geodesic_interpolation(&g_a, &g_b, t).unwrap();
            let dist = fro_dist(&g_t, &g_a);
            assert!(
                dist >= prev_dist - 1e-8,
                "distance should be monotonic: at θ={t}, dist={dist:.6} < prev={prev_dist:.6}"
            );
            prev_dist = dist;
        }
    }

    #[test]
    fn interp_5x5_produces_so() {
        // Test with a larger matrix — build G_A, G_B from small anti-symmetric generators
        let n = 5;
        let mut a_gen = Array2::<f64>::zeros((n, n));
        a_gen[[0, 1]] = 0.3;
        a_gen[[1, 0]] = -0.3;
        a_gen[[2, 3]] = 0.5;
        a_gen[[3, 2]] = -0.5;
        let g_a = matrix_exp_antisym(&a_gen, OrthoMethod::Taylor).unwrap();

        let mut b_gen = Array2::<f64>::zeros((n, n));
        b_gen[[0, 2]] = 0.7;
        b_gen[[2, 0]] = -0.7;
        b_gen[[1, 4]] = 0.4;
        b_gen[[4, 1]] = -0.4;
        let g_b = matrix_exp_antisym(&b_gen, OrthoMethod::Taylor).unwrap();

        let g_mid = so_geodesic_interpolation(&g_a, &g_b, 0.5).unwrap();
        assert_so(&g_mid, 1e-8, "5×5 midpoint");
    }

    #[test]
    fn interp_midpoint_is_equidistant() {
        let g_a = rot3_z(0.3);
        let g_b = rot3_z(1.1).dot(&rot3_x(0.4));
        let g_mid = so_geodesic_interpolation(&g_a, &g_b, 0.5).unwrap();
        let d_start = fro_dist(&g_mid, &g_a);
        let d_end = fro_dist(&g_mid, &g_b);
        assert!(
            (d_start - d_end).abs() < 0.15,
            "midpoint roughly equidistant: d_start={d_start:.4}, d_end={d_end:.4}"
        );
    }

    // ── Error handling ───────────────────────────────────

    #[test]
    fn interp_non_square_returns_error() {
        let g_a = Array2::<f64>::zeros((3, 4));
        let g_b = Array2::<f64>::zeros((3, 4));
        let result = so_geodesic_interpolation(&g_a, &g_b, 0.5);
        assert!(result.is_err());
        match result.unwrap_err() {
            LieGroupError::NotSquare { rows: 3, cols: 4 } => {}
            e => panic!("expected NotSquare, got {e:?}"),
        }
    }

    #[test]
    fn interp_shape_mismatch_returns_error() {
        let g_a = Array2::<f64>::eye(3);
        let g_b = Array2::<f64>::eye(4);
        let result = so_geodesic_interpolation(&g_a, &g_b, 0.5);
        assert!(result.is_err());
        match result.unwrap_err() {
            LieGroupError::ShapeMismatch { .. } => {}
            e => panic!("expected ShapeMismatch, got {e:?}"),
        }
    }
}
