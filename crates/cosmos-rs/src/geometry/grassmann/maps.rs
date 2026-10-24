//! Exponential map, logarithmic map, and parallel transport on Gr(k, n).
//!
//! These are the core Riemannian operations for moving on the manifold:
//!   - [`exponential_map`] — move along geodesic in tangent direction
//!   - [`logarithmic_map`] — compute tangent vector connecting two points
//!   - [`parallel_transport`] — transport tangent vectors along geodesics
//!
use super::{GrassmannError, retract_polar_gram};
use crate::geometry::svd::{SvdConfig, SvdResult, svd};
use ndarray::{Array1, Array2};

/// Re-orthonormalise a `k×k` matrix into a proper orthogonal basis.
///
/// Used inside [`logarithmic_map`] as an MGS re-orthonormalization
/// safeguard on the right singular vectors of the Gram matrix `W1^T W2`.
/// It is a no-op whenever the SVD already returns an orthonormal `V`; it
/// only does work when one or more principal angles equal `π/2` and `V`'s
/// corresponding columns are degenerate, which would otherwise leave the
/// downstream projection `M @ V` with spurious zero or near-collinear
/// columns and collapse the recovered `sin θ_i` (and hence `θ_i`).
///
/// # Algorithm
///
/// Sweep columns left to right. For each column, subtract its
/// projections onto the already-accepted columns (Modified
/// Gram-Schmidt). If the residual has positive norm above `eps`,
/// normalise and accept it. If the residual collapses below `eps`
/// (the column was a duplicate or near-duplicate of earlier ones),
/// fall back to the first canonical basis vector `e_i` that, after
/// Gram-Schmidt against the accepted columns, has positive norm.
///
/// This always succeeds for a `k×k` input because at most `k - 1`
/// accepted columns block at most `k - 1` canonical directions.
fn complete_orthogonal_columns(mat: Array2<f64>) -> Array2<f64> {
    let k = mat.ncols();
    assert_eq!(
        mat.nrows(),
        k,
        "complete_orthogonal_columns: matrix must be square"
    );
    let eps = 1e-10_f64;

    let mut out = Array2::<f64>::zeros((k, k));
    let mut accepted: Vec<Array1<f64>> = Vec::with_capacity(k);

    for j in 0..k {
        // Modified Gram-Schmidt: start from the input column and remove
        // its projections onto previously-accepted directions.
        let mut candidate: Array1<f64> = mat.column(j).to_owned();
        for v in &accepted {
            let dot: f64 = candidate.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            for (idx, &vv) in v.iter().enumerate() {
                candidate[idx] -= dot * vv;
            }
        }
        let norm = candidate.iter().map(|&x| x * x).sum::<f64>().sqrt();

        let final_col: Array1<f64> = if norm > eps {
            let inv = 1.0 / norm;
            candidate.iter().map(|&x| x * inv).collect()
        } else {
            // Column collapsed under Gram-Schmidt (zero or duplicate of
            // earlier columns). Replace it with the first canonical
            // direction that survives orthogonalisation.
            let mut chosen: Option<Array1<f64>> = None;
            for i in 0..k {
                let mut e: Array1<f64> = Array1::zeros(k);
                e[i] = 1.0;
                for v in &accepted {
                    let dot: f64 = e.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                    for (idx, &vv) in v.iter().enumerate() {
                        e[idx] -= dot * vv;
                    }
                }
                let n = e.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if n > eps {
                    let inv = 1.0 / n;
                    chosen = Some(e.iter().map(|&x| x * inv).collect());
                    break;
                }
            }
            chosen.expect(
                "complete_orthogonal_columns: no canonical complement available — impossible for a k×k matrix",
            )
        };

        out.column_mut(j).assign(&final_col);
        accepted.push(final_col);
    }

    out
}

/// Configuration for the Grassmann exponential map.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialMapConfig {
    /// Distance parameter scaling the tangent direction along the geodesic.
    pub t: f64,
}

impl Default for ExponentialMapConfig {
    fn default() -> Self {
        Self { t: 1.0 }
    }
}

/// Exponential map: move along geodesic in tangent direction.
///
/// Given a point W ∈ Gr(k, n) and a tangent vector V ∈ T_W Gr(k, n),
/// compute the point reached by traveling distance t along the geodesic
/// in direction V.
///
/// Uses the canonical formulation from Edelman, Arias, Smith (1998).
///
/// This is the Riemannian exponential map and generates geodesics on
/// the manifold. For t=1, it gives the endpoint of the geodesic starting
/// at W with initial velocity V.
///
/// # Mathematical Foundation:
/// - Exp_W: T_W Gr(k,n) → Gr(k,n) is the inverse of Log_W
/// - Exp_W(0) = W (identity)
/// - Exp_W(tV) traces a geodesic for t ∈ R
/// - ||V||_F = speed of geodesic
///
/// # Arguments
///
/// * `W` - Base point on manifold, shape (n, k) with orthonormal columns.
/// * `tangent_vec` - Tangent vector at W, shape (n, k).
/// * `config` - Configuration with distance parameter.
///
/// # Returns
///
/// New point on Gr(k, n), shape (n, k) with orthonormal columns.
///
/// # Note
/// The tangent vector V should satisfy W^T V = 0. If not, it will be
/// automatically projected to the tangent space.
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if `w` and `tangent_vec`
/// have different shapes.
pub fn exponential_map(
    w: &Array2<f64>,
    tangent_vec: &Array2<f64>,
    config: &ExponentialMapConfig,
) -> Result<Array2<f64>, GrassmannError> {
    // Validate dimensions
    if w.dim() != tangent_vec.dim() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w.nrows(),
            got: tangent_vec.nrows(),
        });
    }

    // Project V onto tangent space: V_proj = V - W(W^T V)
    let v_proj = tangent_vec - &w.dot(&w.t().dot(tangent_vec));

    // Check for zero velocity
    let v_norm = v_proj.mapv(|x| x.powi(2)).sum().sqrt();
    if v_norm < 1e-10 {
        return Ok(w.clone());
    }

    // Scale by t
    let v_scaled = config.t * v_proj;

    // SVD of V: V = U S Vt
    let svd_res: SvdResult = svd(&v_scaled, &SvdConfig::default());

    let cos_s = svd_res.s.mapv(f64::cos);
    let sin_s = svd_res.s.mapv(f64::sin);

    let k = cos_s.len();

    // Build diagonal matrices
    let diag_cos = Array2::from_shape_fn((k, k), |(i, j)| if i == j { cos_s[i] } else { 0.0 });
    let diag_sin = Array2::from_shape_fn((k, k), |(i, j)| if i == j { sin_s[i] } else { 0.0 });

    // term1 = W @ Vt^T @ diag(cos_S) @ Vt
    let term1 = w.dot(&svd_res.vt.t().dot(&diag_cos).dot(&svd_res.vt));

    // term2 = U @ diag(sin_S) @ Vt
    let term2 = svd_res.u.dot(&diag_sin.dot(&svd_res.vt));

    let w_new = term1 + term2;

    // Re-orthonormalize via a compact polar factor on the k x k Gram matrix.
    Ok(retract_polar_gram(&w_new))
}

/// Logarithmic map: compute tangent vector connecting two points.
///
/// Given two points W1, W2 ∈ Gr(k, n), compute the tangent vector V at W1
/// such that Exp_W1(V) = W2. This V points along the geodesic from W1 to W2.
///
/// Uses the canonical formulation from Edelman, Arias, Smith (1998).
///
/// # Mathematical Properties
/// - Log_W is inverse of Exp_W
/// - Log_W(W) = 0 (identity)
/// - ||Log_W1(W2)||_F = geodesic_distance(W1, W2)
/// - Log_W1(W2) ∈ T_W1 Gr(k,n) (tangent space)
///
/// # Arguments
///
/// * `w1` - Starting point, shape (n, k) with orthonormal columns.
/// * `w2` - Target point, shape (n, k) with orthonormal columns.
///
/// # Returns
///
/// Tangent vector at W1 pointing toward W2, shape (n, k).
///
/// # Note
/// For identical subspaces, returns zero vector.
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if `w1` and `w2` have
/// different shapes.
pub fn logarithmic_map(w1: &Array2<f64>, w2: &Array2<f64>) -> Result<Array2<f64>, GrassmannError> {
    // Validate dimensions
    if w1.dim() != w2.dim() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w1.nrows(),
            got: w2.nrows(),
        });
    }

    let (n, k) = w1.dim();

    // Gram matrix: gram = W1^T @ W2, shape (k, k)
    let gram = w1.t().dot(w2);

    // Orthogonal component: M = (I - W1 W1^T) W2 = W2 - W1 @ gram
    let m = w2 - &w1.dot(&gram);

    // Check if same subspace (M ≈ 0 means W1 ≈ W2)
    let m_norm = m.mapv(|x| x.powi(2)).sum().sqrt();
    if m_norm < 1e-10 {
        return Ok(Array2::zeros(w1.dim()));
    }

    // Full SVD of gram: gram = U_g @ diag(cos_θ) @ Vt_g
    // This gives us the principal directions AND cosines of principal angles
    let svd_gram = svd(&gram, &SvdConfig::default());

    // Repair `Vt_g` so its rows are a proper k×k orthonormal basis even
    // when some principal angles equal π/2, where the SVD's `Vt_g` can
    // return degenerate rows (equivalently, degenerate columns in `V_g`)
    // for those directions; without completion, `M @ Vt_g^T` produces a
    // spurious zero column and the recovered θ collapses to zero instead
    // of π/2. This is a no-op whenever the SVD already returns an
    // orthonormal `V_g`.
    let v_g_completed = complete_orthogonal_columns(svd_gram.vt.t().to_owned());
    let vt_g = v_g_completed.t().to_owned();

    // Project M into gram's right singular vector basis.
    // Since M = U_perp @ diag(sin_θ) @ Vt_g (same Vt_g as gram!),
    // then M @ Vt_g^T = U_perp @ diag(sin_θ).
    // Column i has magnitude sin_θ_i and direction u_perp_i.
    let m_in_basis = m.dot(&vt_g.t());

    // Extract θ_i and U_perp from the projected columns
    let mut theta = Array1::<f64>::zeros(k);
    let mut u_perp = Array2::<f64>::zeros((n, k));

    for i in 0..k {
        // Column i of m_in_basis = u_perp_i * sin_θ_i
        let col = m_in_basis.column(i);
        let sin_i = col.dot(&col).sqrt();
        let cos_i = svd_gram.s[i].clamp(-1.0, 1.0);

        // atan2 is more stable than acos or asin alone
        theta[i] = sin_i.atan2(cos_i);

        // Normalize to get u_perp_i
        if sin_i > 1e-10 {
            for row in 0..n {
                u_perp[[row, i]] = col[row] / sin_i;
            }
        }
        // If sin_i ≈ 0, θ_i ≈ 0 too, so u_perp stays zero (multiplied by θ=0 anyway)
    }

    // Build diagonal matrix of principal angles
    let diag_theta = Array2::from_shape_fn((k, k), |(i, j)| if i == j { theta[i] } else { 0.0 });

    // Log_W1(W2) = U_perp @ diag(θ) @ Vt_g — using the completed Vt_g so
    // that directions with cos θ = 0 are rotated by a valid orthonormal
    // basis instead of being zeroed out.
    Ok(u_perp.dot(&diag_theta).dot(&vt_g))
}

/// Parallel transport a tangent vector along geodesic.
///
/// Given a tangent vector V at W_start, transport it along the geodesic
/// to W_end, preserving its length and the angle it makes with the geodesic.
///
/// This is fundamental for comparing gradients or tangent vectors at
/// different points on the manifold.
///
/// # Mathematical Foundation
/// - Parallel transport: Γ_{γ}: T_{W_start}M → T_{W_end}M preserves
///   - Inner product: ⟨Γ(V), Γ(U)⟩ = ⟨V, U⟩
///   - Length: ||Γ(V)|| = ||V||
/// - For Gr(k, n), we use the canonical connection.
///
/// # Arguments
///
/// * `w_start` - Starting point on Gr(k, n), shape (n, k).
/// * `w_end` - Ending point on Gr(k, n), shape (n, k).
/// * `t_start` - Tangent vector at W_start, shape (n, k).
///   Should satisfy W_start^T @ tangent_at_start ≈ 0.
///
/// # Returns
///
/// Transported tangent vector at W_end, shape (n, k).
///
/// # Note
/// For identical points, returns the tangent vector unchanged.
///
/// # Errors
///
/// Returns [`GrassmannError::DimensionMismatch`] if `w_start`, `w_end` and
/// `t_start` do not all share the same shape.
pub fn parallel_transport(
    w_start: &Array2<f64>,
    w_end: &Array2<f64>,
    t_start: &Array2<f64>,
) -> Result<Array2<f64>, GrassmannError> {
    // Validate dimensions: all three must have the same shape
    if w_start.dim() != w_end.dim() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w_start.nrows(),
            got: w_end.nrows(),
        });
    }

    if w_start.dim() != t_start.dim() {
        return Err(GrassmannError::DimensionMismatch {
            expected: w_start.nrows(),
            got: t_start.nrows(),
        });
    }

    // Check if same subspace
    let diff_norm = (w_start - w_end).mapv(|diff| diff.powi(2)).sum().sqrt();
    if diff_norm < 1e-10 {
        return Ok(t_start.clone());
    }

    // Compute geodesic direction via log map
    // Δ = Log_{W_start}(W_end), the initial velocity of the geodesic
    let log_vec = logarithmic_map(w_start, w_end)?;

    // SVD of log vector: Δ = U @ diag(S) @ Vt
    // U spans the "perpendicular directions" of the geodesic
    // S contains the principal angles
    // Vt rotates within the k-dimensional subspace
    let svd_log = svd(&log_vec, &SvdConfig::default());

    let k = svd_log.s.len();

    let diag_cos = Array2::from_shape_fn(
        (k, k),
        |(i, j)| {
            if i == j { svd_log.s[i].cos() } else { 0.0 }
        },
    );

    let diag_sin = Array2::from_shape_fn(
        (k, k),
        |(i, j)| {
            if i == j { svd_log.s[i].sin() } else { 0.0 }
        },
    );

    // Project tangent vector onto geodesic plane: U^T @ V
    let ut_v = svd_log.u.t().dot(t_start);

    // Perpendicular component: V_perp = V - U @ U^T (unchanged by transport)
    let v_perp = t_start - &svd_log.u.dot(&ut_v);

    // Parallel transport formula (Edelman, Arias, Smith 1998):
    //
    // Γ(V) = [-W_start @ Vt^T @ diag(sin(S)) + U @ diag(cos(S))] @ U^T @ V + V_perp
    //
    // The sin term rotates the W_start component toward W_end.
    // The cos term scales the U component.
    // Together they perform a rotation in each geodesic plane.

    // term_sin = W_start @ Vt^T @ diag(sin(S)) @ U^T @ V
    let term_sin = w_start.dot(&svd_log.vt.t().dot(&diag_sin).dot(&ut_v));

    // term_cos = U @ diag(cos(S)) @ U^T @ V
    let term_cos = svd_log.u.dot(&diag_cos.dot(&ut_v));

    // Γ(V) = V_perp - term_sin + term_cos
    Ok(v_perp - &term_sin + &term_cos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::grassmann::distances::{geodesic_distance, principal_angles};
    use ndarray::array;

    /// Frobenius norm helper.
    fn frob_norm(a: &Array2<f64>) -> f64 {
        a.mapv(|x| x.powi(2)).sum().sqrt()
    }

    // ========================================================
    // exponential_map tests
    // ========================================================

    // —— Test 1: Zero tangent vector → same point ——
    #[test]
    fn exp_zero_tangent_returns_same_point() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let zero = Array2::zeros(w.dim());
        let w2 = exponential_map(&w, &zero, &ExponentialMapConfig::default()).unwrap();

        let d = geodesic_distance(&w, &w2).unwrap();
        assert!(d < 1e-8, "Expected same point, got geodesic distance {d}");
    }

    // —— Test 2: Output has orthonormal columns ——
    #[test]
    fn exp_output_is_orthonormal() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let v = array![[0.0, 0.0], [0.0, 0.0], [0.3, 0.0], [0.0, 0.3]];

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();

        // W2^T W2 should be ≈ identity
        let wtw = w2.t().dot(&w2);
        let k = wtw.nrows();
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (wtw[[i, j]] - expected).abs() < 1e-6,
                    "W2^T W2[{i},{j}] = {}, expected {expected}",
                    wtw[[i, j]]
                );
            }
        }
    }

    #[test]
    fn exp_output_is_orthonormal_for_tall_skinny_block() {
        let n = 128;
        let k = 16;
        let w = Array2::from_shape_fn((n, k), |(row, col)| if row == col { 1.0 } else { 0.0 });
        let v = Array2::from_shape_fn((n, k), |(row, col)| {
            if row < k {
                0.0
            } else {
                (((row * 29 + col * 17 + 3) % 97) as f64 / 97.0 - 0.5) * 0.05
            }
        });

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();
        let wtw = w2.t().dot(&w2);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (wtw[[i, j]] - expected).abs() < 1e-5,
                    "W2^T W2[{i},{j}] = {}, expected {expected}",
                    wtw[[i, j]]
                );
            }
        }
    }

    // —— Test 3: Geodesic distance matches tangent norm ——
    #[test]
    fn exp_geodesic_distance_matches_tangent_norm() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        // Tangent orthogonal to W
        let v = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.15]];

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();

        let d = geodesic_distance(&w, &w2).unwrap();
        let v_norm = frob_norm(&v);
        assert!(
            (d - v_norm).abs() < 1e-6,
            "Expected geodesic dist ≈ ||V|| = {v_norm}, got {d}"
        );
    }

    // —— Test 4: t parameter scales the step ——
    #[test]
    fn exp_t_scales_the_step() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let v = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.2]];

        let half = ExponentialMapConfig { t: 0.5 };
        let full = ExponentialMapConfig { t: 1.0 };

        let w_half = exponential_map(&w, &v, &half).unwrap();
        let w_full = exponential_map(&w, &v, &full).unwrap();

        let d_half = geodesic_distance(&w, &w_half).unwrap();
        let d_full = geodesic_distance(&w, &w_full).unwrap();

        assert!(
            (d_full - 2.0 * d_half).abs() < 1e-6,
            "Expected d_full ≈ 2 * d_half: {} vs 2*{}",
            d_full,
            d_half
        );
    }

    // —— Test 5: Tangent not orthogonal to W is projected automatically ——
    #[test]
    fn exp_projects_non_orthogonal_tangent() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        // This has a component along W (non-zero in first two rows)
        let v = array![[0.5, 0.0], [0.0, 0.5], [0.3, 0.0]];

        // Should not error, just project and proceed
        let result = exponential_map(&w, &v, &ExponentialMapConfig::default());
        assert!(result.is_ok(), "Should handle non-orthogonal tangent");
    }

    // —— Test 6: Dimension mismatch → Err ——
    #[test]
    fn exp_dimension_mismatch_returns_error() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let v_bad = array![[0.0, 0.0], [0.0, 0.0]]; // 2×2 vs 3×2

        let result = exponential_map(&w, &v_bad, &ExponentialMapConfig::default());
        assert!(result.is_err());
    }

    // —— Test 7: 1D case — exp along single direction ——
    #[test]
    fn exp_1d_subspace() {
        let w = array![[1.0], [0.0], [0.0]]; // span{e1}

        let v = array![[0.0], [0.3], [0.0]]; // tangent toward e2

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();

        // Principal angle should be 0.3 radians
        let angles = principal_angles(&w, &w2).unwrap();
        assert!(
            (angles[0] - 0.3).abs() < 1e-6,
            "Expected angle ≈ 0.3, got {}",
            angles[0]
        );
    }

    // ========================================================
    // logarithmic_map tests
    // ========================================================

    // —— Test 8: Identical points → zero tangent ——
    #[test]
    fn log_identical_points_give_zero() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let v = logarithmic_map(&w, &w).unwrap();
        assert!(
            frob_norm(&v) < 1e-10,
            "Expected zero tangent, got norm {}",
            frob_norm(&v)
        );
    }

    // —— Test 9: ||Log(W1, W2)||_F = geodesic_distance(W1, W2) ——
    #[test]
    fn log_norm_equals_geodesic_distance() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let v = logarithmic_map(&w1, &w2).unwrap();
        let d = geodesic_distance(&w1, &w2).unwrap();

        let v_norm = frob_norm(&v);
        assert!(
            (v_norm - d).abs() < 1e-6,
            "Expected ||Log|| = d_geo: {} vs {}",
            v_norm,
            d
        );
    }

    // —— Test 10: Log tangent is in tangent space — W^T V ≈ 0 ——
    #[test]
    fn log_result_is_in_tangent_space() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let v = logarithmic_map(&w1, &w2).unwrap();

        // W1^T @ V should be ≈ 0
        let wtv = w1.t().dot(&v);
        let wtv_norm = frob_norm(&wtv);
        assert!(
            wtv_norm < 1e-6,
            "Expected W^T V ≈ 0, got ||W^T V|| = {wtv_norm}"
        );
    }

    // —— Test 11: Symmetry — ||Log(W1, W2)|| = ||Log(W2, W1)|| ——
    #[test]
    fn log_norm_is_symmetric() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let s = (0.5_f64).sqrt();
        let w2 = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];

        let v_ab = logarithmic_map(&w1, &w2).unwrap();
        let v_ba = logarithmic_map(&w2, &w1).unwrap();

        assert!(
            (frob_norm(&v_ab) - frob_norm(&v_ba)).abs() < 1e-6,
            "Expected symmetric norms: {} vs {}",
            frob_norm(&v_ab),
            frob_norm(&v_ba)
        );
    }

    // —— Test 12: Dimension mismatch → Err ——
    #[test]
    fn log_dimension_mismatch_returns_error() {
        let w1 = array![[1.0], [0.0], [0.0]];
        let w2 = array![[1.0], [0.0]];

        let result = logarithmic_map(&w1, &w2);
        assert!(result.is_err());
    }

    // —— Test 13: 1D orthogonal subspaces — tangent norm = π/2 ——
    #[test]
    fn log_orthogonal_1d_gives_pi_over_2() {
        use std::f64::consts::FRAC_PI_2;

        let w1 = array![[1.0], [0.0], [0.0]]; // span{e1}
        let w2 = array![[0.0], [1.0], [0.0]]; // span{e2}

        let v = logarithmic_map(&w1, &w2).unwrap();
        let v_norm = frob_norm(&v);

        assert!(
            (v_norm - FRAC_PI_2).abs() < 1e-6,
            "Expected ||Log|| = π/2 ≈ {}, got {v_norm}",
            FRAC_PI_2
        );
    }

    // —— Test 13b: 3D orthogonal subspaces in R^6 — H^T H = (π/2)^2 · I_3 ——
    //
    // Multi-dimensional analogue of the previous test: takes two
    // fully-orthogonal 3-planes in R^6 (first 3 axes vs. last 3) and
    // verifies that all three principal angles land on π/2. This was
    // the test case used by `cosmos-palace` to expose the rank-deficient
    // SVD on the W1^T W2 path; the cosmos-rs `logarithmic_map` is
    // constructed so that the orthogonal component `M = W2 - W1·gram`
    // already carries the full sin-component information, so this case
    // should be handled natively without any extra orthogonal-column
    // completion. The test is included here to lock that invariant in.
    #[test]
    fn log_orthogonal_3d_in_r6_gives_pi_over_2_identity() {
        use std::f64::consts::FRAC_PI_2;

        let mut w1 = Array2::<f64>::zeros((6, 3));
        let mut w2 = Array2::<f64>::zeros((6, 3));
        for i in 0..3 {
            w1[[i, i]] = 1.0;
            w2[[i + 3, i]] = 1.0;
        }

        let h = logarithmic_map(&w1, &w2).unwrap();

        // H^T H should equal (π/2)^2 · I_3.
        let hth = h.t().dot(&h);
        let expected_diag = FRAC_PI_2.powi(2);
        let eye3 = Array2::<f64>::eye(3);
        let expected = eye3 * expected_diag;
        let diff = frob_norm(&(&hth - &expected));
        assert!(
            diff < 1e-10,
            "log at π/2 should give diag((π/2)^2): diff = {diff}"
        );
    }

    // ========================================================
    // Exp / Log consistency tests
    // ========================================================

    // —— Test 14: Exp(W, Log(W, W2)) ≈ W2 ——
    #[test]
    fn exp_of_log_recovers_target() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        // Target: W1 rotated by a small angle in e3/e4 directions
        let v = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.15]];

        let w2 = exponential_map(&w1, &v, &ExponentialMapConfig::default()).unwrap();

        // Now recover via Log then Exp
        let v_log = logarithmic_map(&w1, &w2).unwrap();
        let w2_rec = exponential_map(&w1, &v_log, &ExponentialMapConfig::default()).unwrap();

        let d = geodesic_distance(&w2, &w2_rec).unwrap();
        assert!(
            d < 1e-6,
            "Exp(Log()) should recover target, got distance {d}"
        );
    }

    // —— Test 15: Log(W, Exp(W, V)) ≈ V ——
    #[test]
    fn log_of_exp_recovers_tangent() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        // Tangent orthogonal to W
        let v = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.15]];

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();
        let v_rec = logarithmic_map(&w, &w2).unwrap();

        let diff = frob_norm(&(&v - &v_rec));
        assert!(
            diff < 1e-6,
            "Log(Exp()) should recover tangent, got ||V - V_rec|| = {diff}"
        );
    }

    // —— Test 16: Exp/Log roundtrip for 1D subspaces ——
    #[test]
    fn exp_log_roundtrip_1d() {
        let w = array![[1.0], [0.0], [0.0]];
        let v = array![[0.0], [0.4], [0.0]]; // tangent toward e2

        let w2 = exponential_map(&w, &v, &ExponentialMapConfig::default()).unwrap();
        let v_rec = logarithmic_map(&w, &w2).unwrap();

        let diff = frob_norm(&(&v - &v_rec));
        assert!(diff < 1e-6, "1D roundtrip failed: ||V - V_rec|| = {diff}");
    }

    // ========================================================
    // parallel_transport tests
    // ========================================================

    // —— Test 17: Transport at same point returns same tangent ——
    #[test]
    fn transport_at_same_point_returns_same() {
        let w = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let t = array![[0.0, 0.0], [0.0, 0.0], [0.5, 0.3]];

        let transported = parallel_transport(&w, &w, &t).unwrap();

        let diff = frob_norm(&(&t - &transported));
        assert!(diff < 1e-10, "Expected same tangent, got diff {diff}");
    }

    // —— Test 18: Parallel transport preserves Frobenius norm ——
    #[test]
    fn transport_preserves_norm() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let delta = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.2]];

        let w2 = exponential_map(&w1, &delta, &ExponentialMapConfig::default()).unwrap();

        let t = array![[0.0, 0.0], [0.0, 0.0], [0.3, 0.0], [0.0, 0.4]];

        let transported = parallel_transport(&w1, &w2, &t).unwrap();

        let n1 = frob_norm(&t);
        let n2 = frob_norm(&transported);
        assert!((n1 - n2).abs() < 1e-6, "Norm changed: {n1} → {n2}");
    }

    // —— Test 19: Transported vector is in tangent space at target ——
    #[test]
    fn transport_result_is_in_target_tangent_space() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let delta = array![[0.0, 0.0], [0.0, 0.0], [0.15, 0.0], [0.0, 0.15]];

        let w2 = exponential_map(&w1, &delta, &ExponentialMapConfig::default()).unwrap();

        let t = array![[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.3]];

        let transported = parallel_transport(&w1, &w2, &t).unwrap();

        // W2^T @ transported should be ≈ 0
        let wtv = w2.t().dot(&transported);
        let wtv_norm = frob_norm(&wtv);
        assert!(
            wtv_norm < 1e-6,
            "Expected W2^T @ Γ(V) ≈ 0, got ||W2^T Γ(V)|| = {wtv_norm}"
        );
    }

    // —— Test 20: Dimension mismatch W_start vs W_end → Err ——
    #[test]
    fn transport_dimension_mismatch_w_start_w_end() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let w2 = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2 vs 3×2

        let t = Array2::zeros(w1.dim());

        let result = parallel_transport(&w1, &w2, &t);
        assert!(result.is_err());
    }

    // —— Test 21: Dimension mismatch W_start vs tangent → Err ——
    #[test]
    fn transport_dimension_mismatch_w_start_tangent() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

        let w2 = w1.clone();

        let t_bad = array![[0.0, 0.0], [0.0, 0.0]]; // 2×2 vs 3×2

        let result = parallel_transport(&w1, &w2, &t_bad);
        assert!(result.is_err());
    }

    // —— Test 22: 1D parallel transport preserves norm ——
    #[test]
    fn transport_1d_preserves_norm() {
        let w1 = array![[1.0], [0.0], [0.0]]; // span{e1}
        let w2 = array![[0.0], [1.0], [0.0]]; // span{e2}

        let t = array![[0.0], [0.0], [0.7]]; // tangent toward e3

        let transported = parallel_transport(&w1, &w2, &t).unwrap();

        let n1 = frob_norm(&t);
        let n2 = frob_norm(&transported);
        assert!((n1 - n2).abs() < 1e-6, "1D norm changed: {n1} → {n2}");
    }

    // —— Test 23: Transport and reverse transport is identity ——
    #[test]
    fn transport_roundtrip_recovers_tangent() {
        let w1 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let delta = array![[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1]];

        let w2 = exponential_map(&w1, &delta, &ExponentialMapConfig::default()).unwrap();

        // Tangent at w1 orthogonal to W
        let t = array![[0.0, 0.0], [0.0, 0.0], [0.3, 0.0], [0.0, 0.2]];

        // Transport forward then backward
        let t_at_w2 = parallel_transport(&w1, &w2, &t).unwrap();
        let t_back = parallel_transport(&w2, &w1, &t_at_w2).unwrap();

        let diff = frob_norm(&(&t - &t_back));
        assert!(
            diff < 1e-5,
            "Roundtrip should recover tangent, got ||T - T_back|| = {diff}"
        );
    }

    // ========================================================
    // ExponentialMapConfig tests
    // ========================================================

    // —— Test 24: Default config has t = 1.0 ——
    #[test]
    fn exp_config_default() {
        let config = ExponentialMapConfig::default();
        assert!((config.t - 1.0).abs() < 1e-15);
    }

    // ========================================================
    // Thin-SVD map invariance
    // ========================================================

    /// Pins [`exponential_map`] and [`retract_polar_gram`] to the results they
    /// produced when [`crate::geometry::svd::svd`] was backed by the full
    /// LAPACK `dgesvd` factorisation instead of the divide-and-conquer thin
    /// `dgesdd` one.
    ///
    /// Both maps are re-implemented verbatim inside this module against a
    /// local full-SVD helper, so the comparison isolates exactly one variable:
    /// which LAPACK driver produced `(U, Σ, Vt)`.
    ///
    /// `dgesdd` and `dgesvd` may disagree on the sign of each singular-vector
    /// pair `(u_i, v_i)`. Both maps only ever form products of the shape
    /// `U f(Σ) Vᵀ`, which are invariant under a simultaneous sign flip of
    /// `u_i` and `v_i`, so the map outputs are comparable even though the
    /// factors are not. Nothing below inspects a factor directly.
    #[cfg(feature = "lapack")]
    mod thin_svd_map_invariance {
        use super::*;
        use crate::geometry::svd::SvdResult;
        use ndarray_linalg::SVD as SVDTrait;

        /// Deterministic xorshift64 matrix generator with entries in `[-1, 1)`.
        fn pseudo_random_matrix(m: usize, n: usize, seed: u64) -> Array2<f64> {
            let mut state = seed | 1;
            Array2::from_shape_fn((m, n), |_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
        }

        /// Pre-change SVD path: full `dgesvd` factors sliced to the leading `k`.
        fn svd_full_then_slice(a: &Array2<f64>) -> SvdResult {
            let (m, n) = a.dim();
            let k = m.min(n);
            let (u_opt, s, vt_opt) = a.svd(true, true).expect("LAPACK SVD failed");
            let u = u_opt.unwrap().slice(ndarray::s![.., ..k]).to_owned();
            let vt = vt_opt.unwrap().slice(ndarray::s![..k, ..]).to_owned();
            SvdResult { u, s, vt }
        }

        /// Copy of `retract_polar_gram` wired to the pre-change SVD path.
        fn retract_polar_gram_old(w_p: &Array2<f64>) -> Array2<f64> {
            let gram_raw = w_p.t().dot(w_p);
            let gram = 0.5 * (&gram_raw + &gram_raw.t());
            let svd_res = svd_full_then_slice(&gram);
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

        /// Copy of [`exponential_map`] wired to the pre-change SVD path.
        fn exponential_map_old(
            w: &Array2<f64>,
            tangent_vec: &Array2<f64>,
            config: &ExponentialMapConfig,
        ) -> Result<Array2<f64>, GrassmannError> {
            if w.dim() != tangent_vec.dim() {
                return Err(GrassmannError::DimensionMismatch {
                    expected: w.nrows(),
                    got: tangent_vec.nrows(),
                });
            }

            let v_proj = tangent_vec - &w.dot(&w.t().dot(tangent_vec));

            let v_norm = v_proj.mapv(|x| x.powi(2)).sum().sqrt();
            if v_norm < 1e-10 {
                return Ok(w.clone());
            }

            let v_scaled = config.t * v_proj;
            let svd_res = svd_full_then_slice(&v_scaled);

            let cos_s = svd_res.s.mapv(f64::cos);
            let sin_s = svd_res.s.mapv(f64::sin);
            let k = cos_s.len();

            let diag_cos =
                Array2::from_shape_fn((k, k), |(i, j)| if i == j { cos_s[i] } else { 0.0 });
            let diag_sin =
                Array2::from_shape_fn((k, k), |(i, j)| if i == j { sin_s[i] } else { 0.0 });

            let term1 = w.dot(&svd_res.vt.t().dot(&diag_cos).dot(&svd_res.vt));
            let term2 = svd_res.u.dot(&diag_sin.dot(&svd_res.vt));

            Ok(retract_polar_gram_old(&(term1 + term2)))
        }

        /// Build a deterministic Stiefel point of shape `(n, k)`.
        ///
        /// The orthonormalisation runs through the current `retract_polar_gram`
        /// for both sides of every comparison, so it contributes no differential
        /// bias — it only has to produce a valid, reproducible base point.
        fn stiefel_point(n: usize, k: usize, seed: u64) -> Array2<f64> {
            retract_polar_gram(&pseudo_random_matrix(n, k, seed))
        }

        #[test]
        fn exponential_map_matches_full_svd_path() {
            const TOL: f64 = 1e-12;

            // `scale` keeps the principal angles well inside (0, π/2) so the
            // geodesic step is a generic, non-degenerate one.
            for (n, k, scale, seed_w, seed_v) in [
                (784_usize, 16_usize, 0.02_f64, 0x0AB1_u64, 0x0AB2_u64),
                (32, 8, 0.10, 0x0CD1, 0x0CD2),
            ] {
                let w = stiefel_point(n, k, seed_w);
                let v = pseudo_random_matrix(n, k, seed_v).mapv(|x| x * scale);

                for t in [1.0_f64, 0.5] {
                    let config = ExponentialMapConfig { t };
                    let new = exponential_map(&w, &v, &config).unwrap();
                    let old = exponential_map_old(&w, &v, &config).unwrap();

                    assert_eq!(new.dim(), old.dim(), "exp shape for ({n},{k})");
                    for i in 0..n {
                        for j in 0..k {
                            let diff = (new[[i, j]] - old[[i, j]]).abs();
                            assert!(
                                diff < TOL,
                                "exp({n},{k}, t={t}) mismatch at ({i},{j}): \
                                 new={} old={} diff={diff}",
                                new[[i, j]],
                                old[[i, j]]
                            );
                        }
                    }

                    // The result must still be a Stiefel point.
                    let wtw = new.t().dot(&new);
                    for i in 0..k {
                        for j in 0..k {
                            let expected = if i == j { 1.0 } else { 0.0 };
                            assert!(
                                (wtw[[i, j]] - expected).abs() < 1e-13,
                                "exp({n},{k}, t={t}) W^T W[{i},{j}] = {}",
                                wtw[[i, j]]
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn retract_polar_gram_matches_full_svd_path() {
            const TOL: f64 = 1e-12;

            for (n, k, seed) in [
                (784_usize, 16_usize, 0x0EF1_u64),
                (32, 8, 0x0EF2),
                (200, 64, 0x0EF3),
            ] {
                // A drifted (non-orthonormal) point, which is what the
                // retraction is actually asked to repair in the update path.
                let raw = pseudo_random_matrix(n, k, seed);
                let new = retract_polar_gram(&raw);
                let old = retract_polar_gram_old(&raw);

                assert_eq!(new.dim(), old.dim(), "retraction shape for ({n},{k})");
                for i in 0..n {
                    for j in 0..k {
                        let diff = (new[[i, j]] - old[[i, j]]).abs();
                        assert!(
                            diff < TOL,
                            "retract({n},{k}) mismatch at ({i},{j}): \
                             new={} old={} diff={diff}",
                            new[[i, j]],
                            old[[i, j]]
                        );
                    }
                }
            }
        }

        #[test]
        fn exponential_map_first_order_expansion_holds_for_small_steps() {
            // Exp_W(t V) = W + t V + O(t²): halving t must quarter the
            // deviation from the first-order prediction.
            for (n, k, seed_w, seed_v) in [
                (784_usize, 16_usize, 0x0FA1_u64, 0x0FA2_u64),
                (32, 8, 0x0FB1, 0x0FB2),
            ] {
                let w = stiefel_point(n, k, seed_w);
                let raw = pseudo_random_matrix(n, k, seed_v);
                // Project onto the tangent space and normalise, so ||V||_F = 1
                // and t is exactly the geodesic arc length.
                let proj = &raw - &w.dot(&w.t().dot(&raw));
                let v = &proj / frob_norm(&proj);

                let residual = |t: f64| {
                    let moved = exponential_map(&w, &v, &ExponentialMapConfig { t }).unwrap();
                    frob_norm(&(&moved - &(&w + &(t * &v))))
                };

                let r1 = residual(1e-3);
                let r2 = residual(5e-4);
                assert!(r1 < 1e-5, "({n},{k}) first-order residual too large: {r1}");
                assert!(
                    (r1 / r2 - 4.0).abs() < 0.1,
                    "({n},{k}) residual should scale as t²: r(1e-3)={r1}, r(5e-4)={r2}"
                );
            }
        }
    }
}
