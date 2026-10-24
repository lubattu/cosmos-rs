//! Matrix exponential utilities for gauge-theoretic evolution.
//!
//! This module provides matrix exponential computation for the
//! GaugeDynamics. The key operation is computing:
//!
//!  U = exp(-η A*)
//!
//! where A* is an anti-symmetric matrix (Lie algebra element of SO(n)).
//!
//! For anti-symmetric matrices, exp(A) is always orthogonal, which is
//! crucial for preserving manifold structure during conjugation updates.
//! ...

#[cfg(not(feature = "lapack"))]
use ndarray::s;
use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::geometry::qr::{QrConfig, qr};
use crate::geometry::svd::{SvdConfig, svd};

/// Errors that can occur in matrix exponential operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MatrixExpError {
    /// Input matrix is not square
    #[error("Input matrix is not square: got [{rows}, {cols}]")]
    NotSquare {
        /// Number of rows the input matrix has.
        rows: usize,
        /// Number of columns the input matrix has.
        cols: usize,
    },

    /// Two matrices have incompatible dimensions.
    #[error("Matrices have incompatible dimensions: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension required by the operation.
        expected: usize,
        /// Dimension actually supplied.
        got: usize,
    },

    /// Linear system solve failed (singular matrix).
    #[error("Linear system solve failed (singular matrix)")]
    SingularMatrix,
}

/// Method for mapping anti-symmetric matrices to orthogonal matrices.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OrthoMethod {
    /// Taylor series with scaling & squaring. General purpose, good accuracy.
    #[default]
    Taylor,
    /// Padé \[6,6\] approximation with scaling * squaring. More accurate than Taylor.
    Pade,
    /// Cayley-based exponential: (I + A/2)(I - A/2)⁻¹ ≈ exp(A). Faster, single matrix solve.
    Cayley,
    /// QR retraction of I + A. Fastest, first-order approximation.
    Qr,
    /// Taylor expansion followed by polar retraction to O(n).
    TaylorPolar,
    /// Householder geodesic for low-rank connections. Falls back to Cayley for rank > threshold.
    Householder,
}

/// Method for retracting a drifted matrix back to O(n).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RetractionMethod {
    /// QR decomposition. Fast, O(n³).
    #[default]
    Qr,
    /// SVD polar decomposition: U @ Vᵀ. Most stable.
    Svd,
    /// Polar decomposition via eigendecomposition of UᵀU. Closest orthogonal matrix.
    Polar,
}

// ── Pure utilities ──────────────────────────────────

/// Compute the anti-symmetric part of a matrix.
///
/// A_antisym = (A - A^T) / 2
///
/// Used to extract the Lie algebra element from the commutator.
///
/// # Arguments
///
/// * `A`- Square matrix.
///
/// # Returns
///
/// Anti-symmetric matrix satisfying A_antisym^T = -A_antisym.
pub fn antisymmetrize(a: &Array2<f64>) -> Array2<f64> {
    (a - &a.t()) / 2.0
}

/// Compute the matrix commutator [A, B] = AB - BA.
///
/// For the gauge evolution law, we compute [G, P] where G is the gradient
/// and P is the projector. The anti-symmetric part of this commutator
/// gives the connection A*.
///
/// # Arguments
///
/// * `A` - First matrix.
/// * `B` - Second matrix.
///
/// # Returns
///
/// Commutator [A, B] = AB - BA.
pub fn commutator(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    &a.dot(b) - &b.dot(a)
}

/// Frobenius norm of `U Uᵀ − I`: the residual of **row** orthonormality.
///
/// For a square `U` this is the usual orthogonality measure and agrees
/// with [`orthogonality_error_columns`]. For a tall `U` of shape
/// `(n, k)` with `n > k` it does not: `U Uᵀ` is then a rank-`k`
/// projector, its distance to the `n × n` identity is `sqrt(n − k)`
/// whatever the columns look like, and the value says nothing about
/// whether the columns are orthonormal.
///
/// State blocks are tall frames whose *columns* carry the invariant.
/// Measure them with [`orthogonality_error_columns`].
///
/// # Arguments
///
/// * `u` - matrix whose rows are expected to be orthonormal.
///
/// # Returns
///
/// Frobenius norm of the deviation of `U Uᵀ` from the identity.
pub fn orthogonality_error(u: &Array2<f64>) -> f64 {
    let n = u.nrows();
    let eye = Array2::<f64>::eye(n);

    (u.dot(&u.t()) - eye).mapv(|x| x.powi(2)).sum().sqrt()
}

/// Frobenius norm of `Uᵀ U − I_k`: the residual of **column** orthonormality.
///
/// This is the invariant every state block preserves. It is defined at
/// any shape: for a square `U` it agrees with [`orthogonality_error`],
/// and for a tall `U` of shape `(n, k)` it is zero exactly when the `k`
/// columns form an orthonormal frame in `n` dimensions.
///
/// # Arguments
///
/// * `u` - matrix whose columns are expected to be orthonormal.
///
/// # Returns
///
/// Frobenius norm of the deviation of `Uᵀ U` from the `k × k` identity.
pub fn orthogonality_error_columns(u: &Array2<f64>) -> f64 {
    let k = u.ncols();
    let eye = Array2::<f64>::eye(k);

    (u.t().dot(u) - eye).mapv(|x| x.powi(2)).sum().sqrt()
}

// ── Linear solve (private) ──────────────────────────

/// Solve the linear system AX = B using LAPACK LU factorization.
///
/// Uses ndarray-linalg's `Solve` trait for numerically stable,
/// LAPACK-backed linear system solving.
///
/// # Arguments
///
/// * `a` - Coefficient matrix of shape (n, n).
/// * `b` - Right-hand side matrix of shape (n, m).
///
/// # Returns
///
/// `Some(X)` where AX = B, or `None` if A is singular.
#[cfg(feature = "lapack")]
fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    use ndarray_linalg::Solve;
    let m = b.ncols();
    if m == 1 {
        // Single column: solve Ax = b directly
        let b_col = b.column(0).to_owned();
        let x_col = a.solve(&b_col).ok()?;
        Some(x_col.into_shape_with_order((a.nrows(), 1)).unwrap())
    } else {
        // Multiple columns: solve column by column
        let n = a.nrows();
        let mut x = Array2::<f64>::zeros((n, m));
        for k in 0..m {
            let b_col = b.column(k).to_owned();
            let x_col = a.solve(&b_col).ok()?;
            x.column_mut(k).assign(&x_col);
        }
        Some(x)
    }
}

/// Solve the linear system AX = B via Gaussian elimination with partial pivoting.
///
/// Fallback implementation when LAPACK is not available.
///
/// # Arguments
///
/// * `a` - Coefficient matrix of shape (n, n).
/// * `b` - Right-hand side matrix of shape (n, m).
///
/// # Returns
///
/// `Some(X)` where AX = B, or `None` if A is singular (pivot < 1e-14).
#[cfg(not(feature = "lapack"))]
fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();

    // Build augmented matrix [A | B], shape n × (n + n)
    let m = b.ncols();
    let mut aug = Array2::<f64>::zeros((n, n + m));
    aug.slice_mut(s![.., ..n]).assign(a);
    aug.slice_mut(s![.., n..]).assign(b);

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot: row with largest absolute value in this column
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Singular check
        if max_val < 1e-14 {
            return None;
        }

        // Swap rows
        if max_row != col {
            for j in 0..(n + m) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(n + m) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution — solve for each column of X
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

// ── Individual algorithms ───────────────────────────

/// Cayley transform: maps anti-symmetric A to orthogonal U.
///
/// The Cayley transform is defined as:
///   U = (I - A)(I + A)^{-1}
///
/// For anti-symmetric A, this produces an orthogonal matrix U.
/// It is computationally cheaper than matrix exponential (single inverse
/// instead of series/Padé) and is equivalent to exp(-2A) + O(||A||³) for
/// small ||A||. Note the sign and rate: this raw form moves in the
/// *opposite* direction of exp(A), twice as fast. For the properly-scaled
/// approximation to exp(A), use [`orthogonal_exponential`] (= Cayley(-A/2)),
/// which is what the [`OrthoMethod::Cayley`] dispatch computes.
///
/// # Mathematical Properties
/// - If A^T = -A, then U^T U = I (orthogonal)
/// - Cayley(A) = exp(-2A) + O(||A||³) for small ||A||
/// - Cayley is a rational approximation to exp on SO(n)
/// - det(U) = 1 when -1 is not an eigenvalue of A
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n). Must satisfy A^T = -A.
///
/// # Returns
///
/// Orthogonal matrix U = (I - A)(I + A)^{-1} of shape (n, n).
///
/// # Errors
///
/// Returns [`MatrixExpError::SingularMatrix`] if `I + A` is singular.
pub fn cayley_transform(a: &Array2<f64>) -> Result<Array2<f64>, MatrixExpError> {
    let n = a.nrows();
    let eye = Array2::<f64>::eye(n);

    // I + A
    let ipa = &eye + a;
    // I - A
    let ima = &eye - a;

    // Solve (I+A)^T . X^T = (I-A)^T -> X = ((I+A)^T \ (I-A)^T)^T
    let x_t = solve_linear_system(&ipa.t().to_owned(), &ima.t().to_owned())
        .ok_or(MatrixExpError::SingularMatrix)?;
    Ok(x_t.t().to_owned())
}

/// Cayley-based orthogonal exponential: rational approximation to exp(A).
///
/// Computes Cayley(-A/2) = (I + A/2)(I - A/2)^{-1}, the \[1,1\] Padé
/// approximant of exp(A). This is the properly-scaled Cayley form that
/// matches the exp(A) direction and rate; the raw [`cayley_transform`]
/// approximates exp(-2A) instead.
///
/// # Mathematical Properties
/// - If A^T = -A, then U^T U = I exactly (orthogonal by construction)
/// - U = exp(A) + O(||A||³) for small ||A||
/// - Single linear solve; cheaper than series/Padé exponentials
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n). Must satisfy A^T = -A.
///
/// # Returns
///
/// Orthogonal matrix U = (I + A/2)(I - A/2)^{-1} of shape (n, n).
///
/// # Errors
///
/// Returns [`MatrixExpError::SingularMatrix`] if `I - A/2` is singular.
pub fn orthogonal_exponential(a: &Array2<f64>) -> Result<Array2<f64>, MatrixExpError> {
    cayley_transform(&(a * -0.5))
}

/// QR retraction: first-order approximation to exp(A).
///
/// Computes the orthogonal factor Q from the QR decomposition of (I + A).
/// This is a retraction on the Stiefel/orthogonal manifold that is tangent
/// to exp(A) at A = 0.
///
/// # Mathematical Properties
/// - Q is always orthogonal by construction
/// - QR(I + A) gives Q ≈ exp(A) + O(||A||²)
/// - Fastest method (QR is highly optimized in LAPACK/cuSOLVER)
/// - Not exact, but sufficient for optimization with small step sizes
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
///
/// # Returns
///
/// Orthogonal matrix Q from QR(I + A) of shape (n, n).
pub fn qr_retraction(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let eye = Array2::<f64>::eye(n);

    let qr_res = qr(&(eye + a), &QrConfig::default());

    // Ensure positive diagonal in R for uniqueness (det Q = +1)
    let mut q = qr_res.q;
    for j in 0..n {
        let r_jj = qr_res.r[[j, j]];
        if r_jj < 0.0 {
            // Flip sign of column j of Q
            for i in 0..n {
                q[[i, j]] = -q[[i, j]];
            }
        }
    }
    q
}

/// Taylor series approximation of exp(A) with scaling & squaring.
///
/// exp(A) = (exp(A / 2^s))^(2^s)
///
/// where s = ceil(log2(||A||)) ensures ||A/2^s|| ≤ 1 for fast convergence.
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
/// * `order` - Number of terms in the Taylor expansion.
///
/// # Returns
///
/// Approximation to exp(A).
fn taylor_exp_antisym(a: &Array2<f64>, order: usize) -> Array2<f64> {
    let n = a.nrows();

    // Scaling: compute s = ceil(log2(||A||_F))
    let norm_a = a.mapv(|x| x * x).sum().sqrt(); // Frobenius norm
    let s = if norm_a > 1.0 {
        norm_a.log2().ceil() as u32
    } else {
        0
    };

    let divisor = (1u64 << s) as f64; // 2^s
    let a_scaled = a / divisor;

    // Taylor expansion: exp(A_scaled) = I + A + A²/2! + A³/3! + ...
    let eye = Array2::<f64>::eye(n);
    let mut result = eye.clone();
    let mut term = eye; // term_k = A^k / k!

    for i in 1..=order {
        term = term.dot(&a_scaled) / (i as f64);
        result = &result + &term;
    }

    // Squaring phase: exp(A) = exp(A/2^s)^(2^s)
    for _ in 0..s {
        result = result.dot(&result);
    }

    result
}

/// Padé \[6,6\] approximation of exp(A) with scaling & squaring.
///
/// Uses the diagonal Padé approximant:
///   exp(A) ≈ N(A) · D(A)^{-1}
///
/// where N and D are polynomials computed from even/odd powers of A.
/// More accurate than Taylor for the same computational cost.
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
///
/// # Returns
///
/// Approximation to exp(A).
///
/// # Errors
///
/// Returns [`MatrixExpError::SingularMatrix`] if the Padé denominator is singular.
fn pade_exp_antisym(a: &Array2<f64>) -> Result<Array2<f64>, MatrixExpError> {
    let n = a.nrows();

    // Scaling for convergence
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let s = if norm_a > 0.5 {
        (norm_a / 0.5).log2().ceil() as u32
    } else {
        0
    };

    let divisor = (1u64 << s) as f64;
    let a_scaled = a / divisor;

    // Correct [6,6] diagonal Padé coefficients:
    // b_j = (2p-j)! p! / ((2p)! j! (p-j)!)  with p = 6
    let b0: f64 = 1.0;
    let b1: f64 = 0.5;
    let b2: f64 = 5.0 / 44.0;
    let b3: f64 = 1.0 / 66.0;
    let b4: f64 = 1.0 / 792.0;
    let b5: f64 = 1.0 / 15840.0;
    let b6: f64 = 1.0 / 665280.0;

    // Compute powers
    let eye = Array2::<f64>::eye(n);
    let a2 = a_scaled.dot(&a_scaled);
    let a4 = a2.dot(&a2);
    let a6 = a4.dot(&a2);

    // Even part: U = b0·I + b2·A² + b4·A⁴ + b6·A⁶
    let u_even = &eye * b0 + &a2 * b2 + &a4 * b4 + &a6 * b6;

    // Odd part: V = b1·I + b3·A² + b5·A⁴
    // so that A·V = b1·A + b3·A³ + b5·A⁵
    let v_odd = &eye * b1 + &a2 * b3 + &a4 * b5;

    // N(A) = U + A·V, D(A) = U - A·V
    let av = a_scaled.dot(&v_odd);
    let numer = &u_even + &av;
    let denom = &u_even - &av;

    // Solve D · X = N  →  X = D⁻¹ N = exp(A_scaled)
    let mut result = solve_linear_system(&denom, &numer).ok_or(MatrixExpError::SingularMatrix)?;

    // Squaring phase
    for _ in 0..s {
        result = result.dot(&result);
    }

    Ok(result)
}

/// Taylor expansion followed by polar retraction to O(n).
///
/// Computes a truncated Taylor series of exp(A), then projects onto
/// the orthogonal group via SVD: U = Ũ → U_svd · Vᵀ (nearest orthogonal).
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
/// * `order` - Number of Taylor terms (default 8).
///
/// # Returns
///
/// Orthogonal matrix U of shape (n, n).
pub fn taylor_polar_retraction(a: &Array2<f64>, order: usize) -> Array2<f64> {
    let n = a.nrows();
    let eye = Array2::<f64>::eye(n);

    // Taylor expansion: Ũ = I + A + A²/2! + ...
    let mut u_tilde = eye.clone();
    let mut term = eye.clone();

    for i in 1..=order {
        term = term.dot(a) / (i as f64);
        u_tilde = &u_tilde + &term;
    }

    // Polar retraction via SVD: U = U_svd · Vᵀ
    // This gives the closest orthogonal matrix to Ũ in Frobenius norm
    let svd_res = svd(&u_tilde, &SvdConfig::default());
    svd_res.u.dot(&svd_res.vt)
}

/// Householder geodesic for low-rank anti-symmetric connections.
///
/// When A has low rank (≤ rank_threshold), computes exp(A) as a planar
/// rotation in the eigenspace. Falls back to Cayley for higher rank.
///
/// For rank-2 A = σ(uv^T - vu^T), the exponential acts as a rotation
/// by angle σ in the span{u, v} plane and identity elsewhere.
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
/// * `rank_threshold` - Maximum rank for Householder method (default 4).
///
/// # Returns
///
/// Orthogonal matrix U ≈ exp(A).
///
/// # Errors
///
/// Returns [`MatrixExpError::SingularMatrix`] if the high-rank fallback's
/// linear solve fails.
pub fn householder_geodesic(
    a: &Array2<f64>,
    rank_threshold: usize,
) -> Result<Array2<f64>, MatrixExpError> {
    let n = a.nrows();

    // Estimate rank via SVD
    let svd_res = svd(a, &SvdConfig::default());
    let s = &svd_res.s;

    // Tolerance for significant singular values
    let tol = if !s.is_empty() && s[0] > 0.0 {
        s[0] * (n as f64) * f64::EPSILON * 10.0
    } else {
        f64::EPSILON
    };

    let rank = s.iter().filter(|&&sv| sv > tol).count();

    if rank > rank_threshold {
        // Fall back to the Cayley-based exponential for higher rank
        return orthogonal_exponential(a);
    }

    if rank == 0 {
        // A ≈ 0, return identity
        return Ok(Array2::<f64>::eye(n));
    }

    // For rank-2: A ≈ σ(uv^T - vu^T)
    // exp(A) rotates by angle σ in span{u, v}
    let sigma = s[0];
    let u = svd_res.u.column(0).to_owned();
    let v = if rank >= 2 {
        svd_res.u.column(1).to_owned()
    } else {
        Array1::zeros(n)
    };

    let eye = Array2::<f64>::eye(n);
    let cos_s = sigma.cos();
    let sin_s = sigma.sin();

    // Projection onto rotation plane: P = uuᵀ + vvᵀ
    let uu = outer(&u, &u);
    let vv = outer(&v, &v);
    let p = &uu + &vv;

    // Rotation generator in the plane: J = uvᵀ - vuᵀ
    let uv = outer(&u, &v);
    let vu = outer(&v, &u);
    let j = &uv - &vu;

    // exp(A) = I + (cos σ - 1)P + sin σ · J
    Ok(&eye + &(&p * (cos_s - 1.0)) + &(&j * sin_s))
}

/// Compute outer product of two vectors: a ⊗ b → matrix (n, m).
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

// ── Dispatch functions ──────────────────────────────

/// Compute orthogonal matrix from anti-symmetric matrix.
///
/// For anti-symmetric A (A^T = -A), returns an orthogonal matrix U
/// using the specified method. This is the main entry point for
/// anti-symmetric → orthogonal mapping.
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
/// * `method` - Computation method (see [`OrthoMethod`]).
///
/// # Returns
///
/// Orthogonal matrix U of shape (n, n).
///
/// # Errors
///
/// Returns [`MatrixExpError::NotSquare`] if `a` is not square.
/// Returns [`MatrixExpError::SingularMatrix`] if the chosen method
/// requires a linear solve that fails.
pub fn matrix_exp_antisym(
    a: &Array2<f64>,
    method: OrthoMethod,
) -> Result<Array2<f64>, MatrixExpError> {
    let (rows, cols) = a.dim();
    if rows != cols {
        return Err(MatrixExpError::NotSquare { rows, cols });
    }

    match method {
        OrthoMethod::Taylor => Ok(taylor_exp_antisym(a, 12)),
        OrthoMethod::Pade => pade_exp_antisym(a),
        OrthoMethod::Cayley => orthogonal_exponential(a),
        OrthoMethod::Qr => Ok(qr_retraction(a)),
        OrthoMethod::TaylorPolar => Ok(taylor_polar_retraction(a, 8)),
        OrthoMethod::Householder => householder_geodesic(a, 4),
    }
}

/// Compute exp(scale * A) for anti-symmetric A.
///
/// Convenience function for U = exp(-η A*) with learning rate η.
///
/// # Arguments
///
/// * `a` - Anti-symmetric matrix of shape (n, n).
/// * `scale` - Scalar multiplier (typically `-lr` for the evolution law).
/// * `method` - Computation method (see [`OrthoMethod`]).
///
/// # Returns
///
/// Orthogonal matrix U = exp(scale * A).
///
/// # Errors
///
/// Returns [`MatrixExpError::NotSquare`] if `a` is not square.
/// Returns [`MatrixExpError::SingularMatrix`] if the chosen method
/// requires a linear solve that fails.
pub fn scaled_matrix_exp_antisym(
    a: &Array2<f64>,
    scale: f64,
    method: OrthoMethod,
) -> Result<Array2<f64>, MatrixExpError> {
    let scaled = a * scale;
    matrix_exp_antisym(&scaled, method)
}

/// Retract a matrix back to the orthogonal group O(n).
///
/// If numerical drift has made U non-orthogonal (U·Uᵀ ≠ I),
/// this projects it back to O(n).
///
/// # Arguments
///
/// * `u` - Near-orthogonal matrix of shape (n, n).
/// * `method` - Retraction method (see [`RetractionMethod`]).
///
/// # Returns
///
/// Orthogonal matrix Q closest to U.
pub fn retract_to_orthogonal(u: &Array2<f64>, method: RetractionMethod) -> Array2<f64> {
    match method {
        RetractionMethod::Qr => {
            let n = u.nrows();
            let qr_res = qr(u, &QrConfig::default());
            let mut q = qr_res.q;
            for j in 0..n {
                if qr_res.r[[j, j]] < 0.0 {
                    for i in 0..n {
                        q[[i, j]] = -q[[i, j]];
                    }
                }
            }
            q
        }
        RetractionMethod::Svd => {
            let svd_res = svd(u, &SvdConfig::default());
            svd_res.u.dot(&svd_res.vt)
        }
        RetractionMethod::Polar => {
            // Polar: Q = U·(UᵀU)^{-1/2}
            // Equivalent to SVD retraction: U_svd · Vᵀ
            let svd_res = svd(u, &SvdConfig::default());
            svd_res.u.dot(&svd_res.vt)
        }
    }
}
// ── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: build a random-ish anti-symmetric matrix from a seed pattern.
    /// Uses a deterministic construction so tests are reproducible.
    fn make_antisym(n: usize, scale: f64) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                // Deterministic "random" value from indices
                let val = scale * ((i * 7 + j * 13) as f64 % 10.0 - 5.0) / 5.0;
                m[[i, j]] = val;
                m[[j, i]] = -val;
            }
        }
        m
    }

    /// Helper: check that a matrix is orthogonal (U·Uᵀ ≈ I).
    fn assert_orthogonal(u: &Array2<f64>, tol: f64, label: &str) {
        let err = orthogonality_error(u);
        assert!(
            err < tol,
            "{label}: orthogonality error {err:.2e} exceeds tol {tol:.2e}"
        );
    }

    // ——————————————————————————————————————————————
    // antisymmetrize
    // ——————————————————————————————————————————————

    #[test]
    fn antisymmetrize_of_symmetric_is_zero() {
        let s = array![[1.0, 2.0], [2.0, 3.0]];
        let result = antisymmetrize(&s);
        for &v in result.iter() {
            assert!(v.abs() < 1e-14, "Expected 0, got {v}");
        }
    }

    #[test]
    fn antisymmetrize_of_antisymmetric_is_itself() {
        let a = array![[0.0, 3.0], [-3.0, 0.0]];
        let result = antisymmetrize(&a);
        for (r, o) in result.iter().zip(a.iter()) {
            assert!((r - o).abs() < 1e-14, "Expected {o}, got {r}");
        }
    }

    #[test]
    fn antisymmetrize_general_matrix() {
        let m = array![[1.0, 4.0], [2.0, 3.0]];
        let result = antisymmetrize(&m);
        // (M - Mᵀ)/2 = [[0, 1], [-1, 0]]
        assert!((result[[0, 1]] - 1.0).abs() < 1e-14);
        assert!((result[[1, 0]] + 1.0).abs() < 1e-14);
        assert!(result[[0, 0]].abs() < 1e-14);
    }

    #[test]
    fn antisymmetrize_result_is_antisymmetric() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let a = antisymmetrize(&m);
        let diff = &a + &a.t();
        for &v in diff.iter() {
            assert!(v.abs() < 1e-14, "A + Aᵀ should be zero, got {v}");
        }
    }

    // ——————————————————————————————————————————————
    // commutator
    // ——————————————————————————————————————————————

    #[test]
    fn commutator_of_identity_is_zero() {
        let eye = Array2::<f64>::eye(3);
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = commutator(&eye, &b);
        for &v in result.iter() {
            assert!(v.abs() < 1e-14, "Expected 0, got {v}");
        }
    }

    #[test]
    fn commutator_with_itself_is_zero() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let result = commutator(&a, &a);
        for &v in result.iter() {
            assert!(v.abs() < 1e-14, "[A,A] should be 0, got {v}");
        }
    }

    #[test]
    fn commutator_antisymmetric_inputs() {
        // [A, B] = AB - BA
        let a = array![[0.0, 1.0], [-1.0, 0.0]];
        let b = array![[0.0, 2.0], [-2.0, 0.0]];
        let result = commutator(&a, &b);
        // AB = [[-2, 0],[0, -2]], BA = [[-2, 0],[0, -2]] → [A,B] = 0
        for &v in result.iter() {
            assert!(v.abs() < 1e-14, "Expected 0, got {v}");
        }
    }

    #[test]
    fn commutator_is_anticommutative() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);
        let sum = &ab + &ba;
        for &v in sum.iter() {
            assert!(v.abs() < 1e-14, "[A,B] + [B,A] should be 0, got {v}");
        }
    }

    // ——————————————————————————————————————————————
    // orthogonality_error
    // ——————————————————————————————————————————————

    #[test]
    fn orthogonality_error_of_identity_is_zero() {
        let eye = Array2::<f64>::eye(5);
        let err = orthogonality_error(&eye);
        assert!(err < 1e-14, "Expected ~0, got {err}");
    }

    #[test]
    fn orthogonality_error_of_rotation_is_zero() {
        let theta = 0.7_f64;
        let r = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
        let err = orthogonality_error(&r);
        assert!(
            err < 1e-14,
            "Rotation should be orthogonal, got error {err}"
        );
    }

    #[test]
    fn orthogonality_error_of_non_orthogonal_is_positive() {
        let m = array![[2.0, 0.0], [0.0, 1.0]];
        let err = orthogonality_error(&m);
        assert!(
            err > 1.0,
            "Scaled matrix should have large error, got {err}"
        );
    }

    // ——————————————————————————————————————————————
    // solve_linear_system
    // ——————————————————————————————————————————————

    #[test]
    fn solve_identity_system() {
        let eye = Array2::<f64>::eye(3);
        let b = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let x = solve_linear_system(&eye, &b).unwrap();
        for (xv, bv) in x.iter().zip(b.iter()) {
            assert!((xv - bv).abs() < 1e-12, "IX = B → X = B, got {xv} != {bv}");
        }
    }

    #[test]
    fn solve_simple_system() {
        // A = [[2, 1], [1, 3]], B = [[5], [7]]
        // Solution: x = [8/5, 9/5] = [1.6, 1.8]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![[5.0], [7.0]];
        let x = solve_linear_system(&a, &b).unwrap();
        assert!((x[[0, 0]] - 1.6).abs() < 1e-12);
        assert!((x[[1, 0]] - 1.8).abs() < 1e-12);
    }

    #[test]
    fn solve_singular_returns_none() {
        let a = array![[1.0, 2.0], [2.0, 4.0]]; // rank 1
        let b = array![[1.0], [2.0]];
        assert!(solve_linear_system(&a, &b).is_none());
    }

    #[test]
    fn solve_verifies_ax_eq_b() {
        let a = array![[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]];
        let b = array![[1.0], [2.0], [3.0]];
        let x = solve_linear_system(&a, &b).unwrap();
        let residual = a.dot(&x) - &b;
        for &v in residual.iter() {
            assert!(v.abs() < 1e-10, "AX - B should be ~0, got {v}");
        }
    }

    // ——————————————————————————————————————————————
    // cayley_transform
    // ——————————————————————————————————————————————

    #[test]
    fn cayley_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = cayley_transform(&zero).unwrap();
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-12, "Cayley(0) should be I");
        }
    }

    #[test]
    fn cayley_produces_orthogonal_2x2() {
        let a = array![[0.0, 0.5], [-0.5, 0.0]];
        let u = cayley_transform(&a).unwrap();
        assert_orthogonal(&u, 1e-12, "cayley_2x2");
    }

    #[test]
    fn cayley_produces_orthogonal_5x5() {
        let a = make_antisym(5, 0.3);
        let u = cayley_transform(&a).unwrap();
        assert_orthogonal(&u, 1e-10, "cayley_5x5");
    }

    #[test]
    fn cayley_produces_orthogonal_8x8() {
        let a = make_antisym(8, 1.0);
        let u = cayley_transform(&a).unwrap();
        assert_orthogonal(&u, 1e-10, "cayley_8x8");
    }

    #[test]
    fn cayley_raw_convention_is_exp_minus_2a() {
        // Pins the raw primitive's convention: Cayley(A) = exp(-2A) + O(||A||³).
        // For A = [[0, θ], [-θ, 0]], exp(-2A) is the rotation by -2θ.
        let theta = 0.1_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = cayley_transform(&a).unwrap();
        let phi = -2.0 * theta;
        assert!((u[[0, 0]] - phi.cos()).abs() < 1e-3, "cos mismatch");
        assert!((u[[0, 1]] - phi.sin()).abs() < 1e-3, "sin mismatch");
        assert!(u[[0, 1]] < 0.0, "raw Cayley must rotate opposite to exp(A)");
    }

    // ——————————————————————————————————————————————
    // orthogonal_exponential
    // ——————————————————————————————————————————————

    #[test]
    fn orthogonal_exponential_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = orthogonal_exponential(&zero).unwrap();
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!(
                (uv - ev).abs() < 1e-12,
                "orthogonal_exponential(0) should be I"
            );
        }
    }

    #[test]
    fn orthogonal_exponential_produces_orthogonal() {
        let a = make_antisym(5, 0.5);
        let u = orthogonal_exponential(&a).unwrap();
        assert_orthogonal(&u, 1e-10, "orthogonal_exponential_5x5");
    }

    #[test]
    fn orthogonal_exponential_matches_exp_direction_on_rotation() {
        // For A = [[0, θ], [-θ, 0]], exp(A) = [[cos θ, sin θ], [-sin θ, cos θ]].
        // Cayley(-A/2) rotates by 2·atan(θ/2) = θ + O(θ³), same direction as exp(A).
        let theta = 0.1_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = orthogonal_exponential(&a).unwrap();
        assert!((u[[0, 0]] - theta.cos()).abs() < 1e-3, "cos mismatch");
        assert!((u[[0, 1]] - theta.sin()).abs() < 1e-3, "sin mismatch");
        assert!((u[[1, 0]] + theta.sin()).abs() < 1e-3, "-sin mismatch");
        assert!(u[[0, 1]] > 0.0, "must rotate in the exp(A) direction");
    }

    // ——————————————————————————————————————————————
    // qr_retraction
    // ——————————————————————————————————————————————

    #[test]
    fn qr_retraction_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = qr_retraction(&zero);
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-12, "QR(I+0) should be I");
        }
    }

    #[test]
    fn qr_retraction_produces_orthogonal() {
        let a = make_antisym(5, 0.1);
        let u = qr_retraction(&a);
        assert_orthogonal(&u, 1e-10, "qr_retraction_5x5");
    }

    #[test]
    fn qr_retraction_small_step_close_to_identity() {
        let a = make_antisym(4, 0.001);
        let u = qr_retraction(&a);
        let eye = Array2::<f64>::eye(4);
        let diff = &u - &eye;
        let norm = diff.mapv(|x| x * x).sum().sqrt();
        assert!(norm < 0.1, "Small step QR should be close to I, got {norm}");
    }

    // ——————————————————————————————————————————————
    // taylor_exp_antisym
    // ——————————————————————————————————————————————

    #[test]
    fn taylor_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = taylor_exp_antisym(&zero, 12);
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-12, "exp(0) should be I");
        }
    }

    #[test]
    fn taylor_produces_orthogonal_small_norm() {
        let a = make_antisym(4, 0.1);
        let u = taylor_exp_antisym(&a, 12);
        assert_orthogonal(&u, 1e-6, "taylor_small_4x4");
    }

    #[test]
    fn taylor_produces_orthogonal_large_norm() {
        let a = make_antisym(5, 2.0);
        let u = taylor_exp_antisym(&a, 12);
        assert_orthogonal(&u, 1e-4, "taylor_large_5x5");
    }

    #[test]
    fn taylor_2x2_matches_exact_rotation() {
        // For A = [[0, θ], [-θ, 0]], exp(A) = [[cos θ, sin θ], [-sin θ, cos θ]]
        let theta = 0.3_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = taylor_exp_antisym(&a, 12);
        assert!((u[[0, 0]] - theta.cos()).abs() < 1e-10, "cos mismatch");
        assert!((u[[0, 1]] - theta.sin()).abs() < 1e-10, "sin mismatch");
        assert!((u[[1, 0]] + theta.sin()).abs() < 1e-10, "-sin mismatch");
        assert!((u[[1, 1]] - theta.cos()).abs() < 1e-10, "cos mismatch");
    }

    // ——————————————————————————————————————————————
    // pade_exp_antisym
    // ——————————————————————————————————————————————

    #[test]
    fn pade_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = pade_exp_antisym(&zero).unwrap();
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-12, "Padé(0) should be I");
        }
    }

    #[test]
    fn pade_produces_orthogonal_small_norm() {
        let a = make_antisym(4, 0.1);
        let u = pade_exp_antisym(&a).unwrap();
        assert_orthogonal(&u, 1e-8, "pade_small_4x4");
    }

    #[test]
    fn pade_produces_orthogonal_large_norm() {
        let a = make_antisym(5, 2.0);
        let u = pade_exp_antisym(&a).unwrap();
        assert_orthogonal(&u, 1e-4, "pade_large_5x5");
    }

    #[test]
    fn pade_2x2_matches_exact_rotation() {
        let theta = 0.3_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = pade_exp_antisym(&a).unwrap();
        assert!((u[[0, 0]] - theta.cos()).abs() < 1e-10, "cos mismatch");
        assert!((u[[0, 1]] - theta.sin()).abs() < 1e-10, "sin mismatch");
    }

    // ——————————————————————————————————————————————
    // taylor_polar_retraction
    // ——————————————————————————————————————————————

    #[test]
    fn taylor_polar_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = taylor_polar_retraction(&zero, 8);
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-10, "taylor_polar(0) should be I");
        }
    }

    #[test]
    fn taylor_polar_produces_orthogonal() {
        let a = make_antisym(5, 0.5);
        let u = taylor_polar_retraction(&a, 8);
        assert_orthogonal(&u, 1e-6, "taylor_polar_5x5");
    }

    #[test]
    fn taylor_polar_2x2_matches_rotation() {
        let theta = 0.4_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = taylor_polar_retraction(&a, 12);
        assert!((u[[0, 0]] - theta.cos()).abs() < 1e-8, "cos mismatch");
        assert!((u[[0, 1]] - theta.sin()).abs() < 1e-8, "sin mismatch");
    }

    // ——————————————————————————————————————————————
    // householder_geodesic
    // ——————————————————————————————————————————————

    #[test]
    fn householder_of_zero_is_identity() {
        let zero = Array2::<f64>::zeros((4, 4));
        let u = householder_geodesic(&zero, 4).unwrap();
        let eye = Array2::<f64>::eye(4);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-10, "householder(0) should be I");
        }
    }

    #[test]
    fn householder_rank2_produces_orthogonal() {
        // Build a rank-2 anti-symmetric: A = uv^T - vu^T
        let u = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 1.0, 0.0, 0.0];
        let a = &outer(&u, &v) - &outer(&v, &u);
        let scaled = &a * 0.5;
        let result = householder_geodesic(&scaled, 4).unwrap();
        assert_orthogonal(&result, 1e-10, "householder_rank2");
    }

    #[test]
    fn householder_falls_back_to_cayley_for_high_rank() {
        // Full-rank anti-symmetric → rank > threshold → falls back to Cayley
        let a = make_antisym(4, 0.3);
        let u = householder_geodesic(&a, 2).unwrap();
        assert_orthogonal(&u, 1e-10, "householder_fallback");
    }

    #[test]
    fn householder_fallback_matches_exp_direction() {
        // The high-rank fallback must also honor the exp(A) contract.
        let a = make_antisym(4, 0.05);
        let fallback = householder_geodesic(&a, 2).unwrap();
        let taylor = taylor_exp_antisym(&a, 12);
        let diff = (&fallback - &taylor).mapv(|x| x * x).sum().sqrt();
        assert!(
            diff < 1e-3,
            "householder fallback deviates from exp(A) by {diff:.2e}"
        );
    }

    // ——————————————————————————————————————————————
    // matrix_exp_antisym (dispatch)
    // ——————————————————————————————————————————————

    #[test]
    fn dispatch_taylor_produces_orthogonal() {
        let a = make_antisym(4, 0.3);
        let u = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        assert_orthogonal(&u, 1e-6, "dispatch_taylor");
    }

    #[test]
    fn dispatch_pade_produces_orthogonal() {
        let a = make_antisym(4, 0.3);
        let u = matrix_exp_antisym(&a, OrthoMethod::Pade).unwrap();
        assert_orthogonal(&u, 1e-6, "dispatch_pade");
    }

    #[test]
    fn dispatch_cayley_produces_orthogonal() {
        let a = make_antisym(4, 0.3);
        let u = matrix_exp_antisym(&a, OrthoMethod::Cayley).unwrap();
        assert_orthogonal(&u, 1e-10, "dispatch_cayley");
    }

    #[test]
    fn dispatch_cayley_matches_exp_direction_on_rotation() {
        // Regression: the Cayley dispatch must honor the exp(A) contract.
        // The unscaled raw form would rotate by -2θ (gradient ascent at 2x
        // when reached via scaled_matrix_exp_antisym(A, -lr, Cayley)).
        let theta = 0.3_f64;
        let a = array![[0.0, theta], [-theta, 0.0]];
        let u = matrix_exp_antisym(&a, OrthoMethod::Cayley).unwrap();
        assert!((u[[0, 0]] - theta.cos()).abs() < 5e-3, "cos mismatch");
        assert!((u[[0, 1]] - theta.sin()).abs() < 5e-3, "sin mismatch");
        assert!(
            u[[0, 1]] > 0.0,
            "dispatch Cayley must rotate in the exp(A) direction"
        );
    }

    #[test]
    fn dispatch_qr_produces_orthogonal() {
        let a = make_antisym(4, 0.1);
        let u = matrix_exp_antisym(&a, OrthoMethod::Qr).unwrap();
        assert_orthogonal(&u, 1e-10, "dispatch_qr");
    }

    #[test]
    fn dispatch_taylor_polar_produces_orthogonal() {
        let a = make_antisym(4, 0.5);
        let u = matrix_exp_antisym(&a, OrthoMethod::TaylorPolar).unwrap();
        assert_orthogonal(&u, 1e-8, "dispatch_taylor_polar");
    }

    #[test]
    fn dispatch_householder_produces_orthogonal() {
        let a = make_antisym(4, 0.3);
        let u = matrix_exp_antisym(&a, OrthoMethod::Householder).unwrap();
        assert_orthogonal(&u, 1e-10, "dispatch_householder");
    }

    #[test]
    fn dispatch_non_square_returns_error() {
        let a = Array2::<f64>::zeros((3, 4));
        let result = matrix_exp_antisym(&a, OrthoMethod::Taylor);
        assert_eq!(result, Err(MatrixExpError::NotSquare { rows: 3, cols: 4 }));
    }

    #[test]
    fn all_methods_agree_on_small_input() {
        let a = make_antisym(3, 0.05);
        let taylor = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        let pade = matrix_exp_antisym(&a, OrthoMethod::Pade).unwrap();
        let tp = matrix_exp_antisym(&a, OrthoMethod::TaylorPolar).unwrap();
        let qr = matrix_exp_antisym(&a, OrthoMethod::Qr).unwrap();
        let cayley = matrix_exp_antisym(&a, OrthoMethod::Cayley).unwrap();

        // For small ||A||, exponential-family methods should produce similar results
        let tol = 1e-3;
        for (a_mat, b_mat, label) in [
            (&taylor, &pade, "taylor vs pade"),
            (&taylor, &tp, "taylor vs taylor_polar"),
            (&taylor, &qr, "taylor vs qr"),
            (&taylor, &cayley, "taylor vs cayley"),
        ] {
            let diff = a_mat - b_mat;
            let norm = diff.mapv(|x| x * x).sum().sqrt();
            assert!(norm < tol, "{label}: diff norm {norm:.2e}");
        }
    }

    // ——————————————————————————————————————————————
    // scaled_matrix_exp_antisym
    // ——————————————————————————————————————————————

    #[test]
    fn scaled_with_zero_scale_is_identity() {
        let a = make_antisym(3, 1.0);
        let u = scaled_matrix_exp_antisym(&a, 0.0, OrthoMethod::Taylor).unwrap();
        let eye = Array2::<f64>::eye(3);
        for (uv, ev) in u.iter().zip(eye.iter()) {
            assert!((uv - ev).abs() < 1e-10, "exp(0*A) should be I");
        }
    }

    #[test]
    fn scaled_with_unit_scale_matches_unscaled() {
        let a = make_antisym(3, 0.2);
        let u1 = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        let u2 = scaled_matrix_exp_antisym(&a, 1.0, OrthoMethod::Taylor).unwrap();
        for (a_val, b_val) in u1.iter().zip(u2.iter()) {
            assert!(
                (a_val - b_val).abs() < 1e-12,
                "scale=1 should match unscaled"
            );
        }
    }

    #[test]
    fn scaled_negative_lr_produces_orthogonal() {
        let a = make_antisym(4, 1.0);
        let u = scaled_matrix_exp_antisym(&a, -0.01, OrthoMethod::Taylor).unwrap();
        assert_orthogonal(&u, 1e-8, "scaled_neg_lr");
    }

    // ——————————————————————————————————————————————
    // retract_to_orthogonal
    // ——————————————————————————————————————————————

    #[test]
    fn retract_qr_of_identity_is_identity() {
        let eye = Array2::<f64>::eye(4);
        let q = retract_to_orthogonal(&eye, RetractionMethod::Qr);
        assert_orthogonal(&q, 1e-12, "retract_qr_identity");
        for i in 0..4 {
            assert!((q[[i, i]] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn retract_svd_of_identity_is_identity() {
        let eye = Array2::<f64>::eye(4);
        let q = retract_to_orthogonal(&eye, RetractionMethod::Svd);
        assert_orthogonal(&q, 1e-12, "retract_svd_identity");
    }

    #[test]
    fn retract_polar_of_identity_is_identity() {
        let eye = Array2::<f64>::eye(4);
        let q = retract_to_orthogonal(&eye, RetractionMethod::Polar);
        assert_orthogonal(&q, 1e-12, "retract_polar_identity");
    }

    #[test]
    fn retract_qr_fixes_drifted_matrix() {
        // Start with rotation, add small noise
        let theta = 0.5_f64;
        let mut u = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
        u[[0, 0]] += 0.05; // drift
        u[[1, 1]] += 0.03;
        assert!(orthogonality_error(&u) > 0.01, "Should have drift");

        let q = retract_to_orthogonal(&u, RetractionMethod::Qr);
        assert_orthogonal(&q, 1e-10, "retract_qr_drifted");
    }

    #[test]
    fn retract_svd_fixes_drifted_matrix() {
        let mut u = Array2::<f64>::eye(3);
        u[[0, 0]] = 1.1;
        u[[1, 1]] = 0.9;
        let q = retract_to_orthogonal(&u, RetractionMethod::Svd);
        assert_orthogonal(&q, 1e-10, "retract_svd_drifted");
    }

    #[test]
    fn retract_all_methods_produce_orthogonal() {
        let mut u = Array2::<f64>::eye(4);
        u[[0, 1]] = 0.1;
        u[[2, 3]] = -0.05;

        for method in [
            RetractionMethod::Qr,
            RetractionMethod::Svd,
            RetractionMethod::Polar,
        ] {
            let q = retract_to_orthogonal(&u, method);
            assert_orthogonal(&q, 1e-9, &format!("retract_{method:?}"));
        }
    }

    // ——————————————————————————————————————————————
    // Integration: round-trip correctness
    // ——————————————————————————————————————————————

    #[test]
    fn exp_then_retract_stays_orthogonal() {
        let a = make_antisym(4, 0.5);
        let u = matrix_exp_antisym(&a, OrthoMethod::Taylor).unwrap();
        assert_orthogonal(&u, 1e-6, "before_retract");

        let q = retract_to_orthogonal(&u, RetractionMethod::Svd);
        assert_orthogonal(&q, 1e-10, "after_retract");
    }

    #[test]
    fn multiple_scaled_steps_stay_orthogonal() {
        let a = make_antisym(4, 1.0);
        let lr = -0.01;

        let mut u = Array2::<f64>::eye(4);
        for _ in 0..10 {
            let step = scaled_matrix_exp_antisym(&a, lr, OrthoMethod::Taylor).unwrap();
            u = step.dot(&u);
        }
        // After 10 steps, U should still be approximately orthogonal
        assert_orthogonal(&u, 1e-4, "10_steps");

        // Retraction cleans it up
        let q = retract_to_orthogonal(&u, RetractionMethod::Qr);
        assert_orthogonal(&q, 1e-10, "10_steps_retracted");
    }

    #[test]
    fn column_orthonormality_error_is_zero_for_a_tall_orthonormal_frame() {
        let mut q = Array2::<f64>::zeros((6, 3));
        q[[0, 0]] = 1.0;
        q[[1, 1]] = 1.0;
        q[[2, 2]] = 1.0;
        assert!(orthogonality_error_columns(&q) < 1e-15);
    }

    #[test]
    fn row_orthogonality_error_of_a_tall_frame_is_the_projector_deficit() {
        let mut q = Array2::<f64>::zeros((6, 3));
        q[[0, 0]] = 1.0;
        q[[1, 1]] = 1.0;
        q[[2, 2]] = 1.0;
        let expected = ((6 - 3) as f64).sqrt();
        assert!((orthogonality_error(&q) - expected).abs() < 1e-12);
    }

    #[test]
    fn the_two_measures_agree_on_a_square_frame() {
        let n = 4;
        let m = make_antisym(n, 0.3);
        let q = pade_exp_antisym(&m).unwrap();
        let rows = orthogonality_error(&q);
        let columns = orthogonality_error_columns(&q);
        assert!(
            (rows - columns).abs() < 1e-12,
            "rows {rows:e} columns {columns:e}"
        );
    }

    #[test]
    fn column_orthonormality_error_is_small_for_a_tall_orthonormalized_frame() {
        use crate::geometry::qr::qr_orthonormalize;

        let raw = Array2::from_shape_fn((784, 32), |(row, col)| {
            let mixed = (row as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((col as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
            ((mixed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        });
        let q = qr_orthonormalize(&raw, &QrConfig::default());
        assert_eq!(q.dim(), (784, 32));
        assert!(orthogonality_error_columns(&q) < 1e-12);
    }

    #[test]
    fn column_orthonormality_error_detects_a_perturbed_column() {
        let mut q = Array2::<f64>::zeros((6, 3));
        q[[0, 0]] = 1.0;
        q[[1, 1]] = 1.0;
        q[[2, 2]] = 1.0;
        let col0 = q.column(0).to_owned();
        let mut col1 = q.column_mut(1);
        col1 += &(1e-6 * &col0);
        let err = orthogonality_error_columns(&q);
        assert!(err > 9e-7, "perturbation not detected: {err:e}");
        assert!(err < 3e-6, "unexpected magnitude: {err:e}");
    }
}
