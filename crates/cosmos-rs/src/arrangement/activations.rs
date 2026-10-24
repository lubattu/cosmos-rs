//! Activation placement helpers for arrangement compositions.
//!
//! The nonlinearities themselves live in [`crate::compute::activation`], below
//! this layer, because the compute device dispatches on them. This module
//! re-exports them and adds the projection helpers that place an activation
//! relative to a Grassmann projection.

use ndarray::Array2;

use super::ArrangementError;
use super::params::SubspaceAction;
use crate::compute::activation::UnknownActivation;

pub use crate::compute::activation::{Activation, ActivationPlacement};

impl From<UnknownActivation> for ArrangementError {
    fn from(err: UnknownActivation) -> Self {
        ArrangementError::UnknownStrategy {
            primitive: err.primitive,
            name: err.name,
        }
    }
}

/// Apply a single projection with optional activation placement.
///
/// Assumes:
///
/// - `w` has shape `(n, k)`
/// - `x` has shape `(batch, n)`
///
/// For row-major batches this computes:
///
/// - `ActivationPlacement::None`: `x W W^T`
/// - `ActivationPlacement::Post`: `sigma(x W W^T)`
/// - `ActivationPlacement::Subspace`: `sigma(x W) W^T`
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use cosmos_rs::arrangement::activations::{
///     project_with_activation, Activation, ActivationPlacement,
/// };
///
/// let w = array![[1.0], [0.0]];
/// let x = array![[2.0, -3.0]];
/// let y = project_with_activation(&w, &x, Activation::Relu, ActivationPlacement::Post);
/// assert_eq!(y.shape(), &[1, 2]);
/// ```
pub fn project_with_activation(
    w: &Array2<f64>,
    x: &Array2<f64>,
    activation: Activation,
    placement: ActivationPlacement,
) -> Array2<f64> {
    let subspace = x.dot(w);

    match placement {
        ActivationPlacement::None => subspace.dot(&w.t()),
        ActivationPlacement::Post => {
            let projected = subspace.dot(&w.t());
            if activation == Activation::Identity {
                projected
            } else {
                activation.apply_array2(&projected)
            }
        }
        ActivationPlacement::Subspace => {
            if activation == Activation::Identity {
                subspace.dot(&w.t())
            } else {
                activation.apply_array2(&subspace).dot(&w.t())
            }
        }
    }
}

/// Apply a gauge-invariant subspace action with optional activation placement.
///
/// Computes `T(x) = d·x + c·f_σ(x)` where `c = α − β` (action coefficient),
/// `d = β` (identity coefficient), and `f_σ` is the placement-dependent projection:
///
/// - `None`:     `f_σ(x) = x W Wᵀ`
/// - `Post`:     `f_σ(x) = σ(x W Wᵀ)`
/// - `Subspace`: `f_σ(x) = σ(x W) Wᵀ`
///
/// For Post placement, the activation wraps the full linear combination:
/// `T(x) = σ(d·x + c·x W Wᵀ)`.
pub fn apply_subspace_action(
    w: &Array2<f64>,
    x: &Array2<f64>,
    activation: Activation,
    placement: ActivationPlacement,
    action: SubspaceAction,
) -> Array2<f64> {
    let c = action.action_coeff();
    let d = action.identity_coeff();

    if c.abs() < 1e-15 {
        return x * d;
    }

    match placement {
        ActivationPlacement::None => {
            let projected = x.dot(w).dot(&w.t());
            &projected * c + x * d
        }
        ActivationPlacement::Post => {
            let projected = x.dot(w).dot(&w.t());
            let pre_act = &projected * c + x * d;
            if activation == Activation::Identity {
                pre_act
            } else {
                activation.apply_array2(&pre_act)
            }
        }
        ActivationPlacement::Subspace => {
            let subspace = x.dot(w);
            let activated = if activation == Activation::Identity {
                subspace
            } else {
                activation.apply_array2(&subspace)
            };
            &activated.dot(&w.t()) * c + x * d
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual}, diff {diff} > {tol}"
        );
    }

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, col), value) in actual.indexed_iter() {
            let expected_value = expected[(row, col)];
            let diff = (*value - expected_value).abs();
            assert!(
                diff <= tol,
                "entry ({row}, {col}): expected {expected_value}, got {value}, diff {diff} > {tol}"
            );
        }
    }

    fn mixed_sign_basis() -> Array2<f64> {
        let inv_sqrt2 = 1.0 / SQRT_2;
        array![[inv_sqrt2], [-inv_sqrt2]]
    }

    fn orthonormal_basis_2d_in_4d() -> Array2<f64> {
        let s = 1.0 / SQRT_2;
        array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]]
    }

    #[test]
    fn unknown_activation_converts_to_unknown_strategy() {
        let err: ArrangementError = "not_an_activation"
            .parse::<Activation>()
            .unwrap_err()
            .into();
        assert!(matches!(
            err,
            ArrangementError::UnknownStrategy {
                primitive: "activation",
                ..
            }
        ));

        let err: ArrangementError = "not_a_placement"
            .parse::<ActivationPlacement>()
            .unwrap_err()
            .into();
        assert!(matches!(
            err,
            ArrangementError::UnknownStrategy {
                primitive: "activation_placement",
                ..
            }
        ));
    }

    #[test]
    fn test_project_with_activation_none() {
        let w = array![[1.0], [0.0]];
        let x = array![[2.0, -3.0], [4.0, 5.0]];
        let out = project_with_activation(&w, &x, Activation::Relu, ActivationPlacement::None);
        let expected = array![[2.0, 0.0], [4.0, 0.0]];
        assert_array2_close(&out, &expected, EPSILON);
    }

    #[test]
    fn test_project_with_activation_none_ignores_activation() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [-1.0, 1.0]];

        let linear =
            project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::None);
        let relu_none =
            project_with_activation(&w, &x, Activation::Relu, ActivationPlacement::None);

        assert_array2_close(&linear, &relu_none, EPSILON);
    }

    #[test]
    fn test_project_with_activation_post_relu() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [-1.0, 1.0]];
        let out = project_with_activation(&w, &x, Activation::Relu, ActivationPlacement::Post);
        let expected = array![[1.0, 0.0], [0.0, 1.0]];
        assert_array2_close(&out, &expected, EPSILON);
    }

    #[test]
    fn test_project_with_activation_subspace_relu() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [-1.0, 1.0]];
        let out = project_with_activation(&w, &x, Activation::Relu, ActivationPlacement::Subspace);
        let expected = array![[1.0, -1.0], [0.0, 0.0]];
        assert_array2_close(&out, &expected, EPSILON);
    }

    #[test]
    fn test_project_with_identity_post_matches_linear() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [-1.0, 1.0]];

        let linear =
            project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::None);
        let post = project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::Post);

        assert_array2_close(&linear, &post, EPSILON);
    }

    #[test]
    fn test_project_with_identity_subspace_matches_linear() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [-1.0, 1.0]];

        let linear =
            project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::None);
        let subspace =
            project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::Subspace);

        assert_array2_close(&linear, &subspace, EPSILON);
    }

    #[test]
    fn test_project_with_activation_preserves_batch_shape() {
        let w = mixed_sign_basis();
        let x = array![[1.0, -1.0], [0.5, 0.5], [-2.0, 2.0]];
        let out = project_with_activation(&w, &x, Activation::Tanh, ActivationPlacement::Post);
        assert_eq!(out.dim(), x.dim());
    }

    #[test]
    fn reflection_none_preserves_norm() {
        let w = orthonormal_basis_2d_in_4d();
        let x = array![[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, -0.5, 2.0]];
        let out = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );

        for i in 0..x.nrows() {
            let input_norm = x.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            let output_norm = out.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_close(output_norm, input_norm, 1e-10);
        }
    }

    #[test]
    fn reflection_none_is_involution() {
        let w = mixed_sign_basis();
        let x = array![[3.0, -7.0], [1.5, 2.5]];
        let r1 = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );
        let r2 = apply_subspace_action(
            &w,
            &r1,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&r2, &x, EPSILON);
    }

    #[test]
    fn reflection_none_equals_2p_minus_x() {
        let w = mixed_sign_basis();
        let x = array![[3.0, -7.0], [1.5, 2.5]];
        let proj = project_with_activation(&w, &x, Activation::Identity, ActivationPlacement::None);
        let expected = &proj * 2.0 - &x;
        let reflected = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&reflected, &expected, EPSILON);
    }

    #[test]
    fn reflection_post_identity_equals_none() {
        let w = orthonormal_basis_2d_in_4d();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let none = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );
        let post = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::Post,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&none, &post, EPSILON);
    }

    #[test]
    fn reflection_subspace_identity_equals_none() {
        let w = orthonormal_basis_2d_in_4d();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let none = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::None,
            SubspaceAction::reflection(),
        );
        let subspace = apply_subspace_action(
            &w,
            &x,
            Activation::Identity,
            ActivationPlacement::Subspace,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&none, &subspace, EPSILON);
    }

    #[test]
    fn reflection_post_applies_activation_to_reflected() {
        let w = array![[1.0], [0.0]];
        let x = array![[-1.0, 3.0]];
        let out = apply_subspace_action(
            &w,
            &x,
            Activation::Relu,
            ActivationPlacement::Post,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&out, &array![[0.0, 0.0]], EPSILON);
    }

    #[test]
    fn reflection_subspace_applies_activation_to_coords() {
        let w = array![[1.0], [0.0]];
        let x = array![[-1.0, 3.0]];
        let out = apply_subspace_action(
            &w,
            &x,
            Activation::Relu,
            ActivationPlacement::Subspace,
            SubspaceAction::reflection(),
        );
        assert_array2_close(&out, &array![[1.0, -3.0]], EPSILON);
    }

    #[test]
    fn projection_action_matches_raw_project_with_activation() {
        let w = orthonormal_basis_2d_in_4d();
        let x = array![[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, -0.5, 2.0]];
        for placement in [
            ActivationPlacement::None,
            ActivationPlacement::Post,
            ActivationPlacement::Subspace,
        ] {
            let proj = project_with_activation(&w, &x, Activation::Relu, placement);
            let proj_via_mode = apply_subspace_action(
                &w,
                &x,
                Activation::Relu,
                placement,
                SubspaceAction::projection(),
            );
            assert_array2_close(&proj, &proj_via_mode, EPSILON);
        }
    }
}
