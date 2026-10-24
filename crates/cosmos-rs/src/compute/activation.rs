//! Pointwise nonlinearities available to the compute kernels.
//!
//! These enums live below the arrangement layer because
//! [`ComputeDevice::apply_activation`](crate::compute::ComputeDevice::apply_activation)
//! dispatches on them. The `arrangement::activations` module re-exports both.

use std::f64::consts::SQRT_2;
use std::fmt;
use std::str::FromStr;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Raised when a string does not name a known activation or placement.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unknown {primitive}: {name}")]
pub struct UnknownActivation {
    /// Which primitive was being parsed, `"activation"` or
    /// `"activation_placement"`.
    pub primitive: &'static str,
    /// The unrecognized name, as supplied.
    pub name: String,
}

/// Where to apply an activation relative to a Grassmann projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationPlacement {
    /// No activation applied.
    None,
    /// Apply after full projection: sigma(x W W^T).
    Post,
    /// Apply in subspace coordinates: sigma(x W) W^T.
    Subspace,
}

impl fmt::Display for ActivationPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Post => write!(f, "post"),
            Self::Subspace => write!(f, "subspace"),
        }
    }
}

impl FromStr for ActivationPlacement {
    type Err = UnknownActivation;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "post" => Ok(Self::Post),
            "subspace" => Ok(Self::Subspace),
            _ => Err(UnknownActivation {
                primitive: "activation_placement",
                name: s.to_string(),
            }),
        }
    }
}

/// Built-in activation functions for arrangement compositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    /// No-op activation.
    Identity,

    /// Rectified linear unit.
    Relu,

    /// Gaussian error linear unit.
    Gelu,

    /// Sigmoid linear unit, also known as Swish.
    Silu,

    /// Hyperbolic tangent.
    Tanh,

    /// Logistic sigmoid.
    Sigmoid,

    /// Softplus: ln(1 + e^x).
    Softplus,

    /// Leaky ReLU with slope 0.01 for negative inputs.
    LeakyRelu,

    /// ELU with alpha = 1.0.
    Elu,
}

impl Activation {
    /// Apply this activation element-wise to a scalar.
    #[must_use]
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Relu => x.max(0.0),
            Self::Gelu => gelu(x),
            Self::Silu => x * sigmoid(x),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => sigmoid(x),
            Self::Softplus => softplus(x),
            Self::LeakyRelu => {
                if x >= 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Self::Elu => {
                if x >= 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
        }
    }

    /// Compute the derivative of this activation at a given input value.
    ///
    /// Returns `d(activation(x))/dx` evaluated at `x`.
    #[must_use]
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Relu => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Gelu => {
                let cdf = 0.5 * (1.0 + erf_approx(x / SQRT_2));
                let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
                cdf + x * pdf
            }
            Self::Silu => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Self::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            Self::Softplus => sigmoid(x),
            Self::LeakyRelu => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Self::Elu => {
                if x >= 0.0 {
                    1.0
                } else {
                    x.exp()
                }
            }
        }
    }

    /// Apply this activation element-wise to a 2D array.
    #[must_use]
    pub fn apply_array2(&self, arr: &Array2<f64>) -> Array2<f64> {
        arr.mapv(|x| self.apply(x))
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => write!(f, "none"),
            Self::Relu => write!(f, "relu"),
            Self::Gelu => write!(f, "gelu"),
            Self::Silu => write!(f, "silu"),
            Self::Tanh => write!(f, "tanh"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Softplus => write!(f, "softplus"),
            Self::LeakyRelu => write!(f, "leaky_relu"),
            Self::Elu => write!(f, "elu"),
        }
    }
}

impl FromStr for Activation {
    type Err = UnknownActivation;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "identity" | "none" => Ok(Self::Identity),
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "silu" | "swish" => Ok(Self::Silu),
            "tanh" => Ok(Self::Tanh),
            "sigmoid" => Ok(Self::Sigmoid),
            "softplus" => Ok(Self::Softplus),
            "leaky_relu" | "leakyrelu" => Ok(Self::LeakyRelu),
            "elu" => Ok(Self::Elu),
            _ => Err(UnknownActivation {
                primitive: "activation",
                name: s.to_string(),
            }),
        }
    }
}

fn gelu(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf_approx(x / SQRT_2))
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Abramowitz and Stegun 7.1.26 approximation for erf(x).
///
/// Max error is approximately 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + P * x_abs);

    let poly = (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-x_abs * x_abs).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    use ndarray::array;

    const EPSILON: f64 = 1e-10;
    const GELU_EPSILON: f64 = 1e-6;

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

    #[test]
    fn test_activation_placement_display() {
        assert_eq!(ActivationPlacement::None.to_string(), "none");
        assert_eq!(ActivationPlacement::Post.to_string(), "post");
        assert_eq!(ActivationPlacement::Subspace.to_string(), "subspace");
    }

    #[test]
    fn test_activation_placement_from_str() {
        assert_eq!(
            "none".parse::<ActivationPlacement>().unwrap(),
            ActivationPlacement::None
        );
        assert_eq!(
            "POST".parse::<ActivationPlacement>().unwrap(),
            ActivationPlacement::Post
        );
        assert_eq!(
            "subspace".parse::<ActivationPlacement>().unwrap(),
            ActivationPlacement::Subspace
        );
    }

    #[test]
    fn test_activation_placement_from_str_invalid() {
        assert!("before".parse::<ActivationPlacement>().is_err());
    }

    #[test]
    fn test_activation_display_roundtrip() {
        for activation in [
            Activation::Identity,
            Activation::Relu,
            Activation::Gelu,
            Activation::Silu,
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::Softplus,
            Activation::LeakyRelu,
            Activation::Elu,
        ] {
            let s = activation.to_string();
            assert_eq!(s.parse::<Activation>().unwrap(), activation);
        }
    }

    #[test]
    fn test_activation_from_str_valid_variants() {
        let cases = [
            ("identity", Activation::Identity),
            ("none", Activation::Identity),
            ("relu", Activation::Relu),
            ("gelu", Activation::Gelu),
            ("SWISH", Activation::Silu),
            ("tanh", Activation::Tanh),
            ("sigmoid", Activation::Sigmoid),
            ("softplus", Activation::Softplus),
            ("leaky_relu", Activation::LeakyRelu),
            ("LEAKYRELU", Activation::LeakyRelu),
            ("elu", Activation::Elu),
        ];

        for (name, expected) in cases {
            assert_eq!(name.parse::<Activation>().unwrap(), expected);
        }
    }

    #[test]
    fn test_activation_from_str_invalid() {
        assert!("mish".parse::<Activation>().is_err());
    }

    #[test]
    fn test_swish_alias_maps_to_silu() {
        assert_eq!("swish".parse::<Activation>().unwrap(), Activation::Silu);
    }

    #[test]
    fn test_leaky_relu_alias_without_underscore() {
        assert_eq!(
            "leakyrelu".parse::<Activation>().unwrap(),
            Activation::LeakyRelu
        );
    }

    #[test]
    fn test_activation_apply_identity() {
        assert_close(Activation::Identity.apply(-1.25), -1.25, EPSILON);
    }

    #[test]
    fn test_activation_apply_relu() {
        assert_close(Activation::Relu.apply(-2.0), 0.0, EPSILON);
        assert_close(Activation::Relu.apply(1.5), 1.5, EPSILON);
    }

    #[test]
    fn test_activation_apply_gelu_zero() {
        assert_close(Activation::Gelu.apply(0.0), 0.0, EPSILON);
    }

    #[test]
    fn test_activation_apply_gelu_known_value() {
        assert_close(
            Activation::Gelu.apply(1.0),
            0.841_344_746_068_542_9,
            GELU_EPSILON,
        );
    }

    #[test]
    fn test_activation_apply_silu() {
        assert_close(
            Activation::Silu.apply(1.0),
            0.731_058_578_630_004_9,
            EPSILON,
        );
    }

    #[test]
    fn test_activation_apply_tanh() {
        assert_close(
            Activation::Tanh.apply(1.0),
            0.761_594_155_955_764_9,
            EPSILON,
        );
    }

    #[test]
    fn test_activation_apply_sigmoid_zero() {
        assert_close(Activation::Sigmoid.apply(0.0), 0.5, EPSILON);
    }

    #[test]
    fn test_activation_apply_softplus_zero() {
        assert_close(Activation::Softplus.apply(0.0), LN_2, EPSILON);
    }

    #[test]
    fn test_activation_apply_leaky_relu() {
        assert_close(Activation::LeakyRelu.apply(-2.0), -0.02, EPSILON);
        assert_close(Activation::LeakyRelu.apply(3.0), 3.0, EPSILON);
    }

    #[test]
    fn test_activation_apply_elu() {
        assert_close(Activation::Elu.apply(-1.0), (-1.0f64).exp() - 1.0, EPSILON);
        assert_close(Activation::Elu.apply(2.0), 2.0, EPSILON);
    }

    #[test]
    fn test_apply_array2_preserves_shape() {
        let x = array![[-1.0, 0.0], [2.0, -3.0]];
        let out = Activation::Sigmoid.apply_array2(&x);
        assert_eq!(out.raw_dim(), x.raw_dim());
    }

    #[test]
    fn test_apply_array2_values() {
        let x = array![[-1.0, 0.0], [1.0, 2.0]];
        let out = Activation::Sigmoid.apply_array2(&x);
        let expected = array![
            [0.268_941_421_369_995_1, 0.5],
            [0.731_058_578_630_004_9, 0.880_797_077_977_882_3]
        ];
        assert_array2_close(&out, &expected, EPSILON);
    }

    #[test]
    fn test_activation_serde_roundtrip() {
        let json = serde_json::to_string(&Activation::Gelu).unwrap();
        let restored: Activation = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, Activation::Gelu);
    }

    #[test]
    fn test_activation_placement_serde_roundtrip() {
        let json = serde_json::to_string(&ActivationPlacement::Subspace).unwrap();
        let restored: ActivationPlacement = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, ActivationPlacement::Subspace);
    }
}
