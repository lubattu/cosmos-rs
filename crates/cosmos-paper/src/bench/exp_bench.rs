//! Matrix-exponential cost measurements.
//!
//! Premises:
//! - Every generator is antisymmetric and rescaled so its Frobenius norm is
//!   exactly the configured target, so the cost comparison is at fixed
//!   generator magnitude.
//! - Each measured pair runs one untimed warm-up call before the timed
//!   repetitions, so a single-repetition size is not biased by first-call
//!   cache effects.
//! - Timings are single-threaded: the crate performs its own linear algebra
//!   and starts no worker pool.

use std::time::Instant;

use cosmos_rs::geometry::matrix_exp::{
    OrthoMethod, antisymmetrize, cayley_transform, matrix_exp_antisym, orthogonality_error,
};
use cosmos_rs::geometry::qr::{QrConfig, qr_orthonormalize};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// One measured `(method, size)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodMeasurement {
    /// Method name.
    pub method: String,
    /// Ambient dimension.
    pub n: usize,
    /// Timed repetitions.
    pub reps: usize,
    /// Median wall-clock seconds.
    pub median_s: f64,
    /// Fastest wall-clock seconds.
    pub min_s: f64,
    /// Orthogonality error of the produced matrix.
    pub orthogonality_error: f64,
    /// Frobenius deviation from the reference method's result.
    pub frob_dev_from_pade: Option<f64>,
}

/// One measured per-block orthonormalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockMeasurement {
    /// Ambient dimension.
    pub n: usize,
    /// Block width.
    pub k: usize,
    /// Timed repetitions.
    pub reps: usize,
    /// Median wall-clock seconds.
    pub median_s: f64,
}

/// The complete measurement set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchResults {
    /// Frobenius norm every generator is scaled to.
    pub target_norm: f64,
    /// Ambient dimensions measured.
    pub sizes: Vec<usize>,
    /// Per-method measurements.
    pub methods: Vec<MethodMeasurement>,
    /// Per-block orthonormalization measurements.
    pub block_qr: Vec<BlockMeasurement>,
    /// Fitted log-log slopes, one per series.
    pub slopes: Vec<(String, f64)>,
}

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn uniform_signed(seed: u64, row: usize, col: usize) -> f64 {
    let bits = splitmix64(seed ^ splitmix64((row as u64) << 32 | col as u64));
    2.0 * (((bits >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)) - 1.0
}

/// A deterministic antisymmetric generator of exactly `target_norm`.
pub fn generator(n: usize, seed: u64, target_norm: f64) -> Array2<f64> {
    let raw = Array2::from_shape_fn((n, n), |(row, col)| uniform_signed(seed, row, col));
    let mut a = antisymmetrize(&raw);
    let norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        a *= target_norm / norm;
    }
    a
}

/// A deterministic `(n, k)` frame.
pub fn frame(n: usize, k: usize, seed: u64) -> Array2<f64> {
    Array2::from_shape_fn((n, k), |(row, col)| uniform_signed(seed, row, col))
}

fn reps_for(n: usize) -> usize {
    match n {
        0..=1024 => 5,
        1025..=2048 => 3,
        _ => 1,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn frobenius_difference(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn apply(method: &str, a: &Array2<f64>) -> Array2<f64> {
    match method {
        "pade" => matrix_exp_antisym(a, OrthoMethod::Pade).expect("square generator"),
        "taylor" => matrix_exp_antisym(a, OrthoMethod::Taylor).expect("square generator"),
        "qr" => matrix_exp_antisym(a, OrthoMethod::Qr).expect("square generator"),
        "cayley" => cayley_transform(a).expect("invertible generator"),
        other => panic!("unknown method {other}"),
    }
}

/// Measure every `(method, size)` pair and the per-block orthonormalization.
///
/// # Panics
///
/// Panics if an unknown method name is given, or if a method fails on a
/// well-formed antisymmetric generator.
pub fn measure(
    sizes: &[usize],
    methods: &[String],
    target_norm: f64,
    block_k: usize,
) -> BenchResults {
    let mut measurements = Vec::new();
    let mut block_qr = Vec::new();

    for &n in sizes {
        let a = generator(n, 42 + n as u64, target_norm);
        let scaled = &a * -0.5;
        let reference = apply("pade", &a);

        for method in methods {
            let input = if method == "cayley" { &scaled } else { &a };
            let _ = apply(method, input);

            let reps = reps_for(n);
            let mut timings = Vec::with_capacity(reps);
            let mut last = None;
            for _ in 0..reps {
                let start = Instant::now();
                let value = apply(method, input);
                timings.push(start.elapsed().as_secs_f64());
                last = Some(value);
            }
            let produced = last.expect("at least one repetition");
            let deviation = if method == "pade" {
                None
            } else {
                Some(frobenius_difference(&produced, &reference))
            };
            measurements.push(MethodMeasurement {
                method: method.clone(),
                n,
                reps,
                median_s: median(timings.clone()),
                min_s: timings.iter().copied().fold(f64::INFINITY, f64::min),
                orthogonality_error: orthogonality_error(&produced),
                frob_dev_from_pade: deviation,
            });
        }

        let raw_frame = frame(n, block_k, 4242 + n as u64);
        let qr_config = QrConfig::default();
        let _ = qr_orthonormalize(&raw_frame, &qr_config);
        let reps = reps_for(n).max(3);
        let mut timings = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let _ = qr_orthonormalize(&raw_frame, &qr_config);
            timings.push(start.elapsed().as_secs_f64());
        }
        block_qr.push(BlockMeasurement {
            n,
            k: block_k,
            reps,
            median_s: median(timings),
        });
    }

    let mut slopes = Vec::new();
    for method in methods {
        let points: Vec<(f64, f64)> = measurements
            .iter()
            .filter(|entry| &entry.method == method && entry.median_s > 0.0)
            .map(|entry| ((entry.n as f64).ln(), entry.median_s.ln()))
            .collect();
        slopes.push((method.clone(), log_log_slope(&points)));
    }
    let block_points: Vec<(f64, f64)> = block_qr
        .iter()
        .filter(|entry| entry.median_s > 0.0)
        .map(|entry| ((entry.n as f64).ln(), entry.median_s.ln()))
        .collect();
    slopes.push(("block_qr".to_string(), log_log_slope(&block_points)));

    BenchResults {
        target_norm,
        sizes: sizes.to_vec(),
        methods: measurements,
        block_qr,
        slopes,
    }
}

fn log_log_slope(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return f64::NAN;
    }
    let count = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / count;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / count;
    let numerator: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let denominator: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_antisymmetric_at_the_target_norm() {
        let a = generator(16, 58, 1.2);
        let norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.2).abs() < 1e-12);
        for row in 0..16 {
            for col in 0..16 {
                assert!((a[[row, col]] + a[[col, row]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn every_method_produces_an_orthogonal_matrix() {
        let methods: Vec<String> = ["pade", "taylor", "cayley", "qr"]
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let results = measure(&[32, 64], &methods, 1.2, 8);
        for entry in &results.methods {
            assert!(
                entry.orthogonality_error < 1e-11,
                "{} at n={} had error {}",
                entry.method,
                entry.n,
                entry.orthogonality_error
            );
        }
        assert_eq!(results.block_qr.len(), 2);
        assert_eq!(results.slopes.len(), 5);
    }

    #[test]
    fn reference_method_has_no_deviation_entry() {
        let methods = vec!["pade".to_string(), "qr".to_string()];
        let results = measure(&[16], &methods, 1.2, 8);
        let pade = results
            .methods
            .iter()
            .find(|entry| entry.method == "pade")
            .expect("pade row");
        assert!(pade.frob_dev_from_pade.is_none());
        let qr = results
            .methods
            .iter()
            .find(|entry| entry.method == "qr")
            .expect("qr row");
        assert!(qr.frob_dev_from_pade.expect("deviation") >= 0.0);
    }

    #[test]
    fn slope_recovers_a_known_exponent() {
        let points: Vec<(f64, f64)> = [1.0f64, 2.0, 4.0, 8.0]
            .iter()
            .map(|n| (n.ln(), 3.0 * n.ln()))
            .collect();
        assert!((log_log_slope(&points) - 3.0).abs() < 1e-12);
    }
}
