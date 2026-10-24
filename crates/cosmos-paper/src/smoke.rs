//! Reference expectations and tolerance checking for shortened runs.
//!
//! Premises:
//! - A shortened run executes the same objective, loss, model and rule as the
//!   full run; only the epoch count and the seed list are reduced.
//! - A value outside tolerance is a defect to investigate. Editing the
//!   reference to make a run pass defeats the purpose of the file.
//! - A quantity that may cross zero and one that is strictly positive need
//!   different treatment near zero: the first is compared against an absolute
//!   floor, the second against a widened relative band, because a relative
//!   band around a signed value near zero is narrower than the arithmetic.

use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Tolerances applied to every comparison.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// Absolute tolerance, in accuracy points, for any accuracy.
    pub accuracy_points: f64,
    /// Absolute upper bound for an orthogonality error.
    pub orthogonality_upper_bound: f64,
    /// Relative tolerance for every other scalar.
    pub relative: f64,
    /// Reference magnitude below which a quantity counts as near zero.
    pub near_zero_reference: f64,
    /// Absolute tolerance applied to a near-zero signed reference.
    pub near_zero_absolute: f64,
    /// Relative tolerance applied to a near-zero strictly positive
    /// reference.
    pub positive_small_relative: f64,
}

/// Whether a compared quantity may cross zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// The quantity may take either sign, so a near-zero reference is
    /// compared against the absolute floor.
    Signed,
    /// The quantity is strictly positive, so a near-zero reference is
    /// compared against the widened relative band.
    Positive,
}

/// One cell's expectations.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellReference {
    /// Cell name, matching the name the runner prints.
    pub name: String,
    /// Cell file the reference belongs to.
    pub config: String,
    /// Engine the cell runs on.
    pub engine: String,
    /// Arm, when the cell has more than one.
    #[serde(default)]
    pub arm: Option<String>,
    /// Seed the shortened run uses.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Epochs the shortened run executes.
    #[serde(default)]
    pub epochs: Option<u32>,
    /// Expected training accuracy per epoch.
    #[serde(default)]
    pub train_acc: Vec<f64>,
    /// Expected validation accuracy per epoch.
    #[serde(default)]
    pub val_acc: Vec<f64>,
    /// Expected test accuracy per epoch.
    #[serde(default)]
    pub test_acc: Vec<f64>,
    /// Expected mean training loss per epoch.
    #[serde(default)]
    pub loss: Vec<f64>,
    /// Expected mean commutator norm per epoch.
    #[serde(default)]
    pub mean_commutator: Vec<f64>,
    /// Expected interior flow ratio per epoch.
    #[serde(default)]
    pub flow_ratio_overall: Vec<f64>,
    /// Expected angular margin per epoch.
    #[serde(default)]
    pub cmr_margin: Vec<f64>,
    /// Upper bound on the aggregate orthogonality error.
    #[serde(default)]
    pub total_orth_error_max: Option<f64>,
    /// Upper bound on validation accuracy, for cells that sit at chance.
    #[serde(default)]
    pub val_acc_upper_bound: Option<f64>,
    /// Expected parameter count.
    #[serde(default)]
    pub param_count: Option<usize>,
}

/// The whole reference file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct References {
    /// Tolerances.
    pub tolerance: Tolerance,
    /// Per-cell expectations.
    #[serde(rename = "cell")]
    pub cells: Vec<CellReference>,
    /// Expectations for the benchmark, which has no per-epoch history.
    #[serde(default)]
    pub bench: Option<BenchReference>,
}

impl References {
    /// Find the entry named `name`.
    pub fn cell(&self, name: &str) -> Option<&CellReference> {
        self.cells.iter().find(|entry| entry.name == name)
    }
}

/// Expectations for the benchmark, whose measurements are structural rather
/// than per-epoch: the primitives must stay orthogonal, every measured series
/// must yield a fitted slope, and the per-block orthonormalization must not
/// get cheaper as the ambient dimension grows.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchReference {
    /// Entry name, matching the name the runner prints.
    pub name: String,
    /// Benchmark cell file the reference belongs to.
    pub config: String,
    /// Engine the entry describes.
    pub engine: String,
    /// Upper bound on the orthogonality error of every method's output.
    pub orthogonality_upper_bound: f64,
    /// Whether a finite slope is required for every measured series.
    pub require_slopes: bool,
    /// Whether the per-block orthonormalization time must not decrease with
    /// the ambient dimension.
    pub require_block_qr_monotone: bool,
    /// Smallest ratio between two ambient dimensions for their times to be
    /// compared. Adjacent sizes that are too close apart are skipped,
    /// because the difference their cost makes is smaller than the spread
    /// of a wall-clock measurement.
    pub monotone_min_size_ratio: f64,
    /// Fraction by which a compared time may fall below the previous one
    /// before it counts as a failure.
    pub monotone_tolerance: f64,
}

/// What a shortened benchmark run actually produced.
#[derive(Debug, Clone, Default)]
pub struct ObservedBench {
    /// Orthogonality error of each measured `(method, ambient dimension)`.
    pub orthogonality_error: Vec<(String, usize, f64)>,
    /// Fitted slope of each measured series.
    pub slopes: Vec<(String, f64)>,
    /// Median per-block orthonormalization time by ambient dimension.
    pub block_qr_median_s: Vec<(usize, f64)>,
}

/// Compare a shortened benchmark run against its reference.
///
/// Returns one description per expectation that was not met.
#[must_use]
pub fn check_bench(reference: &BenchReference, observed: &ObservedBench) -> Vec<String> {
    let mut out = Vec::new();
    let bound = reference.orthogonality_upper_bound;

    if observed.orthogonality_error.is_empty() {
        out.push("no method measurement was produced".to_string());
    }
    for (method, n, error) in &observed.orthogonality_error {
        if error.is_nan() || *error > bound {
            out.push(format!(
                "orthogonality of {method} at n {n}: observed {error:.3e}, bound {bound:.3e}"
            ));
        }
    }

    if reference.require_slopes {
        if observed.slopes.is_empty() {
            out.push("no slope was fitted".to_string());
        }
        for (series, slope) in &observed.slopes {
            if !slope.is_finite() {
                out.push(format!(
                    "slope for {series}: observed {slope}, expected a finite value"
                ));
            }
        }
    }

    if reference.require_block_qr_monotone {
        let mut sorted = observed.block_qr_median_s.clone();
        sorted.sort_by_key(|(n, _)| *n);
        let floor = 1.0 - reference.monotone_tolerance;
        let mut previous: Option<(usize, f64)> = None;
        for (n, seconds) in sorted {
            let Some((previous_n, previous_s)) = previous else {
                previous = Some((n, seconds));
                continue;
            };
            if (n as f64) < reference.monotone_min_size_ratio * previous_n as f64 {
                continue;
            }
            if seconds < floor * previous_s {
                out.push(format!(
                    "per-block time at n {n}: observed {seconds:.6}, below n {previous_n} at {previous_s:.6}"
                ));
            }
            previous = Some((n, seconds));
        }
    }

    out
}

/// What a shortened run actually produced.
#[derive(Debug, Clone, Default)]
pub struct Observed {
    /// Training accuracy per epoch.
    pub train_acc: Vec<f64>,
    /// Validation accuracy per epoch.
    pub val_acc: Vec<f64>,
    /// Test accuracy per epoch.
    pub test_acc: Vec<f64>,
    /// Mean training loss per epoch.
    pub loss: Vec<f64>,
    /// Mean commutator norm per epoch.
    pub mean_commutator: Vec<f64>,
    /// Interior flow ratio per epoch.
    pub flow_ratio_overall: Vec<f64>,
    /// Angular margin per epoch.
    pub cmr_margin: Vec<f64>,
    /// Aggregate orthogonality error per epoch.
    pub total_orth_error: Vec<f64>,
    /// Parameter count.
    pub param_count: Option<usize>,
}

/// One value that fell outside tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct Deviation {
    /// Quantity that deviated.
    pub quantity: String,
    /// Epoch, one-based, or zero for a run-level quantity.
    pub epoch: usize,
    /// Expected value.
    pub expected: f64,
    /// Observed value.
    pub observed: f64,
    /// Tolerance that was applied.
    pub tolerance: f64,
}

/// Read the reference file.
///
/// # Errors
///
/// Returns an I/O error when the file cannot be read or does not parse.
pub fn load(path: &Path) -> std::io::Result<References> {
    let text = fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

/// Tolerance allowed around `reference` for a quantity of the given sign.
pub fn allowed(tolerance: &Tolerance, reference: f64, sign: Sign) -> f64 {
    let magnitude = reference.abs();
    if magnitude >= tolerance.near_zero_reference {
        return tolerance.relative * magnitude;
    }
    match sign {
        Sign::Signed => tolerance.near_zero_absolute,
        Sign::Positive => tolerance.positive_small_relative * magnitude,
    }
}

fn compare(
    quantity: &str,
    expected: &[f64],
    observed: &[f64],
    allowance: impl Fn(f64) -> f64,
    out: &mut Vec<Deviation>,
) {
    for (index, value) in expected.iter().enumerate() {
        let tolerance = allowance(*value);
        let observed = observed.get(index).copied().unwrap_or(f64::NAN);
        let deviation = (observed - value).abs();
        if deviation.is_nan() || deviation > tolerance {
            out.push(Deviation {
                quantity: quantity.to_string(),
                epoch: index + 1,
                expected: *value,
                observed,
                tolerance,
            });
        }
    }
}

/// Compare a shortened run against its reference.
pub fn check(
    reference: &CellReference,
    tolerance: &Tolerance,
    observed: &Observed,
) -> Vec<Deviation> {
    let mut out = Vec::new();
    let points = tolerance.accuracy_points;
    let accuracy = |_: f64| points;
    compare(
        "train_acc",
        &reference.train_acc,
        &observed.train_acc,
        accuracy,
        &mut out,
    );
    compare(
        "val_acc",
        &reference.val_acc,
        &observed.val_acc,
        accuracy,
        &mut out,
    );
    compare(
        "test_acc",
        &reference.test_acc,
        &observed.test_acc,
        accuracy,
        &mut out,
    );

    let positive = |value: f64| allowed(tolerance, value, Sign::Positive);
    compare("loss", &reference.loss, &observed.loss, positive, &mut out);
    compare(
        "mean_commutator",
        &reference.mean_commutator,
        &observed.mean_commutator,
        positive,
        &mut out,
    );
    compare(
        "flow_ratio_overall",
        &reference.flow_ratio_overall,
        &observed.flow_ratio_overall,
        positive,
        &mut out,
    );
    compare(
        "cmr_margin",
        &reference.cmr_margin,
        &observed.cmr_margin,
        |value| allowed(tolerance, value, Sign::Signed),
        &mut out,
    );

    let bound = reference
        .total_orth_error_max
        .unwrap_or(tolerance.orthogonality_upper_bound);
    for (index, value) in observed.total_orth_error.iter().enumerate() {
        if value.is_nan() || *value > bound {
            out.push(Deviation {
                quantity: "total_orth_error".to_string(),
                epoch: index + 1,
                expected: bound,
                observed: *value,
                tolerance: bound,
            });
        }
    }

    if let Some(limit) = reference.val_acc_upper_bound {
        for (index, value) in observed.val_acc.iter().enumerate() {
            if *value > limit {
                out.push(Deviation {
                    quantity: "val_acc_upper_bound".to_string(),
                    epoch: index + 1,
                    expected: limit,
                    observed: *value,
                    tolerance: 0.0,
                });
            }
        }
    }

    if let (Some(expected), Some(got)) = (reference.param_count, observed.param_count) {
        if expected != got {
            out.push(Deviation {
                quantity: "param_count".to_string(),
                epoch: 0,
                expected: expected as f64,
                observed: got as f64,
                tolerance: 0.0,
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
            .join("smoke_references.toml")
    }

    fn tolerance() -> Tolerance {
        Tolerance {
            accuracy_points: 1.5,
            orthogonality_upper_bound: 1e-12,
            relative: 0.05,
            near_zero_reference: 0.05,
            near_zero_absolute: 0.005,
            positive_small_relative: 0.20,
        }
    }

    #[test]
    fn reference_file_loads_with_every_cell() {
        let references = load(&reference_path()).expect("reference file");
        assert_eq!(references.cells.len(), 8);
        assert!((references.tolerance.accuracy_points - 1.5).abs() < 1e-12);
        assert!((references.tolerance.near_zero_absolute - 0.005).abs() < 1e-12);
        assert!((references.tolerance.positive_small_relative - 0.20).abs() < 1e-12);
    }

    #[test]
    fn every_reference_names_a_shipped_cell_file() {
        let references = load(&reference_path()).expect("reference file");
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        for cell in &references.cells {
            assert!(
                root.join(&cell.config).exists(),
                "{} names a missing cell file",
                cell.name
            );
        }
    }

    #[test]
    fn an_entry_that_names_an_arm_is_keyed_by_that_arm() {
        let references = load(&reference_path()).expect("reference file");
        for cell in &references.cells {
            let Some(arm) = cell.arm.as_deref() else {
                continue;
            };
            assert!(
                cell.name.ends_with(&format!(":{arm}")),
                "{} is keyed without its arm, so the runner would apply it to every arm",
                cell.name
            );
        }
    }

    #[test]
    fn accuracy_inside_the_band_passes() {
        let reference = CellReference {
            val_acc: vec![39.94, 55.20],
            ..CellReference::default()
        };
        let observed = Observed {
            val_acc: vec![41.30, 53.80],
            ..Observed::default()
        };
        assert!(check(&reference, &tolerance(), &observed).is_empty());
    }

    #[test]
    fn accuracy_outside_the_band_is_reported() {
        let reference = CellReference {
            val_acc: vec![39.94],
            ..CellReference::default()
        };
        let observed = Observed {
            val_acc: vec![42.0],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance(), &observed);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "val_acc");
    }

    #[test]
    fn a_large_reference_uses_the_relative_band() {
        let tolerance = tolerance();
        assert!((allowed(&tolerance, 2.0, Sign::Positive) - 0.1).abs() < 1e-12);
        assert!((allowed(&tolerance, -2.0, Sign::Signed) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_near_zero_signed_reference_uses_the_absolute_floor() {
        let tolerance = tolerance();
        assert!((allowed(&tolerance, -0.004, Sign::Signed) - 0.005).abs() < 1e-12);

        let reference = CellReference {
            cmr_margin: vec![-0.004],
            ..CellReference::default()
        };
        let inside = Observed {
            cmr_margin: vec![-0.0005],
            ..Observed::default()
        };
        assert!(check(&reference, &tolerance, &inside).is_empty());
        let outside = Observed {
            cmr_margin: vec![0.004],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance, &outside);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "cmr_margin");
    }

    #[test]
    fn a_near_zero_positive_reference_uses_the_widened_relative_band() {
        let tolerance = tolerance();
        assert!((allowed(&tolerance, 0.0272, Sign::Positive) - 0.005_44).abs() < 1e-12);

        let reference = CellReference {
            mean_commutator: vec![0.0272],
            loss: vec![0.0764],
            flow_ratio_overall: vec![0.004_49],
            ..CellReference::default()
        };
        let inside = Observed {
            mean_commutator: vec![0.0320],
            loss: vec![0.0780],
            flow_ratio_overall: vec![0.005_20],
            ..Observed::default()
        };
        assert!(check(&reference, &tolerance, &inside).is_empty());

        let outside = Observed {
            mean_commutator: vec![0.0340],
            loss: vec![0.0764],
            flow_ratio_overall: vec![0.004_49],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance, &outside);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "mean_commutator");
    }

    #[test]
    fn the_absolute_floor_does_not_reach_a_near_zero_positive_quantity() {
        let tolerance = tolerance();
        let positive = allowed(&tolerance, 0.004_49, Sign::Positive);
        let signed = allowed(&tolerance, 0.004_49, Sign::Signed);
        assert!(positive < signed);
        assert!((positive - 0.000_898).abs() < 1e-12);
    }

    #[test]
    fn orthogonality_is_an_upper_bound_not_a_target() {
        let reference = CellReference {
            total_orth_error_max: Some(1e-12),
            ..CellReference::default()
        };
        let observed = Observed {
            total_orth_error: vec![1.5e-14, 1e-30],
            ..Observed::default()
        };
        assert!(check(&reference, &tolerance(), &observed).is_empty());

        let drifted = Observed {
            total_orth_error: vec![1e-9],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance(), &drifted);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "total_orth_error");
    }

    #[test]
    fn a_chance_level_cell_is_held_below_its_ceiling() {
        let reference = CellReference {
            val_acc_upper_bound: Some(15.0),
            ..CellReference::default()
        };
        let observed = Observed {
            val_acc: vec![9.9, 16.2],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance(), &observed);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "val_acc_upper_bound");
        assert_eq!(deviations[0].epoch, 2);
    }

    #[test]
    fn parameter_count_must_match_exactly() {
        let reference = CellReference {
            param_count: Some(150_528),
            ..CellReference::default()
        };
        let observed = Observed {
            param_count: Some(150_529),
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance(), &observed);
        assert_eq!(deviations.len(), 1);
        assert_eq!(deviations[0].quantity, "param_count");
    }

    fn bench_reference() -> BenchReference {
        BenchReference {
            name: "bench".to_string(),
            config: "configs/paper/bench.toml".to_string(),
            engine: "bench".to_string(),
            orthogonality_upper_bound: 1e-11,
            require_slopes: true,
            require_block_qr_monotone: true,
            monotone_min_size_ratio: 2.0,
            monotone_tolerance: 0.10,
        }
    }

    fn bench_observed() -> ObservedBench {
        ObservedBench {
            orthogonality_error: vec![
                ("pade".to_string(), 256, 4.0e-14),
                ("qr".to_string(), 512, 9.0e-13),
            ],
            slopes: vec![("pade".to_string(), 2.71), ("block_qr".to_string(), 1.94)],
            block_qr_median_s: vec![(512, 0.0021), (256, 0.0009), (1024, 0.0044)],
        }
    }

    #[test]
    fn the_benchmark_reference_is_read_from_its_own_table() {
        let references = load(&reference_path()).expect("reference file");
        let bench = references.bench.expect("benchmark reference");
        assert_eq!(bench.name, "bench");
        assert_eq!(bench.engine, "bench");
        assert!((bench.orthogonality_upper_bound - 1e-11).abs() < 1e-24);
        assert!(bench.require_slopes);
        assert!(bench.require_block_qr_monotone);
        assert!((bench.monotone_min_size_ratio - 2.0).abs() < 1e-12);
        assert!((bench.monotone_tolerance - 0.10).abs() < 1e-12);
        assert!(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join(&bench.config)
                .exists()
        );
    }

    #[test]
    fn an_unknown_field_in_the_reference_file_is_refused() {
        let text = concat!(
            "[tolerance]
",
            "accuracy_points = 1.5
",
            "orthogonality_upper_bound = 1e-12
",
            "relative = 0.05
",
            "near_zero_reference = 0.05
",
            "near_zero_absolute = 0.005
",
            "positive_small_relative = 0.20
",
            "[[cell]]
",
            "name = \"e1\"
",
            "config = \"configs/paper/e1.toml\"
",
            "engine = \"suite\"
",
            "require_slopes = true
",
        );
        let parsed = toml::from_str::<References>(text);
        assert!(parsed.is_err(), "an unmodelled field must not be dropped");
    }

    #[test]
    fn a_benchmark_run_inside_its_expectations_passes() {
        assert!(check_bench(&bench_reference(), &bench_observed()).is_empty());
    }

    #[test]
    fn a_method_above_the_orthogonality_bound_is_reported() {
        let mut observed = bench_observed();
        observed
            .orthogonality_error
            .push(("cayley".to_string(), 1024, 3.0e-9));
        let deviations = check_bench(&bench_reference(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(deviations[0].contains("cayley"));
    }

    #[test]
    fn a_missing_slope_is_reported() {
        let mut observed = bench_observed();
        observed.slopes.push(("taylor".to_string(), f64::NAN));
        let deviations = check_bench(&bench_reference(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(deviations[0].contains("taylor"));

        let mut relaxed = bench_reference();
        relaxed.require_slopes = false;
        assert!(check_bench(&relaxed, &observed).is_empty());
    }

    fn with_block_times(times: &[(usize, f64)]) -> ObservedBench {
        ObservedBench {
            block_qr_median_s: times.to_vec(),
            ..bench_observed()
        }
    }

    #[test]
    fn a_per_block_time_that_falls_with_the_dimension_is_reported() {
        let observed = with_block_times(&[(256, 0.0010), (512, 0.0021), (1024, 0.0005)]);
        let deviations = check_bench(&bench_reference(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(deviations[0].contains("per-block time"));

        let mut relaxed = bench_reference();
        relaxed.require_block_qr_monotone = false;
        assert!(check_bench(&relaxed, &observed).is_empty());
    }

    #[test]
    fn a_small_dip_is_within_the_timing_noise() {
        let observed = with_block_times(&[(256, 0.0010), (512, 0.000_95)]);
        assert!(check_bench(&bench_reference(), &observed).is_empty());
    }

    #[test]
    fn a_dip_beyond_the_timing_noise_is_reported() {
        let observed = with_block_times(&[(256, 0.0010), (512, 0.0008)]);
        let deviations = check_bench(&bench_reference(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(deviations[0].contains("n 512"));
    }

    #[test]
    fn a_size_too_close_to_the_previous_one_is_skipped() {
        let observed = with_block_times(&[(512, 0.0020), (784, 0.0005), (1024, 0.0030)]);
        assert!(
            check_bench(&bench_reference(), &observed).is_empty(),
            "784 is below twice 512, so neither comparison against it holds"
        );

        let observed = with_block_times(&[(512, 0.0020), (784, 0.0025), (1024, 0.0004)]);
        let deviations = check_bench(&bench_reference(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(
            deviations[0].contains("below n 512"),
            "the skipped size must not become the point of comparison: {}",
            deviations[0]
        );
    }

    #[test]
    fn an_empty_benchmark_run_is_reported() {
        let deviations = check_bench(&bench_reference(), &ObservedBench::default());
        assert_eq!(deviations.len(), 2);
    }

    #[test]
    fn a_missing_epoch_is_reported() {
        let reference = CellReference {
            val_acc: vec![39.94, 55.20],
            ..CellReference::default()
        };
        let observed = Observed {
            val_acc: vec![39.94],
            ..Observed::default()
        };
        let deviations = check(&reference, &tolerance(), &observed);
        assert_eq!(deviations.len(), 1);
        assert!(deviations[0].observed.is_nan());
    }
}
