//! Expansion of a cell definition into the runs it describes.
//!
//! Premises:
//! - Output is laid out as `<out>/<experiment>/<coordinate>/<stem>.json`, so a
//!   sweep never writes two runs to the same path.
//! - An engine accepts only the cells it was written for; a cell sent to the
//!   wrong engine yields plausible but incomparable numbers, so it is refused
//!   before any run starts.
//! - A shortened run compares itself against the documented reference and
//!   exits non-zero on any deviation.

pub mod e1;
pub mod e2;
pub mod e3;
pub mod e4;
pub mod e5;

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::cli::RunArgs;
use crate::config::{Cell, Engine, expand, load};
use crate::datasets::{self, Dataset, DatasetName};
use crate::engine::{EngineError, artifact_path, relative_artifact_path, suite, table};
use crate::schema::{RetractionAnalysis, SuiteSummary, write_json};
use crate::smoke::{self, Observed};

/// Path of the reference file the shortened runs compare themselves against.
pub const REFERENCE_FILE: &str = "configs/paper/smoke_references.toml";

/// Outcome of running a cell set.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Artifact paths written, relative to the output root.
    pub run_files: Vec<String>,
    /// Deviations found when the set ran in shortened mode.
    pub deviations: Vec<String>,
}

impl Outcome {
    /// Append another set's artifacts and deviations to this one.
    pub fn absorb(&mut self, other: Self) {
        self.run_files.extend(other.run_files);
        self.deviations.extend(other.deviations);
    }
}

fn dataset_for(cells: &[Cell]) -> DatasetName {
    cells
        .first()
        .map_or(DatasetName::FashionMnist, |cell| cell.dataset)
}

/// Run one cell on the full-training-set engine.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] when the cell does not declare that
/// engine or does not use the flat model, and otherwise whatever the engine
/// reports.
pub fn run_suite_cell(
    cell: &Cell,
    data: &Dataset,
    out_dir: &Path,
) -> Result<(String, Observed), EngineError> {
    if cell.engine != Engine::Suite {
        return Err(EngineError::Unknown {
            field: "engine",
            value: cell.engine.to_string(),
        });
    }
    if cell.model.variant != "flat" {
        return Err(EngineError::Unknown {
            field: "model.variant",
            value: cell.model.variant.clone(),
        });
    }
    let artifact = suite::run(cell, data, out_dir)?;
    let observed = Observed {
        train_acc: artifact.history.train_acc.clone(),
        val_acc: artifact.history.val_acc.clone(),
        loss: artifact.history.loss.clone(),
        mean_commutator: artifact.history.mean_commutator.clone(),
        flow_ratio_overall: artifact.history.flow_ratio_overall.clone(),
        cmr_margin: artifact.history.cmr_margin.clone(),
        total_orth_error: artifact.history.per_block.total_orth_error_mean.clone(),
        ..Observed::default()
    };
    Ok((artifact.output_path.clone(), observed))
}

/// Run one cell on the three-way split engine.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] when the cell does not declare that
/// engine, and otherwise whatever the engine reports.
pub fn run_table_cell(
    cell: &Cell,
    data: &Dataset,
    out_dir: &Path,
) -> Result<(String, Observed), EngineError> {
    if cell.engine != Engine::Table {
        return Err(EngineError::Unknown {
            field: "engine",
            value: cell.engine.to_string(),
        });
    }
    let artifact = table::run(cell, data, out_dir)?;
    let observed = Observed {
        train_acc: artifact.history.train_acc.clone(),
        val_acc: artifact.history.val_acc.clone(),
        test_acc: artifact.history.test_acc.clone(),
        loss: artifact.history.loss.clone(),
        param_count: Some(artifact.result.param_count),
        ..Observed::default()
    };
    Ok((relative_artifact_path(cell), observed))
}

fn run_one(cell: &Cell, data: &Dataset, out_dir: &Path) -> Result<(String, Observed), EngineError> {
    match cell.engine {
        Engine::Suite => run_suite_cell(cell, data, out_dir),
        Engine::Table => run_table_cell(cell, data, out_dir),
    }
}

fn retraction_entry(cell: &Cell, observed: &Observed) -> (usize, f64, f64, f64) {
    let last_orth = observed
        .total_orth_error
        .last()
        .copied()
        .unwrap_or(f64::NAN);
    let last_commutator = observed.mean_commutator.last().copied().unwrap_or(f64::NAN);
    let best_val = observed
        .val_acc
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (
        cell.rules.retraction_interval,
        last_orth,
        last_commutator,
        best_val,
    )
}

/// Run every cell of `config_name` under `args`.
///
/// # Errors
///
/// Returns a description when the cell file cannot be read, the dataset
/// cannot be made available, a run fails, or a shortened run deviates from
/// its reference.
///
/// # Panics
///
/// Panics when a run thread panics, propagating that panic to the caller.
pub fn run_cells(args: &RunArgs, config_name: &str) -> Result<Outcome, String> {
    args.validate()?;
    let path = args.config_path(config_name);
    let file = load(&path).map_err(|err| err.to_string())?;
    let cells = expand(&file, args.seeds.as_deref(), args.smoke).map_err(|err| err.to_string())?;
    if cells.is_empty() {
        return Err(format!("{} expands to no run", path.display()));
    }

    let data: Dataset = datasets::load(dataset_for(&cells), &args.data_dir, !args.no_download)
        .map_err(|err| err.to_string())?;

    let started = Instant::now();
    let mut run_files = Vec::new();
    let mut observed_by_cell: Vec<(String, Observed)> = Vec::new();
    let mut analysis_input: Vec<(usize, f64, f64, f64)> = Vec::new();

    for chunk in cells.chunks(args.threads.max(1)) {
        let results: Vec<Result<(String, Observed), EngineError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|cell| {
                    let out = args.out.clone();
                    let data = &data;
                    scope.spawn(move || run_one(cell, data, &out))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("run thread"))
                .collect()
        });

        for (index, result) in results.into_iter().enumerate() {
            let (relative, observed) = result.map_err(|err| err.to_string())?;
            println!("wrote {relative}");
            if chunk[index].engine == Engine::Suite {
                analysis_input.push(retraction_entry(&chunk[index], &observed));
            }
            run_files.push(relative);
            observed_by_cell.push((chunk[index].arm.clone(), observed));
        }
    }

    if cells[0].engine == Engine::Suite {
        write_suite_summary(args, &cells[0], &run_files, &analysis_input, started)?;
    }

    let deviations = if args.smoke {
        check_against_references(&file.experiment, &observed_by_cell)?
    } else {
        Vec::new()
    };

    Ok(Outcome {
        run_files,
        deviations,
    })
}

/// Directory the artifact of `cell` is written into, under `root`.
pub fn output_dir(root: &Path, cell: &Cell) -> PathBuf {
    artifact_path(cell, root)
        .parent()
        .map_or_else(|| root.to_path_buf(), Path::to_path_buf)
}

fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn write_suite_summary(
    args: &RunArgs,
    first: &Cell,
    run_files: &[String],
    analysis_input: &[(usize, f64, f64, f64)],
    started: Instant,
) -> Result<(), String> {
    let mut intervals: Vec<usize> = analysis_input.iter().map(|entry| entry.0).collect();
    intervals.sort_unstable();
    intervals.dedup();

    let mut analysis = Vec::new();
    for interval in intervals {
        let rows: Vec<&(usize, f64, f64, f64)> = analysis_input
            .iter()
            .filter(|entry| entry.0 == interval)
            .collect();
        let orth: Vec<f64> = rows.iter().map(|entry| entry.1).collect();
        let commutator: Vec<f64> = rows.iter().map(|entry| entry.2).collect();
        let accuracy: Vec<f64> = rows.iter().map(|entry| entry.3).collect();
        analysis.push(RetractionAnalysis {
            retraction_interval: interval,
            num_runs: rows.len(),
            orth_error_mean: mean(&orth),
            orth_error_std: population_std(&orth),
            commutator_mean: mean(&commutator),
            commutator_std: population_std(&commutator),
            val_acc_mean: mean(&accuracy),
            val_acc_std: population_std(&accuracy),
            diverged_count: orth.iter().filter(|value| !value.is_finite()).count(),
        });
    }

    let dir = args.out.join(&first.experiment);
    let summary = SuiteSummary {
        runner_version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| err.to_string())?
            .as_secs(),
        suite: first.experiment.clone(),
        output_dir: first.experiment.clone(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        notes: Vec::new(),
        run_files: run_files.to_vec(),
        analysis: analysis.clone(),
    };
    write_json(&dir.join("suite_summary.json"), &summary).map_err(|err| err.to_string())?;
    write_json(&dir.join("analysis.json"), &analysis).map_err(|err| err.to_string())
}

/// Name under which a run of `experiment` on `arm` is listed in the
/// reference file.
pub fn reference_name(experiment: &str, arm: &str) -> String {
    format!("{experiment}:{arm}")
}

fn check_against_references(
    experiment: &str,
    observed: &[(String, Observed)],
) -> Result<Vec<String>, String> {
    let references = smoke::load(Path::new(REFERENCE_FILE)).map_err(|err| err.to_string())?;
    let mut lines = Vec::new();
    for (arm, values) in observed {
        let name = reference_name(experiment, arm);
        let Some(entry) = references
            .cell(&name)
            .or_else(|| references.cell(experiment))
        else {
            println!("no reference for {name}; structural checks only");
            continue;
        };
        for deviation in smoke::check(entry, &references.tolerance, values) {
            lines.push(format!(
                "{} {} epoch {}: expected {:.6}, observed {:.6}, tolerance {:.6}",
                entry.name,
                deviation.quantity,
                deviation.epoch,
                deviation.expected,
                deviation.observed,
                deviation.tolerance
            ));
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::config::CellFile;
    use crate::datasets::DatasetName;
    use ndarray::Array2;

    fn config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
    }

    fn cell_file(name: &str) -> CellFile {
        load(&config_dir().join(name)).unwrap_or_else(|err| panic!("{name}: {err}"))
    }

    fn count(name: &str, smoke: bool) -> usize {
        expand(&cell_file(name), None, smoke).expect("expand").len()
    }

    fn empty_dataset() -> Dataset {
        Dataset {
            train_inputs: Array2::zeros((0, 784)),
            train_labels: Vec::new(),
            test_inputs: Array2::zeros((0, 784)),
            test_labels: Vec::new(),
        }
    }

    #[test]
    fn every_default_cell_file_expands_to_the_documented_run_count() {
        assert_eq!(count("e1.toml", false), 36);
        assert_eq!(count("e2.toml", false), 180);
        assert_eq!(count("e3_baselines.toml", false), 9);
        assert_eq!(count("e3_margin_ablation.toml", false), 12);
        assert_eq!(count("e4_topology.toml", false), 60);
        assert_eq!(count("e4_companion.toml", false), 12);
    }

    #[test]
    fn a_shortened_expansion_is_one_run_per_arm() {
        assert_eq!(count("e1.toml", true), 1);
        assert_eq!(count("e2.toml", true), 1);
        assert_eq!(count("e3_margin_ablation.toml", true), 1);
        assert_eq!(count("e4_topology.toml", true), 1);
        assert_eq!(count("e4_companion.toml", true), 1);
        for name in ["e3_baselines.toml", "e5_cifar_flat.toml"] {
            assert_eq!(count(name, true), 3, "{name}");
        }
        for name in [
            "e5_patchwise_cifar.toml",
            "e5_patchwise_fashion.toml",
            "e5_patchwise_no_activation.toml",
            "e5_cifar_margin.toml",
        ] {
            assert_eq!(count(name, true), 1, "{name}");
        }
    }

    #[test]
    fn the_colour_cells_all_declare_the_colour_dataset_or_its_companion() {
        for name in e5::CELLS {
            let cells = expand(&cell_file(name), None, true).expect("expand");
            assert!(matches!(
                cells[0].dataset,
                DatasetName::Cifar10 | DatasetName::FashionMnist
            ));
        }
    }

    #[test]
    fn a_split_engine_cell_is_refused_by_the_full_training_set_engine() {
        let cells = expand(&cell_file("e3_baselines.toml"), None, true).expect("expand");
        let error = run_suite_cell(&cells[0], &empty_dataset(), Path::new("runs"))
            .expect_err("a table cell must not reach the suite engine");
        assert!(matches!(error, EngineError::Unknown { field, .. } if field == "engine"));
    }

    #[test]
    fn a_patch_wise_cell_is_refused_by_the_full_training_set_engine() {
        let mut cell = expand(&cell_file("e1.toml"), None, true).expect("expand")[0].clone();
        cell.model.variant = "patchwise".to_string();
        let error = run_suite_cell(&cell, &empty_dataset(), Path::new("runs"))
            .expect_err("a patch-wise cell must not reach the flat engine");
        assert!(matches!(error, EngineError::Unknown { field, .. } if field == "model.variant"));
    }

    #[test]
    fn a_full_training_set_cell_is_refused_by_the_split_engine() {
        let cells = expand(&cell_file("e1.toml"), None, true).expect("expand");
        let error = run_table_cell(&cells[0], &empty_dataset(), Path::new("runs"))
            .expect_err("a suite cell must not reach the table engine");
        assert!(matches!(error, EngineError::Unknown { field, .. } if field == "engine"));
    }

    #[test]
    fn every_shortened_arm_finds_its_reference_entry() {
        let references = smoke::load(&config_dir().join("smoke_references.toml")).expect("load");
        for name in [
            "e1.toml",
            "e2.toml",
            "e3_baselines.toml",
            "e4_topology.toml",
        ] {
            let file = cell_file(name);
            let cells = expand(&file, None, true).expect("expand");
            for cell in &cells {
                let key = reference_name(&file.experiment, &cell.arm);
                assert!(
                    references.cell(&key).is_some() || references.cell(&file.experiment).is_some(),
                    "{key} has no reference"
                );
            }
        }
    }

    #[test]
    fn the_artifact_directory_carries_the_sweep_coordinate() {
        let cells = expand(&cell_file("e1.toml"), None, true).expect("expand");
        let dir = output_dir(Path::new("runs"), &cells[0]);
        assert!(dir.ends_with(cells[0].coordinate()));
        assert!(dir.starts_with(Path::new("runs").join(&cells[0].experiment)));
    }
}
