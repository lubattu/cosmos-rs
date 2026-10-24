//! Run artifacts to CSV.
//!
//! Premises:
//! - A table never mixes engines. The two engines train on different
//!   amounts of data, so their accuracies answer different questions.
//! - Aggregates are mean and population standard deviation over the seeds
//!   of one configuration.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::schema::{SuiteArtifact, TableArtifact};

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn population_std(values: &[f64]) -> f64 {
    let centre = mean(values);
    (values
        .iter()
        .map(|value| (value - centre).powi(2))
        .sum::<f64>()
        / values.len().max(1) as f64)
        .sqrt()
}

/// Reason a table could not be assembled.
#[derive(Debug, PartialEq, Eq)]
pub struct MixedEngines {
    /// Engines found among the artifacts.
    pub engines: Vec<String>,
}

impl std::fmt::Display for MixedEngines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a table cannot mix engines; found {}",
            self.engines.join(", ")
        )
    }
}

impl std::error::Error for MixedEngines {}

/// Aggregate a comparison table over arms and seeds.
///
/// # Errors
///
/// Returns [`MixedEngines`] when the artifacts do not all come from the
/// three-way split engine.
pub fn baselines_table(artifacts: &[TableArtifact]) -> Result<String, MixedEngines> {
    let mut engines: Vec<String> = artifacts
        .iter()
        .map(|artifact| artifact.config.engine.clone())
        .collect();
    engines.sort();
    engines.dedup();
    if engines.len() > 1 || engines.first().map(String::as_str) != Some("table") {
        return Err(MixedEngines { engines });
    }

    let mut grouped: BTreeMap<(String, String, u64), Vec<&TableArtifact>> = BTreeMap::new();
    for artifact in artifacts {
        grouped
            .entry((
                artifact.config.dataset.clone(),
                artifact.config.model_type.clone(),
                artifact.config.cmr_margin.to_bits(),
            ))
            .or_default()
            .push(artifact);
    }

    let mut csv = String::from(
        "dataset,model,cmr_margin,runs,param_count,test_acc_at_best_val_mean,test_acc_at_best_val_std,best_val_acc_mean,mean_train_epoch_s\n",
    );
    for ((dataset, model, _margin_bits), rows) in grouped {
        let accuracies: Vec<f64> = rows
            .iter()
            .map(|artifact| artifact.result.test_acc_at_best_val)
            .collect();
        let validations: Vec<f64> = rows
            .iter()
            .map(|artifact| artifact.result.best_val_acc)
            .collect();
        let seconds: Vec<f64> = rows
            .iter()
            .map(|artifact| artifact.result.mean_train_epoch_s)
            .collect();
        let _ = writeln!(
            csv,
            "{dataset},{model},{:.6},{},{},{:.4},{:.4},{:.4},{:.3}",
            rows[0].config.cmr_margin,
            rows.len(),
            rows[0].result.param_count,
            mean(&accuracies),
            population_std(&accuracies),
            mean(&validations),
            mean(&seconds)
        );
    }
    Ok(csv)
}

/// Aggregate a full-train table over a sweep coordinate and its seeds.
///
/// # Errors
///
/// Returns [`MixedEngines`] when the artifacts do not all come from the
/// full-train engine.
pub fn suite_table(artifacts: &[SuiteArtifact]) -> Result<String, MixedEngines> {
    let mut engines: Vec<String> = artifacts
        .iter()
        .map(|artifact| artifact.config.engine.clone())
        .collect();
    engines.sort();
    engines.dedup();
    if engines.len() > 1 || engines.first().map(String::as_str) != Some("suite") {
        return Err(MixedEngines { engines });
    }

    type SuiteKey = (String, usize, usize, String, usize, u64, u64, u64, u64);
    let mut grouped: BTreeMap<SuiteKey, Vec<&SuiteArtifact>> = BTreeMap::new();
    for artifact in artifacts {
        grouped
            .entry((
                artifact.config.experiment_name.clone(),
                artifact.config.retraction_interval,
                artifact.config.num_interior_blocks,
                artifact.config.topology.clone(),
                artifact.config.interior_subspace_dim,
                artifact.config.cmr_margin.to_bits(),
                artifact.config.lr_gauge.to_bits(),
                artifact.config.momentum.to_bits(),
                artifact.config.momentum_max_norm.0.to_bits(),
            ))
            .or_default()
            .push(artifact);
    }

    let mut csv = String::from(
        "experiment,retraction_interval,num_interior_blocks,topology,interior_subspace_dim,cmr_margin,lr_gauge,momentum,momentum_max_norm,runs,best_val_acc_mean,best_val_acc_std,final_orth_error_mean,final_commutator_mean\n",
    );
    for ((experiment, interval, blocks, topology, interior_subspace_dim, ..), rows) in grouped {
        let accuracies: Vec<f64> = rows
            .iter()
            .map(|artifact| artifact.result.best_val_acc)
            .collect();
        let orth: Vec<f64> = rows
            .iter()
            .filter_map(|artifact| {
                artifact
                    .history
                    .per_block
                    .total_orth_error_mean
                    .last()
                    .copied()
            })
            .collect();
        let comm: Vec<f64> = rows
            .iter()
            .filter_map(|artifact| artifact.history.mean_commutator.last().copied())
            .collect();
        let _ = writeln!(
            csv,
            "{experiment},{interval},{blocks},{topology},{interior_subspace_dim},{:.6},{:.6},{:.6},{:.6},{},{:.4},{:.4},{:.3e},{:.6}",
            rows[0].config.cmr_margin,
            rows[0].config.lr_gauge,
            rows[0].config.momentum,
            rows[0].config.momentum_max_norm.0,
            rows.len(),
            mean(&accuracies),
            population_std(&accuracies),
            mean(&orth),
            mean(&comm)
        );
    }
    Ok(csv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{MaxNorm, TableConfig, TableHistory, TableResult};

    fn artifact(model: &str, seed: u64, accuracy: f64, engine: &str) -> TableArtifact {
        margined_artifact(model, seed, accuracy, engine, 0.6283185307179586)
    }

    fn margined_artifact(
        model: &str,
        seed: u64,
        accuracy: f64,
        engine: &str,
        cmr_margin: f64,
    ) -> TableArtifact {
        TableArtifact {
            config: TableConfig {
                model_type: model.to_string(),
                engine: engine.to_string(),
                dataset: "fashion-mnist".to_string(),
                input_dim: 784,
                input_subspace_dim: 32,
                interior_subspace_dim: 16,
                num_interior_blocks: 5,
                num_classes: 10,
                cmr_k: 8,
                activation: "none".to_string(),
                activation_placement: "none".to_string(),
                topology_str: "series".to_string(),
                cmr_margin,
                loss: "geodesic_hinge".to_string(),
                total_epochs: 30,
                batch_size: 2048,
                seed,
                split_seed: 12_345,
                val_size: 5_000,
                drop_last: true,
                lr_gauge: 0.8,
                momentum: 0.9,
                momentum_max_norm: MaxNorm(1.2),
                retraction_interval: 10,
                svd_driver: "gesvd".to_string(),
                mlp_hidden: 189,
                mlp_lr: 1e-3,
                mlp_optimizer: "adam".to_string(),
                rsgd_lr: 0.3,
                anchor_init: "keyed_gaussian_qr".to_string(),
                model_variant: "flat".to_string(),
                image_shape: None,
                patch_size: None,
                patch_subspace_dim: None,
                agg_subspace_dim: None,
            },
            result: TableResult {
                model_type: model.to_string(),
                seed,
                param_count: 150_528,
                best_val_acc: accuracy + 1.0,
                best_epoch: 30,
                test_acc_at_best_val: accuracy,
                final_val_acc: accuracy,
                final_test_acc: accuracy,
                final_train_acc: accuracy,
                train_test_gap: 0.0,
                mean_train_epoch_s: 12.0,
                total_train_time_min: 6.0,
            },
            history: TableHistory::default(),
        }
    }

    #[test]
    fn aggregate_uses_population_standard_deviation() {
        let rows = vec![
            artifact("cosmos", 1, 80.0, "table"),
            artifact("cosmos", 2, 82.0, "table"),
            artifact("cosmos", 3, 84.0, "table"),
        ];
        let csv = baselines_table(&rows).expect("table");
        assert!(csv.contains("fashion-mnist,cosmos,0.628319,3,150528,82.0000,1.6330"));
    }

    #[test]
    fn sweep_margins_stay_in_separate_rows() {
        let rows = vec![
            margined_artifact("cosmos", 1, 80.0, "table", 0.4),
            margined_artifact("cosmos", 2, 81.0, "table", 0.4),
            margined_artifact("cosmos", 1, 90.0, "table", 0.8),
            margined_artifact("cosmos", 2, 91.0, "table", 0.8),
        ];
        let csv = baselines_table(&rows).expect("table");
        let data_lines: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(data_lines.len(), 2);
        for line in data_lines {
            let runs = line.split(',').nth(3).expect("runs column");
            assert_eq!(runs, "2");
        }
    }

    #[test]
    fn mixed_engines_are_refused() {
        let rows = vec![
            artifact("cosmos", 1, 80.0, "table"),
            artifact("cosmos", 2, 82.0, "suite"),
        ];
        let err = baselines_table(&rows).expect_err("refusal");
        assert_eq!(err.engines.len(), 2);
    }

    #[test]
    fn header_names_every_reported_column() {
        let rows = vec![artifact("mlp", 1, 88.0, "table")];
        let csv = baselines_table(&rows).expect("table");
        let header = csv.lines().next().expect("header");
        assert_eq!(header.split(',').count(), 9);
    }
}
