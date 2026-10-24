//! Run-artifact structures and writers.
//!
//! Two shapes exist, one per engine, because the two engines answer
//! different questions and their artifacts have different readers. Field
//! names are the article's.
//!
//! Invariants the analysis scripts depend on:
//! - every `SuiteHistory::per_block` vector except `total_orth_error_mean`
//!   is flattened per-epoch major, per-block minor, and is reshaped by the
//!   reader using the length of `epoch`;
//! - `momentum_max_norm` is always a number, and the number `0.0` means
//!   unconstrained.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Name of the LAPACK SVD driver a run is decomposed with when the
/// configuration does not name one.
pub(crate) fn default_svd_driver() -> String {
    "gesvd".to_string()
}

/// Momentum norm ceiling, where `0.0` means unconstrained.
///
/// The gauge regime grid contains an unconstrained column. Writing that
/// column as a JSON `null` would make it indistinguishable from an absent
/// field, so the ceiling is always a number and `0.0` carries the meaning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MaxNorm(pub f64);

impl MaxNorm {
    /// The ceiling as the rule constructor expects it: `None` when unconstrained.
    pub fn as_option(self) -> Option<f64> {
        if self.0 > 0.0 { Some(self.0) } else { None }
    }

    /// Build a ceiling from a rule-level option.
    pub fn from_option(value: Option<f64>) -> Self {
        Self(value.unwrap_or(0.0))
    }
}

/// Configuration block of a run on the full-train engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteConfig {
    /// Name of the cell this run belongs to.
    pub experiment_name: String,
    /// Engine that produced the run. Always `"suite"` here.
    pub engine: String,
    /// Run seed.
    pub seed: u64,
    /// Steps between forced retractions; `0` disables them.
    pub retraction_interval: usize,
    /// `"flat"` or `"patchwise"`.
    pub model_variant: String,
    /// Epochs requested.
    pub total_epochs: u32,
    /// Batch size.
    pub batch_size: usize,
    /// Ambient input width.
    pub input_dim: usize,
    /// Width of the input subspace.
    pub input_subspace_dim: usize,
    /// Width of each interior subspace.
    pub interior_subspace_dim: usize,
    /// Number of interior blocks.
    pub num_interior_blocks: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Width of each class anchor.
    pub cmr_k: usize,
    /// Interior composition topology.
    pub topology: String,
    /// Gauge learning rate.
    pub lr_gauge: f64,
    /// Gauge momentum.
    pub momentum: f64,
    /// Momentum norm ceiling; `0.0` is unconstrained.
    pub momentum_max_norm: MaxNorm,
    /// Orthogonal-update method.
    pub exp_method: String,
    /// LAPACK SVD driver: `"gesvd"` or `"gesdd"`.
    #[serde(default = "default_svd_driver")]
    pub svd_driver: String,
    /// Block update method.
    pub update_method: String,
    /// Retraction method.
    pub retraction_method: String,
    /// Angular margin in radians.
    pub cmr_margin: f64,
    /// Supervised loss.
    pub loss: String,
    /// Interior nonlinearity.
    pub activation: String,
    /// Where the nonlinearity is applied.
    pub activation_placement: String,
    /// Subspace action used by the interior blocks.
    pub subspace_action: String,
}

/// Summary metrics of a run on the full-train engine.
///
/// This engine reports no separate test accuracy: `val_acc` already is the
/// accuracy on the canonical evaluation split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteResult {
    /// Batch size actually used.
    pub batch_size: usize,
    /// Best evaluation accuracy over the run, in percent.
    pub best_val_acc: f64,
    /// Epoch at which `best_val_acc` occurred.
    pub best_epoch: u32,
    /// Evaluation accuracy at the last epoch, in percent.
    pub final_val_acc: f64,
    /// Training accuracy at the last epoch, in percent.
    pub final_train_acc: f64,
    /// `final_train_acc - final_val_acc`.
    pub train_val_gap: f64,
    /// Wall-clock training time in minutes.
    pub training_time_minutes: f64,
    /// Epochs completed per minute.
    pub epochs_per_minute: f64,
    /// Training samples processed per second.
    pub samples_per_second: f64,
}

/// Per-block history, flattened per-epoch major and per-block minor.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerBlockHistory {
    /// Connection norm of each block, flattened.
    pub connection_norms_mean: Vec<f64>,
    /// Commutator norm of each block, flattened.
    pub commutator_norms_mean: Vec<f64>,
    /// Orthogonality error of each block, flattened.
    pub orthogonality_errors_mean: Vec<f64>,
    /// One aggregate orthogonality error per epoch.
    pub total_orth_error_mean: Vec<f64>,
    /// Drift of each block from its initial basis, flattened.
    pub drift_mean: Vec<f64>,
}

/// Scalar rule diagnostics, one entry per epoch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionScalars {
    /// Mean commutator norm across blocks.
    pub mean_comm_norm: Vec<f64>,
    /// Mean commutator target across blocks.
    pub mean_comm_target: Vec<f64>,
    /// Mean geometric error across blocks.
    pub mean_geom_error: Vec<f64>,
    /// Fraction of blocks inside the deadband.
    pub deadband_fraction: Vec<f64>,
    /// Rule step counter.
    pub global_step: Vec<usize>,
    /// Gauge learning rate in force.
    pub lr: Vec<f64>,
}

/// Raw per-block vectors kept at a snapshot epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Epoch the snapshot was taken at.
    pub epoch: u32,
    /// Commutator norm of each block.
    pub commutator_norms: Vec<f64>,
    /// Connection norm of each block.
    pub connection_norms: Vec<f64>,
    /// Orthogonality error of each block.
    pub orthogonality_errors: Vec<f64>,
    /// Aggregate orthogonality error.
    pub total_orth_error: f64,
}

/// Per-epoch history of a run on the full-train engine.
///
/// Every field below except the optional flattened extras is required on
/// deserialization: an artifact missing or misnaming a field is a defect in
/// the run, and parsing must fail loudly rather than silently substitute a
/// default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteHistory {
    /// Epoch indices, one-based.
    pub epoch: Vec<u32>,
    /// Wall-clock seconds per epoch.
    pub time_epoch_s: Vec<f64>,
    /// Training accuracy per epoch, in percent.
    pub train_acc: Vec<f64>,
    /// Evaluation accuracy per epoch, in percent.
    pub val_acc: Vec<f64>,
    /// Mean training loss per epoch.
    pub loss: Vec<f64>,
    /// Mean commutator norm per epoch.
    pub mean_commutator: Vec<f64>,
    /// Per-block vectors.
    pub per_block: PerBlockHistory,
    /// Scalar rule diagnostics.
    pub evolution_scalars: EvolutionScalars,
    /// Snapshots kept at the configured cadence.
    pub snapshots: Vec<Snapshot>,
    /// Mean ratio of interior output norm to input norm, per epoch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flow_ratio_overall: Vec<f64>,
    /// Mean per-stage projection ratio, per epoch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projection_ratio_mean: Vec<f64>,
    /// Mean angle to the correct class anchor, per epoch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mean_angle_correct: Vec<f64>,
    /// Mean smallest angle to a wrong class anchor, per epoch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mean_angle_wrong_min: Vec<f64>,
    /// `mean_angle_wrong_min` minus `mean_angle_correct`, per epoch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cmr_margin: Vec<f64>,
}

/// Distance of each block from its initial basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerBlockDistances {
    /// Projector distance from the initial basis, per block.
    pub projection_distance_from_init: Vec<f64>,
    /// Grassmann geodesic distance from the initial basis, per block.
    pub grassmann_distance_from_init: Vec<f64>,
}

/// Timing of one epoch, restricted to the quantities the article's timing
/// claims are built from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochProfile {
    /// Seconds spent in the training pass.
    pub train_epoch_total_sec: f64,
    /// Seconds spent in the evaluation pass.
    pub eval_epoch_total_sec: f64,
    /// Seconds for the whole epoch.
    pub epoch_total_sec: f64,
    /// Seconds spent inside rule steps.
    pub evolution_step_sec: f64,
    /// Seconds spent inside retractions.
    pub evolution_retraction_sec: f64,
    /// Number of retractions performed.
    pub retraction_steps: usize,
}

/// One entry of the per-epoch record list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochRecord {
    /// Epoch index, one-based.
    pub epoch: u32,
    /// Wall-clock seconds for the epoch.
    pub time_epoch_s: f64,
    /// Training accuracy, in percent.
    pub train: f64,
    /// Evaluation accuracy, in percent.
    pub val: f64,
    /// Timing breakdown.
    pub profile: EpochProfile,
    /// Block distances, present at the configured cadence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_block_distances: Option<PerBlockDistances>,
}

/// Complete artifact of a run on the full-train engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteArtifact {
    /// Path of this artifact relative to the output root.
    pub output_path: String,
    /// Configuration the run was built from.
    pub config: SuiteConfig,
    /// Summary metrics.
    pub result: SuiteResult,
    /// Per-epoch timing.
    pub runtime_profile: Vec<EpochProfile>,
    /// Per-epoch history.
    pub history: SuiteHistory,
    /// Per-epoch records.
    pub epochs: Vec<EpochRecord>,
    /// Free-form notes recorded by the cell.
    pub notes: Vec<String>,
    /// Angle from each true class to each class anchor at the last epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_angle_matrix_final: Option<Vec<Vec<f64>>>,
}

/// Sidecar describing a finished set of runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteSummary {
    /// Driver version that produced the set.
    pub runner_version: String,
    /// Unix timestamp at which the set finished.
    pub generated_at_unix: u64,
    /// Cell name.
    pub suite: String,
    /// Output directory relative to the output root.
    pub output_dir: String,
    /// Wall-clock seconds for the whole set.
    pub elapsed_seconds: f64,
    /// Free-form notes.
    pub notes: Vec<String>,
    /// Artifact file names produced by the set.
    pub run_files: Vec<String>,
    /// Aggregated analysis, one entry per retraction interval.
    pub analysis: Vec<RetractionAnalysis>,
}

/// Aggregation across the seeds of one retraction interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetractionAnalysis {
    /// Retraction interval the entry aggregates.
    pub retraction_interval: usize,
    /// Number of runs aggregated.
    pub num_runs: usize,
    /// Mean final orthogonality error.
    pub orth_error_mean: f64,
    /// Population standard deviation of the final orthogonality error.
    pub orth_error_std: f64,
    /// Mean final commutator norm.
    pub commutator_mean: f64,
    /// Population standard deviation of the final commutator norm.
    pub commutator_std: f64,
    /// Mean best evaluation accuracy.
    pub val_acc_mean: f64,
    /// Population standard deviation of the best evaluation accuracy.
    pub val_acc_std: f64,
    /// Runs whose orthogonality error left the numerical range.
    pub diverged_count: usize,
}

/// Configuration block of a run on the three-way split engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableConfig {
    /// Arm: `"cosmos"`, `"mlp"` or `"rsgd"`.
    pub model_type: String,
    /// Engine that produced the run. Always `"table"` here.
    pub engine: String,
    /// Dataset name.
    pub dataset: String,
    /// Ambient input width.
    pub input_dim: usize,
    /// Width of the input subspace.
    pub input_subspace_dim: usize,
    /// Width of each interior subspace.
    pub interior_subspace_dim: usize,
    /// Number of interior blocks.
    pub num_interior_blocks: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Width of each class anchor.
    pub cmr_k: usize,
    /// Interior nonlinearity.
    pub activation: String,
    /// Where the nonlinearity is applied.
    pub activation_placement: String,
    /// Interior composition topology.
    pub topology_str: String,
    /// Angular margin in radians.
    pub cmr_margin: f64,
    /// Supervised loss.
    pub loss: String,
    /// Epochs requested.
    pub total_epochs: u32,
    /// Batch size.
    pub batch_size: usize,
    /// Run seed.
    pub seed: u64,
    /// Split seed, independent of the run seed.
    pub split_seed: u64,
    /// Validation split size.
    pub val_size: usize,
    /// Whether the trailing partial training batch is discarded.
    pub drop_last: bool,
    /// Gauge learning rate.
    pub lr_gauge: f64,
    /// Gauge momentum.
    pub momentum: f64,
    /// Momentum norm ceiling; `0.0` is unconstrained.
    pub momentum_max_norm: MaxNorm,
    /// Steps between forced retractions.
    pub retraction_interval: usize,
    /// LAPACK SVD driver: `"gesvd"` or `"gesdd"`.
    #[serde(default = "default_svd_driver")]
    pub svd_driver: String,
    /// Hidden width of the two-layer comparison network.
    pub mlp_hidden: usize,
    /// Learning rate of the two-layer comparison network.
    pub mlp_lr: f64,
    /// Optimizer of the two-layer comparison network.
    pub mlp_optimizer: String,
    /// Learning rate of the Riemannian comparison arm.
    pub rsgd_lr: f64,
    /// Initialization family for the class anchors.
    pub anchor_init: String,
    /// `"flat"` or `"patchwise"`.
    pub model_variant: String,
    /// Image shape for the patchwise variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_shape: Option<[usize; 3]>,
    /// Patch side length for the patchwise variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_size: Option<usize>,
    /// Width of each patch subspace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_subspace_dim: Option<usize>,
    /// Width of the aggregation subspace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg_subspace_dim: Option<usize>,
}

/// Summary metrics of a run on the three-way split engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableResult {
    /// Arm.
    pub model_type: String,
    /// Run seed.
    pub seed: u64,
    /// Number of learnable parameters.
    pub param_count: usize,
    /// Best validation accuracy, in percent.
    pub best_val_acc: f64,
    /// Epoch at which `best_val_acc` occurred.
    pub best_epoch: u32,
    /// Test accuracy recorded at the best-validation epoch, in percent.
    pub test_acc_at_best_val: f64,
    /// Validation accuracy at the last epoch, in percent.
    pub final_val_acc: f64,
    /// Test accuracy at the last epoch, in percent.
    pub final_test_acc: f64,
    /// Training accuracy at the last epoch, in percent.
    pub final_train_acc: f64,
    /// `final_train_acc` minus `final_test_acc`.
    pub train_test_gap: f64,
    /// Mean seconds per training epoch, over the training pass alone:
    /// validation and test evaluation are excluded, for every arm.
    pub mean_train_epoch_s: f64,
    /// Total training time in minutes.
    pub total_train_time_min: f64,
}

/// Per-epoch history of a run on the three-way split engine.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableHistory {
    /// Epoch indices, one-based.
    pub epoch: Vec<u32>,
    /// Training accuracy per epoch, in percent.
    pub train_acc: Vec<f64>,
    /// Validation accuracy per epoch, in percent.
    pub val_acc: Vec<f64>,
    /// Test accuracy per epoch, in percent.
    pub test_acc: Vec<f64>,
    /// Mean training loss per epoch.
    pub loss: Vec<f64>,
    /// Seconds of the training pass, per epoch. Validation and test
    /// evaluation are excluded, for every arm.
    pub train_epoch_s: Vec<f64>,
}

/// Complete artifact of a run on the three-way split engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableArtifact {
    /// Configuration the run was built from.
    pub config: TableConfig,
    /// Summary metrics.
    pub result: TableResult,
    /// Per-epoch history.
    pub history: TableHistory,
}

/// Write `value` as pretty JSON, creating parent directories as needed.
///
/// # Errors
///
/// Returns an I/O error when a directory or the file cannot be created, or
/// when serialization fails.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(value)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    text.push('\n');
    fs::write(path, text)
}

/// Read a JSON artifact back.
///
/// # Errors
///
/// Returns an I/O error when the file cannot be read, or an `InvalidData`
/// error when its contents do not match the requested shape.
pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("{path:?}: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_suite() -> SuiteArtifact {
        SuiteArtifact {
            output_path: "e1/ring_L5_d16_lr0.01_m0.9_n0_mar0.300000/run_r10_seed_1.json"
                .to_string(),
            config: SuiteConfig {
                experiment_name: "e1".to_string(),
                engine: "suite".to_string(),
                seed: 1,
                retraction_interval: 10,
                model_variant: "flat".to_string(),
                total_epochs: 2,
                batch_size: 2048,
                input_dim: 784,
                input_subspace_dim: 32,
                interior_subspace_dim: 16,
                num_interior_blocks: 5,
                num_classes: 10,
                cmr_k: 8,
                topology: "series".to_string(),
                lr_gauge: 0.8,
                momentum: 0.9,
                momentum_max_norm: MaxNorm(1.2),
                exp_method: "qr".to_string(),
                svd_driver: "gesvd".to_string(),
                update_method: "grassmann".to_string(),
                retraction_method: "qr".to_string(),
                cmr_margin: std::f64::consts::PI / 5.0,
                loss: "geodesic_hinge".to_string(),
                activation: "none".to_string(),
                activation_placement: "none".to_string(),
                subspace_action: "projection".to_string(),
            },
            result: SuiteResult {
                batch_size: 2048,
                best_val_acc: 55.2,
                best_epoch: 2,
                final_val_acc: 55.2,
                final_train_acc: 51.35,
                train_val_gap: -3.85,
                training_time_minutes: 0.5,
                epochs_per_minute: 4.0,
                samples_per_second: 8000.0,
            },
            runtime_profile: vec![EpochProfile::default(), EpochProfile::default()],
            history: SuiteHistory {
                epoch: vec![1, 2],
                time_epoch_s: vec![7.5, 7.4],
                train_acc: vec![27.75, 51.35],
                val_acc: vec![39.94, 55.20],
                loss: vec![0.0764, 0.0252],
                mean_commutator: vec![0.0272, 0.00725],
                per_block: PerBlockHistory {
                    connection_norms_mean: vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                    commutator_norms_mean: vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                    orthogonality_errors_mean: vec![0.0; 6],
                    total_orth_error_mean: vec![1.49e-14, 1.51e-14],
                    drift_mean: vec![0.0; 6],
                },
                evolution_scalars: EvolutionScalars {
                    mean_comm_norm: vec![0.0272, 0.00725],
                    mean_comm_target: vec![0.03, 0.01],
                    mean_geom_error: vec![0.0, 0.0],
                    deadband_fraction: vec![0.0, 0.0],
                    global_step: vec![29, 58],
                    lr: vec![0.8, 0.8],
                },
                snapshots: vec![],
                flow_ratio_overall: vec![],
                projection_ratio_mean: vec![],
                mean_angle_correct: vec![],
                mean_angle_wrong_min: vec![],
                cmr_margin: vec![],
            },
            epochs: vec![],
            notes: vec![],
            class_angle_matrix_final: None,
        }
    }

    #[test]
    fn suite_artifact_round_trips() {
        let artifact = sample_suite();
        let text = serde_json::to_string(&artifact).expect("serialize");
        let back: SuiteArtifact = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(artifact, back);
    }

    #[test]
    fn per_block_vectors_stay_flattened() {
        let artifact = sample_suite();
        let epochs = artifact.history.epoch.len();
        let blocks = artifact.history.per_block.connection_norms_mean.len() / epochs;
        assert_eq!(epochs, 2);
        assert_eq!(blocks, 3);
        assert_eq!(
            artifact.history.per_block.total_orth_error_mean.len(),
            epochs
        );
    }

    #[test]
    fn unconstrained_max_norm_serializes_as_zero() {
        let value = MaxNorm::from_option(None);
        assert_eq!(serde_json::to_string(&value).expect("serialize"), "0.0");
        let back: MaxNorm = serde_json::from_str("0.0").expect("deserialize");
        assert_eq!(back.as_option(), None);
        let bounded: MaxNorm = serde_json::from_str("1.2").expect("deserialize");
        assert_eq!(bounded.as_option(), Some(1.2));
    }

    #[test]
    fn table_history_carries_loss() {
        let history = TableHistory {
            epoch: vec![1],
            train_acc: vec![28.78],
            val_acc: vec![36.92],
            test_acc: vec![36.46],
            loss: vec![0.0801],
            train_epoch_s: vec![12.0],
        };
        let text = serde_json::to_string(&history).expect("serialize");
        assert!(text.contains("\"loss\""));
        let back: TableHistory = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(history, back);
    }

    #[test]
    fn optional_flat_history_fields_are_omitted_when_empty() {
        let artifact = sample_suite();
        let text = serde_json::to_string(&artifact.history).expect("serialize");
        assert!(!text.contains("flow_ratio_overall"));
        assert!(text.contains("mean_commutator"));
    }

    #[test]
    fn an_unknown_config_key_is_rejected() {
        let text = r#"{"epoch":[1],"train_acc":[1.0],"val_acc":[1.0],
            "test_acc":[1.0],"loss":[0.1],"train_epoch_s":[1.0],"extra":1}"#;
        let parsed: Result<TableHistory, _> = serde_json::from_str(text);
        assert!(parsed.is_err(), "unknown keys must not be accepted");
    }
}
