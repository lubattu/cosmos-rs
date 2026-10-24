//! Serializable training metrics.
//!
//! [`EpochMetrics`] is one row of a run's history and [`BaseExperimentResult`] is
//! its summary. Both derive `Serialize`/`Deserialize` so a driver can write a run
//! record without redefining the schema.

use serde::{Deserialize, Serialize};

/// Phase label attached to a recorded epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingPhase {
    /// Standard training.
    #[default]
    Standard,
}

/// Metrics recorded for one training epoch.
///
/// Losses are in nats and accuracies in percentage points.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// One-based index of the epoch this row describes.
    pub epoch: u32,
    /// Phase label carried by the adapter that produced the epoch.
    pub phase: TrainingPhase,
    /// Sample-weighted mean training loss over the epoch.
    pub train_loss: f64,
    /// Sample-weighted mean training accuracy over the epoch.
    pub train_acc: f64,
    /// Held-out accuracy, or `None` when the epoch ran without validation.
    pub val_a_as_a: Option<f64>,
    /// Validation loss under the run's own supervised objective — the
    /// held-out counterpart of `train_loss`, aggregated from the same
    /// evaluation forward that produces `val_a_as_a`. `None` when the epoch
    /// ran without a validation split.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub val_loss: Option<f64>,
    /// Mean commutator norm across the epoch's evolution steps.
    pub mean_commutator: f64,
    /// Wall time of the epoch, training plus validation, in seconds.
    pub time_sec: f64,
    /// Optional per-block commutator norms captured for the epoch.
    pub comm_snapshot: Option<Vec<f64>>,
}

/// Summary statistics for a training run.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BaseExperimentResult {
    /// Size of the first batch seen by the run.
    pub batch_size: u16,
    /// Best held-out accuracy across epochs, falling back to the best
    /// training accuracy when the run never validated.
    pub best_val_acc: f64,
    /// Epoch index at which `best_val_acc` was reached.
    pub best_epoch: u32,
    /// Held-out accuracy of the last epoch that validated.
    pub final_val_acc: f64,
    /// Training accuracy of the last epoch.
    pub final_train_acc: f64,
    /// `final_train_acc - final_val_acc`.
    pub train_val_gap: f64,
    /// Wall time of the whole run, in minutes.
    pub training_time_minutes: f64,
    /// Epochs completed per minute of wall time.
    pub epochs_per_minute: f64,
    /// Training samples processed per second of wall time.
    pub samples_per_second: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_phase_default_is_standard() {
        assert_eq!(TrainingPhase::default(), TrainingPhase::Standard);
    }

    #[test]
    fn training_phase_serde_roundtrip() {
        let phase = TrainingPhase::Standard;
        let json = serde_json::to_string(&phase).unwrap();
        assert_eq!(json, "\"standard\"");
        let recovered: TrainingPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(phase, recovered);
    }

    #[test]
    fn epoch_metrics_serde_roundtrip_with_options() {
        let with_options = EpochMetrics {
            epoch: 5,
            phase: TrainingPhase::Standard,
            train_loss: 0.42,
            val_a_as_a: Some(0.91),
            val_loss: Some(0.55),
            comm_snapshot: Some(vec![0.1, 0.2, 0.3]),
            ..Default::default()
        };

        let json = serde_json::to_string(&with_options).unwrap();
        let recovered: EpochMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(with_options, recovered);
    }

    #[test]
    fn epoch_metrics_skips_absent_val_loss() {
        let json = serde_json::to_string(&EpochMetrics::default()).unwrap();
        assert!(!json.contains("val_loss"));
    }

    #[test]
    fn base_experiment_result_serde_roundtrip() {
        let original = BaseExperimentResult {
            batch_size: 64,
            best_val_acc: 91.5,
            best_epoch: 7,
            final_val_acc: 90.0,
            final_train_acc: 95.0,
            train_val_gap: 5.0,
            training_time_minutes: 3.25,
            epochs_per_minute: 2.0,
            samples_per_second: 1200.0,
        };

        let json = serde_json::to_string(&original).unwrap();
        let recovered: BaseExperimentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(original, recovered);
    }
}
