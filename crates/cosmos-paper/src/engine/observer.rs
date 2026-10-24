//! Per-epoch geometry, flow and class-angle observer.
//!
//! Premises:
//! - The diagnostics run on a fixed evaluation batch taken once before the
//!   run and reused at every epoch, so the epoch-to-epoch difference is the
//!   model's, not the sample's.
//! - The diagnostic pass reads state only and uses its own adapter, so the
//!   training path is untouched.

use cosmos_rs::diagnostics::geometry::{
    compute_block_geometry_from_state, snapshot_initial_blocks_from_state,
};
use cosmos_rs::geometry::grassmann::{geodesic_distance, projection_distance};
use cosmos_rs::prelude::{
    CosmosModel, CosmosModelPatchwise, CosmosState, EpochMetrics, FlatClassificationAdapter,
    FlatClassificationAdapterConfig, GaugeDynamics, TrainError,
};
use cosmos_rs::training::trainer::{EpochObserver, EpochTraceCapture};
use ndarray::Array2;

use crate::schema::{EvolutionScalars, PerBlockDistances, PerBlockHistory, Snapshot, SuiteHistory};

/// Upper bound on the number of evaluation samples the diagnostics use.
pub const DIAGNOSTIC_SAMPLES: usize = 1024;

/// Fixed evaluation batch used by the flow and angle diagnostics.
#[derive(Debug, Clone)]
pub struct DiagnosticBatch {
    /// Inputs, at most [`DIAGNOSTIC_SAMPLES`] rows.
    pub inputs: Array2<f64>,
    /// Labels matching `inputs`.
    pub labels: Vec<usize>,
}

impl DiagnosticBatch {
    /// Take the first [`DIAGNOSTIC_SAMPLES`] evaluation samples.
    ///
    /// # Panics
    ///
    /// Panics when `labels` is shorter than the number of rows in `inputs`.
    pub fn take(inputs: &Array2<f64>, labels: &[usize]) -> Self {
        let rows = inputs.nrows().min(DIAGNOSTIC_SAMPLES);
        assert!(
            labels.len() >= rows,
            "labels must cover every evaluation row"
        );
        Self {
            inputs: inputs.slice(ndarray::s![..rows, ..]).to_owned(),
            labels: labels[..rows].to_vec(),
        }
    }
}

/// Accumulates the per-epoch history of a run.
#[derive(Debug)]
pub struct GeometryObserver {
    initial_blocks: Vec<Array2<f64>>,
    snapshot_every: u32,
    compute_block_every: u32,
    diagnostic_batch: Option<DiagnosticBatch>,
    adapter: FlatClassificationAdapter,
    num_classes: usize,
    total_epochs: u32,
    /// History accumulated so far.
    pub history: SuiteHistory,
    /// Block distances recorded at the configured cadence.
    pub distances: Vec<(u32, PerBlockDistances)>,
    /// Angle matrix from the last epoch, when the diagnostics are enabled.
    pub class_angle_matrix_final: Option<Vec<Vec<f64>>>,
}

impl GeometryObserver {
    /// Create an observer over the state's current bases.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the diagnostic adapter cannot be built.
    pub fn new(
        state: &CosmosState,
        snapshot_every: u32,
        compute_block_every: u32,
        diagnostic_batch: Option<DiagnosticBatch>,
        num_classes: usize,
    ) -> Result<Self, TrainError> {
        Ok(Self {
            initial_blocks: snapshot_initial_blocks_from_state(state),
            snapshot_every,
            compute_block_every,
            diagnostic_batch,
            adapter: FlatClassificationAdapter::new(FlatClassificationAdapterConfig::default())?,
            num_classes,
            total_epochs: u32::MAX,
            history: SuiteHistory::default(),
            distances: Vec::new(),
            class_angle_matrix_final: None,
        })
    }

    /// Total epochs the run will execute, needed to detect the last one.
    ///
    /// Until this is called the observer treats no epoch as the last, so the
    /// final angle matrix is simply never recorded.
    pub fn set_total_epochs(&mut self, total: u32) {
        self.total_epochs = total;
    }

    fn record_state(
        &mut self,
        state: &CosmosState,
        law: &GaugeDynamics,
        metrics: &EpochMetrics,
    ) -> Result<(), TrainError> {
        let geometry = compute_block_geometry_from_state(state, &self.initial_blocks, law)
            .map_err(|err| TrainError::task_adapter(err.to_string()))?;
        let diagnostics = law.get_diagnostics(state);

        self.history.epoch.push(metrics.epoch);
        self.history.time_epoch_s.push(metrics.time_sec);
        self.history.train_acc.push(metrics.train_acc);
        self.history
            .val_acc
            .push(metrics.val_a_as_a.unwrap_or(f64::NAN));
        self.history.loss.push(metrics.train_loss);
        self.history.mean_commutator.push(metrics.mean_commutator);

        let per_block: &mut PerBlockHistory = &mut self.history.per_block;
        per_block
            .connection_norms_mean
            .extend_from_slice(&diagnostics.connection_norms);
        per_block
            .commutator_norms_mean
            .extend_from_slice(&diagnostics.commutator_norms);
        per_block
            .orthogonality_errors_mean
            .extend_from_slice(&geometry.orth_error);
        per_block
            .total_orth_error_mean
            .push(diagnostics.total_orth_error);
        per_block.drift_mean.extend_from_slice(&geometry.drift);

        let scalars: &mut EvolutionScalars = &mut self.history.evolution_scalars;
        scalars.mean_comm_norm.push(diagnostics.mean_comm_norm);
        scalars.mean_comm_target.push(diagnostics.mean_comm_target);
        scalars.mean_geom_error.push(diagnostics.mean_geom_error);
        scalars
            .deadband_fraction
            .push(diagnostics.deadband_fraction);
        scalars.global_step.push(diagnostics.global_step);
        scalars.lr.push(diagnostics.lr);

        if self.snapshot_every > 0 && metrics.epoch % self.snapshot_every == 0 {
            self.history.snapshots.push(Snapshot {
                epoch: metrics.epoch,
                commutator_norms: diagnostics.commutator_norms.clone(),
                connection_norms: diagnostics.connection_norms.clone(),
                orthogonality_errors: diagnostics.orthogonality_errors.clone(),
                total_orth_error: diagnostics.total_orth_error,
            });
        }

        if self.compute_block_every > 0 && metrics.epoch % self.compute_block_every == 0 {
            let mut projection = Vec::with_capacity(self.initial_blocks.len());
            let mut geodesic = Vec::with_capacity(self.initial_blocks.len());
            for (index, initial) in self.initial_blocks.iter().enumerate() {
                let current = state
                    .get_basis(index)
                    .map_err(|err| TrainError::task_adapter(err.to_string()))?;
                projection.push(
                    projection_distance(initial, current)
                        .map_err(|err| TrainError::task_adapter(err.to_string()))?,
                );
                geodesic.push(
                    geodesic_distance(initial, current)
                        .map_err(|err| TrainError::task_adapter(err.to_string()))?,
                );
            }
            self.distances.push((
                metrics.epoch,
                PerBlockDistances {
                    projection_distance_from_init: projection,
                    grassmann_distance_from_init: geodesic,
                },
            ));
        }
        Ok(())
    }

    fn record_flat_diagnostics(
        &mut self,
        model: &CosmosModel,
        is_last_epoch: bool,
    ) -> Result<(), TrainError> {
        let Some(batch) = self.diagnostic_batch.clone() else {
            return Ok(());
        };

        let (per_stage, overall) = model
            .interior_flow_norms(&batch.inputs)
            .map_err(|err| TrainError::task_adapter(err.to_string()))?;
        let ratios: Vec<f64> = per_stage
            .iter()
            .map(|norm| norm.mean_out_norm / norm.mean_in_norm.max(1e-12))
            .collect();
        let projection_ratio = if ratios.is_empty() {
            f64::NAN
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        };
        self.history.flow_ratio_overall.push(overall);
        self.history.projection_ratio_mean.push(projection_ratio);

        let angles = self.adapter.class_angles(model, &batch.inputs)?;
        let samples = angles.nrows();
        let mut correct_sum = 0.0;
        let mut wrong_sum = 0.0;
        for (row, &label) in batch.labels.iter().enumerate() {
            correct_sum += angles[[row, label]];
            let mut smallest = f64::INFINITY;
            for class in 0..self.num_classes {
                if class != label {
                    smallest = smallest.min(angles[[row, class]]);
                }
            }
            wrong_sum += smallest;
        }
        let mean_correct = correct_sum / samples as f64;
        let mean_wrong = wrong_sum / samples as f64;
        self.history.mean_angle_correct.push(mean_correct);
        self.history.mean_angle_wrong_min.push(mean_wrong);
        self.history.cmr_margin.push(mean_wrong - mean_correct);

        if is_last_epoch {
            let mut matrix = vec![vec![0.0; self.num_classes]; self.num_classes];
            let mut counts = vec![0usize; self.num_classes];
            for (row, &label) in batch.labels.iter().enumerate() {
                counts[label] += 1;
                for class in 0..self.num_classes {
                    matrix[label][class] += angles[[row, class]];
                }
            }
            for (label, count) in counts.iter().enumerate() {
                if *count > 0 {
                    for class in 0..self.num_classes {
                        matrix[label][class] /= *count as f64;
                    }
                }
            }
            self.class_angle_matrix_final = Some(matrix);
        }
        Ok(())
    }
}

impl EpochObserver<CosmosModel> for GeometryObserver {
    fn on_epoch_end(
        &mut self,
        model: &CosmosModel,
        evolution_law: &GaugeDynamics,
        metrics: &EpochMetrics,
        _trace_capture: &EpochTraceCapture,
    ) -> Result<(), TrainError> {
        self.record_state(model.get_state(), evolution_law, metrics)?;
        let is_last = metrics.epoch >= self.total_epochs;
        self.record_flat_diagnostics(model, is_last)
    }
}

impl EpochObserver<CosmosModelPatchwise> for GeometryObserver {
    fn on_epoch_end(
        &mut self,
        model: &CosmosModelPatchwise,
        evolution_law: &GaugeDynamics,
        metrics: &EpochMetrics,
        _trace_capture: &EpochTraceCapture,
    ) -> Result<(), TrainError> {
        self.record_state(model.get_state(), evolution_law, metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_diagnostic_batch_is_capped() {
        let inputs = Array2::from_shape_fn((2048, 4), |(row, col)| (row + col) as f64);
        let labels: Vec<usize> = (0..2048).map(|row| row % 10).collect();
        let batch = DiagnosticBatch::take(&inputs, &labels);
        assert_eq!(batch.inputs.nrows(), DIAGNOSTIC_SAMPLES);
        assert_eq!(batch.labels.len(), DIAGNOSTIC_SAMPLES);
        assert_eq!(batch.inputs[[3, 1]], 4.0);
    }

    #[test]
    fn a_short_evaluation_split_is_taken_whole() {
        let inputs = Array2::from_shape_fn((7, 3), |(row, col)| (row * 3 + col) as f64);
        let labels: Vec<usize> = (0..7).map(|row| row % 2).collect();
        let batch = DiagnosticBatch::take(&inputs, &labels);
        assert_eq!(batch.inputs.nrows(), 7);
        assert_eq!(batch.labels, labels);
    }
}
