//! Full-train engine.
//!
//! Premises:
//! - Training uses the whole training set; the canonical evaluation split is
//!   the per-epoch signal. There is no third split, so no separate test
//!   accuracy is reported.
//! - The batch order of epoch `e` is the permutation seeded with
//!   `run_seed + e * 10_000`.

use std::path::Path;
use std::time::Instant;

use cosmos_rs::models::count_state_parameters;
use cosmos_rs::prelude::{
    DeviceBackend, EvolutionConfig, EvolutionGovernor, FlatClassificationAdapter,
    FlatClassificationAdapterConfig, TrainError,
};

use crate::config::Cell;
use crate::datasets::{Dataset, epoch_seed, flat_batches, shuffled_indices};
use crate::engine::observer::{DiagnosticBatch, GeometryObserver};
use crate::engine::{
    EngineError, artifact_path, build_flat_model, build_loss, build_rules, relative_artifact_path,
};
use crate::schema::{
    EpochProfile, EpochRecord, MaxNorm, SuiteArtifact, SuiteConfig, SuiteResult, write_json,
};

/// Run one cell on the full-train engine and write its artifact.
///
/// # Errors
///
/// Returns [`EngineError`] when the model or rule cannot be built, when the
/// governor rejects the run, or when the artifact cannot be written.
pub fn run(cell: &Cell, data: &Dataset, out_dir: &Path) -> Result<SuiteArtifact, EngineError> {
    let input_dim = data.train_inputs.ncols();
    let mut model = build_flat_model(cell, input_dim)?;
    let num_blocks = model.get_state().num_blocks();
    let mut law = build_rules(cell, num_blocks)?;

    let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: build_loss(cell)?,
        ..FlatClassificationAdapterConfig::default()
    })
    .map_err(|err| EngineError::Train(err.to_string()))?;

    let eval_batches = flat_batches(
        &data.test_inputs,
        &data.test_labels,
        &(0..data.test_labels.len()).collect::<Vec<usize>>(),
        cell.batch_size,
        false,
    );

    let diagnostic_batch = if cell.diagnostics.flow_diagnostics {
        Some(DiagnosticBatch::take(&data.test_inputs, &data.test_labels))
    } else {
        None
    };

    let mut observer = GeometryObserver::new(
        model.get_state(),
        cell.diagnostics.snapshot_every,
        cell.diagnostics.compute_block_every,
        diagnostic_batch,
        cell.model.num_classes,
    )
    .map_err(|err| EngineError::Train(err.to_string()))?;
    observer.set_total_epochs(cell.epochs);

    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: cell.epochs,
            validate_every: Some(1),
            collect_traces: false,
        },
        DeviceBackend::default(),
    )
    .map_err(|err| EngineError::Train(err.to_string()))?;

    let train_inputs = &data.train_inputs;
    let train_labels = &data.train_labels;
    let batch_size = cell.batch_size;
    let seed = cell.seed;
    let provider = move |epoch: u32| -> Result<Vec<_>, TrainError> {
        let order = shuffled_indices(train_labels.len(), epoch_seed(seed, epoch));
        Ok(flat_batches(
            train_inputs,
            train_labels,
            &order,
            batch_size,
            false,
        ))
    };

    let started = Instant::now();
    let artifacts = governor
        .fit_with_observer_and_train_provider(
            &mut model,
            &mut law,
            &mut adapter,
            provider,
            Some(&eval_batches),
            &mut observer,
        )
        .map_err(|err| EngineError::Train(err.to_string()))?;
    let elapsed = started.elapsed().as_secs_f64();

    let history = observer.history;
    let profiles: Vec<EpochProfile> = artifacts
        .epoch_profiles
        .iter()
        .map(|profile| EpochProfile {
            train_epoch_total_sec: profile.train_epoch_total_sec,
            eval_epoch_total_sec: profile.eval_epoch_total_sec,
            epoch_total_sec: profile.epoch_total_sec,
            evolution_step_sec: profile.evolution.total_step_sec,
            evolution_retraction_sec: profile.evolution.retraction_sec,
            retraction_steps: profile.evolution.retraction_count,
        })
        .collect();

    let mut epochs = Vec::with_capacity(history.epoch.len());
    for (index, &epoch) in history.epoch.iter().enumerate() {
        epochs.push(EpochRecord {
            epoch,
            time_epoch_s: history.time_epoch_s[index],
            train: history.train_acc[index],
            val: history.val_acc[index],
            profile: profiles.get(index).cloned().unwrap_or_default(),
            per_block_distances: observer
                .distances
                .iter()
                .find(|(recorded, _)| *recorded == epoch)
                .map(|(_, distances)| distances.clone()),
        });
    }

    let summary = artifacts.summary;
    let result = SuiteResult {
        batch_size: cell.batch_size,
        best_val_acc: summary.best_val_acc,
        best_epoch: summary.best_epoch,
        final_val_acc: summary.final_val_acc,
        final_train_acc: summary.final_train_acc,
        train_val_gap: summary.train_val_gap,
        training_time_minutes: elapsed / 60.0,
        epochs_per_minute: summary.epochs_per_minute,
        samples_per_second: summary.samples_per_second,
    };

    let artifact = SuiteArtifact {
        output_path: relative_artifact_path(cell),
        config: SuiteConfig {
            experiment_name: cell.experiment.clone(),
            engine: cell.engine.to_string(),
            seed: cell.seed,
            retraction_interval: cell.rules.retraction_interval,
            model_variant: cell.model.variant.clone(),
            total_epochs: cell.epochs,
            batch_size: cell.batch_size,
            input_dim,
            input_subspace_dim: cell.model.input_subspace_dim,
            interior_subspace_dim: cell.model.interior_subspace_dim,
            num_interior_blocks: cell.model.num_interior_blocks,
            num_classes: cell.model.num_classes,
            cmr_k: cell.model.cmr_k,
            topology: cell.model.topology.clone(),
            lr_gauge: cell.rules.lr_gauge,
            momentum: cell.rules.momentum,
            momentum_max_norm: MaxNorm(cell.rules.momentum_max_norm),
            exp_method: cell.rules.exp_method.clone(),
            svd_driver: cell.rules.svd_driver.clone(),
            update_method: cell.rules.update_method.clone(),
            retraction_method: cell.rules.retraction_method.clone(),
            cmr_margin: cell.objective.cmr_margin,
            loss: cell.objective.loss.clone(),
            activation: cell.model.activation.clone(),
            activation_placement: cell.model.activation_placement.clone(),
            subspace_action: cell.model.action.clone(),
        },
        result,
        runtime_profile: profiles,
        history,
        epochs,
        notes: vec![format!(
            "state parameters: {}",
            count_state_parameters(model.get_state())
        )],
        class_angle_matrix_final: observer.class_angle_matrix_final,
    };

    write_json(&artifact_path(cell, out_dir), &artifact)?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{expand, load};
    use ndarray::Array2;

    fn synthetic(samples: usize, dim: usize, classes: usize) -> (Array2<f64>, Vec<usize>) {
        let inputs = Array2::from_shape_fn((samples, dim), |(row, col)| {
            let class = row % classes;
            let centre = ((class * 37 + col * 11) % 97) as f64 / 97.0;
            centre + ((row * 31 + col * 7) % 13) as f64 / 260.0
        });
        let labels = (0..samples).map(|row| row % classes).collect();
        (inputs, labels)
    }

    fn short_cell() -> Cell {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
            .join("e1.toml");
        let file = load(&path).expect("cell file");
        let mut cell = expand(&file, Some(&[319_635_789]), true)
            .expect("expand")
            .remove(0);
        cell.epochs = 2;
        cell.batch_size = 512;
        cell.model.input_subspace_dim = 8;
        cell.model.interior_subspace_dim = 4;
        cell.model.num_interior_blocks = 2;
        cell.model.cmr_k = 2;
        cell
    }

    fn tiny_dataset(dim: usize) -> Dataset {
        let (train_inputs, train_labels) = synthetic(512, dim, 10);
        let (test_inputs, test_labels) = synthetic(256, dim, 10);
        Dataset {
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
        }
    }

    fn out_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join("cosmos-paper-tests").join(name)
    }

    #[test]
    fn two_epoch_run_records_a_consistent_history() {
        let cell = short_cell();
        let data = tiny_dataset(64);
        let out = out_dir("suite_engine");
        let artifact = run(&cell, &data, &out).expect("run");

        assert_eq!(artifact.history.epoch, vec![1, 2]);
        let epochs = artifact.history.epoch.len();
        assert_eq!(
            artifact.history.per_block.total_orth_error_mean.len(),
            epochs
        );
        let blocks = artifact.history.per_block.connection_norms_mean.len() / epochs;
        assert_eq!(
            artifact.history.per_block.connection_norms_mean.len(),
            blocks * epochs
        );
        assert_eq!(
            artifact.history.per_block.orthogonality_errors_mean.len(),
            blocks * epochs
        );
        assert_eq!(artifact.history.per_block.drift_mean.len(), blocks * epochs);
        assert_eq!(artifact.history.evolution_scalars.lr.len(), epochs);
        assert!(
            artifact
                .history
                .per_block
                .total_orth_error_mean
                .iter()
                .all(|error| *error < 1e-10)
        );
        assert!(artifact.epochs[0].per_block_distances.is_some());
        assert_eq!(artifact.config.engine, "suite");
        assert_eq!(artifact.config.input_dim, 64);
        assert_eq!(artifact.runtime_profile.len(), epochs);
        assert!(artifact.history.val_acc.iter().all(|acc| acc.is_finite()));
        assert!(artifact.class_angle_matrix_final.is_none());

        let written = artifact_path(&cell, &out);
        assert!(written.exists());
        assert!(written.ends_with(&artifact.output_path));
        let back: SuiteArtifact = crate::schema::read_json(&written).expect("read back");
        assert_eq!(back.history.epoch, artifact.history.epoch);
    }

    #[test]
    fn flow_diagnostics_fill_the_angle_history_and_final_matrix() {
        let mut cell = short_cell();
        cell.diagnostics.flow_diagnostics = true;
        let data = tiny_dataset(64);
        let artifact = run(&cell, &data, &out_dir("suite_engine_flow")).expect("run");

        assert_eq!(artifact.history.flow_ratio_overall.len(), 2);
        assert_eq!(artifact.history.projection_ratio_mean.len(), 2);
        assert_eq!(artifact.history.mean_angle_correct.len(), 2);
        assert_eq!(artifact.history.mean_angle_wrong_min.len(), 2);
        assert_eq!(artifact.history.cmr_margin.len(), 2);
        for index in 0..2 {
            let margin = artifact.history.mean_angle_wrong_min[index]
                - artifact.history.mean_angle_correct[index];
            assert!((artifact.history.cmr_margin[index] - margin).abs() < 1e-12);
        }
        let matrix = artifact
            .class_angle_matrix_final
            .expect("final angle matrix");
        assert_eq!(matrix.len(), 10);
        assert!(matrix.iter().all(|row| row.len() == 10));
        assert!(matrix.iter().flatten().all(|angle| angle.is_finite()));
    }
}
