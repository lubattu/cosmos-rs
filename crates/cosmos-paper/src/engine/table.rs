//! Three-way split engine.
//!
//! Premises:
//! - The partition is drawn with a split seed that does not depend on the
//!   run seed, so every arm and every seed of a comparison sees the same
//!   training, validation and test sets.
//! - The reported number is the test accuracy of the epoch with the best
//!   validation accuracy, so the test split never selects a model.
//! - Every arm reports the same quantity as its epoch time: the training
//!   pass alone, with validation and test evaluation excluded.

use std::path::Path;
use std::time::Instant;

use cosmos_rs::models::count_state_parameters;
use cosmos_rs::prelude::{
    CosmosModel, DeviceBackend, EvolutionConfig, EvolutionGovernor, FlatClassificationAdapter,
    FlatClassificationAdapterConfig, FlatClassificationBatch, PatchwiseClassificationAdapter,
    PatchwiseClassificationAdapterConfig, TaskAdapter,
};

use crate::baselines::{Arm, mlp::Mlp, rsgd};
use crate::config::Cell;
use crate::datasets::{
    Dataset, epoch_seed, flat_batches, patch_batches, shuffled_indices, split_indices,
};
use crate::engine::{
    EngineError, artifact_path, build_flat_model, build_loss, build_patchwise_model, build_rules,
};
use crate::schema::{MaxNorm, TableArtifact, TableConfig, TableHistory, TableResult, write_json};

#[derive(Debug, Default)]
struct BestTracker {
    best_val_acc: f64,
    best_epoch: u32,
    test_acc_at_best_val: f64,
}

impl BestTracker {
    fn observe(&mut self, epoch: u32, val_acc: f64, test_acc: f64) {
        if epoch == 1 || val_acc > self.best_val_acc {
            self.best_val_acc = val_acc;
            self.best_epoch = epoch;
            self.test_acc_at_best_val = test_acc;
        }
    }
}

struct Partition {
    train: Vec<usize>,
    val: Vec<usize>,
    test: Vec<usize>,
    drop_last: bool,
}

fn partition(cell: &Cell, data: &Dataset) -> Result<Partition, EngineError> {
    let split = cell.split.as_ref().ok_or(EngineError::Unknown {
        field: "split",
        value: "absent".to_string(),
    })?;
    let (train, val) = split_indices(data.train_labels.len(), split.val_size, split.split_seed);
    Ok(Partition {
        train,
        val,
        test: (0..data.test_labels.len()).collect(),
        drop_last: split.drop_last,
    })
}

fn epoch_order(base: &[usize], seed: u64, epoch: u32) -> Vec<usize> {
    let permutation = shuffled_indices(base.len(), epoch_seed(seed, epoch));
    permutation.into_iter().map(|index| base[index]).collect()
}

fn run_cosmos(
    cell: &Cell,
    data: &Dataset,
    parts: &Partition,
) -> Result<(TableResult, TableHistory, usize), EngineError> {
    let mut history = TableHistory::default();
    let mut tracker = BestTracker::default();
    let started = Instant::now();
    let mut epoch_seconds = Vec::new();

    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: 1,
            validate_every: Some(1),
            collect_traces: false,
        },
        DeviceBackend::default(),
    )
    .map_err(|err| EngineError::Train(err.to_string()))?;

    let param_count;
    if cell.model.variant == "patchwise" {
        let shape = cell.model.image_shape.ok_or(EngineError::Unknown {
            field: "model.image_shape",
            value: "absent".to_string(),
        })?;
        let shape = (shape[0], shape[1], shape[2]);
        let mut model = build_patchwise_model(cell)?;
        param_count = count_state_parameters(model.get_state());
        let mut law = build_rules(cell, model.get_state().num_blocks())?;
        let mut adapter =
            PatchwiseClassificationAdapter::new(PatchwiseClassificationAdapterConfig {
                loss: build_loss(cell)?,
                ..PatchwiseClassificationAdapterConfig::default()
            })
            .map_err(|err| EngineError::Train(err.to_string()))?;

        let val_batches = patch_batches(
            &data.train_inputs,
            &data.train_labels,
            &parts.val,
            cell.batch_size,
            false,
            shape,
        );
        let test_batches = patch_batches(
            &data.test_inputs,
            &data.test_labels,
            &parts.test,
            cell.batch_size,
            false,
            shape,
        );

        for epoch in 1..=cell.epochs {
            let order = epoch_order(&parts.train, cell.seed, epoch);
            let train_batches = patch_batches(
                &data.train_inputs,
                &data.train_labels,
                &order,
                cell.batch_size,
                parts.drop_last,
                shape,
            );
            let artifacts = governor
                .fit(
                    &mut model,
                    &mut law,
                    &mut adapter,
                    &train_batches,
                    Some(&val_batches),
                )
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let seconds = artifacts.epoch_profiles[0].train_epoch_total_sec;
            let metrics = artifacts.epoch_metrics[0].clone();
            let test_artifacts = governor
                .evaluate_only(&mut model, &mut adapter, &test_batches, epoch)
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let test_acc = test_artifacts.epoch_metrics[0]
                .val_a_as_a
                .unwrap_or(f64::NAN);
            record_epoch(
                &mut history,
                &mut tracker,
                &mut epoch_seconds,
                epoch,
                metrics.train_acc,
                metrics.val_a_as_a.unwrap_or(f64::NAN),
                test_acc,
                metrics.train_loss,
                seconds,
            );
        }
    } else {
        let mut model = build_flat_model(cell, data.train_inputs.ncols())?;
        param_count = count_state_parameters(model.get_state());
        let mut law = build_rules(cell, model.get_state().num_blocks())?;
        let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
            loss: build_loss(cell)?,
            ..FlatClassificationAdapterConfig::default()
        })
        .map_err(|err| EngineError::Train(err.to_string()))?;

        let val_batches = flat_batches(
            &data.train_inputs,
            &data.train_labels,
            &parts.val,
            cell.batch_size,
            false,
        );
        let test_batches = flat_batches(
            &data.test_inputs,
            &data.test_labels,
            &parts.test,
            cell.batch_size,
            false,
        );

        for epoch in 1..=cell.epochs {
            let order = epoch_order(&parts.train, cell.seed, epoch);
            let train_batches = flat_batches(
                &data.train_inputs,
                &data.train_labels,
                &order,
                cell.batch_size,
                parts.drop_last,
            );
            let artifacts = governor
                .fit(
                    &mut model,
                    &mut law,
                    &mut adapter,
                    &train_batches,
                    Some(&val_batches),
                )
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let seconds = artifacts.epoch_profiles[0].train_epoch_total_sec;
            let metrics = artifacts.epoch_metrics[0].clone();
            let test_artifacts = governor
                .evaluate_only(&mut model, &mut adapter, &test_batches, epoch)
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let test_acc = test_artifacts.epoch_metrics[0]
                .val_a_as_a
                .unwrap_or(f64::NAN);
            record_epoch(
                &mut history,
                &mut tracker,
                &mut epoch_seconds,
                epoch,
                metrics.train_acc,
                metrics.val_a_as_a.unwrap_or(f64::NAN),
                test_acc,
                metrics.train_loss,
                seconds,
            );
        }
    }

    Ok((
        summarize(
            cell,
            &history,
            &tracker,
            &epoch_seconds,
            started,
            param_count,
        ),
        history,
        param_count,
    ))
}

fn run_mlp(
    cell: &Cell,
    data: &Dataset,
    parts: &Partition,
) -> Result<(TableResult, TableHistory, usize), EngineError> {
    let baselines = cell.baselines.as_ref().ok_or(EngineError::Unknown {
        field: "baselines",
        value: "absent".to_string(),
    })?;
    let mut network = Mlp::new(
        data.train_inputs.ncols(),
        baselines.mlp_hidden,
        cell.model.num_classes,
        cell.seed,
    );
    let param_count = network.param_count();

    let val_batches = flat_batches(
        &data.train_inputs,
        &data.train_labels,
        &parts.val,
        cell.batch_size,
        false,
    );
    let test_batches = flat_batches(
        &data.test_inputs,
        &data.test_labels,
        &parts.test,
        cell.batch_size,
        false,
    );

    let mut history = TableHistory::default();
    let mut tracker = BestTracker::default();
    let mut epoch_seconds = Vec::new();
    let started = Instant::now();

    for epoch in 1..=cell.epochs {
        let order = epoch_order(&parts.train, cell.seed, epoch);
        let train_batches = flat_batches(
            &data.train_inputs,
            &data.train_labels,
            &order,
            cell.batch_size,
            parts.drop_last,
        );
        let epoch_start = Instant::now();
        let (loss, train_acc) = network.train_epoch(&train_batches, baselines.mlp_lr);
        let seconds = epoch_start.elapsed().as_secs_f64();
        let val_acc = network.evaluate(&val_batches);
        let test_acc = network.evaluate(&test_batches);
        record_epoch(
            &mut history,
            &mut tracker,
            &mut epoch_seconds,
            epoch,
            train_acc,
            val_acc,
            test_acc,
            loss,
            seconds,
        );
    }

    Ok((
        summarize(
            cell,
            &history,
            &tracker,
            &epoch_seconds,
            started,
            param_count,
        ),
        history,
        param_count,
    ))
}

fn run_rsgd(
    cell: &Cell,
    data: &Dataset,
    parts: &Partition,
) -> Result<(TableResult, TableHistory, usize), EngineError> {
    let baselines = cell.baselines.as_ref().ok_or(EngineError::Unknown {
        field: "baselines",
        value: "absent".to_string(),
    })?;
    let mut model = build_flat_model(cell, data.train_inputs.ncols())?;
    let param_count = count_state_parameters(model.get_state());
    let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: build_loss(cell)?,
        ..FlatClassificationAdapterConfig::default()
    })
    .map_err(|err| EngineError::Train(err.to_string()))?;
    let mut backend = DeviceBackend::default();

    let val_batches = flat_batches(
        &data.train_inputs,
        &data.train_labels,
        &parts.val,
        cell.batch_size,
        false,
    );
    let test_batches = flat_batches(
        &data.test_inputs,
        &data.test_labels,
        &parts.test,
        cell.batch_size,
        false,
    );

    let mut history = TableHistory::default();
    let mut tracker = BestTracker::default();
    let mut epoch_seconds = Vec::new();
    let started = Instant::now();

    for epoch in 1..=cell.epochs {
        let order = epoch_order(&parts.train, cell.seed, epoch);
        let train_batches = flat_batches(
            &data.train_inputs,
            &data.train_labels,
            &order,
            cell.batch_size,
            parts.drop_last,
        );
        let epoch_start = Instant::now();
        model
            .set_training(true)
            .map_err(|err| EngineError::Train(err.to_string()))?;
        let mut loss_sum = 0.0;
        let mut accuracy_sum = 0.0;
        let mut samples = 0usize;
        for (batch_idx, batch) in train_batches.iter().enumerate() {
            model
                .set_epoch_batch(u64::from(epoch), batch_idx as u64)
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let output = adapter
                .train_batch(&model, batch, &mut backend, false)
                .map_err(|err| EngineError::Train(err.to_string()))?;
            let batch_samples = output.metrics.samples;
            loss_sum += output.metrics.loss * batch_samples as f64;
            accuracy_sum += output.metrics.accuracy.unwrap_or(0.0) * batch_samples as f64;
            samples += batch_samples;
            rsgd::step(model.get_state_mut(), &output.gradients, baselines.rsgd_lr)
                .map_err(|err| EngineError::Train(err.to_string()))?;
        }
        let seconds = epoch_start.elapsed().as_secs_f64();
        let train_acc = accuracy_sum / samples.max(1) as f64;
        let loss = loss_sum / samples.max(1) as f64;
        let val_acc = evaluate(&model, &mut adapter, &mut backend, &val_batches)?;
        let test_acc = evaluate(&model, &mut adapter, &mut backend, &test_batches)?;
        record_epoch(
            &mut history,
            &mut tracker,
            &mut epoch_seconds,
            epoch,
            train_acc,
            val_acc,
            test_acc,
            loss,
            seconds,
        );
    }

    Ok((
        summarize(
            cell,
            &history,
            &tracker,
            &epoch_seconds,
            started,
            param_count,
        ),
        history,
        param_count,
    ))
}

fn evaluate(
    model: &CosmosModel,
    adapter: &mut FlatClassificationAdapter,
    backend: &mut DeviceBackend,
    batches: &[FlatClassificationBatch],
) -> Result<f64, EngineError> {
    model
        .set_training(false)
        .map_err(|err| EngineError::Train(err.to_string()))?;
    let mut correct_weight = 0.0;
    let mut samples = 0usize;
    for batch in batches {
        let output = adapter
            .eval_batch(model, batch, backend, false)
            .map_err(|err| EngineError::Train(err.to_string()))?;
        correct_weight += output.metrics.accuracy.unwrap_or(0.0) * output.metrics.samples as f64;
        samples += output.metrics.samples;
    }
    model
        .set_training(true)
        .map_err(|err| EngineError::Train(err.to_string()))?;
    Ok(correct_weight / samples.max(1) as f64)
}

fn record_epoch(
    history: &mut TableHistory,
    tracker: &mut BestTracker,
    epoch_seconds: &mut Vec<f64>,
    epoch: u32,
    train_acc: f64,
    val_acc: f64,
    test_acc: f64,
    loss: f64,
    seconds: f64,
) {
    history.epoch.push(epoch);
    history.train_acc.push(train_acc);
    history.val_acc.push(val_acc);
    history.test_acc.push(test_acc);
    history.loss.push(loss);
    history.train_epoch_s.push(seconds);
    epoch_seconds.push(seconds);
    tracker.observe(epoch, val_acc, test_acc);
}

fn summarize(
    cell: &Cell,
    history: &TableHistory,
    tracker: &BestTracker,
    epoch_seconds: &[f64],
    started: Instant,
    param_count: usize,
) -> TableResult {
    let final_train = history.train_acc.last().copied().unwrap_or(f64::NAN);
    let final_test = history.test_acc.last().copied().unwrap_or(f64::NAN);
    TableResult {
        model_type: cell.arm.clone(),
        seed: cell.seed,
        param_count,
        best_val_acc: tracker.best_val_acc,
        best_epoch: tracker.best_epoch,
        test_acc_at_best_val: tracker.test_acc_at_best_val,
        final_val_acc: history.val_acc.last().copied().unwrap_or(f64::NAN),
        final_test_acc: final_test,
        final_train_acc: final_train,
        train_test_gap: final_train - final_test,
        mean_train_epoch_s: epoch_seconds.iter().sum::<f64>() / epoch_seconds.len().max(1) as f64,
        total_train_time_min: started.elapsed().as_secs_f64() / 60.0,
    }
}

/// Run one cell on the three-way split engine and write its artifact.
///
/// # Errors
///
/// Returns [`EngineError`] when the arm is unknown, the model or rule
/// cannot be built, the run fails, or the artifact cannot be written.
pub fn run(cell: &Cell, data: &Dataset, out_dir: &Path) -> Result<TableArtifact, EngineError> {
    let parts = partition(cell, data)?;
    let arm = Arm::parse(&cell.arm).map_err(|value| EngineError::Unknown {
        field: "arm",
        value,
    })?;
    let (result, history, _) = match arm {
        Arm::Cosmos => run_cosmos(cell, data, &parts)?,
        Arm::Mlp => run_mlp(cell, data, &parts)?,
        Arm::Rsgd => run_rsgd(cell, data, &parts)?,
    };

    let split = cell.split.as_ref().ok_or(EngineError::Unknown {
        field: "split",
        value: "absent".to_string(),
    })?;
    let baselines = cell.baselines.clone();
    let artifact = TableArtifact {
        config: TableConfig {
            model_type: cell.arm.clone(),
            engine: cell.engine.to_string(),
            dataset: cell.dataset.to_string(),
            input_dim: data.train_inputs.ncols(),
            input_subspace_dim: cell.model.input_subspace_dim,
            interior_subspace_dim: cell.model.interior_subspace_dim,
            num_interior_blocks: cell.model.num_interior_blocks,
            num_classes: cell.model.num_classes,
            cmr_k: cell.model.cmr_k,
            activation: cell.model.activation.clone(),
            activation_placement: cell.model.activation_placement.clone(),
            topology_str: cell.model.topology.clone(),
            cmr_margin: cell.objective.cmr_margin,
            loss: cell.objective.loss.clone(),
            total_epochs: cell.epochs,
            batch_size: cell.batch_size,
            seed: cell.seed,
            split_seed: split.split_seed,
            val_size: split.val_size,
            drop_last: split.drop_last,
            lr_gauge: cell.rules.lr_gauge,
            momentum: cell.rules.momentum,
            momentum_max_norm: MaxNorm(cell.rules.momentum_max_norm),
            retraction_interval: cell.rules.retraction_interval,
            svd_driver: cell.rules.svd_driver.clone(),
            mlp_hidden: baselines.as_ref().map_or(0, |value| value.mlp_hidden),
            mlp_lr: baselines.as_ref().map_or(0.0, |value| value.mlp_lr),
            mlp_optimizer: "adam".to_string(),
            rsgd_lr: baselines.as_ref().map_or(0.0, |value| value.rsgd_lr),
            anchor_init: "keyed_gaussian_qr".to_string(),
            model_variant: cell.model.variant.clone(),
            image_shape: cell.model.image_shape,
            patch_size: cell.model.patch_size,
            patch_subspace_dim: cell.model.patch_subspace_dim,
            agg_subspace_dim: cell.model.agg_subspace_dim,
        },
        result,
        history,
    };

    write_json(&artifact_path(cell, out_dir), &artifact)?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{expand, load};
    use ndarray::Array2;

    fn out_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join("cosmos-paper-tests").join(name)
    }

    fn synthetic(samples: usize, dim: usize, classes: usize) -> (Array2<f64>, Vec<usize>) {
        let inputs = Array2::from_shape_fn((samples, dim), |(row, col)| {
            let class = row % classes;
            let centre = ((class * 37 + col * 11) % 97) as f64 / 97.0;
            centre + ((row * 31 + col * 7) % 13) as f64 / 260.0
        });
        (inputs, (0..samples).map(|row| row % classes).collect())
    }

    fn cell_for(arm: &str) -> Cell {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
            .join("e3_baselines.toml");
        let file = load(&path).expect("cell file");
        let mut cell = expand(&file, Some(&[319_635_789]), true)
            .expect("expand")
            .into_iter()
            .find(|candidate| candidate.arm == arm)
            .expect("arm");
        cell.epochs = 2;
        cell.batch_size = 256;
        if let Some(split) = cell.split.as_mut() {
            split.val_size = 128;
        }
        cell
    }

    fn data() -> Dataset {
        let (train_inputs, train_labels) = synthetic(1024, 784, 10);
        let (test_inputs, test_labels) = synthetic(256, 784, 10);
        Dataset {
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
        }
    }

    #[test]
    fn best_tracker_records_the_test_accuracy_of_the_best_validation_epoch() {
        let mut tracker = BestTracker::default();
        tracker.observe(1, 40.0, 39.0);
        tracker.observe(2, 55.0, 54.0);
        tracker.observe(3, 50.0, 60.0);
        assert_eq!(tracker.best_epoch, 2);
        assert!((tracker.best_val_acc - 55.0).abs() < 1e-12);
        assert!((tracker.test_acc_at_best_val - 54.0).abs() < 1e-12);
    }

    #[test]
    fn partition_is_shared_by_every_arm() {
        let dataset = data();
        let first = partition(&cell_for("cosmos"), &dataset).expect("partition");
        let second = partition(&cell_for("mlp"), &dataset).expect("partition");
        assert_eq!(first.train, second.train);
        assert_eq!(first.val, second.val);
    }

    #[test]
    fn two_epoch_cosmos_run_produces_a_valid_artifact() {
        let artifact = run(&cell_for("cosmos"), &data(), &out_dir("table_cosmos")).expect("run");
        assert_eq!(artifact.history.epoch, vec![1, 2]);
        assert_eq!(artifact.history.loss.len(), 2);
        assert_eq!(artifact.config.engine, "table");
        assert_eq!(artifact.config.split_seed, 12_345);
        assert_eq!(artifact.result.param_count, 150_528);
    }

    #[test]
    fn two_epoch_network_run_matches_the_expected_parameter_count() {
        let artifact = run(&cell_for("mlp"), &data(), &out_dir("table_mlp")).expect("run");
        assert_eq!(artifact.result.param_count, 150_265);
        assert_eq!(artifact.result.model_type, "mlp");
    }

    #[test]
    fn two_epoch_riemannian_run_records_params_and_epochs() {
        let artifact = run(&cell_for("rsgd"), &data(), &out_dir("table_rsgd")).expect("run");
        assert_eq!(artifact.result.param_count, 150_528);
        assert_eq!(artifact.history.epoch.len(), 2);
    }
}
