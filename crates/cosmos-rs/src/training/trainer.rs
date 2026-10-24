//! Generic CPU-first evolution governance for COSMOS models.

use std::time::Instant;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    compute::device::ComputeDevice,
    metrics::{BaseExperimentResult, EpochMetrics, TrainingPhase},
    models::{CosmosModel, CosmosModelPatchwise, ModelError},
    rules::{GaugeDynamics, GaugeDynamicsStepProfile},
    state::cosmos_state::CosmosState,
};

use super::{
    TrainError,
    backend::TrainingBackend,
    task_adapter::{
        BatchMetrics, BatchTimingProfile, EvalBatchOutput, TaskAdapter, TrainBatchOutput,
    },
    trace::BatchTrace,
};

/// Minimal model contract required by the evolution governor.
pub trait TrainingModel {
    /// Borrow the unified COSMOS state.
    fn state(&self) -> &CosmosState;

    /// Mutably borrow the unified COSMOS state.
    fn state_mut(&mut self) -> &mut CosmosState;

    /// Switch the model runtime between training and evaluation modes.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the model runtime rejects the call.
    fn set_training(&self, training: bool) -> Result<(), ModelError>;

    /// Notify the runtime that a new epoch has started.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the model runtime rejects the call.
    fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError>;

    /// Update the runtime with the current epoch and batch index.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the model runtime rejects the call.
    fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError>;
}

impl TrainingModel for CosmosModel {
    fn state(&self) -> &CosmosState {
        self.get_state()
    }

    fn state_mut(&mut self) -> &mut CosmosState {
        self.get_state_mut()
    }

    fn set_training(&self, training: bool) -> Result<(), ModelError> {
        self.set_training(training)
    }

    fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError> {
        self.on_epoch_start(epoch)
    }

    fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError> {
        self.set_epoch_batch(epoch, batch_idx)
    }
}

impl TrainingModel for CosmosModelPatchwise {
    fn state(&self) -> &CosmosState {
        self.get_state()
    }

    fn state_mut(&mut self) -> &mut CosmosState {
        self.get_state_mut()
    }

    fn set_training(&self, training: bool) -> Result<(), ModelError> {
        self.set_training(training)
    }

    fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError> {
        self.on_epoch_start(epoch)
    }

    fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError> {
        self.set_epoch_batch(epoch, batch_idx)
    }
}

/// Governor configuration shared across tasks.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionConfig {
    /// Number of epochs to execute.
    pub epochs: u32,
    /// Validation cadence in epochs. `None` disables validation.
    pub validate_every: Option<u32>,
    /// Whether adapters should retain the first batch trace of each split.
    pub collect_traces: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            validate_every: None,
            collect_traces: false,
        }
    }
}

impl EvolutionConfig {
    fn validate(&self) -> Result<(), TrainError> {
        if self.epochs == 0 {
            return Err(TrainError::InvalidConfig {
                field: "epochs",
                reason: "must be greater than zero".to_string(),
            });
        }

        if let Some(validate_every) = self.validate_every {
            if validate_every == 0 {
                return Err(TrainError::InvalidConfig {
                    field: "validate_every",
                    reason: "must be greater than zero when provided".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Optional trace snapshots retained for one epoch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EpochTraceCapture {
    /// First training-batch trace captured in the epoch.
    pub train: Option<BatchTrace>,
    /// First validation-batch trace captured in the epoch.
    pub eval: Option<BatchTrace>,
}

/// Aggregated adapter timings for one epoch split.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AdapterEpochProfile {
    /// Number of batches processed for this split.
    pub batch_count: usize,
    /// Total time spent in adapter forwards.
    pub forward_sec: f64,
    /// Total time spent in adapter backwards.
    pub backward_sec: f64,
    /// Total time spent in anchor-score backward kernels.
    pub backward_anchor_sec: f64,
    /// Total time spent in interior backward propagation.
    pub backward_interior_sec: f64,
    /// Total time spent in the input-projector basis gradient.
    pub backward_input_projector_sec: f64,
    /// Total time spent materializing traces.
    pub trace_sec: f64,
}

impl AdapterEpochProfile {
    fn record(&mut self, timing: BatchTimingProfile) {
        self.batch_count += 1;
        self.forward_sec += timing.forward_sec;
        self.backward_sec += timing.backward_sec;
        self.backward_anchor_sec += timing.backward_anchor_sec;
        self.backward_interior_sec += timing.backward_interior_sec;
        self.backward_input_projector_sec += timing.backward_input_projector_sec;
        self.trace_sec += timing.trace_sec;
    }
}

/// Aggregated evolution-law timings for one training epoch.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EvolutionEpochProfile {
    /// Number of evolution steps executed in the epoch.
    pub step_count: usize,
    /// Number of steps that triggered a retraction.
    pub retraction_count: usize,
    /// Time spent probing commutator norms before the evolution step.
    pub commutator_probe_sec: f64,
    /// Total time spent inside `GaugeDynamics::step`.
    pub total_step_sec: f64,
    /// Time spent building commutator-derived connections.
    pub connection_build_sec: f64,
    /// Time spent applying momentum/clipping.
    pub momentum_sec: f64,
    /// Time spent updating commutator-target statistics.
    pub comm_target_sec: f64,
    /// Time spent computing internal step diagnostics.
    pub diagnostics_sec: f64,
    /// Time spent caching connection norms.
    pub connection_norms_sec: f64,
    /// Time spent applying orthogonal updates.
    pub apply_updates_sec: f64,
    /// Time spent constructing orthogonal update matrices.
    pub update_exp_sec: f64,
    /// Time spent applying update matrices to the state.
    pub update_state_apply_sec: f64,
    /// Time spent inside periodic retraction.
    pub retraction_sec: f64,
    /// Orthogonality error of block 0 after each evolution step.
    pub batch_input_block_orth_error: Vec<f64>,
    /// Total orthogonality error across all blocks after each evolution step.
    pub batch_total_orth_error: Vec<f64>,
}

impl EvolutionEpochProfile {
    fn record_step(
        &mut self,
        step: GaugeDynamicsStepProfile,
        commutator_probe_sec: f64,
        input_block_orth_error: f64,
        total_orth_error: f64,
    ) {
        self.step_count += 1;
        self.commutator_probe_sec += commutator_probe_sec;
        self.total_step_sec += step.total_sec;
        self.connection_build_sec += step.connection_build_sec;
        self.momentum_sec += step.momentum_sec;
        self.comm_target_sec += step.comm_target_sec;
        self.diagnostics_sec += step.diagnostics_sec;
        self.connection_norms_sec += step.connection_norms_sec;
        self.apply_updates_sec += step.apply_updates_sec;
        self.update_exp_sec += step.update_exp_sec;
        self.update_state_apply_sec += step.update_state_apply_sec;
        self.retraction_sec += step.retraction_sec;
        self.retraction_count += usize::from(step.used_retraction);
        self.batch_input_block_orth_error
            .push(input_block_orth_error);
        self.batch_total_orth_error.push(total_orth_error);
    }
}

/// Runtime profile aggregated for one epoch.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EpochRuntimeProfile {
    /// Adapter timings for the training split.
    pub train_adapter: AdapterEpochProfile,
    /// Adapter timings for the evaluation split.
    pub eval_adapter: AdapterEpochProfile,
    /// Evolution-law timings captured during training.
    pub evolution: EvolutionEpochProfile,
    /// Time spent inside the observer hook after the epoch.
    pub observer_sec: f64,
    /// Wall time of the full training epoch.
    pub train_epoch_total_sec: f64,
    /// Wall time of the full evaluation epoch.
    pub eval_epoch_total_sec: f64,
    /// Total epoch wall time including observer work.
    pub epoch_total_sec: f64,
}

/// High-level run artifacts for one call to [`EvolutionGovernor::fit`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrainingArtifacts {
    /// Per-epoch metrics.
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Per-epoch runtime profile breakdowns.
    pub epoch_profiles: Vec<EpochRuntimeProfile>,
    /// Summary over the full run.
    pub summary: BaseExperimentResult,
    /// Optional trace captures per epoch.
    pub trace_captures: Vec<EpochTraceCapture>,
}

/// Observer hook fired after each epoch completes.
pub trait EpochObserver<M> {
    /// Called at the start of every epoch, before the model runtime is put
    /// into training mode and before any batch of that epoch is processed.
    ///
    /// Both the model and the evolution law are handed over mutably, so this
    /// is the seam for between-epoch changes to the arrangement or the
    /// evolver — an interior topology swap, a momentum reset — that must
    /// never land mid-epoch. Default implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the observer aborts the run; the governor
    /// propagates it without completing the epoch.
    fn on_epoch_start(
        &mut self,
        _model: &mut M,
        _evolution_law: &mut GaugeDynamics,
        _epoch: u32,
    ) -> Result<(), TrainError> {
        Ok(())
    }

    /// Called after the governor has built epoch metrics and captured traces.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the observer aborts the run; the governor
    /// propagates it without recording further epochs.
    fn on_epoch_end(
        &mut self,
        _model: &M,
        _evolution_law: &GaugeDynamics,
        _metrics: &EpochMetrics,
        _trace_capture: &EpochTraceCapture,
    ) -> Result<(), TrainError> {
        Ok(())
    }

    /// Called after every gauge-dynamics step with the report that step
    /// produced. This is the seam for per-step diagnostics that would
    /// otherwise require rewriting the inner training loop. Default
    /// implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError`] when the observer aborts the run; the governor
    /// propagates it without finishing the epoch.
    fn on_batch_end(
        &mut self,
        _model: &M,
        _evolution_law: &GaugeDynamics,
        _step_report: &crate::rules::GaugeDynamicsStepReport,
        _epoch: u32,
        _batch_idx: usize,
    ) -> Result<(), TrainError> {
        Ok(())
    }
}

/// Default no-op observer used by [`EvolutionGovernor::fit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoopEpochObserver;

impl<M> EpochObserver<M> for NoopEpochObserver {}

/// Generic evolution governor over a configurable numerical backend.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionGovernor<B> {
    config: EvolutionConfig,
    backend: B,
}

impl<B> EvolutionGovernor<B>
where
    B: TrainingBackend,
{
    /// Construct a new evolution governor.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidConfig`] when the configuration asks for
    /// zero epochs or a zero validation interval.
    pub fn new(config: EvolutionConfig, backend: B) -> Result<Self, TrainError> {
        config.validate()?;
        Ok(Self { config, backend })
    }

    /// Borrow the governor configuration.
    pub fn config(&self) -> &EvolutionConfig {
        &self.config
    }

    /// Borrow the numerical backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutably borrow the numerical backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Evolve a model using explicit per-block gradients returned by the task adapter.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::EmptyDataset`] when a required split is empty,
    /// and propagates any adapter, model or evolution-law failure.
    pub fn fit<M, A>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        train_batches: &[A::Batch],
        val_batches: Option<&[A::Batch]>,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
    {
        let mut observer = NoopEpochObserver;
        self.fit_impl(
            model,
            evolution_law,
            adapter,
            train_batches,
            val_batches,
            &mut observer,
        )
    }

    /// Evolve a model while notifying an epoch observer after each epoch.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::EmptyDataset`] when a required split is empty,
    /// and propagates any adapter, observer, model or evolution-law failure.
    pub fn fit_with_observer<M, A, O>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        train_batches: &[A::Batch],
        val_batches: Option<&[A::Batch]>,
        observer: &mut O,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
        O: EpochObserver<M>,
    {
        self.fit_impl(
            model,
            evolution_law,
            adapter,
            train_batches,
            val_batches,
            observer,
        )
    }

    fn fit_impl<M, A, O>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        train_batches: &[A::Batch],
        val_batches: Option<&[A::Batch]>,
        observer: &mut O,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
        O: EpochObserver<M>,
    {
        if train_batches.is_empty() {
            return Err(TrainError::EmptyDataset { split: "train" });
        }

        if let Some(val_batches) = val_batches {
            if self.config.validate_every.is_some() && val_batches.is_empty() {
                return Err(TrainError::EmptyDataset { split: "val" });
            }
        }

        let run_start = Instant::now();
        let mut epoch_metrics = Vec::with_capacity(self.config.epochs as usize);
        let mut epoch_profiles = Vec::with_capacity(self.config.epochs as usize);
        let mut trace_captures = Vec::with_capacity(self.config.epochs as usize);
        let mut total_train_samples = 0usize;
        let mut batch_size = 0u16;

        for epoch in 1..=self.config.epochs {
            observer.on_epoch_start(model, evolution_law, epoch)?;
            model.set_training(true)?;
            model.on_epoch_start(epoch as u64)?;

            let train_run = self.run_train_epoch_with_step_observer::<M, A, O>(
                model,
                evolution_law,
                adapter,
                train_batches,
                epoch,
                Some(observer),
            )?;
            total_train_samples += train_run.aggregates.samples;
            if batch_size == 0 {
                batch_size = train_run
                    .aggregates
                    .first_batch_size
                    .unwrap_or(0)
                    .min(u16::MAX as usize) as u16;
            }

            let eval_run = if self.should_validate(epoch) {
                let Some(val_batches) = val_batches else {
                    return Err(TrainError::EmptyDataset { split: "val" });
                };
                model.set_training(false)?;
                Some(self.run_eval_epoch(model, adapter, val_batches, epoch)?)
            } else {
                None
            };

            let phase = adapter.training_phase();
            let epoch_metric = build_epoch_metrics(epoch, phase, &train_run, eval_run.as_ref());
            let trace_capture = EpochTraceCapture {
                train: train_run.aggregates.first_trace.clone(),
                eval: eval_run
                    .as_ref()
                    .and_then(|run| run.aggregates.first_trace.clone()),
            };
            let observer_start = Instant::now();
            observer.on_epoch_end(model, evolution_law, &epoch_metric, &trace_capture)?;
            let observer_sec = observer_start.elapsed().as_secs_f64();
            let epoch_profile =
                build_epoch_runtime_profile(&train_run, eval_run.as_ref(), observer_sec);
            epoch_metrics.push(epoch_metric);
            epoch_profiles.push(epoch_profile);
            trace_captures.push(trace_capture);
        }

        Ok(TrainingArtifacts {
            summary: build_summary(
                &epoch_metrics,
                batch_size,
                total_train_samples,
                run_start.elapsed().as_secs_f64(),
            ),
            epoch_metrics,
            epoch_profiles,
            trace_captures,
        })
    }

    /// Evolve a model with per-epoch training batches generated on demand.
    ///
    /// This is useful for data pipelines that reshuffle or rebuild batches at
    /// every epoch while still reusing the generic governor loop.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::EmptyDataset`] when the provider yields no batch
    /// for an epoch or the validation split is empty, and propagates any
    /// adapter, observer, model or evolution-law failure.
    pub fn fit_with_observer_and_train_provider<M, A, O, F>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        train_batches_for_epoch: F,
        val_batches: Option<&[A::Batch]>,
        observer: &mut O,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
        O: EpochObserver<M>,
        F: FnMut(u32) -> Result<Vec<A::Batch>, TrainError>,
    {
        self.fit_train_provider_impl(
            model,
            evolution_law,
            adapter,
            train_batches_for_epoch,
            val_batches,
            observer,
        )
    }

    fn fit_train_provider_impl<M, A, O, F>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        mut train_batches_for_epoch: F,
        val_batches: Option<&[A::Batch]>,
        observer: &mut O,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
        O: EpochObserver<M>,
        F: FnMut(u32) -> Result<Vec<A::Batch>, TrainError>,
    {
        if let Some(val_batches) = val_batches {
            if self.config.validate_every.is_some() && val_batches.is_empty() {
                return Err(TrainError::EmptyDataset { split: "val" });
            }
        }

        let run_start = Instant::now();
        let mut epoch_metrics = Vec::with_capacity(self.config.epochs as usize);
        let mut epoch_profiles = Vec::with_capacity(self.config.epochs as usize);
        let mut trace_captures = Vec::with_capacity(self.config.epochs as usize);
        let mut total_train_samples = 0usize;
        let mut batch_size = 0u16;

        for epoch in 1..=self.config.epochs {
            let train_batches = train_batches_for_epoch(epoch)?;
            if train_batches.is_empty() {
                return Err(TrainError::EmptyDataset { split: "train" });
            }

            observer.on_epoch_start(model, evolution_law, epoch)?;
            model.set_training(true)?;
            model.on_epoch_start(epoch as u64)?;

            let train_run = self.run_train_epoch_with_step_observer::<M, A, O>(
                model,
                evolution_law,
                adapter,
                &train_batches,
                epoch,
                Some(observer),
            )?;
            total_train_samples += train_run.aggregates.samples;
            if batch_size == 0 {
                batch_size = train_run
                    .aggregates
                    .first_batch_size
                    .unwrap_or(0)
                    .min(u16::MAX as usize) as u16;
            }

            let eval_run = if self.should_validate(epoch) {
                let Some(val_batches) = val_batches else {
                    return Err(TrainError::EmptyDataset { split: "val" });
                };
                model.set_training(false)?;
                Some(self.run_eval_epoch(model, adapter, val_batches, epoch)?)
            } else {
                None
            };

            let phase = adapter.training_phase();
            let epoch_metric = build_epoch_metrics(epoch, phase, &train_run, eval_run.as_ref());
            let trace_capture = EpochTraceCapture {
                train: train_run.aggregates.first_trace.clone(),
                eval: eval_run
                    .as_ref()
                    .and_then(|run| run.aggregates.first_trace.clone()),
            };
            let observer_start = Instant::now();
            observer.on_epoch_end(model, evolution_law, &epoch_metric, &trace_capture)?;
            let observer_sec = observer_start.elapsed().as_secs_f64();
            let epoch_profile =
                build_epoch_runtime_profile(&train_run, eval_run.as_ref(), observer_sec);
            epoch_metrics.push(epoch_metric);
            epoch_profiles.push(epoch_profile);
            trace_captures.push(trace_capture);
        }

        Ok(TrainingArtifacts {
            summary: build_summary(
                &epoch_metrics,
                batch_size,
                total_train_samples,
                run_start.elapsed().as_secs_f64(),
            ),
            epoch_metrics,
            epoch_profiles,
            trace_captures,
        })
    }

    /// Score a model on a held-out split without evolving anything.
    ///
    /// This is the inference-only counterpart of [`Self::fit`]: it runs
    /// the very same evaluation pass the training loop runs after each
    /// epoch — the same adapter call over the same batches, aggregated the
    /// same way — and returns artifacts holding that single record. A
    /// model restored from a checkpoint therefore reports exactly the
    /// number the run that saved it reported at its last epoch, digit for
    /// digit.
    ///
    /// `epoch` is the epoch index the record is stamped with and the one
    /// handed to the model's per-batch counters, so an inference pass over
    /// a restored model can reproduce the epoch-dependent runtime state
    /// its source run was in.
    ///
    /// The training columns of the record are empty by construction: no
    /// training batch was seen.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::EmptyDataset`] when `val_batches` is empty, and
    /// propagates any adapter or model failure.
    pub fn evaluate_only<M, A>(
        &mut self,
        model: &mut M,
        adapter: &mut A,
        val_batches: &[A::Batch],
        epoch: u32,
    ) -> Result<TrainingArtifacts, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
    {
        if val_batches.is_empty() {
            return Err(TrainError::EmptyDataset { split: "val" });
        }

        let run_start = Instant::now();
        model.set_training(false)?;
        let eval_run = self.run_eval_epoch(model, adapter, val_batches, epoch)?;

        let train_run = EpochRun {
            aggregates: EpochAggregates::default(),
            elapsed_sec: 0.0,
            profile: EpochRunProfile::default(),
        };
        let phase = adapter.training_phase();
        let epoch_metric = build_epoch_metrics(epoch, phase, &train_run, Some(&eval_run));
        let trace_capture = EpochTraceCapture {
            train: None,
            eval: eval_run.aggregates.first_trace.clone(),
        };
        let epoch_profile = build_epoch_runtime_profile(&train_run, Some(&eval_run), 0.0);
        let batch_size = eval_run
            .aggregates
            .first_batch_size
            .unwrap_or(0)
            .min(u16::MAX as usize) as u16;
        let epoch_metrics = vec![epoch_metric];

        Ok(TrainingArtifacts {
            summary: build_summary(
                &epoch_metrics,
                batch_size,
                0,
                run_start.elapsed().as_secs_f64(),
            ),
            epoch_metrics,
            epoch_profiles: vec![epoch_profile],
            trace_captures: vec![trace_capture],
        })
    }

    fn should_validate(&self, epoch: u32) -> bool {
        self.config
            .validate_every
            .map(|interval| epoch % interval == 0)
            .unwrap_or(false)
    }

    fn run_train_epoch_with_step_observer<M, A, O>(
        &mut self,
        model: &mut M,
        evolution_law: &mut GaugeDynamics,
        adapter: &mut A,
        batches: &[A::Batch],
        epoch: u32,
        mut step_observer: Option<&mut O>,
    ) -> Result<EpochRun, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
        O: EpochObserver<M>,
    {
        let start = Instant::now();
        let mut aggregates = EpochAggregates::default();
        let mut profile = EpochRunProfile::default();

        for (batch_idx, batch) in batches.iter().enumerate() {
            model.set_epoch_batch(epoch as u64, batch_idx as u64)?;

            let TrainBatchOutput {
                metrics,
                gradients,
                trace,
                timing,
            } = adapter.train_batch(model, batch, &mut self.backend, self.config.collect_traces)?;
            profile.adapter.record(timing);

            validate_batch_metrics(&metrics)?;
            validate_gradient_count(model.state(), &gradients)?;

            // Download device-resident gradients to CPU for GaugeDynamics
            let device = self.backend.device();
            let gradient_arrays: Vec<Option<Array2<f64>>> = gradients
                .iter()
                .map(|g| g.as_ref().map(|buf| device.download_2d(buf)))
                .collect();
            let gradient_refs: Vec<Option<&Array2<f64>>> =
                gradient_arrays.iter().map(|g| g.as_ref()).collect();

            let step_report = evolution_law.step(model.state_mut(), &gradient_refs)?;

            if let Some(obs) = step_observer.as_deref_mut() {
                obs.on_batch_end(model, evolution_law, &step_report, epoch, batch_idx)?;
            }

            let orth_errors = model.state().orthogonality_errors();
            let input_block_orth_error = orth_errors.first().copied().unwrap_or(0.0);
            // RMS across blocks, matching `CosmosState::total_orthogonality_error`.
            // Derived from the per-block vector already in hand so the whole
            // per-block Gram sweep is not walked a second time every step.
            let total_orth_error = if orth_errors.is_empty() {
                0.0
            } else {
                let sum_sq: f64 = orth_errors.iter().map(|e| e * e).sum();
                (sum_sq / orth_errors.len() as f64).sqrt()
            };
            profile.evolution.record_step(
                step_report.profile,
                0.0,
                input_block_orth_error,
                total_orth_error,
            );

            aggregates.record(metrics, Some(step_report.mean_commutator), trace);
        }

        Ok(EpochRun {
            aggregates,
            elapsed_sec: start.elapsed().as_secs_f64(),
            profile,
        })
    }

    fn run_eval_epoch<M, A>(
        &mut self,
        model: &mut M,
        adapter: &mut A,
        batches: &[A::Batch],
        epoch: u32,
    ) -> Result<EpochRun, TrainError>
    where
        M: TrainingModel,
        A: TaskAdapter<M, B>,
    {
        let start = Instant::now();
        let mut aggregates = EpochAggregates::default();
        let mut profile = EpochRunProfile::default();

        for (batch_idx, batch) in batches.iter().enumerate() {
            model.set_epoch_batch(epoch as u64, batch_idx as u64)?;

            let EvalBatchOutput {
                metrics,
                trace,
                timing,
            } = adapter.eval_batch(model, batch, &mut self.backend, self.config.collect_traces)?;
            profile.adapter.record(timing);

            validate_batch_metrics(&metrics)?;
            aggregates.record(metrics, None, trace);
        }

        Ok(EpochRun {
            aggregates,
            elapsed_sec: start.elapsed().as_secs_f64(),
            profile,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EpochRun {
    aggregates: EpochAggregates,
    elapsed_sec: f64,
    profile: EpochRunProfile,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct EpochRunProfile {
    adapter: AdapterEpochProfile,
    evolution: EvolutionEpochProfile,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct EpochAggregates {
    samples: usize,
    weighted_loss_sum: f64,
    weighted_accuracy_sum: f64,
    accuracy_weight: usize,
    weighted_commutator_sum: f64,
    commutator_weight: usize,
    first_batch_size: Option<usize>,
    first_trace: Option<BatchTrace>,
}

impl EpochAggregates {
    fn record(
        &mut self,
        metrics: BatchMetrics,
        commutator_mean: Option<f64>,
        trace: Option<BatchTrace>,
    ) {
        let samples = metrics.samples;
        self.samples += samples;
        self.weighted_loss_sum += metrics.loss * samples as f64;
        self.first_batch_size.get_or_insert(samples);

        if let Some(accuracy) = metrics.accuracy {
            self.weighted_accuracy_sum += accuracy * samples as f64;
            self.accuracy_weight += samples;
        }

        if let Some(commutator_mean) = commutator_mean {
            self.weighted_commutator_sum += commutator_mean;
            self.commutator_weight += 1;
        }

        if self.first_trace.is_none() {
            self.first_trace = trace;
        }
    }

    fn mean_loss(&self) -> f64 {
        weighted_average(self.weighted_loss_sum, self.samples)
    }

    fn mean_accuracy(&self) -> Option<f64> {
        if self.accuracy_weight == 0 {
            None
        } else {
            Some(weighted_average(
                self.weighted_accuracy_sum,
                self.accuracy_weight,
            ))
        }
    }

    fn mean_commutator(&self) -> f64 {
        weighted_average(self.weighted_commutator_sum, self.commutator_weight)
    }
}

fn build_epoch_metrics(
    epoch: u32,
    phase: TrainingPhase,
    train_run: &EpochRun,
    eval_run: Option<&EpochRun>,
) -> EpochMetrics {
    let train_acc = train_run.aggregates.mean_accuracy().unwrap_or(0.0);
    let val_acc = eval_run.and_then(|run| run.aggregates.mean_accuracy());
    // The evaluation forward already scores the adapter's own loss on every
    // validation batch, so the held-out loss is a reduction over metrics
    // that exist — not a second pass.
    let val_loss = eval_run.map(|run| run.aggregates.mean_loss());

    EpochMetrics {
        epoch,
        phase,
        train_loss: train_run.aggregates.mean_loss(),
        train_acc,
        val_a_as_a: val_acc,
        val_loss,
        mean_commutator: train_run.aggregates.mean_commutator(),
        time_sec: train_run.elapsed_sec + eval_run.map(|run| run.elapsed_sec).unwrap_or(0.0),
        comm_snapshot: None,
    }
}

fn build_epoch_runtime_profile(
    train_run: &EpochRun,
    eval_run: Option<&EpochRun>,
    observer_sec: f64,
) -> EpochRuntimeProfile {
    let eval_epoch_total_sec = eval_run.map(|run| run.elapsed_sec).unwrap_or(0.0);

    EpochRuntimeProfile {
        train_adapter: train_run.profile.adapter.clone(),
        eval_adapter: eval_run
            .map(|run| run.profile.adapter.clone())
            .unwrap_or_default(),
        evolution: train_run.profile.evolution.clone(),
        observer_sec,
        train_epoch_total_sec: train_run.elapsed_sec,
        eval_epoch_total_sec,
        epoch_total_sec: train_run.elapsed_sec + eval_epoch_total_sec + observer_sec,
    }
}

fn build_summary(
    epoch_metrics: &[EpochMetrics],
    batch_size: u16,
    total_train_samples: usize,
    elapsed_sec: f64,
) -> BaseExperimentResult {
    let mut best_epoch = 0;
    let mut best_val_acc = f64::NEG_INFINITY;
    let mut saw_val = false;

    for metric in epoch_metrics {
        if let Some(val_acc) = metric.val_a_as_a {
            saw_val = true;
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = metric.epoch;
            }
        }
    }

    if !saw_val {
        for metric in epoch_metrics {
            if metric.train_acc > best_val_acc {
                best_val_acc = metric.train_acc;
                best_epoch = metric.epoch;
            }
        }
    }

    let final_train_acc = epoch_metrics
        .last()
        .map(|metric| metric.train_acc)
        .unwrap_or(0.0);
    let final_val_acc = epoch_metrics
        .iter()
        .rev()
        .find_map(|metric| metric.val_a_as_a)
        .unwrap_or(0.0);
    let minutes = elapsed_sec / 60.0;

    BaseExperimentResult {
        batch_size,
        best_val_acc: if best_val_acc.is_finite() {
            best_val_acc
        } else {
            0.0
        },
        best_epoch,
        final_val_acc,
        final_train_acc,
        train_val_gap: final_train_acc - final_val_acc,
        training_time_minutes: minutes,
        epochs_per_minute: if minutes > 0.0 {
            epoch_metrics.len() as f64 / minutes
        } else {
            0.0
        },
        samples_per_second: if elapsed_sec > 0.0 {
            total_train_samples as f64 / elapsed_sec
        } else {
            0.0
        },
    }
}

fn validate_batch_metrics(metrics: &BatchMetrics) -> Result<(), TrainError> {
    if metrics.samples == 0 {
        return Err(TrainError::task_adapter(
            "batch metrics must report at least one sample",
        ));
    }
    if !metrics.loss.is_finite() {
        return Err(TrainError::task_adapter("batch loss must be finite"));
    }
    if let Some(accuracy) = metrics.accuracy {
        if !accuracy.is_finite() {
            return Err(TrainError::task_adapter(
                "batch accuracy must be finite when provided",
            ));
        }
    }
    Ok(())
}

fn validate_gradient_count(
    state: &CosmosState,
    gradients: &super::BlockGradients,
) -> Result<(), TrainError> {
    if gradients.len() != state.num_blocks() {
        return Err(TrainError::task_adapter(format!(
            "expected {} block gradients, got {}",
            state.num_blocks(),
            gradients.len()
        )));
    }

    for (block_idx, gradient) in gradients.iter().enumerate() {
        let Some(gradient) = gradient.as_ref() else {
            continue;
        };
        let expected = state.get_basis(block_idx)?.dim();
        let got = gradient.dim();
        if got != expected {
            return Err(TrainError::task_adapter(format!(
                "gradient shape mismatch for block {block_idx}: expected ({}, {}), got ({}, {})",
                expected.0, expected.1, got.0, got.1
            )));
        }
    }

    Ok(())
}

fn weighted_average(sum: f64, weight: usize) -> f64 {
    if weight == 0 {
        0.0
    } else {
        sum / weight as f64
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use ndarray::array;

    use super::*;
    use crate::{
        rules::{GaugeDynamicsConfig, RulesError},
        training::{CpuBackend, trace::BatchTrace},
    };

    #[derive(Debug, Clone, PartialEq)]
    struct DummyBatch {
        samples: usize,
        loss: f64,
        accuracy: f64,
    }

    #[derive(Debug)]
    struct DummyModel {
        state: CosmosState,
        training_flag: Cell<bool>,
        epoch_starts: RefCell<Vec<u64>>,
        batch_updates: RefCell<Vec<(u64, u64)>>,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                state: CosmosState::new(&[(4, 2)]).unwrap(),
                training_flag: Cell::new(true),
                epoch_starts: RefCell::new(Vec::new()),
                batch_updates: RefCell::new(Vec::new()),
            }
        }
    }

    impl TrainingModel for DummyModel {
        fn state(&self) -> &CosmosState {
            &self.state
        }

        fn state_mut(&mut self) -> &mut CosmosState {
            &mut self.state
        }

        fn set_training(&self, training: bool) -> Result<(), ModelError> {
            self.training_flag.set(training);
            Ok(())
        }

        fn on_epoch_start(&self, epoch: u64) -> Result<(), ModelError> {
            self.epoch_starts.borrow_mut().push(epoch);
            Ok(())
        }

        fn set_epoch_batch(&self, epoch: u64, batch_idx: u64) -> Result<(), ModelError> {
            self.batch_updates.borrow_mut().push((epoch, batch_idx));
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct DummyAdapter;

    #[derive(Debug, Default)]
    struct RecordingObserver {
        epochs: Vec<u32>,
        train_acc: Vec<f64>,
        saw_train_trace: Vec<bool>,
        saw_eval_trace: Vec<bool>,
    }

    impl EpochObserver<DummyModel> for RecordingObserver {
        fn on_epoch_end(
            &mut self,
            _model: &DummyModel,
            _evolution_law: &GaugeDynamics,
            metrics: &EpochMetrics,
            trace_capture: &EpochTraceCapture,
        ) -> Result<(), TrainError> {
            self.epochs.push(metrics.epoch);
            self.train_acc.push(metrics.train_acc);
            self.saw_train_trace.push(trace_capture.train.is_some());
            self.saw_eval_trace.push(trace_capture.eval.is_some());
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct FailingObserver;

    impl EpochObserver<DummyModel> for FailingObserver {
        fn on_epoch_end(
            &mut self,
            _model: &DummyModel,
            _evolution_law: &GaugeDynamics,
            metrics: &EpochMetrics,
            _trace_capture: &EpochTraceCapture,
        ) -> Result<(), TrainError> {
            Err(TrainError::task_adapter(format!(
                "observer failed at epoch {}",
                metrics.epoch
            )))
        }
    }

    impl TaskAdapter<DummyModel, CpuBackend> for DummyAdapter {
        type Batch = DummyBatch;

        fn train_batch(
            &mut self,
            model: &DummyModel,
            batch: &Self::Batch,
            backend: &mut CpuBackend,
            collect_trace: bool,
        ) -> Result<TrainBatchOutput, TrainError> {
            let gradient = backend.zeros_like(model.state().get_basis(0)?);
            Ok(TrainBatchOutput {
                metrics: BatchMetrics::new(batch.samples, batch.loss, Some(batch.accuracy)),
                gradients: vec![Some(crate::compute::Buffer2::from_array(gradient))],
                trace: collect_trace.then(|| BatchTrace {
                    scalar_loss: Some(batch.loss),
                    ..Default::default()
                }),
                timing: BatchTimingProfile::default(),
            })
        }

        fn eval_batch(
            &mut self,
            _model: &DummyModel,
            batch: &Self::Batch,
            _backend: &mut CpuBackend,
            collect_trace: bool,
        ) -> Result<EvalBatchOutput, TrainError> {
            Ok(EvalBatchOutput {
                metrics: BatchMetrics::new(batch.samples, batch.loss, Some(batch.accuracy)),
                trace: collect_trace.then(|| BatchTrace {
                    scalar_loss: Some(batch.loss),
                    ..Default::default()
                }),
                timing: BatchTimingProfile::default(),
            })
        }
    }

    fn make_evolution_law(num_blocks: usize) -> Result<GaugeDynamics, RulesError> {
        GaugeDynamics::new(
            num_blocks,
            GaugeDynamicsConfig {
                lr: 0.0,
                ..Default::default()
            },
        )
    }

    #[test]
    fn governor_runs_epochs_and_updates_evolution_steps() {
        let mut governor = EvolutionGovernor::new(
            EvolutionConfig {
                epochs: 2,
                validate_every: Some(1),
                collect_traces: true,
            },
            CpuBackend::default(),
        )
        .unwrap();
        let mut model = DummyModel::new();
        let mut evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();
        let mut adapter = DummyAdapter;
        let train_batches = vec![
            DummyBatch {
                samples: 4,
                loss: 0.8,
                accuracy: 50.0,
            },
            DummyBatch {
                samples: 2,
                loss: 0.2,
                accuracy: 100.0,
            },
        ];
        let val_batches = vec![DummyBatch {
            samples: 3,
            loss: 0.4,
            accuracy: 75.0,
        }];

        let artifacts = governor
            .fit(
                &mut model,
                &mut evolution_law,
                &mut adapter,
                &train_batches,
                Some(&val_batches),
            )
            .unwrap();

        assert_eq!(evolution_law.global_step(), 4);
        assert_eq!(artifacts.epoch_metrics.len(), 2);
        assert_eq!(artifacts.epoch_profiles.len(), 2);
        assert_eq!(artifacts.trace_captures.len(), 2);
        assert!(artifacts.trace_captures[0].train.is_some());
        assert!(artifacts.trace_captures[0].eval.is_some());
        assert_eq!(artifacts.epoch_profiles[0].train_adapter.batch_count, 2);
        assert_eq!(artifacts.epoch_profiles[0].eval_adapter.batch_count, 1);
        assert_eq!(artifacts.epoch_profiles[0].evolution.step_count, 2);
        assert_eq!(
            artifacts.epoch_profiles[0].evolution.commutator_probe_sec,
            0.0
        );
        assert!((artifacts.epoch_metrics[0].train_acc - 66.666_666_666_7).abs() < 1e-9);
        assert_eq!(artifacts.epoch_metrics[0].val_a_as_a, Some(75.0));
        assert_eq!(artifacts.summary.batch_size, 4);
        assert_eq!(artifacts.summary.best_epoch, 1);
        assert_eq!(model.epoch_starts.borrow().as_slice(), &[1, 2]);
        assert!(!model.training_flag.get());
    }

    #[test]
    fn governor_notifies_epoch_observer() {
        let mut governor = EvolutionGovernor::new(
            EvolutionConfig {
                epochs: 2,
                validate_every: Some(1),
                collect_traces: true,
            },
            CpuBackend::default(),
        )
        .unwrap();
        let mut model = DummyModel::new();
        let mut evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();
        let mut adapter = DummyAdapter;
        let train_batches = vec![DummyBatch {
            samples: 4,
            loss: 0.5,
            accuracy: 75.0,
        }];
        let val_batches = vec![DummyBatch {
            samples: 2,
            loss: 0.4,
            accuracy: 50.0,
        }];
        let mut observer = RecordingObserver::default();

        governor
            .fit_with_observer(
                &mut model,
                &mut evolution_law,
                &mut adapter,
                &train_batches,
                Some(&val_batches),
                &mut observer,
            )
            .unwrap();

        assert_eq!(observer.epochs, vec![1, 2]);
        assert_eq!(observer.train_acc, vec![75.0, 75.0]);
        assert_eq!(observer.saw_train_trace, vec![true, true]);
        assert_eq!(observer.saw_eval_trace, vec![true, true]);
    }

    #[test]
    fn governor_propagates_epoch_observer_errors() {
        let mut governor = EvolutionGovernor::new(
            EvolutionConfig {
                epochs: 2,
                ..Default::default()
            },
            CpuBackend::default(),
        )
        .unwrap();
        let mut model = DummyModel::new();
        let mut evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();
        let mut adapter = DummyAdapter;
        let train_batches = vec![DummyBatch {
            samples: 1,
            loss: 0.5,
            accuracy: 100.0,
        }];
        let mut observer = FailingObserver;

        let err = governor
            .fit_with_observer(
                &mut model,
                &mut evolution_law,
                &mut adapter,
                &train_batches,
                None::<&[DummyBatch]>,
                &mut observer,
            )
            .unwrap_err();

        match err {
            TrainError::TaskAdapter { message } => {
                assert!(message.contains("observer failed at epoch 1"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn governor_uses_epoch_train_provider() {
        let mut governor = EvolutionGovernor::new(
            EvolutionConfig {
                epochs: 2,
                validate_every: None,
                collect_traces: false,
            },
            CpuBackend::default(),
        )
        .unwrap();
        let mut model = DummyModel::new();
        let mut evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();
        let mut adapter = DummyAdapter;
        let mut requested_epochs = Vec::new();
        let mut observer = RecordingObserver::default();

        let artifacts = governor
            .fit_with_observer_and_train_provider(
                &mut model,
                &mut evolution_law,
                &mut adapter,
                |epoch| {
                    requested_epochs.push(epoch);
                    Ok(vec![DummyBatch {
                        samples: 2,
                        loss: epoch as f64,
                        accuracy: epoch as f64 * 10.0,
                    }])
                },
                None::<&[DummyBatch]>,
                &mut observer,
            )
            .unwrap();

        assert_eq!(requested_epochs, vec![1, 2]);
        assert_eq!(artifacts.epoch_metrics[0].train_loss, 1.0);
        assert_eq!(artifacts.epoch_metrics[1].train_loss, 2.0);
        assert_eq!(observer.epochs, vec![1, 2]);
    }

    #[test]
    fn governor_rejects_empty_train_split() {
        let mut governor =
            EvolutionGovernor::new(EvolutionConfig::default(), CpuBackend::default()).unwrap();
        let mut model = DummyModel::new();
        let mut evolution_law = make_evolution_law(model.state().num_blocks()).unwrap();
        let mut adapter = DummyAdapter;

        let err = governor
            .fit(
                &mut model,
                &mut evolution_law,
                &mut adapter,
                &[],
                None::<&[DummyBatch]>,
            )
            .unwrap_err();

        match err {
            TrainError::EmptyDataset { split } => assert_eq!(split, "train"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn evolution_config_rejects_zero_epochs() {
        let err = EvolutionGovernor::new(
            EvolutionConfig {
                epochs: 0,
                ..Default::default()
            },
            CpuBackend::default(),
        )
        .unwrap_err();

        match err {
            TrainError::InvalidConfig { field, .. } => assert_eq!(field, "epochs"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn validate_gradient_count_rejects_shape_mismatch() {
        let state = CosmosState::new(&[(4, 2)]).unwrap();
        let gradients = vec![Some(crate::compute::Buffer2::from_array(array![
            [1.0],
            [2.0]
        ]))];

        let err = validate_gradient_count(&state, &gradients).unwrap_err();

        match err {
            TrainError::TaskAdapter { message } => {
                assert!(message.contains("gradient shape mismatch"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
