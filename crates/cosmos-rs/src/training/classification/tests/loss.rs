//! Loss, configuration-validation and end-to-end training tests for the flat
//! classification adapter.

use super::*;

#[test]
fn adapter_accepts_models_with_multiple_anchors_per_class() {
    // Shape sanity for `anchors_per_class > 1`: eval and train both run end to
    // end on a two-anchor model. The gradient flow itself is validated in
    // `flat_multi_anchor_gradient_finite_difference`.
    let model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1)
            .with_anchors_per_class(2),
    )
    .unwrap();
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();
    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0], [0.0, 1.0]], vec![0, 1]);

    let eval_out = adapter
        .eval_batch(&model, &batch, &mut backend, false)
        .expect("eval should succeed for multi-anchor models");
    assert_eq!(eval_out.metrics.samples, 2);

    let train_out = adapter
        .train_batch(&model, &batch, &mut backend, false)
        .expect("train should succeed for multi-anchor models");
    // Every CMR anchor block has a gradient buffer allocated by
    // the kernel; the losers carry zero entries.
    let state = model.get_state();
    for &idx in Model::cmr_block_indices(&model) {
        assert!(
            train_out.gradients[idx].is_some(),
            "CMR anchor gradient missing at idx={idx}"
        );
        let shape = train_out.gradients[idx].as_ref().unwrap().dim();
        assert_eq!(shape, state.get_basis(idx).unwrap().dim());
    }
}

#[test]
fn trainer_with_flat_adapter_reduces_loss_on_easy_dataset() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), diagonal_xy()]);
    let batch = FlatClassificationBatch::single_label(
        array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        vec![0, 0, 1, 1],
    );
    let mut adapter = FlatClassificationAdapter::default();
    let mut eval_backend = CpuBackend::default();
    let initial = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: 20,
            validate_every: None,
            collect_traces: false,
        },
        CpuBackend::default(),
    )
    .unwrap();
    let mut evolution_law = make_evolution_law(model.get_state().num_blocks(), 0.2);

    governor
        .fit(
            &mut model,
            &mut evolution_law,
            &mut adapter,
            std::slice::from_ref(&batch),
            None::<&[FlatClassificationBatch]>,
        )
        .unwrap();

    let mut eval_backend = CpuBackend::default();
    let final_loss = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    assert!(
        final_loss < initial,
        "expected final loss {final_loss} to be lower than initial loss {initial}"
    );
}

/// Mid-run topology swapping: the governor's between-epoch hook can
/// recompose the interior over the same evolving state. Training must
/// continue through the seam with finite losses, the swap must land
/// exactly once at the requested epoch, and every later epoch must run
/// under the new composition.
#[test]
fn governor_swaps_interior_topology_between_epochs() {
    use crate::{
        metrics::EpochMetrics,
        training::{EpochObserver, EpochTraceCapture},
    };

    #[derive(Debug, Default)]
    struct SwapObserver {
        swap_epochs: Vec<u32>,
        epoch_topologies: Vec<InteriorTopology>,
        losses: Vec<f64>,
    }

    impl EpochObserver<CosmosModel> for SwapObserver {
        fn on_epoch_start(
            &mut self,
            model: &mut CosmosModel,
            evolution_law: &mut GaugeDynamics,
            epoch: u32,
        ) -> Result<(), TrainError> {
            if epoch == 2 {
                model.set_interior_topology(InteriorTopology::Parallel)?;
                evolution_law.reset_momentum();
                self.swap_epochs.push(epoch);
            }
            Ok(())
        }

        fn on_epoch_end(
            &mut self,
            model: &CosmosModel,
            _evolution_law: &GaugeDynamics,
            metrics: &EpochMetrics,
            _trace_capture: &EpochTraceCapture,
        ) -> Result<(), TrainError> {
            self.epoch_topologies
                .push(model.interior_topology().expect("preset topology"));
            self.losses.push(metrics.train_loss);
            Ok(())
        }
    }

    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(2)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(InteriorTopology::Series),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), x_axis(), diagonal_xy(), x_axis(), y_axis()],
    );
    let batch = FlatClassificationBatch::single_label(
        array![[1.0, 0.2], [0.9, -0.1], [0.1, 1.0], [-0.2, 0.8]],
        vec![0, 0, 1, 1],
    );

    let mut adapter = FlatClassificationAdapter::default();
    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: 3,
            validate_every: None,
            collect_traces: false,
        },
        CpuBackend::default(),
    )
    .unwrap();
    let mut evolution_law = make_evolution_law(model.get_state().num_blocks(), 0.2);
    let mut observer = SwapObserver::default();

    let artifacts = governor
        .fit_with_observer(
            &mut model,
            &mut evolution_law,
            &mut adapter,
            std::slice::from_ref(&batch),
            None::<&[FlatClassificationBatch]>,
            &mut observer,
        )
        .unwrap();

    assert_eq!(observer.swap_epochs, vec![2]);
    assert_eq!(
        observer.epoch_topologies,
        vec![
            InteriorTopology::Series,
            InteriorTopology::Parallel,
            InteriorTopology::Parallel,
        ]
    );
    assert_eq!(artifacts.epoch_metrics.len(), 3);
    for loss in &observer.losses {
        assert!(loss.is_finite(), "loss must stay finite across the swap");
    }
    assert_eq!(model.interior_topology(), Some(InteriorTopology::Parallel));
}

#[test]
fn hinge_config_rejects_negative_margin() {
    let result = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: ClassificationLoss::GeodesicHinge { margin: -0.1 },
        ..Default::default()
    });
    assert!(result.is_err());
}

#[test]
fn hinge_trainer_reduces_loss_on_easy_dataset() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), diagonal_xy()]);
    let batch = FlatClassificationBatch::single_label(
        array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        vec![0, 0, 1, 1],
    );
    let mut adapter = hinge_adapter(std::f64::consts::PI / 5.0);
    let mut eval_backend = CpuBackend::default();
    let initial = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: 20,
            validate_every: None,
            collect_traces: false,
        },
        CpuBackend::default(),
    )
    .unwrap();
    let mut evolution_law = make_evolution_law(model.get_state().num_blocks(), 0.2);

    governor
        .fit(
            &mut model,
            &mut evolution_law,
            &mut adapter,
            std::slice::from_ref(&batch),
            None::<&[FlatClassificationBatch]>,
        )
        .unwrap();

    let mut eval_backend = CpuBackend::default();
    let final_loss = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    assert!(
        final_loss < initial,
        "expected final hinge loss {final_loss} to be lower than initial loss {initial}"
    );
}

#[test]
fn reflection_trainer_reduces_loss() {
    let mut model = make_reflection_model(
        InteriorTopology::Series,
        Activation::LeakyRelu,
        ActivationPlacement::Post,
    );
    // Use an asymmetric basis so gradients are non-zero
    let basis = {
        let angle = 0.3_f64;
        array![[angle.cos()], [angle.sin()]]
    };
    set_global_blocks(&mut model, &[Array2::eye(2), basis, x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(
        array![[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        vec![0, 0, 1, 1],
    );
    let mut adapter = FlatClassificationAdapter::default();
    let mut eval_backend = CpuBackend::default();
    let initial = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    let mut governor = EvolutionGovernor::new(
        EvolutionConfig {
            epochs: 30,
            validate_every: None,
            collect_traces: false,
        },
        CpuBackend::default(),
    )
    .unwrap();
    let mut evolution_law = make_evolution_law(model.get_state().num_blocks(), 0.15);

    governor
        .fit(
            &mut model,
            &mut evolution_law,
            &mut adapter,
            std::slice::from_ref(&batch),
            None::<&[FlatClassificationBatch]>,
        )
        .unwrap();

    let mut eval_backend = CpuBackend::default();
    let final_loss = adapter
        .eval_batch(&model, &batch, &mut eval_backend, false)
        .unwrap()
        .metrics
        .loss;

    assert!(
        final_loss < initial,
        "reflection+residual: expected final loss {final_loss} < initial loss {initial}"
    );
}
