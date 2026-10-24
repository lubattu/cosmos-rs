//! Forward-pass tests for the flat classification adapter.

use super::*;

#[test]
fn eval_batch_supports_default_interior_activation() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_activation(Activation::Relu, ActivationPlacement::Post),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), diagonal_x_neg_y(), x_axis(), y_axis()],
    );
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();
    let batch = FlatClassificationBatch::single_label(array![[1.0, -1.0]], vec![0]);

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_array2_close(&interior.output.unwrap(), &array![[1.0, 0.0]]);
    assert_eq!(output.metrics.accuracy, Some(100.0));
}

#[test]
fn eval_batch_matches_known_axis_scores_and_loss() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0], [0.0, 1.0]], vec![0, 1]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let expected_scores = array![
        [1.0, (-(std::f64::consts::FRAC_PI_2.powi(2))).exp()],
        [(-(std::f64::consts::FRAC_PI_2.powi(2))).exp(), 1.0]
    ];
    let expected_loss =
        direct_cmr_loss(&expected_scores, &batch.targets, Reduction::Mean).unwrap()[0];

    assert!((output.metrics.loss - expected_loss).abs() < EPSILON);
    assert_eq!(output.metrics.accuracy, Some(100.0));
    let trace = output.trace.expect("trace should be captured");
    assert_array2_close(&trace.input_projection.unwrap().output, &batch.inputs);
    assert_array2_close(&trace.cmr.unwrap().scores, &expected_scores);
}

#[test]
fn class_angles_match_known_axis_geometry() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), y_axis()]);
    let adapter = FlatClassificationAdapter::default();

    let angles = adapter
        .class_angles(&model, &array![[1.0, 0.0], [0.0, 1.0]])
        .unwrap();

    let right_angle = std::f64::consts::FRAC_PI_2;
    assert_array2_close(&angles, &array![[0.0, right_angle], [right_angle, 0.0]]);
}

#[test]
fn cmr_margin_is_positive_for_well_separated_data() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), y_axis()]);
    let adapter = FlatClassificationAdapter::default();
    let inputs = array![[1.0, 0.0], [0.0, 1.0]];

    // Correct labels: each sample sits on its own class anchor, so
    // margin = θ_closest_wrong − θ_correct = π/2 − 0 per sample.
    let margin = adapter.cmr_margin(&model, &inputs, &[0, 1]).unwrap();
    assert!(
        (margin - std::f64::consts::FRAC_PI_2).abs() < EPSILON,
        "expected margin π/2, got {margin}"
    );

    // Swapped labels invert the sign.
    let inverted = adapter.cmr_margin(&model, &inputs, &[1, 0]).unwrap();
    assert!(
        (inverted + std::f64::consts::FRAC_PI_2).abs() < EPSILON,
        "expected margin -π/2, got {inverted}"
    );
}

#[test]
fn eval_batch_matches_series_interior_projection_trace() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(InteriorTopology::Series),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), diagonal_xy(), diagonal_xy(), y_axis()],
    );

    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");
    let expected_embedding = array![[0.5, 0.5]];
    let expected_scores = array![[1.0, (-(std::f64::consts::FRAC_PI_4.powi(2))).exp()]];

    assert_array2_close(&interior.output.unwrap(), &expected_embedding);
    assert_array2_close(&trace.cmr.unwrap().scores, &expected_scores);
}

#[test]
fn eval_batch_matches_parallel_interior_projection_trace() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(2)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(InteriorTopology::Parallel),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), x_axis(), diagonal_xy(), x_axis(), y_axis()],
    );

    let batch = FlatClassificationBatch::single_label(array![[1.0, 1.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_array2_close(&interior.output.unwrap(), &array![[1.0, 0.5]]);
}

#[test]
fn eval_batch_matches_residual_interior_projection_trace() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(InteriorTopology::Residual),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), diagonal_xy(), x_axis(), y_axis()],
    );

    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_array2_close(&interior.output.unwrap(), &array![[1.5, 0.5]]);
}

#[test]
fn eval_batch_supports_custom_spec_with_multiple_blocks() {
    let spec = InteriorBuilder::new()
        .add_block_builder(BlockBuilder::new().with_indices(vec![1]).compose_series())
        .add_block_builder(BlockBuilder::new().with_indices(vec![2]).compose_series())
        .build_spec();
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(2)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_spec(spec),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), diagonal_xy(), x_axis(), x_axis(), y_axis()],
    );
    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_eq!(interior.blocks.len(), 2);
    assert_array2_close(&interior.blocks[0].output, &array![[0.5, 0.5]]);
    assert_array2_close(&interior.blocks[1].output, &array![[0.5, 0.0]]);
    assert_array2_close(&interior.output.unwrap(), &array![[0.5, 0.0]]);
}

#[test]
fn eval_batch_supports_custom_spec_subspace_activation() {
    let spec = InteriorBuilder::new()
        .add_block_builder(
            BlockBuilder::new()
                .with_indices(vec![1])
                .compose_series()
                .activate(Activation::Relu, ActivationPlacement::Subspace),
        )
        .build_spec();
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_spec(spec),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), diagonal_x_neg_y(), x_axis(), y_axis()],
    );
    let batch = FlatClassificationBatch::single_label(array![[-1.0, 1.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_array2_close(&interior.output.unwrap(), &array![[0.0, 0.0]]);
}

#[test]
fn eval_batch_supports_custom_spec_global_activation_and_normalization() {
    let spec = InteriorBuilder::new()
        .with_global_activation(Activation::Relu)
        .with_normalize_final(true)
        .build_spec();
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1)
            .with_interior_spec(spec),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), y_axis()]);
    let batch = FlatClassificationBatch::single_label(array![[3.0, -4.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");

    assert_array2_close(&interior.output.unwrap(), &array![[1.0, 0.0]]);
    assert_eq!(output.metrics.accuracy, Some(100.0));
}

#[test]
fn hinge_eval_loss_matches_manual_reference() {
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), diagonal_xy()]);
    let batch = FlatClassificationBatch::single_label(array![[0.8, 0.6], [0.3, -0.9]], vec![0, 1]);
    let margin = 0.9;
    let mut adapter = hinge_adapter(margin);
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, false)
        .unwrap();

    // With anchors_per_class == 1, the per-class distances are the
    // raw angles. Expected loss = hinge(Mean over batch) / C to
    // match the Python `geodesic_cmr_loss(reduction="mean")`
    // normalization by batch · num_classes.
    let distances = manual_angles(&model, &batch.inputs);
    let expected = geodesic_cmr_hinge_loss(&distances, &batch.targets, margin, Reduction::Mean)
        .unwrap()[0]
        / 2.0;
    assert!(
        (output.metrics.loss - expected).abs() < EPSILON,
        "hinge loss {} != manual reference {}",
        output.metrics.loss,
        expected
    );
}

#[test]
fn reflection_series_forward_matches_expected() {
    let mut model = make_reflection_model(
        InteriorTopology::Series,
        Activation::Identity,
        ActivationPlacement::None,
    );
    // basis = [1,0]^T projects onto x-axis
    // R(x) = 2*P(x) - x = 2*[x1, 0] - [x1, x2] = [x1, -x2]
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(array![[3.0, 4.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");
    // R([3,4]) = [3, -4]
    assert_array2_close(&interior.output.unwrap(), &array![[3.0, -4.0]]);
}

#[test]
fn reflection_residual_forward_matches_expected() {
    let mut model = make_reflection_model(
        InteriorTopology::Residual,
        Activation::Identity,
        ActivationPlacement::None,
    );
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(array![[3.0, 4.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");
    // x + R(x) = [3,4] + [3,-4] = [6, 0]
    assert_array2_close(&interior.output.unwrap(), &array![[6.0, 0.0]]);
}

#[test]
fn reflection_post_activation_forward_matches_expected() {
    let mut model = make_reflection_model(
        InteriorTopology::Series,
        Activation::Relu,
        ActivationPlacement::Post,
    );
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(array![[3.0, 4.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");
    // R([3,4]) = [3, -4], relu([3,-4]) = [3, 0]
    assert_array2_close(&interior.output.unwrap(), &array![[3.0, 0.0]]);
}

#[test]
fn reflection_subspace_activation_forward_matches_expected() {
    let spec = InteriorBuilder::new()
        .add_block_builder(
            BlockBuilder::new()
                .with_indices(vec![1])
                .compose_series()
                .activate(Activation::Relu, ActivationPlacement::Subspace)
                .subspace_action(SubspaceAction::reflection()),
        )
        .build_spec();
    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_spec(spec),
    )
    .unwrap();
    set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), x_axis(), y_axis()]);

    let batch = FlatClassificationBatch::single_label(array![[-3.0, 4.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior = trace.interior.expect("interior trace should be present");
    // xW = [-3], relu(-3) = 0, 2*0*W^T - x = [0,0] - [-3,4] = [3, -4]
    assert_array2_close(&interior.output.unwrap(), &array![[3.0, -4.0]]);
}

/// The device-path parallel-series block forward must match the host
/// `CompositionStrategy::ParallelSeries` on the same projector list.
#[test]
fn parallel_series_topology_matches_host_composition() {
    use crate::arrangement::compositions::{CompositionContext, CompositionStrategy};

    let mut model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(2)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(InteriorTopology::ParallelSeries),
    )
    .unwrap();
    set_global_blocks(
        &mut model,
        &[Array2::eye(2), x_axis(), diagonal_xy(), x_axis(), y_axis()],
    );

    let inputs = array![[3.0, 4.0], [-1.0, 2.0]];
    let batch = FlatClassificationBatch::single_label(inputs.clone(), vec![0, 1]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .eval_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let trace = output.trace.expect("trace should be captured");
    let interior_out = trace
        .interior
        .expect("interior trace should be present")
        .output
        .unwrap();

    // Identity input projection: the interior consumes `inputs` directly.
    let projectors = vec![x_axis(), diagonal_xy()];
    let ctx = CompositionContext::uniform(SubspaceAction::projection(), projectors.len());
    let expected = CompositionStrategy::ParallelSeries
        .apply(&projectors, &inputs, &ctx)
        .unwrap();
    assert_array2_close(&interior_out, &expected);
}
