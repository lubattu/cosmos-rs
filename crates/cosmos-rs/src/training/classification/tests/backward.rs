//! Backward-pass and gradient-checking tests for the flat classification
//! adapter.

use super::*;

/// Gradient limit the score kernel clamps to when the cosine saturates at one.
const COSINE_ONE_GRAD_LIMIT: f64 = 2.0;

/// Chain factors of one `(embedding, anchor)` score, used as the scalar
/// reference the batched CMR kernel is checked against.
struct ScoreGradientFactors {
    dscore_dc: f64,
    dc_de: Array1<f64>,
    dc_dw: Option<Array2<f64>>,
}

fn score_gradient_factors(
    embedding: &Array1<f64>,
    anchor: &Array2<f64>,
    eps: f64,
) -> ScoreGradientFactors {
    use ndarray::Axis;

    let embedding_norm_sq = embedding.iter().map(|value| value * value).sum::<f64>();
    let embedding_norm = embedding_norm_sq.sqrt();
    if embedding_norm <= eps {
        return ScoreGradientFactors {
            dscore_dc: 0.0,
            dc_de: Array1::zeros(embedding.len()),
            dc_dw: None,
        };
    }

    let coords = embedding.dot(anchor);
    let projected_norm_sq = coords.iter().map(|value| value * value).sum::<f64>();
    let projected_norm = projected_norm_sq.sqrt();
    let cosine = (projected_norm / embedding_norm).clamp(0.0, 1.0);
    let dscore_dc = score_derivative_from_cosine(cosine, eps);

    if projected_norm <= eps {
        return ScoreGradientFactors {
            dscore_dc,
            dc_de: Array1::zeros(embedding.len()),
            dc_dw: None,
        };
    }

    let projected = anchor.dot(&coords);
    let denom = projected_norm * embedding_norm;
    let mut dc_de = projected.mapv(|value| value / denom);
    let correction = projected_norm / embedding_norm.powi(3);
    dc_de -= &embedding.mapv(|value| value * correction);

    let outer = embedding
        .view()
        .insert_axis(Axis(1))
        .dot(&coords.view().insert_axis(Axis(0)))
        .mapv(|value| value / denom);

    ScoreGradientFactors {
        dscore_dc,
        dc_de,
        dc_dw: Some(outer),
    }
}

fn score_derivative_from_cosine(cosine: f64, eps: f64) -> f64 {
    if cosine >= 1.0 - eps {
        return COSINE_ONE_GRAD_LIMIT;
    }

    let theta = cosine.acos();
    let score = (-(theta * theta)).exp();
    let denom = (1.0 - cosine * cosine).sqrt().max(eps);
    2.0 * theta * score / denom
}

#[test]
fn training_step_returns_gradients_for_input_and_anchor_blocks() {
    let model = CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(0)
            .with_cmr_k(1),
    )
    .unwrap();
    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0], [0.0, 1.0]], vec![0, 1]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .train_batch(&model, &batch, &mut backend, true)
        .unwrap();

    assert_eq!(output.gradients.len(), model.get_state().num_blocks());
    assert!(output.gradients[model.input_block_idx()].is_some());
    for block_idx in model.cmr_block_range() {
        assert!(output.gradients[block_idx].is_some());
    }
    assert!(output.trace.is_some());
}

#[test]
fn batched_anchor_gradients_match_scalar_reference() {
    let embeddings = array![[1.0, 0.5], [0.2, 1.2], [0.7, -0.3]];
    let embedding_norms = Array1::from_iter(
        embeddings
            .outer_iter()
            .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt()),
    );
    let anchor = diagonal_xy();
    let score_grads = array![0.3, -0.2, 0.5];
    let mut batched_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
    let mut batched_grad_anchor = Array2::<f64>::zeros(anchor.dim());
    let mut workspace = CmrAnchorBackwardWorkspace::default();
    cmr_anchor_backward(
        &embeddings,
        &embedding_norms,
        &anchor,
        score_grads.view(),
        DEFAULT_EPS,
        &mut batched_grad_embeddings,
        &mut batched_grad_anchor,
        &mut workspace,
    );

    let mut scalar_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
    let mut scalar_grad_anchor = Array2::<f64>::zeros(anchor.dim());
    for sample_idx in 0..embeddings.nrows() {
        let embedding = embeddings.row(sample_idx).to_owned();
        let score_grad = score_gradient_factors(&embedding, &anchor, DEFAULT_EPS);
        let scale = score_grads[sample_idx] * score_grad.dscore_dc;
        scalar_grad_embeddings
            .row_mut(sample_idx)
            .scaled_add(scale, &score_grad.dc_de);
        if let Some(dc_dw) = score_grad.dc_dw {
            scalar_grad_anchor.scaled_add(scale, &dc_dw);
        }
    }

    assert_array2_close(&batched_grad_embeddings, &scalar_grad_embeddings);
    assert_array2_close(&batched_grad_anchor, &scalar_grad_anchor);
}

#[test]
fn training_step_returns_gradient_for_series_interior_block() {
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
        &[Array2::eye(2), diagonal_xy(), x_axis(), y_axis()],
    );
    let batch = FlatClassificationBatch::single_label(array![[1.0, 0.0]], vec![0]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .train_batch(&model, &batch, &mut backend, true)
        .unwrap();
    let interior_grad = output.gradients[1]
        .as_ref()
        .expect("interior gradient should be present");

    assert_eq!(interior_grad.dim(), (2, 1));
    assert!(
        output
            .trace
            .expect("trace should be present")
            .interior
            .is_some()
    );
}

#[test]
fn training_step_returns_gradient_for_activated_custom_block() {
    let spec = InteriorBuilder::new()
        .add_block_builder(
            BlockBuilder::new()
                .with_indices(vec![1])
                .compose_series()
                .activate(Activation::Relu, ActivationPlacement::Post),
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
    let batch = FlatClassificationBatch::single_label(array![[1.0, -1.0]], vec![1]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .train_batch(&model, &batch, &mut backend, true)
        .unwrap();

    assert_eq!(
        output.gradients[1]
            .as_ref()
            .expect("interior gradient should be present")
            .dim(),
        (2, 1)
    );
    assert!(
        output
            .trace
            .expect("trace should be present")
            .interior
            .is_some()
    );
}

/// Finite-difference check of the full backward in hinge mode:
/// input-projector and anchor gradients must match central
/// differences of the reported loss. The margin is chosen so no
/// baseline distance sits near the hinge kink (verified below).
#[test]
fn flat_hinge_gradient_finite_difference() {
    let make_model = || {
        let mut model = CosmosModel::new(
            CosmosModelConfig::new(2, 2)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(0)
                .with_cmr_k(1),
        )
        .unwrap();
        set_global_blocks(&mut model, &[Array2::eye(2), x_axis(), diagonal_xy()]);
        model
    };
    let model = make_model();
    let batch = FlatClassificationBatch::single_label(array![[0.8, 0.6], [0.3, -0.9]], vec![0, 1]);
    let margin = 0.9;

    // Precondition: every baseline distance is far from the hinge
    // kink at θ = margin, so ±h perturbations cannot cross it.
    let distances = manual_angles(&model, &batch.inputs);
    for d in distances.iter() {
        assert!(
            (d - margin).abs() > 1e-2,
            "baseline distance {d} too close to margin {margin} for a clean FD check"
        );
    }

    let mut adapter = hinge_adapter(margin);
    let mut backend = CpuBackend::default();
    let output = adapter
        .train_batch(&model, &batch, &mut backend, false)
        .unwrap();
    let analytical_grads: Vec<_> = output.gradients.to_vec();

    let h = 1e-6;
    let block_shapes = model.get_state().block_shapes().to_vec();
    let global = model.get_state().assemble_global_basis();

    let mut tested = 0usize;
    for block_idx in 0..block_shapes.len() {
        let Some(ref analytical) = analytical_grads[block_idx] else {
            continue;
        };
        let analytical_arr = analytical.as_array();
        let (n, k) = block_shapes[block_idx];
        let (row_offset, col_offset) = block_offsets(&block_shapes, block_idx);

        for i in 0..n {
            for j in 0..k {
                let mut model_plus = make_model();
                let mut perturbed = global.clone();
                perturbed[(row_offset + i, col_offset + j)] += h;
                model_plus
                    .get_state_mut()
                    .update_from_global_basis(&perturbed);
                let loss_plus = adapter
                    .eval_batch(&model_plus, &batch, &mut backend, false)
                    .unwrap()
                    .metrics
                    .loss;

                let mut model_minus = make_model();
                let mut perturbed = global.clone();
                perturbed[(row_offset + i, col_offset + j)] -= h;
                model_minus
                    .get_state_mut()
                    .update_from_global_basis(&perturbed);
                let loss_minus = adapter
                    .eval_batch(&model_minus, &batch, &mut backend, false)
                    .unwrap()
                    .metrics
                    .loss;

                let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
                let ana_grad = analytical_arr[(i, j)];
                let diff = (fd_grad - ana_grad).abs();
                let scale = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
                assert!(
                    diff / scale < 1e-3,
                    "hinge FD mismatch block {block_idx} ({i},{j}): analytical={ana_grad:.8e}, fd={fd_grad:.8e}, rel={:.3e}",
                    diff / scale
                );
                tested += 1;
            }
        }
    }
    assert!(tested > 0, "no gradient entries were tested");
}

#[test]
fn reflection_gradient_finite_difference_none() {
    reflection_gradient_finite_difference(Activation::Identity, ActivationPlacement::None);
}

#[test]
fn reflection_gradient_finite_difference_post() {
    reflection_gradient_finite_difference(Activation::LeakyRelu, ActivationPlacement::Post);
}

#[test]
fn reflection_gradient_finite_difference_subspace() {
    reflection_gradient_finite_difference(Activation::LeakyRelu, ActivationPlacement::Subspace);
}

/// Finite-difference validation of the flat adapter's multi-anchor
/// backward. Uses `input_dim=4`,
/// `input_subspace_dim=2`, `cmr_k=1`, `anchors_per_class=2` so the
/// input projector is strictly rank-reducing and the CMR
/// projection is rank-1 with cos θ in `(0, 1)` generically.
/// Matches the configuration shape of the patchwise multi-anchor
/// FD test (which uses `agg_subspace_dim=3`, `cmr_k=1`).
///
/// Keep `cmr_k < input_subspace_dim`
/// and the embedding ambient dim strictly larger than the rank of
/// `W_cmr`, otherwise the CMR projection preserves norm and the
/// score-derivative limit branch breaks finite-difference symmetry.
#[test]
fn flat_multi_anchor_gradient_finite_difference() {
    // Use embedding_dim=3 (input_subspace_dim=3) and cmr_k=1 so
    // the CMR projection is strictly rank-1 in 3D, giving the
    // per-class max-pool comfortable margins under the QR random
    // init. Smaller h reduces the chance of perturbations flipping the argmax
    // in a one-sided way, which would produce a spurious factor-of-2 mismatch.
    let make_model = || {
        CosmosModel::new(
            CosmosModelConfig::new(6, 2)
                .with_input_subspace_dim(3)
                .with_num_interior_blocks(0)
                .with_cmr_k(1)
                .with_anchors_per_class(2),
        )
        .unwrap()
    };
    let model = make_model();
    let batch = FlatClassificationBatch::single_label(
        array![
            [0.8, 0.6, -0.2, 0.4, 0.1, -0.3],
            [0.3, -0.9, 0.1, 0.5, -0.2, 0.6],
        ],
        vec![0, 1],
    );
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .train_batch(&model, &batch, &mut backend, false)
        .unwrap();
    let analytical_grads: Vec<_> = output.gradients.to_vec();

    let h = 1e-6;
    let abs_tol = 2e-4_f64;
    let rel_tol = 2e-2_f64;
    let block_shapes = model.get_state().block_shapes().to_vec();
    let global = model.get_state().assemble_global_basis();

    // Identify per-(sample, class) argmax at the unperturbed
    // basis. Any (block, entry) whose perturbation flips the
    // argmax for at least one (sample, class) lies on a kink of
    // the loss landscape; central differences are then a chord
    // across the kink and do not match the per-side analytic
    // gradient. We skip such entries here — they are diagnostic,
    // not a bug. The remaining entries (the bulk) must match
    // tightly.
    let baseline_argmax = compute_argmax(&model, &batch);

    let mut mismatches: Vec<String> = Vec::new();
    let mut tested = 0usize;
    let mut skipped = 0usize;
    for block_idx in 0..block_shapes.len() {
        let Some(ref analytical) = analytical_grads[block_idx] else {
            continue;
        };
        let analytical_arr = analytical.as_array();
        let (n, k) = block_shapes[block_idx];
        let (row_offset, col_offset) = block_offsets(&block_shapes, block_idx);

        for i in 0..n {
            for j in 0..k {
                let mut model_plus = make_model();
                let mut perturbed = global.clone();
                perturbed[(row_offset + i, col_offset + j)] += h;
                model_plus
                    .get_state_mut()
                    .update_from_global_basis(&perturbed);
                let arg_plus = compute_argmax(&model_plus, &batch);
                let loss_plus = adapter
                    .eval_batch(&model_plus, &batch, &mut backend, false)
                    .unwrap()
                    .metrics
                    .loss;

                let mut model_minus = make_model();
                let mut perturbed = global.clone();
                perturbed[(row_offset + i, col_offset + j)] -= h;
                model_minus
                    .get_state_mut()
                    .update_from_global_basis(&perturbed);
                let arg_minus = compute_argmax(&model_minus, &batch);
                let loss_minus = adapter
                    .eval_batch(&model_minus, &batch, &mut backend, false)
                    .unwrap()
                    .metrics
                    .loss;

                if arg_plus != baseline_argmax || arg_minus != baseline_argmax {
                    skipped += 1;
                    continue;
                }

                let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
                let ana_grad = analytical_arr[(i, j)];
                let diff = (fd_grad - ana_grad).abs();
                let rel = diff / ana_grad.abs().max(1e-12);
                if !(diff < abs_tol || rel < rel_tol) {
                    mismatches.push(format!(
                        "block {block_idx} ({i},{j}): analytical={ana_grad:.6e}, fd={fd_grad:.6e}, abs_err={diff:.3e}, rel_err={rel:.3e}"
                    ));
                }
                tested += 1;
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "multi-anchor FD mismatches ({} tested, {} skipped due to argmax flip):\n  {}",
        tested,
        skipped,
        mismatches.join("\n  ")
    );
    assert!(
        tested > 0,
        "no FD entries tested — every perturbation flipped the argmax, which suggests a degenerate model config"
    );
}

/// Compute the per-class argmax over anchors for the given batch.
/// Used by the multi-anchor FD test to detect argmax flips
/// induced by perturbations (which would break central-difference
/// symmetry).
fn compute_argmax(model: &CosmosModel, batch: &FlatClassificationBatch) -> Array2<usize> {
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();
    // Use eval_batch to drive the full forward; then re-do the
    // per-(sample, class, anchor) scoring on CPU to reach the
    // argmax. The CosmosModel doesn't expose the raw per-anchor
    // scores via the trait, so we recompute via embeddings.
    let _ = adapter
        .eval_batch(model, batch, &mut backend, false)
        .unwrap();
    // Recompute embeddings from the model directly.
    let inputs = batch.inputs.clone();
    let embeddings = Model::get_embeddings(model, &inputs).unwrap();
    let num_classes = Model::num_classes(model);
    let anchors_per_class = Model::anchors_per_class(model);
    let cmr_indices = Model::cmr_block_indices(model);
    let state = model.get_state();
    let batch_size = embeddings.nrows();
    let mut argmax = Array2::<usize>::zeros((batch_size, num_classes));
    for sample_idx in 0..batch_size {
        let z = embeddings.row(sample_idx);
        let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        for class_idx in 0..num_classes {
            let mut best = f64::NEG_INFINITY;
            let mut best_local = 0usize;
            for anchor_local in 0..anchors_per_class {
                let block_idx = cmr_indices[class_idx * anchors_per_class + anchor_local];
                let anchor = state.get_basis(block_idx).unwrap();
                let coords = z.dot(anchor);
                let proj_norm: f64 = coords.iter().map(|v| v * v).sum::<f64>().sqrt();
                let cos_theta = (proj_norm / z_norm).clamp(0.0, 1.0);
                let theta = cos_theta.acos();
                let s = (-(theta * theta)).exp();
                if s > best {
                    best = s;
                    best_local = anchor_local;
                }
            }
            argmax[[sample_idx, class_idx]] = best_local;
        }
    }
    argmax
}

fn reflection_gradient_finite_difference(activation: Activation, placement: ActivationPlacement) {
    let spec = InteriorBuilder::new()
        .add_block_builder(
            BlockBuilder::new()
                .with_indices(vec![1])
                .compose_series()
                .activate(activation, placement)
                .subspace_action(SubspaceAction::reflection()),
        )
        .build_spec();

    let make_model = || {
        CosmosModel::new(
            CosmosModelConfig::new(2, 2)
                .with_input_subspace_dim(2)
                .with_num_interior_blocks(1)
                .with_interior_subspace_dim(1)
                .with_cmr_k(1)
                .with_interior_spec(spec.clone()),
        )
        .unwrap()
    };

    let model = make_model();
    let batch = FlatClassificationBatch::single_label(array![[0.8, 0.6], [0.3, -0.9]], vec![0, 1]);
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    // Get analytical gradients
    let output = adapter
        .train_batch(&model, &batch, &mut backend, false)
        .unwrap();
    let analytical_grads: Vec<_> = output.gradients.to_vec();

    let h = 1e-6;
    let block_shapes = model.get_state().block_shapes().to_vec();
    let global = model.get_state().assemble_global_basis();

    // Check each block's gradient via finite difference
    for block_idx in 0..block_shapes.len() {
        if let Some(ref analytical) = analytical_grads[block_idx] {
            let analytical_arr = analytical.as_array();
            let (n, k) = block_shapes[block_idx];
            let (row_offset, col_offset) = block_offsets(&block_shapes, block_idx);

            for i in 0..n {
                for j in 0..k {
                    // Perturb +h
                    let mut model_plus = make_model();
                    let mut perturbed = global.clone();
                    perturbed[(row_offset + i, col_offset + j)] += h;
                    model_plus
                        .get_state_mut()
                        .update_from_global_basis(&perturbed);
                    let loss_plus = adapter
                        .eval_batch(&model_plus, &batch, &mut backend, false)
                        .unwrap()
                        .metrics
                        .loss;

                    // Perturb -h
                    let mut model_minus = make_model();
                    let mut perturbed = global.clone();
                    perturbed[(row_offset + i, col_offset + j)] -= h;
                    model_minus
                        .get_state_mut()
                        .update_from_global_basis(&perturbed);
                    let loss_minus = adapter
                        .eval_batch(&model_minus, &batch, &mut backend, false)
                        .unwrap()
                        .metrics
                        .loss;

                    let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
                    let ana_grad = analytical_arr[(i, j)];
                    let diff = (fd_grad - ana_grad).abs();
                    let scale = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
                    assert!(
                        diff / scale < 1e-3,
                        "block {block_idx} ({i},{j}): analytical={ana_grad:.8}, fd={fd_grad:.8}, rel_diff={:.6}",
                        diff / scale
                    );
                }
            }
        }
    }
}

/// Finite-difference validation of the parallel-series device
/// backward (phase-2 series chain into phase-1 mean-pool fan-out).
#[test]
fn parallel_series_gradient_finite_difference() {
    let make_model = || {
        CosmosModel::new(
            CosmosModelConfig::new(4, 2)
                .with_input_subspace_dim(3)
                .with_num_interior_blocks(2)
                .with_interior_subspace_dim(2)
                .with_cmr_k(1)
                .with_interior_topology(InteriorTopology::ParallelSeries),
        )
        .unwrap()
    };

    let model = make_model();
    let batch = FlatClassificationBatch::single_label(
        array![[0.8, 0.6, -0.2, 0.4], [0.3, -0.9, 0.1, 0.5]],
        vec![0, 1],
    );
    let mut adapter = FlatClassificationAdapter::default();
    let mut backend = CpuBackend::default();

    let output = adapter
        .train_batch(&model, &batch, &mut backend, false)
        .unwrap();
    let analytical_grads: Vec<_> = output.gradients.to_vec();

    let h = 1e-6;
    let block_shapes = model.get_state().block_shapes().to_vec();
    let global = model.get_state().assemble_global_basis();

    for block_idx in 0..block_shapes.len() {
        if let Some(ref analytical) = analytical_grads[block_idx] {
            let analytical_arr = analytical.as_array();
            let (n, k) = block_shapes[block_idx];
            let (row_offset, col_offset) = block_offsets(&block_shapes, block_idx);

            for i in 0..n {
                for j in 0..k {
                    let mut model_plus = make_model();
                    let mut perturbed = global.clone();
                    perturbed[(row_offset + i, col_offset + j)] += h;
                    model_plus
                        .get_state_mut()
                        .update_from_global_basis(&perturbed);
                    let loss_plus = adapter
                        .eval_batch(&model_plus, &batch, &mut backend, false)
                        .unwrap()
                        .metrics
                        .loss;

                    let mut model_minus = make_model();
                    let mut perturbed = global.clone();
                    perturbed[(row_offset + i, col_offset + j)] -= h;
                    model_minus
                        .get_state_mut()
                        .update_from_global_basis(&perturbed);
                    let loss_minus = adapter
                        .eval_batch(&model_minus, &batch, &mut backend, false)
                        .unwrap()
                        .metrics
                        .loss;

                    let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
                    let ana_grad = analytical_arr[(i, j)];
                    let diff = (fd_grad - ana_grad).abs();
                    let scale = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
                    assert!(
                        diff / scale < 1e-3,
                        "block {block_idx} ({i},{j}): analytical={ana_grad:.8}, fd={fd_grad:.8}, rel_diff={:.6}",
                        diff / scale
                    );
                }
            }
        }
    }
}

fn block_offsets(shapes: &[(usize, usize)], block_idx: usize) -> (usize, usize) {
    let mut row = 0;
    let mut col = 0;
    for (n, k) in &shapes[..block_idx] {
        row += n;
        col += k;
    }
    (row, col)
}
