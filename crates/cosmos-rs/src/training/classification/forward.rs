//! Forward pass of the flat classification adapter.
//!
//! Every buffer the backward pass needs is cached in the returned
//! `ForwardBatch` rather than recomputed.

use ndarray::Array2;

use crate::{
    compute::device::{Buffer2, ComputeDevice},
    models::{CosmosModel, Model},
    objectives::{
        Reduction,
        cmr_losses::{direct_cmr_loss, geodesic_cmr_hinge_loss},
    },
    training::{TrainError, backend::TrainingBackend},
};

use super::{
    ClassificationLoss, FlatClassificationAdapterConfig, FlatClassificationBatch,
    loss::{validate_batch_targets, validate_model_support},
    shared::{
        InteriorForwardDevice, forward_supported_interior_device, pool_class_angles,
        single_label_accuracy, upload_basis,
    },
};

/// Everything the backward pass reads from one forward pass.
#[derive(Debug)]
pub(super) struct ForwardBatch {
    pub(super) input_block_idx: usize,
    pub(super) anchor_indices: Vec<usize>,
    pub(super) projected_inputs: Array2<f64>,
    /// Per-class scores used by loss, accuracy and the score gradient. Shape
    /// `(batch, num_classes)`. For multi-anchor models this is the max-pool of
    /// the raw `(batch, num_anchors)` scores over each class's anchors; for
    /// single-anchor models it equals the raw scores.
    pub(super) scores: Array2<f64>,
    /// For models with `anchors_per_class > 1`, the local anchor index (in
    /// `0..anchors_per_class`) that wins the max-pool per `(sample, class)`.
    /// The backward routes the per-class score gradient exclusively to that
    /// winning anchor. `None` when `anchors_per_class == 1`.
    pub(super) argmax_anchor: Option<Array2<usize>>,
    /// Number of anchors per class, cached so the backward can fan out score
    /// gradients without re-querying the model.
    pub(super) anchors_per_class: usize,
    /// Raw per-(sample, anchor) geodesic angles `θ` from the score kernel.
    /// Populated only for [`ClassificationLoss::GeodesicHinge`], whose backward
    /// re-expresses angle gradients as score gradients.
    pub(super) angles: Option<Array2<f64>>,
    /// Per-(sample, class) pooled angles — the winning anchor's `θ` under the
    /// score max-pool. Populated only for [`ClassificationLoss::GeodesicHinge`].
    pub(super) class_distances: Option<Array2<f64>>,
    pub(super) samples: usize,
    pub(super) loss: f64,
    pub(super) accuracy: Option<f64>,
    pub(super) inputs_buf: Buffer2<f64>,
    pub(super) embeddings_buf: Buffer2<f64>,
    pub(super) interior_device: Option<InteriorForwardDevice>,
}

pub(super) fn forward_batch<B: TrainingBackend>(
    model: &CosmosModel,
    batch: &FlatClassificationBatch,
    config: &FlatClassificationAdapterConfig,
    backend: &mut B,
) -> Result<ForwardBatch, TrainError> {
    validate_batch_targets(&batch.targets, batch.samples())?;
    validate_model_support(model)?;

    let state = model.get_state();
    let input_block_idx = model.input_block_idx();
    let input_basis = state.get_basis(input_block_idx)?;
    if batch.inputs.ncols() != input_basis.nrows() {
        return Err(TrainError::task_adapter(format!(
            "input dimension mismatch: expected {}, got {}",
            input_basis.nrows(),
            batch.inputs.ncols()
        )));
    }

    let inputs_buf: Buffer2<f64>;
    let proj_buf: Buffer2<f64>;
    let proj_cpu: Array2<f64>;
    {
        let device = backend.device();
        inputs_buf = upload_basis(device, &batch.inputs);
        let input_basis_buf = upload_basis(device, input_basis);
        proj_buf = device.project_rows(&inputs_buf, &input_basis_buf);
        proj_cpu = device.download_2d(&proj_buf);
    }

    let interior_block_configs = model.interior_block_configs()?;
    let interior_block_actions = model.interior_block_actions()?;
    let interior_global_activation = model.interior_global_activation()?;
    let interior_normalize_final = model.interior_normalize_final()?;
    let interior_device = forward_supported_interior_device(
        &interior_block_configs,
        &interior_block_actions,
        interior_global_activation,
        interior_normalize_final,
        state,
        &proj_buf,
        backend,
    )?;
    let device = backend.device();
    let embeddings_buf: Buffer2<f64> = match interior_device {
        Some(ref int_dev) => device.clone_buffer(&int_dev.output),
        None => device.clone_buffer(&proj_buf),
    };

    // `compute_geodesic_scores` returns shape `(batch, num_anchors)` where
    // `num_anchors = num_classes * anchors_per_class`.
    let anchor_indices = Model::cmr_block_indices(model);
    let anchor_bufs: Vec<Buffer2<f64>> = anchor_indices
        .iter()
        .map(|&idx| Ok(upload_basis(device, state.get_basis(idx)?)))
        .collect::<Result<Vec<_>, TrainError>>()?;
    let anchor_refs: Vec<&Buffer2<f64>> = anchor_bufs.iter().collect();
    let (scores_buf, angles_buf) =
        device.compute_geodesic_scores(&embeddings_buf, &anchor_refs, config.eps);
    let raw_scores = device.download_2d(&scores_buf);

    // Max-pool over the anchors of each class. For `anchors_per_class == 1` the
    // operation is the identity and the bookkeeping is skipped; otherwise the
    // winner's local index is recorded so the backward can route the per-class
    // gradient back to that single anchor.
    let num_classes = Model::num_classes(model);
    let anchors_per_class = Model::anchors_per_class(model);
    let (scores, argmax_anchor) = if anchors_per_class == 1 {
        (raw_scores, None)
    } else {
        let batch_size = raw_scores.nrows();
        let mut per_class = Array2::<f64>::zeros((batch_size, num_classes));
        let mut argmax = Array2::<usize>::zeros((batch_size, num_classes));
        for sample_idx in 0..batch_size {
            for class_idx in 0..num_classes {
                let mut best = f64::NEG_INFINITY;
                let mut best_local = 0usize;
                for anchor_local in 0..anchors_per_class {
                    let col = class_idx * anchors_per_class + anchor_local;
                    let v = raw_scores[[sample_idx, col]];
                    if v > best {
                        best = v;
                        best_local = anchor_local;
                    }
                }
                per_class[[sample_idx, class_idx]] = best;
                argmax[[sample_idx, class_idx]] = best_local;
            }
        }
        (per_class, Some(argmax))
    };

    let (loss, angles, class_distances) = match config.loss {
        ClassificationLoss::DirectMembership => (
            direct_cmr_loss(&scores, &batch.targets, Reduction::Mean)?[0],
            None,
            None,
        ),
        ClassificationLoss::GeodesicHinge { margin } => {
            let angles_f64 = device.download_2d(&angles_buf);
            let class_distances = pool_class_angles(
                &angles_f64,
                argmax_anchor.as_ref(),
                num_classes,
                anchors_per_class,
            );
            // `Reduction::Mean` divides by the batch size only, while the
            // reference `geodesic_cmr_loss(reduction="mean")` divides by
            // `batch · num_classes` — the same normalization folded into
            // `geodesic_cmr_loss_grad`. Dividing by `num_classes` keeps the
            // reported loss consistent with the analytic gradient.
            let loss =
                geodesic_cmr_hinge_loss(&class_distances, &batch.targets, margin, Reduction::Mean)?
                    [0]
                    / num_classes as f64;
            (loss, Some(angles_f64), Some(class_distances))
        }
    };
    let accuracy = single_label_accuracy(&scores, &batch.targets);

    Ok(ForwardBatch {
        input_block_idx,
        anchor_indices: anchor_indices.to_vec(),
        projected_inputs: proj_cpu,
        scores,
        argmax_anchor,
        anchors_per_class,
        angles,
        class_distances,
        samples: batch.samples(),
        loss,
        accuracy,
        inputs_buf,
        embeddings_buf,
        interior_device,
    })
}
