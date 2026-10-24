//! Loss-side validation and score gradients.
//!
//! Holds the arrangement support checks the adapter enforces before it runs a
//! batch, and the softmax cross-entropy gradient of the direct-membership loss.

use std::str::FromStr;

use ndarray::{Array2, s};

use crate::{
    arrangement::{
        ArrangementError,
        activations::{Activation, ActivationPlacement},
        blocks::BlockConfig,
        compositions::CompositionStrategy,
        orderings::OrderingStrategy,
        params::ParallelMode,
        selections::SelectionStrategy,
    },
    models::{CosmosModel, Model},
    objectives::utils::Targets,
    training::TrainError,
};

use super::DEFAULT_EPS;

pub(super) fn validate_batch_targets(
    targets: &Targets,
    expected_batch: usize,
) -> Result<(), TrainError> {
    match targets {
        Targets::SingleLabel(indices) => {
            if indices.len() != expected_batch {
                return Err(TrainError::task_adapter(format!(
                    "single-label target count mismatch: expected {expected_batch}, got {}",
                    indices.len()
                )));
            }
        }
        Targets::MultiLabel(probs) => {
            if probs.nrows() != expected_batch {
                return Err(TrainError::task_adapter(format!(
                    "multi-label batch size mismatch: expected {expected_batch}, got {}",
                    probs.nrows()
                )));
            }
        }
    }

    Ok(())
}

pub(super) fn validate_model_support(model: &CosmosModel) -> Result<(), TrainError> {
    if Model::anchors_per_class(model) == 0 {
        return Err(TrainError::task_adapter(
            "FlatClassificationAdapter: anchors_per_class must be >= 1",
        ));
    }

    for config in model.interior_block_configs()? {
        validate_block_config_support(&config)?;
    }

    Ok(())
}

/// Reject any arrangement the adapter has no backward path for.
///
/// The adapter accepts the `all` selection, the `fixed` ordering, `mean`
/// aggregation and the four composition strategies. Every other routing
/// primitive is unconstructible in this crate, so this is a defensive check
/// that keeps the refusal explicit rather than implicit.
///
/// # Errors
///
/// Returns [`TrainError::TaskAdapter`] naming the rejected primitive, or
/// [`TrainError::Arrangement`] when a composition, activation or action name
/// does not parse.
pub(crate) fn validate_block_config_support(config: &BlockConfig) -> Result<(), TrainError> {
    match SelectionStrategy::from_str(&config.selection.strategy) {
        Ok(SelectionStrategy::All) => {}
        _ => {
            return Err(TrainError::task_adapter(format!(
                "FlatClassificationAdapter supports only the `all` selection strategy, got `{}`",
                config.selection.strategy
            )));
        }
    }

    match OrderingStrategy::from_str(&config.ordering.strategy) {
        Ok(OrderingStrategy::Fixed) => {}
        _ => {
            return Err(TrainError::task_adapter(format!(
                "FlatClassificationAdapter supports only the `fixed` ordering strategy, got `{}`",
                config.ordering.strategy
            )));
        }
    }

    let composition = CompositionStrategy::from_str(&config.composition.strategy)?;
    if composition == CompositionStrategy::Parallel
        && config.composition.parallel_mode != ParallelMode::Stack
    {
        return Err(TrainError::task_adapter(
            "FlatClassificationAdapter supports only `ParallelMode::Stack` for parallel interior blocks",
        ));
    }

    if config
        .aggregation
        .as_ref()
        .is_some_and(|aggregation| aggregation.strategy != "mean")
    {
        return Err(TrainError::task_adapter(
            "FlatClassificationAdapter supports only mean aggregation for parallel interior blocks",
        ));
    }

    let _ = Activation::from_str(&config.composition.activation_name)
        .map_err(ArrangementError::from)?;
    let _ = ActivationPlacement::from_str(&config.composition.activation_placement)
        .map_err(ArrangementError::from)?;

    Ok(())
}

/// Gradient of the softmax cross-entropy over membership scores.
///
/// # Errors
///
/// Returns [`TrainError::TaskAdapter`] for an empty batch, a target count that
/// does not match the batch, an out-of-range label, or a multi-label target
/// shape that does not match the score matrix.
pub(crate) fn direct_membership_score_grad(
    scores: &Array2<f64>,
    targets: &Targets,
) -> Result<Array2<f64>, TrainError> {
    let batch_size = scores.nrows();
    if batch_size == 0 {
        return Err(TrainError::task_adapter(
            "cannot compute score gradients for an empty batch",
        ));
    }

    match targets {
        Targets::SingleLabel(labels) => {
            if labels.len() != batch_size {
                return Err(TrainError::task_adapter(format!(
                    "single-label target count mismatch: expected {batch_size}, got {}",
                    labels.len()
                )));
            }

            let mut grads = stable_softmax(scores);
            for (row_idx, &label) in labels.iter().enumerate() {
                if label >= scores.ncols() {
                    return Err(TrainError::task_adapter(format!(
                        "target class {label} out of range for {} classes",
                        scores.ncols()
                    )));
                }
                grads[[row_idx, label]] -= 1.0;
            }

            grads.mapv_inplace(|value| value / batch_size as f64);
            Ok(grads)
        }
        Targets::MultiLabel(target_probs) => {
            if target_probs.dim() != scores.dim() {
                return Err(TrainError::task_adapter(format!(
                    "multi-label target shape mismatch: expected {:?}, got {:?}",
                    scores.dim(),
                    target_probs.dim()
                )));
            }

            Ok((scores - target_probs).mapv(|value| 2.0 * value / batch_size as f64))
        }
    }
}

pub(super) fn stable_softmax(scores: &Array2<f64>) -> Array2<f64> {
    let mut probs = Array2::<f64>::zeros(scores.dim());

    for (row_idx, row) in scores.outer_iter().enumerate() {
        let max_value = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_row = row.mapv(|value| (value - max_value).exp());
        let denom = exp_row.sum().max(DEFAULT_EPS);
        probs
            .slice_mut(s![row_idx, ..])
            .assign(&exp_row.mapv(|value| value / denom));
    }

    probs
}
