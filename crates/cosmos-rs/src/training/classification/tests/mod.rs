//! Tests for the flat classification adapter, grouped by the pass under test.

pub(super) use ndarray::{Array1, Array2, array, s};

pub(super) use super::*;
pub(super) use crate::{
    arrangement::params::SubspaceAction,
    arrangement::{
        activations::{Activation, ActivationPlacement},
        builder::{BlockBuilder, InteriorBuilder},
    },
    compute::{CmrAnchorBackwardWorkspace, cmr_anchor_backward},
    models::{CosmosModelConfig, InteriorTopology},
    objectives::{
        Reduction,
        cmr_losses::{direct_cmr_loss, geodesic_cmr_hinge_loss},
    },
    rules::{GaugeDynamics, GaugeDynamicsConfig},
    training::{CpuBackend, EvolutionConfig, EvolutionGovernor},
};

mod backward;
mod forward;
mod loss;

pub(super) const EPSILON: f64 = 1e-8;

pub(super) fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < EPSILON,
            "expected {e}, got {a} (diff = {})",
            (a - e).abs()
        );
    }
}

pub(super) fn x_axis() -> Array2<f64> {
    array![[1.0], [0.0]]
}

pub(super) fn y_axis() -> Array2<f64> {
    array![[0.0], [1.0]]
}

pub(super) fn diagonal_xy() -> Array2<f64> {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    array![[inv_sqrt2], [inv_sqrt2]]
}

pub(super) fn diagonal_x_neg_y() -> Array2<f64> {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    array![[inv_sqrt2], [-inv_sqrt2]]
}

pub(super) fn set_global_blocks(model: &mut CosmosModel, blocks: &[Array2<f64>]) {
    let block_shapes = model.get_state().block_shapes().to_vec();
    assert_eq!(blocks.len(), block_shapes.len());

    let total_n: usize = block_shapes.iter().map(|(n, _)| n).sum();
    let total_k: usize = block_shapes.iter().map(|(_, k)| k).sum();
    let mut global = Array2::<f64>::zeros((total_n, total_k));

    let mut row_offset = 0;
    let mut col_offset = 0;
    for ((n, k), block) in block_shapes.iter().copied().zip(blocks.iter()) {
        assert_eq!(block.dim(), (n, k));
        global
            .slice_mut(s![row_offset..row_offset + n, col_offset..col_offset + k])
            .assign(block);
        row_offset += n;
        col_offset += k;
    }

    model.get_state_mut().update_from_global_basis(&global);
}

pub(super) fn make_evolution_law(num_blocks: usize, lr: f64) -> GaugeDynamics {
    GaugeDynamics::new(
        num_blocks,
        GaugeDynamicsConfig {
            lr,
            momentum: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
}

pub(super) fn hinge_adapter(margin: f64) -> FlatClassificationAdapter {
    FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: ClassificationLoss::GeodesicHinge { margin },
        ..Default::default()
    })
    .unwrap()
}

/// Per-(sample, anchor) angles recomputed on CPU with the same
/// aggregated-cosine convention as `compute_geodesic_scores`.
pub(super) fn manual_angles(model: &CosmosModel, inputs: &Array2<f64>) -> Array2<f64> {
    let embeddings = Model::get_embeddings(model, inputs).unwrap();
    let cmr_indices = Model::cmr_block_indices(model);
    let state = model.get_state();
    let mut angles = Array2::<f64>::zeros((embeddings.nrows(), cmr_indices.len()));
    for i in 0..embeddings.nrows() {
        let z = embeddings.row(i);
        let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        for (j, &block_idx) in cmr_indices.iter().enumerate() {
            let anchor = state.get_basis(block_idx).unwrap();
            let coords = z.dot(anchor);
            let proj_norm: f64 = coords.iter().map(|v| v * v).sum::<f64>().sqrt();
            angles[[i, j]] = (proj_norm / z_norm).clamp(0.0, 1.0).acos();
        }
    }
    angles
}

pub(super) fn make_reflection_model(
    topology: InteriorTopology,
    activation: Activation,
    placement: ActivationPlacement,
) -> CosmosModel {
    CosmosModel::new(
        CosmosModelConfig::new(2, 2)
            .with_input_subspace_dim(2)
            .with_num_interior_blocks(1)
            .with_interior_subspace_dim(1)
            .with_cmr_k(1)
            .with_interior_topology(topology)
            .with_activation(activation, placement)
            .with_subspace_action(SubspaceAction::reflection()),
    )
    .unwrap()
}
