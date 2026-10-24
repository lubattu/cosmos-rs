//! Assertions that run without any dataset.
//!
//! The comparison table is only meaningful if the arms are matched on
//! parameter count and if the Riemannian arm really stays on the manifold.
//! These checks assert both, plus the determinism and disjointness of the
//! partition, in under a second and without touching the network.

use cosmos_rs::compute::Buffer2;
use cosmos_rs::models::count_state_parameters;
use cosmos_rs::prelude::{
    CosmosModel, CosmosModelConfig, CosmosModelPatchwise, PatchwiseModelConfig,
};
use ndarray::Array2;

use crate::baselines::{mlp, rsgd};
use crate::datasets::split_indices;
use crate::init::seed_state;

/// Expected frame count of the flat grayscale configuration.
pub const FLAT_GRAYSCALE_PARAMS: usize = 150_528;
/// Expected parameter count of its matched two-layer network.
pub const FLAT_GRAYSCALE_MLP_PARAMS: usize = 150_265;
/// Expected frame count of the flat colour configuration.
pub const FLAT_COLOUR_PARAMS: usize = 589_824;
/// Expected parameter count of its matched two-layer network.
pub const FLAT_COLOUR_MLP_PARAMS: usize = 588_863;
/// Expected frame count of the patch-wise colour configuration.
pub const PATCHWISE_COLOUR_PARAMS: usize = 443_136;

/// Run every assertion and return one line per check.
///
/// # Errors
///
/// Returns a description of the first assertion that did not hold.
pub fn run_all() -> Result<Vec<String>, String> {
    let mut lines = Vec::new();

    let flat = flat_model(784)?;
    let count = count_state_parameters(flat.get_state());
    if count != FLAT_GRAYSCALE_PARAMS {
        return Err(format!(
            "flat grayscale frame count is {count}, expected {FLAT_GRAYSCALE_PARAMS}"
        ));
    }
    lines.push(format!("flat grayscale frame parameters: {count}"));

    let colour = flat_model(3072)?;
    let count = count_state_parameters(colour.get_state());
    if count != FLAT_COLOUR_PARAMS {
        return Err(format!(
            "flat colour frame count is {count}, expected {FLAT_COLOUR_PARAMS}"
        ));
    }
    lines.push(format!("flat colour frame parameters: {count}"));

    let patchwise = patchwise_model()?;
    let count = count_state_parameters(patchwise.get_state());
    if count != PATCHWISE_COLOUR_PARAMS {
        return Err(format!(
            "patch-wise colour frame count is {count}, expected {PATCHWISE_COLOUR_PARAMS}"
        ));
    }
    lines.push(format!("patch-wise colour frame parameters: {count}"));

    let matched = mlp::param_count(784, 189, 10);
    if matched != FLAT_GRAYSCALE_MLP_PARAMS {
        return Err(format!(
            "matched grayscale network has {matched} parameters, expected {FLAT_GRAYSCALE_MLP_PARAMS}"
        ));
    }
    let gap = (FLAT_GRAYSCALE_PARAMS as f64 - matched as f64).abs() / FLAT_GRAYSCALE_PARAMS as f64;
    if gap >= 0.02 {
        return Err(format!(
            "grayscale parameter gap is {gap:.4}, expected below 0.02"
        ));
    }
    lines.push(format!(
        "matched grayscale network: {matched} parameters, gap {:.3} %",
        100.0 * gap
    ));

    let matched = mlp::param_count(3072, 191, 10);
    if matched != FLAT_COLOUR_MLP_PARAMS {
        return Err(format!(
            "matched colour network has {matched} parameters, expected {FLAT_COLOUR_MLP_PARAMS}"
        ));
    }
    lines.push(format!("matched colour network: {matched} parameters"));

    let mut stepped = flat_model(784)?;
    let state = stepped.get_state_mut();
    let shapes: Vec<(usize, usize)> = state.block_shapes().to_vec();
    let gradients: Vec<Option<Buffer2<f64>>> = shapes
        .into_iter()
        .enumerate()
        .map(|(index, (rows, cols))| {
            Some(Buffer2::from_array(Array2::from_shape_fn(
                (rows, cols),
                |(row, col)| ((row * 13 + col * 5 + index) % 17) as f64 - 8.0,
            )))
        })
        .collect();
    rsgd::step(state, &gradients, 0.3).map_err(|err| err.to_string())?;
    for (index, error) in state.orthogonality_errors().into_iter().enumerate() {
        if error >= 1e-12 {
            return Err(format!("block {index} orthogonality error is {error:e}"));
        }
    }
    lines.push("Riemannian step preserves orthonormality on every block".to_string());

    let (train, val) = split_indices(60_000, 5_000, 12_345);
    if train.len() != 55_000 || val.len() != 5_000 {
        return Err(format!(
            "grayscale partition is {} / {}, expected 55000 / 5000",
            train.len(),
            val.len()
        ));
    }
    let mut all = train.clone();
    all.extend_from_slice(&val);
    all.sort_unstable();
    all.dedup();
    if all.len() != 60_000 {
        return Err("partition splits overlap".to_string());
    }
    if split_indices(60_000, 5_000, 12_345) != (train, val) {
        return Err("partition is not deterministic".to_string());
    }
    lines.push("partition is deterministic, disjoint and correctly sized".to_string());

    Ok(lines)
}

/// Reference seed of the calibration dump.
pub const CALIBRATION_SEED: u64 = 319_635_789;

/// First row of the first block's keyed Gaussian draw, at full precision.
///
/// The dump exists so the draw can be compared entry by entry against an
/// externally held reference vector when a shortened run misses its
/// expectation. It reads nothing and writes nothing.
pub fn first_row_dump() -> Vec<String> {
    let block = crate::init::gaussian_block(CALIBRATION_SEED, 0, 784, 32);
    let mut lines = vec![format!("seed {CALIBRATION_SEED} block 0 row 0")];
    lines.extend(block.row(0).iter().map(|value| format!("{value:.17e}")));
    lines
}

fn flat_model(input_dim: usize) -> Result<CosmosModel, String> {
    let config = CosmosModelConfig::new(input_dim, 10)
        .with_input_subspace_dim(32)
        .with_interior_subspace_dim(16)
        .with_num_interior_blocks(5)
        .with_cmr_k(8)
        .with_anchors_per_class(1);
    let mut model = CosmosModel::new(config).map_err(|err| err.to_string())?;
    seed_state(model.get_state_mut(), CALIBRATION_SEED).map_err(|err| err.to_string())?;
    Ok(model)
}

fn patchwise_model() -> Result<CosmosModelPatchwise, String> {
    let config = PatchwiseModelConfig::new((3, 32, 32), 10)
        .with_patch_size(4)
        .with_patch_subspace_dim(16)
        .with_agg_subspace_dim(256)
        .with_interior_subspace_dim(128)
        .with_num_interior_blocks(3)
        .with_cmr_k(32)
        .with_anchors_per_class(1);
    let mut model = CosmosModelPatchwise::new(config).map_err(|err| err.to_string())?;
    seed_state(model.get_state_mut(), CALIBRATION_SEED).map_err(|err| err.to_string())?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_assertion_holds() {
        let lines = run_all().expect("self-check");
        assert_eq!(lines.len(), 7);
    }

    #[test]
    fn the_calibration_dump_is_a_full_input_row() {
        let lines = first_row_dump();
        assert_eq!(lines.len(), 33);
        assert!(lines[0].contains("319635789"));
    }
}
