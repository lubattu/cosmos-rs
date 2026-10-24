//! SAOR contracts, encoded mechanically.
//!
//! A contribution that breaks one of these fails CI, which spec §7.1 makes an
//! automatic rejection.

mod common;

use common::{gaussian_clusters, paper_e3_config, seed_state, small_config};
use cosmos_rs::objectives::cmr_losses::{
    RegionMetrics, direct_cmr_loss, geodesic_cmr_hinge_loss, geodesic_cmr_loss,
};
use cosmos_rs::objectives::utils::Targets;
use cosmos_rs::objectives::{ObjectiveError, Reduction};
use cosmos_rs::prelude::*;
use ndarray::{Array1, Array2};

const SEED: u64 = 319_635_789;

type LossResult = Result<Array1<f64>, ObjectiveError>;
type DirectLossFn = fn(&Array2<f64>, &Targets, Reduction) -> LossResult;
type GeodesicLossFn = fn(&Array2<f64>, &[RegionMetrics], &Targets, Reduction) -> LossResult;
type HingeLossFn = fn(&Array2<f64>, &Targets, f64, Reduction) -> LossResult;

/// Contract 1 — everything that learns lives in the State.
///
/// The model reports exactly the parameters the state holds, so no model,
/// arrangement or head field can be a learnable tensor hidden from the count.
#[test]
fn every_parameter_is_a_state_block() {
    let model = CosmosModel::new(paper_e3_config()).expect("model builds");
    let state = model.get_state();

    let from_shapes: usize = state.block_shapes().iter().map(|(n, k)| n * k).sum();
    assert_eq!(count_state_parameters(state), from_shapes);

    let reported = model.count_parameters();
    assert_eq!(reported["total"], from_shapes);
    assert_eq!(reported["state_total"], from_shapes);
}

/// The E3 configuration's parameter count.
///
/// Layout: one input block `(784, 32)`, five interior blocks `(784, 16)` and ten
/// CMR anchor blocks `(784, 8)` — the ambient frame keeps every downstream block
/// on the ambient input dimension. That is
/// `25_088 + 5 * 12_544 + 10 * 6_272 = 150_528`, the figure the article reports.
/// The assertion is written against the layout so a change in the layout fails
/// here rather than silently in a results table.
#[test]
fn the_e3_configuration_holds_150_528_parameters() {
    let model = CosmosModel::new(paper_e3_config()).expect("model builds");
    let shapes = model.get_state().block_shapes();

    assert_eq!(shapes.len(), 1 + 5 + 10);
    assert_eq!(shapes[0], (784, 32));
    assert!(shapes[1..6].iter().all(|&s| s == (784, 16)));
    assert!(shapes[6..].iter().all(|&s| s == (784, 8)));

    let expected = 784 * 32 + 5 * (784 * 16) + 10 * (784 * 8);
    assert_eq!(expected, 150_528);
    assert_eq!(count_state_parameters(model.get_state()), 150_528);
}

/// Contract 2 — the Arrangement is gradient-free.
///
/// An evaluation pass routes through every arrangement primitive. If any of them
/// owned a mutable learnable tensor, it would have to live in the state, and the
/// state's serialized bytes would move.
#[test]
fn evaluation_does_not_move_the_state() {
    let (inputs, labels) = gaussian_clusters(16, 4, 32, 0.6, SEED);
    let batch = FlatClassificationBatch::single_label(inputs, labels);

    let mut model = CosmosModel::new(small_config(32, 4)).expect("model builds");
    seed_state(model.get_state_mut(), SEED);

    let mut before = Checkpoint::new();
    store_state(model.get_state(), &mut before);

    let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: ClassificationLoss::DirectMembership,
        ..Default::default()
    })
    .expect("adapter builds");
    let mut backend = DeviceBackend::default();
    adapter
        .eval_batch(&model, &batch, &mut backend, false)
        .expect("evaluation runs");

    let mut after = Checkpoint::new();
    store_state(model.get_state(), &mut after);

    assert_eq!(before.to_bytes(), after.to_bytes());
}

/// Contract 3 — objectives evaluate; they do not act.
///
/// Every objective coerces to a `fn` pointer whose every parameter is a shared
/// reference or a value. Adding a `&mut` parameter to any of them makes this
/// test fail to compile.
#[test]
fn objectives_take_no_mutable_reference() {
    let _direct: DirectLossFn = direct_cmr_loss;
    let _geodesic: GeodesicLossFn = geodesic_cmr_loss;
    let _hinge: HingeLossFn = geodesic_cmr_hinge_loss;
}
