//! Invariants the gauge update preserves by construction.
//!
//! Assumptions:
//! - One gauge step runs per epoch here, so an epoch count is a step count: an
//!   arm that must exercise the periodic retraction has to run more epochs than
//!   its retraction interval.

mod common;

use common::{TrainOutcome, gaussian_clusters, seed_state, small_config, train_epochs};
use cosmos_rs::prelude::*;
use ndarray::Array2;

const SEED: u64 = 319_635_789;

fn trained_model(retraction_interval: usize, epochs: usize) -> (CosmosModel, TrainOutcome) {
    let (inputs, labels) = gaussian_clusters(24, 4, 32, 0.6, SEED);
    let batch = FlatClassificationBatch::single_label(inputs, labels);
    let mut model = CosmosModel::new(small_config(32, 4)).expect("model builds");
    seed_state(model.get_state_mut(), SEED);
    let outcome =
        train_epochs(&mut model, &batch, retraction_interval, epochs).expect("training runs");
    (model, outcome)
}

#[test]
fn orthogonality_is_preserved_for_every_retraction_interval() {
    // The interval-10 arm runs 25 steps so the retraction fires at steps 10 and
    // 20; without that the branch would never be entered.
    for (retraction_interval, epochs, expected_retractions) in
        [(0usize, 5usize, 0usize), (10, 25, 2)]
    {
        let (model, outcome) = trained_model(retraction_interval, epochs);
        assert_eq!(
            outcome.retractions, expected_retractions,
            "retraction_interval={retraction_interval} ran {} retraction(s) over {epochs} steps",
            outcome.retractions
        );

        for (block, error) in model.get_state().orthogonality_errors().iter().enumerate() {
            assert!(
                *error <= 1e-13,
                "block {block} drifted to {error:e} after {epochs} epochs at retraction_interval={retraction_interval}"
            );
        }
    }
}

#[test]
fn every_block_projector_is_idempotent() {
    let (model, _) = trained_model(0, 5);
    let state = model.get_state();
    for block in 0..state.num_blocks() {
        let p: Array2<f64> = state.get_projector(block).expect("projector exists");
        let pp = p.dot(&p);
        let max_dev = (&pp - &p).iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_dev <= 1e-12,
            "block {block}: ||P P - P||_max = {max_dev:e}"
        );
    }
}

#[test]
fn a_fixed_seed_gives_bit_identical_output() {
    fn run() -> Vec<f64> {
        let (model, outcome) = trained_model(10, 11);
        assert_eq!(outcome.retractions, 1, "the run must cross a retraction");
        model
            .get_state()
            .orthogonality_errors()
            .into_iter()
            .chain(std::iter::once(
                model.get_state().total_orthogonality_error(),
            ))
            .collect()
    }

    assert_eq!(run(), run(), "two runs with the same seed diverged");
}
