//! Train a COSMOS classifier on synthetic Gaussian clusters.
//!
//! Seeding the state with an orthonormal basis before training avoids the
//! identity-basis plateau; six epochs at `lr = 0.1` then reach peak accuracy.
//!
//! Run with `cargo run --example quickstart_synthetic`.

use cosmos_rs::prelude::*;
use ndarray::Array2;

fn normal(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (dim, classes, per_class) = (32, 4, 64);
    let mut rng = 319_635_789u64;
    let centres = Array2::from_shape_fn((classes, dim), |_| normal(&mut rng) * 4.0);
    let inputs = Array2::from_shape_fn((classes * per_class, dim), |(row, col)| {
        centres[[row / per_class, col]] + 0.6 * normal(&mut rng)
    });
    let labels: Vec<usize> = (0..classes * per_class).map(|i| i / per_class).collect();
    let batch = FlatClassificationBatch::single_label(inputs, labels);

    let mut model = CosmosModel::new(
        CosmosModelConfig::new(dim, classes)
            .with_input_subspace_dim(8)
            .with_interior_subspace_dim(4)
            .with_num_interior_blocks(2)
            .with_cmr_k(4)
            .with_interior_topology(InteriorTopology::Series),
    )?;
    model.get_state_mut().seed_orthonormal(319_635_789)?;

    let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
        loss: ClassificationLoss::GeodesicHinge {
            margin: std::f64::consts::PI / 5.0,
        },
        ..Default::default()
    })?;
    let mut backend = DeviceBackend::default();
    let mut rules = GaugeDynamics::new(
        model.get_state().num_blocks(),
        GaugeDynamicsConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        },
    )?;

    for epoch in 1u64..=6 {
        model.set_epoch_batch(epoch, 0)?;
        let out = adapter.train_batch(&model, &batch, &mut backend, false)?;
        let grads: Vec<Option<ndarray::Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| backend.device().download_2d(b)))
            .collect();
        let refs: Vec<Option<&ndarray::Array2<f64>>> = grads.iter().map(|g| g.as_ref()).collect();
        rules.step(model.get_state_mut(), &refs)?;
        println!(
            "epoch {epoch}: accuracy {:.4}",
            out.metrics.accuracy.unwrap_or(0.0)
        );
    }

    println!(
        "orthogonality error {:.3e}",
        model.get_state().total_orthogonality_error()
    );
    Ok(())
}
