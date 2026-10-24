//! Train the paper's flat configuration, print its learning curve, and
//! round-trip a checkpoint.
//!
//! Demonstrates three things together: the paper's default flat
//! configuration (`784 -> 32`, five interior blocks of 16, `cmr_k = 8`), the
//! shape of its learning curve over eight epochs, and a full
//! `Checkpoint::save`/`load` round trip through `store_state`/`restore_state`.
//!
//! The library ships no dataset loader: the reproduction driver `cosmos-paper`
//! owns Fashion-MNIST and CIFAR-10. This example therefore takes its data as an
//! `Array2<f64>` of shape `(samples, 784)` plus one label per row, and `main`
//! supplies a synthetic stand-in. Replace `synthetic_batch` with your own loader
//! and nothing else changes.

use cosmos_rs::prelude::*;
use ndarray::Array2;

fn normal(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn synthetic_batch(
    samples: usize,
    dim: usize,
    classes: usize,
    seed: u64,
) -> (Array2<f64>, Vec<usize>) {
    let mut rng = seed;
    let centres = Array2::from_shape_fn((classes, dim), |_| normal(&mut rng) * 4.0);
    let inputs = Array2::from_shape_fn((samples, dim), |(row, col)| {
        centres[[row % classes, col]] + 0.6 * normal(&mut rng)
    });
    let labels: Vec<usize> = (0..samples).map(|row| row % classes).collect();
    (inputs, labels)
}

fn run(inputs: Array2<f64>, labels: Vec<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let seed = 319_635_789u64;
    let batch = FlatClassificationBatch::single_label(inputs, labels);
    let config = CosmosModelConfig::new(784, 10)
        .with_input_subspace_dim(32)
        .with_interior_subspace_dim(16)
        .with_num_interior_blocks(5)
        .with_cmr_k(8)
        .with_interior_topology(InteriorTopology::Series);
    let mut model = CosmosModel::new(config.clone())?;
    model.get_state_mut().seed_orthonormal(seed)?;

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
            lr: 0.01,
            momentum: 0.9,
            momentum_max_norm: Some(1.2),
            retraction_interval: 10,
            ..Default::default()
        },
    )?;

    for epoch in 1u64..=8 {
        model.set_epoch_batch(epoch, 0)?;
        let out = adapter.train_batch(&model, &batch, &mut backend, false)?;
        let grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| backend.device().download_2d(b)))
            .collect();
        let refs: Vec<Option<&Array2<f64>>> = grads.iter().map(|g| g.as_ref()).collect();
        rules.step(model.get_state_mut(), &refs)?;
        println!(
            "epoch {epoch}: accuracy {:.4}",
            out.metrics.accuracy.unwrap_or(0.0)
        );
    }

    let mut checkpoint = Checkpoint::new();
    store_state(model.get_state(), &mut checkpoint);
    let path = std::env::temp_dir().join("cosmos_flat_classifier.ckpt");
    checkpoint.save(&path)?;

    let reloaded = Checkpoint::load(&path)?;
    let mut restored = CosmosModel::new(config)?;
    restore_state(restored.get_state_mut(), &reloaded)?;
    std::fs::remove_file(&path)?;
    println!("checkpoint round-tripped through {}", path.display());

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (inputs, labels) = synthetic_batch(512, 784, 10, 319_635_789);
    run(inputs, labels)
}
