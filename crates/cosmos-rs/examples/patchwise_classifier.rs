//! Train the patchwise model on synthetic 3x8x8 images.
//!
//! The paper's E5 arm uses CIFAR-10 at 3x32x32 with patch 4, `k_p` 16,
//! `d_agg` 256 and three interior blocks of 128. This example keeps the shape of
//! that configuration and shrinks every dimension so it runs as an example.

use cosmos_rs::prelude::*;
use ndarray::Array4;

fn normal(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (samples, channels, height, width, classes) = (64, 3, 8, 8, 4);
    let mut rng = 319_635_789u64;
    let images = Array4::from_shape_fn((samples, channels, height, width), |_| normal(&mut rng));
    let labels: Vec<usize> = (0..samples).map(|i| i % classes).collect();
    let batch = PatchwiseClassificationBatch::single_label(images, labels);

    let mut model = CosmosModelPatchwise::new(
        PatchwiseModelConfig::new((channels, height, width), classes)
            .with_patch_size(4)
            .with_patch_subspace_dim(8)
            .with_agg_subspace_dim(32)
            .with_interior_subspace_dim(16)
            .with_num_interior_blocks(2)
            .with_cmr_k(8)
            .with_anchors_per_class(1)
            .with_interior_topology(InteriorTopology::Series),
    )?;

    let mut adapter = PatchwiseClassificationAdapter::new(PatchwiseClassificationAdapterConfig {
        loss: ClassificationLoss::GeodesicHinge {
            margin: std::f64::consts::PI / 5.0,
        },
        ..Default::default()
    })?;
    let mut backend = DeviceBackend::default();
    let mut rules = GaugeDynamics::new(
        model.get_state().num_blocks(),
        GaugeDynamicsConfig {
            lr: 0.05,
            momentum: 0.9,
            ..Default::default()
        },
    )?;

    let mut accuracy = 0.0;
    for epoch in 1u64..=3 {
        model.set_epoch_batch(epoch, 0)?;
        let out = adapter.train_batch(&model, &batch, &mut backend, false)?;
        let grads: Vec<Option<ndarray::Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| backend.device().download_2d(b)))
            .collect();
        let refs: Vec<Option<&ndarray::Array2<f64>>> = grads.iter().map(|g| g.as_ref()).collect();
        rules.step(model.get_state_mut(), &refs)?;
        accuracy = out.metrics.accuracy.unwrap_or(0.0);
    }

    println!("patchwise accuracy after 3 epochs: {accuracy:.4}");
    Ok(())
}
