//! Shared fixtures for the integration tests.
//!
//! Assumptions:
//! - Every generator here is deterministic in its seed. No test may depend on
//!   system entropy, thread scheduling or wall-clock time.
//! - Data is synthetic. The crate ships no dataset loader; reproducing the paper
//!   is the business of the `cosmos-paper` crate.
//! - Each test binary compiles its own copy of this module and uses a subset of
//!   it, so unused items are expected here.

#![allow(dead_code)]

use cosmos_rs::geometry::qr::{QrConfig, qr_orthonormalize};
use cosmos_rs::prelude::*;
use ndarray::Array2;

/// Deterministic 64-bit mixer (SplitMix64).
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One standard normal draw, by Box-Muller over two uniforms.
pub fn standard_normal(state: &mut u64) -> f64 {
    let u1 = ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Isotropic Gaussian clusters, one per class, on well-separated centres.
///
/// Returns `(inputs, labels)` with `inputs` of shape
/// `(samples_per_class * num_classes, dim)` in class-major order.
pub fn gaussian_clusters(
    samples_per_class: usize,
    num_classes: usize,
    dim: usize,
    spread: f64,
    seed: u64,
) -> (Array2<f64>, Vec<usize>) {
    let mut centre_rng = seed ^ 0x5DEE_CE66_D000_0000;
    let centres = Array2::from_shape_fn((num_classes, dim), |_| {
        standard_normal(&mut centre_rng) * 4.0
    });

    let rows = samples_per_class * num_classes;
    let mut inputs = Array2::<f64>::zeros((rows, dim));
    let mut labels = Vec::with_capacity(rows);
    let mut rng = seed;

    for class in 0..num_classes {
        for i in 0..samples_per_class {
            let row = class * samples_per_class + i;
            for col in 0..dim {
                inputs[[row, col]] = centres[[class, col]] + spread * standard_normal(&mut rng);
            }
            labels.push(class);
        }
    }

    (inputs, labels)
}

/// Seed every state block with an orthonormalized Gaussian matrix.
///
/// This is the initialization the paper harness uses: draw `N(0, 1)`, then take
/// the `Q` factor, so each block starts uniformly on its Stiefel manifold.
///
/// # Panics
///
/// Panics if a block basis is rejected by the state, which cannot happen since
/// every matrix is drawn at the state's own block shape.
pub fn seed_state(state: &mut CosmosState, seed: u64) {
    let qr = QrConfig::default();
    let shapes = state.block_shapes().to_vec();

    for (block, (n, k)) in shapes.into_iter().enumerate() {
        let mut rng = seed ^ ((block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let raw = Array2::from_shape_fn((n, k), |_| standard_normal(&mut rng));
        state
            .set_block_basis(block, qr_orthonormalize(&raw, &qr))
            .expect("block basis matches the state layout");
    }
}

/// A small flat config that trains in well under a second.
pub fn small_config(input_dim: usize, num_classes: usize) -> CosmosModelConfig {
    CosmosModelConfig::new(input_dim, num_classes)
        .with_input_subspace_dim(8)
        .with_interior_subspace_dim(4)
        .with_num_interior_blocks(2)
        .with_cmr_k(4)
        .with_anchors_per_class(1)
        .with_interior_topology(InteriorTopology::Series)
}

/// The paper's E3 flat configuration: 784 -> 32, five interior blocks of 16,
/// `cmr_k = 8`, ten classes, one anchor per class, series topology.
pub fn paper_e3_config() -> CosmosModelConfig {
    CosmosModelConfig::new(784, 10)
        .with_input_subspace_dim(32)
        .with_interior_subspace_dim(16)
        .with_num_interior_blocks(5)
        .with_cmr_k(8)
        .with_anchors_per_class(1)
        .with_interior_topology(InteriorTopology::Series)
}

/// What a short training run reports back to a test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainOutcome {
    /// Training accuracy of the last epoch.
    pub accuracy: f64,
    /// Number of gauge steps that ran the periodic retraction.
    pub retractions: usize,
}

/// Train `epochs` full-batch epochs, one gauge step per epoch.
///
/// # Errors
///
/// Returns any adapter, model or gauge-dynamics failure raised on the way.
pub fn train_epochs(
    model: &mut CosmosModel,
    batch: &FlatClassificationBatch,
    retraction_interval: usize,
    epochs: usize,
) -> Result<TrainOutcome, Box<dyn std::error::Error>> {
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
            lr: 0.05,
            momentum: 0.9,
            retraction_interval,
            ..Default::default()
        },
    )?;

    let mut accuracy = 0.0;
    let mut retractions = 0usize;
    for epoch in 1..=epochs {
        model.set_epoch_batch(epoch as u64, 0)?;
        let out = adapter.train_batch(model, batch, &mut backend, false)?;
        let device = backend.device();
        let grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| device.download_2d(b)))
            .collect();
        let refs: Vec<Option<&Array2<f64>>> = grads.iter().map(|g| g.as_ref()).collect();
        let report = rules.step(model.get_state_mut(), &refs)?;
        if report.profile.used_retraction {
            retractions += 1;
        }
        accuracy = out.metrics.accuracy.unwrap_or(0.0);
    }

    Ok(TrainOutcome {
        accuracy,
        retractions,
    })
}
