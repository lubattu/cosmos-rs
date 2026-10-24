//! Use COSMOS on your own data.
//!
//! You supply two things: a feature matrix `Array2<f64>` of shape
//! `(samples, features)` and one class index per row. You implement no trait and
//! write no layer. This example walks the four decisions that matter:
//!
//! 1. `input_subspace_dim` — the width of the subspace the input is projected
//!    onto. Smaller is a stronger bottleneck.
//! 2. `num_interior_blocks` and `interior_subspace_dim` — the depth and width of
//!    the interior chain.
//! 3. `cmr_k` — the dimension of each class's anchor subspace. It must not exceed
//!    `input_subspace_dim`.
//! 4. `topology` — `Series`, `Parallel`, `Residual` or `ParallelSeries`.
//!
//! Features should be centred and comparably scaled: the geometry reads angles
//! between subspaces, so a feature with a dominant scale dominates the geometry.
//!
//! Interleaved spirals need many gauge steps at this learning rate to move
//! past the default identity-basis plateau, so this example seeds the state
//! with [`CosmosState::seed_orthonormal`] and trains for 100 epochs; ten
//! epochs stay near chance.

use cosmos_rs::prelude::*;
use ndarray::Array2;

fn normal(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Two interleaved spirals generated in 2D, then embedded into `dim`
/// dimensions through a fixed random linear map so every column carries
/// signal rather than the last `dim - 2` columns being pure noise.
///
/// COSMOS scores an embedding against a class anchor by the angle between the
/// embedding vector and the anchor subspace, which cannot tell a vector from
/// its negation. The two arms are offset by a quarter turn plus a per-arm
/// radius shift rather than the classic half-turn rotation, so no point in
/// one arm is the negation of a point in the other.
fn two_spirals(points_per_arm: usize, dim: usize, seed: u64) -> (Array2<f64>, Vec<usize>) {
    let mut rng = seed;
    let embed = Array2::from_shape_fn((2, dim), |_| normal(&mut rng));

    let rows = points_per_arm * 2;
    let mut plane = Array2::<f64>::zeros((rows, 2));
    let mut labels = Vec::with_capacity(rows);

    for arm in 0..2 {
        for i in 0..points_per_arm {
            let row = arm * points_per_arm + i;
            let t = 1.2 * std::f64::consts::PI * (i as f64 + 1.0) / points_per_arm as f64;
            let radius = 0.5 + 0.4 * t / (1.2 * std::f64::consts::PI) + arm as f64 * 0.6;
            let angle = t + arm as f64 * std::f64::consts::FRAC_PI_2;
            plane[[row, 0]] = radius * angle.cos();
            plane[[row, 1]] = radius * angle.sin();
            labels.push(arm);
        }
    }

    let mut inputs = plane.dot(&embed);
    for value in inputs.iter_mut() {
        *value += 0.05 * normal(&mut rng);
    }

    (inputs, labels)
}

fn standardize(inputs: &mut Array2<f64>) {
    let (rows, cols) = inputs.dim();
    for col in 0..cols {
        let mean: f64 = (0..rows).map(|row| inputs[[row, col]]).sum::<f64>() / rows as f64;
        let variance: f64 = (0..rows)
            .map(|row| (inputs[[row, col]] - mean).powi(2))
            .sum::<f64>()
            / rows as f64;
        let std_dev = variance.sqrt().max(1e-12);
        for row in 0..rows {
            inputs[[row, col]] = (inputs[[row, col]] - mean) / std_dev;
        }
    }
}

fn split_80_20(
    inputs: &Array2<f64>,
    labels: &[usize],
    seed: u64,
) -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
    let rows = inputs.nrows();
    let mut order: Vec<usize> = (0..rows).collect();
    let mut rng = seed;
    for i in (1..order.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % (i + 1);
        order.swap(i, j);
    }

    let split = rows * 4 / 5;
    let train_idx = &order[..split];
    let val_idx = &order[split..];

    let gather = |idx: &[usize]| -> (Array2<f64>, Vec<usize>) {
        let cols = inputs.ncols();
        let mut out = Array2::<f64>::zeros((idx.len(), cols));
        let mut lbls = Vec::with_capacity(idx.len());
        for (dst, &src) in idx.iter().enumerate() {
            for col in 0..cols {
                out[[dst, col]] = inputs[[src, col]];
            }
            lbls.push(labels[src]);
        }
        (out, lbls)
    };

    let (train_inputs, train_labels) = gather(train_idx);
    let (val_inputs, val_labels) = gather(val_idx);
    (train_inputs, train_labels, val_inputs, val_labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 10;
    let seed = 319_635_789u64;
    let (mut inputs, labels) = two_spirals(150, dim, seed);
    standardize(&mut inputs);
    let (train_inputs, train_labels, val_inputs, val_labels) = split_80_20(&inputs, &labels, seed);

    let train_batch = FlatClassificationBatch::single_label(train_inputs, train_labels);
    let val_batch = FlatClassificationBatch::single_label(val_inputs, val_labels);

    let mut model = CosmosModel::new(
        CosmosModelConfig::new(dim, 2)
            .with_input_subspace_dim(8)
            .with_interior_subspace_dim(4)
            .with_num_interior_blocks(2)
            .with_cmr_k(4)
            .with_interior_topology(InteriorTopology::Series),
    )?;
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
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        },
    )?;

    for epoch in 1u64..=100 {
        model.set_epoch_batch(epoch, 0)?;
        let out = adapter.train_batch(&model, &train_batch, &mut backend, false)?;
        let grads: Vec<Option<Array2<f64>>> = out
            .gradients
            .iter()
            .map(|g| g.as_ref().map(|b| backend.device().download_2d(b)))
            .collect();
        let refs: Vec<Option<&Array2<f64>>> = grads.iter().map(|g| g.as_ref()).collect();
        rules.step(model.get_state_mut(), &refs)?;
    }

    let train_out = adapter.eval_batch(&model, &train_batch, &mut backend, false)?;
    let val_out = adapter.eval_batch(&model, &val_batch, &mut backend, false)?;
    println!(
        "train accuracy {:.4}, validation accuracy {:.4}",
        train_out.metrics.accuracy.unwrap_or(0.0),
        val_out.metrics.accuracy.unwrap_or(0.0)
    );
    Ok(())
}
