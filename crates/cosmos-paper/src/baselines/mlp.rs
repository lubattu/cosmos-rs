//! Two-layer network with Adam.
//!
//! Premises:
//! - Weights and biases of both layers are drawn uniformly from
//!   `[-1/sqrt(fan_in), +1/sqrt(fan_in)]`, the convention of the reference
//!   implementations this arm is compared against.
//! - Adam runs with bias correction and no weight decay, at a constant
//!   learning rate.
//! - The draw stream is tagged separately from the frame initialization, so
//!   the two never alias.

use cosmos_rs::prelude::{FlatClassificationBatch, Targets};
use ndarray::{Array1, Array2};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn uniform(seed: u64, tag: u64, row: usize, col: usize, bound: f64) -> f64 {
    let mut h = splitmix64(seed ^ 0x5DEE_CE66_D1B6_4B7D);
    h = splitmix64(h ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    h = splitmix64(h ^ (row as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    h = splitmix64(h ^ (col as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    let unit = ((h >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
    (2.0 * unit - 1.0) * bound
}

#[derive(Debug, Clone)]
struct AdamState {
    mean: Array2<f64>,
    variance: Array2<f64>,
}

impl AdamState {
    fn zeros(shape: (usize, usize)) -> Self {
        Self {
            mean: Array2::zeros(shape),
            variance: Array2::zeros(shape),
        }
    }
}

/// A two-layer network with a rectified hidden layer.
#[derive(Debug, Clone)]
pub struct Mlp {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    m1: AdamState,
    m2: AdamState,
    mb1: AdamState,
    mb2: AdamState,
    step: u64,
}

impl Mlp {
    /// Build the network with the reference initialization.
    pub fn new(input_dim: usize, hidden: usize, classes: usize, seed: u64) -> Self {
        let bound1 = 1.0 / (input_dim as f64).sqrt();
        let bound2 = 1.0 / (hidden as f64).sqrt();
        Self {
            w1: Array2::from_shape_fn((input_dim, hidden), |(row, col)| {
                uniform(seed, 1, row, col, bound1)
            }),
            b1: Array1::from_shape_fn(hidden, |col| uniform(seed, 2, 0, col, bound1)),
            w2: Array2::from_shape_fn((hidden, classes), |(row, col)| {
                uniform(seed, 3, row, col, bound2)
            }),
            b2: Array1::from_shape_fn(classes, |col| uniform(seed, 4, 0, col, bound2)),
            m1: AdamState::zeros((input_dim, hidden)),
            m2: AdamState::zeros((hidden, classes)),
            mb1: AdamState::zeros((1, hidden)),
            mb2: AdamState::zeros((1, classes)),
            step: 0,
        }
    }

    /// Number of learnable parameters.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn forward(&self, inputs: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut hidden = inputs.dot(&self.w1);
        for mut row in hidden.rows_mut() {
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value + self.b1[index]).max(0.0);
            }
        }
        let mut logits = hidden.dot(&self.w2);
        for mut row in logits.rows_mut() {
            for (index, value) in row.iter_mut().enumerate() {
                *value += self.b2[index];
            }
        }
        (hidden, logits)
    }

    fn softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
        let mut probabilities = logits.clone();
        for mut row in probabilities.rows_mut() {
            let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for value in row.iter_mut() {
                *value = (*value - peak).exp();
                total += *value;
            }
            for value in row.iter_mut() {
                *value /= total;
            }
        }
        probabilities
    }

    fn accuracy(logits: &Array2<f64>, labels: &[usize]) -> (usize, usize) {
        let mut correct = 0;
        for (row, &label) in labels.iter().enumerate() {
            let mut best = 0;
            let mut best_value = f64::NEG_INFINITY;
            for class in 0..logits.ncols() {
                if logits[[row, class]] > best_value {
                    best_value = logits[[row, class]];
                    best = class;
                }
            }
            if best == label {
                correct += 1;
            }
        }
        (correct, labels.len())
    }

    fn adam_update(
        parameter: &mut Array2<f64>,
        gradient: &Array2<f64>,
        state: &mut AdamState,
        lr: f64,
        step: u64,
    ) {
        let correction1 = 1.0 - BETA1.powi(step as i32);
        let correction2 = 1.0 - BETA2.powi(step as i32);
        for ((value, grad), (mean, variance)) in parameter
            .iter_mut()
            .zip(gradient.iter())
            .zip(state.mean.iter_mut().zip(state.variance.iter_mut()))
        {
            *mean = BETA1 * *mean + (1.0 - BETA1) * grad;
            *variance = BETA2 * *variance + (1.0 - BETA2) * grad * grad;
            let hat_mean = *mean / correction1;
            let hat_variance = *variance / correction2;
            *value -= lr * hat_mean / (hat_variance.sqrt() + EPS);
        }
    }

    /// Train one epoch and return the mean loss and the accuracy in percent.
    ///
    /// # Panics
    ///
    /// Panics when a batch carries multi-label supervision.
    pub fn train_epoch(&mut self, batches: &[FlatClassificationBatch], lr: f64) -> (f64, f64) {
        let mut loss_sum = 0.0;
        let mut sample_count = 0usize;
        let mut correct_total = 0usize;

        for batch in batches {
            let labels = single_labels(batch);
            let samples = labels.len();
            let (hidden, logits) = self.forward(&batch.inputs);
            let probabilities = Self::softmax_rows(&logits);

            let mut batch_loss = 0.0;
            let mut delta = probabilities.clone();
            for (row, &label) in labels.iter().enumerate() {
                batch_loss -= probabilities[[row, label]].max(1e-15).ln();
                delta[[row, label]] -= 1.0;
            }
            delta /= samples as f64;

            let (correct, _) = Self::accuracy(&logits, &labels);
            correct_total += correct;
            loss_sum += batch_loss;
            sample_count += samples;

            let grad_w2 = hidden.t().dot(&delta);
            let grad_b2 = delta.sum_axis(ndarray::Axis(0));
            let mut grad_hidden = delta.dot(&self.w2.t());
            for (value, activation) in grad_hidden.iter_mut().zip(hidden.iter()) {
                if *activation <= 0.0 {
                    *value = 0.0;
                }
            }
            let grad_w1 = batch.inputs.t().dot(&grad_hidden);
            let grad_b1 = grad_hidden.sum_axis(ndarray::Axis(0));

            self.step += 1;
            let step = self.step;
            Self::adam_update(&mut self.w1, &grad_w1, &mut self.m1, lr, step);
            Self::adam_update(&mut self.w2, &grad_w2, &mut self.m2, lr, step);
            let mut b1 = self.b1.clone().insert_axis(ndarray::Axis(0));
            let g1 = grad_b1.clone().insert_axis(ndarray::Axis(0));
            Self::adam_update(&mut b1, &g1, &mut self.mb1, lr, step);
            self.b1 = b1.remove_axis(ndarray::Axis(0));
            let mut b2 = self.b2.clone().insert_axis(ndarray::Axis(0));
            let g2 = grad_b2.clone().insert_axis(ndarray::Axis(0));
            Self::adam_update(&mut b2, &g2, &mut self.mb2, lr, step);
            self.b2 = b2.remove_axis(ndarray::Axis(0));
        }

        (
            loss_sum / sample_count.max(1) as f64,
            100.0 * correct_total as f64 / sample_count.max(1) as f64,
        )
    }

    /// Accuracy in percent over `batches`.
    ///
    /// # Panics
    ///
    /// Panics when a batch carries multi-label supervision.
    pub fn evaluate(&self, batches: &[FlatClassificationBatch]) -> f64 {
        let mut correct_total = 0usize;
        let mut sample_count = 0usize;
        for batch in batches {
            let labels = single_labels(batch);
            let (_, logits) = self.forward(&batch.inputs);
            let (correct, samples) = Self::accuracy(&logits, &labels);
            correct_total += correct;
            sample_count += samples;
        }
        100.0 * correct_total as f64 / sample_count.max(1) as f64
    }
}

/// Parameter count of a two-layer network with the given shape.
pub fn param_count(input_dim: usize, hidden: usize, classes: usize) -> usize {
    input_dim * hidden + hidden + hidden * classes + classes
}

fn single_labels(batch: &FlatClassificationBatch) -> Vec<usize> {
    match &batch.targets {
        Targets::SingleLabel(labels) => labels.clone(),
        Targets::MultiLabel(_) => {
            panic!("the comparison arms are trained on single-label batches")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_counts_match_the_comparison_targets() {
        assert_eq!(param_count(784, 189, 10), 150_265);
        assert_eq!(param_count(3072, 191, 10), 588_863);
    }

    #[test]
    fn parameter_gap_to_the_frame_count_is_below_two_percent() {
        let gap = (150_528_f64 - 150_265.0).abs() / 150_528.0;
        assert!(gap < 0.02);
        let gap = (589_824_f64 - 588_863.0).abs() / 589_824.0;
        assert!(gap < 0.02);
    }

    #[test]
    fn initialization_is_deterministic_and_bounded() {
        let first = Mlp::new(16, 8, 4, 7);
        let second = Mlp::new(16, 8, 4, 7);
        assert_eq!(first.w1, second.w1);
        let bound = 1.0 / 16.0_f64.sqrt();
        assert!(first.w1.iter().all(|value| value.abs() <= bound));
        assert_eq!(first.param_count(), param_count(16, 8, 4));
    }

    #[test]
    fn one_epoch_reduces_the_loss_on_a_separable_problem() {
        use ndarray::Array2;
        let inputs =
            Array2::from_shape_fn((64, 4), |(row, col)| if col == row % 4 { 1.0 } else { 0.0 });
        let labels: Vec<usize> = (0..64).map(|row| row % 4).collect();
        let batches = vec![cosmos_rs::prelude::FlatClassificationBatch::single_label(
            inputs, labels,
        )];
        let mut model = Mlp::new(4, 16, 4, 11);
        let (first_loss, _) = model.train_epoch(&batches, 1e-2);
        let mut last = first_loss;
        for _ in 0..20 {
            let (loss, _) = model.train_epoch(&batches, 1e-2);
            last = loss;
        }
        assert!(last < first_loss);
        assert!(model.evaluate(&batches) > 50.0);
    }
}
