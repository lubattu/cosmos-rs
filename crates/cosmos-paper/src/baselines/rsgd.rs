//! Per-frame Riemannian SGD on the same state.
//!
//! Premises:
//! - This arm is not a different architecture. It trains the same state,
//!   the same readout and the same objective as the COSMOS arm, and differs
//!   only in how a gradient becomes a state update, so its parameter count
//!   is identical.
//! - There is no momentum term. The step is projection, descent,
//!   retraction.

use cosmos_rs::compute::Buffer2;
use cosmos_rs::geometry::grassmann::retract_qr;
use cosmos_rs::prelude::{CosmosState, CosmosStateError};
use ndarray::Array2;

fn symmetric_part(matrix: &Array2<f64>) -> Array2<f64> {
    0.5 * (matrix + &matrix.t().to_owned())
}

/// Project `gradient` onto the tangent space of the frame `basis`.
pub fn tangent_projection(basis: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
    let inner = basis.t().dot(gradient);
    gradient - &basis.dot(&symmetric_part(&inner))
}

/// Take one Riemannian step on every block that has a gradient.
///
/// # Errors
///
/// Returns [`CosmosStateError`] when a block index is out of range or a
/// retracted basis has the wrong shape.
pub fn step(
    state: &mut CosmosState,
    gradients: &[Option<Buffer2<f64>>],
    lr: f64,
) -> Result<(), CosmosStateError> {
    for (block_idx, gradient) in gradients.iter().enumerate() {
        let Some(gradient) = gradient else {
            continue;
        };
        let basis = state.get_basis(block_idx)?.clone();
        let tangent = tangent_projection(&basis, gradient.as_array());
        let moved = &basis - &(lr * &tangent);
        state.set_block_basis(block_idx, retract_qr(&moved))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Array2<f64> {
        let raw = Array2::from_shape_fn((12, 4), |(row, col)| {
            ((row * 7 + col * 3) % 11) as f64 - 5.0
        });
        retract_qr(&raw)
    }

    #[test]
    fn tangent_direction_is_orthogonal_in_the_symmetric_sense() {
        let basis = frame();
        let gradient = Array2::from_shape_fn((12, 4), |(row, col)| (row as f64) - (col as f64));
        let tangent = tangent_projection(&basis, &gradient);
        let inner = basis.t().dot(&tangent);
        let symmetric = symmetric_part(&inner);
        assert!(symmetric.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn a_step_preserves_orthonormality() {
        let basis = frame();
        let mut state = CosmosState::new(&[(12, 4)]).expect("state");
        state.set_block_basis(0, basis).expect("install");
        let gradient =
            Array2::from_shape_fn((12, 4), |(row, col)| ((row * 5 + col) % 7) as f64 - 3.0);
        step(&mut state, &[Some(Buffer2::from_array(gradient))], 0.3).expect("step");
        assert!(state.orthogonality_errors()[0] < 1e-12);
    }

    #[test]
    fn a_zero_gradient_leaves_the_frame_where_it_was() {
        let basis = frame();
        let mut state = CosmosState::new(&[(12, 4)]).expect("state");
        state.set_block_basis(0, basis.clone()).expect("install");
        step(
            &mut state,
            &[Some(Buffer2::from_array(Array2::zeros((12, 4))))],
            0.3,
        )
        .expect("step");
        let updated = state.get_basis(0).expect("basis");
        for row in 0..12 {
            for col in 0..4 {
                assert!((updated[[row, col]].abs() - basis[[row, col]].abs()).abs() < 1e-10);
            }
        }
    }
}
