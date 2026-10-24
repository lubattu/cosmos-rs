//! Flat input projection into a state-defined subspace.
//!
//! [`FlatInputProjector`] applies one block projector from [`CosmosState`] to a
//! batch-major input:
//!
//! `y = x W W^T`
//!
//! where `W` is the orthonormal basis stored at a chosen block index.

use ndarray::{Array1, Array2, Axis};

use crate::{
    arrangement::{ArrangementError, utils::get_w_block},
    state::cosmos_state::CosmosState,
};

/// Parameter-free projector for flat feature vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlatInputProjector {
    block_idx: usize,
}

impl FlatInputProjector {
    /// Create a projector bound to one block index in [`CosmosState`].
    pub fn new(block_idx: usize) -> Self {
        Self { block_idx }
    }

    /// Return the bound block index.
    pub fn block_idx(&self) -> usize {
        self.block_idx
    }

    /// Return the ambient input dimension expected by this projector.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InvalidParameter`] with `param: "block_idx"`
    /// when the configured block index is out of range for `state`.
    pub fn input_dim(&self, state: &CosmosState) -> Result<usize, ArrangementError> {
        let w = get_w_block(state, self.block_idx)?;
        Ok(w.nrows())
    }

    /// Return the subspace dimension of the selected state block.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InvalidParameter`] with `param: "block_idx"`
    /// when the configured block index is out of range for `state`.
    pub fn subspace_dim(&self, state: &CosmosState) -> Result<usize, ArrangementError> {
        let w = get_w_block(state, self.block_idx)?;
        Ok(w.ncols())
    }

    /// Materialize the projector matrix `W W^T`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrangementError::InvalidParameter`] with `param: "block_idx"`
    /// when the configured block index is out of range for `state`.
    pub fn projector(&self, state: &CosmosState) -> Result<Array2<f64>, ArrangementError> {
        let w = get_w_block(state, self.block_idx)?;
        Ok(w.dot(&w.t()))
    }

    /// Project a batch-major input through the selected state block.
    ///
    /// `x` must have shape `(batch, n)` where `n` matches the ambient
    /// dimension of the selected state block.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward(
        &self,
        x: &Array2<f64>,
        state: &CosmosState,
    ) -> Result<Array2<f64>, ArrangementError> {
        let w = get_w_block(state, self.block_idx)?;

        if x.ncols() != w.nrows() {
            return Err(ArrangementError::ShapeMismatch {
                expected: format!("batch-major input with {} columns", w.nrows()),
                got: format!("({}, {})", x.nrows(), x.ncols()),
            });
        }

        Ok(x.dot(&w).dot(&w.t()))
    }

    /// Project a single sample through the selected state block.
    ///
    /// # Errors
    ///
    /// Returns an [`ArrangementError`] when the operand shapes do not line up or a
    /// parameter fails validation.
    pub fn forward_one(
        &self,
        x: &Array1<f64>,
        state: &CosmosState,
    ) -> Result<Array1<f64>, ArrangementError> {
        let batch = x.clone().insert_axis(Axis(0));
        let output = self.forward(&batch, state)?;
        Ok(output.index_axis(Axis(0), 0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;

    const EPSILON: f64 = 1e-10;

    fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < EPSILON,
                "expected {e}, got {a} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn x_axis() -> Array2<f64> {
        array![[1.0], [0.0]]
    }

    fn y_axis() -> Array2<f64> {
        array![[0.0], [1.0]]
    }

    fn diagonal_xy() -> Array2<f64> {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        array![[inv_sqrt2], [inv_sqrt2]]
    }

    fn build_state(blocks: &[Array2<f64>]) -> CosmosState {
        let mut state = CosmosState::new(&[]).unwrap();
        for block in blocks {
            state.register_external_block(block).unwrap();
        }
        state
    }

    #[test]
    fn new_stores_block_index() {
        let projector = FlatInputProjector::new(3);
        assert_eq!(projector.block_idx(), 3);
    }

    #[test]
    fn input_dim_and_subspace_dim_match_state_block() {
        let state = build_state(&[x_axis()]);
        let projector = FlatInputProjector::new(0);

        assert_eq!(projector.input_dim(&state).unwrap(), 2);
        assert_eq!(projector.subspace_dim(&state).unwrap(), 1);
    }

    #[test]
    fn projector_matches_state_basis_product() {
        let state = build_state(&[diagonal_xy()]);
        let projector = FlatInputProjector::new(0);
        let expected = array![[0.5, 0.5], [0.5, 0.5]];

        let actual = projector.projector(&state).unwrap();

        assert_array2_close(&actual, &expected);
    }

    #[test]
    fn forward_projects_onto_selected_axis() {
        let state = build_state(&[x_axis()]);
        let projector = FlatInputProjector::new(0);
        let x = array![[1.0, 2.0], [-3.0, 4.0]];

        let out = projector.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 0.0], [-3.0, 0.0]]);
    }

    #[test]
    fn forward_with_diagonal_subspace_matches_manual_projection() {
        let state = build_state(&[diagonal_xy()]);
        let projector = FlatInputProjector::new(0);
        let x = array![[1.0, 1.0], [1.0, -1.0]];

        let out = projector.forward(&x, &state).unwrap();

        assert_array2_close(&out, &array![[1.0, 1.0], [0.0, 0.0]]);
    }

    #[test]
    fn forward_one_matches_batch_forward() {
        let state = build_state(&[y_axis()]);
        let projector = FlatInputProjector::new(0);
        let x = array![1.0, 2.0];

        let single = projector.forward_one(&x, &state).unwrap();
        let batch = projector
            .forward(&x.clone().insert_axis(Axis(0)), &state)
            .unwrap();

        assert_array1_close(&single, &batch.index_axis(Axis(0), 0).to_owned());
    }

    #[test]
    fn forward_rejects_mismatched_input_dimension() {
        let state = build_state(&[x_axis()]);
        let projector = FlatInputProjector::new(0);
        let x = array![[1.0, 2.0, 3.0]];

        let err = projector.forward(&x, &state).unwrap_err();

        match err {
            ArrangementError::ShapeMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn forward_rejects_invalid_block_index() {
        let state = build_state(&[x_axis()]);
        let projector = FlatInputProjector::new(1);

        let err = projector.forward(&array![[1.0, 2.0]], &state).unwrap_err();

        match err {
            ArrangementError::InvalidParameter { param, .. } => {
                assert_eq!(param, "block_idx");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn forward_preserves_batch_shape() {
        let state = build_state(&[y_axis()]);
        let projector = FlatInputProjector::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]];

        let out = projector.forward(&x, &state).unwrap();

        assert_eq!(out.dim(), x.dim());
    }

    #[test]
    fn zero_input_maps_to_zero() {
        let state = build_state(&[x_axis()]);
        let projector = FlatInputProjector::new(0);
        let out = projector.forward(&array![[0.0, 0.0]], &state).unwrap();

        assert_array2_close(&out, &array![[0.0, 0.0]]);
    }
}
