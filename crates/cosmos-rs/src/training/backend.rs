//! Numerical helpers used by training-time task adapters.
//!
//! Adapters never touch the device directly: they call a [`TrainingBackend`],
//! which owns the `Array2` ↔ `Buffer2` boundary and routes the actual work
//! through a [`Device`].

use ndarray::{Array1, Array2};

use crate::compute::{ComputeDevice, Device, Scalar};

/// Numerical kernels consumed by task adapters and manual backward passes.
///
/// All methods stay generic over [`Scalar`]; `f64` is the only implementation
/// this crate ships, and the geometry layer is `f64` throughout.
pub trait TrainingBackend {
    /// The underlying compute device.
    fn device(&self) -> &Device;

    /// Allocate a zero matrix.
    fn zeros<S: Scalar>(&mut self, rows: usize, cols: usize) -> Array2<S>;

    /// Allocate a zero matrix matching the provided template.
    fn zeros_like<S: Scalar>(&mut self, template: &Array2<S>) -> Array2<S> {
        self.zeros(template.nrows(), template.ncols())
    }

    /// Allocate a zero vector.
    fn zeros_1d<S: Scalar>(&mut self, len: usize) -> Array1<S>;

    /// Matrix product `lhs @ rhs`.
    fn matmul<S: Scalar>(&mut self, lhs: &Array2<S>, rhs: &Array2<S>) -> Array2<S>;

    /// Materialize an owned transpose.
    fn transpose_owned<S: Scalar>(&mut self, matrix: &Array2<S>) -> Array2<S>;

    /// Project `x` through the subspace basis `w` via `x W W^T`.
    fn project<S: Scalar>(&mut self, x: &Array2<S>, w: &Array2<S>) -> Array2<S> {
        let coords = self.matmul(x, w);
        let w_t = self.transpose_owned(w);
        self.matmul(&coords, &w_t)
    }

    /// Per-row quadratic energies `||x W||^2`.
    fn quadratic_energies<S: Scalar>(&mut self, x: &Array2<S>, w: &Array2<S>) -> Array1<S> {
        let coords = self.matmul(x, w);
        Array1::from_iter(
            coords
                .outer_iter()
                .map(|row| row.iter().copied().map(|v| v * v).sum::<S>()),
        )
    }

    /// Mean value over a vector. Empty input returns zero.
    fn mean<S: Scalar>(&mut self, values: &Array1<S>) -> S {
        if values.is_empty() {
            S::zero()
        } else {
            values.iter().copied().sum::<S>() / S::from_f64(values.len() as f64)
        }
    }
}

/// Training backend that delegates to a [`Device`] for computation.
///
/// Wraps `Array2` ↔ `Buffer2` at the boundary so that [`TrainingBackend`]
/// callers keep working with plain arrays while the computation is routed
/// through the device abstraction.
#[derive(Debug, Clone)]
pub struct DeviceBackend {
    device: Device,
}

impl DeviceBackend {
    /// Create a backend that dispatches through the given device.
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// The underlying compute device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Default for DeviceBackend {
    fn default() -> Self {
        Self::new(Device::default())
    }
}

/// Convenience alias for the CPU backend.
pub type CpuBackend = DeviceBackend;

impl TrainingBackend for DeviceBackend {
    fn device(&self) -> &Device {
        &self.device
    }

    fn zeros<S: Scalar>(&mut self, rows: usize, cols: usize) -> Array2<S> {
        self.device.zeros_2d(rows, cols).into_array()
    }

    fn zeros_1d<S: Scalar>(&mut self, len: usize) -> Array1<S> {
        self.device.zeros_1d(len).into_array()
    }

    fn matmul<S: Scalar>(&mut self, lhs: &Array2<S>, rhs: &Array2<S>) -> Array2<S> {
        let a = self.device.upload_2d(lhs);
        let b = self.device.upload_2d(rhs);
        self.device.download_2d(&self.device.matmul(&a, &b))
    }

    fn transpose_owned<S: Scalar>(&mut self, matrix: &Array2<S>) -> Array2<S> {
        let buf = self.device.upload_2d(matrix);
        self.device.download_2d(&self.device.transpose(&buf))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const EPSILON: f64 = 1e-10;

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

    #[test]
    fn project_matches_manual_axis_projection() {
        let mut backend = DeviceBackend::default();
        let x = array![[1.0, 2.0], [-3.0, 4.0]];
        let w = array![[1.0], [0.0]];

        let projected: Array2<f64> = backend.project(&x, &w);

        assert_array2_close(&projected, &array![[1.0, 0.0], [-3.0, 0.0]]);
    }

    #[test]
    fn quadratic_energies_match_expected_values() {
        let mut backend = DeviceBackend::default();
        let x = array![[3.0, 4.0], [0.0, 2.0]];
        let w = array![[1.0], [0.0]];

        let energies: Array1<f64> = backend.quadratic_energies(&x, &w);

        assert_eq!(energies, array![9.0, 0.0]);
    }

    #[test]
    fn matmul_matches_ndarray_dot() {
        let mut backend = DeviceBackend::new(Device::default());
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result: Array2<f64> = backend.matmul(&a, &b);

        assert_eq!(result, a.dot(&b));
    }

    #[test]
    fn transpose_owned_matches_ndarray_transpose() {
        let mut backend = DeviceBackend::default();
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let result: Array2<f64> = backend.transpose_owned(&a);

        assert_eq!(result, a.t().to_owned());
    }

    #[test]
    fn cpu_backend_alias_resolves_to_device_backend() {
        let mut backend = CpuBackend::default();
        let zeros: Array2<f64> = backend.zeros(2, 3);

        assert_eq!(zeros.dim(), (2, 3));
        assert!(zeros.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn mean_of_empty_vector_is_zero() {
        let mut backend = DeviceBackend::default();
        let empty: Array1<f64> = backend.zeros_1d(0);

        assert_eq!(backend.mean(&empty), 0.0);
    }
}
