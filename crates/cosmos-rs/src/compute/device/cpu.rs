//! CPU implementation of [`ComputeDevice`] backed by `ndarray`.

use ndarray::{Array1, Array2, linalg::general_mat_mul};

use super::{Buffer1, Buffer2, ComputeDevice, DeviceKind};
use crate::compute::Scalar;

/// CPU compute device that delegates all operations to `ndarray`.
///
/// On CPU, [`Buffer2`] / [`Buffer1`] are thin wrappers around owned arrays,
/// so upload/download are just clones and allocation is `Array::zeros`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuDevice;

impl ComputeDevice for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    // -- allocation --------------------------------------------------------

    fn zeros_2d<S: Scalar>(&self, rows: usize, cols: usize) -> Buffer2<S> {
        Buffer2::from_array(Array2::zeros((rows, cols)))
    }

    fn zeros_1d<S: Scalar>(&self, len: usize) -> Buffer1<S> {
        Buffer1::from_array(Array1::zeros(len))
    }

    // -- transfer ----------------------------------------------------------

    fn upload_2d<S: Scalar>(&self, data: &Array2<S>) -> Buffer2<S> {
        Buffer2::from_array(data.clone())
    }

    fn download_2d<S: Scalar>(&self, buf: &Buffer2<S>) -> Array2<S> {
        buf.as_array().clone()
    }

    fn upload_1d<S: Scalar>(&self, data: &Array1<S>) -> Buffer1<S> {
        Buffer1::from_array(data.clone())
    }

    fn download_1d<S: Scalar>(&self, buf: &Buffer1<S>) -> Array1<S> {
        buf.as_array().clone()
    }

    // -- BLAS-like ---------------------------------------------------------

    fn gemm<S: Scalar>(
        &self,
        alpha: S,
        a: &Buffer2<S>,
        b: &Buffer2<S>,
        beta: S,
        c: &mut Buffer2<S>,
    ) {
        general_mat_mul(
            alpha,
            &a.as_array().view(),
            &b.as_array().view(),
            beta,
            c.as_array_mut(),
        );
    }

    fn transpose<S: Scalar>(&self, a: &Buffer2<S>) -> Buffer2<S> {
        Buffer2::from_array(a.as_array().t().to_owned())
    }

    // -- reductions --------------------------------------------------------

    fn row_norms_sq<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S> {
        let arr = x.as_array();
        Buffer1::from_array(Array1::from_iter(
            arr.outer_iter()
                .map(|row| row.iter().copied().map(|v| v * v).sum::<S>()),
        ))
    }

    fn sum_1d<S: Scalar>(&self, x: &Buffer1<S>) -> S {
        x.as_array().iter().copied().sum::<S>()
    }

    // -- element-wise operations -----------------------------------------------

    fn row_norms<S: Scalar>(&self, x: &Buffer2<S>) -> Buffer1<S> {
        let arr = x.as_array();
        Buffer1::from_array(Array1::from_iter(
            arr.outer_iter()
                .map(|row| row.iter().copied().map(|v| v * v).sum::<S>().sqrt()),
        ))
    }

    fn elementwise_mul_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>) {
        a.as_array_mut()
            .zip_mut_with(b.as_array(), |x, &y| *x = *x * y);
    }

    fn add_assign<S: Scalar>(&self, a: &mut Buffer2<S>, b: &Buffer2<S>) {
        a.as_array_mut()
            .zip_mut_with(b.as_array(), |x, &y| *x = *x + y);
    }

    fn add_scaled<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S, b: &Buffer2<S>) {
        a.as_array_mut()
            .zip_mut_with(b.as_array(), |x, &y| *x = *x + alpha * y);
    }

    fn scale_inplace<S: Scalar>(&self, a: &mut Buffer2<S>, alpha: S) {
        a.as_array_mut().mapv_inplace(|v| v * alpha);
    }

    fn clone_buffer<S: Scalar>(&self, src: &Buffer2<S>) -> Buffer2<S> {
        Buffer2::from_array(src.as_array().clone())
    }

    fn broadcast_col_mul<S: Scalar>(&self, mat: &Buffer2<S>, col: &Buffer1<S>) -> Buffer2<S> {
        let arr = mat.as_array();
        let col_arr = col.as_array();
        let mut out = arr.clone();
        for (mut row, &scale) in out.outer_iter_mut().zip(col_arr.iter()) {
            row.mapv_inplace(|v| v * scale);
        }
        Buffer2::from_array(out)
    }

    fn sub_broadcast_col_mul<S: Scalar>(
        &self,
        a: &mut Buffer2<S>,
        b: &Buffer2<S>,
        col: &Buffer1<S>,
    ) {
        let b_arr = b.as_array();
        let col_arr = col.as_array();
        let a_arr = a.as_array_mut();
        for ((mut a_row, b_row), &scale) in a_arr
            .outer_iter_mut()
            .zip(b_arr.outer_iter())
            .zip(col_arr.iter())
        {
            a_row.zip_mut_with(&b_row, |a_val, &b_val| *a_val = *a_val - b_val * scale);
        }
    }

    fn assign_cols<S: Scalar>(&self, dst: &mut Buffer2<S>, start: usize, src: &Buffer2<S>) {
        use ndarray::s;
        let ncols = src.ncols();
        dst.as_array_mut()
            .slice_mut(s![.., start..start + ncols])
            .assign(src.as_array());
    }

    fn slice_cols<S: Scalar>(&self, src: &Buffer2<S>, start: usize, ncols: usize) -> Buffer2<S> {
        use ndarray::s;
        Buffer2::from_array(
            src.as_array()
                .slice(s![.., start..start + ncols])
                .to_owned(),
        )
    }

    fn reciprocal_1d<S: Scalar>(&self, x: &Buffer1<S>) -> Buffer1<S> {
        Buffer1::from_array(x.as_array().mapv(|v| {
            if v == S::zero() {
                S::zero()
            } else {
                S::one() / v
            }
        }))
    }

    fn apply_activation<S: Scalar>(
        &self,
        buf: &mut Buffer2<S>,
        activation: crate::compute::activation::Activation,
    ) {
        buf.as_array_mut()
            .mapv_inplace(|x| S::from_f64(activation.apply(x.to_f64())));
    }

    fn activation_derivative<S: Scalar>(
        &self,
        input: &Buffer2<S>,
        activation: crate::compute::activation::Activation,
    ) -> Buffer2<S> {
        Buffer2::from_array(
            input
                .as_array()
                .mapv(|x| S::from_f64(activation.derivative(x.to_f64()))),
        )
    }

    fn compute_geodesic_scores<S: Scalar>(
        &self,
        embeddings: &Buffer2<S>,
        anchors: &[&Buffer2<S>],
        eps: S,
    ) -> (Buffer2<S>, Buffer2<S>) {
        let n = embeddings.dim().0;
        let num_anchors = anchors.len();
        let emb = embeddings.as_array();

        // Embedding norms (N,) with eps floor
        let embed_norms: Array1<S> = Array1::from_iter(emb.outer_iter().map(|row| {
            let norm_sq: S = row.iter().map(|&v| v * v).fold(S::zero(), |a, b| a + b);
            let norm = S::from_f64(norm_sq.to_f64().sqrt());
            if norm < eps { eps } else { norm }
        }));

        let mut scores = Array2::<S>::zeros((n, num_anchors));
        let mut angles = Array2::<S>::zeros((n, num_anchors));

        for (j, anchor) in anchors.iter().enumerate() {
            let coords = emb.dot(anchor.as_array()); // (N, k_j)
            for i in 0..n {
                let proj_norm_sq: S = coords
                    .row(i)
                    .iter()
                    .map(|&v| v * v)
                    .fold(S::zero(), |a, b| a + b);
                let proj_norm = S::from_f64(proj_norm_sq.to_f64().sqrt());
                let cosine = (proj_norm.to_f64() / embed_norms[i].to_f64()).clamp(0.0, 1.0);
                let angle = cosine.acos();
                angles[[i, j]] = S::from_f64(angle);
                scores[[i, j]] = S::from_f64((-angle * angle).exp());
            }
        }

        (Buffer2::from_array(scores), Buffer2::from_array(angles))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn cpu_device_zeros_2d_shape() {
        let dev = CpuDevice;
        let buf: Buffer2<f64> = dev.zeros_2d(3, 4);
        assert_eq!(buf.dim(), (3, 4));
        assert!(buf.as_array().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cpu_device_upload_download_roundtrip() {
        let dev = CpuDevice;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let buf = dev.upload_2d(&data);
        let back = dev.download_2d(&buf);
        assert_eq!(data, back);
    }

    #[test]
    fn cpu_device_matmul_matches_ndarray() {
        let dev = CpuDevice;
        let a = Buffer2::from_array(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = Buffer2::from_array(array![[5.0, 6.0], [7.0, 8.0]]);
        let c = dev.matmul(&a, &b);
        assert_eq!(*c.as_array(), array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn cpu_device_gemm_accumulates() {
        let dev = CpuDevice;
        let a = Buffer2::from_array(array![[1.0, 0.0], [0.0, 1.0]]);
        let b = Buffer2::from_array(array![[2.0, 3.0], [4.0, 5.0]]);
        let mut c = Buffer2::from_array(array![[10.0, 10.0], [10.0, 10.0]]);
        dev.gemm(1.0, &a, &b, 1.0, &mut c);
        assert_eq!(*c.as_array(), array![[12.0, 13.0], [14.0, 15.0]]);
    }

    #[test]
    fn cpu_device_transpose() {
        let dev = CpuDevice;
        let a = Buffer2::from_array(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = dev.transpose(&a);
        assert_eq!(t.dim(), (3, 2));
        assert_eq!(*t.as_array(), array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn cpu_device_project_rows() {
        let dev = CpuDevice;
        let x = Buffer2::from_array(array![[1.0, 2.0], [-3.0, 4.0]]);
        let w = Buffer2::from_array(array![[1.0], [0.0]]);
        let projected = dev.project_rows(&x, &w);
        assert_eq!(*projected.as_array(), array![[1.0, 0.0], [-3.0, 0.0]]);
    }

    #[test]
    fn cpu_device_row_norms_sq() {
        let dev = CpuDevice;
        let x = Buffer2::from_array(array![[3.0, 4.0], [0.0, 2.0]]);
        let norms = dev.row_norms_sq(&x);
        assert_eq!(*norms.as_array(), array![25.0, 4.0]);
    }

    #[test]
    fn cpu_device_sum_and_mean() {
        let dev = CpuDevice;
        let v = Buffer1::from_array(array![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dev.sum_1d(&v), 10.0);
        assert_eq!(dev.mean_1d(&v), 2.5);
    }

    #[test]
    fn cpu_device_mean_empty() {
        let dev = CpuDevice;
        let v: Buffer1<f64> = dev.zeros_1d(0);
        assert_eq!(dev.mean_1d(&v), 0.0);
    }

    #[test]
    fn cpu_device_buffer_kind() {
        let dev = CpuDevice;
        let buf: Buffer2<f64> = dev.zeros_2d(2, 2);
        assert_eq!(buf.device_kind(), DeviceKind::Cpu);
        assert_eq!(dev.kind(), DeviceKind::Cpu);
    }

    #[test]
    fn cpu_device_row_norms() {
        let dev = CpuDevice;
        let x = Buffer2::from_array(array![[3.0_f64, 4.0], [0.0, 2.0]]);
        let norms = dev.row_norms(&x);
        assert!((norms.as_array()[0] - 5.0).abs() < 1e-10);
        assert!((norms.as_array()[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_device_elementwise_mul_assign() {
        let dev = CpuDevice;
        let mut a = Buffer2::from_array(array![[1.0_f64, 2.0], [3.0, 4.0]]);
        let b = Buffer2::from_array(array![[5.0, 6.0], [7.0, 8.0]]);
        dev.elementwise_mul_assign(&mut a, &b);
        assert_eq!(*a.as_array(), array![[5.0, 12.0], [21.0, 32.0]]);
    }

    #[test]
    fn cpu_device_add_scaled() {
        let dev = CpuDevice;
        let mut a = Buffer2::from_array(array![[1.0_f64, 2.0], [3.0, 4.0]]);
        let b = Buffer2::from_array(array![[10.0, 20.0], [30.0, 40.0]]);
        dev.add_scaled(&mut a, 0.5, &b);
        assert_eq!(*a.as_array(), array![[6.0, 12.0], [18.0, 24.0]]);
    }

    #[test]
    fn cpu_device_broadcast_col_mul() {
        let dev = CpuDevice;
        let mat = Buffer2::from_array(array![[1.0_f64, 2.0], [3.0, 4.0]]);
        let col = Buffer1::from_array(array![10.0, 0.5]);
        let result = dev.broadcast_col_mul(&mat, &col);
        assert_eq!(*result.as_array(), array![[10.0, 20.0], [1.5, 2.0]]);
    }

    #[test]
    fn cpu_device_slice_and_assign_cols() {
        let dev = CpuDevice;
        let src = Buffer2::from_array(array![[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let sliced = dev.slice_cols(&src, 1, 2);
        assert_eq!(*sliced.as_array(), array![[2.0, 3.0], [6.0, 7.0]]);

        let mut dst: Buffer2<f64> = dev.zeros_2d(2, 4);
        dev.assign_cols(&mut dst, 2, &sliced);
        assert_eq!(
            *dst.as_array(),
            array![[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0]]
        );
    }
}
