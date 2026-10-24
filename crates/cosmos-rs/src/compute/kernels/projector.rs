//! Projector-specific kernels shared by COSMOS runtimes.

use ndarray::{Array2, linalg::general_mat_mul};

use crate::compute::Scalar;
use crate::compute::device::{Buffer2, ComputeDevice, Device};
use crate::compute::workspace::ProjectorKernelWorkspace;

/// Compute `X W W^T` into `out`.
pub fn project_rows<S: Scalar>(
    rows: &Array2<S>,
    basis: &Array2<S>,
    out: &mut Array2<S>,
    workspace: &mut ProjectorKernelWorkspace<S>,
) {
    let one = S::one();
    let zero = S::zero();
    workspace.resize(rows.nrows(), rows.ncols(), basis.ncols());
    general_mat_mul(
        one,
        &rows.view(),
        &basis.view(),
        zero,
        &mut workspace.left_coords,
    );
    general_mat_mul(one, &workspace.left_coords.view(), &basis.t(), zero, out);
}

/// Compute `(X^T G + G^T X) W` into `grad_basis`.
///
/// When the forward pass already cached `X W`, pass it as `input_coords` to
/// skip one matmul inside the kernel.
pub fn projector_basis_gradient<S: Scalar>(
    inputs: &Array2<S>,
    input_coords: Option<&Array2<S>>,
    grad_projected: &Array2<S>,
    basis: &Array2<S>,
    grad_basis: &mut Array2<S>,
    workspace: &mut ProjectorKernelWorkspace<S>,
) {
    let one = S::one();
    let zero = S::zero();
    workspace.resize(inputs.nrows(), inputs.ncols(), basis.ncols());
    general_mat_mul(
        one,
        &grad_projected.view(),
        &basis.view(),
        zero,
        &mut workspace.left_coords,
    );
    general_mat_mul(
        one,
        &inputs.t(),
        &workspace.left_coords.view(),
        zero,
        grad_basis,
    );

    if let Some(coords) = input_coords {
        general_mat_mul(one, &grad_projected.t(), &coords.view(), one, grad_basis);
    } else {
        general_mat_mul(
            one,
            &inputs.view(),
            &basis.view(),
            zero,
            &mut workspace.right_coords,
        );
        general_mat_mul(
            one,
            &grad_projected.t(),
            &workspace.right_coords.view(),
            one,
            grad_basis,
        );
    }
}

/// Backward kernel for `X W W^T` with optional post-projection derivative.
pub fn projector_backward_projected<S: Scalar>(
    inputs: &Array2<S>,
    input_coords: Option<&Array2<S>>,
    grad_output: &Array2<S>,
    basis: &Array2<S>,
    projected_derivative: Option<&Array2<S>>,
    grad_input: &mut Array2<S>,
    grad_basis: &mut Array2<S>,
    workspace: &mut ProjectorKernelWorkspace<S>,
) {
    let one = S::one();
    let zero = S::zero();
    workspace.resize(inputs.nrows(), inputs.ncols(), basis.ncols());

    if let Some(derivative) = projected_derivative {
        workspace.row_buffer.assign(grad_output);
        workspace
            .row_buffer
            .zip_mut_with(derivative, |value, deriv| *value = *value * *deriv);
        general_mat_mul(
            one,
            &workspace.row_buffer.view(),
            &basis.view(),
            zero,
            &mut workspace.left_coords,
        );
        general_mat_mul(
            one,
            &workspace.left_coords.view(),
            &basis.t(),
            zero,
            grad_input,
        );
        general_mat_mul(
            one,
            &inputs.t(),
            &workspace.left_coords.view(),
            zero,
            grad_basis,
        );
        if let Some(coords) = input_coords {
            general_mat_mul(
                one,
                &workspace.row_buffer.t(),
                &coords.view(),
                one,
                grad_basis,
            );
        } else {
            general_mat_mul(
                one,
                &inputs.view(),
                &basis.view(),
                zero,
                &mut workspace.right_coords,
            );
            general_mat_mul(
                one,
                &workspace.row_buffer.t(),
                &workspace.right_coords.view(),
                one,
                grad_basis,
            );
        }
    } else {
        project_rows(grad_output, basis, grad_input, workspace);
        projector_basis_gradient(
            inputs,
            input_coords,
            grad_output,
            basis,
            grad_basis,
            workspace,
        );
    }
}

/// Backward kernel for subspace-activated projector applications.
pub fn projector_backward_subspace<S: Scalar>(
    inputs: &Array2<S>,
    grad_output: &Array2<S>,
    basis: &Array2<S>,
    activated_subspace: &Array2<S>,
    subspace_derivative: &Array2<S>,
    grad_input: &mut Array2<S>,
    grad_basis: &mut Array2<S>,
    workspace: &mut ProjectorKernelWorkspace<S>,
) {
    let one = S::one();
    let zero = S::zero();
    workspace.resize(inputs.nrows(), inputs.ncols(), basis.ncols());
    general_mat_mul(
        one,
        &grad_output.view(),
        &basis.view(),
        zero,
        &mut workspace.left_coords,
    );
    workspace.right_coords.assign(&workspace.left_coords);
    workspace
        .right_coords
        .zip_mut_with(subspace_derivative, |value, deriv| *value = *value * *deriv);
    general_mat_mul(
        one,
        &workspace.right_coords.view(),
        &basis.t(),
        zero,
        grad_input,
    );
    general_mat_mul(
        one,
        &grad_output.t(),
        &activated_subspace.view(),
        zero,
        grad_basis,
    );
    general_mat_mul(
        one,
        &inputs.t(),
        &workspace.right_coords.view(),
        one,
        grad_basis,
    );
}

// ---------------------------------------------------------------------------
// Device-dispatched variants (operate on Buffer2 via Device)
// ---------------------------------------------------------------------------

/// Device-dispatched `X W W^T`.
///
/// Equivalent to [`project_rows`] but works on device-resident buffers.
/// Uses `Device::matmul` and `Device::transpose` — no host round-trips.
pub fn project_rows_device<S: Scalar>(
    device: &Device,
    rows: &Buffer2<S>,
    basis: &Buffer2<S>,
) -> Buffer2<S> {
    device.project_rows(rows, basis)
}

/// Device-dispatched `(X^T G + G^T X) W`.
///
/// Equivalent to [`projector_basis_gradient`] but on device-resident buffers.
pub fn projector_basis_gradient_device<S: Scalar>(
    device: &Device,
    inputs: &Buffer2<S>,
    input_coords: Option<&Buffer2<S>>,
    grad_projected: &Buffer2<S>,
    basis: &Buffer2<S>,
    grad_basis: &mut Buffer2<S>,
) {
    let one = S::one();
    let zero = S::zero();

    // left_coords = grad_projected @ basis
    let left_coords = device.matmul(grad_projected, basis);

    // grad_basis = inputs^T @ left_coords
    let inputs_t = device.transpose(inputs);
    device.gemm(one, &inputs_t, &left_coords, zero, grad_basis);

    // grad_basis += grad_projected^T @ (input_coords or inputs @ basis)
    let grad_projected_t = device.transpose(grad_projected);
    if let Some(coords) = input_coords {
        device.gemm(one, &grad_projected_t, coords, one, grad_basis);
    } else {
        let right_coords = device.matmul(inputs, basis);
        device.gemm(one, &grad_projected_t, &right_coords, one, grad_basis);
    }
}

/// Device-dispatched backward for `X W W^T` with optional post-projection derivative.
pub fn projector_backward_projected_device<S: Scalar>(
    device: &Device,
    inputs: &Buffer2<S>,
    input_coords: Option<&Buffer2<S>>,
    grad_output: &Buffer2<S>,
    basis: &Buffer2<S>,
    projected_derivative: Option<&Buffer2<S>>,
    grad_input: &mut Buffer2<S>,
    grad_basis: &mut Buffer2<S>,
) {
    let one = S::one();
    let zero = S::zero();

    if let Some(derivative) = projected_derivative {
        // effective_grad = grad_output * derivative (element-wise)
        let effective_grad = device.elementwise_mul(grad_output, derivative);

        // left_coords = effective_grad @ basis
        let left_coords = device.matmul(&effective_grad, basis);

        // grad_input = left_coords @ basis^T
        let basis_t = device.transpose(basis);
        device.gemm(one, &left_coords, &basis_t, zero, grad_input);

        // grad_basis = inputs^T @ left_coords
        let inputs_t = device.transpose(inputs);
        device.gemm(one, &inputs_t, &left_coords, zero, grad_basis);

        // grad_basis += effective_grad^T @ (input_coords or inputs @ basis)
        let effective_grad_t = device.transpose(&effective_grad);
        if let Some(coords) = input_coords {
            device.gemm(one, &effective_grad_t, coords, one, grad_basis);
        } else {
            let right_coords = device.matmul(inputs, basis);
            device.gemm(one, &effective_grad_t, &right_coords, one, grad_basis);
        }
    } else {
        *grad_input = device.project_rows(grad_output, basis);

        projector_basis_gradient_device(
            device,
            inputs,
            input_coords,
            grad_output,
            basis,
            grad_basis,
        );
    }
}

/// Device-dispatched backward for subspace-activated projector.
pub fn projector_backward_subspace_device<S: Scalar>(
    device: &Device,
    inputs: &Buffer2<S>,
    grad_output: &Buffer2<S>,
    basis: &Buffer2<S>,
    activated_subspace: &Buffer2<S>,
    subspace_derivative: &Buffer2<S>,
    grad_input: &mut Buffer2<S>,
    grad_basis: &mut Buffer2<S>,
) {
    let one = S::one();
    let zero = S::zero();

    // left_coords = grad_output @ basis
    let left_coords = device.matmul(grad_output, basis);

    // right_coords = left_coords * subspace_derivative (element-wise)
    let right_coords = device.elementwise_mul(&left_coords, subspace_derivative);

    // grad_input = right_coords @ basis^T
    let basis_t = device.transpose(basis);
    device.gemm(one, &right_coords, &basis_t, zero, grad_input);

    // grad_basis = grad_output^T @ activated_subspace + inputs^T @ right_coords
    let grad_output_t = device.transpose(grad_output);
    device.gemm(one, &grad_output_t, activated_subspace, zero, grad_basis);
    let inputs_t = device.transpose(inputs);
    device.gemm(one, &inputs_t, &right_coords, one, grad_basis);
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

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
    fn projector_basis_gradient_matches_explicit_reference() {
        let inputs = array![[1.0, 2.0, -1.0], [0.5, -0.25, 1.5]];
        let basis = array![[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]];
        let grad_projected = array![[0.2, -0.1, 0.4], [-0.3, 0.7, 0.5]];
        let mut workspace = ProjectorKernelWorkspace::default();
        let mut grad_basis = Array2::<f64>::zeros(basis.dim());

        projector_basis_gradient(
            &inputs,
            None,
            &grad_projected,
            &basis,
            &mut grad_basis,
            &mut workspace,
        );

        let reference =
            (inputs.t().dot(&grad_projected) + grad_projected.t().dot(&inputs)).dot(&basis);
        assert_array2_close(&grad_basis, &reference);
    }

    #[test]
    fn projector_backward_projected_matches_reference_with_derivative() {
        let inputs = array![[1.0, 2.0], [-1.0, 0.5]];
        let basis = array![[1.0], [1.0]] / 2.0_f64.sqrt();
        let grad_output = array![[0.4, -0.2], [0.1, 0.3]];
        let derivative = array![[1.0, 0.5], [0.25, 2.0]];
        let local_grad = &grad_output * &derivative;
        let mut workspace = ProjectorKernelWorkspace::default();
        let mut grad_input = Array2::<f64>::zeros(inputs.dim());
        let mut grad_basis = Array2::<f64>::zeros(basis.dim());

        projector_backward_projected(
            &inputs,
            None,
            &grad_output,
            &basis,
            Some(&derivative),
            &mut grad_input,
            &mut grad_basis,
            &mut workspace,
        );

        let reference_input = local_grad.dot(&basis).dot(&basis.t());
        let reference_basis =
            (inputs.t().dot(&local_grad) + local_grad.t().dot(&inputs)).dot(&basis);
        assert_array2_close(&grad_input, &reference_input);
        assert_array2_close(&grad_basis, &reference_basis);
    }

    #[test]
    fn projector_backward_subspace_matches_reference() {
        let inputs = array![[1.0, -2.0], [0.25, 0.75]];
        let basis = array![[1.0], [1.0]] / 2.0_f64.sqrt();
        let grad_output = array![[0.3, -0.4], [0.2, 0.1]];
        let activated_subspace = array![[1.5], [0.0]];
        let subspace_derivative = array![[0.5], [1.0]];
        let grad_subspace = grad_output.dot(&basis) * &subspace_derivative;
        let mut workspace = ProjectorKernelWorkspace::default();
        let mut grad_input = Array2::<f64>::zeros(inputs.dim());
        let mut grad_basis = Array2::<f64>::zeros(basis.dim());

        projector_backward_subspace(
            &inputs,
            &grad_output,
            &basis,
            &activated_subspace,
            &subspace_derivative,
            &mut grad_input,
            &mut grad_basis,
            &mut workspace,
        );

        let reference_input = grad_subspace.dot(&basis.t());
        let reference_basis =
            grad_output.t().dot(&activated_subspace) + inputs.t().dot(&grad_subspace);
        assert_array2_close(&grad_input, &reference_input);
        assert_array2_close(&grad_basis, &reference_basis);
    }

    #[test]
    fn device_projector_basis_gradient_matches_cpu() {
        let dev = Device::default();
        let inputs = array![[1.0, 2.0, -1.0], [0.5, -0.25, 1.5]];
        let basis = array![[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]];
        let grad_projected = array![[0.2, -0.1, 0.4], [-0.3, 0.7, 0.5]];

        let inputs_buf = dev.upload_2d(&inputs);
        let basis_buf = dev.upload_2d(&basis);
        let grad_projected_buf = dev.upload_2d(&grad_projected);
        let mut grad_basis_buf: Buffer2<f64> = dev.zeros_2d(3, 2);

        projector_basis_gradient_device(
            &dev,
            &inputs_buf,
            None,
            &grad_projected_buf,
            &basis_buf,
            &mut grad_basis_buf,
        );

        let reference =
            (inputs.t().dot(&grad_projected) + grad_projected.t().dot(&inputs)).dot(&basis);
        let result = dev.download_2d(&grad_basis_buf);
        assert_array2_close(&result, &reference);
    }

    #[test]
    fn device_projector_backward_subspace_matches_cpu() {
        let dev = Device::default();
        let inputs = array![[1.0, -2.0], [0.25, 0.75]];
        let basis = array![[1.0], [1.0]] / 2.0_f64.sqrt();
        let grad_output = array![[0.3, -0.4], [0.2, 0.1]];
        let activated_subspace = array![[1.5], [0.0]];
        let subspace_derivative = array![[0.5], [1.0]];

        let inputs_buf = dev.upload_2d(&inputs);
        let basis_buf = dev.upload_2d(&basis);
        let grad_output_buf = dev.upload_2d(&grad_output);
        let activated_buf = dev.upload_2d(&activated_subspace);
        let derivative_buf = dev.upload_2d(&subspace_derivative);
        let mut grad_input_buf: Buffer2<f64> = dev.zeros_2d(2, 2);
        let mut grad_basis_buf: Buffer2<f64> = dev.zeros_2d(2, 1);

        projector_backward_subspace_device(
            &dev,
            &inputs_buf,
            &grad_output_buf,
            &basis_buf,
            &activated_buf,
            &derivative_buf,
            &mut grad_input_buf,
            &mut grad_basis_buf,
        );

        // Reference from CPU kernel
        let mut workspace = ProjectorKernelWorkspace::default();
        let mut ref_grad_input = Array2::<f64>::zeros(inputs.dim());
        let mut ref_grad_basis = Array2::<f64>::zeros(basis.dim());
        projector_backward_subspace(
            &inputs,
            &grad_output,
            &basis,
            &activated_subspace,
            &subspace_derivative,
            &mut ref_grad_input,
            &mut ref_grad_basis,
            &mut workspace,
        );

        assert_array2_close(&dev.download_2d(&grad_input_buf), &ref_grad_input);
        assert_array2_close(&dev.download_2d(&grad_basis_buf), &ref_grad_basis);
    }
}
