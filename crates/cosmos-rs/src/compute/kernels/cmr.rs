//! CMR-specific backward kernels.

use ndarray::{Array1, Array2, ArrayView1, Axis, linalg::general_mat_mul};

use ndarray::s;

use crate::compute::Scalar;
use crate::compute::device::{Buffer1, Buffer2, ComputeDevice, Device};
use crate::compute::workspace::{CmrAnchorBackwardWorkspace, CmrMultiAnchorBackwardWorkspace};

fn score_derivative_from_cosine<S: Scalar>(cosine: S, eps: S) -> S {
    let one = S::one();
    let two = one + one;
    let limit = two;

    if cosine >= one - eps {
        return limit;
    }

    let theta = cosine.acos();
    let score = (-(theta * theta)).exp();
    let denom = (one - cosine * cosine).sqrt().max(eps);
    two * theta * score / denom
}

/// Batched backward kernel for one CMR anchor block.
///
/// Accumulates gradient contributions into `grad_embeddings` and overwrites
/// `grad_anchor` with the gradient for the provided `anchor`.
pub fn cmr_anchor_backward<S: Scalar>(
    embeddings: &Array2<S>,
    embedding_norms: &Array1<S>,
    anchor: &Array2<S>,
    score_grads: ArrayView1<'_, S>,
    eps: S,
    grad_embeddings: &mut Array2<S>,
    grad_anchor: &mut Array2<S>,
    workspace: &mut CmrAnchorBackwardWorkspace<S>,
) {
    let batch_size = embeddings.nrows();
    let embedding_dim = embeddings.ncols();
    let anchor_dim = anchor.ncols();
    let zero = S::zero();
    let one = S::one();

    workspace.resize(batch_size, embedding_dim, anchor_dim);
    workspace.projected_norms.fill(zero);
    workspace.denom_scale.fill(zero);
    workspace.correction.fill(zero);
    workspace.scaled_coords.fill(zero);
    grad_anchor.fill(zero);

    general_mat_mul(
        one,
        &embeddings.view(),
        &anchor.view(),
        zero,
        &mut workspace.coords,
    );
    general_mat_mul(
        one,
        &workspace.coords.view(),
        &anchor.t(),
        zero,
        &mut workspace.projected,
    );

    for sample_idx in 0..batch_size {
        let projected_norm = workspace
            .coords
            .row(sample_idx)
            .dot(&workspace.coords.row(sample_idx))
            .sqrt();
        workspace.projected_norms[sample_idx] = projected_norm;

        let grad_scale = score_grads[sample_idx];
        if grad_scale == zero {
            continue;
        }

        let embedding_norm = embedding_norms[sample_idx];
        if embedding_norm <= eps {
            continue;
        }

        let cosine = (projected_norm / embedding_norm).min(one).max(zero);
        let dscore_dc = score_derivative_from_cosine(cosine, eps);
        if dscore_dc == zero || projected_norm <= eps {
            continue;
        }

        let scale = grad_scale * dscore_dc;
        workspace.denom_scale[sample_idx] = scale / (projected_norm * embedding_norm);
        workspace.correction[sample_idx] =
            scale * projected_norm / (embedding_norm * embedding_norm * embedding_norm);
    }

    let denom_scale_col = workspace.denom_scale.view().insert_axis(Axis(1));
    let correction_col = workspace.correction.view().insert_axis(Axis(1));
    grad_embeddings.zip_mut_with(&(&workspace.projected * &denom_scale_col), |g, v| {
        *g = *g + *v
    });
    grad_embeddings.zip_mut_with(&(embeddings * &correction_col), |g, v| *g = *g - *v);

    workspace
        .scaled_coords
        .assign(&(&workspace.coords * &denom_scale_col));
    general_mat_mul(
        one,
        &embeddings.t(),
        &workspace.scaled_coords.view(),
        zero,
        grad_anchor,
    );
}

/// Fused backward kernel for multiple CMR anchors with identical shapes.
pub fn cmr_multi_anchor_backward<S: Scalar>(
    embeddings: &Array2<S>,
    embedding_norms: &Array1<S>,
    anchors: &[&Array2<S>],
    score_grads: &Array2<S>,
    eps: S,
    grad_embeddings: &mut Array2<S>,
    grad_anchors: &mut [Array2<S>],
    workspace: &mut CmrMultiAnchorBackwardWorkspace<S>,
) {
    let num_anchors = anchors.len();
    if num_anchors == 0 {
        return;
    }

    let batch_size = embeddings.nrows();
    let embedding_dim = embeddings.ncols();
    let anchor_dim = anchors[0].ncols();
    let zero = S::zero();
    let one = S::one();
    debug_assert_eq!(score_grads.dim(), (batch_size, num_anchors));
    debug_assert_eq!(grad_anchors.len(), num_anchors);

    workspace.resize(batch_size, embedding_dim, anchor_dim, num_anchors);
    workspace.scaled_coords.fill(zero);
    workspace.correction.fill(zero);

    for (anchor_idx, anchor) in anchors.iter().enumerate() {
        debug_assert_eq!(anchor.dim(), (embedding_dim, anchor_dim));
        let start = anchor_idx * anchor_dim;
        let end = start + anchor_dim;
        workspace
            .stacked_anchors
            .slice_mut(s![.., start..end])
            .assign(anchor);
        workspace
            .stacked_anchor_t
            .slice_mut(s![start..end, ..])
            .assign(&anchor.t());
    }

    general_mat_mul(
        one,
        &embeddings.view(),
        &workspace.stacked_anchors.view(),
        zero,
        &mut workspace.coords,
    );

    for anchor_idx in 0..num_anchors {
        let start = anchor_idx * anchor_dim;
        let score_grad_col = score_grads.column(anchor_idx);

        for sample_idx in 0..batch_size {
            let grad_scale = score_grad_col[sample_idx];
            if grad_scale == zero {
                continue;
            }

            let embedding_norm = embedding_norms[sample_idx];
            if embedding_norm <= eps {
                continue;
            }

            let mut projected_norm_sq = zero;
            for col_idx in 0..anchor_dim {
                let value = workspace.coords[[sample_idx, start + col_idx]];
                projected_norm_sq = projected_norm_sq + value * value;
            }
            let projected_norm = projected_norm_sq.sqrt();
            if projected_norm <= eps {
                continue;
            }

            let cosine = (projected_norm / embedding_norm).min(one).max(zero);
            let dscore_dc = score_derivative_from_cosine(cosine, eps);
            if dscore_dc == zero {
                continue;
            }

            let scale = grad_scale * dscore_dc;
            let denom_scale = scale / (projected_norm * embedding_norm);
            workspace.correction[sample_idx] = workspace.correction[sample_idx]
                + scale * projected_norm / (embedding_norm * embedding_norm * embedding_norm);

            for col_idx in 0..anchor_dim {
                workspace.scaled_coords[[sample_idx, start + col_idx]] =
                    workspace.coords[[sample_idx, start + col_idx]] * denom_scale;
            }
        }
    }

    general_mat_mul(
        one,
        &workspace.scaled_coords.view(),
        &workspace.stacked_anchor_t.view(),
        zero,
        &mut workspace.projected,
    );
    let correction_col = workspace.correction.view().insert_axis(Axis(1));
    grad_embeddings.zip_mut_with(&workspace.projected, |g, v| *g = *g + *v);
    grad_embeddings.zip_mut_with(&(embeddings * &correction_col), |g, v| *g = *g - *v);

    general_mat_mul(
        one,
        &embeddings.t(),
        &workspace.scaled_coords.view(),
        zero,
        &mut workspace.stacked_anchors,
    );
    for (anchor_idx, grad_anchor) in grad_anchors.iter_mut().enumerate() {
        let start = anchor_idx * anchor_dim;
        let end = start + anchor_dim;
        grad_anchor.assign(&workspace.stacked_anchors.slice(s![.., start..end]));
    }
}

// ---------------------------------------------------------------------------
// Device-dispatched variant
// ---------------------------------------------------------------------------

/// Device-dispatched fused backward for multiple CMR anchors.
///
/// The big matmuls (coords, projected, grad_anchors) run on-device via
/// `Device::gemm`.  The per-sample scoring derivative loop downloads the
/// small `coords` matrix (batch × total_anchor_dim) to CPU, computes
/// `scaled_coords` and `correction`, then uploads for the remaining matmuls.
pub fn cmr_multi_anchor_backward_device<S: Scalar>(
    device: &Device,
    embeddings: &Buffer2<S>,
    embedding_norms: &Buffer1<S>,
    anchors: &[&Buffer2<S>],
    score_grads: &Buffer2<S>,
    eps: S,
    grad_embeddings: &mut Buffer2<S>,
    grad_anchors: &mut [Buffer2<S>],
) {
    let num_anchors = anchors.len();
    if num_anchors == 0 {
        return;
    }

    let batch_size = embeddings.nrows();
    let embedding_dim = embeddings.ncols();
    let anchor_dim = anchors[0].ncols();
    let total_anchor_dim = anchor_dim * num_anchors;
    let zero = S::zero();
    let one = S::one();

    // Stack anchors into a single (embedding_dim × total_anchor_dim) buffer
    let mut stacked_anchors: Buffer2<S> = device.zeros_2d(embedding_dim, total_anchor_dim);
    for (i, anchor) in anchors.iter().enumerate() {
        device.assign_cols(&mut stacked_anchors, i * anchor_dim, anchor);
    }

    // coords = embeddings @ stacked_anchors  (batch × total_anchor_dim) — on device
    let coords = device.matmul(embeddings, &stacked_anchors);

    // Download coords, embedding_norms, score_grads for per-sample scoring loop
    let coords_cpu = device.download_2d(&coords);
    let norms_cpu = device.download_1d(embedding_norms);
    let score_grads_cpu = device.download_2d(score_grads);

    let mut scaled_coords_cpu = Array2::<S>::zeros((batch_size, total_anchor_dim));
    let mut correction_cpu = Array1::<S>::zeros(batch_size);

    for anchor_idx in 0..num_anchors {
        let start = anchor_idx * anchor_dim;

        for sample_idx in 0..batch_size {
            let grad_scale = score_grads_cpu[[sample_idx, anchor_idx]];
            if grad_scale == zero {
                continue;
            }

            let embedding_norm = norms_cpu[sample_idx];
            if embedding_norm <= eps {
                continue;
            }

            let mut projected_norm_sq = zero;
            for col_idx in 0..anchor_dim {
                let v = coords_cpu[[sample_idx, start + col_idx]];
                projected_norm_sq = projected_norm_sq + v * v;
            }
            let projected_norm = projected_norm_sq.sqrt();
            if projected_norm <= eps {
                continue;
            }

            let cosine = (projected_norm / embedding_norm).min(one).max(zero);
            let dscore_dc = score_derivative_from_cosine(cosine, eps);
            if dscore_dc == zero {
                continue;
            }

            let scale = grad_scale * dscore_dc;
            let denom_scale = scale / (projected_norm * embedding_norm);
            correction_cpu[sample_idx] = correction_cpu[sample_idx]
                + scale * projected_norm / (embedding_norm * embedding_norm * embedding_norm);

            for col_idx in 0..anchor_dim {
                scaled_coords_cpu[[sample_idx, start + col_idx]] =
                    coords_cpu[[sample_idx, start + col_idx]] * denom_scale;
            }
        }
    }

    // Upload scaled_coords and correction back to device
    let scaled_coords = device.upload_2d(&scaled_coords_cpu);
    let correction = device.upload_1d(&correction_cpu);

    // projected = scaled_coords @ stacked_anchors^T  (batch × embedding_dim) — on device
    let stacked_anchor_t = device.transpose(&stacked_anchors);
    let projected = device.matmul(&scaled_coords, &stacked_anchor_t);

    // grad_embeddings += projected - embeddings * correction (broadcast col)
    device.add_assign(grad_embeddings, &projected);
    device.sub_broadcast_col_mul(grad_embeddings, embeddings, &correction);

    // grad_anchor_stacked = embeddings^T @ scaled_coords  (embedding_dim × total_anchor_dim)
    let embeddings_t = device.transpose(embeddings);
    let grad_anchor_stacked = device.matmul(&embeddings_t, &scaled_coords);

    // Slice out per-anchor gradients
    for (i, grad_anchor) in grad_anchors.iter_mut().enumerate() {
        *grad_anchor = device.slice_cols(&grad_anchor_stacked, i * anchor_dim, anchor_dim);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Axis, array};

    use super::*;

    struct ScoreGradientFactors {
        dscore_dc: f64,
        dc_de: Array1<f64>,
        dc_dw: Option<Array2<f64>>,
    }

    fn score_gradient_factors(
        embedding: &Array1<f64>,
        anchor: &Array2<f64>,
        eps: f64,
    ) -> ScoreGradientFactors {
        let embedding_norm_sq = embedding.iter().map(|value| value * value).sum::<f64>();
        let embedding_norm = embedding_norm_sq.sqrt();
        if embedding_norm <= eps {
            return ScoreGradientFactors {
                dscore_dc: 0.0,
                dc_de: Array1::zeros(embedding.len()),
                dc_dw: None,
            };
        }

        let coords = embedding.dot(anchor);
        let projected_norm_sq = coords.iter().map(|value| value * value).sum::<f64>();
        let projected_norm = projected_norm_sq.sqrt();
        let cosine = (projected_norm / embedding_norm).clamp(0.0, 1.0);
        let dscore_dc = score_derivative_from_cosine(cosine, eps);

        if projected_norm <= eps {
            return ScoreGradientFactors {
                dscore_dc,
                dc_de: Array1::zeros(embedding.len()),
                dc_dw: None,
            };
        }

        let projected = anchor.dot(&coords);
        let denom = projected_norm * embedding_norm;
        let mut dc_de = projected.mapv(|value| value / denom);
        let correction = projected_norm / embedding_norm.powi(3);
        dc_de -= &embedding.mapv(|value| value * correction);

        let outer = embedding
            .view()
            .insert_axis(Axis(1))
            .dot(&coords.view().insert_axis(Axis(0)))
            .mapv(|value| value / denom);

        ScoreGradientFactors {
            dscore_dc,
            dc_de,
            dc_dw: Some(outer),
        }
    }

    #[test]
    fn cmr_anchor_backward_matches_scalar_reference() {
        let embeddings = array![[1.0, 0.5], [0.2, 1.2], [0.7, -0.3]];
        let embedding_norms = Array1::from_iter(
            embeddings
                .outer_iter()
                .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt()),
        );
        let anchor = array![[1.0], [1.0]] / 2.0_f64.sqrt();
        let score_grads = array![0.3, -0.2, 0.5];
        let mut workspace = CmrAnchorBackwardWorkspace::default();
        let mut batched_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
        let mut batched_grad_anchor = Array2::<f64>::zeros(anchor.dim());

        cmr_anchor_backward(
            &embeddings,
            &embedding_norms,
            &anchor,
            score_grads.view(),
            1e-8,
            &mut batched_grad_embeddings,
            &mut batched_grad_anchor,
            &mut workspace,
        );

        let mut scalar_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
        let mut scalar_grad_anchor = Array2::<f64>::zeros(anchor.dim());
        for sample_idx in 0..embeddings.nrows() {
            let embedding = embeddings.row(sample_idx).to_owned();
            let score_grad = score_gradient_factors(&embedding, &anchor, 1e-8);
            let scale = score_grads[sample_idx] * score_grad.dscore_dc;
            scalar_grad_embeddings
                .row_mut(sample_idx)
                .scaled_add(scale, &score_grad.dc_de);
            if let Some(dc_dw) = score_grad.dc_dw {
                scalar_grad_anchor.scaled_add(scale, &dc_dw);
            }
        }

        let emb_diff = (&batched_grad_embeddings - &scalar_grad_embeddings)
            .mapv(|value| value * value)
            .sum()
            .sqrt();
        let anchor_diff = (&batched_grad_anchor - &scalar_grad_anchor)
            .mapv(|value| value * value)
            .sum()
            .sqrt();

        assert!(emb_diff < 1e-10, "embedding grad diff = {emb_diff}");
        assert!(anchor_diff < 1e-10, "anchor grad diff = {anchor_diff}");
    }

    #[test]
    fn cmr_multi_anchor_backward_matches_repeated_single_anchor_reference() {
        let embeddings = array![[1.0, 0.5], [0.2, 1.2], [0.7, -0.3]];
        let embedding_norms = Array1::from_iter(
            embeddings
                .outer_iter()
                .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt()),
        );
        let anchor_a = array![[1.0], [1.0]] / 2.0_f64.sqrt();
        let anchor_b = array![[1.0], [-1.0]] / 2.0_f64.sqrt();
        let anchors = vec![&anchor_a, &anchor_b];
        let score_grads = array![[0.3, -0.1], [-0.2, 0.4], [0.5, 0.2]];

        let mut fused_workspace = CmrMultiAnchorBackwardWorkspace::default();
        let mut fused_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
        let mut fused_grad_anchors = vec![
            Array2::<f64>::zeros(anchor_a.dim()),
            Array2::<f64>::zeros(anchor_b.dim()),
        ];
        cmr_multi_anchor_backward(
            &embeddings,
            &embedding_norms,
            &anchors,
            &score_grads,
            1e-8,
            &mut fused_grad_embeddings,
            &mut fused_grad_anchors,
            &mut fused_workspace,
        );

        let mut repeated_workspace = CmrAnchorBackwardWorkspace::default();
        let mut repeated_grad_embeddings = Array2::<f64>::zeros(embeddings.dim());
        let mut repeated_grad_anchors = [
            Array2::<f64>::zeros(anchor_a.dim()),
            Array2::<f64>::zeros(anchor_b.dim()),
        ];
        for (anchor_idx, anchor) in anchors.iter().enumerate() {
            cmr_anchor_backward(
                &embeddings,
                &embedding_norms,
                anchor,
                score_grads.column(anchor_idx),
                1e-8,
                &mut repeated_grad_embeddings,
                &mut repeated_grad_anchors[anchor_idx],
                &mut repeated_workspace,
            );
        }

        let emb_diff = (&fused_grad_embeddings - &repeated_grad_embeddings)
            .mapv(|value| value * value)
            .sum()
            .sqrt();
        assert!(emb_diff < 1e-10, "embedding grad diff = {emb_diff}");

        for (fused, repeated) in fused_grad_anchors.iter().zip(repeated_grad_anchors.iter()) {
            let anchor_diff = (fused - repeated).mapv(|value| value * value).sum().sqrt();
            assert!(anchor_diff < 1e-10, "anchor grad diff = {anchor_diff}");
        }
    }

    #[test]
    fn device_cmr_multi_anchor_backward_matches_cpu() {
        use crate::compute::device::{Buffer2, ComputeDevice, Device};

        let dev = Device::default();
        let embeddings = array![[1.0_f64, 0.5], [0.2, 1.2], [0.7, -0.3]];
        let embedding_norms = Array1::from_iter(
            embeddings
                .outer_iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt()),
        );
        let anchor_a = array![[1.0_f64], [1.0]] / 2.0_f64.sqrt();
        let anchor_b = array![[1.0_f64], [-1.0]] / 2.0_f64.sqrt();
        let score_grads = array![[0.3_f64, -0.1], [-0.2, 0.4], [0.5, 0.2]];

        // CPU reference
        let anchors_ref = vec![&anchor_a, &anchor_b];
        let mut cpu_workspace = CmrMultiAnchorBackwardWorkspace::default();
        let mut cpu_grad_emb = Array2::<f64>::zeros(embeddings.dim());
        let mut cpu_grad_anchors = vec![
            Array2::<f64>::zeros(anchor_a.dim()),
            Array2::<f64>::zeros(anchor_b.dim()),
        ];
        cmr_multi_anchor_backward(
            &embeddings,
            &embedding_norms,
            &anchors_ref,
            &score_grads,
            1e-8,
            &mut cpu_grad_emb,
            &mut cpu_grad_anchors,
            &mut cpu_workspace,
        );

        // Device variant
        let emb_buf = dev.upload_2d(&embeddings);
        let norms_buf = dev.upload_1d(&embedding_norms);
        let anchor_a_buf = dev.upload_2d(&anchor_a);
        let anchor_b_buf = dev.upload_2d(&anchor_b);
        let anchors_buf: Vec<&Buffer2<f64>> = vec![&anchor_a_buf, &anchor_b_buf];
        let score_grads_buf = dev.upload_2d(&score_grads);
        let mut dev_grad_emb: Buffer2<f64> = dev.zeros_2d(3, 2);
        let mut dev_grad_anchors = vec![dev.zeros_2d(2, 1), dev.zeros_2d(2, 1)];

        cmr_multi_anchor_backward_device(
            &dev,
            &emb_buf,
            &norms_buf,
            &anchors_buf,
            &score_grads_buf,
            1e-8,
            &mut dev_grad_emb,
            &mut dev_grad_anchors,
        );

        let dev_emb = dev.download_2d(&dev_grad_emb);
        let emb_diff = (&dev_emb - &cpu_grad_emb).mapv(|v| v * v).sum().sqrt();
        assert!(emb_diff < 1e-10, "device embedding grad diff = {emb_diff}");

        for (dev_a, cpu_a) in dev_grad_anchors.iter().zip(cpu_grad_anchors.iter()) {
            let a_diff = (&dev.download_2d(dev_a) - cpu_a)
                .mapv(|v| v * v)
                .sum()
                .sqrt();
            assert!(a_diff < 1e-10, "device anchor grad diff = {a_diff}");
        }
    }
}
