//! Tests for the gauge dynamics evolution law.

use super::*;
use crate::geometry::grassmann::{ExponentialMapConfig, exponential_map};
use crate::state::cosmos_state::CosmosState;
use ndarray::Array2;
use std::collections::HashSet;

/// Helper: create a CosmosState and GaugeDynamics with default config.
fn make_default(shapes: &[(usize, usize)]) -> (CosmosState, GaugeDynamics) {
    let state = CosmosState::new(shapes).unwrap();
    let gd = GaugeDynamics::new(state.num_blocks(), Default::default()).unwrap();
    (state, gd)
}

/// Helper: create random-ish gradient for block i (deterministic).
fn make_gradient(n: usize, k: usize, seed: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, k), |(r, c)| {
        let v = ((r * 7 + c * 13 + seed * 31) % 97) as f64 / 97.0 - 0.5;
        v * 0.1
    })
}

fn frobenius(a: &Array2<f64>) -> f64 {
    a.mapv(|x| x * x).sum().sqrt()
}

// ══════════ Config & Defaults ══════════

#[test]
fn config_default_matches_constants() {
    let cfg = GaugeDynamicsConfig::default();
    assert_eq!(cfg.lr, super::super::DEFAULT_LR);
    assert_eq!(cfg.momentum, super::super::DEFAULT_MOMENTUM);
    assert_eq!(
        cfg.retraction_interval,
        super::super::DEFAULT_RETRACTION_INTERVAL
    );
    assert_eq!(cfg.comm_target_beta, super::super::DEFAULT_COMM_TARGET_BETA);
    assert_eq!(
        cfg.spectral_norm_power_iters,
        super::super::DEFAULT_SPECTRAL_NORM_POWER_ITERS
    );
    assert!(cfg.use_block_updates);
    assert!(!cfg.check_nan);
    assert!(!cfg.skip_antisymmetrize);
    assert!(cfg.momentum_max_norm.is_none());
}

#[test]
fn config_struct_update_syntax() {
    let cfg = GaugeDynamicsConfig {
        lr: 0.05,
        momentum: 0.9,
        ..Default::default()
    };
    assert_eq!(cfg.lr, 0.05);
    assert_eq!(cfg.momentum, 0.9);
    assert!(cfg.use_block_updates); // still default
}

// ══════════ Constructor Validation ══════════

#[test]
fn new_with_defaults_succeeds() {
    let gd = GaugeDynamics::new(3, Default::default());
    assert!(gd.is_ok());
    let gd = gd.unwrap();
    assert_eq!(gd.num_blocks(), 3);
    assert_eq!(gd.global_step(), 0);
    assert_eq!(gd.lr(), super::super::DEFAULT_LR);
}

#[test]
fn new_rejects_negative_lr() {
    let cfg = GaugeDynamicsConfig {
        lr: -0.01,
        ..Default::default()
    };
    let err = GaugeDynamics::new(2, cfg).unwrap_err();
    assert!(matches!(err, RulesError::InvalidLearningRate(lr) if lr == -0.01));
}

#[test]
fn new_accepts_zero_lr() {
    let cfg = GaugeDynamicsConfig {
        lr: 0.0,
        ..Default::default()
    };
    assert!(GaugeDynamics::new(2, cfg).is_ok());
}

#[test]
fn new_rejects_grassmann_global_updates() {
    let cfg = GaugeDynamicsConfig {
        block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
        use_block_updates: false,
        ..Default::default()
    };
    assert!(matches!(
        GaugeDynamics::new(2, cfg).unwrap_err(),
        RulesError::GrassmannRequiresBlockUpdates
    ));
}

#[test]
fn new_rejects_momentum_out_of_range() {
    // momentum >= 1.0
    let cfg = GaugeDynamicsConfig {
        momentum: 1.0,
        ..Default::default()
    };
    assert!(matches!(
        GaugeDynamics::new(2, cfg).unwrap_err(),
        RulesError::InvalidMomentum(_)
    ));
    // momentum < 0
    let cfg = GaugeDynamicsConfig {
        momentum: -0.1,
        ..Default::default()
    };
    assert!(matches!(
        GaugeDynamics::new(2, cfg).unwrap_err(),
        RulesError::InvalidMomentum(_)
    ));
}

#[test]
fn new_rejects_negative_max_norm() {
    let cfg = GaugeDynamicsConfig {
        momentum_max_norm: Some(-1.0),
        ..Default::default()
    };
    assert!(matches!(
        GaugeDynamics::new(2, cfg).unwrap_err(),
        RulesError::InvalidMomentumMaxNorm(_)
    ));
}

#[test]
fn new_rejects_zero_max_norm() {
    let cfg = GaugeDynamicsConfig {
        momentum_max_norm: Some(0.0),
        ..Default::default()
    };
    assert!(matches!(
        GaugeDynamics::new(2, cfg).unwrap_err(),
        RulesError::InvalidMomentumMaxNorm(_)
    ));
}

// ══════════ Getters & Setters ══════════

#[test]
fn set_lr_updates_value() {
    let mut gd = GaugeDynamics::new(2, Default::default()).unwrap();
    gd.set_lr(0.05);
    assert_eq!(gd.lr(), 0.05);
}

#[test]
fn global_step_increments_on_step() {
    let (mut state, mut gd) = make_default(&[(4, 2), (4, 2)]);
    let g0 = make_gradient(4, 2, 0);
    let g1 = make_gradient(4, 2, 1);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];

    assert_eq!(gd.global_step(), 0);
    gd.step(&mut state, &grads).unwrap();
    assert_eq!(gd.global_step(), 1);
    gd.step(&mut state, &grads).unwrap();
    assert_eq!(gd.global_step(), 2);
}

// ══════════ Block Masking ══════════

#[test]
fn default_has_no_active_block_mask() {
    let gd = GaugeDynamics::new(3, Default::default()).unwrap();
    assert!(gd.get_active_blocks().is_none());
}

#[test]
fn set_active_blocks_restricts() {
    let mut gd = GaugeDynamics::new(4, Default::default()).unwrap();
    let active: HashSet<usize> = [0, 2].iter().copied().collect();
    gd.set_active_blocks(Some(active));
    assert!(gd.get_active_blocks().is_some());

    let frozen = gd.get_frozen_blocks();
    assert!(frozen.contains(&1));
    assert!(frozen.contains(&3));
    assert!(!frozen.contains(&0));
    assert!(!frozen.contains(&2));
}

#[test]
fn clear_active_blocks_activates_all() {
    let mut gd = GaugeDynamics::new(3, Default::default()).unwrap();
    gd.set_active_blocks(Some([0].iter().copied().collect()));
    gd.clear_active_blocks();
    assert!(gd.get_active_blocks().is_none());
    assert!(gd.get_frozen_blocks().is_empty());
}

// ══════════ Step Basics ══════════

#[test]
fn step_rejects_wrong_gradient_count() {
    let (mut state, mut gd) = make_default(&[(4, 2), (4, 2)]);
    // Pass 3 gradients for 2 blocks
    let grads: Vec<Option<&Array2<f64>>> = vec![None, None, None];
    let err = gd.step(&mut state, &grads).unwrap_err();
    assert!(matches!(
        err,
        RulesError::BlockCountMismatch {
            expected: 2,
            got: 3
        }
    ));
}

#[test]
fn step_with_none_gradients_is_noop() {
    let (mut state, mut gd) = make_default(&[(4, 2), (4, 2)]);
    let w0_before = state.get_basis(0).unwrap().clone();
    let w1_before = state.get_basis(1).unwrap().clone();

    let grads: Vec<Option<&Array2<f64>>> = vec![None, None];
    gd.step(&mut state, &grads).unwrap();

    let w0_after = state.get_basis(0).unwrap();
    let w1_after = state.get_basis(1).unwrap();
    assert_eq!(&w0_before, w0_after);
    assert_eq!(&w1_before, w1_after);
}

#[test]
fn step_with_zero_lr_preserves_state() {
    let mut state = CosmosState::new(&[(6, 3), (6, 3)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        lr: 0.0,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(2, cfg).unwrap();

    let w0_before = state.get_basis(0).unwrap().clone();
    let g0 = make_gradient(6, 3, 42);
    let g1 = make_gradient(6, 3, 99);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];

    gd.step(&mut state, &grads).unwrap();
    let diff = (&w0_before - state.get_basis(0).unwrap())
        .mapv(|x| x * x)
        .sum();
    assert!(
        diff < 1e-24,
        "State should not change with lr=0, diff={diff}"
    );
}

// ══════════ Orthogonality Preservation ══════════

#[test]
fn step_preserves_orthogonality() {
    let (mut state, mut gd) = make_default(&[(8, 4), (6, 3)]);
    let g0 = make_gradient(8, 4, 1);
    let g1 = make_gradient(6, 3, 2);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];

    for _ in 0..5 {
        gd.step(&mut state, &grads).unwrap();
    }

    // Check W^T W ≈ I for each block
    for i in 0..2 {
        let w = state.get_basis(i).unwrap();
        let (_, k) = state.block_shapes()[i];
        let wtw = w.t().dot(w);
        let eye = Array2::<f64>::eye(k);
        let err = (&wtw - &eye).mapv(|x| x * x).sum().sqrt();
        assert!(err < 1e-10, "Block {i} orthogonality error: {err}");
    }
}

#[test]
fn parallel_block_evolution_matches_sequential_bit_for_bit() {
    // The host Grassmann block evolution can spread the independent
    // per-block maps across worker threads. The result must be
    // byte-identical to the sequential loop: each block is a pure map
    // and the updates are applied in index order regardless of thread
    // scheduling. Uneven block sizes exercise the work-stealing queue.
    let shapes = [(12, 4), (6, 2), (10, 5), (8, 3), (6, 2)];
    let make_gd = || {
        let cfg = GaugeDynamicsConfig {
            block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
            ..Default::default()
        };
        GaugeDynamics::new(shapes.len(), cfg).unwrap()
    };

    let grads: Vec<Array2<f64>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(n, k))| make_gradient(n, k, 100 + i))
        .collect();
    let grad_refs: Vec<Option<&Array2<f64>>> = grads.iter().map(Some).collect();

    // `CosmosState::new` is deterministic; assert the two initial
    // states agree before evolving so a parity failure can only come
    // from the threaded evolution itself.
    let mut state_seq = CosmosState::new(&shapes).unwrap();
    let mut state_par = CosmosState::new(&shapes).unwrap();
    for i in 0..shapes.len() {
        assert_eq!(
            state_seq.get_basis(i).unwrap(),
            state_par.get_basis(i).unwrap(),
            "initial basis {i} must match before evolving"
        );
    }

    let mut gd_seq = make_gd();
    let mut gd_par = make_gd();
    gd_seq.set_evolve_threads(1);
    gd_par.set_evolve_threads(4);

    for _ in 0..4 {
        gd_seq.step(&mut state_seq, &grad_refs).unwrap();
        gd_par.step(&mut state_par, &grad_refs).unwrap();
    }

    for i in 0..shapes.len() {
        let a = state_seq.get_basis(i).unwrap();
        let b = state_par.get_basis(i).unwrap();
        assert_eq!(a.dim(), b.dim(), "block {i} shape");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "block {i} diverged between sequential and parallel evolution"
            );
        }
    }
}

// ══════════ Connections & Spectral Norm ══════════

#[test]
fn compute_connections_zeros_for_skipped_blocks() {
    let (state, gd) = make_default(&[(4, 2), (4, 2)]);
    let g0 = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), None];
    let skip_set = HashSet::new();

    let (a_blocks, norms) = gd.compute_connections(&state, &grads, &skip_set);
    // Block 0 has gradient → non-trivial
    // Block 1 has None → should be zeros
    assert!(a_blocks[1].mapv(|x| x.abs()).sum() < 1e-15);
    assert_eq!(norms[1], 0.0);
}

#[test]
fn compute_connections_skips_inactive_blocks() {
    let (state, gd) = make_default(&[(4, 2), (4, 2)]);
    let g0 = make_gradient(4, 2, 0);
    let g1 = make_gradient(4, 2, 1);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];
    let skip_set: HashSet<usize> = [1].iter().copied().collect();

    let (a_blocks, norms) = gd.compute_connections(&state, &grads, &skip_set);
    assert!(a_blocks[0].mapv(|x| x.abs()).sum() > 0.0);
    assert!(a_blocks[1].mapv(|x| x.abs()).sum() < 1e-15);
    assert_eq!(norms[1], 0.0);
}

#[test]
fn connections_are_antisymmetric() {
    let (state, gd) = make_default(&[(6, 3)]);
    let g = make_gradient(6, 3, 42);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];
    let skip_set = HashSet::new();

    let (a_blocks, _) = gd.compute_connections(&state, &grads, &skip_set);
    let a = &a_blocks[0];
    let diff = (a + &a.t()).mapv(|x| x * x).sum().sqrt();
    assert!(
        diff < 1e-12,
        "Connection A should be antisymmetric, ||A + A^T|| = {diff}"
    );
}

#[test]
fn compute_connections_matches_projector_commutator_reference() {
    let (state, gd) = make_default(&[(6, 3)]);
    let g = make_gradient(6, 3, 7);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];
    let skip_set = HashSet::new();

    let (a_blocks, norms) = gd.compute_connections(&state, &grads, &skip_set);
    let a = &a_blocks[0];
    let w = state.get_basis(0).unwrap();
    let p = w.dot(&w.t());
    let wt_grad = w.t().dot(&g);
    let g_tangent = &g - &w.dot(&wt_grad);
    let g_p = g_tangent.dot(&w.t()) + w.dot(&g_tangent.t());
    let c = g_p.dot(&p) - p.dot(&g_p);
    let reference_norm = c.mapv(|x| x * x).sum().sqrt();
    let diff = (a - &c).mapv(|x| x * x).sum().sqrt();

    assert!(diff < 1e-10, "connection diff = {diff}");
    assert!(
        (norms[0] - reference_norm).abs() < 1e-10,
        "norm mismatch: {} vs {}",
        norms[0],
        reference_norm
    );
}

#[test]
fn spectral_norm_of_identity_is_one() {
    let eye = Array2::<f64>::eye(5);
    let sigma = GaugeDynamics::spectral_norm_approx(&eye, 10);
    assert!(
        (sigma - 1.0).abs() < 1e-6,
        "σ_max(I) should be 1.0, got {sigma}"
    );
}

#[test]
fn spectral_norm_of_scaled_matrix() {
    let mut m = Array2::<f64>::eye(4);
    m[[0, 0]] = 3.0;
    m[[1, 1]] = 2.0;
    let sigma = GaugeDynamics::spectral_norm_approx(&m, 20);
    assert!(
        (sigma - 3.0).abs() < 1e-4,
        "σ_max should be 3.0, got {sigma}"
    );
}

#[test]
fn spectral_norm_of_zero_matrix() {
    let z = Array2::<f64>::zeros((4, 4));
    let sigma = GaugeDynamics::spectral_norm_approx(&z, 5);
    assert!(sigma < 1e-15);
}

#[test]
fn spectral_norm_of_empty_matrix() {
    let z = Array2::<f64>::zeros((0, 0));
    let sigma = GaugeDynamics::spectral_norm_approx(&z, 5);
    assert_eq!(sigma, 0.0);
}

// ══════════ Momentum ══════════

#[test]
fn no_momentum_does_not_populate_buffers() {
    let (mut state, mut gd) = make_default(&[(4, 2)]);
    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    gd.step(&mut state, &grads).unwrap();
    assert!(gd.connection_momentum[0].is_none());
}

#[test]
fn momentum_populates_buffer_on_first_step() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        momentum: 0.9,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];
    gd.step(&mut state, &grads).unwrap();
    assert!(gd.connection_momentum[0].is_some());
}

#[test]
fn grassmann_momentum_buffer_uses_tangent_shape() {
    let mut state = CosmosState::new(&[(6, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        momentum: 0.9,
        block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(6, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];
    gd.step(&mut state, &grads).unwrap();

    assert_eq!(gd.connection_momentum[0].as_ref().unwrap().dim(), (6, 2));
}

#[test]
fn momentum_accumulates_over_steps() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        momentum: 0.9,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    gd.step(&mut state, &grads).unwrap();
    let norm1 = gd.connection_momentum[0]
        .as_ref()
        .unwrap()
        .mapv(|x| x * x)
        .sum()
        .sqrt();

    gd.step(&mut state, &grads).unwrap();
    let norm2 = gd.connection_momentum[0]
        .as_ref()
        .unwrap()
        .mapv(|x| x * x)
        .sum()
        .sqrt();

    // With momentum accumulation, norm should generally grow
    assert!(
        norm2 > norm1 * 0.5,
        "Momentum buffer should accumulate: norm1={norm1}, norm2={norm2}"
    );
}

// ══════════ Spectral Clipping ══════════

#[test]
fn spectral_clipping_enforces_max_norm() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        momentum: 0.9,
        momentum_max_norm: Some(0.001),
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    // Run several steps so momentum builds up
    for _ in 0..10 {
        gd.step(&mut state, &grads).unwrap();
    }

    let buf = gd.connection_momentum[0].as_ref().unwrap();
    let sigma = GaugeDynamics::spectral_norm_approx(buf, 10);
    assert!(
        sigma <= 0.001 + 1e-6,
        "Spectral norm should be clipped to 0.001, got {sigma}"
    );
}

// ══════════ Block Updates vs Global Update ══════════

#[test]
fn block_and_global_updates_produce_similar_results() {
    let shapes = &[(4, 2), (4, 2)];
    let g0 = make_gradient(4, 2, 10);
    let g1 = make_gradient(4, 2, 20);

    // Block updates
    let mut state_b = CosmosState::new(shapes).unwrap();
    let cfg_b = GaugeDynamicsConfig {
        use_block_updates: true,
        ..Default::default()
    };
    let mut gd_b = GaugeDynamics::new(2, cfg_b).unwrap();
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];
    gd_b.step(&mut state_b, &grads).unwrap();

    // Global update
    let mut state_g = CosmosState::new(shapes).unwrap();
    let cfg_g = GaugeDynamicsConfig {
        use_block_updates: false,
        ..Default::default()
    };
    let mut gd_g = GaugeDynamics::new(2, cfg_g).unwrap();
    gd_g.step(&mut state_g, &grads).unwrap();

    // Results should be numerically very close
    for i in 0..2 {
        let wb = state_b.get_basis(i).unwrap();
        let wg = state_g.get_basis(i).unwrap();
        let diff = (wb - wg).mapv(|x| x * x).sum().sqrt();
        assert!(diff < 1e-10, "Block {i}: block vs global diff = {diff}");
    }
}

// ══════════ Commutator Target EMA ══════════

#[test]
fn bootstrap_copies_raw_norms() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        comm_target_bootstrap_steps: 3,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    gd.step(&mut state, &grads).unwrap();
    // During bootstrap, EMA should equal raw norms
    assert_eq!(gd.comm_target_ema[0], gd.last_commutator_norms[0]);
}

#[test]
fn ema_updates_after_bootstrap() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        comm_target_bootstrap_steps: 1,
        comm_target_beta: 0.9,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    // Step 1: bootstrap
    gd.step(&mut state, &grads).unwrap();
    let ema_after_bootstrap = gd.comm_target_ema[0];

    // Step 2: EMA update
    gd.step(&mut state, &grads).unwrap();
    let ema_after = gd.comm_target_ema[0];

    // EMA should have changed from raw copy
    assert!(
        (ema_after - ema_after_bootstrap).abs() > 1e-15,
        "EMA should update after bootstrap"
    );
}

// ══════════ Diagnostics ══════════

#[test]
fn diagnostics_fields_match_block_count() {
    let (mut state, mut gd) = make_default(&[(4, 2), (6, 3), (4, 2)]);
    let g0 = make_gradient(4, 2, 0);
    let g1 = make_gradient(6, 3, 1);
    let g2 = make_gradient(4, 2, 2);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1), Some(&g2)];

    gd.step(&mut state, &grads).unwrap();
    let diag = gd.get_diagnostics(&state);

    assert_eq!(diag.connection_norms.len(), 3);
    assert_eq!(diag.commutator_norms.len(), 3);
    assert_eq!(diag.commutator_targets.len(), 3);
    assert_eq!(diag.geom_error.len(), 3);
    assert_eq!(diag.deadband_mask.len(), 3);
    assert_eq!(diag.orthogonality_errors.len(), 3);
    assert_eq!(diag.global_step, 1);
    assert_eq!(diag.lr, super::super::DEFAULT_LR);
}

#[test]
fn diagnostics_orthogonality_near_zero_for_fresh_state() {
    let (state, gd) = make_default(&[(4, 2), (4, 2)]);
    let diag = gd.get_diagnostics(&state);

    for (i, &err) in diag.orthogonality_errors.iter().enumerate() {
        assert!(
            err < 1e-14,
            "Block {i}: orth error should be ≈ 0 for fresh state, got {err}"
        );
    }
    assert!(diag.total_orth_error < 1e-14);
}

#[test]
fn compute_commutator_norms_now_matches_step() {
    let (mut state, mut gd) = make_default(&[(4, 2)]);
    let g = make_gradient(4, 2, 5);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    // Before step: compute norms manually
    let norms_before = gd.compute_commutator_norms_now(&state, &grads);

    // Step uses the same compute_connections internally
    gd.step(&mut state, &grads).unwrap();

    // The norms computed before step should match the cached ones
    // (state was the same at that point)
    assert!(
        (norms_before[0] - gd.last_commutator_norms[0]).abs() < 1e-12,
        "compute_commutator_norms_now should agree with step: {} vs {}",
        norms_before[0],
        gd.last_commutator_norms[0]
    );
}

// ══════════ Deadband ══════════

#[test]
fn deadband_zeroes_small_errors() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        comm_geom_deadband: 10.0, // huge deadband — all errors inside
        comm_target_bootstrap_steps: 0,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    gd.step(&mut state, &grads).unwrap();

    // With huge deadband, geom error should be zeroed and mask should be true
    assert_eq!(gd.last_geom_error[0], 0.0);
    assert!(gd.last_deadband_mask[0]);
}

// ══════════ Retraction ══════════

#[test]
fn retraction_fires_at_correct_interval() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        lr: 0.5, // large lr to introduce drift
        retraction_interval: 3,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    // Steps 1, 2: no retraction
    gd.step(&mut state, &grads).unwrap();
    gd.step(&mut state, &grads).unwrap();

    // Step 3: retraction fires (3 % 3 == 0)
    gd.step(&mut state, &grads).unwrap();
    let w = state.get_basis(0).unwrap();
    let wtw = w.t().dot(w);
    let eye = Array2::<f64>::eye(2);
    let err = (&wtw - &eye).mapv(|x| x * x).sum().sqrt();
    assert!(
        err < 1e-12,
        "After retraction at step 3, orth error should be tiny: {err}"
    );
}

#[test]
fn step_profile_marks_retraction_and_records_nonnegative_times() {
    let mut state = CosmosState::new(&[(4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        retraction_interval: 1,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 2, 0);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g)];

    gd.step(&mut state, &grads).unwrap();
    let profile = gd.last_step_profile();

    assert!(profile.used_retraction);
    assert!(profile.total_sec >= 0.0);
    assert!(profile.connection_build_sec >= 0.0);
    assert!(profile.apply_updates_sec >= 0.0);
    assert!(profile.retraction_sec >= 0.0);
}

#[test]
fn grassmann_block_update_matches_grassmann_exponential_map() {
    let mut state = CosmosState::new(&[(6, 2)]).unwrap();
    let initial_basis = state.get_basis(0).unwrap().clone();
    let grad = make_gradient(6, 2, 7);
    let tangent = &grad - &initial_basis.dot(&initial_basis.t().dot(&grad));
    let expected =
        exponential_map(&initial_basis, &tangent, &ExponentialMapConfig { t: -0.2 }).unwrap();

    let cfg = GaugeDynamicsConfig {
        lr: 0.2,
        momentum: 0.0,
        block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
        retraction_interval: 0,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&grad)];

    gd.step(&mut state, &grads).unwrap();

    let actual = state.get_basis(0).unwrap();
    assert!(frobenius(&(actual - &expected)) < 1e-8);
    assert!(state.orthogonality_errors()[0] < 1e-10);
}

#[test]
fn grassmann_block_update_stays_close_to_ambient_projector_for_small_step() {
    let initial_basis = CosmosState::new(&[(6, 2)])
        .unwrap()
        .get_basis(0)
        .unwrap()
        .clone();
    let grad = make_gradient(6, 2, 11);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&grad)];

    let mut state_grassmann = CosmosState::new(&[(6, 2)]).unwrap();
    state_grassmann
        .set_block_basis(0, initial_basis.clone())
        .unwrap();
    let mut state_ambient = CosmosState::new(&[(6, 2)]).unwrap();
    state_ambient.set_block_basis(0, initial_basis).unwrap();

    let mut grassmann = GaugeDynamics::new(
        1,
        GaugeDynamicsConfig {
            lr: 1e-3,
            momentum: 0.0,
            block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
            retraction_interval: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut ambient = GaugeDynamics::new(
        1,
        GaugeDynamicsConfig {
            lr: 1e-3,
            momentum: 0.0,
            exp_method: OrthoMethod::Taylor,
            block_update_method: GaugeBlockUpdateMethod::AmbientExp,
            retraction_interval: 0,
            ..Default::default()
        },
    )
    .unwrap();

    grassmann.step(&mut state_grassmann, &grads).unwrap();
    ambient.step(&mut state_ambient, &grads).unwrap();

    let p_grassmann = state_grassmann.get_projector(0).unwrap();
    let p_ambient = state_ambient.get_projector(0).unwrap();
    assert!(frobenius(&(&p_grassmann - &p_ambient)) < 1e-5);
}

// ══════════ Refresh for New Blocks ══════════

#[test]
fn refresh_resets_state_but_preserves_config() {
    let mut gd = GaugeDynamics::new(3, Default::default()).unwrap();
    let (mut state, _) = make_default(&[(4, 2), (4, 2), (4, 2)]);
    let g0 = make_gradient(4, 2, 0);
    let g1 = make_gradient(4, 2, 1);
    let g2 = make_gradient(4, 2, 2);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1), Some(&g2)];
    gd.step(&mut state, &grads).unwrap();

    let original_lr = gd.lr();

    gd.refresh_for_new_blocks(5);
    assert_eq!(gd.num_blocks(), 5);
    assert_eq!(gd.lr(), original_lr); // config preserved
    assert_eq!(gd.global_step(), 1); // step count preserved
    assert_eq!(gd.last_connection_norms.len(), 5);
    assert_eq!(gd.comm_target_ema.len(), 5);
    assert_eq!(gd.connection_momentum.len(), 5);
    assert!(gd.connection_momentum.iter().all(|b| b.is_none()));
}

// ══════════ Display ══════════

#[test]
fn display_includes_key_info() {
    let gd = GaugeDynamics::new(4, Default::default()).unwrap();
    let s = format!("{gd}");
    assert!(s.contains("GaugeDynamics"));
    assert!(s.contains("lr="));
    assert!(s.contains("momentum="));
    assert!(s.contains("blocks=4"));
    assert!(s.contains("step=0"));
}

// ══════════ Active Blocks During Step ══════════

#[test]
fn frozen_blocks_are_not_updated() {
    let (mut state, mut gd) = make_default(&[(4, 2), (4, 2)]);
    let w1_before = state.get_basis(1).unwrap().clone();

    // Only update block 0
    gd.set_active_blocks(Some([0].iter().copied().collect()));

    let g0 = make_gradient(4, 2, 0);
    let g1 = make_gradient(4, 2, 1);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];

    gd.step(&mut state, &grads).unwrap();

    let w1_after = state.get_basis(1).unwrap();
    assert_eq!(&w1_before, w1_after, "Frozen block 1 should not change");
}

// ══════════ NaN Check ══════════

#[test]
fn nan_gradients_skipped_when_check_nan_enabled() {
    let mut state = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
    let cfg = GaugeDynamicsConfig {
        check_nan: true,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(2, cfg).unwrap();

    let w0_before = state.get_basis(0).unwrap().clone();

    let mut g_nan = make_gradient(4, 2, 0);
    g_nan[[0, 0]] = f64::NAN;
    let g1 = make_gradient(4, 2, 1);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g_nan), Some(&g1)];

    gd.step(&mut state, &grads).unwrap();

    // Block 0 had NaN gradient — should be unchanged
    let w0_after = state.get_basis(0).unwrap();
    assert_eq!(
        &w0_before, w0_after,
        "Block with NaN gradient should not change"
    );
}

// ══════════ Multi-Step Stability ══════════

#[test]
fn twenty_steps_remain_stable() {
    let (mut state, mut gd) = make_default(&[(8, 4), (6, 3)]);
    let g0 = make_gradient(8, 4, 7);
    let g1 = make_gradient(6, 3, 11);
    let grads: Vec<Option<&Array2<f64>>> = vec![Some(&g0), Some(&g1)];

    for _ in 0..20 {
        gd.step(&mut state, &grads).unwrap();
    }

    // State should still have finite values and reasonable orthogonality
    for i in 0..2 {
        let w = state.get_basis(i).unwrap();
        assert!(
            w.iter().all(|v| v.is_finite()),
            "Block {i} has non-finite values after 20 steps"
        );

        let (_, k) = state.block_shapes()[i];
        let wtw = w.t().dot(w);
        let eye = Array2::<f64>::eye(k);
        let err = (&wtw - &eye).mapv(|x| x * x).sum().sqrt();
        assert!(
            err < 1e-6,
            "Block {i} orthogonality error after 20 steps: {err}"
        );
    }
}

// ══════════ Square (pure-fiber) blocks ══════════

/// A square block is pure fiber: `Gr(k, k)` is a point, so the
/// Grassmann horizontal projection annihilates its gradient. The
/// evolution law must still move it — the gauge-token head's token
/// frames are exactly such blocks — while keeping `WᵀW = I` at
/// machine precision.
#[test]
fn square_blocks_evolve_and_stay_orthonormal() {
    for method in [
        GaugeBlockUpdateMethod::GrassmannExp,
        GaugeBlockUpdateMethod::AmbientExp,
    ] {
        // A square gauge block beside an ordinary tall block, so a
        // mixed block layout is covered too.
        let mut state = CosmosState::new(&[(6, 3), (4, 4), (4, 4)]).unwrap();
        let initial: Vec<Array2<f64>> = (0..3)
            .map(|i| state.get_basis(i).unwrap().clone())
            .collect();
        let cfg = GaugeDynamicsConfig {
            lr: 0.2,
            momentum: 0.9,
            block_update_method: method,
            retraction_interval: 0,
            ..Default::default()
        };
        let mut gd = GaugeDynamics::new(3, cfg).unwrap();

        let grads = [
            make_gradient(6, 3, 1),
            make_gradient(4, 4, 2),
            make_gradient(4, 4, 3),
        ];
        let grad_refs: Vec<Option<&Array2<f64>>> = grads.iter().map(Some).collect();

        for _ in 0..5 {
            gd.step(&mut state, &grad_refs).unwrap();
        }

        for i in 0..3 {
            let w = state.get_basis(i).unwrap();
            let moved = frobenius(&(w - &initial[i]));
            assert!(
                moved > 1e-6,
                "{method:?}: block {i} did not move after 5 steps (delta {moved})"
            );

            let wtw = w.t().dot(w);
            let eye = Array2::<f64>::eye(w.ncols());
            let err = frobenius(&(&wtw - &eye));
            assert!(
                err < 1e-12,
                "{method:?}: block {i} orthogonality error {err} after 5 steps"
            );
        }
    }
}

/// The pure-fiber direction is the skew part of `WᵀG`, and the
/// resulting frame update is exactly the canonical `O(k)` step
/// `W <- W exp(-2·lr·skew(WᵀG))`.
#[test]
fn square_block_step_matches_the_orthogonal_group_update() {
    use crate::geometry::matrix_exp::{OrthoMethod, scaled_matrix_exp_antisym};

    let mut state = CosmosState::new(&[(4, 4)]).unwrap();
    let w0 = state.get_basis(0).unwrap().clone();
    let cfg = GaugeDynamicsConfig {
        lr: 0.15,
        momentum: 0.0,
        block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
        exp_method: OrthoMethod::Taylor,
        retraction_interval: 0,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(4, 4, 7);
    gd.step(&mut state, &[Some(&g)]).unwrap();

    // exp(-lr·A)·W with A = 2·W·skew·Wᵀ equals W·exp(-2·lr·skew).
    let wtg = w0.t().dot(&g);
    let skew = (&wtg - &wtg.t()).mapv(|v| 0.5 * v);
    let expected =
        w0.dot(&scaled_matrix_exp_antisym(&skew, -2.0 * 0.15, OrthoMethod::Taylor).unwrap());

    let diff = frobenius(&(state.get_basis(0).unwrap() - &expected));
    assert!(diff < 1e-10, "square-block step deviates by {diff}");
}

/// The pure-fiber branch must not disturb rectangular blocks: with no
/// square block present the trajectory is the historical one.
#[test]
fn rectangular_blocks_keep_the_grassmann_trajectory() {
    use crate::geometry::grassmann::{ExponentialMapConfig, exponential_map};

    let mut state = CosmosState::new(&[(6, 2)]).unwrap();
    let w0 = state.get_basis(0).unwrap().clone();
    let cfg = GaugeDynamicsConfig {
        lr: 0.25,
        momentum: 0.0,
        block_update_method: GaugeBlockUpdateMethod::GrassmannExp,
        retraction_interval: 0,
        ..Default::default()
    };
    let mut gd = GaugeDynamics::new(1, cfg).unwrap();

    let g = make_gradient(6, 2, 4);
    gd.step(&mut state, &[Some(&g)]).unwrap();

    let tangent = &g - &w0.dot(&w0.t().dot(&g));
    let expected = exponential_map(&w0, &tangent, &ExponentialMapConfig { t: -0.25 }).unwrap();

    let diff = frobenius(&(state.get_basis(0).unwrap() - &expected));
    assert!(diff < 1e-12, "rectangular trajectory changed by {diff}");
}
