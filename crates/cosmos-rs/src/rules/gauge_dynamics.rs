//! GaugeDynamics - Global gauge-theoretic evolver for COSMOS.
//!
//! This module implements the R (Rules) component of the COSMOS tuple (S, A, O, R).
//!
//! GaugeDynamics treats all Grassmann projectors as one geometry entity
//! and updates each block coherently via orthogonal conjugation / left action.
//!
//! The core update uses an anti-symmetric connection A derived from commutators
//! with projectors, then evolves each block either through an ambient
//! orthogonal action or directly on the Grassmann manifold:
//! ```text
//! U = exp(-lr * A)
//! W <- U @ W
//! ```
//! This preserves the Stiefel manifold structure (W^T W = I) required by the
//! S (State) component.

use std::collections::HashSet;
use std::fmt;
use std::time::Instant;

use ndarray::{Array1, Array2};

use crate::geometry::matrix_exp::{OrthoMethod, RetractionMethod};
use crate::state::cosmos_state::CosmosState;

use super::{
    DEFAULT_COMM_GEOM_DEADBAND, DEFAULT_COMM_TARGET_BETA, DEFAULT_COMM_TARGET_BOOTSTRAP_STEPS,
    DEFAULT_LR, DEFAULT_MOMENTUM, DEFAULT_RETRACTION_INTERVAL, DEFAULT_SPECTRAL_NORM_POWER_ITERS,
    EPS, RulesError,
};

/// Block-local evolution map used by [`GaugeDynamics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GaugeBlockUpdateMethod {
    /// Form an ambient orthogonal matrix from the connection and apply `W <- U @ W`.
    #[default]
    AmbientExp,
    /// Evolve the subspace directly through the Grassmann exponential map.
    GrassmannExp,
}

// ─── Configuration ────────────────────────────────────────────────

/// Configuration for the [`GaugeDynamics`] evolver.
///
/// All fields have sensible defaults via the `Default` trait.
/// Use struct update syntax to override only what you need.
///
/// ```rust
/// # use cosmos_rs::rules::GaugeDynamicsConfig;
/// let config = GaugeDynamicsConfig {
///     lr: 0.05,
///     momentum: 0.9,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GaugeDynamicsConfig {
    /// Learning rate for projector updates. Must be >= 0.
    pub lr: f64,
    /// Momentum coefficient in [0, 1).
    pub momentum: f64,
    /// Optional spectral-norm clip for momentum buffers.
    pub momentum_max_norm: Option<f64>,
    /// Method for computing orthogonal transform from anti-symmetric connection.
    pub exp_method: OrthoMethod,
    /// Block-local evolution map applied after building the connection.
    pub block_update_method: GaugeBlockUpdateMethod,
    /// If true, compute exp per block. If false, build block-diagonal A* and do one global exp.
    pub use_block_updates: bool,
    /// If true, skip blocks with non-finite gradients instead of propagating NaN.
    pub check_nan: bool,
    /// Re-orthonormalize W_blocks every N steps. 0 disables.
    pub retraction_interval: usize,
    /// Retraction method used for periodic re-orthonormalization.
    pub retraction_method: RetractionMethod,
    /// EMA beta for commutator target tracking.
    pub comm_target_beta: f64,
    /// Bootstrap steps for commutator target (copies raw norms during bootstrap).
    pub comm_target_bootstrap_steps: usize,
    /// If true, freeze commutator target after bootstrap phase.
    pub freeze_comm_target_after_bootstrap: bool,
    /// Deadband for geometric error — errors below this are zeroed.
    pub comm_geom_deadband: f64,
    /// Power-iteration steps for spectral norm approximation.
    pub spectral_norm_power_iters: usize,
    /// If true, skip antisymmetrize(C) step (faster, less numerically robust).
    pub skip_antisymmetrize: bool,
}

impl Default for GaugeDynamicsConfig {
    fn default() -> Self {
        Self {
            lr: DEFAULT_LR,
            momentum: DEFAULT_MOMENTUM,
            momentum_max_norm: None,
            exp_method: OrthoMethod::default(),
            block_update_method: GaugeBlockUpdateMethod::default(),
            use_block_updates: true,
            check_nan: false,
            retraction_interval: DEFAULT_RETRACTION_INTERVAL,
            retraction_method: RetractionMethod::default(),
            comm_target_beta: DEFAULT_COMM_TARGET_BETA,
            comm_target_bootstrap_steps: DEFAULT_COMM_TARGET_BOOTSTRAP_STEPS,
            freeze_comm_target_after_bootstrap: false,
            comm_geom_deadband: DEFAULT_COMM_GEOM_DEADBAND,
            spectral_norm_power_iters: DEFAULT_SPECTRAL_NORM_POWER_ITERS,
            skip_antisymmetrize: false,
        }
    }
}

// ─── Diagnostics ──────────────────────────────────────────────────

/// Diagnostics snapshot from the most recent [`GaugeDynamics::step`] call.
#[derive(Debug, Clone)]
pub struct GaugeDynamicsDiagnostics {
    /// Frobenius norm of each block's connection matrix A_i.
    pub connection_norms: Vec<f64>,

    /// Frobenius norm of each block's commutator [G_P, P].
    pub commutator_norms: Vec<f64>,

    /// EMA targets for commutator norms.
    pub commutator_targets: Vec<f64>,

    /// Geometric error: log(||C|| / target) per block.
    pub geom_error: Vec<f64>,

    /// Per-block deadband mask (true = error was inside deadband).
    pub deadband_mask: Vec<bool>,

    /// Per-block orthogonality error ||W^T W - I||_F.
    pub orthogonality_errors: Vec<f64>,

    /// RMS orthogonality error across all blocks.
    pub total_orth_error: f64,

    /// Current evolver step count.
    pub global_step: usize,

    /// Current learning rate.
    pub lr: f64,

    /// Mean commutator norm across blocks.
    pub mean_comm_norm: f64,

    /// Mean commutator target across blocks.
    pub mean_comm_target: f64,

    /// Mean geometric error across blocks.
    pub mean_geom_error: f64,

    /// Fraction of blocks inside the deadband.
    pub deadband_fraction: f64,
}

// ─── Step Metrics (private) ───────────────────────────────────────

/// Aggregated metrics computed every N steps (private).
#[derive(Debug, Clone)]
struct StepMetrics {
    mean_comm_norm: f64,
    mean_comm_target: f64,
    mean_geom_error: f64,
    deadband_fraction: f64,
}

/// Runtime timings captured for the most recent evolution step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GaugeDynamicsStepProfile {
    /// Total wall time spent inside `GaugeDynamics::step`.
    pub total_sec: f64,
    /// Time spent building commutator-derived connections.
    pub connection_build_sec: f64,
    /// Time spent applying momentum buffers and clipping.
    pub momentum_sec: f64,
    /// Time spent updating the commutator target statistics.
    pub comm_target_sec: f64,
    /// Time spent computing internal step diagnostics.
    pub diagnostics_sec: f64,
    /// Time spent caching connection norms for later inspection.
    pub connection_norms_sec: f64,
    /// Time spent applying orthogonal updates to the state.
    pub apply_updates_sec: f64,
    /// Time spent building orthogonal update matrices inside the step.
    pub update_exp_sec: f64,
    /// Time spent applying update matrices to the state bases.
    pub update_state_apply_sec: f64,
    /// Time spent in periodic retraction.
    pub retraction_sec: f64,
    /// Whether this step executed a retraction.
    pub used_retraction: bool,
}

/// Lightweight report returned by one evolution step.
#[derive(Debug, Clone, PartialEq)]
pub struct GaugeDynamicsStepReport {
    /// Mean commutator norm across blocks for this step.
    pub mean_commutator: f64,
    /// Per-block commutator norms for this step.
    pub commutator_norms: Vec<f64>,
    /// Runtime timings captured during this step.
    pub profile: GaugeDynamicsStepProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct ApplyUpdateBreakdown {
    exp_sec: f64,
    state_apply_sec: f64,
}

// ─── GaugeDynamics ────────────────────────────────────────────────

/// Global gauge-theoretic evolver using conjugation updates.
///
/// Updates each projector block via an orthogonal transform computed from a
/// commutator-derived connection:
/// ```text
/// P = W W^T
/// C = [G_P, P]
/// A = antisym(C)
/// U = exp(-lr * A)
/// W <- U W
/// ```
///
/// # Example
/// ```
/// use cosmos_rs::state::cosmos_state::CosmosState;
/// use cosmos_rs::rules::{GaugeDynamics, GaugeDynamicsConfig};
///
/// let mut state = CosmosState::new(&[(8, 4), (8, 4)]).unwrap();
/// let config = GaugeDynamicsConfig { lr: 0.01, ..Default::default() };
/// let mut opt = GaugeDynamics::new(state.num_blocks(), config).unwrap();
/// ```
#[derive(Debug)]
pub struct GaugeDynamics {
    // ── Config (flattened for direct access) ──
    lr: f64,
    momentum: f64,
    momentum_max_norm: Option<f64>,
    exp_method: OrthoMethod,
    block_update_method: GaugeBlockUpdateMethod,
    use_block_updates: bool,
    check_nan: bool,
    retraction_interval: usize,
    retraction_method: RetractionMethod,
    comm_target_beta: f64,
    comm_target_bootstrap_steps: usize,
    freeze_comm_target_after_bootstrap: bool,
    comm_geom_deadband: f64,
    spectral_norm_power_iters: usize,
    skip_antisymmetrize: bool,

    /// Optional compute device for the heavy per-block evolution ops
    /// (currently the Grassmann exponential map). `None` (default)
    /// keeps the evolution fully on host — the canonical reference
    /// path. Set via [`Self::set_device`]; the math is identical (same
    /// geodesic formulation and polar-Gram retraction), only the
    /// execution placement changes.
    device: Option<crate::compute::Device>,

    /// Worker-thread count for the host Grassmann block evolution.
    /// `1` (default) keeps the canonical sequential loop. Values `> 1`
    /// spread the independent per-block exponential maps across scoped
    /// threads; the result is bit-identical to the sequential path
    /// because each block is an independent pure map applied in index
    /// order. Only relevant when the evolution runs on host
    /// (`device == None`); ignored on the device path. Set via
    /// [`Self::set_evolve_threads`].
    evolve_threads: usize,

    // ── Mutable evolver state ──
    num_blocks: usize,
    global_step: usize,
    active_blocks: Option<HashSet<usize>>,

    /// Per-block momentum buffer over the active evolution signal.
    ///
    /// This stores ambient connections for `AmbientExp` and tangent blocks for
    /// `GrassmannExp`.
    connection_momentum: Vec<Option<Array2<f64>>>,

    /// EMA of commutator norms (one per block).
    comm_target_ema: Vec<f64>,

    // ── Cached diagnostics from last step ──
    last_connection_norms: Vec<f64>,
    last_commutator_norms: Vec<f64>,
    last_commutator_targets: Vec<f64>,
    last_geom_error: Vec<f64>,
    last_deadband_mask: Vec<bool>,
    last_metrics: Option<StepMetrics>,
    last_step_profile: GaugeDynamicsStepProfile,
}

// ─── Constructor + Getters ────────────────────────────────────────

impl GaugeDynamics {
    /// Create a new GaugeDynamics evolver.
    ///
    /// # Arguments
    ///
    /// * `num_blocks` - Number of blocks in the CosmosState this evolver manages.
    /// * `config` - Configuration parameters (use `Default::default()` for sensible defaults).
    ///
    /// # Errors
    ///
    /// Returns [`RulesError`] if `lr < 0`, `momentum` is outside `[0, 1)`,
    /// or `momentum_max_norm` is `Some(v)` with `v <= 0`.
    pub fn new(num_blocks: usize, config: GaugeDynamicsConfig) -> Result<Self, RulesError> {
        if config.lr < 0.0 {
            return Err(RulesError::InvalidLearningRate(config.lr));
        }
        if config.momentum < 0.0 || config.momentum >= 1.0 {
            return Err(RulesError::InvalidMomentum(config.momentum));
        }
        if let Some(max_norm) = config.momentum_max_norm {
            if max_norm <= 0.0 {
                return Err(RulesError::InvalidMomentumMaxNorm(max_norm));
            }
        }
        if matches!(
            config.block_update_method,
            GaugeBlockUpdateMethod::GrassmannExp
        ) && !config.use_block_updates
        {
            return Err(RulesError::GrassmannRequiresBlockUpdates);
        }

        Ok(Self {
            lr: config.lr,
            momentum: config.momentum,
            momentum_max_norm: config.momentum_max_norm,
            exp_method: config.exp_method,
            block_update_method: config.block_update_method,
            use_block_updates: config.use_block_updates,
            check_nan: config.check_nan,
            retraction_interval: config.retraction_interval,
            retraction_method: config.retraction_method,
            comm_target_beta: config.comm_target_beta,
            comm_target_bootstrap_steps: config.comm_target_bootstrap_steps,
            freeze_comm_target_after_bootstrap: config.freeze_comm_target_after_bootstrap,
            comm_geom_deadband: config.comm_geom_deadband,
            spectral_norm_power_iters: config.spectral_norm_power_iters,
            skip_antisymmetrize: config.skip_antisymmetrize,
            device: None,
            evolve_threads: 1,

            num_blocks,
            global_step: 0,
            active_blocks: None,
            connection_momentum: vec![None; num_blocks],
            comm_target_ema: vec![0.0; num_blocks],

            last_connection_norms: vec![0.0; num_blocks],
            last_commutator_norms: vec![0.0; num_blocks],
            last_commutator_targets: vec![0.0; num_blocks],
            last_geom_error: vec![0.0; num_blocks],
            last_deadband_mask: vec![false; num_blocks],
            last_metrics: None,
            last_step_profile: GaugeDynamicsStepProfile::default(),
        })
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Update the learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Current evolver step count.
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Route the heavy per-block evolution ops (Grassmann exponential
    /// map) through a compute device. The math is unchanged — same
    /// Edelman-Arias-Smith geodesic and polar-Gram retraction — only
    /// the execution placement moves; state remains host-resident f64
    /// and every update is downloaded back before it touches
    /// `CosmosState`. Not setting a device (the default) keeps the
    /// canonical host path.
    pub fn set_device(&mut self, device: crate::compute::Device) {
        self.device = Some(device);
    }

    /// Set the worker-thread count for the host Grassmann block
    /// evolution. `1` (default) is the sequential loop; `> 1` runs the
    /// independent per-block exponential maps on that many scoped
    /// threads and applies the results in index order, so the output is
    /// bit-identical to the sequential path. A count of `0` is treated
    /// as `1`.
    ///
    /// Scope note: this only helps when the process has spare cores.
    /// The MKL backend is the sequential (`-seq`) build, so per-block
    /// maps are single-threaded and stacking them across threads is
    /// safe (no BLAS oversubscription). But when many single-process
    /// runs already saturate the machine (the usual parallel-suite
    /// workflow), intra-process threads oversubscribe and lose — keep
    /// this at `1` there. Only relevant on the host path; the device
    /// path ignores it.
    pub fn set_evolve_threads(&mut self, threads: usize) {
        self.evolve_threads = threads.max(1);
    }

    /// Clear the per-block momentum buffers.
    ///
    /// The next step then rebuilds momentum from the current connection
    /// alone, as if the run had just started. Useful when the arrangement
    /// changes mid-run (for example an interior topology swap) and the
    /// accumulated direction refers to a composition that no longer applies.
    /// Learned state, the step counter, and the commutator-target EMA are
    /// left untouched.
    pub fn reset_momentum(&mut self) {
        for buffer in &mut self.connection_momentum {
            *buffer = None;
        }
    }

    // ─── Block Masking ────────────────────────────────────────────

    /// Restrict updates to a subset of blocks.
    ///
    /// Blocks not in `indices` will be skipped during `step()`.
    /// Pass `None` to activate all blocks.
    pub fn set_active_blocks(&mut self, indices: Option<HashSet<usize>>) {
        self.active_blocks = indices;
    }

    /// Clear block masking so all blocks receive updates.
    pub fn clear_active_blocks(&mut self) {
        self.active_blocks = None;
    }

    /// Get the current active-block mask, or `None` if all are active.
    pub fn get_active_blocks(&self) -> Option<&HashSet<usize>> {
        self.active_blocks.as_ref()
    }

    /// Get indices of currently frozen (inactive) blocks.
    pub fn get_frozen_blocks(&self) -> HashSet<usize> {
        match &self.active_blocks {
            None => HashSet::new(),
            Some(active) => (0..self.num_blocks)
                .filter(|i| !active.contains(i))
                .collect(),
        }
    }

    // ─── Private: Core Math ───────────────────────────────────────

    /// Compute per-block anti-symmetric connections A_i from gradients.
    ///
    /// For each active block i with a gradient:
    /// ```text
    /// G_tangent = grad - W_i @ (W_i^T @ grad)           (tangent projection)
    /// A_i = G_tangent @ W_i^T - W_i @ G_tangent^T       (closed-form commutator)
    /// ```
    ///
    /// Returns `(a_blocks, commutator_norms)` where:
    /// - `a_blocks[i]` has shape `(n_i, n_i)` — zero if block was skipped
    /// - `commutator_norms[i]` is `||A_i||_F` — zero if block was skipped
    fn compute_connections(
        &self,
        state: &CosmosState,
        gradients: &[Option<&Array2<f64>>],
        skip_set: &HashSet<usize>,
    ) -> (Vec<Array2<f64>>, Vec<f64>) {
        let _skip_antisymmetrize = self.skip_antisymmetrize;
        let mut a_blocks: Vec<Array2<f64>> = Vec::with_capacity(self.num_blocks);
        let mut commutator_norms: Vec<f64> = vec![0.0; self.num_blocks];

        for i in 0..self.num_blocks {
            let (n, _k) = state.block_shapes()[i];

            // Skip inactive blocks or blocks without gradients
            if skip_set.contains(&i) || gradients[i].is_none() {
                a_blocks.push(Array2::zeros((n, n)));
                continue;
            }

            let grad = gradients[i].unwrap();
            let w = state.get_basis(i).unwrap();

            // Check for non-finite gradients if requested
            if self.check_nan && !grad.iter().all(|v| v.is_finite()) {
                a_blocks.push(Array2::zeros((n, n)));
                continue;
            }

            let g_tangent = Self::riemannian_direction(w, grad);

            // Closed form of the commutator-derived connection. For orthonormal
            // bases this is exactly the anti-symmetric evolution law; in finite
            // precision it also enforces anti-symmetry by construction.
            let a = g_tangent.dot(&w.t()) - w.dot(&g_tangent.t());
            commutator_norms[i] = Self::frobenius_norm(&a);

            a_blocks.push(a);
        }

        (a_blocks, commutator_norms)
    }

    /// Compute per-block tangent directions G_i from gradients.
    ///
    /// For each active block i with a gradient:
    /// ```text
    /// G_i = grad - W_i @ (W_i^T @ grad)
    /// ```
    ///
    /// Returns `(tangent_blocks, commutator_norms)` where:
    /// - `tangent_blocks[i]` has shape `(n_i, k_i)` — zero if block was skipped
    /// - `commutator_norms[i]` is `||A_i||_F = sqrt(2) ||G_i||_F`
    fn compute_tangent_blocks(
        &self,
        state: &CosmosState,
        gradients: &[Option<&Array2<f64>>],
        skip_set: &HashSet<usize>,
    ) -> (Vec<Array2<f64>>, Vec<f64>) {
        let mut tangent_blocks: Vec<Array2<f64>> = Vec::with_capacity(self.num_blocks);
        let mut commutator_norms: Vec<f64> = vec![0.0; self.num_blocks];

        for i in 0..self.num_blocks {
            let (n, k) = state.block_shapes()[i];

            if skip_set.contains(&i) || gradients[i].is_none() {
                tangent_blocks.push(Array2::zeros((n, k)));
                continue;
            }

            let grad = gradients[i].unwrap();
            let w = state.get_basis(i).unwrap();

            if self.check_nan && !grad.iter().all(|v| v.is_finite()) {
                tangent_blocks.push(Array2::zeros((n, k)));
                continue;
            }

            let g_tangent = Self::riemannian_direction(w, grad);
            commutator_norms[i] = Self::tangent_connection_norm(&g_tangent);
            tangent_blocks.push(g_tangent);
        }

        (tangent_blocks, commutator_norms)
    }

    /// Whether a block is a full-rank square frame.
    ///
    /// Such a block is *pure fiber*: `Gr(k, k)` is a single point, so it
    /// carries no Grassmann content and its learnable object is the frame
    /// itself, an element of `O(k)`.
    fn is_square_frame(w: &Array2<f64>) -> bool {
        w.nrows() == w.ncols()
    }

    /// Riemannian gradient direction of a block.
    ///
    /// A block with `n > k` is a point of `Gr(k, n)`; the horizontal lift
    /// of a Euclidean gradient is `G − W(WᵀG)` and the vertical (fiber)
    /// part is quotiented away.
    ///
    /// A square block is pure fiber, and that same projection annihilates
    /// it exactly: `W Wᵀ = I` for a square orthonormal `W`, so the
    /// horizontal component is identically zero and the block would never
    /// move. Its Riemannian gradient on `O(k)` is the skew part
    /// `W · skew(Wᵀ G)`, which is what the gauge step needs in order to
    /// evolve a learnable frame at all.
    fn riemannian_direction(w: &Array2<f64>, candidate: &Array2<f64>) -> Array2<f64> {
        let wt_candidate = w.t().dot(candidate);
        if Self::is_square_frame(w) {
            let skew = (&wt_candidate - &wt_candidate.t()).mapv(|value| 0.5 * value);
            w.dot(&skew)
        } else {
            candidate - &w.dot(&wt_candidate)
        }
    }

    /// Build the ambient antisymmetric generator `A = T W^T - W T^T`
    /// from a tangent block `T = G - W (W^T G)` and the basis `W`.
    ///
    /// This is the same generator the AmbientExp path forms directly
    /// in [`Self::compute_connections`]; the GrassmannExp path stores
    /// only the tangent block, so this helper reconstructs the
    /// equivalent ambient matrix for diagnostics / spectral capture.
    /// Antisymmetric by construction.
    fn ambient_from_tangent(w: &Array2<f64>, tangent: &Array2<f64>) -> Array2<f64> {
        let tw_t = tangent.dot(&w.t());
        &tw_t - &tw_t.t()
    }

    fn frobenius_norm(a: &Array2<f64>) -> f64 {
        a.mapv(|x| x * x).sum().sqrt()
    }

    fn tangent_connection_norm(tangent: &Array2<f64>) -> f64 {
        std::f64::consts::SQRT_2 * Self::frobenius_norm(tangent)
    }

    /// Approximate spectral norm of a dense matrix via power iteration.
    ///
    /// Computes an estimate of the largest singular value σ_max(A) using:
    /// ```text
    /// for each iteration:
    ///   v <- A^T (A v)
    ///   v <- v / ||v||
    /// σ_max ≈ ||A v||
    /// ```
    fn spectral_norm_approx(a: &Array2<f64>, num_iters: usize) -> f64 {
        let n = a.ncols();
        if n == 0 {
            return 0.0;
        }

        // Initialize v as [1, 0, 0, ...] (deterministic, avoids rand dependency)
        let mut v = Array1::zeros(n);
        v[0] = 1.0;

        for _ in 0..num_iters {
            // v <- A^T (A v)
            let av = a.dot(&v);
            v = a.t().dot(&av);

            // Normalize
            let norm = v.dot(&v).sqrt();
            if norm > EPS {
                v.mapv_inplace(|x| x / norm);
            }
        }

        // σ_max ≈ ||A v||
        let av = a.dot(&v);
        av.dot(&av).sqrt()
    }

    // ─── Step + Update Application ───────────────────────────────

    /// Perform a single gauge-theoretic optimization step.
    ///
    /// # Algorithm
    /// 1. Compute anti-symmetric connections A_i from gradients
    /// 2. Apply momentum (if configured)
    /// 3. Update commutator target EMA
    /// 4. Compute diagnostics (every 10 steps or first 5)
    /// 5. Apply orthogonal update: W_i <- exp(-lr * A_i) @ W_i
    /// 6. Periodic retraction to re-orthonormalize
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to the CosmosState being optimized.
    /// * `gradients` - One gradient per block. `None` means no gradient for that block.
    ///
    /// # Errors
    ///
    /// Returns [`RulesError::BlockCountMismatch`] if `gradients.len() != num_blocks`.
    /// Returns [`RulesError::MatrixExp`] if the matrix exponential fails.
    ///
    /// # Panics
    ///
    /// Panics if `state` reports fewer blocks than `num_blocks`, which would
    /// indicate the state and evolver have fallen out of sync.
    pub fn step(
        &mut self,
        state: &mut CosmosState,
        gradients: &[Option<&Array2<f64>>],
    ) -> Result<GaugeDynamicsStepReport, RulesError> {
        let step_start = Instant::now();

        // Validate gradient count
        if gradients.len() != self.num_blocks {
            return Err(RulesError::BlockCountMismatch {
                expected: self.num_blocks,
                got: gradients.len(),
            });
        }

        self.global_step += 1;

        // Build skip set from inactive blocks + blocks without gradients
        let mut skip_set: HashSet<usize> = match &self.active_blocks {
            None => HashSet::new(),
            Some(active) => (0..self.num_blocks)
                .filter(|i| !active.contains(i))
                .collect(),
        };

        // Also skip blocks without gradients
        for (i, g) in gradients.iter().enumerate() {
            if g.is_none() {
                skip_set.insert(i);
            }
        }

        // ===== 1) Compute per-block evolution signals =====
        let connection_build_start = Instant::now();
        let (mut a_blocks, mut tangent_blocks, commutator_norms) = match self.block_update_method {
            GaugeBlockUpdateMethod::AmbientExp => {
                let (a_blocks, commutator_norms) =
                    self.compute_connections(state, gradients, &skip_set);
                (Some(a_blocks), None, commutator_norms)
            }
            GaugeBlockUpdateMethod::GrassmannExp => {
                let (tangent_blocks, commutator_norms) =
                    self.compute_tangent_blocks(state, gradients, &skip_set);
                (None, Some(tangent_blocks), commutator_norms)
            }
        };
        let connection_build_sec = connection_build_start.elapsed().as_secs_f64();

        // ===== 2) Apply momentum =====
        let momentum_start = Instant::now();
        match self.block_update_method {
            GaugeBlockUpdateMethod::AmbientExp => {
                let a_blocks = a_blocks
                    .as_mut()
                    .expect("ambient path should build A blocks");
                if self.momentum > 0.0 {
                    for i in 0..self.num_blocks {
                        if skip_set.contains(&i) {
                            continue;
                        }
                        match &mut self.connection_momentum[i] {
                            Some(buf) => {
                                buf.zip_mut_with(&a_blocks[i], |b, &a| *b = self.momentum * *b + a);
                            }
                            None => {
                                self.connection_momentum[i] = Some(a_blocks[i].clone());
                            }
                        }
                    }
                }

                if let Some(max_norm) = self.momentum_max_norm {
                    for i in 0..self.num_blocks {
                        if skip_set.contains(&i) {
                            continue;
                        }
                        if let Some(buf) = &mut self.connection_momentum[i] {
                            let sigma =
                                Self::spectral_norm_approx(buf, self.spectral_norm_power_iters);
                            if sigma > max_norm {
                                let scale = max_norm / (sigma + EPS);
                                buf.mapv_inplace(|x| x * scale);
                            }
                        }
                    }
                }

                for i in 0..self.num_blocks {
                    if skip_set.contains(&i) {
                        continue;
                    }
                    if let Some(buf) = &self.connection_momentum[i] {
                        a_blocks[i] = buf.clone();
                    }
                }
            }
            GaugeBlockUpdateMethod::GrassmannExp => {
                let tangent_blocks = tangent_blocks
                    .as_mut()
                    .expect("grassmann path should build tangent blocks");
                if self.momentum > 0.0 {
                    for i in 0..self.num_blocks {
                        if skip_set.contains(&i) {
                            continue;
                        }
                        match &mut self.connection_momentum[i] {
                            Some(buf) => {
                                buf.zip_mut_with(&tangent_blocks[i], |b, &t| {
                                    *b = self.momentum * *b + t
                                });
                            }
                            None => {
                                self.connection_momentum[i] = Some(tangent_blocks[i].clone());
                            }
                        }
                    }
                }

                if let Some(max_norm) = self.momentum_max_norm {
                    for i in 0..self.num_blocks {
                        if skip_set.contains(&i) {
                            continue;
                        }
                        if let Some(buf) = &mut self.connection_momentum[i] {
                            let sigma =
                                Self::spectral_norm_approx(buf, self.spectral_norm_power_iters);
                            if sigma > max_norm {
                                let scale = max_norm / (sigma + EPS);
                                buf.mapv_inplace(|x| x * scale);
                            }
                        }
                    }
                }

                for i in 0..self.num_blocks {
                    if skip_set.contains(&i) {
                        continue;
                    }
                    let w = state.get_basis(i).unwrap();
                    if let Some(buf) = &mut self.connection_momentum[i] {
                        *buf = Self::riemannian_direction(w, buf);
                        tangent_blocks[i] = buf.clone();
                    } else {
                        tangent_blocks[i] = Self::riemannian_direction(w, &tangent_blocks[i]);
                    }
                }
            }
        }
        let momentum_sec = momentum_start.elapsed().as_secs_f64();

        // ===== 3) Commutator target EMA =====
        let comm_target_start = Instant::now();
        if self.global_step <= self.comm_target_bootstrap_steps {
            // Bootstrap: copy raw norms directly
            self.comm_target_ema = commutator_norms.clone();
        } else if !self.freeze_comm_target_after_bootstrap {
            // EMA update: target = β * target + (1 - β) * norm
            let beta = self.comm_target_beta;
            for i in 0..self.num_blocks {
                self.comm_target_ema[i] =
                    beta * self.comm_target_ema[i] + (1.0 - beta) * commutator_norms[i];
            }
        }
        let comm_target_sec = comm_target_start.elapsed().as_secs_f64();

        // ===== 4) Diagnostics (every 10 steps or first 5) =====
        let compute_diagnostics = (self.global_step % 10 == 0) || (self.global_step <= 5);

        let diagnostics_start = Instant::now();
        if compute_diagnostics {
            self.last_commutator_targets = self.comm_target_ema.clone();

            let mut geom_error = vec![0.0; self.num_blocks];
            let mut deadband_mask = vec![false; self.num_blocks];

            for i in 0..self.num_blocks {
                // Geometric error: log(||C|| / target)
                let err = (commutator_norms[i] + EPS).ln() - (self.comm_target_ema[i] + EPS).ln();

                if self.comm_geom_deadband > 0.0 {
                    let abs_err = err.abs();
                    if abs_err <= self.comm_geom_deadband {
                        deadband_mask[i] = true;
                        geom_error[i] = 0.0;
                    } else {
                        geom_error[i] = err.signum() * (abs_err - self.comm_geom_deadband);
                    }
                } else {
                    geom_error[i] = err;
                }
            }

            self.last_geom_error = geom_error;
            self.last_deadband_mask = deadband_mask;

            let n = self.num_blocks as f64;
            self.last_metrics = Some(StepMetrics {
                mean_comm_norm: commutator_norms.iter().sum::<f64>() / n,
                mean_comm_target: self.comm_target_ema.iter().sum::<f64>() / n,
                mean_geom_error: self.last_geom_error.iter().sum::<f64>() / n,
                deadband_fraction: self.last_deadband_mask.iter().filter(|&&m| m).count() as f64
                    / n,
            });
        }
        let diagnostics_sec = diagnostics_start.elapsed().as_secs_f64();

        // ===== 5) Connection norms =====
        let connection_norms_start = Instant::now();
        let mut connection_norms = vec![0.0; self.num_blocks];
        match self.block_update_method {
            GaugeBlockUpdateMethod::AmbientExp => {
                let a_blocks = a_blocks
                    .as_ref()
                    .expect("ambient path should retain A blocks");
                for i in 0..self.num_blocks {
                    if !skip_set.contains(&i) {
                        connection_norms[i] = Self::frobenius_norm(&a_blocks[i]);
                    }
                }
            }
            GaugeBlockUpdateMethod::GrassmannExp => {
                let tangent_blocks = tangent_blocks
                    .as_ref()
                    .expect("grassmann path should retain tangent blocks");
                for i in 0..self.num_blocks {
                    if !skip_set.contains(&i) {
                        connection_norms[i] = Self::tangent_connection_norm(&tangent_blocks[i]);
                    }
                }
            }
        }
        self.last_connection_norms = connection_norms;
        let connection_norms_sec = connection_norms_start.elapsed().as_secs_f64();

        // ===== 6) Apply orthogonal update =====
        let apply_updates_start = Instant::now();
        let update_breakdown = if self.use_block_updates {
            match self.block_update_method {
                GaugeBlockUpdateMethod::AmbientExp => self.apply_block_updates_ambient(
                    state,
                    a_blocks
                        .as_ref()
                        .expect("ambient path should retain A blocks"),
                    &skip_set,
                )?,
                GaugeBlockUpdateMethod::GrassmannExp => self.apply_block_updates_grassmann(
                    state,
                    tangent_blocks
                        .as_ref()
                        .expect("grassmann path should retain tangent blocks"),
                    &skip_set,
                )?,
            }
        } else {
            self.apply_global_update(
                state,
                a_blocks
                    .as_ref()
                    .expect("global updates require ambient A blocks"),
                &skip_set,
            )?
        };
        let apply_updates_sec = apply_updates_start.elapsed().as_secs_f64();

        // ===== 7) Periodic retraction =====
        let used_retraction =
            self.retraction_interval > 0 && (self.global_step % self.retraction_interval == 0);
        let retraction_start = Instant::now();
        if used_retraction {
            state.orthonormalize_all(self.retraction_method);
        }

        self.last_step_profile = GaugeDynamicsStepProfile {
            total_sec: step_start.elapsed().as_secs_f64(),
            connection_build_sec,
            momentum_sec,
            comm_target_sec,
            diagnostics_sec,
            connection_norms_sec,
            apply_updates_sec,
            update_exp_sec: update_breakdown.exp_sec,
            update_state_apply_sec: update_breakdown.state_apply_sec,
            retraction_sec: retraction_start.elapsed().as_secs_f64(),
            used_retraction,
        };

        let mean_commutator = if commutator_norms.is_empty() {
            0.0
        } else {
            commutator_norms.iter().sum::<f64>() / commutator_norms.len() as f64
        };
        let report_norms = commutator_norms.clone();
        self.last_commutator_norms = commutator_norms;

        Ok(GaugeDynamicsStepReport {
            mean_commutator,
            commutator_norms: report_norms,
            profile: self.last_step_profile,
        })
    }

    fn apply_block_updates_ambient(
        &self,
        state: &mut CosmosState,
        a_blocks: &[Array2<f64>],
        skip_set: &HashSet<usize>,
    ) -> Result<ApplyUpdateBreakdown, RulesError> {
        use crate::geometry::matrix_exp::scaled_matrix_exp_antisym;

        let mut breakdown = ApplyUpdateBreakdown::default();
        for i in 0..self.num_blocks {
            if !skip_set.contains(&i) {
                let exp_start = Instant::now();
                let u = scaled_matrix_exp_antisym(&a_blocks[i], -self.lr, self.exp_method)?;
                breakdown.exp_sec += exp_start.elapsed().as_secs_f64();
                let apply_start = Instant::now();
                state.apply_block_left_action(i, &u);
                breakdown.state_apply_sec += apply_start.elapsed().as_secs_f64();
            }
        }
        Ok(breakdown)
    }

    fn apply_block_updates_grassmann(
        &self,
        state: &mut CosmosState,
        tangent_blocks: &[Array2<f64>],
        skip_set: &HashSet<usize>,
    ) -> Result<ApplyUpdateBreakdown, RulesError> {
        use crate::geometry::grassmann::{ExponentialMapConfig, GrassmannError, exponential_map};

        let mut breakdown = ApplyUpdateBreakdown::default();

        // Square blocks take the ambient orthogonal action instead of the
        // Grassmann geodesic. The geodesic map re-projects its tangent
        // onto the horizontal space, which is trivial for `Gr(k, k)`, so
        // it would return every square block unchanged; the frame is
        // evolved by `W <- exp(-lr A) W` with `A = T Wᵀ - W Tᵀ` built
        // from the same tangent, which stays inside `O(k)` exactly.
        let square_blocks: HashSet<usize> = (0..self.num_blocks)
            .filter(|i| {
                !skip_set.contains(i) && state.block_shapes().get(*i).is_some_and(|&(n, k)| n == k)
            })
            .collect();
        if !square_blocks.is_empty() {
            let breakdown_fiber =
                self.apply_block_updates_square_frames(state, tangent_blocks, &square_blocks)?;
            breakdown.exp_sec += breakdown_fiber.exp_sec;
            breakdown.state_apply_sec += breakdown_fiber.state_apply_sec;
        }
        let skip_set = &skip_set
            .union(&square_blocks)
            .copied()
            .collect::<HashSet<_>>();

        // Device path: same geodesic map, executed by the compute
        // device (`ComputeDevice::grassmann_exp` at t = 1, with the
        // step size folded into the uploaded tangent — the projection
        // inside the map is linear, so pre-scaling is equivalent to
        // the host call at t = −lr). State stays host-resident: each
        // block round-trips basis + tangent up, updated basis down.
        if let Some(device) = &self.device {
            use crate::compute::ComputeDevice;
            for i in 0..self.num_blocks {
                if skip_set.contains(&i) {
                    continue;
                }

                let w = state.get_basis(i)?;
                let evolve_start = Instant::now();
                let scaled_tangent = tangent_blocks[i].mapv(|v| -self.lr * v);
                let w_dev = device.upload_2d(w);
                let h_dev = device.upload_2d(&scaled_tangent);
                let w_new_dev = device.grassmann_exp(&w_dev, &h_dev)?;
                let w_new = device.download_2d(&w_new_dev);
                breakdown.exp_sec += evolve_start.elapsed().as_secs_f64();

                let apply_start = Instant::now();
                state.set_block_basis(i, w_new)?;
                breakdown.state_apply_sec += apply_start.elapsed().as_secs_f64();
            }
            return Ok(breakdown);
        }

        // Host path. Each per-block Grassmann exponential is an
        // independent pure map of (basis, tangent); the blocks share no
        // state, so they can be evaluated in any order and the results
        // applied in index order.
        let active: Vec<usize> = (0..self.num_blocks)
            .filter(|i| !skip_set.contains(i))
            .collect();

        // Sequential path (default, `evolve_threads == 1`, or a single
        // active block): the canonical loop, byte-for-byte unchanged.
        if self.evolve_threads <= 1 || active.len() <= 1 {
            for &i in &active {
                let w = state.get_basis(i)?.to_owned();
                let evolve_start = Instant::now();
                let w_new = exponential_map(
                    &w,
                    &tangent_blocks[i],
                    &ExponentialMapConfig { t: -self.lr },
                )?;
                breakdown.exp_sec += evolve_start.elapsed().as_secs_f64();

                let apply_start = Instant::now();
                state.set_block_basis(i, w_new)?;
                breakdown.state_apply_sec += apply_start.elapsed().as_secs_f64();
            }
            return Ok(breakdown);
        }

        // Parallel host path. Snapshot the active bases (a cheap clone)
        // so the worker threads read no shared `state`, then evaluate
        // the independent maps across scoped threads that pull blocks
        // from a shared counter — dynamic balancing for the uneven
        // block sizes (one large input/aggregator block among many
        // small interior blocks). The MKL backend is the sequential
        // build, so each map is single-threaded and stacking them does
        // not oversubscribe BLAS. Results are sorted by block index and
        // applied in order, making the outcome bit-identical to the
        // sequential path regardless of thread scheduling.
        let inputs: Vec<(usize, Array2<f64>)> = active
            .iter()
            .map(|&i| Ok((i, state.get_basis(i)?.to_owned())))
            .collect::<Result<Vec<_>, RulesError>>()?;

        let lr = self.lr;
        let n_threads = self.evolve_threads.min(inputs.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let inputs_ref = &inputs;
        let tangents = tangent_blocks;

        let evolve_start = Instant::now();
        let collected: Result<Vec<(usize, Array2<f64>)>, GrassmannError> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n_threads)
                    .map(|_| {
                        let next_ref = &next;
                        scope.spawn(
                            move || -> Result<Vec<(usize, Array2<f64>)>, GrassmannError> {
                                let mut local = Vec::new();
                                loop {
                                    let idx =
                                        next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if idx >= inputs_ref.len() {
                                        break;
                                    }
                                    let (block_idx, w) = &inputs_ref[idx];
                                    let w_new = exponential_map(
                                        w,
                                        &tangents[*block_idx],
                                        &ExponentialMapConfig { t: -lr },
                                    )?;
                                    local.push((*block_idx, w_new));
                                }
                                Ok(local)
                            },
                        )
                    })
                    .collect();

                let mut out: Vec<(usize, Array2<f64>)> = Vec::with_capacity(inputs_ref.len());
                let mut first_err: Option<GrassmannError> = None;
                for handle in handles {
                    match handle.join().expect("evolution worker thread panicked") {
                        Ok(mut local) => out.append(&mut local),
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
                match first_err {
                    Some(e) => Err(e),
                    None => Ok(out),
                }
            });

        let mut results = collected?;
        breakdown.exp_sec += evolve_start.elapsed().as_secs_f64();

        results.sort_by_key(|(i, _)| *i);
        let apply_start = Instant::now();
        for (i, w_new) in results {
            state.set_block_basis(i, w_new)?;
        }
        breakdown.state_apply_sec += apply_start.elapsed().as_secs_f64();

        Ok(breakdown)
    }

    /// Evolve pure-fiber (square) blocks through the ambient orthogonal
    /// action generated by their tangent directions.
    ///
    /// `W <- exp(-lr A) W` with `A = T Wᵀ - W Tᵀ` is the same generator
    /// the `AmbientExp` path builds; for `T = W·skew(Wᵀ G)` it reduces to
    /// the canonical `O(k)` gradient step `W <- W exp(-2·lr·skew(Wᵀ G))`,
    /// keeping the frame exactly orthogonal.
    fn apply_block_updates_square_frames(
        &self,
        state: &mut CosmosState,
        tangent_blocks: &[Array2<f64>],
        square_blocks: &HashSet<usize>,
    ) -> Result<ApplyUpdateBreakdown, RulesError> {
        use crate::geometry::matrix_exp::scaled_matrix_exp_antisym;

        let mut breakdown = ApplyUpdateBreakdown::default();
        for &i in square_blocks {
            let w = state.get_basis(i)?;
            let a = Self::ambient_from_tangent(w, &tangent_blocks[i]);
            let exp_start = Instant::now();
            let u = scaled_matrix_exp_antisym(&a, -self.lr, self.exp_method)?;
            breakdown.exp_sec += exp_start.elapsed().as_secs_f64();
            let apply_start = Instant::now();
            state.apply_block_left_action(i, &u);
            breakdown.state_apply_sec += apply_start.elapsed().as_secs_f64();
        }
        Ok(breakdown)
    }

    /// Apply global block-diagonal orthogonal update.
    ///
    /// Assembles block-diagonal A*, computes U = exp(-lr * A*), applies globally.
    fn apply_global_update(
        &self,
        state: &mut CosmosState,
        a_blocks: &[Array2<f64>],
        skip_set: &HashSet<usize>,
    ) -> Result<ApplyUpdateBreakdown, RulesError> {
        use crate::geometry::matrix_exp::scaled_matrix_exp_antisym;

        let total_n = state.total_n();
        let mut a_star = Array2::zeros((total_n, total_n));

        let mut offset = 0;
        for (i, &(n, _k)) in state.block_shapes().iter().enumerate() {
            if !skip_set.contains(&i) {
                // Copy A_i into the block-diagonal position
                let block = a_blocks[i].view();
                a_star
                    .slice_mut(ndarray::s![offset..offset + n, offset..offset + n])
                    .assign(&block);
            }
            offset += n;
        }

        // U = exp(-lr * A*)
        let exp_start = Instant::now();
        let u = scaled_matrix_exp_antisym(&a_star, -self.lr, self.exp_method)?;
        let exp_sec = exp_start.elapsed().as_secs_f64();
        let apply_start = Instant::now();
        state.apply_conjugation(&u);
        Ok(ApplyUpdateBreakdown {
            exp_sec,
            state_apply_sec: apply_start.elapsed().as_secs_f64(),
        })
    }

    // ─── Public: Diagnostics & Utilities ─────────────────────────

    /// Number of blocks this evolver manages.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Return runtime timings captured during the most recent `step()`.
    pub fn last_step_profile(&self) -> GaugeDynamicsStepProfile {
        self.last_step_profile
    }

    /// Return a snapshot of the diagnostics from the most recent `step()`.
    ///
    /// Orthogonality errors are computed live from the current state.
    ///
    /// # Panics
    ///
    /// Panics if `state` reports fewer blocks than this evolver manages.
    pub fn get_diagnostics(&self, state: &CosmosState) -> GaugeDynamicsDiagnostics {
        // Compute per-block orthogonality error: ||W^T W - I||_F
        let mut orth_errors = Vec::with_capacity(self.num_blocks);
        let mut sum_sq = 0.0;

        for i in 0..self.num_blocks {
            let w = state.get_basis(i).unwrap();
            let (_n, k) = state.block_shapes()[i];
            let wtw = w.t().dot(w);
            let eye = Array2::<f64>::eye(k);
            let diff = &wtw - &eye;
            let err = diff.mapv(|x| x * x).sum().sqrt();
            sum_sq += err * err;
            orth_errors.push(err);
        }

        let total_orth_error = sum_sq.sqrt();

        // Use cached metrics or compute defaults
        let metrics = self.last_metrics.as_ref();

        GaugeDynamicsDiagnostics {
            connection_norms: self.last_connection_norms.clone(),
            commutator_norms: self.last_commutator_norms.clone(),
            commutator_targets: self.last_commutator_targets.clone(),
            geom_error: self.last_geom_error.clone(),
            deadband_mask: self.last_deadband_mask.clone(),
            orthogonality_errors: orth_errors,
            total_orth_error,
            global_step: self.global_step,
            lr: self.lr,
            mean_comm_norm: metrics.map_or(0.0, |m| m.mean_comm_norm),
            mean_comm_target: metrics.map_or(0.0, |m| m.mean_comm_target),
            mean_geom_error: metrics.map_or(0.0, |m| m.mean_geom_error),
            deadband_fraction: metrics.map_or(0.0, |m| m.deadband_fraction),
        }
    }

    /// Compute commutator norms from the current state and gradients, *without* running a step.
    ///
    /// Useful for monitoring convergence between optimization steps.
    pub fn compute_commutator_norms_now(
        &self,
        state: &CosmosState,
        gradients: &[Option<&Array2<f64>>],
    ) -> Vec<f64> {
        let skip_set = HashSet::new();
        match self.block_update_method {
            GaugeBlockUpdateMethod::AmbientExp => {
                let (_a_blocks, norms) = self.compute_connections(state, gradients, &skip_set);
                norms
            }
            GaugeBlockUpdateMethod::GrassmannExp => {
                let (_tangent_blocks, norms) =
                    self.compute_tangent_blocks(state, gradients, &skip_set);
                norms
            }
        }
    }

    /// Reset internal state for a new block count.
    ///
    /// Call this when the number of blocks in the CosmosState changes
    /// (e.g., after architecture modification). Preserves all config,
    /// resets momentum buffers, EMA targets, and cached diagnostics.
    pub fn refresh_for_new_blocks(&mut self, num_blocks: usize) {
        self.num_blocks = num_blocks;
        self.connection_momentum = vec![None; num_blocks];
        self.comm_target_ema = vec![0.0; num_blocks];
        self.last_connection_norms = vec![0.0; num_blocks];
        self.last_commutator_norms = vec![0.0; num_blocks];
        self.last_commutator_targets = vec![0.0; num_blocks];
        self.last_geom_error = vec![0.0; num_blocks];
        self.last_deadband_mask = vec![false; num_blocks];
        self.last_metrics = None;
        self.last_step_profile = GaugeDynamicsStepProfile::default();
        // Note: global_step is NOT reset — the evolver's age persists
    }
}

// ─── Display ──────────────────────────────────────────────────────

impl fmt::Display for GaugeDynamics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GaugeDynamics(lr={}, momentum={}, exp_method={:?}, block_update_method={:?}, \
            blocks={}, step={})",
            self.lr,
            self.momentum,
            self.exp_method,
            self.block_update_method,
            self.num_blocks,
            self.global_step,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
