//! COSMOS — Continuous Subspace Manifold Operating System.
//!
//! COSMOS treats learning as gauge dynamics on the Grassmannian. The State holds
//! every learnable quantity as orthonormal subspace bases. The Arrangement routes
//! signals and carries no trainable tensors. The Objectives evaluate a configuration
//! without mutating it. The Rules move the State through global, coherent gauge
//! updates that preserve orthonormality by construction.
//!
//! This crate is the reference implementation accompanying the article
//! "COSMOS: Continuous Subspace Manifold Operating System" (Neural Computing and
//! Applications, 2026).
//!
//! Four contracts hold across the whole crate (SAOR, after State, Arrangement,
//! Objectives, Rules): everything that learns lives in the State, so no other
//! module owns a trainable tensor; the Arrangement is gradient-free, routing
//! activations without ever holding a parameter; Objectives evaluate a
//! configuration and return a value, never mutating the State or any other
//! input; and Rules update the State only through orthogonal conjugations,
//! which preserve orthonormality by construction. `tests/contracts.rs` in
//! the repository encodes all four mechanically.
//!
//! # Features
//!
//! `default` is pure Rust: the Grassmannian geometry runs entirely on an
//! in-crate QR/SVD implementation, with no external numerical dependency.
//! `lapack` is opt-in and routes the same operations through a BLAS/LAPACK
//! backend; enabling it requires the consumer to also select one of
//! `ndarray-linalg`'s backend features (for example `openblas-system`).
//! `openblas-system` enables `lapack` together with a system OpenBLAS
//! installation in one step.
//!
//! # Example
//!
//! ```
//! use cosmos_rs::prelude::*;
//!
//! let config = CosmosModelConfig::new(64, 4)
//!     .with_input_subspace_dim(16)
//!     .with_interior_subspace_dim(8)
//!     .with_num_interior_blocks(2)
//!     .with_cmr_k(4)
//!     .with_interior_topology(InteriorTopology::Series);
//! let model = CosmosModel::new(config)?;
//!
//! let rules = GaugeDynamics::new(
//!     model.get_state().num_blocks(),
//!     GaugeDynamicsConfig { lr: 0.01, momentum: 0.9, ..Default::default() },
//! )?;
//!
//! assert_eq!(model.get_state().num_blocks(), 1 + 2 + 4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Numerical code indexes several arrays by the same (row, col) and passes one
// parameter per cached tensor through the manual forward/backward kernels.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]

pub mod arrangement;
pub mod checkpoint;
pub mod compute;
pub mod diagnostics;
pub mod geometry;
pub mod metrics;
pub mod models;
pub mod objectives;
pub mod prelude;
pub mod rules;
pub mod state;
pub mod training;

/// Returns the version of this crate, as declared in its `Cargo.toml`.
///
/// # Examples
///
/// ```
/// assert!(!cosmos_rs::version().is_empty());
/// ```
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
