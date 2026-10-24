//! Diagnostics surface for observational geometry helpers.
//!
//! The diagnostics module stays outside the `(S, A, O, R)` core. It exposes
//! read-only helpers used by experiments and notebook-style analyses.

pub mod geometry;

pub use geometry::{
    BlockGeometryDiagnostics, GeometryDiagnosticsError, compute_block_geometry,
    compute_block_geometry_from_state, snapshot_initial_blocks, snapshot_initial_blocks_from_state,
};
