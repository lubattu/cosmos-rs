//! Parameter-free output heads over state-owned subspaces.
//!
//! The released crate ships the geometric Class-Membership-Region family: a
//! head reads an embedding against anchor subspaces held in the State and
//! returns per-class scores or angles. It stores nothing and learns nothing.

pub mod geometric_heads;

pub use geometric_heads::{
    AnchorGeometricLogitHead, GeometricLogitHead, GeometricScoreMode,
    differential_log_energy_ratio, differential_log_energy_ratio_batched,
    quadratic_energies_batched, quadratic_energies_from_anchors, quadratic_energies_from_blocks,
    quadratic_energy, rank1_distances_batched, rank1_distances_from_anchors,
    rank1_distances_from_blocks, rank1_lift_distance,
};
