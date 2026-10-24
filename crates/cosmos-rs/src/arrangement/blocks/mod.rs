//! Runtime block wrappers for arrangement primitives.
//!
//! This submodule packages the lower-level arrangement primitives into
//! reusable runtime units.
//!
//! - [`BlockConfig`]: immutable configuration for one routing block
//! - [`ExecutionContext`]: runtime inputs shared across a block execution
//! - [`ModularBlock`]: batch-major Select -> Order -> Compose -> Aggregate
//!   pipeline with diagnostics

pub mod modular_block;

pub use modular_block::{BlockConfig, ExecutionContext, ModularBlock};
