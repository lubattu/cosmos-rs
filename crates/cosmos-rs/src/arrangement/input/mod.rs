//! Input projectors for arrangement pipelines.
//!
//! Input projectors provide the first arrangement-facing transformation over
//! raw features before routing enters the interior/runtime block stack.
//!
//! - [`flat_input`]: direct projection of flat feature vectors through one
//!   state block
//! - [`patchwise`]: non-overlapping patch extraction, shared patch projection,
//!   spatial pooling, and patch aggregation

pub mod flat_input;
pub mod patchwise;

pub use flat_input::FlatInputProjector;
pub use patchwise::{
    PatchAggregationMode, PatchAggregator, PatchProjector, Patchify, PatchifyOutput, PoolingMode,
    SpatialPooling,
};
