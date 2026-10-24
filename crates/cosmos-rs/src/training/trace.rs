//! Forward traces retained for explicit backward implementations.
//!
//! These structs intentionally store owned arrays for now because the first
//! manual backward path will target CPU execution. If future performance work
//! requires buffer reuse or borrowing, this module can evolve without changing
//! the outer trainer/task interfaces.

use ndarray::Array2;

/// Trace for the input projection stage.
#[derive(Debug, Clone, PartialEq)]
pub struct InputProjectionTrace {
    /// State block used by the input projector.
    pub block_idx: usize,
    /// Batch input before projection.
    pub input: Array2<f64>,
    /// Projected batch output.
    pub output: Array2<f64>,
}

/// Selection and ordering metadata for one interior block.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BlockSelectionTrace {
    /// Candidate block ids considered by the block runtime.
    pub candidate_indices: Vec<usize>,
    /// Final selected block ids after selection.
    pub selected_indices: Vec<usize>,
    /// Ordered block ids after ordering.
    pub ordered_indices: Vec<usize>,
}

/// Forward cache for one modular interior block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockTrace {
    /// Selection/order bookkeeping for this block.
    pub selection: BlockSelectionTrace,
    /// Input entering the block.
    pub input: Array2<f64>,
    /// Output produced by the block.
    pub output: Array2<f64>,
}

/// Trace for the full interior runtime.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InteriorTrace {
    /// Batch entering the interior.
    pub input: Option<Array2<f64>>,
    /// Batch leaving the interior.
    pub output: Option<Array2<f64>>,
    /// Per-block forward traces.
    pub blocks: Vec<BlockTrace>,
}

/// Trace for CMR scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct CmrTrace {
    /// Embeddings entering the CMR scorer.
    pub embeddings: Array2<f64>,
    /// Final class scores.
    pub scores: Array2<f64>,
    /// Winning anchor indices per sample/class when multi-anchor scoring is used.
    pub winning_anchor_indices: Option<Array2<usize>>,
}

/// Top-level trace retained for one batch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchTrace {
    /// Optional input projection trace.
    pub input_projection: Option<InputProjectionTrace>,
    /// Optional interior runtime trace.
    pub interior: Option<InteriorTrace>,
    /// Optional CMR scoring trace.
    pub cmr: Option<CmrTrace>,
    /// Optional scalar loss captured by the adapter.
    pub scalar_loss: Option<f64>,
}
