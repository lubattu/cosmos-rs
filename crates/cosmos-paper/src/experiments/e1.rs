//! Invariant preservation across retraction intervals.

use crate::cli::RunArgs;
use crate::experiments::{Outcome, run_cells};

/// Run the cell.
///
/// # Errors
///
/// Returns a description when the cell cannot be read or a run fails.
pub fn run(args: &RunArgs) -> Result<Outcome, String> {
    run_cells(args, "e1.toml")
}
