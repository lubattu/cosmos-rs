//! Topology robustness and the companion control.

use crate::cli::RunArgs;
use crate::experiments::{Outcome, run_cells};

/// Cell files the command runs when no explicit cell file is given, in order.
pub const CELLS: [&str; 2] = ["e4_topology.toml", "e4_companion.toml"];

/// Run the topology cell and, when no explicit cell file is given, the
/// companion control as well.
///
/// # Errors
///
/// Returns a description when a cell cannot be read or a run fails.
pub fn run(args: &RunArgs) -> Result<Outcome, String> {
    let mut outcome = run_cells(args, CELLS[0])?;
    if args.config.is_none() {
        outcome.absorb(run_companion(args)?);
    }
    Ok(outcome)
}

/// Run the companion control cell.
///
/// # Errors
///
/// Returns a description when the cell cannot be read or a run fails.
pub fn run_companion(args: &RunArgs) -> Result<Outcome, String> {
    run_cells(args, CELLS[1])
}
