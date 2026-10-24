//! Comparison baselines and the margin ablation.

use crate::cli::RunArgs;
use crate::experiments::{Outcome, run_cells};

/// Cell files the command runs when no explicit cell file is given, in order.
pub const CELLS: [&str; 2] = ["e3_baselines.toml", "e3_margin_ablation.toml"];

/// Run the comparison baselines and, when no explicit cell file is given, the
/// margin ablation as well.
///
/// # Errors
///
/// Returns a description when a cell cannot be read or a run fails.
pub fn run(args: &RunArgs) -> Result<Outcome, String> {
    let mut outcome = run_cells(args, CELLS[0])?;
    if args.config.is_none() {
        outcome.absorb(run_cells(args, CELLS[1])?);
    }
    Ok(outcome)
}
