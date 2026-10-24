//! Colour-dataset and patch-wise cells.

use crate::cli::RunArgs;
use crate::experiments::{Outcome, run_cells};

/// Cell files the command runs when no explicit cell file is given, in order.
pub const CELLS: [&str; 5] = [
    "e5_cifar_flat.toml",
    "e5_patchwise_cifar.toml",
    "e5_patchwise_fashion.toml",
    "e5_patchwise_no_activation.toml",
    "e5_cifar_margin.toml",
];

/// Run the colour-dataset and patch-wise cells.
///
/// # Errors
///
/// Returns a description when a cell cannot be read or a run fails.
pub fn run(args: &RunArgs) -> Result<Outcome, String> {
    if args.config.is_some() {
        return run_cells(args, CELLS[0]);
    }
    let mut outcome = Outcome::default();
    for cell in CELLS {
        outcome.absorb(run_cells(args, cell)?);
    }
    Ok(outcome)
}
