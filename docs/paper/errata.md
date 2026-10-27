# Errata and numerical notes

This file records every place where the published text and this repository
disagree, and every numerical caveat a reader reproducing the article should
know about. It is updated whenever a reproduction finds a discrepancy.

## Default gauge momentum

The article's default-configuration table prints a gauge momentum of 0.7.
Every certified run used 0.9, and 0.9 is what `configs/paper/*.toml` ship.
The printed 0.7 is a transcription error in the table, not a different
experiment: no reported number was produced at 0.7.

## Initialization

Every reported number depends on the initial block bases. This repository
draws them in `crates/cosmos-paper/src/init.rs`: a standard-normal entry
keyed on `(seed, block index, row, column)`, orthonormalized with the
library's QR routine. The library's own `CosmosState::seed_orthonormal` is a
convenience initializer with a different generator and a different
distribution; it is never used by the reproduction driver. The keyed
Gaussian draw is offered as a second library constructor,
`CosmosState::seed_orthonormal_gaussian`, implemented in
`state::keyed_init` and re-exported by the driver's `init` module.

## Linear algebra backend

The article's own runs were produced with a LAPACK-backed build. The
default build in this repository performs its own QR and SVD in pure Rust
instead. Results may differ from a LAPACK-backed build in the last digits,
and that sensitivity is largest in early epochs; converged, end-of-run
numbers are compared against the article within tolerance rather than
expected to match digit for digit. The comparison tolerance for a full
protocol run against the article's tables is plus or minus 1.5 accuracy
points, absolute. `configs/paper/smoke_references.toml` is a separate,
narrower tolerance set: it holds this crate's own shortened-run outputs as
a regression guard against this crate's own prior behavior, not a
comparison to the article. A systematic deviation beyond the article
tolerance is recorded here rather than absorbed by widening it.

On arms that stay at chance accuracy (for example deep series
compositions), early-epoch geometric diagnostics such as the flow ratio
and the CMR margin vary by up to an order of magnitude across otherwise
equivalent builds of the linear-algebra backend, while accuracies agree.
Comparisons on such arms are meaningful only for converged runs and for
the reported aggregates.

## Patch-wise diagnostics

The interior flow ratio and the class-angle diagnostics are available for
the flat arrangement only. No reported number depends on them for the
patch-wise arrangement, which reports accuracy, timing and memory. The
asymmetry is deliberate and is recorded so a reader does not look for
missing fields in a patch-wise artifact.

## Memory figures

Resident-set size is measured outside the process by
`scripts/paper/rss_probe.ps1` and `scripts/paper/rss_probe.sh`, which poll
the operating system while a two-epoch run executes and write their own
`poll_log.csv` and `summary.json`. Run artifacts carry no memory field. The
memory probes are two-epoch runs and are not the accuracy runs; their
durations are not epoch timings.

## Flow diagnostics of the margin-ablation cell

The suite engine records `history.mean_angle_correct`,
`history.mean_angle_wrong_min`, `history.cmr_margin` and
`class_angle_matrix_final` only when `[diagnostics] flow_diagnostics` is
enabled. `configs/paper/e3_margin_ablation.toml` enables it since 0.1.1;
artifacts from earlier runs of the cell do not carry those fields. The
diagnostics are computed on a held-out batch outside the training loop
and do not alter the trajectory: with the flag on and off, the same seed
produces bit-identical accuracy histories.

## Margin ablation: rounded margin literals and backend sensitivity

The article's margin-ablation runs were launched with the margin rounded
to ten digits — `0.6283185307`, `0.3141592654`, `0.1570796327`,
`0.0785398163` — not with π/5, π/10, π/20 and π/40 at full `f64`
precision. `configs/paper/e3_margin_ablation.toml` carries those exact
literals since 0.1.1, because at the two smaller margins the hinge signal
is weak enough that the trajectory is metastable: the epoch at which a run
escapes the chance plateau moves under last-digit changes to the margin or
to the linear-algebra backend, and with the 30-epoch budget a late escape
ends far below a converged one.

With the article's literals and a `lapack` build linked against a
sequential Intel MKL with the `gesvd` driver, this repository reproduces
all twelve runs of the cell bit-identically: `val_acc`, `train_acc`,
`loss` and `mean_commutator` match the article's run artifacts exactly at
every epoch, and the committed `results/paper/e3_margin_ablation.csv` is
from that re-run.

The default pure-Rust backend reproduces the two larger margins and not
the two smaller ones: 79.38 ± 0.57 against the article's 79.25 ± 0.60 at
π/10, 75.64 ± 0.98 against 77.60 ± 2.07 at π/5 (inside the article arm's
own seed spread), but only 40.41 ± 18.85 and 39.68 ± 13.94 against
74.00 ± 1.82 and 68.75 ± 4.61 at π/20 and π/40, where every seed escapes
the plateau late or not at all within the 30 epochs. The deviation is a
property of the metastable small-margin regime, not of either backend's
correctness; it is recorded here per this file's rule rather than
absorbed by widening the tolerance.
