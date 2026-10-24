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
distribution; it is never used by the reproduction driver. Offering the
keyed Gaussian draw as a second library constructor is tracked as a separate
request: <https://github.com/ryuuft/cosmos-rs/issues/2>.

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

## Figures generated in a later release

Two figures of the classification-margin experiment — the margin evolution
and the inter-class angle matrix — are generated in a later release. The
per-epoch fields they need are already recorded by the full-train engine:
`history.mean_angle_correct`, `history.mean_angle_wrong_min`,
`history.cmr_margin` and `class_angle_matrix_final`. Nothing about the
experiment needs to be re-run to produce them.
