# Reproducing the article

The repository contains a second crate, `cosmos-paper`, that reproduces
every experiment reported in the COSMOS article. It is not published to the
registry; clone the repository to use it.

## The contract

Each experiment is a **cell**: a configuration file under `configs/paper/`
that fixes every hyperparameter, the seeds and the epoch count. The
configuration file is the only source of hyperparameters — nothing is
hard-coded in the driver and nothing is passed on the command line except
paths and the seed list.

A cell declares one of two engines:

| Engine | Data protocol | Used by |
|---|---|---|
| full-train | the whole training set, evaluated on the canonical test split each epoch | the invariant, regime-map and topology cells |
| three-way split | train / validation / test, split seed `12345`, trailing partial batch discarded | the comparison-table and colour-dataset cells |

The engine is recorded in every artifact, and the report command refuses to
aggregate a table from artifacts of mixed engines.

## Running a cell

```text
cargo run -p cosmos-paper --release -- fetch fashion-mnist
cargo run -p cosmos-paper --release -- fetch cifar10
cargo run -p cosmos-paper --release -- e3 --smoke
cargo run -p cosmos-paper --release -- e3
cargo run -p cosmos-paper --release -- report
```

Datasets are cached under `data/` and verified against pinned SHA-256
digests before extraction. Run artifacts go to `runs/`. Both directories
are ignored by version control; only configurations and the small result
tables under `results/paper/` are committed.

## Always run the shortened form first

`--smoke` runs one seed for a few epochs and compares the result against
expectations stored in `configs/paper/smoke_references.toml`. It changes no
objective, no loss, no model and no rule — only the number of seeds and
epochs. Run it before starting anything that will take hours:

```text
cargo run -p cosmos-paper --release -- e1 --smoke
```

A deviation outside tolerance is printed with the expected value, the
observed value and the width of the band. Tolerances are stated in the same
file: accuracies are compared within an absolute band, the orthonormality
residual against an absolute upper bound, and other scalars relatively,
with a wider relative band for strictly positive small quantities and an
absolute floor for signed quantities that can cross zero.

The stored expectations are this crate's own shortened-run outputs. They
are a regression guard for the driver, not a restatement of the published
numbers. Certifying a number against the article is done only by a full
run, three canonical seeds, compared with the published table.

## What an artifact contains

One JSON file per run, with the configuration it was built from, the
summary metrics, the per-epoch history and the per-epoch timing. The
full-train engine additionally writes a set summary and an aggregation
sidecar. Artifacts contain no absolute path, no device or precision field,
no thread count and no memory field: the driver is CPU double precision
throughout, and resident-set size is measured outside the process by the
pollers in `scripts/paper/`.

Per-block history vectors other than the aggregate orthonormality residual
are stored flattened, per-epoch major and per-block minor. The analysis
scripts reshape them using the length of the epoch vector. This layout is
part of the artifact contract.

You can check any artifact, or a whole directory of them, against the
schema:

```text
cargo run -p cosmos-paper --release -- validate runs
```

## Archived artifacts

The repository holds configurations and small tables. The full run
artifacts behind the published tables and figures are attached to the
corresponding release as archives, one per cell, with a checksum file and a
manifest that maps each archive to the tables and figures it backs. Each
archive passes `validate`.

## Provenance and errata

`docs/paper/provenance.md` maps every published item to its configuration,
its command and its output. `docs/paper/errata.md` records the differences
between the published text and this repository — the default gauge
momentum, the initialization, the linear-algebra backend, the patch-wise
diagnostic asymmetry, the memory-measurement protocol, and the figures
scheduled for a later release.

Read both before reporting a mismatch.
