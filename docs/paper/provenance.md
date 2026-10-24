# Provenance

Every table and figure of the article maps to a configuration file, a
command and an output. Run everything from the repository root.

Dataset files are fetched on first use into `data/` and verified against
pinned SHA-256 digests. To fetch them ahead of time:

```
cargo run -p cosmos-paper --release -- fetch fashion-mnist
cargo run -p cosmos-paper --release -- fetch cifar10
```

Add `--no-download` to any experiment command to make a missing dataset an
error instead of a download.

| Article item | Configuration | Command | Output |
|---|---|---|---|
| Invariant preservation table and figure | `configs/paper/e1.toml` | `cargo run -p cosmos-paper --release -- e1` | `runs/e1/**/run_r*_seed_*.json`, `results/paper/e1.csv` |
| Gauge regime figures and prose figures | `configs/paper/e2.toml` | `cargo run -p cosmos-paper --release -- e2` | `runs/e2/**/run_r10_seed_*.json`, `results/paper/e2.csv` (from `report`), `results/paper/e2_metrics.csv` (from `scripts/paper/figures/e2_analysis.py --out results/paper`) |
| Comparison baselines table | `configs/paper/e3_baselines.toml` | `cargo run -p cosmos-paper --release -- e3` | `runs/e3_baselines/**/*.json`, `results/paper/e3_baselines.csv` |
| Margin ablation | `configs/paper/e3_margin_ablation.toml` | `cargo run -p cosmos-paper --release -- e3` | `runs/e3_margin_ablation/**/*.json`, `results/paper/e3_margin_ablation.csv` |
| Topology robustness table and figure | `configs/paper/e4_topology.toml` | `cargo run -p cosmos-paper --release -- e4` | `runs/e4_topology/**/*.json`, `results/paper/e4_topology.csv` |
| Companion control | `configs/paper/e4_companion.toml` | `cargo run -p cosmos-paper --release -- e4` | `runs/e4_companion/**/*.json`, `results/paper/e4_companion.csv` |
| Colour-dataset flat baselines | `configs/paper/e5_cifar_flat.toml` | `cargo run -p cosmos-paper --release -- e5` | `runs/e5_cifar_flat/**/*.json`, `results/paper/e5_cifar_flat.csv` |
| Patch-wise colour results | `configs/paper/e5_patchwise_cifar.toml` | `cargo run -p cosmos-paper --release -- e5` | `runs/e5_patchwise_cifar/**/*.json`, `results/paper/e5_patchwise_cifar.csv` |
| Patch-wise grayscale results | `configs/paper/e5_patchwise_fashion.toml` | `cargo run -p cosmos-paper --release -- e5` | `runs/e5_patchwise_fashion/**/*.json`, `results/paper/e5_patchwise_fashion.csv` |
| Patch-wise ablation without subspace nonlinearity | `configs/paper/e5_patchwise_no_activation.toml` | `cargo run -p cosmos-paper --release -- e5` | `runs/e5_patchwise_no_activation/**/*.json` |
| Colour-dataset margin sweep (one seed) | `configs/paper/e5_cifar_margin.toml` | `cargo run -p cosmos-paper --release -- e5` | `runs/e5_cifar_margin/**/*.json` |
| Orthogonal update cost table | `configs/paper/bench.toml` | `cargo run -p cosmos-paper --release -- bench` | `runs/bench/results.json`, `runs/bench/report.html` |
| Memory figures | none | `scripts/paper/rss_probe.ps1` or `.sh` around a two-epoch run | `runs/rss/<probe>/poll_log.csv`, `summary.json` |

Tables are produced from finished artifacts with:

```
cargo run -p cosmos-paper --release -- report
```

Figures are produced from the same artifacts with the scripts in
`scripts/paper/figures/`; install their pinned requirements first.

## Sanity probe before a long run

Every cell has a shortened form that runs one seed for a few epochs and
compares itself against `configs/paper/smoke_references.toml`:

```
cargo run -p cosmos-paper --release -- e1 --smoke
```

The benchmark has the same shortened form, which measures the smaller
ambient dimensions and checks its structural expectations:

```
cargo run -p cosmos-paper --release -- bench --smoke
```

`configs/paper/smoke_references.toml` holds this crate's own shortened-run
outputs and serves as a regression guard against this crate's own prior
behavior; it is not a comparison against the article's numbers. The command
exits non-zero and prints every quantity outside its own tolerance. Run it
before committing to a multi-hour cell.

### Reproduced tables

- `results/paper/e3_baselines.csv` — three seeds, three arms, 30 epochs,
  produced with
  `cargo run -p cosmos-paper --release -- e3 --config configs/paper/e3_baselines.toml --no-download --out runs`
  followed by
  `cargo run -p cosmos-paper --release -- report --runs runs --out results/paper`,
  on 2026-09-19. All three means fell within the article's own reported
  spread; see "Certifying a number against the article" for the rule. Each
  table's grouping key includes every swept configuration axis, and its
  `runs` column counts the rows aggregated into that group.

`history.projection_ratio_mean` is the per-epoch mean of the per-stage
projection ratios recorded across a run's interior blocks; it is not itself
broken out per block.

## Shortened-run results

Every cell was run in its shortened form from the repository root, on the
default numerical backend, with `--no-download` against datasets fetched
beforehand. The values those runs produced are the entries now in
`configs/paper/smoke_references.toml`; re-running each cell against them
reports no deviation. The table records what each cell is compared on.

| Cell | Epochs | Seed | Compared on |
|---|---|---|---|
| `e1.toml` | 3 | 319635789 | `val_acc`, `train_acc`, `loss`, `mean_commutator`, orthogonality bound |
| `e2.toml` | 3 | 319635789 | `val_acc`, `train_acc`, `mean_commutator`, orthogonality bound |
| `e3_baselines.toml` | 3 | 319635789 | `train_acc`, `val_acc`, `param_count` for each of the three arms, and `test_acc` for the first |
| `e4_topology.toml` | 3 | 319635789 | `val_acc`, `flow_ratio_overall`, `cmr_margin`, orthogonality bound |
| `e5_patchwise_cifar.toml` | 3 | 1489023746 | `train_acc`, `val_acc`, `test_acc`, `param_count` |
| `e5_cifar_flat.toml` | 3 | 1489023746 | chance-level ceiling on `val_acc` and `param_count`, first arm only |
| `bench.toml` | — | — | orthogonality bound on every method, a fitted slope per series, per-block time not falling with the ambient dimension |

The topology cell sits at chance by construction; what the reference holds
it to is the flow ratio, the angular margin and the orthogonality bound.
The colour cells take a few minutes each even in their shortened form.

The five cells with no entry of their own — `e3_margin_ablation.toml`,
`e4_companion.toml`, `e5_patchwise_fashion.toml`,
`e5_patchwise_no_activation.toml` and `e5_cifar_margin.toml` — print
`no reference for <cell>; structural checks only` and are checked by the
run completing and its artifact validating.

An entry whose cell file carries more than one arm is keyed as
`<cell>:<arm>`, so it is matched to that arm alone; a bare cell name is
matched to every arm of the file.

## Certifying a number against the article

Article numbers are certified only by full protocol runs, not by the
shortened form above. `results/paper/*.csv` are produced by the `report`
command from full runs. The comparison tolerance between a full run's table
and the article's tables is plus or minus 1.5 accuracy points, absolute. See
`docs/paper/errata.md` for the numerical-backend note behind that tolerance:
the article's runs used a LAPACK-backed build, this crate's default build is
pure Rust, and early-epoch values are sensitive to last-digit differences
while converged results are compared within tolerance.

## Numerical backend of the reference artifacts

The article's run artifacts were computed with the `lapack` feature linked
against a sequential Intel MKL and the `gesvd` driver. Each run JSON records
its driver as `svd_driver` in the config block. The default pure-Rust build
selects no driver and reproduces the tables within the tolerances held in
`configs/paper/smoke_references.toml`.

## Reference artifacts

The archived run artifacts of the article are attached to the corresponding
release, together with the tables in `results/paper/`. The repository
itself holds configurations and small tables only; full artifacts are too
large to version.

## One seed, stated plainly

The colour-dataset margin sweep was run at a single seed. Its configuration
says so (`seeds = [319635789]`), and any number derived from it carries no
spread.

## Figures scheduled for a later release

The margin-evolution and inter-class-angle figures are generated in a later
release. The per-epoch fields they read are already recorded, so no
experiment needs re-running. See `docs/paper/errata.md`.

## Checking an artifact

Any run artifact, single file or whole directory, can be checked against
the schema this crate writes:

```
cargo run -p cosmos-paper --release -- validate runs
```

The command prints one line per file and exits non-zero when a file does
not match. Files that are not run artifacts are reported as skipped.
`--require-artifacts` additionally fails when nothing at all was
recognized, which is the form to use on an archive that is supposed to
contain runs.
