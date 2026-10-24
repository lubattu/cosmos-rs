<p align="center">
  <img src="assets/cosmos-logo.png" alt="COSMOS" width="220">
</p>

# cosmos-rs

*Learning as gauge dynamics on the Grassmannian — a reference implementation in Rust.*

[![crates.io](https://img.shields.io/crates/v/cosmos-rs.svg)](https://crates.io/crates/cosmos-rs)
[![docs.rs](https://docs.rs/cosmos-rs/badge.svg)](https://docs.rs/cosmos-rs)
[![CI](https://github.com/ryuuft/cosmos-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ryuuft/cosmos-rs/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/rustc-1.85%2B-orange.svg)](https://github.com/ryuuft/cosmos-rs)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**COSMOS — Continuous Subspace Manifold Operating System** is a learning framework in
which every learnable quantity is an orthonormal subspace basis and learning is gauge
dynamics on the Grassmannian rather than Euclidean descent on weights. A parameter-free
Arrangement routes signals, Objectives evaluate configurations without mutating them, and
Rules move the State by global, coherent gauge updates that preserve orthonormality by
construction. `cosmos-rs` is the reference Rust implementation of the framework and of
every experiment reported in the accompanying article.

## Status

Version 0.1.0. The library provides `state`, `geometry`, `compute`,
`arrangement`, `objectives`, `rules`, `models`, `training`, `diagnostics`,
`metrics` and `checkpoint`, plus a curated `prelude`; the flat and
patch-wise classification models with the four interior topologies; gauge
dynamics; and the checkpoint round trip. Four runnable examples cover the
library end to end, listed under [Getting started](#getting-started). The
reproduction driver for the article is described under
[Reproducing the article](#reproducing-the-article).

The public API follows Semantic Versioning from this version onward. While
the major version is zero, a minor bump may change the API; a patch bump
will not.

## Repository layout

| Path | Contents |
|---|---|
| `crates/cosmos-rs` | the published library: state, geometry, compute, arrangement, objectives, rules, models, training, diagnostics, metrics, checkpoint |
| `crates/cosmos-rs/examples` | four runnable examples, from a synthetic quickstart to bringing your own data |
| `crates/cosmos-paper` | the reproduction driver for the article's experiments (not published to crates.io) |
| `xtask` | the `cargo run-checks` runner: format, clippy, typos, dependency audit, tests, docs |
| `configs/paper` | one TOML per experiment cell; the only source of hyperparameters |
| `results/paper` | the small CSV tables that reproduce the article's tables |
| `scripts/paper` | portable launchers, resident-set pollers and the figure scripts |
| `docs/book` | the guide |
| `docs/adr` | architecture decision records |
| `docs/paper` | provenance of every published item, and the errata |

## Getting started

Add the library to a project:

```toml
[dependencies]
cosmos-rs = "0.1"
```

The library uses `f64` throughout. The default feature set is pure Rust;
the optional `lapack` feature links a generic BLAS/LAPACK for the dense decompositions.

Building a model and its update rule takes a few lines:

```rust
use cosmos_rs::prelude::*;

let config = CosmosModelConfig::new(64, 4)
    .with_input_subspace_dim(16)
    .with_interior_subspace_dim(8)
    .with_num_interior_blocks(2)
    .with_cmr_k(4)
    .with_interior_topology(InteriorTopology::Series);
let model = CosmosModel::new(config)?;

let rules = GaugeDynamics::new(
    model.get_state().num_blocks(),
    GaugeDynamicsConfig { lr: 0.01, momentum: 0.9, ..Default::default() },
)?;

assert_eq!(model.get_state().num_blocks(), 1 + 2 + 4);
```

Runnable examples ship under `crates/cosmos-rs/examples`, each covering a
different corner of the library end to end:

```text
cargo run --release --example quickstart_synthetic
cargo run --release --example flat_classifier
cargo run --release --example patchwise_classifier
cargo run --release --example custom_data
```

## Documentation

The guide lives under [`docs/book`](docs/book): concepts, a getting-started
walkthrough, how to bring your own data, how to reproduce the article, and
the contribution rules. Build it locally with `mdbook build docs/book` and
open `docs/book/book/index.html`. The API reference is on
[docs.rs](https://docs.rs/cosmos-rs). Planned extensions and their order are
in [`ROADMAP.md`](ROADMAP.md).

## Reproducing the article

The repository contains a reproduction driver, `crates/cosmos-paper`, that
runs every experiment reported in the COSMOS article from a configuration
file under `configs/paper/`. It is not published to the registry; clone the
repository to use it.

```text
cargo run -p cosmos-paper --release -- fetch fashion-mnist
cargo run -p cosmos-paper --release -- e3 --smoke
cargo run -p cosmos-paper --release -- e3
cargo run -p cosmos-paper --release -- report
```

`--smoke` runs one seed for a few epochs and checks the result against
documented expectations before a long run is started. Datasets are cached
under `data/` and verified against pinned SHA-256 digests; run artifacts go
to `runs/`. Both directories are ignored by git.

`docs/paper/provenance.md` maps every table and figure to its configuration,
its command and its output. `docs/paper/errata.md` records the differences
between the published text and this repository.

## Citation

If you use COSMOS in academic work, cite the article:

```bibtex
@article{cosmos2026,
  author  = {Batt{\'u}, Lucas},
  title   = {COSMOS: Continuous Subspace Manifold Operating System},
  journal = {Neural Computing and Applications},
  year    = {2026},
  note    = {In press. Manuscript NCAA-D-26-00622R2}
}
```

Machine-readable metadata lives in [CITATION.cff](CITATION.cff).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. The four SAOR contracts are hard acceptance
rules: contributions that violate them are closed without further negotiation. See
[RELEASING.md](RELEASING.md) for how a version of this workspace is released.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
