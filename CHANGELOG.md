# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added

- Cargo workspace with the `cosmos-rs` library, the `cosmos-paper` reproduction driver and
  the `xtask` check runner.
- `cargo run-checks`: format, clippy, typos, dependency audit, tests and documentation.
- Continuous integration on Linux, Windows and macOS for stable and the 1.85 MSRV.
- Governance documents: contributing guide with the SAOR contracts as acceptance rules,
  code of conduct, security policy, issue forms and pull-request template.
- Architecture decision records 0001 to 0006.
- `cosmos-rs` library: `state`, `geometry`, `compute`, `arrangement`, `objectives`,
  `rules`, `models`, `training`, `diagnostics`, `metrics`, `checkpoint` and a curated
  `prelude`.
- Flat and patchwise classification models with the four interior topologies (series,
  parallel, residual, parallel-series) and the geodesic CMR head.
- Gauge dynamics with momentum, momentum clipping and periodic retraction.
- `CosmosState::seed_orthonormal` for deterministic orthonormal seeding of state blocks.
- `CosmosState::seed_orthonormal_gaussian` and `state::keyed_init`, the position-keyed
  Gaussian initializer behind the article's reproduction runs.
- Pure-Rust one-sided Jacobi SVD as the default dense decomposition, with the optional
  `lapack` feature delegating to a generic BLAS/LAPACK instead.
- Selectable LAPACK SVD driver (`gesvd` default, `gesdd` optional), recorded in run
  artifacts.
- Integration tests for the SAOR contracts, the orthogonality invariant and the checkpoint
  round trip.
- Examples: `quickstart_synthetic`, `flat_classifier`, `patchwise_classifier`,
  `custom_data`.
- Self-describing, `f64`-exact checkpoint container: a magic header, a format version, an
  ordered metadata section, named matrices and a trailing FNV-1a-64 checksum. The reader
  accepts the format version this crate writes and rejects any other.
- `InteriorSpec`, `BuilderBlockConfig` and the block parameter structs reject unknown
  fields when deserialized.
- `cosmos-paper`, a reproduction driver for the experiments reported in the
  COSMOS article: two execution engines, dataset acquisition with pinned
  digests, the run-artifact schema, comparison baselines, the
  orthogonal-update cost benchmark, result tables and a self-check that
  runs without any dataset.
- `configs/paper/`, one configuration file per experiment cell, the only
  source of hyperparameters, plus the expectations and tolerances used by
  the shortened form of every cell.
- `scripts/paper/`, launchers for each experiment, external resident-set
  pollers and the figure scripts with pinned requirements.
- `docs/paper/errata.md` and `docs/paper/provenance.md`.
- `results/paper/e3_baselines.csv`, the reproduced comparison table.
- `geometry::matrix_exp::orthogonality_error_columns`, the column-orthonormality
  residual `‖UᵀU − I‖_F`, defined at any shape; `orthogonality_error` documents
  that it measures rows and is only equivalent for square frames.
- `cosmos-paper validate`, which checks run artifacts against the schema the
  driver writes and reports one line per file. The artifact structures reject
  unknown fields.
- A guide covering the concepts, a getting-started walkthrough, bringing your
  own data, reproducing the article and the contribution rules, built in
  continuous integration.

[Unreleased]: https://github.com/ryuuft/cosmos-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ryuuft/cosmos-rs/releases/tag/v0.1.0
