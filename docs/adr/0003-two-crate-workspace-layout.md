# ADR-0003: Two-crate workspace, library published, paper driver not

Status: Accepted
Date: 2026-09-19

## Context

The repository has to serve two audiences that want opposite things. A library user wants
a small dependency that compiles fast and exposes a stable API. A reviewer reproducing
the article wants dataset downloads, experiment drivers, a command-line interface and
report generation. Putting both in one crate would force the library user to compile the
experiment machinery, and would tie the library's release cadence to changes in the
experiment driver.

## Decision

A Cargo workspace with `crates/cosmos-rs` (`publish = true`, the library) and
`crates/cosmos-paper` (`publish = false`, the reproduction driver), plus an `xtask` crate
for repository checks. `cargo add cosmos-rs` installs only the library; the driver is
reachable only by cloning the repository.

## Consequences

The published crate stays lean and its dependency list stays auditable, while the driver
is free to take heavy dependencies for datasets and reporting. Shared metadata and lints
live once in `[workspace.package]` and `[workspace.lints]`. The cost is that the driver
is never versioned or archived on crates.io, so its provenance depends on git tags and
release assets instead.
