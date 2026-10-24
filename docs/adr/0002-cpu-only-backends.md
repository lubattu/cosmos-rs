# ADR-0002: CPU-only backends with an optional LAPACK feature

Status: Accepted
Date: 2026-09-19

## Context

The article's protocol is CPU in `f64`; every certified number was produced that way.
Hardware acceleration would make results depend on a driver stack and would pull in the
external GPU and ML framework crates that the project refuses on principle. Separately,
the dense decompositions can either be written in Rust or delegated to a system
BLAS/LAPACK.

## Decision

The crate computes on the CPU in `f64`. The default feature set is empty and pure Rust.
An optional `lapack` feature links a generic BLAS/LAPACK for the dense decompositions; no
vendored or statically linked MKL. That feature's SVD driver is `dgesvd` by default, with
`dgesdd` selectable per run through `[rules] svd_driver`. Hardware acceleration is out of
scope, and `deny.toml` bans external GPU and ML framework crates outright.

## Consequences

The crate builds anywhere Rust builds, with no system dependency by default, and the
Linux, Windows and macOS CI legs test the same code path that produced the article's
numbers. The cost is speed: a large sweep is bounded by the CPU it runs on. Replacing LAPACK
with pure-Rust decompositions may change last-digit results, so the certified tables are
produced with the default features and any delta against the `lapack` feature is recorded.

The two paths also diverge on rank-deficient input: the pure-Rust one-sided Jacobi SVD
leaves the columns of `U` that belong to zero singular values unfilled, while LAPACK
completes them into a full orthonormal basis. Every shipped call site consumes only the
columns that belong to non-zero singular values, so the divergence is inert for the crate
as published, and a future call site that needs the full basis must complete it itself.
