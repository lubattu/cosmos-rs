# ADR-0001: Library scope follows the published article

Status: Accepted
Date: 2026-09-19

## Context

A framework library can be scoped around what its author finds interesting or around what
has been validated in public. The first grows an API surface faster than one maintainer
can keep correct, and lends the credibility of a peer-reviewed result to code that has
never been measured. The COSMOS framework has a published article behind it, which gives
the project a scope boundary that is already defended.

## Decision

The library implements the model family of the NCAA article: the flat and patchwise
classifiers, the geodesic CMR readout, gauge dynamics, the four interior topologies
(series, parallel, residual, parallel-series), and CPU arithmetic in `f64`. A module,
feature or configuration option belongs in the crate when a table, figure or prose number
of a publication depends on it.

The library grows by versioned releases tied to publications: each release states which
results it implements, and new capability enters the crate alongside the work that
validates it.

## Consequences

The public API is small, fully exercised by the reproduction driver and the example
programs, and every public behavior has a certified number behind it. A contributor
proposing a capability is asked which published result it reproduces, which is a friction
cost paid deliberately; the experiment proposal flow in `CONTRIBUTING.md` is the route for
work that does not have one yet. Release notes double as a scope statement, so the
question "what does this version cover" is answered by `CHANGELOG.md` rather than by
reading the module tree.
