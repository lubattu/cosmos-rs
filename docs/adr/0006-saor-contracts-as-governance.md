# ADR-0006: SAOR contracts are hard acceptance rules

Status: Accepted
Date: 2026-09-19

## Context

COSMOS is defined by four architectural contracts, not by an implementation. A
contribution that adds a learnable tensor outside the State, or that mutates state from an
objective, still compiles and may even improve a benchmark, while destroying the property
the framework exists to demonstrate. A single maintainer cannot afford to relitigate that
boundary in every review.

## Decision

The four SAOR contracts are non-negotiable review criteria, published in
`CONTRIBUTING.md#saor-contracts`: everything that learns lives in the State; the
Arrangement is gradient-free; Objectives evaluate and do not act; relevant updates are
global and coherent via gauge. Contributions that violate them are closed with a pointer
to that section, without further negotiation. Contracts 1 to 3 are encoded mechanically by
the `contracts.rs` integration tests, and a CI failure on them is an automatic rejection.

## Consequences

Review becomes a rule check instead of an argument, which is what makes solo maintenance
viable, and the framework's defining property is protected by tests rather than by
vigilance. The cost is a blunt instrument: a well-intentioned contribution can be closed
on a rule it did not know it was breaking, so the contracts are stated in the contributing
guide, in the issue forms and in the pull-request checklist before any work starts.
