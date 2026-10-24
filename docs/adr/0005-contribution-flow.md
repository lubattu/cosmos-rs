# ADR-0005: Every change arrives as a reviewed pull request

Status: Accepted
Date: 2026-09-19

## Context

The repository serves both as a library and as the evidence behind published numbers, so
its history has to stay readable and every commit has to be attributable to a review. A
single maintainer pushing directly to `main` makes the history cheap to produce and
impossible to audit, and it gives contributors no visible standard to aim at.

## Decision

`main` is the only long-lived branch, and it advances only through pull requests. Work
happens on a topic branch, opens a pull request against `main`, and merges once
`cargo run-checks` is green and the SAOR contracts in `CONTRIBUTING.md` are satisfied.
The maintainer's own changes follow the same route.

Experiments follow the proposal-then-pull-request flow: an experiment proposal issue
states the question, the configuration and the acceptance criteria and is agreed before
the work starts; the pull request that follows carries the config, the aggregated table
and the code the run needed.

## Consequences

Every commit on `main` has a pull request that records why it exists and what was checked,
so the history reads as a product history and the review bar is public rather than
implicit. The cost is latency: a single maintainer reviewing everything means changes
wait, and trivial fixes pay the same ceremony as substantial ones. Proposals agreed before
the work starts keep that cost from falling on a contributor after the fact.
