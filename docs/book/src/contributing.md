# Contributing

Read `CONTRIBUTING.md` in the repository for the full text. This chapter is
the short form, and the parts of it that are unusual.

## The contracts are acceptance rules

The four SAOR contracts are not style preferences. A contribution that
violates one is closed with a pointer to the contracts section, without
further negotiation:

1. Everything that learns lives in the State.
2. The Arrangement is gradient-free.
3. Objectives evaluate; they do not act.
4. Relevant updates are global and coherent via gauge.

The integration tests encode the first three mechanically. A failure there
is an automatic rejection, not a discussion.

## Evidence rules for numbers

Any number proposed for the documentation, the readme or a table must be
produced on the CPU in `f64`, with the three canonical seeds, and the run
artifacts must be attached. Screening results are candidates, never
citations.

A contributed experiment arrives as an experiment-proposal issue first, and
then as a pull request containing the configuration file, the run
artifacts and the result table — in that order, so that the protocol is
agreed before the compute is spent.

## Dependencies

No external machine-learning framework, and no external accelerator
library. The numerics stay in the crate; the dependency audit enforces the
ban. A new dependency of any kind needs an issue first, with the
alternative-considered written down.

## Process

- Conventional Commits, single subject line, no attribution trailers.
- Small pull requests, each linked to an issue that was discussed before
  the work started.
- `cargo run-checks` green locally: format, clippy with warnings denied,
  spelling, dependency audit, tests, documentation.
- The changelog is updated under `## [Unreleased]` when public behaviour
  changes, and this book is updated when public behaviour changes visibly.
- `main` is protected: pull request required, checks required, linear
  history.
- Releases follow SemVer. A tagged release publishes the library after a
  manual approval and archives the release for citation.

## One maintainer

This project has a single maintainer. Review latency is measured in days,
not hours, and a pull request that sits unanswered for a while has not been
ignored. Pull requests with no response from their author for thirty days
are closed; reopening one is a comment away.

The open-problem catalogue and the wider research conversation do not live
in this repository's issue tracker. Issues here are about this code.
