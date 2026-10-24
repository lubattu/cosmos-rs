# Contributing to cosmos-rs

Thank you for considering a contribution. This project has a narrow scope and a strict
review standard; reading this page before opening a pull request will save you time.

## SAOR contracts

The four SAOR contracts are the architecture of COSMOS. They are not style preferences:
they are non-negotiable review criteria.

1. Everything that learns lives in the State (no learnable quantity outside `CosmosState`).
2. The Arrangement is gradient-free (no trainable tensors in topology/routing/input/head code).
3. Objectives evaluate; they do not act (no state mutation from `objectives`).
4. Relevant updates are global and coherent via gauge (no Euclidean descent on bases;
   invariants preserved by construction; compatibility diagnostics exposed).

Contributions that violate any of these contracts are closed with a pointer to this
section, without further negotiation.

Contracts 1 to 3 are encoded mechanically by the `contracts.rs` integration tests. A CI
failure on those tests is an automatic rejection, not a discussion.

## Standing rules

**No external GPU or ML frameworks.** Numerics stay in-crate. `deny.toml` bans `cudarc`,
`cust`, `tch`, `candle-core`, `burn`, `ort` and `wgpu`; a pull request that introduces one
of them fails `cargo deny check` and is closed.

**Every public number must be reproducible.** Any number proposed for the
documentation, the README or a paper table must be produced on CPU in `f64`, with the
three canonical seeds

```text
319635789
1489023746
2874510913
```

and the corresponding run JSONs attached to the pull request.

**Contributed experiments arrive as a proposal first.** Open an `experiment_proposal`
issue describing the hypothesis and the design. Once it is accepted, open a pull request
containing the config TOML, the run artifacts and the result CSV — a manifest-style
contribution, not a narrative.

## Development setup

Install the toolchain declared by `rust-toolchain.toml` (stable, with `rustfmt` and
`clippy`), then the two external check tools:

```text
cargo install --locked typos-cli
cargo install --locked cargo-deny
```

`mdbook` is only needed once the book is added and is not required here.

You do not need the GitHub CLI (`gh`) to contribute. Cloning over HTTPS or SSH, pushing a
branch and opening a pull request in the browser is the whole workflow. The maintainer
uses `gh` for repository administration only.

`.cargo/config.toml` is ignored by git, so you can keep a personal one without it ever
being committed. Use it to set `build.target-dir` to a path outside the repository if you
want build output redirected elsewhere, or set `CARGO_TARGET_DIR` in your environment
instead. Without either, `cargo build` creates `target/` in the repository root as usual.

Before every push:

```text
cargo run-checks
```

It runs, in order, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
-- -D warnings`, `typos`, `cargo deny check`, `cargo test --workspace`, and
`cargo doc --workspace --no-deps` with `RUSTDOCFLAGS=-D warnings`. It must be green.

## Process rules

- **Commits follow Conventional Commits**: `type(scope): description`. Allowed types:
  `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `build`, `ci`, `style`,
  `revert`. Scope is optional. Use a single subject line in the imperative mood, lowercase
  after the type, no trailing period, under 72 characters. Do not add a body unless the
  reason is genuinely non-obvious.
- **No attribution trailers.** No `Co-Authored-By`, no generated-with lines, no tool or
  author signature of any kind, in commits or pull-request descriptions.
- **Small pull requests, linked to an issue.** One logical change per pull request. Open
  the issue first; unlinked pull requests are triaged last.
- **`cargo run-checks` must be green** before you request review.
- **Update `CHANGELOG.md`** under `## [Unreleased]` whenever public behavior changes. See
  [RELEASING.md](RELEASING.md) for how a version of this workspace is released.
- **No new dependency without an issue.** Propose it, state what it replaces, and wait for
  a decision before adding it to a manifest.
- **Stale pull requests are closed after 30 days** without a response from the author.
  Reopening is fine once you are ready to continue.

## Code style

- English everywhere: code, comments, doc comments, commit messages, issues.
- Comments explain premises and invariants, never what the code does or why an edit was
  made. Public items carry doc comments describing the contract.
- No commented-out code, no `TODO` or `FIXME` markers, no `#[allow(dead_code)]`.
- Configuration field names follow the article's notation.

## Review expectations

This project has a single maintainer. Review latency is measured in days, not hours, and
can be longer during heavy review periods. A slow response is not a rejection; a closed
pull request with a pointer to a rule is.

Changes are reviewed and merged directly in this repository: every commit on `main`
arrives through a pull request opened here.

The open-problem catalogue and the wider research programme live on the project's future
book site, not in this repository's issue tracker. Issues here are about the library, the
reproduction of the article, and the tooling.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security issues go
to [SECURITY.md](SECURITY.md), never to the public tracker.
