# Releasing cosmos-rs

This page describes how a version of this workspace is released. It is aimed at the
maintainer; contributors do not need it to open a pull request.

## Versioning

The workspace shares a single version across its crates, set once in the root
`Cargo.toml`. Versioning follows [Semantic Versioning](https://semver.org/). While the
major version is zero, a minor bump (`0.Y`) may change the public API; a patch bump
(`0.Y.Z`) does not.

Each version gets exactly one tag, `vX.Y.Z`, annotated, created on a commit of `main`.
Release tags are protected by a repository ruleset: only the repository administrator can
create, update or delete a `v*` tag.

## Before tagging

Run through this checklist before creating a tag:

- `cargo run-checks` is green on `main`.
- The version in the root `Cargo.toml` matches the tag you are about to create.
- `CHANGELOG.md` has its `## [Unreleased]` section renamed to `## [X.Y.Z] - YYYY-MM-DD`,
  and the compare-link definitions at the bottom of the file are updated to match.
- `CITATION.cff` has its `version` field updated to `X.Y.Z`.
- The `crates-io` environment used by the release workflow has its required reviewer
  configured.
- The archive integration (Zenodo) is enabled for the repository, so the tag it reacts to
  produces an archived snapshot.

## Tagging

Create and push the tag from a clean checkout of `main`:

```text
git tag -a vX.Y.Z -m "cosmos-rs vX.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag starts the release workflow (`.github/workflows/release.yml`):

1. **checks** — runs `cargo run-checks`, builds the book with `mdbook build docs/book`,
   and packages the library with `cargo publish --dry-run -p cosmos-rs`.
2. **publish** — confirms the tag matches the version in the manifest, then waits for the
   maintainer to approve the deployment to the `crates-io` environment before running
   `cargo publish -p cosmos-rs`. This is where the maintainer approves the publish.
3. **release** — extracts the notes for this version from `CHANGELOG.md` (the section
   headed `## [X.Y.Z] - YYYY-MM-DD`) and creates the GitHub Release from them. If that
   section is missing, this job fails and no release is created.

## After the workflow

Once the GitHub Release exists:

- Attach the run-artifact archives referenced by `docs/paper/provenance.md`, and their
  `SHA256SUMS` file, to the release:

  ```text
  gh release upload vX.Y.Z <files>
  ```

- Verify the published crate in a fresh project:

  ```text
  cargo add cosmos-rs@X.Y.Z
  ```

- Once the archive service mints a DOI for the release, add it to `CITATION.cff` (under
  `identifiers`) and to the README badge, in a follow-up commit on `main`. The tag itself
  is never moved.

## Pre-releases

Pre-release versions are not published. To try a release candidate, run the checklist and
the workflow steps on a branch instead of tagging `main`.

## Release notes

Release notes come only from the `CHANGELOG.md` section for that version. Keep entries
under `Added`, `Changed` and `Fixed`, and write them as each change lands rather than at
release time, so the `[Unreleased]` section is always ready to be renamed.
