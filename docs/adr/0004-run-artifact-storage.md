# ADR-0004: Configs and result tables in the repository, run JSONs in releases

Status: Accepted
Date: 2026-09-19

## Context

Reproducing the article requires three kinds of artifact: the hyperparameters, the raw
per-run output, and the aggregated tables. The raw per-run JSON output is large and
numerous; committing it would bloat every clone forever and would make git
history dominated by machine output. Omitting them entirely would leave the published
numbers unverifiable.

## Decision

The repository holds the config TOMLs under `configs/paper` and the small aggregated CSV
tables under `results/paper`. The config TOMLs are the only source of hyperparameters:
no experiment reads a value that is not in one of them. Full run JSONs are attached to
GitHub Release assets and archived with a Zenodo DOI per release.

## Consequences

A clone stays small and every hyperparameter is diffable and reviewable. Anyone can check
a published table against the CSV in the tree, and can go down to per-seed evidence by
downloading the release asset. The cost is that the raw evidence lives outside git, so a
release must not be deleted, and `docs/paper/provenance.md` has to map every table to its
config, its command, its CSV row and its archived artifact.
