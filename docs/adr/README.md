# Architecture decision records

Each record states the context that forced a decision, the decision taken, and the
consequences the project accepts because of it. Records are immutable: a decision that
changes gets a new record that supersedes the old one.

| # | Title | Status |
|---|---|---|
| [0001](0001-library-scope.md) | Library scope follows the published article | Accepted |
| [0002](0002-cpu-only-backends.md) | CPU-only backends with an optional LAPACK feature | Accepted |
| [0003](0003-two-crate-workspace-layout.md) | Two-crate workspace: library published, paper driver not | Accepted |
| [0004](0004-run-artifact-storage.md) | Configs and result tables in the repository, run JSONs in releases | Accepted |
| [0005](0005-contribution-flow.md) | Every change arrives as a reviewed pull request | Accepted |
| [0006](0006-saor-contracts-as-governance.md) | SAOR contracts are hard acceptance rules | Accepted |

New records start from [0000-template.md](0000-template.md).
