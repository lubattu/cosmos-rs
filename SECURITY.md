# Security policy

## Supported versions

Only the latest released version of `cosmos-rs` receives security fixes. Until v0.1.0 is
tagged, only the `main` branch is supported.

## Reporting a vulnerability

Do not open a public issue for a security problem.

Report it privately by email to **lucasdryuu@gmail.com** with the subject line
`cosmos-rs security report`. Include the affected version or commit, a description of the
issue, and a reproduction if you have one.

You will receive an acknowledgement within 7 days and a status update within 30 days.

## Disclosure

Coordinated disclosure with a 90-day deadline: a fix is released and the report is
published within 90 days of the acknowledgement, or earlier if a fix ships sooner. If the
deadline is reached without a fix, the reporter is free to disclose.

## Scope

`cosmos-rs` is a numerical library and a reproduction driver. It performs no network
access other than the dataset downloads in `cosmos-paper`, which will verify SHA-256
pins. Reports about dataset download integrity, deserialization of checkpoint files,
and dependency advisories are in scope.
