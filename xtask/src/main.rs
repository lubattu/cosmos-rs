//! Repository check runner for cosmos-rs.
//!
//! Invoked through the `run-checks` cargo alias declared in `.cargo/config.toml`.
//! `typos` and `cargo-deny` must be installed; see `CONTRIBUTING.md`.

use std::process::{Command, ExitCode};

struct Check {
    name: &'static str,
    program: &'static str,
    args: &'static [&'static str],
    rustdocflags: Option<&'static str>,
}

const CHECKS: &[Check] = &[
    Check {
        name: "format",
        program: "cargo",
        args: &["fmt", "--all", "--", "--check"],
        rustdocflags: None,
    },
    Check {
        name: "clippy",
        program: "cargo",
        args: &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        rustdocflags: None,
    },
    Check {
        name: "typos",
        program: "typos",
        args: &[],
        rustdocflags: None,
    },
    Check {
        name: "deny",
        program: "cargo",
        args: &["deny", "check"],
        rustdocflags: None,
    },
    Check {
        name: "test",
        program: "cargo",
        args: &["test", "--workspace"],
        rustdocflags: None,
    },
    Check {
        name: "doc",
        program: "cargo",
        args: &["doc", "--workspace", "--no-deps"],
        rustdocflags: Some("-D warnings"),
    },
];

fn resolve_program(program: &str) -> String {
    if program == "cargo" {
        std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
    } else {
        program.to_string()
    }
}

fn run(check: &Check) -> bool {
    println!("== {} ==", check.name);

    let mut command = Command::new(resolve_program(check.program));
    command.args(check.args);
    if let Some(flags) = check.rustdocflags {
        command.env("RUSTDOCFLAGS", flags);
    }

    match command.status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            eprintln!("check `{}` failed: {status}", check.name);
            false
        }
        Err(error) => {
            eprintln!("check `{}` could not start: {error}", check.name);
            false
        }
    }
}

fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() != Some("checks") {
        eprintln!("usage: cargo run-checks");
        return ExitCode::FAILURE;
    }

    for check in CHECKS {
        if !run(check) {
            return ExitCode::FAILURE;
        }
    }

    println!("all checks passed");
    ExitCode::SUCCESS
}
