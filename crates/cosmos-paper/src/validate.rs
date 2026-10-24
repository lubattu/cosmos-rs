//! Validation of run artifacts against the schema this crate writes.
//!
//! A file is classified by its shape and then deserialized into the
//! matching structure. The structures reject unknown fields, so a file
//! that carries a key this crate does not write is reported as invalid
//! rather than silently accepted.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use walkdir::WalkDir;

use crate::schema::{RetractionAnalysis, SuiteArtifact, SuiteSummary, TableArtifact};

/// Shape a validated file was recognized as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A run on the full-train engine.
    SuiteRun,
    /// A run on the three-way split engine.
    TableRun,
    /// The sidecar describing a finished set of runs.
    SuiteSummary,
    /// The aggregation sidecar of a finished set of runs.
    Analysis,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SuiteRun => "full-train run",
            Self::TableRun => "split run",
            Self::SuiteSummary => "set summary",
            Self::Analysis => "set analysis",
        };
        f.write_str(name)
    }
}

/// Result of validating one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The file matches the named shape.
    Valid(Kind),
    /// The file does not match any shape, with the reason.
    Invalid(String),
    /// The file is not a run artifact, with the reason.
    Skipped(String),
}

/// One line of a validation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path of the file, as the walk found it.
    pub path: PathBuf,
    /// What validating it produced.
    pub outcome: Outcome,
}

/// Counts and lines of a validation run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// One entry per file examined.
    pub entries: Vec<Entry>,
    /// Number of files that matched a shape.
    pub valid: usize,
    /// Number of files that matched none.
    pub invalid: usize,
    /// Number of files that are not run artifacts.
    pub skipped: usize,
    /// Whether the caller demanded at least one artifact.
    pub require_artifacts: bool,
}

impl Report {
    /// Whether the run is a pass.
    ///
    /// A pass has no invalid file and, when artifacts were required, at
    /// least one valid one.
    pub fn is_success(&self) -> bool {
        self.invalid == 0 && !(self.require_artifacts && self.valid == 0)
    }
}

/// Validate a single file.
pub fn validate_file(path: &Path) -> Outcome {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => return Outcome::Invalid(format!("unreadable: {error}")),
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(error) => return Outcome::Invalid(format!("not JSON: {error}")),
    };

    match classify(&value) {
        Classification::Kind(Kind::SuiteRun) => check::<SuiteArtifact>(&text, Kind::SuiteRun),
        Classification::Kind(Kind::TableRun) => check::<TableArtifact>(&text, Kind::TableRun),
        Classification::Kind(Kind::SuiteSummary) => {
            check::<SuiteSummary>(&text, Kind::SuiteSummary)
        }
        Classification::Kind(Kind::Analysis) => {
            check::<Vec<RetractionAnalysis>>(&text, Kind::Analysis)
        }
        Classification::Invalid(reason) => Outcome::Invalid(reason),
        Classification::Skipped => Outcome::Skipped("not a run artifact".to_string()),
    }
}

fn check<T: for<'de> serde::Deserialize<'de>>(text: &str, kind: Kind) -> Outcome {
    match serde_json::from_str::<T>(text) {
        Ok(_) => Outcome::Valid(kind),
        Err(error) => Outcome::Invalid(error.to_string()),
    }
}

/// Result of classifying a value's shape, before schema validation.
enum Classification {
    /// The value matches a known shape.
    Kind(Kind),
    /// The value carries a `config` key that does not name a recognised engine.
    Invalid(String),
    /// The value is not a run artifact.
    Skipped,
}

fn classify(value: &Value) -> Classification {
    if let Some(items) = value.as_array() {
        return match items.first() {
            Some(first) if first.get("retraction_interval").is_some() => {
                Classification::Kind(Kind::Analysis)
            }
            _ => Classification::Skipped,
        };
    }

    let Some(object) = value.as_object() else {
        return Classification::Skipped;
    };
    if object.contains_key("runner_version") && object.contains_key("run_files") {
        return Classification::Kind(Kind::SuiteSummary);
    }

    let Some(config) = object.get("config") else {
        return Classification::Skipped;
    };
    match config.get("engine").and_then(Value::as_str) {
        Some("suite") => Classification::Kind(Kind::SuiteRun),
        Some("table") => Classification::Kind(Kind::TableRun),
        Some(other) => Classification::Invalid(format!("unrecognized config.engine: {other:?}")),
        None => Classification::Invalid("missing config.engine".to_string()),
    }
}

/// Validate every `*.json` file under `path`, or `path` itself when it is
/// a file.
///
/// `require_artifacts` makes an examination that found no valid artifact
/// a failure, which is what a caller checking a distributed archive
/// wants.
///
/// # Errors
///
/// Returns a description when the tree cannot be walked.
pub fn validate_path(path: &Path, require_artifacts: bool) -> Result<Report, String> {
    let mut report = Report {
        require_artifacts,
        ..Report::default()
    };

    if path.is_file() {
        push(&mut report, path.to_path_buf(), validate_file(path));
        return Ok(report);
    }

    for entry in WalkDir::new(path).sort_by_file_name() {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let outcome = validate_file(entry.path());
        push(&mut report, entry.path().to_path_buf(), outcome);
    }

    Ok(report)
}

fn push(report: &mut Report, path: PathBuf, outcome: Outcome) {
    match &outcome {
        Outcome::Valid(_) => report.valid += 1,
        Outcome::Invalid(_) => report.invalid += 1,
        Outcome::Skipped(_) => report.skipped += 1,
    }
    report.entries.push(Entry { path, outcome });
}

/// Print a report, one line per file, then the counts.
pub fn print(report: &Report) {
    for entry in &report.entries {
        let path = entry.path.display();
        match &entry.outcome {
            Outcome::Valid(kind) => println!("ok    {path}  {kind}"),
            Outcome::Invalid(reason) => println!("FAIL  {path}  {reason}"),
            Outcome::Skipped(reason) => println!("skip  {path}  {reason}"),
        }
    }
    println!(
        "{} valid, {} invalid, {} skipped",
        report.valid, report.invalid, report.skipped
    );
}
