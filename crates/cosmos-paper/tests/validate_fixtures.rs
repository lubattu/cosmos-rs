//! Fixture-driven checks of the artifact validator.

use std::path::Path;

use cosmos_paper::validate::{Kind, Outcome, validate_file, validate_path};

#[test]
fn a_well_formed_suite_artifact_validates() {
    let path = Path::new("tests/fixtures/validate/valid_suite_run.json");
    match validate_file(path) {
        Outcome::Valid(Kind::SuiteRun) => {}
        other => panic!("expected a valid full-train run, got {other:?}"),
    }
}

#[test]
fn an_unknown_field_is_rejected() {
    let path = Path::new("tests/fixtures/validate/unknown_field_run.json");
    match validate_file(path) {
        Outcome::Invalid(message) => {
            assert!(
                message.contains("telemetry_channel"),
                "the report must name the offending key, got: {message}"
            );
        }
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn an_unrecognized_engine_is_rejected() {
    let path = Path::new("tests/fixtures/validate/unknown_engine_run.json");
    match validate_file(path) {
        Outcome::Invalid(message) => {
            assert!(
                message.contains("engine"),
                "the report must name the missing or unrecognized engine, got: {message}"
            );
        }
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn a_directory_walk_reports_one_entry_per_file() {
    let report = validate_path(Path::new("tests/fixtures/validate"), false).expect("walk");
    assert_eq!(report.entries.len(), 3);
    assert_eq!(report.valid, 1);
    assert_eq!(report.invalid, 2);
    assert_eq!(report.skipped, 0);
    assert!(!report.is_success());
}

#[test]
fn requiring_artifacts_fails_on_an_empty_directory() {
    let dir = std::env::temp_dir().join("cosmos-paper-empty-validate");
    std::fs::create_dir_all(&dir).expect("create");
    let report = validate_path(&dir, true).expect("walk");
    assert_eq!(report.entries.len(), 0);
    assert!(!report.is_success());
    std::fs::remove_dir_all(&dir).expect("clean");
}
