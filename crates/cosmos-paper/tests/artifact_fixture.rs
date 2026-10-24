//! The committed artifact fixture parses into the schema and keeps its
//! key names and its flattened per-block layout across a round trip.

use cosmos_paper::schema::SuiteHistory;

fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("suite_history.json"),
    )
    .expect("fixture");
    serde_json::from_str(&text).expect("fixture parses")
}

#[test]
fn the_fixture_carries_the_expected_top_level_keys() {
    let value = fixture();
    let object = value.as_object().expect("object");
    for key in ["config", "result", "history", "epochs"] {
        assert!(object.contains_key(key), "missing top-level key {key}");
    }
}

#[test]
fn the_history_round_trips_and_stays_flattened() {
    let value = fixture();
    let history: SuiteHistory =
        serde_json::from_value(value["history"].clone()).expect("history parses");
    let epochs = history.epoch.len();
    assert_eq!(epochs, 2);
    assert_eq!(history.per_block.connection_norms_mean.len(), 3 * epochs);
    assert_eq!(history.per_block.total_orth_error_mean.len(), epochs);

    let text = serde_json::to_string(&history).expect("serialize");
    let back: SuiteHistory = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(history, back);

    let reserialized: serde_json::Value = serde_json::from_str(&text).expect("value");
    let original = value["history"].as_object().expect("object");
    for key in original.keys() {
        assert!(
            reserialized.get(key).is_some(),
            "key {key} disappeared across the round trip"
        );
    }
}

#[test]
fn a_missing_required_history_key_fails_to_parse() {
    let value = fixture();
    let mut history = value["history"].clone();
    history.as_object_mut().expect("object").remove("train_acc");
    let result: Result<SuiteHistory, _> = serde_json::from_value(history);
    assert!(result.is_err(), "missing train_acc should not deserialize");
}
