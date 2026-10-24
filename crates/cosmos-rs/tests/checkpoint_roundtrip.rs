//! Checkpoint container round trip and version handling.
//!
//! Assumptions:
//! - The container ends in an FNV-1a-64 checksum over everything before it, so
//!   any test that rewrites a header field must rewrite the trailer too;
//!   otherwise the reader rejects the file for the checksum, not the field.

mod common;

use common::{seed_state, small_config};
use cosmos_rs::checkpoint::{CHECKPOINT_VERSION, Checkpoint};
use cosmos_rs::prelude::*;

const SEED: u64 = 1_489_023_746;

fn seeded_model() -> CosmosModel {
    let mut model = CosmosModel::new(small_config(32, 4)).expect("model builds");
    seed_state(model.get_state_mut(), SEED);
    model
}

fn stored(model: &CosmosModel) -> Checkpoint {
    let mut checkpoint = Checkpoint::new();
    store_state(model.get_state(), &mut checkpoint);
    checkpoint
}

/// Rewrite the format version of a serialized checkpoint, keeping the file
/// otherwise valid by recomputing the trailing checksum.
fn with_version(mut bytes: Vec<u8>, version: u32) -> Vec<u8> {
    bytes[8..12].copy_from_slice(&version.to_le_bytes());

    let body_end = bytes.len() - 8;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in &bytes[..body_end] {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    bytes[body_end..].copy_from_slice(&hash.to_le_bytes());
    bytes
}

#[test]
fn a_state_round_trips_through_bytes() {
    let source = seeded_model();
    let bytes = stored(&source).to_bytes();

    let restored_checkpoint =
        Checkpoint::from_bytes(&bytes, "round-trip test").expect("checkpoint parses");
    let mut target = CosmosModel::new(small_config(32, 4)).expect("model builds");
    restore_state(target.get_state_mut(), &restored_checkpoint).expect("state restores");

    for block in 0..source.get_state().num_blocks() {
        let a = source.get_state().get_basis(block).expect("source block");
        let b = target.get_state().get_basis(block).expect("target block");
        assert_eq!(a, b, "block {block} differs after the round trip");
    }
}

#[test]
fn the_written_format_records_its_version() {
    let bytes = stored(&seeded_model()).to_bytes();
    assert_eq!(&bytes[..8], b"COSMOSCK");
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        CHECKPOINT_VERSION
    );
}

#[test]
fn an_unknown_format_version_is_rejected() {
    let bytes = with_version(stored(&seeded_model()).to_bytes(), CHECKPOINT_VERSION + 1);

    let err =
        Checkpoint::from_bytes(&bytes, "version test").expect_err("an unknown version is refused");
    assert!(matches!(err, CheckpointError::UnsupportedVersion { .. }));
}
