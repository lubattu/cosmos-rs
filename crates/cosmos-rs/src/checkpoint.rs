//! Versioned, self-describing, f64-exact model checkpoints.
//!
//! A checkpoint is a flat container of two ordered sections — UTF-8
//! metadata entries and named `f64` matrices — behind a magic header and a
//! format version. It carries no schema of its own: what a given key or
//! tensor name means is decided by whoever writes it. That keeps the
//! container reusable across model families while letting each caller
//! record the exact configuration its own reconstruction needs.
//!
//! The format is hand-rolled and length-prefixed so the crate acquires no
//! serialization dependency, and every float travels as its raw IEEE-754
//! little-endian bit pattern: a save/load round trip reproduces every
//! tensor bit-for-bit, not to some tolerance.
//!
//! # Byte layout
//!
//! Every integer is unsigned little-endian. Offsets below are relative to
//! the start of the field, since the sections are variable-length.
//!
//! ```text
//! magic            8 bytes    "COSMOSCK"
//! version          u32        format version
//! metadata_count   u64        number of metadata entries
//!   per entry:
//!     key_len      u64        length of the key in bytes
//!     key          key_len    UTF-8, unique within the file
//!     value_len    u64        length of the value in bytes
//!     value        value_len  UTF-8
//! tensor_count     u64        number of tensors
//!   per tensor:
//!     name_len     u64        length of the name in bytes
//!     name         name_len   UTF-8, unique within the file
//!     rows         u64        matrix rows
//!     cols         u64        matrix columns
//!     values       rows*cols*8  f64 little-endian, row-major
//! checksum         u64        FNV-1a over every preceding byte
//! ```
//!
//! The trailing checksum turns truncation, byte-level corruption and a
//! partially written file into a clean load error instead of a plausible
//! but wrong model.
//!
//! [`Checkpoint::load`] and [`Checkpoint::from_bytes`] accept exactly the
//! format version this crate writes, [`CHECKPOINT_VERSION`]; any other
//! value is a [`CheckpointError::UnsupportedVersion`].
//!
//! A [`CosmosState`] round-trips through the metadata keys
//! `state.num_blocks` and `state.block_shapes` plus one tensor per block,
//! named by [`block_tensor_name`].
//!
//! # Scope
//!
//! Checkpoints written here are **inference-grade**: they hold the state a
//! forward pass reads, so a loaded model can be evaluated and sampled from.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use ndarray::Array2;
use thiserror::Error;

use crate::state::cosmos_state::{CosmosState, CosmosStateError};

/// File signature identifying a COSMOS checkpoint.
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"COSMOSCK";

/// On-disk format version written and read by this crate.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Metadata key holding the number of state blocks in the file.
const KEY_NUM_BLOCKS: &str = "state.num_blocks";
/// Metadata key holding the `n x k` shape list of the state blocks.
const KEY_BLOCK_SHAPES: &str = "state.block_shapes";

/// Failures reading or writing a checkpoint.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// A filesystem read or write failed.
    #[error("checkpoint i/o failed for {path}: {source}")]
    Io {
        /// Path the operation targeted.
        path: String,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The file does not start with the checkpoint magic header.
    #[error(
        "{path} is not a COSMOS checkpoint: expected the magic bytes {expected:?} at offset 0, found {found:?}"
    )]
    BadMagic {
        /// Path or origin name of the file.
        path: String,
        /// Rendering of the expected magic bytes.
        expected: String,
        /// Rendering of the bytes actually found.
        found: String,
    },

    /// The file's format version is not the one this build reads.
    #[error(
        "{path} is a COSMOS checkpoint of format version {found}; this build reads format version {expected}"
    )]
    UnsupportedVersion {
        /// Path or origin name of the file.
        path: String,
        /// Format version stored in the file.
        found: u32,
        /// Format version this build reads.
        expected: u32,
    },

    /// The byte stream ended before a length-prefixed field could be read.
    #[error(
        "checkpoint is truncated or corrupt: {context} needs {needed} byte(s) at offset {offset}, but only {available} remain"
    )]
    Truncated {
        /// What was being read when the stream ran out.
        context: &'static str,
        /// Byte offset where the read was attempted.
        offset: usize,
        /// Number of bytes the read needed.
        needed: u64,
        /// Number of bytes actually remaining.
        available: usize,
    },

    /// A string field's bytes are not valid UTF-8.
    #[error("checkpoint is corrupt: {context} is not valid UTF-8")]
    InvalidUtf8 {
        /// What was being decoded.
        context: &'static str,
    },

    /// The trailing checksum does not match the decoded payload.
    #[error(
        "checkpoint is corrupt: checksum mismatch (stored {stored:#018x}, computed {computed:#018x})"
    )]
    ChecksumMismatch {
        /// Checksum recorded in the file.
        stored: u64,
        /// Checksum recomputed from the decoded bytes.
        computed: u64,
    },

    /// A metadata key or tensor name appears more than once.
    #[error("checkpoint is corrupt: duplicate {kind} {name:?}")]
    DuplicateName {
        /// Which section held the duplicate (`"metadata key"` or `"tensor"`).
        kind: &'static str,
        /// The repeated name.
        name: String,
    },

    /// A tensor a caller required is absent from the checkpoint.
    #[error("checkpoint is missing the required tensor {name:?}")]
    MissingTensor {
        /// Name of the missing tensor.
        name: String,
    },

    /// A metadata key a caller required is absent from the checkpoint.
    #[error("checkpoint is missing the required metadata key {key:?}")]
    MissingMetadata {
        /// The missing key.
        key: String,
    },

    /// A metadata value could not be parsed as the type its reader expects.
    #[error("checkpoint metadata key {key:?} holds {value:?}, which is not a valid {expected}")]
    MalformedMetadata {
        /// The offending key.
        key: String,
        /// The stored value that failed to parse.
        value: String,
        /// Description of the expected type.
        expected: &'static str,
    },

    /// The checkpoint's block layout does not match the model it is being
    /// restored into.
    #[error("checkpoint does not match this model: {reason}")]
    StateMismatch {
        /// Description of the mismatch.
        reason: String,
    },

    /// A state-layer operation failed while restoring a checkpoint.
    #[error(transparent)]
    State(#[from] CosmosStateError),
}

/// An in-memory checkpoint: ordered metadata plus named `f64` matrices.
///
/// Insertion order is preserved on write, so a file produced twice from
/// the same model and the same configuration is byte-identical.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    metadata: Vec<(String, String)>,
    tensors: Vec<(String, Array2<f64>)>,
}

impl Checkpoint {
    /// Create an empty checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    // ─── Metadata ─────────────────────────────────────────────────

    /// Record a metadata entry, replacing any previous value for `key`
    /// while keeping the key's original position.
    pub fn put_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        match self.metadata.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.metadata.push((key, value)),
        }
    }

    /// Read a metadata value.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Whether a metadata key is present.
    pub fn has_meta(&self, key: &str) -> bool {
        self.metadata.iter().any(|(k, _)| k == key)
    }

    /// Read a metadata value, failing when the key is absent.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::MissingMetadata`] if `key` is absent.
    pub fn require_meta(&self, key: &str) -> Result<&str, CheckpointError> {
        self.meta(key)
            .ok_or_else(|| CheckpointError::MissingMetadata {
                key: key.to_string(),
            })
    }

    /// Read a metadata value parsed as `T`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::MissingMetadata`] if `key` is absent, or
    /// [`CheckpointError::MalformedMetadata`] if its value does not parse
    /// as `T`.
    pub fn require_meta_parsed<T: std::str::FromStr>(
        &self,
        key: &str,
        expected: &'static str,
    ) -> Result<T, CheckpointError> {
        let raw = self.require_meta(key)?;
        raw.parse::<T>()
            .map_err(|_| CheckpointError::MalformedMetadata {
                key: key.to_string(),
                value: raw.to_string(),
                expected,
            })
    }

    /// Every metadata entry, in write order.
    pub fn meta_entries(&self) -> &[(String, String)] {
        &self.metadata
    }

    // ─── Tensors ──────────────────────────────────────────────────

    /// Record a tensor, replacing any previous tensor of the same name
    /// while keeping the name's original position.
    pub fn put_tensor(&mut self, name: impl Into<String>, tensor: Array2<f64>) {
        let name = name.into();
        match self.tensors.iter_mut().find(|(n, _)| *n == name) {
            Some(entry) => entry.1 = tensor,
            None => self.tensors.push((name, tensor)),
        }
    }

    /// Borrow a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&Array2<f64>> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Borrow a tensor by name, failing when it is absent.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::MissingTensor`] if `name` is absent.
    pub fn require_tensor(&self, name: &str) -> Result<&Array2<f64>, CheckpointError> {
        self.tensor(name)
            .ok_or_else(|| CheckpointError::MissingTensor {
                name: name.to_string(),
            })
    }

    /// Every tensor, in write order.
    pub fn tensors(&self) -> &[(String, Array2<f64>)] {
        &self.tensors
    }

    /// Number of stored tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Total number of stored `f64` values across every tensor.
    pub fn value_count(&self) -> usize {
        self.tensors.iter().map(|(_, t)| t.len()).sum()
    }

    // ─── Encoding ─────────────────────────────────────────────────

    /// Encode the checkpoint into its on-disk byte form.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.value_count() * 8);
        out.extend_from_slice(&CHECKPOINT_MAGIC);
        out.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());

        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            put_bytes(&mut out, key.as_bytes());
            put_bytes(&mut out, value.as_bytes());
        }

        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        for (name, tensor) in &self.tensors {
            put_bytes(&mut out, name.as_bytes());
            out.extend_from_slice(&(tensor.nrows() as u64).to_le_bytes());
            out.extend_from_slice(&(tensor.ncols() as u64).to_le_bytes());
            // `iter` walks logical row-major order whatever the memory
            // layout is, which is the order the reader rebuilds from.
            for value in tensor.iter() {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }

        let checksum = fnv1a64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Write the checkpoint to `path`, creating parent directories.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the parent directories or the
    /// file itself cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|source| CheckpointError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
        }
        fs::write(path, self.to_bytes()).map_err(|source| CheckpointError::Io {
            path: path.display().to_string(),
            source,
        })
    }

    // ─── Decoding ─────────────────────────────────────────────────

    /// Decode a checkpoint from its on-disk byte form.
    ///
    /// `origin` names the source in error messages; pass the file path
    /// when one exists.
    ///
    /// # Errors
    ///
    /// Returns an error when the magic header, format version, checksum,
    /// or any length-prefixed field is invalid or truncated.
    ///
    /// # Panics
    ///
    /// Never panics: the trailing-checksum slice is sized to exactly 8
    /// bytes before conversion.
    pub fn from_bytes(bytes: &[u8], origin: &str) -> Result<Self, CheckpointError> {
        let mut cursor = Cursor::new(bytes);

        let magic = cursor.take(CHECKPOINT_MAGIC.len() as u64, "the magic header")?;
        if magic != CHECKPOINT_MAGIC {
            return Err(CheckpointError::BadMagic {
                path: origin.to_string(),
                expected: describe_bytes(&CHECKPOINT_MAGIC),
                found: describe_bytes(magic),
            });
        }

        let version = cursor.take_u32("the format version")?;
        if version != CHECKPOINT_VERSION {
            return Err(CheckpointError::UnsupportedVersion {
                path: origin.to_string(),
                found: version,
                expected: CHECKPOINT_VERSION,
            });
        }

        // The checksum covers everything before itself, so it can only be
        // verified once the trailer's position is known.
        let body_end = bytes
            .len()
            .checked_sub(8)
            .ok_or(CheckpointError::Truncated {
                context: "the trailing checksum",
                offset: bytes.len(),
                needed: 8,
                available: 0,
            })?;
        let stored = u64::from_le_bytes(bytes[body_end..].try_into().expect("8 bytes"));
        let computed = fnv1a64(&bytes[..body_end]);
        if stored != computed {
            return Err(CheckpointError::ChecksumMismatch { stored, computed });
        }

        let mut checkpoint = Self::new();
        let mut seen_keys = BTreeSet::new();

        let metadata_count = cursor.take_u64("the metadata entry count")?;
        for _ in 0..metadata_count {
            let key = cursor.take_string("a metadata key")?;
            let value = cursor.take_string("a metadata value")?;
            if !seen_keys.insert(key.clone()) {
                return Err(CheckpointError::DuplicateName {
                    kind: "metadata key",
                    name: key,
                });
            }
            checkpoint.metadata.push((key, value));
        }

        let mut seen_names = BTreeSet::new();
        let tensor_count = cursor.take_u64("the tensor count")?;
        for _ in 0..tensor_count {
            let name = cursor.take_string("a tensor name")?;
            let rows = cursor.take_usize("a tensor row count")?;
            let cols = cursor.take_usize("a tensor column count")?;
            let len = rows
                .checked_mul(cols)
                .ok_or_else(|| CheckpointError::StateMismatch {
                    reason: format!("tensor {name:?} declares an unrepresentable {rows}x{cols}"),
                })?;
            let mut values = Vec::with_capacity(len.min(1 << 20));
            for _ in 0..len {
                values.push(cursor.take_f64("a tensor value")?);
            }
            let tensor = Array2::from_shape_vec((rows, cols), values).map_err(|_| {
                CheckpointError::StateMismatch {
                    reason: format!("tensor {name:?} does not fill its declared {rows}x{cols}"),
                }
            })?;
            if !seen_names.insert(name.clone()) {
                return Err(CheckpointError::DuplicateName {
                    kind: "tensor",
                    name,
                });
            }
            checkpoint.tensors.push((name, tensor));
        }

        Ok(checkpoint)
    }

    /// Read a checkpoint from `path`.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, or when
    /// [`Checkpoint::from_bytes`] rejects its contents.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let bytes = fs::read(path).map_err(|source| CheckpointError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes, &path.display().to_string())
    }
}

/// Path helper: insert `stem_suffix` before the extension of `path`.
///
/// Used by callers that write one checkpoint per run out of a single
/// operator-supplied path.
pub fn path_with_stem_suffix(path: &Path, stem_suffix: &str) -> PathBuf {
    let extension = path
        .extension()
        .map(|ext| ext.to_string_lossy().to_string());
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_default();
    let file_name = match extension {
        Some(ext) => format!("{stem}_{stem_suffix}.{ext}"),
        None => format!("{stem}_{stem_suffix}"),
    };
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

// ─── CosmosState round trip ───────────────────────────────────────

/// Write every inference-relevant field of `state` into `checkpoint`.
///
/// The block layout goes into metadata and each block basis `W_i` into its
/// own tensor.
///
/// # Panics
///
/// Never panics: every index iterated is below `state.num_blocks()`, so
/// each `get_basis` lookup succeeds.
pub fn store_state(state: &CosmosState, checkpoint: &mut Checkpoint) {
    checkpoint.put_meta(KEY_NUM_BLOCKS, state.num_blocks().to_string());
    checkpoint.put_meta(KEY_BLOCK_SHAPES, format_block_shapes(state.block_shapes()));

    for idx in 0..state.num_blocks() {
        let basis = state
            .get_basis(idx)
            .expect("block index below num_blocks is in range");
        checkpoint.put_tensor(block_tensor_name(idx), basis.to_owned());
    }
}

/// Restore every field [`store_state`] wrote into an already-shaped state.
///
/// The state must already have the block layout the checkpoint was
/// written from: a differing block count or a differing block shape is a
/// hard error, never a reshape.
///
/// # Errors
///
/// Returns [`CheckpointError::StateMismatch`] for a block-count or
/// block-shape mismatch, and propagates any metadata or tensor lookup
/// failure.
pub fn restore_state(
    state: &mut CosmosState,
    checkpoint: &Checkpoint,
) -> Result<(), CheckpointError> {
    let stored_blocks: usize = checkpoint.require_meta_parsed(KEY_NUM_BLOCKS, "block count")?;
    if stored_blocks != state.num_blocks() {
        return Err(CheckpointError::StateMismatch {
            reason: format!(
                "checkpoint holds {stored_blocks} state block(s), this model has {}",
                state.num_blocks()
            ),
        });
    }

    let stored_shapes = checkpoint.require_meta(KEY_BLOCK_SHAPES)?;
    let model_shapes = format_block_shapes(state.block_shapes());
    if stored_shapes != model_shapes {
        return Err(CheckpointError::StateMismatch {
            reason: format!(
                "block shapes differ — checkpoint [{stored_shapes}], model [{model_shapes}]"
            ),
        });
    }

    for idx in 0..stored_blocks {
        let name = block_tensor_name(idx);
        let basis = checkpoint.require_tensor(&name)?;
        state.set_block_basis(idx, basis.to_owned())?;
    }

    Ok(())
}

/// Tensor name of state block `idx`.
pub fn block_tensor_name(idx: usize) -> String {
    format!("state.block.{idx}")
}

/// Canonical `n x k` rendering of a block shape list.
fn format_block_shapes(shapes: &[(usize, usize)]) -> String {
    shapes
        .iter()
        .map(|&(n, k)| format!("{n}x{k}"))
        .collect::<Vec<_>>()
        .join(",")
}

// ─── Byte helpers ─────────────────────────────────────────────────

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// FNV-1a, 64-bit. Chosen for having no dependencies and no state: the
/// checksum exists to catch truncation and bit rot, not tampering.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Render up to eight bytes for an error message, printable ASCII as
/// characters and everything else as an escape.
fn describe_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(8)
        .map(|&byte| {
            if byte.is_ascii_graphic() {
                (byte as char).to_string()
            } else {
                format!("\\x{byte:02x}")
            }
        })
        .collect()
}

/// Bounds-checked forward reader over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: u64, context: &'static str) -> Result<&'a [u8], CheckpointError> {
        let available = self.bytes.len().saturating_sub(self.offset);
        let len_usize = usize::try_from(len).unwrap_or(usize::MAX);
        if len_usize > available {
            return Err(CheckpointError::Truncated {
                context,
                offset: self.offset,
                needed: len,
                available,
            });
        }
        let slice = &self.bytes[self.offset..self.offset + len_usize];
        self.offset += len_usize;
        Ok(slice)
    }

    fn take_u32(&mut self, context: &'static str) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(
            self.take(4, context)?.try_into().expect("4 bytes"),
        ))
    }

    fn take_u64(&mut self, context: &'static str) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(
            self.take(8, context)?.try_into().expect("8 bytes"),
        ))
    }

    fn take_usize(&mut self, context: &'static str) -> Result<usize, CheckpointError> {
        let value = self.take_u64(context)?;
        usize::try_from(value).map_err(|_| CheckpointError::Truncated {
            context,
            offset: self.offset,
            needed: value,
            available: self.bytes.len().saturating_sub(self.offset),
        })
    }

    fn take_f64(&mut self, context: &'static str) -> Result<f64, CheckpointError> {
        Ok(f64::from_le_bytes(
            self.take(8, context)?.try_into().expect("8 bytes"),
        ))
    }

    fn take_string(&mut self, context: &'static str) -> Result<String, CheckpointError> {
        let len = self.take_u64(context)?;
        let bytes = self.take(len, context)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| CheckpointError::InvalidUtf8 { context })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> Checkpoint {
        let mut checkpoint = Checkpoint::new();
        checkpoint.put_meta("config.head", "cmr");
        checkpoint.put_meta("provenance.seed", "319635789");
        checkpoint.put_tensor("state.block.0", array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        checkpoint.put_tensor("state.block.1", array![[-0.5, 0.25]]);
        checkpoint
    }

    #[test]
    fn round_trip_preserves_metadata_order_and_tensor_bits() {
        let original = sample();
        let bytes = original.to_bytes();
        let decoded = Checkpoint::from_bytes(&bytes, "memory").unwrap();

        assert_eq!(decoded.meta_entries(), original.meta_entries());
        assert_eq!(decoded.tensor_count(), original.tensor_count());
        for (name, tensor) in original.tensors() {
            let restored = decoded.require_tensor(name).unwrap();
            assert_eq!(restored.dim(), tensor.dim());
            for (a, b) in tensor.iter().zip(restored.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "tensor {name} differs bitwise");
            }
        }
    }

    #[test]
    fn round_trip_is_bit_exact_for_hostile_floats() {
        let hostile = array![[
            f64::MIN_POSITIVE,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            1.0 / 3.0,
            std::f64::consts::PI,
        ]];
        let mut checkpoint = Checkpoint::new();
        checkpoint.put_tensor("t", hostile.clone());
        let decoded = Checkpoint::from_bytes(&checkpoint.to_bytes(), "memory").unwrap();
        let restored = decoded.require_tensor("t").unwrap();
        for (a, b) in hostile.iter().zip(restored.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        assert_eq!(sample().to_bytes(), sample().to_bytes());
    }

    #[test]
    fn a_foreign_file_is_rejected_by_its_magic() {
        let err = Checkpoint::from_bytes(b"NOTACOSMOSFILEXXXXXXXXXXXXXXXXXX", "bogus.ckpt")
            .expect_err("a file without the magic header must be rejected");
        assert!(matches!(err, CheckpointError::BadMagic { .. }), "{err}");
        assert!(err.to_string().contains("bogus.ckpt"), "{err}");
    }

    #[test]
    fn a_future_format_version_is_rejected() {
        let mut bytes = sample().to_bytes();
        bytes[8..12].copy_from_slice(&(CHECKPOINT_VERSION + 1).to_le_bytes());
        let err = Checkpoint::from_bytes(&bytes, "future.ckpt")
            .expect_err("an unknown format version must be rejected");
        assert!(
            matches!(err, CheckpointError::UnsupportedVersion { found, .. } if found == CHECKPOINT_VERSION + 1),
            "{err}"
        );
    }

    #[test]
    fn a_flipped_payload_bit_is_caught_by_the_checksum() {
        let mut bytes = sample().to_bytes();
        let last_value = bytes.len() - 12;
        bytes[last_value] ^= 0x01;
        let err = Checkpoint::from_bytes(&bytes, "corrupt.ckpt")
            .expect_err("a corrupted payload must be rejected");
        assert!(
            matches!(err, CheckpointError::ChecksumMismatch { .. }),
            "{err}"
        );
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let bytes = sample().to_bytes();
        let err = Checkpoint::from_bytes(&bytes[..bytes.len() / 2], "short.ckpt")
            .expect_err("a truncated file must be rejected");
        assert!(
            matches!(
                err,
                CheckpointError::ChecksumMismatch { .. } | CheckpointError::Truncated { .. }
            ),
            "{err}"
        );
    }

    #[test]
    fn an_empty_file_is_rejected_without_panicking() {
        let err = Checkpoint::from_bytes(&[], "empty.ckpt")
            .expect_err("an empty file is not a checkpoint");
        assert!(matches!(err, CheckpointError::Truncated { .. }), "{err}");
    }

    #[test]
    fn a_missing_tensor_names_itself() {
        let err = sample().require_tensor("state.block.9").unwrap_err();
        assert!(err.to_string().contains("state.block.9"), "{err}");
    }

    #[test]
    fn state_round_trip_restores_every_block_bitwise() {
        let mut state = CosmosState::new(&[(6, 3), (4, 4), (4, 2)]).unwrap();
        // Move the blocks off their identity initialization so the
        // comparison is not trivially satisfied.
        let perturbed = state.get_basis(0).unwrap().mapv(|x| x + 0.125);
        state.set_block_basis(0, perturbed).unwrap();

        let mut checkpoint = Checkpoint::new();
        store_state(&state, &mut checkpoint);
        let decoded = Checkpoint::from_bytes(&checkpoint.to_bytes(), "memory").unwrap();

        let mut restored = CosmosState::new(&[(6, 3), (4, 4), (4, 2)]).unwrap();
        restore_state(&mut restored, &decoded).unwrap();

        for idx in 0..state.num_blocks() {
            let before = state.get_basis(idx).unwrap();
            let after = restored.get_basis(idx).unwrap();
            assert_eq!(before.dim(), after.dim());
            for (a, b) in before.iter().zip(after.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "block {idx} differs bitwise");
            }
        }
    }

    #[test]
    fn a_block_count_mismatch_is_a_hard_error() {
        let source = CosmosState::new(&[(4, 2), (4, 2)]).unwrap();
        let mut checkpoint = Checkpoint::new();
        store_state(&source, &mut checkpoint);

        let mut target = CosmosState::new(&[(4, 2)]).unwrap();
        let err = restore_state(&mut target, &checkpoint)
            .expect_err("a differing block count must be rejected");
        assert!(err.to_string().contains("2 state block(s)"), "{err}");
    }

    #[test]
    fn a_block_shape_mismatch_is_a_hard_error() {
        let source = CosmosState::new(&[(6, 3)]).unwrap();
        let mut checkpoint = Checkpoint::new();
        store_state(&source, &mut checkpoint);

        let mut target = CosmosState::new(&[(6, 2)]).unwrap();
        let err = restore_state(&mut target, &checkpoint)
            .expect_err("a differing block shape must be rejected");
        assert!(err.to_string().contains("block shapes differ"), "{err}");
        assert!(err.to_string().contains("6x3"), "{err}");
    }

    #[test]
    fn per_run_paths_keep_the_extension() {
        assert_eq!(
            path_with_stem_suffix(Path::new("runs/model.ckpt"), "seed_7"),
            PathBuf::from("runs").join("model_seed_7.ckpt")
        );
        assert_eq!(
            path_with_stem_suffix(Path::new("model"), "seed_7"),
            PathBuf::from("model_seed_7")
        );
    }
}
