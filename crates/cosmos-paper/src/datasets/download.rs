//! Fetch, digest check, gzip and tar extraction.
//!
//! Premises:
//! - An archive is verified against its pinned digest before it is read.
//!   A mirror that serves different bytes is a failure, not a fallback.
//! - Nothing under the data directory is ever committed.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

/// One archive the driver may need to fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Archive {
    /// File name under the dataset directory.
    pub file_name: &'static str,
    /// Absolute source URL.
    pub url: &'static str,
    /// Size of the archive in bytes.
    pub size: u64,
    /// Lowercase hexadecimal SHA-256 of the archive.
    pub sha256: &'static str,
}

/// Reason an archive could not be obtained.
#[derive(Debug)]
pub enum FetchError {
    /// The archive is absent and fetching is disabled.
    Missing {
        /// Archive file name.
        file: String,
        /// Directory that was searched.
        dir: PathBuf,
        /// Command that would obtain it.
        hint: String,
    },
    /// The transfer failed.
    Transfer {
        /// Source URL.
        url: String,
        /// Underlying reason.
        reason: String,
    },
    /// The bytes on disk do not match the pinned digest.
    Digest {
        /// Archive file name.
        file: String,
        /// Digest that was pinned.
        expected: String,
        /// Digest that was computed.
        got: String,
    },
    /// A filesystem operation failed.
    Io(io::Error),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { file, dir, hint } => write!(
                f,
                "{file} is not present in {}; run `{hint}` or drop the file there",
                dir.display()
            ),
            Self::Transfer { url, reason } => write!(f, "could not fetch {url}: {reason}"),
            Self::Digest {
                file,
                expected,
                got,
            } => write!(
                f,
                "{file} has digest {got}, expected {expected}; the file is not the pinned archive"
            ),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for FetchError {}

impl From<io::Error> for FetchError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Lowercase hexadecimal SHA-256 of a byte slice.
pub fn digest_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Ensure `archive` is present under `dir` and matches its pinned digest.
///
/// # Errors
///
/// Returns [`FetchError`] when the archive is absent and `allow_download`
/// is false, when the transfer fails, when the digest does not match, or
/// when a filesystem operation fails.
pub fn ensure_archive(
    archive: &Archive,
    dir: &Path,
    allow_download: bool,
    fetch_hint: &str,
) -> Result<PathBuf, FetchError> {
    fs::create_dir_all(dir)?;
    let path = dir.join(archive.file_name);

    if !path.exists() {
        if !allow_download {
            return Err(FetchError::Missing {
                file: archive.file_name.to_string(),
                dir: dir.to_path_buf(),
                hint: fetch_hint.to_string(),
            });
        }
        println!("fetching {} ({} bytes)", archive.file_name, archive.size);
        let mut body = Vec::with_capacity(archive.size as usize);
        let mut response = ureq::get(archive.url)
            .call()
            .map_err(|err| FetchError::Transfer {
                url: archive.url.to_string(),
                reason: err.to_string(),
            })?;
        response
            .body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|err| FetchError::Transfer {
                url: archive.url.to_string(),
                reason: err.to_string(),
            })?;
        let mut file = fs::File::create(&path)?;
        file.write_all(&body)?;
    }

    let bytes = fs::read(&path)?;
    let got = digest_hex(&bytes);
    if got != archive.sha256 {
        return Err(FetchError::Digest {
            file: archive.file_name.to_string(),
            expected: archive.sha256.to_string(),
            got,
        });
    }
    Ok(path)
}

/// Decompress a gzip archive into memory.
///
/// # Errors
///
/// Returns the underlying I/O error when the stream is not valid gzip.
pub fn gunzip(path: &Path) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

/// Extract a gzipped tar archive under `dest`.
///
/// # Errors
///
/// Returns the underlying I/O error when the stream is not a valid gzipped
/// tar or when a member cannot be written.
pub fn untar_gz(path: &Path, dest: &Path) -> io::Result<()> {
    fs::create_dir_all(dest)?;
    let file = fs::File::open(path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_of_empty_input_is_the_published_value() {
        assert_eq!(
            digest_hex(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn digest_of_abc_is_the_published_value() {
        assert_eq!(
            digest_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
