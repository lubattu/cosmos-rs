//! IDX reader with header validation.
//!
//! An IDX file begins with a four-byte magic number whose third byte is the
//! element type and whose fourth byte is the number of dimensions, followed
//! by one big-endian `u32` per dimension and then the raw elements. Only
//! unsigned-byte elements occur in the datasets this driver reads.

use std::fmt;

/// Reason an IDX file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdxError {
    /// The file is shorter than its own header.
    Truncated {
        /// Bytes the header requires.
        expected: usize,
        /// Bytes present.
        got: usize,
    },
    /// The magic number does not describe unsigned bytes.
    UnsupportedElementType(u8),
    /// The dimension count is not the one the caller asked for.
    DimensionMismatch {
        /// Dimensions the caller expects.
        expected: u8,
        /// Dimensions declared by the file.
        got: u8,
    },
}

impl fmt::Display for IdxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { expected, got } => {
                write!(
                    f,
                    "IDX file truncated: expected {expected} bytes, got {got}"
                )
            }
            Self::UnsupportedElementType(code) => {
                write!(f, "IDX element type {code:#04x} is not unsigned byte")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "IDX file declares {got} dimensions, expected {expected}")
            }
        }
    }
}

impl std::error::Error for IdxError {}

/// A decoded IDX payload: its dimension sizes and its raw elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdxData {
    /// Size of each dimension, outermost first.
    pub dims: Vec<usize>,
    /// Raw unsigned-byte elements in row-major order.
    pub data: Vec<u8>,
}

/// Decode an IDX byte stream that declares `expected_dims` dimensions.
///
/// # Errors
///
/// Returns [`IdxError`] when the header is truncated, declares a different
/// dimension count, or declares an element type other than unsigned byte.
///
/// # Panics
///
/// Never panics: the truncation check above guarantees each dimension slice
/// is exactly four bytes before it is converted.
pub fn decode(bytes: &[u8], expected_dims: u8) -> Result<IdxData, IdxError> {
    if bytes.len() < 4 {
        return Err(IdxError::Truncated {
            expected: 4,
            got: bytes.len(),
        });
    }
    let element_type = bytes[2];
    if element_type != 0x08 {
        return Err(IdxError::UnsupportedElementType(element_type));
    }
    let dim_count = bytes[3];
    if dim_count != expected_dims {
        return Err(IdxError::DimensionMismatch {
            expected: expected_dims,
            got: dim_count,
        });
    }
    let header_len = 4 + 4 * dim_count as usize;
    if bytes.len() < header_len {
        return Err(IdxError::Truncated {
            expected: header_len,
            got: bytes.len(),
        });
    }

    let mut dims = Vec::with_capacity(dim_count as usize);
    for index in 0..dim_count as usize {
        let start = 4 + 4 * index;
        let raw: [u8; 4] = bytes[start..start + 4].try_into().expect("four bytes");
        dims.push(u32::from_be_bytes(raw) as usize);
    }

    let elements: usize = dims.iter().product();
    let total = header_len + elements;
    if bytes.len() < total {
        return Err(IdxError::Truncated {
            expected: total,
            got: bytes.len(),
        });
    }

    Ok(IdxData {
        dims,
        data: bytes[header_len..total].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_images(count: u32, rows: u32, cols: u32) -> Vec<u8> {
        let mut bytes = vec![0x00, 0x00, 0x08, 0x03];
        bytes.extend_from_slice(&count.to_be_bytes());
        bytes.extend_from_slice(&rows.to_be_bytes());
        bytes.extend_from_slice(&cols.to_be_bytes());
        let elements = (count * rows * cols) as usize;
        bytes.extend((0..elements).map(|index| (index % 251) as u8));
        bytes
    }

    #[test]
    fn decodes_three_dimensional_images() {
        let bytes = synthetic_images(3, 4, 5);
        let decoded = decode(&bytes, 3).expect("decode");
        assert_eq!(decoded.dims, vec![3, 4, 5]);
        assert_eq!(decoded.data.len(), 60);
        assert_eq!(decoded.data[0], 0);
        assert_eq!(decoded.data[59], 59);
    }

    #[test]
    fn decodes_one_dimensional_labels() {
        let mut bytes = vec![0x00, 0x00, 0x08, 0x01];
        bytes.extend_from_slice(&4u32.to_be_bytes());
        bytes.extend_from_slice(&[7, 2, 9, 0]);
        let decoded = decode(&bytes, 1).expect("decode");
        assert_eq!(decoded.dims, vec![4]);
        assert_eq!(decoded.data, vec![7, 2, 9, 0]);
    }

    #[test]
    fn rejects_a_truncated_payload() {
        let mut bytes = synthetic_images(3, 4, 5);
        bytes.truncate(bytes.len() - 1);
        assert_eq!(
            decode(&bytes, 3),
            Err(IdxError::Truncated {
                expected: 76,
                got: 75
            })
        );
    }

    #[test]
    fn rejects_a_wrong_dimension_count() {
        let bytes = synthetic_images(3, 4, 5);
        assert_eq!(
            decode(&bytes, 1),
            Err(IdxError::DimensionMismatch {
                expected: 1,
                got: 3
            })
        );
    }

    #[test]
    fn rejects_a_non_byte_element_type() {
        let mut bytes = synthetic_images(1, 1, 1);
        bytes[2] = 0x0D;
        assert_eq!(
            decode(&bytes, 3),
            Err(IdxError::UnsupportedElementType(0x0D))
        );
    }
}
