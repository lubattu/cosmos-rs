//! Comparison arms trained on the same data protocol as the COSMOS arm.

pub mod mlp;
pub mod rsgd;

/// Which arm a run trains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The COSMOS arm, trained by gauge dynamics.
    Cosmos,
    /// A two-layer network trained by Adam.
    Mlp,
    /// The same state trained by per-frame Riemannian SGD.
    Rsgd,
}

impl Arm {
    /// Parse an arm name.
    ///
    /// # Errors
    ///
    /// Returns the offending name when it is not one of the three arms.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cosmos" => Ok(Self::Cosmos),
            "mlp" => Ok(Self::Mlp),
            "rsgd" => Ok(Self::Rsgd),
            other => Err(other.to_string()),
        }
    }
}
