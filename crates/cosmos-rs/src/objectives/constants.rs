//! Named constants for COSMOS objective hyperparameters.
//!
//! Using named constants instead of magic numbers improves readability
//! and provides a single source of truth for default values.

use std::f64::consts::PI;

/// Default margin for geodesic distance losses (π/4 ≈ 0.785 radians = 45°).
///
/// Represents the minimum angular separation between active and inactive
/// class subspaces on the Grassmann manifold.
///
/// - 0° = no separation (degenerate)
/// - 45° = moderate separation (default)
/// - 90° = maximum separation (orthogonal subspaces)
pub const DEFAULT_MARGIN: f64 = PI / 4.0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn default_margin_is_pi_over_4() {
        assert!((DEFAULT_MARGIN - PI / 4.0).abs() < 1e-15);
    }
}
