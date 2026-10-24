//! Construction and execution of a single training run.
//!
//! Premises:
//! - Every learnable quantity lives in the state, so a run is fully
//!   determined by the state's initial bases, the rule configuration and
//!   the batch order.
//! - The two engines differ only in data protocol and result assembly; the
//!   construction in this module is shared.
//! - A stem is unique only inside one sweep coordinate, so an artifact path
//!   always carries the coordinate directory.

pub mod observer;
pub mod suite;
pub mod table;

use std::fmt;
use std::path::{Path, PathBuf};

use cosmos_rs::geometry::matrix_exp::{OrthoMethod, RetractionMethod};
use cosmos_rs::geometry::svd::set_lapack_svd_driver;
use cosmos_rs::prelude::{
    Activation, ActivationPlacement, ClassificationLoss, CosmosModel, CosmosModelConfig,
    CosmosModelPatchwise, GaugeBlockUpdateMethod, GaugeDynamics, GaugeDynamicsConfig,
    InteriorTopology, ModelError, PatchwiseModelConfig, SubspaceAction,
};

use crate::config::{Cell, svd_driver_from_str};
use crate::init::seed_state;
use crate::schema::MaxNorm;

/// Reason a run could not be constructed or executed.
#[derive(Debug)]
pub enum EngineError {
    /// A configured name is not one the library offers.
    Unknown {
        /// Which field carried the name.
        field: &'static str,
        /// The name that was configured.
        value: String,
    },
    /// The library rejected the construction.
    Model(String),
    /// The library rejected the run.
    Train(String),
    /// An artifact could not be written.
    Io(std::io::Error),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { field, value } => write!(f, "unknown {field}: {value}"),
            Self::Model(reason) => write!(f, "model construction failed: {reason}"),
            Self::Train(reason) => write!(f, "run failed: {reason}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ModelError> for EngineError {
    fn from(value: ModelError) -> Self {
        Self::Model(value.to_string())
    }
}

/// Path of a run's artifact relative to the output root.
///
/// The stem alone is not unique inside a sweep, so the coordinate directory
/// is part of the path and not an optional grouping.
pub fn relative_artifact_path(cell: &Cell) -> String {
    format!(
        "{}/{}/{}.json",
        cell.experiment,
        cell.coordinate(),
        cell.stem()
    )
}

/// Absolute path a run's artifact is written to under `out_dir`.
///
/// Mirrors [`relative_artifact_path`], including the mandatory coordinate
/// directory.
pub fn artifact_path(cell: &Cell, out_dir: &Path) -> PathBuf {
    out_dir
        .join(&cell.experiment)
        .join(cell.coordinate())
        .join(format!("{}.json", cell.stem()))
}

/// Map a configured topology name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_topology(value: &str) -> Result<InteriorTopology, EngineError> {
    match value {
        "series" => Ok(InteriorTopology::Series),
        "parallel" => Ok(InteriorTopology::Parallel),
        "parallel_series" => Ok(InteriorTopology::ParallelSeries),
        "residual" => Ok(InteriorTopology::Residual),
        other => Err(EngineError::Unknown {
            field: "topology",
            value: other.to_string(),
        }),
    }
}

/// Map a configured nonlinearity name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_activation(value: &str) -> Result<Activation, EngineError> {
    match value {
        "none" | "identity" => Ok(Activation::Identity),
        "relu" => Ok(Activation::Relu),
        "leaky_relu" => Ok(Activation::LeakyRelu),
        "tanh" => Ok(Activation::Tanh),
        other => Err(EngineError::Unknown {
            field: "activation",
            value: other.to_string(),
        }),
    }
}

/// Map a configured placement name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_placement(value: &str) -> Result<ActivationPlacement, EngineError> {
    match value {
        "none" => Ok(ActivationPlacement::None),
        "post" => Ok(ActivationPlacement::Post),
        "subspace" => Ok(ActivationPlacement::Subspace),
        other => Err(EngineError::Unknown {
            field: "activation_placement",
            value: other.to_string(),
        }),
    }
}

/// Map a configured orthogonal-update method name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_exp_method(value: &str) -> Result<OrthoMethod, EngineError> {
    match value {
        "qr" => Ok(OrthoMethod::Qr),
        "taylor" => Ok(OrthoMethod::Taylor),
        "pade" => Ok(OrthoMethod::Pade),
        "cayley" => Ok(OrthoMethod::Cayley),
        other => Err(EngineError::Unknown {
            field: "exp_method",
            value: other.to_string(),
        }),
    }
}

/// Map a configured retraction method name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_retraction(value: &str) -> Result<RetractionMethod, EngineError> {
    match value {
        "qr" => Ok(RetractionMethod::Qr),
        "svd" => Ok(RetractionMethod::Svd),
        "polar" => Ok(RetractionMethod::Polar),
        other => Err(EngineError::Unknown {
            field: "retraction_method",
            value: other.to_string(),
        }),
    }
}

/// Map a configured block update method name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_update_method(value: &str) -> Result<GaugeBlockUpdateMethod, EngineError> {
    match value {
        "grassmann" => Ok(GaugeBlockUpdateMethod::GrassmannExp),
        "ambient" => Ok(GaugeBlockUpdateMethod::AmbientExp),
        other => Err(EngineError::Unknown {
            field: "update_method",
            value: other.to_string(),
        }),
    }
}

/// Map a configured subspace action name.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a name the library does not offer.
pub fn parse_action(value: &str) -> Result<SubspaceAction, EngineError> {
    match value {
        "projection" => Ok(SubspaceAction::projection()),
        "reflection" => Ok(SubspaceAction::reflection()),
        "identity" => Ok(SubspaceAction::identity()),
        other => Err(EngineError::Unknown {
            field: "action",
            value: other.to_string(),
        }),
    }
}

/// Build the supervised loss a cell declares.
///
/// # Errors
///
/// Returns [`EngineError::Unknown`] for a loss name the library does not
/// offer.
pub fn build_loss(cell: &Cell) -> Result<ClassificationLoss, EngineError> {
    match cell.objective.loss.as_str() {
        "geodesic_hinge" => Ok(ClassificationLoss::GeodesicHinge {
            margin: cell.objective.cmr_margin,
        }),
        "direct_membership" => Ok(ClassificationLoss::DirectMembership),
        other => Err(EngineError::Unknown {
            field: "loss",
            value: other.to_string(),
        }),
    }
}

/// Build a flat model with the cell's initial bases installed.
///
/// # Errors
///
/// Returns [`EngineError`] when a configured name is unknown or the library
/// rejects the configuration.
pub fn build_flat_model(cell: &Cell, input_dim: usize) -> Result<CosmosModel, EngineError> {
    let config = CosmosModelConfig::new(input_dim, cell.model.num_classes)
        .with_input_subspace_dim(cell.model.input_subspace_dim)
        .with_interior_subspace_dim(cell.model.interior_subspace_dim)
        .with_num_interior_blocks(cell.model.num_interior_blocks)
        .with_cmr_k(cell.model.cmr_k)
        .with_anchors_per_class(cell.model.anchors_per_class)
        .with_interior_topology(parse_topology(&cell.model.topology)?)
        .with_activation(
            parse_activation(&cell.model.activation)?,
            parse_placement(&cell.model.activation_placement)?,
        )
        .with_subspace_action(parse_action(&cell.model.action)?);
    let mut model = CosmosModel::new(config)?;
    seed_state(model.get_state_mut(), cell.seed)
        .map_err(|err| EngineError::Model(err.to_string()))?;
    Ok(model)
}

/// Build a patch-wise model with the cell's initial bases installed.
///
/// # Errors
///
/// Returns [`EngineError`] when a configured name is unknown, a patch-wise
/// field is absent, or the library rejects the configuration.
pub fn build_patchwise_model(cell: &Cell) -> Result<CosmosModelPatchwise, EngineError> {
    let shape = cell.model.image_shape.ok_or(EngineError::Unknown {
        field: "model.image_shape",
        value: "absent".to_string(),
    })?;
    let config = PatchwiseModelConfig::new((shape[0], shape[1], shape[2]), cell.model.num_classes)
        .with_patch_size(cell.model.patch_size.unwrap_or(4))
        .with_patch_subspace_dim(cell.model.patch_subspace_dim.unwrap_or(16))
        .with_agg_subspace_dim(cell.model.agg_subspace_dim.unwrap_or(256))
        .with_interior_subspace_dim(cell.model.interior_subspace_dim)
        .with_num_interior_blocks(cell.model.num_interior_blocks)
        .with_cmr_k(cell.model.cmr_k)
        .with_anchors_per_class(cell.model.anchors_per_class)
        .with_interior_topology(parse_topology(&cell.model.topology)?)
        .with_activation(
            parse_activation(&cell.model.activation)?,
            parse_placement(&cell.model.activation_placement)?,
        );
    let mut model = CosmosModelPatchwise::new(config)?;
    seed_state(model.get_state_mut(), cell.seed)
        .map_err(|err| EngineError::Model(err.to_string()))?;
    Ok(model)
}

/// Build the gauge dynamics rule a cell declares.
///
/// # Errors
///
/// Returns [`EngineError`] when a configured name is unknown or the library
/// rejects the configuration.
pub fn build_rules(cell: &Cell, num_blocks: usize) -> Result<GaugeDynamics, EngineError> {
    let driver =
        svd_driver_from_str(&cell.rules.svd_driver).ok_or_else(|| EngineError::Unknown {
            field: "rules.svd_driver",
            value: cell.rules.svd_driver.clone(),
        })?;
    set_lapack_svd_driver(driver);
    let config = GaugeDynamicsConfig {
        lr: cell.rules.lr_gauge,
        momentum: cell.rules.momentum,
        momentum_max_norm: MaxNorm(cell.rules.momentum_max_norm).as_option(),
        exp_method: parse_exp_method(&cell.rules.exp_method)?,
        block_update_method: parse_update_method(&cell.rules.update_method)?,
        retraction_interval: cell.rules.retraction_interval,
        retraction_method: parse_retraction(&cell.rules.retraction_method)?,
        ..GaugeDynamicsConfig::default()
    };
    GaugeDynamics::new(num_blocks, config).map_err(|err| EngineError::Model(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    #[test]
    fn unconstrained_ceiling_maps_to_none() {
        let mut cell = crate::config::tests_support::sample_cell();
        cell.rules.momentum_max_norm = 0.0;
        assert_eq!(MaxNorm(cell.rules.momentum_max_norm).as_option(), None);
        let rules = build_rules(&cell, 3).expect("rules");
        assert_eq!(rules.num_blocks(), 3);

        cell.rules.momentum_max_norm = 1.2;
        assert_eq!(MaxNorm(cell.rules.momentum_max_norm).as_option(), Some(1.2));
        assert!(build_rules(&cell, 3).is_ok());
    }

    #[test]
    fn unknown_topology_is_rejected_by_name() {
        let err = parse_topology("spiral").expect_err("unknown");
        assert!(err.to_string().contains("spiral"));
    }

    #[test]
    fn hinge_margin_travels_into_the_loss() {
        let cell = crate::config::tests_support::sample_cell();
        match build_loss(&cell).expect("loss") {
            cosmos_rs::prelude::ClassificationLoss::GeodesicHinge { margin } => {
                assert!((margin - std::f64::consts::PI / 5.0).abs() < 1e-12);
            }
            other => panic!("unexpected loss {other:?}"),
        }
    }

    #[test]
    fn two_coordinates_never_share_an_artifact_path() {
        let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper");
        let out = PathBuf::from("runs");
        let mut checked = 0;
        for entry in std::fs::read_dir(&config_dir).expect("config dir") {
            let path = entry.expect("entry").path();
            let name = path.file_name().and_then(|name| name.to_str());
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml")
                || name == Some("bench.toml")
                || name == Some("smoke_references.toml")
            {
                continue;
            }
            let file = crate::config::load(&path).expect("cell file");
            let cells = crate::config::expand(&file, None, false).expect("expand");
            let mut paths: HashSet<PathBuf> = HashSet::new();
            for cell in &cells {
                assert!(
                    paths.insert(artifact_path(cell, &out)),
                    "{}: repeated artifact path",
                    path.display()
                );
                assert!(
                    artifact_path(cell, &out).ends_with(relative_artifact_path(cell)),
                    "relative and absolute paths disagree"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 11);

        let file = crate::config::load(&config_dir.join("e2.toml")).expect("cell file");
        let cells = crate::config::expand(&file, None, false).expect("expand");
        let stems: HashSet<String> = cells.iter().map(Cell::stem).collect();
        assert!(
            stems.len() < cells.len(),
            "the coordinate directory is what separates these runs"
        );
    }
}
