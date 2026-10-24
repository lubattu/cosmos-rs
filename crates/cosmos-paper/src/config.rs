//! Experiment cell definitions read from the config directory.
//!
//! Premises:
//! - A config file is the only source of hyperparameters; the CLI selects a
//!   file, seeds, an output directory and a mode.
//! - `engine` is required. It decides the data protocol, and a cell run on
//!   the wrong engine yields plausible but incomparable numbers.
//! - `momentum_max_norm = 0.0` means unconstrained.

use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use cosmos_rs::geometry::svd::LapackSvdDriver;

use crate::datasets::DatasetName;

/// Which engine a cell runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// Full training set, canonical evaluation split used per epoch.
    Suite,
    /// Training, validation and test splits with a fixed split seed.
    Table,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Suite => write!(f, "suite"),
            Self::Table => write!(f, "table"),
        }
    }
}

/// Model section of a cell.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    /// `"flat"` or `"patchwise"`.
    #[serde(default = "default_variant")]
    pub variant: String,
    /// Width of the input subspace.
    #[serde(default)]
    pub input_subspace_dim: usize,
    /// Width of each interior subspace.
    pub interior_subspace_dim: usize,
    /// Number of interior blocks.
    pub num_interior_blocks: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Width of each class anchor.
    pub cmr_k: usize,
    /// Anchors per class.
    #[serde(default = "default_anchors")]
    pub anchors_per_class: usize,
    /// Interior composition topology.
    pub topology: String,
    /// Interior nonlinearity.
    pub activation: String,
    /// Where the nonlinearity is applied.
    pub activation_placement: String,
    /// Subspace action used by the interior blocks.
    #[serde(default = "default_action")]
    pub action: String,
    /// Image shape for the patchwise variant.
    #[serde(default)]
    pub image_shape: Option<[usize; 3]>,
    /// Patch side length for the patchwise variant.
    #[serde(default)]
    pub patch_size: Option<usize>,
    /// Width of each patch subspace.
    #[serde(default)]
    pub patch_subspace_dim: Option<usize>,
    /// Width of the aggregation subspace.
    #[serde(default)]
    pub agg_subspace_dim: Option<usize>,
}

fn default_variant() -> String {
    "flat".to_string()
}

fn default_anchors() -> usize {
    1
}

fn default_action() -> String {
    "projection".to_string()
}

/// Rules section of a cell.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesSection {
    /// Gauge learning rate.
    pub lr_gauge: f64,
    /// Gauge momentum.
    pub momentum: f64,
    /// Momentum norm ceiling; `0.0` is unconstrained.
    pub momentum_max_norm: f64,
    /// Steps between forced retractions; `0` disables them.
    pub retraction_interval: usize,
    /// Orthogonal-update method.
    pub exp_method: String,
    /// LAPACK SVD driver: `"gesvd"` or `"gesdd"`.
    #[serde(default = "crate::schema::default_svd_driver")]
    pub svd_driver: String,
    /// Block update method.
    pub update_method: String,
    /// Retraction method.
    pub retraction_method: String,
}

/// Map a driver name to the library selector.
///
/// Returns `None` when the name is not one the crate knows.
pub fn svd_driver_from_str(value: &str) -> Option<LapackSvdDriver> {
    match value {
        "gesvd" => Some(LapackSvdDriver::Gesvd),
        "gesdd" => Some(LapackSvdDriver::Gesdd),
        _ => None,
    }
}

/// Objective section of a cell.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveSection {
    /// Supervised loss.
    pub loss: String,
    /// Angular margin in radians.
    pub cmr_margin: f64,
}

/// Diagnostics section of a cell.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticsSection {
    /// Epochs between raw per-block snapshots.
    pub snapshot_every: u32,
    /// Epochs between block-distance records.
    pub compute_block_every: u32,
    /// Whether interior flow and class-angle diagnostics are recorded.
    #[serde(default)]
    pub flow_diagnostics: bool,
}

/// Split section, required by the three-way split engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitSection {
    /// Validation split size.
    pub val_size: usize,
    /// Split seed, independent of the run seed.
    pub split_seed: u64,
    /// Whether the trailing partial training batch is discarded.
    pub drop_last: bool,
}

/// Baseline hyperparameters, required when a baseline arm is requested.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselinesSection {
    /// Hidden width of the two-layer comparison network.
    pub mlp_hidden: usize,
    /// Learning rate of the two-layer comparison network.
    pub mlp_lr: f64,
    /// Learning rate of the Riemannian comparison arm.
    pub rsgd_lr: f64,
}

/// Axes a cell sweeps over.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepSection {
    /// Retraction intervals to run.
    #[serde(default)]
    pub retraction_interval: Vec<usize>,
    /// Gauge learning rates to run.
    #[serde(default)]
    pub lr_gauge: Vec<f64>,
    /// Gauge momenta to run.
    #[serde(default)]
    pub momentum: Vec<f64>,
    /// Momentum norm ceilings to run.
    #[serde(default)]
    pub momentum_max_norm: Vec<f64>,
    /// Angular margins to run.
    #[serde(default)]
    pub cmr_margin: Vec<f64>,
    /// Topologies to run.
    #[serde(default)]
    pub topology: Vec<String>,
    /// Interior block counts to run.
    #[serde(default)]
    pub num_interior_blocks: Vec<usize>,
    /// Interior subspace widths to run.
    #[serde(default)]
    pub interior_subspace_dim: Vec<usize>,
}

/// Shortened-run settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmokeSection {
    /// Epochs a shortened run executes.
    pub epochs: u32,
    /// Seeds a shortened run uses.
    pub seeds: Vec<u64>,
    /// Sweep coordinates a shortened run restricts itself to, as
    /// `axis = [value]` entries.
    #[serde(default)]
    pub sweep: SweepSection,
}

/// A cell file as written on disk.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellFile {
    /// Cell name; also the output subdirectory.
    pub experiment: String,
    /// Dataset the cell trains on.
    pub dataset: String,
    /// Engine the cell runs on.
    pub engine: Engine,
    /// Seeds the cell runs.
    pub seeds: Vec<u64>,
    /// Epochs each run executes.
    pub epochs: u32,
    /// Batch size.
    pub batch_size: usize,
    /// Arms to run; defaults to the single COSMOS arm.
    #[serde(default = "default_arms")]
    pub arms: Vec<String>,
    /// Model section.
    pub model: ModelSection,
    /// Rules section.
    pub rules: RulesSection,
    /// Objective section.
    pub objective: ObjectiveSection,
    /// Diagnostics section.
    pub diagnostics: DiagnosticsSection,
    /// Split section.
    #[serde(default)]
    pub split: Option<SplitSection>,
    /// Baseline section.
    #[serde(default)]
    pub baselines: Option<BaselinesSection>,
    /// Sweep section.
    #[serde(default)]
    pub sweep: SweepSection,
    /// Shortened-run section.
    pub smoke: SmokeSection,
}

fn default_arms() -> Vec<String> {
    vec!["cosmos".to_string()]
}

/// One fully resolved run.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Cell name.
    pub experiment: String,
    /// Engine.
    pub engine: Engine,
    /// Dataset.
    pub dataset: DatasetName,
    /// Arm.
    pub arm: String,
    /// Run seed.
    pub seed: u64,
    /// Epochs.
    pub epochs: u32,
    /// Batch size.
    pub batch_size: usize,
    /// Model section with sweep coordinates applied.
    pub model: ModelSection,
    /// Rules section with sweep coordinates applied.
    pub rules: RulesSection,
    /// Objective section with sweep coordinates applied.
    pub objective: ObjectiveSection,
    /// Diagnostics section.
    pub diagnostics: DiagnosticsSection,
    /// Split section, present on the three-way split engine.
    pub split: Option<SplitSection>,
    /// Baseline section, present when a baseline arm runs.
    pub baselines: Option<BaselinesSection>,
}

impl Cell {
    /// File-name stem for this run's artifact.
    ///
    /// The stem is unique only inside one sweep coordinate, so an artifact
    /// path pairs it with [`Cell::coordinate`].
    pub fn stem(&self) -> String {
        match self.engine {
            Engine::Suite => format!("run_r{}_seed_{}", self.rules.retraction_interval, self.seed),
            Engine::Table => format!("{}_seed_{}", self.arm, self.seed),
        }
    }

    /// Subdirectory name for the sweep coordinate this run sits at.
    pub fn coordinate(&self) -> String {
        format!(
            "{}_L{}_d{}_lr{}_m{}_n{}_mar{:.6}",
            self.model.topology,
            self.model.num_interior_blocks,
            self.model.interior_subspace_dim,
            self.rules.lr_gauge,
            self.rules.momentum,
            self.rules.momentum_max_norm,
            self.objective.cmr_margin
        )
    }
}

/// Reason a cell file could not be used.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not valid TOML or is missing a required field.
    Parse(String),
    /// The file is internally inconsistent.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse(reason) => write!(f, "config parse error: {reason}"),
            Self::Invalid(reason) => write!(f, "config is inconsistent: {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Read and validate a cell file.
///
/// # Errors
///
/// Returns [`ConfigError`] when the file cannot be read, does not parse, or
/// is internally inconsistent.
pub fn load(path: &Path) -> Result<CellFile, ConfigError> {
    let text = fs::read_to_string(path).map_err(ConfigError::Io)?;
    let file: CellFile =
        toml::from_str(&text).map_err(|err| ConfigError::Parse(err.to_string()))?;
    validate(&file)?;
    Ok(file)
}

fn validate(file: &CellFile) -> Result<(), ConfigError> {
    if file.seeds.is_empty() {
        return Err(ConfigError::Invalid("seeds must not be empty".to_string()));
    }
    if file.engine == Engine::Table && file.split.is_none() {
        return Err(ConfigError::Invalid(
            "the table engine requires a [split] section".to_string(),
        ));
    }
    if file.engine == Engine::Suite && file.split.is_some() {
        return Err(ConfigError::Invalid(
            "the suite engine trains on the full training set and takes no [split]".to_string(),
        ));
    }
    let needs_baselines = file.arms.iter().any(|arm| arm != "cosmos");
    if needs_baselines && file.baselines.is_none() {
        return Err(ConfigError::Invalid(
            "a baseline arm requires a [baselines] section".to_string(),
        ));
    }
    if file.model.variant == "patchwise" && file.model.image_shape.is_none() {
        return Err(ConfigError::Invalid(
            "the patchwise variant requires model.image_shape".to_string(),
        ));
    }
    if svd_driver_from_str(&file.rules.svd_driver).is_none() {
        return Err(ConfigError::Invalid(format!(
            "unknown rules.svd_driver {}; expected gesvd or gesdd",
            file.rules.svd_driver
        )));
    }
    for arm in &file.arms {
        if !matches!(arm.as_str(), "cosmos" | "mlp" | "rsgd") {
            return Err(ConfigError::Invalid(format!(
                "unknown arm {arm}; expected cosmos, mlp or rsgd"
            )));
        }
    }
    smoke_axis_off_grid(
        "retraction_interval",
        &file.smoke.sweep.retraction_interval,
        &file.sweep.retraction_interval,
        file.rules.retraction_interval,
    )?;
    smoke_axis_off_grid(
        "lr_gauge",
        &file.smoke.sweep.lr_gauge,
        &file.sweep.lr_gauge,
        file.rules.lr_gauge,
    )?;
    smoke_axis_off_grid(
        "momentum",
        &file.smoke.sweep.momentum,
        &file.sweep.momentum,
        file.rules.momentum,
    )?;
    smoke_axis_off_grid(
        "momentum_max_norm",
        &file.smoke.sweep.momentum_max_norm,
        &file.sweep.momentum_max_norm,
        file.rules.momentum_max_norm,
    )?;
    smoke_axis_off_grid(
        "cmr_margin",
        &file.smoke.sweep.cmr_margin,
        &file.sweep.cmr_margin,
        file.objective.cmr_margin,
    )?;
    smoke_axis_off_grid(
        "topology",
        &file.smoke.sweep.topology,
        &file.sweep.topology,
        file.model.topology.clone(),
    )?;
    smoke_axis_off_grid(
        "num_interior_blocks",
        &file.smoke.sweep.num_interior_blocks,
        &file.sweep.num_interior_blocks,
        file.model.num_interior_blocks,
    )?;
    smoke_axis_off_grid(
        "interior_subspace_dim",
        &file.smoke.sweep.interior_subspace_dim,
        &file.sweep.interior_subspace_dim,
        file.model.interior_subspace_dim,
    )?;
    Ok(())
}

/// Rejects a `[smoke.sweep]` coordinate absent from the effective
/// `[sweep]` grid for that axis (the file's list, or the base value when
/// the axis is not swept).
fn smoke_axis_off_grid<T: PartialEq + std::fmt::Display + Clone>(
    axis_name: &str,
    smoke_values: &[T],
    sweep_values: &[T],
    base_value: T,
) -> Result<(), ConfigError> {
    if smoke_values.is_empty() {
        return Ok(());
    }
    let allowed = axis(sweep_values, base_value);
    for value in smoke_values {
        if !allowed.iter().any(|candidate| candidate == value) {
            return Err(ConfigError::Invalid(format!(
                "smoke.sweep.{axis_name} value {value} is not in the sweep.{axis_name} grid"
            )));
        }
    }
    Ok(())
}

fn axis<T: Clone>(sweep: &[T], fallback: T) -> Vec<T> {
    if sweep.is_empty() {
        vec![fallback]
    } else {
        sweep.to_vec()
    }
}

/// Expand a cell file into the runs it describes.
///
/// The product is taken over every non-empty sweep axis, then over the
/// arms, then over the seeds. `seeds` overrides the file's seed list.
/// `smoke` substitutes the `[smoke]` epochs, seeds and sweep restriction.
///
/// # Errors
///
/// Returns [`ConfigError::Invalid`] when the dataset name is unknown.
pub fn expand(
    file: &CellFile,
    seeds: Option<&[u64]>,
    smoke: bool,
) -> Result<Vec<Cell>, ConfigError> {
    let dataset: DatasetName = file
        .dataset
        .parse()
        .map_err(|reason: String| ConfigError::Invalid(reason))?;

    let epochs = if smoke {
        file.smoke.epochs
    } else {
        file.epochs
    };
    let seed_list: Vec<u64> = match (seeds, smoke) {
        (Some(explicit), _) => explicit.to_vec(),
        (None, true) => file.smoke.seeds.clone(),
        (None, false) => file.seeds.clone(),
    };
    let sweep = if smoke {
        &file.smoke.sweep
    } else {
        &file.sweep
    };

    let retractions = axis(&sweep.retraction_interval, file.rules.retraction_interval);
    let lrs = axis(&sweep.lr_gauge, file.rules.lr_gauge);
    let momenta = axis(&sweep.momentum, file.rules.momentum);
    let ceilings = axis(&sweep.momentum_max_norm, file.rules.momentum_max_norm);
    let margins = axis(&sweep.cmr_margin, file.objective.cmr_margin);
    let topologies = axis(&sweep.topology, file.model.topology.clone());
    let depths = axis(&sweep.num_interior_blocks, file.model.num_interior_blocks);
    let widths = axis(
        &sweep.interior_subspace_dim,
        file.model.interior_subspace_dim,
    );

    let mut cells = Vec::new();
    for topology in &topologies {
        for &depth in &depths {
            for &width in &widths {
                for &retraction in &retractions {
                    for &lr in &lrs {
                        for &momentum in &momenta {
                            for &ceiling in &ceilings {
                                for &margin in &margins {
                                    for arm in &file.arms {
                                        for &seed in &seed_list {
                                            let mut model = file.model.clone();
                                            model.topology = topology.clone();
                                            model.num_interior_blocks = depth;
                                            model.interior_subspace_dim = width;
                                            let mut rules = file.rules.clone();
                                            rules.retraction_interval = retraction;
                                            rules.lr_gauge = lr;
                                            rules.momentum = momentum;
                                            rules.momentum_max_norm = ceiling;
                                            let mut objective = file.objective.clone();
                                            objective.cmr_margin = margin;
                                            cells.push(Cell {
                                                experiment: file.experiment.clone(),
                                                engine: file.engine,
                                                dataset,
                                                arm: arm.clone(),
                                                seed,
                                                epochs,
                                                batch_size: file.batch_size,
                                                model,
                                                rules,
                                                objective,
                                                diagnostics: file.diagnostics.clone(),
                                                split: file.split.clone(),
                                                baselines: file.baselines.clone(),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
    }

    #[test]
    fn every_shipped_cell_file_loads() {
        let mut count = 0;
        for entry in fs::read_dir(config_dir()).expect("config dir") {
            let path = entry.expect("entry").path();
            if path.file_name().and_then(|name| name.to_str()) == Some("bench.toml")
                || path.file_name().and_then(|name| name.to_str()) == Some("smoke_references.toml")
            {
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                load(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
                count += 1;
            }
        }
        assert_eq!(count, 11);
    }

    #[test]
    fn retraction_sweep_expands_to_six_by_six() {
        let file = load(&config_dir().join("e1.toml")).expect("load");
        let cells = expand(&file, None, false).expect("expand");
        assert_eq!(cells.len(), 36);
        assert_eq!(cells[0].epochs, 40);
    }

    #[test]
    fn regime_grid_expands_to_one_hundred_and_eighty() {
        let file = load(&config_dir().join("e2.toml")).expect("load");
        let cells = expand(&file, None, false).expect("expand");
        assert_eq!(cells.len(), 180);
        assert!(cells.iter().any(|cell| cell.rules.momentum_max_norm == 0.0));
    }

    #[test]
    fn topology_grid_expands_to_sixty() {
        let file = load(&config_dir().join("e4_topology.toml")).expect("load");
        let cells = expand(&file, None, false).expect("expand");
        assert_eq!(cells.len(), 60);
    }

    #[test]
    fn baselines_cell_expands_to_nine_runs() {
        let file = load(&config_dir().join("e3_baselines.toml")).expect("load");
        let cells = expand(&file, None, false).expect("expand");
        assert_eq!(cells.len(), 9);
        assert_eq!(cells[0].engine, Engine::Table);
    }

    #[test]
    fn smoke_mode_shortens_without_changing_the_objective() {
        let file = load(&config_dir().join("e3_baselines.toml")).expect("load");
        let full = expand(&file, None, false).expect("expand");
        let short = expand(&file, None, true).expect("expand");
        assert_eq!(short[0].epochs, 3);
        assert_eq!(short.len(), 3);
        assert_eq!(short[0].objective.loss, full[0].objective.loss);
        assert_eq!(short[0].objective.cmr_margin, full[0].objective.cmr_margin);
        assert_eq!(short[0].rules.lr_gauge, full[0].rules.lr_gauge);
    }

    #[test]
    fn seed_override_replaces_the_file_list() {
        let file = load(&config_dir().join("e1.toml")).expect("load");
        let cells = expand(&file, Some(&[7]), false).expect("expand");
        assert_eq!(cells.len(), 6);
        assert!(cells.iter().all(|cell| cell.seed == 7));
    }

    #[test]
    fn off_grid_smoke_margin_is_refused() {
        let path = config_dir().join("e5_cifar_margin.toml");
        let text = fs::read_to_string(&path).expect("read");
        let broken = text.replace(
            "[smoke.sweep]\ncmr_margin = [0.3141592653589793]",
            "[smoke.sweep]\ncmr_margin = [0.5]",
        );
        assert_ne!(
            text, broken,
            "fixture no longer contains the expected block"
        );
        let file: CellFile = toml::from_str(&broken).expect("parse");
        let err = validate(&file).expect_err("off-grid smoke margin");
        match err {
            ConfigError::Invalid(message) => {
                assert!(message.contains("cmr_margin"));
                assert!(message.contains("0.5"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rules_without_a_driver_default_to_gesvd() {
        let file = load(&config_dir().join("e1.toml")).expect("load");
        assert_eq!(file.rules.svd_driver, "gesvd");
    }

    #[test]
    fn driver_names_map_to_the_library_selector() {
        assert_eq!(svd_driver_from_str("gesvd"), Some(LapackSvdDriver::Gesvd));
        assert_eq!(svd_driver_from_str("gesdd"), Some(LapackSvdDriver::Gesdd));
        assert_eq!(svd_driver_from_str("jacobi"), None);
    }

    #[test]
    fn a_declared_driver_is_carried_onto_every_cell() {
        let path = config_dir().join("e1.toml");
        let text = fs::read_to_string(&path).expect("read");
        let edited = text.replace(
            "[rules]",
            "[rules]
svd_driver = \"gesdd\"",
        );
        assert_ne!(text, edited, "fixture no longer contains a [rules] table");
        let file: CellFile = toml::from_str(&edited).expect("parse");
        validate(&file).expect("valid driver");
        let cells = expand(&file, Some(&[7]), true).expect("expand");
        assert!(cells.iter().all(|cell| cell.rules.svd_driver == "gesdd"));
    }

    #[test]
    fn an_unknown_driver_is_refused() {
        let path = config_dir().join("e1.toml");
        let text = fs::read_to_string(&path).expect("read");
        let broken = text.replace(
            "[rules]",
            "[rules]
svd_driver = \"jacobi\"",
        );
        assert_ne!(text, broken, "fixture no longer contains a [rules] table");
        let file: CellFile = toml::from_str(&broken).expect("parse");
        let err = validate(&file).expect_err("unknown driver");
        match err {
            ConfigError::Invalid(message) => {
                assert!(message.contains("svd_driver"));
                assert!(message.contains("jacobi"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// A resolved cell matching the default configuration, for tests that
    /// need a cell but not a file.
    pub(crate) fn sample_cell() -> Cell {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
            .join("paper")
            .join("e1.toml");
        let file = load(&path).expect("sample cell file");
        expand(&file, Some(&[319_635_789]), true)
            .expect("expand")
            .remove(0)
    }
}
