//! Cost measurements for the orthogonal-update primitives.

pub mod drift;
pub mod exp_bench;
pub mod report;

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

use drift::{DriftConfig, drift_curve};
use exp_bench::{BenchResults, measure};

/// The benchmark cell file.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchConfig {
    /// Ambient dimensions measured in a full run.
    pub sizes: Vec<usize>,
    /// Ambient dimensions measured in a shortened run.
    pub quick_sizes: Vec<usize>,
    /// Frobenius norm every generator is scaled to.
    pub target_norm: f64,
    /// Block width for the per-block orthonormalization.
    pub block_k: usize,
    /// Methods measured.
    pub methods: Vec<String>,
    /// Drift simulation parameters.
    pub drift: DriftConfig,
}

/// Read the benchmark cell file.
///
/// # Errors
///
/// Returns an I/O error when the file cannot be read or does not parse.
pub fn load_config(path: &Path) -> io::Result<BenchConfig> {
    let text = fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Run the benchmark and the drift simulation, writing both artifacts.
///
/// # Errors
///
/// Returns an I/O error when an artifact cannot be written.
pub fn run(config: &BenchConfig, quick: bool, out: &Path) -> io::Result<BenchResults> {
    let sizes = if quick {
        &config.quick_sizes
    } else {
        &config.sizes
    };
    let results = measure(sizes, &config.methods, config.target_norm, config.block_k);
    crate::schema::write_json(&out.join("results.json"), &results)?;
    fs::write(out.join("report.html"), report::render_html(&results))?;

    let curve = drift_curve(&config.drift);
    crate::schema::write_json(&out.join("ambient_drift.json"), &curve)?;
    Ok(results)
}

/// Restate a measurement set as the quantities the shortened-run checker
/// compares.
#[must_use]
pub fn observed(results: &BenchResults) -> crate::smoke::ObservedBench {
    crate::smoke::ObservedBench {
        orthogonality_error: results
            .methods
            .iter()
            .map(|entry| (entry.method.clone(), entry.n, entry.orthogonality_error))
            .collect(),
        slopes: results.slopes.clone(),
        block_qr_median_s: results
            .block_qr
            .iter()
            .map(|entry| (entry.n, entry.median_s))
            .collect(),
    }
}

/// Rebuild the report page from a stored measurement set.
///
/// # Errors
///
/// Returns an I/O error when the stored set cannot be read or the page
/// cannot be written.
pub fn render_from(results_path: &Path, out: &Path) -> io::Result<()> {
    let results: BenchResults = crate::schema::read_json(results_path)?;
    fs::create_dir_all(out)?;
    fs::write(out.join("report.html"), report::render_html(&results))
}
