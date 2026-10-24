//! Aggregation of finished run artifacts into result tables.

pub mod tables;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::schema::{SuiteArtifact, TableArtifact, read_json};

fn json_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    if !root.exists() {
        return Ok(found);
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            found.extend(json_files(&path)?);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name != "suite_summary.json" && name != "analysis.json" {
                found.push(path);
            }
        }
    }
    Ok(found)
}

/// Read every artifact under `runs` and write one table per experiment.
///
/// # Errors
///
/// Returns an I/O error when the tree cannot be walked, an artifact does
/// not parse as either shape, or a table cannot be written.
pub fn run(runs: &Path, out: &Path) -> io::Result<Vec<String>> {
    fs::create_dir_all(out)?;
    let mut suite: BTreeMap<String, Vec<SuiteArtifact>> = BTreeMap::new();
    let mut table: BTreeMap<String, Vec<TableArtifact>> = BTreeMap::new();

    for path in json_files(runs)? {
        if let Ok(artifact) = read_json::<SuiteArtifact>(&path) {
            suite
                .entry(artifact.config.experiment_name.clone())
                .or_default()
                .push(artifact);
            continue;
        }
        let artifact: TableArtifact = read_json(&path)?;
        let experiment = path
            .parent()
            .and_then(|dir| dir.parent())
            .and_then(|dir| dir.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("table")
            .to_string();
        table.entry(experiment).or_default().push(artifact);
    }

    let mut written = Vec::new();
    for (experiment, artifacts) in suite {
        let csv = tables::suite_table(&artifacts)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let path = out.join(format!("{experiment}.csv"));
        fs::write(&path, csv)?;
        written.push(path.display().to_string());
    }
    for (experiment, artifacts) in table {
        let csv = tables::baselines_table(&artifacts)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let path = out.join(format!("{experiment}.csv"));
        fs::write(&path, csv)?;
        written.push(path.display().to_string());
    }
    Ok(written)
}
