//! Command-line grammar and cross-flag validation.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Reproduction driver for the experiments reported in the COSMOS article.
#[derive(Debug, Parser)]
#[command(name = "cosmos-paper", version, about)]
pub struct Cli {
    /// Command to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Flags shared by every experiment command.
#[derive(Debug, Clone, clap::Args)]
pub struct RunArgs {
    /// Cell file to run. Defaults to the cell's file under `configs/paper`.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Seeds to run, overriding the cell file's list.
    #[arg(long, value_delimiter = ',')]
    pub seeds: Option<Vec<u64>>,
    /// Output root for run artifacts.
    #[arg(long, default_value = "runs")]
    pub out: PathBuf,
    /// Dataset cache root.
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
    /// Fail instead of fetching a missing dataset.
    #[arg(long)]
    pub no_download: bool,
    /// Run the shortened form of the cell and check it against the
    /// documented reference.
    #[arg(long)]
    pub smoke: bool,
    /// Cells to run concurrently.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
}

/// Commands the driver offers.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Invariant preservation across retraction intervals.
    E1(RunArgs),
    /// Gauge regime map.
    E2(RunArgs),
    /// Comparison baselines and the margin ablation.
    E3(RunArgs),
    /// Topology robustness and the companion control.
    E4(RunArgs),
    /// Colour-dataset and patch-wise cells.
    E5(RunArgs),
    /// Cost of the orthogonal-update primitives.
    Bench {
        /// Benchmark cell file.
        #[arg(long, default_value = "configs/paper/bench.toml")]
        config: PathBuf,
        /// Measure the smaller ambient dimensions only.
        #[arg(long)]
        quick: bool,
        /// Measure the shortened set of ambient dimensions and check the
        /// result against the documented benchmark reference.
        #[arg(long)]
        smoke: bool,
        /// Rebuild the report page from a stored measurement set.
        #[arg(long)]
        render_from: Option<PathBuf>,
        /// Output directory.
        #[arg(long, default_value = "runs/bench")]
        out: PathBuf,
    },
    /// Download and verify a dataset.
    Fetch {
        /// Dataset name.
        dataset: String,
        /// Dataset cache root.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// Aggregate finished run artifacts into result tables.
    Report {
        /// Run artifact root.
        #[arg(long, default_value = "runs")]
        runs: PathBuf,
        /// Output directory for the tables.
        #[arg(long, default_value = "results/paper")]
        out: PathBuf,
    },
    /// Parameter, orthonormality and partition assertions.
    SelfCheck {
        /// Also print the first row of the first block's basis draw for the
        /// reference seed, at full precision, for calibration against a
        /// reference vector.
        #[arg(long)]
        dump_first_row: bool,
    },
    /// Check run artifacts against the schema this crate writes.
    Validate {
        /// File or directory to examine.
        path: PathBuf,
        /// Fail when no artifact is found.
        #[arg(long)]
        require_artifacts: bool,
    },
}

impl RunArgs {
    /// Resolve the cell file, defaulting to the named cell under the config
    /// directory.
    pub fn config_path(&self, default_name: &str) -> PathBuf {
        self.config
            .clone()
            .unwrap_or_else(|| PathBuf::from("configs/paper").join(default_name))
    }

    /// Validate flag combinations.
    ///
    /// # Errors
    ///
    /// Returns a description when `threads` is zero.
    pub fn validate(&self) -> Result<(), String> {
        if self.threads == 0 {
            return Err("--threads must be at least 1".to_string());
        }
        Ok(())
    }
}

/// Parse the command line and execute the selected command.
///
/// # Errors
///
/// Returns a description of the first failure the command reported.
pub fn run() -> Result<(), String> {
    use clap::Parser as _;

    use crate::{bench, datasets, experiments, selfcheck, validate};

    let cli = Cli::parse();
    match &cli.command {
        Command::E1(args) => report_outcome(&experiments::e1::run(args)?),
        Command::E2(args) => report_outcome(&experiments::e2::run(args)?),
        Command::E3(args) => report_outcome(&experiments::e3::run(args)?),
        Command::E4(args) => report_outcome(&experiments::e4::run(args)?),
        Command::E5(args) => report_outcome(&experiments::e5::run(args)?),
        Command::Bench {
            config,
            quick,
            smoke,
            render_from,
            out,
        } => {
            if let Some(stored) = render_from {
                bench::render_from(stored, out).map_err(|err| err.to_string())?;
                println!("wrote {}", out.join("report.html").display());
                return Ok(());
            }
            let parsed = bench::load_config(config).map_err(|err| err.to_string())?;
            let results =
                bench::run(&parsed, *quick || *smoke, out).map_err(|err| err.to_string())?;
            for (name, slope) in &results.slopes {
                println!("{name}: slope {slope:.3}");
            }
            if *smoke {
                return check_bench_reference(&results);
            }
            Ok(())
        }
        Command::Fetch { dataset, data_dir } => {
            let name: datasets::DatasetName = dataset.parse()?;
            datasets::fetch(name, data_dir).map_err(|err| err.to_string())?;
            println!("{name} is present and verified in {}", data_dir.display());
            Ok(())
        }
        Command::Report { runs, out } => {
            let written = crate::report::run(runs, out).map_err(|err| err.to_string())?;
            for path in written {
                println!("wrote {path}");
            }
            Ok(())
        }
        Command::SelfCheck { dump_first_row } => {
            for line in selfcheck::run_all()? {
                println!("{line}");
            }
            if *dump_first_row {
                for line in selfcheck::first_row_dump() {
                    println!("{line}");
                }
            }
            Ok(())
        }
        Command::Validate {
            path,
            require_artifacts,
        } => {
            let report = validate::validate_path(path, *require_artifacts)?;
            validate::print(&report);
            if report.is_success() {
                Ok(())
            } else {
                Err(format!(
                    "{} invalid, {} valid",
                    report.invalid, report.valid
                ))
            }
        }
    }
}

fn check_bench_reference(results: &crate::bench::exp_bench::BenchResults) -> Result<(), String> {
    use std::path::Path;

    use crate::{bench, experiments, smoke};

    let references =
        smoke::load(Path::new(experiments::REFERENCE_FILE)).map_err(|err| err.to_string())?;
    let Some(reference) = references.bench.as_ref() else {
        println!("no reference for bench; structural checks only");
        return Ok(());
    };
    let deviations = smoke::check_bench(reference, &bench::observed(results));
    if deviations.is_empty() {
        println!("{} checked, no deviations", reference.name);
        return Ok(());
    }
    for line in &deviations {
        eprintln!("{} {line}", reference.name);
    }
    Err(format!("{} value(s) outside tolerance", deviations.len()))
}

fn report_outcome(outcome: &crate::experiments::Outcome) -> Result<(), String> {
    println!("{} run artifacts written", outcome.run_files.len());
    if outcome.deviations.is_empty() {
        return Ok(());
    }
    for line in &outcome.deviations {
        eprintln!("{line}");
    }
    Err(format!(
        "{} value(s) outside tolerance",
        outcome.deviations.len()
    ))
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn grammar_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn seeds_parse_as_a_comma_separated_list() {
        let cli = Cli::parse_from(["cosmos-paper", "e1", "--seeds", "1,2,3"]);
        match cli.command {
            Command::E1(args) => assert_eq!(args.seeds, Some(vec![1, 2, 3])),
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn defaults_are_relative_paths() {
        let cli = Cli::parse_from(["cosmos-paper", "e4"]);
        match cli.command {
            Command::E4(args) => {
                assert_eq!(args.out, PathBuf::from("runs"));
                assert_eq!(args.data_dir, PathBuf::from("data"));
                assert_eq!(
                    args.config_path("e4_topology.toml"),
                    PathBuf::from("configs/paper").join("e4_topology.toml")
                );
            }
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn zero_threads_is_rejected() {
        let cli = Cli::parse_from(["cosmos-paper", "e2", "--threads", "0"]);
        match cli.command {
            Command::E2(args) => assert!(args.validate().is_err()),
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn the_subcommand_list_is_the_documented_one() {
        let command = Cli::command();
        let names: Vec<String> = command
            .get_subcommands()
            .map(|sub| sub.get_name().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "e1",
                "e2",
                "e3",
                "e4",
                "e5",
                "bench",
                "fetch",
                "report",
                "self-check",
                "validate"
            ]
        );
        for sub in command.get_subcommands() {
            assert!(
                sub.get_about().is_some(),
                "{} has no help line",
                sub.get_name()
            );
        }
    }

    #[test]
    fn the_benchmark_takes_a_shortened_form() {
        let cli = Cli::parse_from(["cosmos-paper", "bench"]);
        match cli.command {
            Command::Bench { quick, smoke, .. } => {
                assert!(!quick);
                assert!(!smoke);
            }
            other => panic!("unexpected command {other:?}"),
        }

        let cli = Cli::parse_from(["cosmos-paper", "bench", "--smoke"]);
        match cli.command {
            Command::Bench {
                quick, smoke, out, ..
            } => {
                assert!(!quick);
                assert!(smoke);
                assert_eq!(out, PathBuf::from("runs/bench"));
            }
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn a_shortened_run_is_opt_in() {
        let cli = Cli::parse_from(["cosmos-paper", "e5", "--smoke", "--no-download"]);
        match cli.command {
            Command::E5(args) => {
                assert!(args.smoke);
                assert!(args.no_download);
                assert_eq!(args.threads, 1);
            }
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn the_dataset_command_takes_a_positional_name() {
        let cli = Cli::parse_from(["cosmos-paper", "fetch", "cifar10", "--data-dir", "cache"]);
        match cli.command {
            Command::Fetch { dataset, data_dir } => {
                assert_eq!(dataset, "cifar10");
                assert_eq!(data_dir, PathBuf::from("cache"));
            }
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn the_calibration_dump_is_a_flag_of_the_assertion_command() {
        let cli = Cli::parse_from(["cosmos-paper", "self-check", "--dump-first-row"]);
        match cli.command {
            Command::SelfCheck { dump_first_row } => assert!(dump_first_row),
            other => panic!("unexpected command {other:?}"),
        }
    }
}
