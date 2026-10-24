//! Reproduction driver for the experiments reported in the COSMOS article.
//!
//! The driver reads a cell definition from `configs/paper/`, runs it on the
//! engine that cell declares, and writes one JSON artifact per run plus the
//! sidecars the analysis scripts consume. It adds no numerical capability to
//! the library: every quantity it records is read back from the public
//! `cosmos-rs` surface.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod baselines;
pub mod bench;
pub mod cli;
pub mod config;
pub mod datasets;
pub mod engine;
pub mod experiments;
pub mod init;
pub mod report;
pub mod schema;
pub mod selfcheck;
pub mod smoke;
pub mod validate;
