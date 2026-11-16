// Project : KadanoffBaym
// File    : util/mod.rs
// Purpose : the shared utilities of the example programs and the
//           integration tests (configuration, output, propagation
//           drivers, kbref IO, order gate)

//! Shared utilities of the example programs and the integration tests.
//!
//! The only accepted environment variable is
//! `KB_TEST_PROFILE = quick | release` (default: quick), and the
//! convergence-order gate is the single shared implementation for all
//! order tests. `kbref` is the format reader/writer of the reference
//! data.

// This module is #[path]-included into every example binary and test
// target, and each of them consumes only a subset of the items below
// (the same subset-consumption rule as `kbref.rs`); one file-level
// allow pair keeps the per-target lints quiet.
#![allow(unused_imports, dead_code)]

pub mod prelude {
    pub use kadanoffbaym::complex::Complex;
    pub use kadanoffbaym::linalg::DenseMat;
    pub use kadanoffbaym::types::ContourGreen;
    pub use kadanoffbaym::types::GreenTimeStep;
    pub use kadanoffbaym::types::LessComp;
    pub use kadanoffbaym::types::LmixComp;
    pub use kadanoffbaym::types::MATSUBARA;
    pub use kadanoffbaym::types::MatComp;
    pub use kadanoffbaym::types::RetComp;
    pub use kadanoffbaym::types::TimeFunction;
    pub use kadanoffbaym::types::{Contour, Sign};
}

mod config;
mod format;
mod kbref;
mod ness;
mod order;
mod output;
mod profile;
mod propagate;
mod toml;

pub use config::{
    ContourParams, ExampleArgs, ExampleConfig, SolverParams, contour_of, contour_params,
    integrator_of, load_example_config, load_tvector, parse_example_args, solver_params,
};
pub use format::{fmt_17g, fmt_c_e17, fmt_e12};
pub use kbref::{
    KBREF_CG_SCALARS, KBREF_OBSERVABLE_META, KBREF_SELECTED_GREEN_KEYS, KBREF_SERIES_CASE_SETS,
    KbrefData, KbrefRow, OBSERVABLE_MODELS, approximations, observable_class,
    observable_components, read_kbref, series_class, series_components, series_generator,
    write_kbref,
};
pub use ness::{inp_toml, ness_iter_row, ness_outdir, tget_f, tget_i, tget_s, write_ness_gf};
pub use order::{OrderGate, assert_order_gate, check_order_gate};
pub use output::{GreenRow, write_green_components, write_table};
pub use profile::is_release;
pub use propagate::{
    ConvergenceError, LoopEvent, iterate_bootstrap, iterate_matsubara, propagate_timesteps,
};
pub use toml::{TomlError, TomlValue, parse_toml, parse_toml_file};
