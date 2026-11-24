// Project : KadanoffBaym
// File    : util/mod.rs
// Purpose : the shared utilities of the example programs and the
//           integration tests (configuration, output, propagation
//           drivers, ref IO, order gate)
//
// Layout: this root only declares the modules and re-exports their
// public items; every helper lives in the file of its topic. The only
// accepted environment variable is `KB_TEST_PROFILE = quick | release`
// (default: quick); `order` is the single shared convergence-order
// gate and `ref` the reader/writer of the reference data.
//

// This module is #[path]-included into every example binary and test
// target, and each of them consumes only a subset of the items below
// (the same subset-consumption rule as `ref`). The single file-level
// allow keeps the per-target lints quiet for the whole subtree, so the
// sub-modules carry no lint attributes of their own.
#![allow(unused_imports, dead_code, clippy::duplicate_mod)]

mod args;
mod config;
mod fixtures;
mod format;
mod green;
mod order;
mod output;
mod profile;
mod propagate;
mod reference;
mod rng;
mod support;
mod toml;

pub mod prelude;

// ------------------------------------------------------------------
// types
// ------------------------------------------------------------------

pub use config::{ContourParams, ExampleArgs, ExampleConfig, SolverParams};
pub use order::OrderGate;
pub use output::GreenRow;
pub use propagate::{ConvergenceError, LoopEvent};
pub use reference::{RefData, RefRow};
pub use toml::{TomlError, TomlValue};

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

pub use reference::{
    OBSERVABLE_MODELS, REF_CG_SCALARS, REF_OBSERVABLE_META, REF_SELECTED_GREEN_KEYS,
    REF_SERIES_CASE_SETS,
};

// ------------------------------------------------------------------
// functions
// ------------------------------------------------------------------

pub use args::arg_or_default;
pub use config::{
    contour_params, load_example_config, load_tvector, parse_example_args, solver_params,
};
pub use fixtures::{test_hamiltonian_a, test_hamiltonian_b, test_interaction};
pub use format::{fmt_e12, fmt_e17, fmt_g17};
pub use green::{
    add_green_step, block_norm, dyson_moving_step, moving_row_error, vie2_moving_step,
    zero_green_step,
};
pub use order::{assert_order_gate, check_order_gate};
pub use output::{ness_iter_row, ness_outdir, write_green_components, write_ness_gf, write_table};
pub use profile::is_release;
pub use propagate::{iterate_bootstrap, iterate_matsubara, propagate_timesteps};
pub use reference::{
    approximations, observable_class, observable_components, read_ref, series_class,
    series_components, series_generator, write_ref,
};
pub use rng::{prand, rand_c64, rand_c64_scaled, rand_unit, randn, splitmix64};
pub use support::{c, panics, tmpdir, write_file};
pub use toml::{get_f64, get_i64, get_str, load_toml_config, parse_toml, parse_toml_file};
