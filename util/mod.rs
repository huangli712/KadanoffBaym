// Project : KadanoffBaym
// File    : util/mod.rs
// Purpose : the shared utilities of the example programs and the
//           integration tests (configuration, output, propagation
//           drivers, kbref IO, order gate)
//
// Layout: this root only declares the modules and re-exports their
// public items; every helper lives in the file of its topic. The only
// accepted environment variable is `KB_TEST_PROFILE = quick | release`
// (default: quick); `order` is the single shared convergence-order
// gate and `kbref` the reader/writer of the reference data.
//

// This module is #[path]-included into every example binary and test
// target, and each of them consumes only a subset of the items below
// (the same subset-consumption rule as `kbref`). The single file-level
// allow keeps the per-target lints quiet for the whole subtree, so the
// sub-modules carry no lint attributes of their own.
#![allow(unused_imports, dead_code, clippy::duplicate_mod)]

mod args;
mod config;
mod fixtures;
mod format;
mod green;
mod kbref;
mod order;
mod output;
mod profile;
mod propagate;
mod rng;
mod toml;

pub mod prelude;

pub use args::arg_or;
pub use config::{
    ContourParams, ExampleArgs, ExampleConfig, SolverParams, contour_params, load_example_config,
    load_tvector, parse_example_args, solver_params,
};
pub use fixtures::{test_ha, test_hb, test_vv};
pub use format::{fmt_e12, fmt_e17, fmt_g17};
pub use green::{add_green_step, moving_rowerr, zero_green_step};
pub use kbref::{
    KBREF_CG_SCALARS, KBREF_OBSERVABLE_META, KBREF_SELECTED_GREEN_KEYS, KBREF_SERIES_CASE_SETS,
    KbrefData, KbrefRow, OBSERVABLE_MODELS, approximations, observable_class,
    observable_components, read_kbref, series_class, series_components, series_generator,
    write_kbref,
};
pub use order::{OrderGate, assert_order_gate, check_order_gate};
pub use output::{
    GreenRow, ness_iter_row, ness_outdir, write_green_components, write_ness_gf, write_table,
};
pub use profile::is_release;
pub use propagate::{
    ConvergenceError, LoopEvent, iterate_bootstrap, iterate_matsubara, propagate_timesteps,
};
pub use rng::{prand, rand_c64, rand_unit, randn, splitmix64};
pub use toml::{
    TomlError, TomlValue, inp_toml, parse_toml, parse_toml_file, tget_f, tget_i, tget_s,
};
