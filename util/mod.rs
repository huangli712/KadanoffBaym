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

#[allow(unused_imports)]
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

// Each including target uses only a subset of the re-exported items.
#[allow(unused_imports)]
pub use config::{ContourParams, ExampleArgs, ExampleConfig, SolverParams, contour_of,
    contour_params, integrator_of, load_example_config, load_tvector, parse_example_args,
    solver_params};
#[allow(unused_imports)]
pub use format::{fmt_c_e17, fmt_e12, fmt_17g};
#[allow(unused_imports)]
pub use kbref::{KBREF_CG_SCALARS, KBREF_OBSERVABLE_META, KBREF_SELECTED_GREEN_KEYS,
    KBREF_SERIES_CASE_SETS, KbrefData, KbrefRow, OBSERVABLE_MODELS, approximations,
    observable_class, observable_components, read_kbref, series_class, series_components,
    series_generator, write_kbref};
#[allow(unused_imports)]
pub use ness::{inp_toml, ness_iter_row, ness_outdir, tget_f, tget_i, tget_s, write_ness_gf};
#[allow(unused_imports)]
pub use order::{OrderGate, assert_order_gate, check_order_gate};
#[allow(unused_imports)]
pub use output::{GreenRow, write_green_components, write_table};
#[allow(unused_imports)]
pub use profile::is_release;
#[allow(unused_imports)]
pub use propagate::{ConvergenceError, LoopEvent, iterate_bootstrap, iterate_matsubara,
    propagate_timesteps};
#[allow(unused_imports)]
pub use toml::{TomlError, TomlValue, parse_toml, parse_toml_file};
