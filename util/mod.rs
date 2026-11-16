//! Shared utilities of the example programs and the integration tests.
//!
//! The only accepted environment variable is
//! `KB_TEST_PROFILE = quick | release` (default: quick), and the
//! convergence-order gate is the single shared implementation for all
//! order tests. `kbref_raw` is the kbref format reader; `kbref` the
//! reference-comparison layer built on top of it.

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

pub mod config;
pub mod contour_loop;
pub mod kbref;
pub mod kbref_raw;
pub mod ness_util;
pub mod order_gate;
pub mod output;
pub mod profile;
pub mod toml;

#[allow(unused_imports)]
pub use order_gate::{OrderGate, assert_order_gate, check_order_gate};
pub use profile::is_release;
