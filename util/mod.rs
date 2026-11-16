//! Shared utilities of the example programs and the integration tests.
//!
//! The only accepted environment variable is
//! `KB_TEST_PROFILE = quick | release` (default: quick), and the
//! convergence-order gate is the single shared implementation for all
//! order tests. `kbref` is the format reader; `compare` the
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
pub mod propagate;
pub mod compare;
pub mod kbref;
pub mod ness;
pub mod order;
pub mod output;
pub mod profile;
pub mod toml;

#[allow(unused_imports)]
pub use order::{OrderGate, assert_order_gate, check_order_gate};
#[allow(unused_imports)]
pub use profile::is_release;
