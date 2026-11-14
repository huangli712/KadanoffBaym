//! Shared helpers for the integration tests.
//!
//! Mirrors `julia/test/common/profile.jl` and `order_gate.jl`: the only
//! accepted environment variable is `KB_TEST_PROFILE = quick | release`
//! (default: quick), and the convergence-order gate is the single shared
//! implementation for all order tests (P2, P3, P4, P7).

/// Common imports for the ported test files.
#[allow(unused_imports)]
pub mod prelude {
    pub use kadanoffbaym::complex::Complex;
    pub use kadanoffbaym::linalg::mat::DenseMat;
    pub use kadanoffbaym::types::ContourGreen;
    pub use kadanoffbaym::types::step::GreenTimeStep;
    pub use kadanoffbaym::types::function::TimeFunction;
    pub use kadanoffbaym::types::less::LessComp;
    pub use kadanoffbaym::types::lmix::LmixComp;
    pub use kadanoffbaym::types::ret::RetComp;
    pub use kadanoffbaym::types::mat::MatComp;
    pub use kadanoffbaym::types::MATSUBARA;
    pub use kadanoffbaym::types::{Contour, Sign};
}

pub mod order_gate;
pub mod profile;

#[allow(unused_imports)]
pub use order_gate::{assert_order_gate, check_order_gate, OrderGate};
#[allow(unused_imports)]
pub use profile::{is_release, test_profile};
