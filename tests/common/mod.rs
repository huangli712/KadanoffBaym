//! Shared helpers for the integration tests.
//!
//! Mirrors `julia/test/common/profile.jl` and `order_gate.jl`: the only
//! accepted environment variable is `KB_TEST_PROFILE = quick | release`
//! (default: quick), and the convergence-order gate is the single shared
//! implementation for all order tests (P2, P3, P4, P7).

/// Common imports for the ported test files.
#[allow(unused_imports)]
pub mod prelude {
    pub use KadanoffBaym::complex::Complex;
    pub use KadanoffBaym::linalg::DenseMat;
    pub use KadanoffBaym::types::ContourGreen;
    pub use KadanoffBaym::types::GreenTimeStep;
    pub use KadanoffBaym::types::TimeFunction;
    pub use KadanoffBaym::types::LessComp;
    pub use KadanoffBaym::types::LmixComp;
    pub use KadanoffBaym::types::RetComp;
    pub use KadanoffBaym::types::MatComp;
    pub use KadanoffBaym::types::MATSUBARA;
    pub use KadanoffBaym::types::{Contour, Sign};
}

pub mod order_gate;
#[path = "../../tools/kbref.rs"]
pub mod kbref_raw;
pub mod profile;

#[allow(unused_imports)]
pub use order_gate::{assert_order_gate, check_order_gate, OrderGate};
#[allow(unused_imports)]
pub use profile::{is_release, test_profile};
