//! Shared helpers for the integration tests.
//!
//! Mirrors `julia/test/common/profile.jl` and `order_gate.jl`: the only
//! accepted environment variable is `KB_TEST_PROFILE = quick | release`
//! (default: quick), and the convergence-order gate is the single shared
//! implementation for all order tests (P2, P3, P4, P7).

/// Common imports for the ported test files.
pub mod prelude {
    
    
    
    
    
}

pub mod order_gate;
pub mod profile;

