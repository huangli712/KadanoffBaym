// Project : KadanoffBaym
// File    : util/rng.rs
// Purpose : the deterministic pseudo-random streams of the tests and
//           the benchmark drivers (splitmix64, and the LCG of the benchmark driver
//           side)
//

use std::f64::consts::PI;

use kadanoffbaym::complex::Complex;

// ------------------------------------------------------------------
// splitmix64 stream of the tests
// ------------------------------------------------------------------

/// Deterministic splitmix64 stream (golden-gamma advance).
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Uniform complex in `[-0.5, 0.5)^2`.
pub fn rand_c64(state: &mut u64) -> Complex {
    rand_c64_scaled(state, 1.0)
}

/// Uniform complex in `[-scale/2, scale/2)^2`.
pub fn rand_c64_scaled(state: &mut u64, scale: f64) -> Complex {
    let re = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(re * scale, im * scale)
}

/// Uniform unit interval `f64`.
pub fn rand_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal `f64` (Box-Muller).
pub fn randn(state: &mut u64) -> f64 {
    let u1 = rand_unit(state).max(1e-300);
    let u2 = rand_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ------------------------------------------------------------------
// the LCG of the benchmark driver side (the benchmark drivers)
// ------------------------------------------------------------------

/// Deterministic pseudo-random complex in (-0.5, 0.5); the same
/// as the benchmark driver side (values need not agree between the two programs,
/// only determinism matters)
pub fn prand(state: &mut u64) -> Complex {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let re = ((*state >> 33) % 1000) as f64 / 1000.0 - 0.5;
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let im = ((*state >> 33) % 1000) as f64 / 1000.0 - 0.5;
    Complex::new(re, im)
}
