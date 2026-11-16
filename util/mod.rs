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

// ---------------------------------------------------------------------------
// shared deterministic RNG (splitmix64) and step helpers of the tests and
// the example programs
// ---------------------------------------------------------------------------

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, MATSUBARA};

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
    let re = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(re, im)
}

/// Uniform unit interval `f64`.
pub fn rand_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal `f64` (Box-Muller).
pub fn randn(state: &mut u64) -> f64 {
    let u1 = rand_unit(state).max(1e-300);
    let u2 = rand_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// the deterministic 2x2 test Hamiltonians of the t_dyson / t_vie suites
// ---------------------------------------------------------------------------

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// Blueprint `t_dyson` Hamiltonian `A`.
pub fn test_ha() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(-1.0, 0.0));
    m.set(0, 1, c(0.0, 0.3));
    m.set(1, 0, c(0.0, -0.3));
    m.set(1, 1, c(0.5, 0.0));
    m
}

/// Blueprint `t_dyson` Hamiltonian `B`.
pub fn test_hb() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(1.0, 0.0));
    m.set(0, 1, c(0.2, 0.1));
    m.set(1, 0, c(0.2, -0.1));
    m.set(1, 1, c(1.5, 0.0));
    m
}

/// Blueprint `t_dyson` interaction matrix `V`.
pub fn test_vv() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(0.4, 0.0));
    m.set(0, 1, c(0.0, 0.1));
    m.set(1, 0, c(-0.2, 0.0));
    m.set(1, 1, c(0.3, 0.0));
    m
}

/// Row-wise (ret, less) max-error of a moving window vs the full green at
/// step `n` (shared by the dyson/vie2 moving tests, blueprint
/// `dm_rowerr`).
pub fn moving_rowerr(
    gm: &kadanoffbaym::types::MovingGreen,
    g: &kadanoffbaym::types::ContourGreen,
    n: i64,
) -> (f64, f64) {
    let (mut er, mut el) = (0.0f64, 0.0f64);
    for j in 0..=gm.tc {
        if n - j < 0 {
            continue;
        }
        let gr = gm.ret_block(n, n - j);
        let fr = g.ret_block(n, n - j);
        for (x, y) in gr.iter().zip(fr.raw().iter()) {
            er = er.max((*x - *y).abs());
        }
        let gl = gm.less_block(n - j, n);
        let fl = g.less_block(n - j, n);
        for (x, y) in gl.iter().zip(fl.raw().iter()) {
            el = el.max((*x - *y).abs());
        }
    }
    (er, el)
}

// ---------------------------------------------------------------------------
// single-step green helpers shared by the example programs (blueprint
// `memcpy!` / the local `add_step` / `zeros!`)
// ---------------------------------------------------------------------------

/// Copy a single step of `src` into `dst` (blueprint `memcpy!`).
pub fn copy_green_step(src: &ContourGreen, dst: &mut ContourGreen, t: i64) {
    if t == MATSUBARA {
        dst.mat.memcpy_from(&src.mat);
    } else {
        for j in 0..=t {
            let v = src.ret.raw_block_at(t, j).to_vec();
            dst.ret.raw_block_at_mut(t, j).copy_from_slice(&v);
        }
        for i in 0..=t {
            let v = src.less.raw_block_at(i, t).to_vec();
            dst.less.raw_block_at_mut(i, t).copy_from_slice(&v);
        }
        for m in 0..=src.ntau as i64 {
            let v = src.lmix.raw_block_at(t, m).to_vec();
            dst.lmix.raw_block_at_mut(t, m).copy_from_slice(&v);
        }
    }
}

/// `dst += alpha * src` at a single step.
pub fn add_green_step(dst: &mut ContourGreen, src: &ContourGreen, t: i64, alpha: Complex) {
    if t == MATSUBARA {
        for q in 0..dst.mat.raw().len() {
            dst.mat.raw_mut()[q] += alpha * src.mat.raw()[q];
        }
    } else {
        for j in 0..=t {
            for q in 0..dst.ret.raw_block_at(t, j).len() {
                dst.ret.raw_block_at_mut(t, j)[q] += alpha * src.ret.raw_block_at(t, j)[q];
            }
        }
        for i in 0..=t {
            for q in 0..dst.less.raw_block_at(i, t).len() {
                dst.less.raw_block_at_mut(i, t)[q] += alpha * src.less.raw_block_at(i, t)[q];
            }
        }
        for m in 0..=dst.ntau as i64 {
            for q in 0..dst.lmix.raw_block_at(t, m).len() {
                dst.lmix.raw_block_at_mut(t, m)[q] += alpha * src.lmix.raw_block_at(t, m)[q];
            }
        }
    }
}

/// Zero the single step `t` of `g` (blueprint `zeros!`).
pub fn zero_green_step(g: &mut ContourGreen, t: i64) {
    if t == MATSUBARA {
        for q in g.mat.raw_mut() {
            *q = Complex::ZERO;
        }
    } else {
        for j in 0..=t {
            for q in g.ret.raw_block_at_mut(t, j) {
                *q = Complex::ZERO;
            }
        }
        for i in 0..=t {
            for q in g.less.raw_block_at_mut(i, t) {
                *q = Complex::ZERO;
            }
        }
        for m in 0..=g.ntau as i64 {
            for q in g.lmix.raw_block_at_mut(t, m) {
                *q = Complex::ZERO;
            }
        }
    }
}


pub fn arg_or(name: &str, default: i64) -> i64 {
    let args: Vec<String> = std::env::args().collect();
    let mut val = default;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == name
            && let Some(v) = it.next()
        {
            val = v.parse().unwrap_or(default);
        }
    }
    val
}

/// deterministic pseudo-random complex in (-0.5, 0.5); the same LCG
/// as the C++ side (values need not agree between the two programs,
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

