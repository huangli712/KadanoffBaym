// Project : KadanoffBaym
// File    : util/green.rs
// Purpose : the single-step helpers of the contour containers shared
//           by the example programs (reference `memcpy!` / the local
//           `add_step` / `zeros!`), the Bethe lattice equilibrium, and
//           the moving-window row error of the dyson/vie2 tests
//

use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::MovingDysonWorkspace;
use kadanoffbaym::math::BetheDos;
use kadanoffbaym::types::{ContourGreen, MovingGreen, MovingTimeFunction};
use kadanoffbaym::vie2::MovingVie2Workspace;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::types::MATSUBARA;

use kadanoffbaym::dyson::dyson_moving_timestep;
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::vie2::vie2_moving_timestep;

// ------------------------------------------------------------------
// single-step helpers
// ------------------------------------------------------------------

/// Map a KB step number (0 = Matsubara) to the library step.
pub fn matsubara_step(n: i64) -> i64 {
    if n == 0 { MATSUBARA } else { n - 1 }
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

/// Zero the single step `t` of `g` .
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

// ------------------------------------------------------------------
// block norm
// ------------------------------------------------------------------

/// Frobenius norm of a block of components.
pub fn block_norm(block: &[Complex]) -> f64 {
    block.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt()
}

// ------------------------------------------------------------------
// Bethe lattice equilibrium
// ------------------------------------------------------------------

/// The noninteracting equilibrium Green's function of the Bethe lattice
/// of half width 2 at `mu = 0` (the reference `from_dos_mat_bethe!`, `V = 1`
/// and `E0 = 0`), on the whole contour of `g` at the inverse temperature
/// `beta`.
pub fn bethe_lattice_equilibrium(g: &mut ContourGreen, beta: f64) {
    let rho = BetheDos::new(-2.0, 2.0);
    from_dos_mat(
        g,
        &|eps: f64| rho.at(eps),
        rho.lo,
        rho.hi,
        beta,
        0.0,
        100,
        20,
    );
}

// ------------------------------------------------------------------
// moving-window row error
// ------------------------------------------------------------------

/// Row-wise (ret, less) max-error of a moving window vs the full green at
/// step `n` (shared by the dyson/vie2 moving tests, reference
/// `dm_rowerr`).
pub fn moving_row_error(gm: &MovingGreen, g: &ContourGreen, n: i64) -> (f64, f64) {
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

// ------------------------------------------------------------------
// moving one-shot steps
// ------------------------------------------------------------------

/// One-shot moving VIE2 step: builds a fresh `Integrator` and a fresh
/// workspace for the call.
pub fn vie2_moving_step(
    tstp: i64,
    g: &mut MovingGreen,
    f: &MovingGreen,
    fcc: &MovingGreen,
    q: &MovingGreen,
    hstep: f64,
    k: i64,
) {
    let integ = Integrator::new(k);
    let mut ws = MovingVie2Workspace::new(k, g.ret.size1);
    vie2_moving_timestep(tstp, g, f, fcc, q, hstep, &integ, &mut ws);
}

/// One-shot moving Dyson step: builds a fresh `Integrator` and a fresh
/// workspace for the call.
pub fn dyson_moving_step(
    tstp: i64,
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    k: i64,
) {
    let integ = Integrator::new(k);
    let mut ws = MovingDysonWorkspace::new(k, g.ret.size1);
    dyson_moving_timestep(tstp, g, mu, h, sigma, hstep, &integ, &mut ws);
}
