// Project : KadanoffBaym
// File    : util/green.rs
// Purpose : the single-step helpers of the contour containers shared
//           by the example programs (blueprint `memcpy!` / the local
//           `add_step` / `zeros!`), and the moving-window row error of
//           the dyson/vie2 tests
//

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, MATSUBARA};

// ------------------------------------------------------------------
// single-step helpers
// ------------------------------------------------------------------

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

// ------------------------------------------------------------------
// moving-window row error
// ------------------------------------------------------------------

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
