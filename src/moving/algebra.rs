// Project : KadanoffBaym
// File    : src/moving/algebra.rs
// Purpose : algebra of the moving (truncated) time steps (blueprint
//           julia/src/traits/{incr,smul}.jl and
//           julia/src/{math/extrap,properties/distance}.jl, the
//           moving parts)
//
// The traits of the moving types act on one time step (a row of the
// window): an owning `MovingGreenTimeStep` or a logical row of a
// `MovingGreen`. The owning variants operate on the contiguous row
// slices directly; the row variants write through to the parent.

use crate::complex::Complex;
use crate::linalg::mat::DenseMat;
use crate::types::moving::{MovingTimeFunction, MovingGreen, MovingGreenTimeStep};
use crate::weights::Integrator;

/// Check that the single-time function `f` fits to the moving time
/// step `x` (blueprint `_check_mfunc`): initialized, the same matrix
/// size, the window large enough, and the leading time steps agree.
pub(crate) fn check_mfunc(x_tc: i64, x_tstp: i64, size1: usize, size2: usize, f: &MovingTimeFunction) {
    assert!(f.initialized, "the moving object is not initialized");
    assert!(
        f.size1 == size1 && f.size2 == size2,
        "the matrix sizes differ"
    );
    assert!(
        f.tc >= x_tc,
        "the cut-off of f must be >= the one of the time step"
    );
    assert!(
        x_tstp == f.tstp,
        "the time step {x_tstp} is not the leading step {} of f",
        f.tstp
    );
}

/// `dst[i] += a * src[i]` on both components of a moving time step
/// (blueprint `incr!` on `𝒻ᵐᵒᵛ`).
pub fn incr_step(dst: &mut MovingGreenTimeStep, src: &MovingGreenTimeStep, a: Complex) {
    assert!(
        dst.tc == src.tc && dst.ret.size1 == src.ret.size1 && dst.ret.size2 == src.ret.size2,
        "the cut-offs or matrix sizes differ"
    );
    for (x, s) in [
        (dst.ret.raw_mut(), src.ret.raw()),
        (dst.less.raw_mut(), src.less.raw()),
    ] {
        for q in 0..x.len() {
            x[q] += a * s[q];
        }
    }
}

/// Multiply both components of a moving time step by the scalar `a`
/// (blueprint `smul!` on `𝒻ᵐᵒᵛ`).
pub fn smul_step(x: &mut MovingGreenTimeStep, a: Complex) {
    for c in [x.ret.raw_mut(), x.less.raw_mut()] {
        for q in 0..c.len() {
            c[q] = a * c[q];
        }
    }
}

/// `B <- F * B` for every block `B` of the two rows (private helper).
fn lmul_rows(f0: &DenseMat, rows: [&mut [Complex]; 2], d: usize) {
    let b = d * d;
    for row in rows {
        for jblk in 0..(row.len() / b) {
            let blk = &mut row[jblk * b..(jblk + 1) * b];
            let bm = DenseMat::from_vec(d, d, blk.to_vec());
            let p = f0.mat_mul(&bm);
            blk.copy_from_slice(p.raw());
        }
    }
}

/// `B <- B * Fj` for every block `B` with the matching slot `Fj`
/// (private helper).
fn rmul_rows(f: &MovingTimeFunction, adj: bool, rows: [&mut [Complex]; 2], d: usize) {
    let b = d * d;
    for row in rows {
        for jblk in 0..(row.len() / b) {
            let src = f.fslot(jblk as i64).to_vec();
            let fj = if adj {
                let mut out = DenseMat::zeros(d);
                for a in 0..d {
                    for bb in 0..d {
                        out.set(a, bb, src[bb * d + a].conj());
                    }
                }
                out
            } else {
                DenseMat::from_vec(d, d, src)
            };
            let blk = &mut row[jblk * b..(jblk + 1) * b];
            let bm = DenseMat::from_vec(d, d, blk.to_vec());
            let p = bm.mat_mul(&fj);
            blk.copy_from_slice(p.raw());
        }
    }
}

/// Left multiplication of the leading time step of a moving Green's
/// function, `G(t, t - delt_j) <- f(t) G(t, t - delt_j)`, where `f(t)`
/// is the slot 0 of `f`. With `adj`, `f(t)^dagger` is used
/// (blueprint `smul!(f, x)` on the moving types).
pub fn lmul_step(f: &MovingTimeFunction, x: &mut MovingGreenTimeStep, adj: bool) {
    check_mfunc(x.tc, x.tstp, x.ret.size1, x.ret.size2, f);
    let d = x.ret.size1;
    let src = f.fslot(0).to_vec();
    let f0 = if adj {
        let mut out = DenseMat::zeros(d);
        for a in 0..d {
            for bb in 0..d {
                out.set(a, bb, src[bb * d + a].conj());
            }
        }
        out
    } else {
        DenseMat::from_vec(d, d, src)
    };
    lmul_rows(&f0, [x.ret.raw_mut(), x.less.raw_mut()], d);
}

/// Right multiplication of the leading time step of a moving Green's
/// function, `G(t, t - delt_j) <- G(t, t - delt_j) f(t - delt_j)`,
/// where `f(t - delt_j)` is the slot `deltj` of `f`. With `adj`,
/// `f(t - delt_j)^dagger` is used (blueprint `smul!(x, f)`).
pub fn rmul_step(f: &MovingTimeFunction, x: &mut MovingGreenTimeStep, adj: bool) {
    check_mfunc(x.tc, x.tstp, x.ret.size1, x.ret.size2, f);
    let d = x.ret.size1;
    rmul_rows(f, adj, [x.ret.raw_mut(), x.less.raw_mut()], d);
}

/// Extrapolate the leading row of the window from the rows 1, .., k+1
/// (blueprint `extrapolate_timestep!` on `ℱᵐᵒᵛ`). Requires
/// `tc >= 2k + 1`.
pub fn extrapolate_timestep_moving(g: &mut MovingGreen, integ: &Integrator) {
    assert!(g.initialized, "the moving object is not initialized");
    let k = integ.k;
    let tc = g.tc;
    assert!(
        tc >= 2 * k + 1,
        "extrapolation of order {k} requires tc >= {}, got {tc}",
        2 * k + 1
    );
    let p = crate::math::extrap::extrapolation_weights_order(k);
    let d = g.ret.size1;
    let b = d * d;
    let tcl = (tc + 1) as usize;
    for comp in 0..2 {
        // snapshot the rows 1, .., k+1 (the leading row is written
        // while the history rows are read)
        let mut hist: Vec<Complex> = vec![Complex::ZERO; (k + 1) as usize * tcl * b];
        for l in 0..=(k as usize) {
            let src = if comp == 0 {
                g.ret.mrow(l as i64 + 1)
            } else {
                g.less.mrow(l as i64 + 1)
            };
            hist[l * tcl * b..(l + 1) * tcl * b].copy_from_slice(src);
        }
        let lead = if comp == 0 {
            g.ret.mrow_mut(0)
        } else {
            g.less.mrow_mut(0)
        };
        for j in 0..=tc {
            // blueprint: jj = j < tc - k ? j : j - l - 1 (per l)
            let mut gtmp = vec![Complex::ZERO; b];
            for l in 0..=(k as usize) {
                let j_l = if j < tc - k {
                    j as usize
                } else {
                    (j - l as i64 - 1) as usize
                };
                let w = Complex::new(p[l], 0.0);
                let blk = &hist[l * tcl * b + j_l * b..l * tcl * b + (j_l + 1) * b];
                for q in 0..b {
                    gtmp[q] += w * blk[q];
                }
            }
            lead[j as usize * b..(j as usize + 1) * b].copy_from_slice(&gtmp);
        }
    }
}

/// `extrapolate_timestep_moving` with the order `order`; `order == 0`
/// copies the row 1 into the leading row (blueprint
/// `extrapolate_timestep!(G, order)`).
pub fn extrapolate_timestep_moving_order(g: &mut MovingGreen, order: i64) {
    assert!((0..=5).contains(&order), "order must be in 0:5, got {order}");
    if order == 0 {
        assert!(g.initialized, "the moving object is not initialized");
        assert!(g.tc >= 1, "extrapolation requires tc >= 1");
        let r1 = g.ret.mrow(1).to_vec();
        let l1 = g.less.mrow(1).to_vec();
        g.ret.mrow_mut(0).copy_from_slice(&r1);
        g.less.mrow_mut(0).copy_from_slice(&l1);
        return;
    }
    let integ = Integrator::new(order);
    extrapolate_timestep_moving(g, &integ);
}

/// Frobenius norm of the difference of two raw blocks.
fn blkdist(a: &[Complex], b: &[Complex]) -> f64 {
    let mut s = 0.0f64;
    for q in 0..a.len() {
        let dz = a[q] - b[q];
        s += dz.re * dz.re + dz.im * dz.im;
    }
    s.sqrt()
}

/// Distance between two moving time steps: the sum of the Frobenius
/// norms of the block differences of both components (blueprint
/// `distance(x, y)` on `𝒻ᵐᵒᵛ`).
pub fn distance_moving(x: &MovingGreenTimeStep, y: &MovingGreenTimeStep) -> f64 {
    assert!(
        x.tc == y.tc && x.ret.size1 == y.ret.size1 && x.ret.size2 == y.ret.size2,
        "the cut-offs or matrix sizes differ"
    );
    let mut err = 0.0;
    let b = x.ret.size1 * x.ret.size2;
    for (xa, ya) in [
        (x.ret.raw(), y.ret.raw()),
        (x.less.raw(), y.less.raw()),
    ] {
        for jblk in 0..(xa.len() / b) {
            err += blkdist(
                &xa[jblk * b..(jblk + 1) * b],
                &ya[jblk * b..(jblk + 1) * b],
            );
        }
    }
    err
}

/// Distance between the rows of the absolute time step `tstp` of two
/// moving Green's functions (blueprint `distance(G1, G2, tstp)`).
pub fn distance_moving_at(g1: &MovingGreen, g2: &MovingGreen, tstp: i64) -> f64 {
    assert!(g1.initialized && g2.initialized, "the moving object is not initialized");
    let d1 = g1.tstp - tstp;
    let d2 = g2.tstp - tstp;
    assert!(
        0 <= d1 && d1 <= g1.tc && 0 <= d2 && d2 <= g2.tc,
        "distance_moving_at: tstp {tstp} outside the windows"
    );
    let b = g1.ret.size1 * g1.ret.size2;
    let mut err = 0.0;
    for (a, c) in [(g1.ret.mrow(d1), g2.ret.mrow(d2)), (g1.less.mrow(d1), g2.less.mrow(d2))] {
        for jblk in 0..(a.len() / b) {
            err += blkdist(&a[jblk * b..(jblk + 1) * b], &c[jblk * b..(jblk + 1) * b]);
        }
    }
    err
}
