//
// Project : KadanoffBaym
// File    : src/variants/moving.rs
// Purpose : moving (truncated) window container algebra, propagation
//           stages, and distances
//

//!
//! *Remarks* : *Moving Window*
//!
//! The moving container stores a truncated window of time steps.  The
//! window algebra (copying, scaling, left/right multiplication, and
//! distances) and the propagation stages (initialization, extrapolation,
//! and advance) live together here; the time-step solves live in
//! `dyson`/`vie2`.
//!

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, FftDomain, SteadyGreen, TimeFunction};
use crate::types::{MovingGreen, MovingGreenTimeStep, MovingTimeFunction};

use crate::algebra::frobenius_distance;

/// Return an owning copy (`MovingGreenTimeStep`) of the logical row `delti`.
pub fn copy_timestep(g: &MovingGreen, delti: i64) -> MovingGreenTimeStep {
    assert!(g.is_initialized(), "the moving object is not initialized");
    assert!(
        0 <= delti && delti <= g.tc,
        "delti {delti} outside 0..={}",
        g.tc
    );
    let mut x = MovingGreenTimeStep::new(g.tc, g.ret.size1, g.tstp - delti, g.sign);
    x.ret.raw_mut().copy_from_slice(g.ret.mrow(delti));
    x.less.raw_mut().copy_from_slice(g.less.mrow(delti));
    x
}

// ------------------------------------------------------------------
// algebra of the moving time steps
// ------------------------------------------------------------------

/// `dst[i] += a * src[i]` on both components of a moving time step.
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

/// Multiply both components of a moving time step by the scalar `a`.
pub fn smul_step(x: &mut MovingGreenTimeStep, a: Complex) {
    for c in [x.ret.raw_mut(), x.less.raw_mut()] {
        for z in c.iter_mut() {
            *z = a * *z;
        }
    }
}

/// Left multiplication of the leading time step of a moving Green's
/// function, `G(t, t - delt_j) <- f(t) G(t, t - delt_j)`, where `f(t)`
/// is the slot 0 of `f`. With `adj`, `f(t)^dagger` is used.
pub fn lmul_step(x: &mut MovingGreenTimeStep, f: &MovingTimeFunction, adj: bool) {
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
/// `f(t - delt_j)^dagger` is used.
pub fn rmul_step(x: &mut MovingGreenTimeStep, f: &MovingTimeFunction, adj: bool) {
    check_mfunc(x.tc, x.tstp, x.ret.size1, x.ret.size2, f);
    let d = x.ret.size1;
    rmul_rows(f, adj, [x.ret.raw_mut(), x.less.raw_mut()], d);
}

// ------------------------------------------------------------------
// distances
// ------------------------------------------------------------------

/// Distance between two moving time steps: the sum of the Frobenius
/// norms of the block differences of both components.
pub fn distance_moving(x: &MovingGreenTimeStep, y: &MovingGreenTimeStep) -> f64 {
    assert!(
        x.tc == y.tc && x.ret.size1 == y.ret.size1 && x.ret.size2 == y.ret.size2,
        "the cut-offs or matrix sizes differ"
    );
    let mut err = 0.0;
    let b = x.ret.size1 * x.ret.size2;
    for (xa, ya) in [(x.ret.raw(), y.ret.raw()), (x.less.raw(), y.less.raw())] {
        for jblk in 0..(xa.len() / b) {
            err += frobenius_distance(&xa[jblk * b..(jblk + 1) * b], &ya[jblk * b..(jblk + 1) * b]);
        }
    }
    err
}

/// Distance between the rows of the absolute time step `tstp` of two
/// moving Green's functions.
pub fn distance_moving_at(g1: &MovingGreen, g2: &MovingGreen, tstp: i64) -> f64 {
    assert!(
        g1.is_initialized() && g2.is_initialized(),
        "the moving object is not initialized"
    );
    let d1 = g1.tstp - tstp;
    let d2 = g2.tstp - tstp;
    assert!(
        0 <= d1 && d1 <= g1.tc && 0 <= d2 && d2 <= g2.tc,
        "distance_moving_at: tstp {tstp} outside the windows"
    );
    let b = g1.ret.size1 * g1.ret.size2;
    let mut err = 0.0;
    for (a, c) in [
        (g1.ret.mrow(d1), g2.ret.mrow(d2)),
        (g1.less.mrow(d1), g2.less.mrow(d2)),
    ] {
        for jblk in 0..(a.len() / b) {
            err += frobenius_distance(&a[jblk * b..(jblk + 1) * b], &c[jblk * b..(jblk + 1) * b]);
        }
    }
    err
}

// ------------------------------------------------------------------
// propagation stages
// ------------------------------------------------------------------

/// Initialize the window from the full contour Green's function `g` at
/// the 0-based time step `n0`: the row `delti` is the time step
/// `n0 - delti` of `g`,
/// `Gm^{R}(delti, deltj) = G^{R}(t_n, t_{n-deltj})` and
/// `Gm^{<}(delti, deltj) = -Gcc^{<}(t_{n-deltj}, t_n)^dagger` with
/// `n = n0 - delti`, for `deltj <= min(tc, n)`; the remaining blocks are
/// zero. `gcc` is the hermitian conjugate partner of `g` (`gcc = g` for
/// hermitian `g`). It requires `n0 >= tc` and `n0 <= g.nt` (and the same
/// for `gcc`). Afterwards `gm.tstp == n0`.
pub fn init_from_green(gm: &mut MovingGreen, g: &ContourGreen, gcc: &ContourGreen, n0: i64) {
    let tc = gm.tc;
    let d = gm.ret.size1;
    assert!(
        g.size1 == d && g.size2 == d && gcc.size1 == d && gcc.size2 == d,
        "the matrix sizes do not match"
    );
    assert!(
        n0 >= tc && n0 <= g.nt as i64 && n0 <= gcc.nt as i64,
        "init_from_green requires n0 >= tc and n0 <= nt"
    );
    assert!(g.sign == gm.sign, "the statistics of g and gm differ");
    gm.ret.raw_mut().fill(Complex::ZERO);
    gm.less.raw_mut().fill(Complex::ZERO);
    for delti in 0..=tc {
        let n = n0 - delti;
        for deltj in 0..=tc.min(n) {
            let m = n - deltj;
            gm.ret
                .mblock_mut(delti, deltj)
                .copy_from_slice(g.ret.raw_block_at(n, m));
            let s = gcc.less.raw_block_at(m, n);
            let b = gm.less.mblock_mut(delti, deltj);
            for a in 0..d {
                for bb in 0..d {
                    b[a * d + bb] = -s[bb * d + a].conj();
                }
            }
        }
    }
    gm.tstp = n0;
    gm.set_initialized(true);
    gm.bump_generation();
}

/// `init_from_green` with the hermitian partner `gcc = g`.
pub fn init_from_green_herm(gm: &mut MovingGreen, g: &ContourGreen, n0: i64) {
    init_from_green(gm, g, g, n0);
}

/// Initialize the slots from the single-time function `cf`: the slot
/// `delti` is `cf[n0 - delti]`. It requires `n0 >= tc` (the Matsubara
/// value never enters the window). Afterwards `f.tstp == n0`.
pub fn init_from_function(f: &mut MovingTimeFunction, cf: &TimeFunction, n0: i64) {
    assert!(
        cf.size1 == f.size1 && cf.size2 == f.size2,
        "the matrix sizes do not match"
    );
    assert!(
        n0 - f.tc >= 0 && n0 <= cf.nt as i64,
        "init_from_function requires n0 >= tc and n0 <= nt"
    );
    for delti in 0..=f.tc {
        f.fslot_mut(delti).copy_from_slice(cf.block(n0 - delti));
    }
    f.tstp = n0;
    f.set_initialized(true);
}

/// Initialize every row of the window with the time-translation
/// invariant steady state `gsteady`:
/// `G^{R/<}_m(delt_i, delt_j) = G^{R/<}_{steady}(delt_j h)` for
/// `delt_j <= min(tc, Nft/2 - 1)`, the remaining blocks are zero.
/// The relative steady step 0 is the 0-based time step 0, the first
/// `forward` leads to the step 1.
pub fn init_from_steady(gm: &mut MovingGreen, gsteady: &SteadyGreen) {
    let d = gm.ret.size1;
    assert!(
        gsteady.ret.size1 == d && gsteady.ret.size2 == d,
        "init_from_steady: matrix size mismatch"
    );
    let tc = gm.tc;
    let dtmax = tc.min(gsteady.grid.nft / 2 - 1);
    gm.ret.raw_mut().fill(Complex::ZERO);
    gm.less.raw_mut().fill(Complex::ZERO);
    for delti in 0..=tc {
        for dt in 0..=dtmax {
            let src = gsteady.ret.block(FftDomain::Time, dt).to_vec();
            gm.ret.mblock_mut(delti, dt).copy_from_slice(&src);
            let src = gsteady.les.block(FftDomain::Time, dt).to_vec();
            gm.less.mblock_mut(delti, dt).copy_from_slice(&src);
        }
    }
    gm.tstp = 0;
    gm.set_initialized(true);
    gm.bump_generation();
}

/// Move the window forward by one time step. The physical rows are
/// rotated, the new leading row (`delti = 0`) is set to zero, `tstp` is
/// incremented and every existing view becomes stale. No data are copied.
pub fn forward(g: &mut MovingGreen) {
    assert!(g.is_initialized(), "the moving object is not initialized");
    g.ret.rotate_forward();
    g.less.rotate_forward();
    g.tstp += 1;
    g.bump_generation();
}

// ------------------------------------------------------------------
// private helpers
// ------------------------------------------------------------------

/// Check that the single-time function `f` fits to the moving time
/// step `x`: initialized, the same matrix size, the window large enough,
/// and the leading time steps agree.
fn check_mfunc(x_tc: i64, x_tstp: i64, size1: usize, size2: usize, f: &MovingTimeFunction) {
    assert!(f.is_initialized(), "the moving object is not initialized");
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

/// `B <- F * B` for every block `B` of the two rows (private helper).
fn lmul_rows(f0: &DenseMat, rows: [&mut [Complex]; 2], d: usize) {
    let b = d * d;
    for row in rows {
        for jblk in 0..(row.len() / b) {
            let blk = &mut row[jblk * b..(jblk + 1) * b];
            let bm = DenseMat::from_slice(d, d, blk);
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
            let bm = DenseMat::from_slice(d, d, blk);
            let p = bm.mat_mul(&fj);
            blk.copy_from_slice(p.raw());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    use crate::types::Sign;

    #[test]
    fn moving_uninitialized_objects_are_rejected() {
        let mut g = MovingGreen::new(3, 2, Sign::Fermi);
        assert!(catch_unwind(|| g.ret_block(1, 1)).is_err());
        assert!(catch_unwind(|| g.less_block(1, 1)).is_err());
        assert!(catch_unwind(|| copy_timestep(&g, 0)).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| forward(&mut g))).is_err());
        let mut f = MovingTimeFunction::new(3, 2);
        assert!(catch_unwind(AssertUnwindSafe(|| f.forward())).is_err());
    }

    #[test]
    fn forward_requires_initialization() {
        let mut g = MovingGreen::new(3, 2, Sign::Fermi);
        assert!(catch_unwind(AssertUnwindSafe(|| forward(&mut g))).is_err());
    }
}
