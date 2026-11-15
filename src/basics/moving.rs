// Project : KadanoffBaym
// File    : src/basics/moving.rs
// Purpose : the moving (truncated) window layer: window lifecycle,
//           moving-timestep algebra, and the moving solver checks
//           (blueprint julia/src/moving/{window,layout}.jl and the
//           moving parts of traits/{incr,smul}.jl)
//
// Merged from the former src/moving/ folder: the moving types live in
// types/moving.rs, the moving solvers in dyson/moving.rs and
// vie2/moving.rs, the moving bubbles in convolution/bubble.rs.

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, Sign, SteadyGreen, TimeFunction};
use crate::types::{MovingGreen, MovingGreenTimeStep, MovingTimeFunction};
use crate::weights::Integrator;

// ------------------------------------------------------------------
// window lifecycle
// ------------------------------------------------------------------

/// Initialize the window from the full contour Green's function `g` at
/// the 0-based time step `n0`: the row `delti` is the time step
/// `n0 - delti` of `g`,
/// `Gm^{R}(delti, deltj) = G^{R}(t_n, t_{n-deltj})` and
/// `Gm^{<}(delti, deltj) = -Gcc^{<}(t_{n-deltj}, t_n)^dagger` with
/// `n = n0 - delti` and `deltj <= n` (the remaining blocks are zero).
/// `gcc` is the hermitian conjugate partner of `g` (`gcc = g` for
/// hermitian `g`). It requires `n0 >= tc` and `n0 <= g.nt`. Afterwards
/// `gm.tstp == n0` (blueprint `init_from_G!`).
///
/// Initialize the window from the full contour Green's function `G` at the
/// KB time step `n₀`: the row `delti` is the time step `n₀ - delti` of `G`,
///
/// ```math
/// G_m^{R}(\\delta_i, \\delta_j) = G^{R}(t_{n}, t_{n-\\delta_j}),\\quad
/// G_m^{<}(\\delta_i, \\delta_j) = -G_{cc}^{<}(t_{n-\\delta_j}, t_n)^{\\dagger},
/// ```
///
/// with `n = n₀ - delti` and `deltj ≤ n - 1` (the remaining blocks are
/// zero). `Gcc` is the hermitian conjugate of `G` (`Gcc = G` for hermitian
/// `G`). It requires `n₀ - tc ≥ 1`. Afterwards `Gm.tstp = n₀`.
///
/// *References* :
///
/// Please see [`NESSi`] `herm_matrix_moving::set_from_G_backward` and
/// `herm_matrix_timestep_moving_view::set_timestep`.
pub fn init_from_g(gm: &mut MovingGreen, g: &ContourGreen, gcc: &ContourGreen, n0: i64) {
    let tc = gm.tc;
    let d = gm.ret.size1;
    assert!(
        g.size1 == d && g.size2 == d && gcc.size1 == d && gcc.size2 == d,
        "the matrix sizes do not match"
    );
    assert!(
        n0 >= tc && n0 <= g.nt as i64 && n0 <= gcc.nt as i64,
        "init_from_g requires n0 >= tc and n0 <= nt"
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
    gm.initialized = true;
    gm.generation += 1;
}

/// `init_from_g` with the hermitian partner `gcc = g` (blueprint
/// `init_from_G!(Gm, G, n0)`).
pub fn init_from_g_herm(gm: &mut MovingGreen, g: &ContourGreen, n0: i64) {
    init_from_g(gm, g, g, n0);
}

/// Initialize the slots from the single-time function `cf`: the slot
/// `delti` is `cf[n0 - delti]`. It requires `n0 >= tc` (the Matsubara
/// value never enters the window). Afterwards `f.tstp == n0`
/// (blueprint `init_from_function!`).
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
    f.initialized = true;
}

/// Initialize every row of the window with the time-translation
/// invariant steady state `gness` (blueprint `init_from_ness!`):
/// `G^{R/<}_m(delt_i, delt_j) = G^{R/<}_{ness}(delt_j h)` for
/// `delt_j <= min(tc, Nft/2 - 1)`, the remaining blocks are zero.
/// The relative NESS step 0 is the 0-based time step 0, the first
/// `forward` leads to the step 1.
pub fn init_from_ness(gm: &mut MovingGreen, gness: &SteadyGreen) {
    use crate::types::FftDomain;
    let d = gm.ret.size1;
    assert!(
        gness.ret.size1 == d && gness.ret.size2 == d,
        "init_from_ness: matrix size mismatch"
    );
    let tc = gm.tc;
    let dtmax = tc.min(gness.grid.nft / 2 - 1);
    gm.ret.raw_mut().fill(Complex::ZERO);
    gm.less.raw_mut().fill(Complex::ZERO);
    for delti in 0..=tc {
        for dt in 0..=dtmax {
            let src = gness.ret.block(FftDomain::Time, dt).to_vec();
            gm.ret.mblock_mut(delti, dt).copy_from_slice(&src);
            let src = gness.les.block(FftDomain::Time, dt).to_vec();
            gm.less.mblock_mut(delti, dt).copy_from_slice(&src);
        }
    }
    gm.tstp = 0;
    gm.initialized = true;
    gm.generation += 1;
}

/// Move the window forward by one time step. The physical rows are
/// rotated, the new leading row (`delti = 0`) is set to zero, `tstp` is
/// incremented and every existing view becomes stale. No data are
/// copied (blueprint `forward!`).
pub fn forward(g: &mut MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
    g.ret.rotate_forward();
    g.less.rotate_forward();
    g.tstp += 1;
    g.generation += 1;
}

/// Return an owning copy (`MovingGreenTimeStep`) of the logical row
/// `delti` (blueprint `copy_timestep`).
pub fn copy_timestep(g: &MovingGreen, delti: i64) -> MovingGreenTimeStep {
    assert!(g.initialized, "the moving object is not initialized");
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
        for z in c.iter_mut() {
            *z = a * *z;
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
        tc > 2 * k,
        "extrapolation of order {k} requires tc >= {}, got {tc}",
        2 * k + 1
    );
    let p = crate::math::extrapolation_weights_order(k);
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
    assert!(
        (0..=5).contains(&order),
        "order must be in 0:5, got {order}"
    );
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
    for (xa, ya) in [(x.ret.raw(), y.ret.raw()), (x.less.raw(), y.less.raw())] {
        for jblk in 0..(xa.len() / b) {
            err += blkdist(&xa[jblk * b..(jblk + 1) * b], &ya[jblk * b..(jblk + 1) * b]);
        }
    }
    err
}

/// Distance between the rows of the absolute time step `tstp` of two
/// moving Green's functions (blueprint `distance(G1, G2, tstp)`).
pub fn distance_moving_at(g1: &MovingGreen, g2: &MovingGreen, tstp: i64) -> f64 {
    assert!(
        g1.initialized && g2.initialized,
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
            err += blkdist(&a[jblk * b..(jblk + 1) * b], &c[jblk * b..(jblk + 1) * b]);
        }
    }
    err
}

// ------------------------------------------------------------------
// sanity checks of the moving solvers
// ------------------------------------------------------------------

/// Check the moving objects of the truncated solvers: initialized, the
/// same cut-off, the leading time step `tstp`, the matrix size, and
/// `tc >= 2k + 2` (blueprint `_check_moving_solver`).
pub(crate) fn check_moving_solver(tstp: i64, k: i64, objs: &[&MovingGreen]) {
    assert!((1..=5).contains(&k), "solve_order must be in 1:5, got {k}");
    for x in objs {
        assert!(x.initialized, "the moving object is not initialized");
        assert!(x.tc == objs[0].tc, "the cut-offs differ");
        assert!(
            x.tstp == tstp,
            "tstp = {tstp} is not the leading time step {}",
            x.tstp
        );
        assert!(
            x.ret.size1 == objs[0].ret.size1 && x.ret.size2 == objs[0].ret.size2,
            "the matrix sizes differ"
        );
    }
    assert!(
        objs[0].tc >= 2 * k + 2,
        "the truncated solver of order {k} requires tc >= {}, got {}",
        2 * k + 2,
        objs[0].tc
    );
}

/// Check the moving Hamiltonian of the truncated Dyson solver
/// (blueprint part of `dyson_timestep!`): initialized, `h.tc >= g.tc`,
/// the leading time step, and the matrix size.
pub(crate) fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    check_initialized_cfmoving(h);
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}

// ------------------------------------------------------------------
// private helpers
// ------------------------------------------------------------------

/// Check that the single-time function `f` fits to the moving time
/// step `x` (blueprint `_check_mfunc`): initialized, the same matrix
/// size, the window large enough, and the leading time steps agree.
fn check_mfunc(x_tc: i64, x_tstp: i64, size1: usize, size2: usize, f: &MovingTimeFunction) {
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

/// Frobenius norm of the difference of two raw blocks.
fn blkdist(a: &[Complex], b: &[Complex]) -> f64 {
    let mut s = 0.0f64;
    for q in 0..a.len() {
        let dz = a[q] - b[q];
        s += dz.re * dz.re + dz.im * dz.im;
    }
    s.sqrt()
}

#[allow(dead_code)]
/// Throw if the moving object `g` is not initialized (blueprint
/// `_check_initialized` for `ℱᵐᵒᵛ`).
fn check_initialized_moving(g: &MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
}

/// Throw if the moving single-time function is not initialized.
fn check_initialized_cfmoving(f: &MovingTimeFunction) {
    assert!(f.initialized, "the moving object is not initialized");
}

#[allow(dead_code)]
/// The sign of a moving object (convenience for generic checks).
fn getsign_moving(g: &MovingGreen) -> Sign {
    g.sign
}
