// Project : KadanoffBaym
// File    : src/dyson/moving.rs
// Purpose : the truncated (moving) Dyson equation at one time step
//           (`_dyson_moving_*`)
//
// All kernels act on the relative window indices `(delti, deltj)`
// directly; the leading row is `delti = 0`. Only the retarded and
// lesser components are stored; the hermitian relations
// `G^{R/<}(t_n, t_m) = -G^{R/<}(t_m, t_n)^dagger` reconstruct the
// blocks with `n < m`.

//!
//! *Remarks* : *Truncated Dyson Equation*
//!
//! The truncated (moving) Dyson equation determines the leading row
//! ``G(t, t - \delta_j)``, ``\delta_j = 0, \cdots, t_c``, of a moving Green's
//! function. The memory integrals are restricted to the window
//! ``[t - t_c, t]``, and the Matsubara and mixing components do not exist.
//! In the following, `G(i,j)` denotes the block `(delti, deltj)` of the
//! window, i.e., ``G(t - i, t - i - j)``.
//!
//! **Retarded component**: ``G^{R}(0,0) = -i``. The blocks `j = 1, ⋯, k`
//! are the solution of a 𝑘 × 𝑘 start-up system with polynomial
//! differentiation and Gregory weights. The blocks `j = k+1, ⋯, tc` are
//! obtained independently from
//!
//! ```math
//! \left[\frac{i}{h}b_0 + \mu - H(t) - h\omega_0\Sigma^{R}(t,t)\right]
//! G^{R}(t,t-j) = h\sum_{l=1}^{j} w_{jl}\Sigma^{R}(t,t-l)G^{R}(t-l,t-j)
//! - \frac{i}{h}\sum_{p=1}^{k+1} b_p G^{R}(t-p,t-j).
//! ```
//!
//! **Lesser component**: the blocks `j = tc, ⋯, k+1` are obtained
//! independently from the equation of motion in the first time argument
//! (the source contains ``\Sigma^{R} * G^{<}`` and ``\Sigma^{<} * G^{A}``
//! over the window), then the blocks `j = k, ⋯, 0` sequentially from the
//! equation of motion in the second time argument. The order of the second
//! part must not be changed.
//!
//! The weights are applied literally, e.g. ``\omega_0`` is used for
//! the diagonal term of every column. The mixed initial correlations
//! ``\Sigma^{\rceil} \star G^{\lceil}`` are not represented. Hence the result
//! agrees with the full Dyson equation only up to the time propagation
//! error, even if the window covers the whole history.

use std::ops::RangeInclusive;

use crate::complex::Complex;
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{MovingGreen, MovingTimeFunction, RingComp};
use crate::weights::Integrator;

use crate::base::MovingBlockScratch;
use crate::base::check_moving_solver;
use crate::dyson::workspace::MovingDysonWorkspace;
use crate::linalg::axpy_c;
use crate::linalg::{solve_block_left, solve_block_left_single, solve_block_right_single};
use crate::types::adjoint_block_into;

/// Retarded component of the truncated Dyson equation at the leading row.
fn dyson_moving_ret(
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut MovingDysonWorkspace,
) {
    let ki = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;

    // Clear the leading row, G^{R}(t, t) = -i
    let gr = &mut g.ret;
    let sr = &sigma.ret;
    for z in gr.mrow_mut(0) {
        *z = Complex::ZERO;
    }
    let id = &ws.id;
    {
        let g00 = gr.mblock_mut(0, 0);
        for a in 0..d {
            for b in 0..d {
                g00[a * d + b] = if a == b { -Complex::I } else { Complex::ZERO };
            }
        }
    }
    ws.g00.clear();
    ws.g00.extend_from_slice(gr.mblock(0, 0));
    let g00 = &ws.g00;

    // Start values G^{R}(t, t-j), j = 1, .., k: sum_q X_q M_{pq} = Q_p.
    // The blocks are accumulated, so they must be re-zeroed every call.
    for row in ws.blocks.mblocks.iter_mut().flatten() {
        row.raw_mut().fill(Complex::ZERO);
    }
    for blk in ws.blocks.rhsblocks.iter_mut() {
        blk.raw_mut().fill(Complex::ZERO);
    }
    for blk in ws.blocks.xblocks.iter_mut() {
        blk.raw_mut().fill(Complex::ZERO);
    }
    let MovingBlockScratch {
        mblocks,
        rhsblocks,
        xblocks,
        ws: block_ws,
    } = &mut ws.blocks;
    for j in 1..=ki as usize {
        let jj = j as i64;
        // Derivatives
        for l in 0..=ki {
            let cweight =
                Complex::I / Complex::from(hstep) * Complex::from(integ.pdw.at(j, l as usize));
            if l == 0 {
                let src = DenseMat::from_slice(d, d, g00);
                axpy_c(&mut rhsblocks[j - 1], -cweight, &src);
            } else {
                let mut m = id.clone();
                for q in 0..(d * d) {
                    m.raw_mut()[q] *= cweight;
                }
                axpy_c(&mut mblocks[j - 1][l as usize - 1], Complex::ONE, &m);
            }
        }
        // Hamiltonian: M[j, j] -= H(j) - mu Id
        let mut hh = DenseMat::from_slice(d, d, h.fslot(jj));
        for a in 0..d {
            let v = hh.get(a, a) - Complex::from(mu);
            hh.set(a, a, v);
        }
        axpy_c(&mut mblocks[j - 1][j - 1], -Complex::ONE, &hh);
        // Integral
        for l in 0..=ki {
            let mut weight = Complex::from(hstep * integ.giw.at(j, l as usize));
            let sblk: Vec<Complex> = if jj >= l {
                sr.mblock(l, jj - l).to_vec()
            } else {
                // S = Sigma^{R}(j, l - j)^dagger, weight = -weight
                let src = sr.mblock(jj, l - jj);
                weight = -weight;
                let mut out = vec![Complex::ZERO; d * d];
                adjoint_block_into(src, &mut out, d, d);
                out
            };
            if l == 0 {
                // Q[j] += weight * G00 * S
                let sm = DenseMat::from_slice(d, d, &sblk);
                let gm = DenseMat::from_slice(d, d, g00);
                let mut p = gm.mat_mul(&sm);
                for q in 0..(d * d) {
                    p.raw_mut()[q] *= weight;
                }
                axpy_c(&mut rhsblocks[j - 1], Complex::ONE, &p);
            } else {
                let mut sm = DenseMat::from_slice(d, d, &sblk);
                for q in 0..(d * d) {
                    sm.raw_mut()[q] *= weight;
                }
                axpy_c(&mut mblocks[j - 1][l as usize - 1], -Complex::ONE, &sm);
            }
        }
    }
    solve_block_left(xblocks, mblocks, rhsblocks, block_ws);
    for j in 1..=ki as usize {
        gr.mblock_mut(0, j as i64)
            .copy_from_slice(xblocks[j - 1].raw());
    }

    // The rest, j = k+1, .., tc: independent 1 x 1 problems M X = qq
    ws.diffw.clear();
    ws.diffw.extend(
        (0..=(ki + 1) as usize)
            .map(|p| Complex::I / Complex::from(hstep) * Complex::from(integ.bdw.at(p))),
    );
    let diffw = &ws.diffw;
    let mm = &mut ws.mm;
    mm.set_from(id);
    axpy_c(
        mm,
        Complex::I / Complex::from(hstep) * Complex::from(integ.bdw.at(0)) + Complex::from(mu)
            - Complex::ONE,
        id,
    );
    {
        let mut t = DenseMat::from_slice(d, d, sr.mblock(0, 0));
        for q in 0..(d * d) {
            t.raw_mut()[q] *= Complex::from(-hstep * integ.giw.omega[0]);
        }
        axpy_c(mm, Complex::ONE, &t);
        let mut hh = DenseMat::from_slice(d, d, h.fslot(0));
        for q in 0..(d * d) {
            hh.raw_mut()[q] *= -Complex::ONE;
        }
        axpy_c(mm, Complex::ONE, &hh);
    }
    let ws1 = &mut ws.ws1;
    dyson_moving_ret_far((ki + 1)..=tc, gr, sr, mm, diffw, hstep, integ, ws1);
}

/// Independent retarded blocks `G^{R}(t, t-j)`, `j > k`, of the
/// truncated Dyson equation. Only the history rows are read.
#[allow(clippy::too_many_arguments)]
fn dyson_moving_ret_far(
    js: RangeInclusive<i64>,
    gr: &mut RingComp,
    sr: &RingComp,
    mm: &DenseMat,
    diffw: &[Complex],
    hstep: f64,
    integ: &Integrator,
    ws1: &mut BlockSolver,
) {
    let k = integ.k;
    let d = mm.nrows();
    let mut qq = DenseMat::zeros(d);
    let mut stemp = DenseMat::zeros(d);
    for j in js {
        stemp.raw_mut().fill(Complex::ZERO);
        if j < 2 * k + 2 {
            for l in 1..=j {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    Complex::from(integ.giw.at(j as usize, l as usize)),
                    gr.mblock(l, j - l),
                    d,
                );
            }
        } else {
            for l in 1..=k {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    Complex::from(integ.giw.omega[l as usize]),
                    gr.mblock(l, j - l),
                    d,
                );
            }
            for l in (k + 1)..=(j - k - 1) {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    Complex::ONE,
                    gr.mblock(l, j - l),
                    d,
                );
            }
            for l in (j - k)..=j {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    Complex::from(integ.giw.omega[(j - l) as usize]),
                    gr.mblock(l, j - l),
                    d,
                );
            }
        }
        qq.set_from(&stemp);
        for z in qq.raw_mut() {
            *z *= Complex::from(hstep);
        }
        for (p, w) in diffw.iter().enumerate().skip(1).take(k as usize + 1) {
            let src = DenseMat::from_slice(d, d, gr.mblock(p as i64, j - p as i64));
            axpy_c(&mut qq, -*w, &src);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_right_single(&mut xj, mm, &qq, ws1);
        gr.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Lesser component of the truncated Dyson equation at the leading
/// row. The retarded component of the leading row must be known.
fn dyson_moving_less(
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut MovingDysonWorkspace,
) {
    let k = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;
    ws.diffw.clear();
    ws.diffw.extend(
        (0..=(k + 1) as usize)
            .map(|p| Complex::I / Complex::from(hstep) * Complex::from(integ.bdw.at(p))),
    );
    let diffw = &ws.diffw;
    for z in g.less.mrow_mut(0) {
        *z = Complex::ZERO;
    }

    let id = &ws.id;
    let mm = &mut ws.mm;
    mm.set_from(id);
    axpy_c(
        mm,
        Complex::I / Complex::from(hstep) * Complex::from(integ.bdw.at(0)) + Complex::from(mu)
            - Complex::ONE,
        id,
    );
    {
        let mut t = DenseMat::from_slice(d, d, sigma.ret.mblock(0, 0));
        for q in 0..(d * d) {
            t.raw_mut()[q] *= Complex::from(-hstep * integ.giw.omega[0]);
        }
        axpy_c(mm, Complex::ONE, &t);
        let mut hh = DenseMat::from_slice(d, d, h.fslot(0));
        for q in 0..(d * d) {
            hh.raw_mut()[q] *= -Complex::ONE;
        }
        axpy_c(mm, Complex::ONE, &hh);
    }

    // G^{<}(t, t-j), j = tc, .., k+1, from the first time argument.
    // These blocks are independent, only the history rows are read.
    let ws1 = &mut ws.ws1;
    let (ret, less) = (&g.ret, &mut g.less);
    dyson_moving_less_far(
        (k + 1)..=tc,
        ret,
        less,
        &sigma.ret,
        &sigma.less,
        mm,
        diffw,
        hstep,
        integ,
        ws1,
    );

    dyson_moving_less_near(g, mu, h, sigma, hstep, integ, ws);
}

/// Independent lesser blocks `G^{<}(t, t-j)`, `j > k`.
#[allow(clippy::too_many_arguments)]
fn dyson_moving_less_far(
    js: RangeInclusive<i64>,
    gr: &RingComp,
    gl: &mut RingComp,
    sr: &RingComp,
    sl: &RingComp,
    mm: &DenseMat,
    diffw: &[Complex],
    hstep: f64,
    integ: &Integrator,
    ws1: &mut BlockSolver,
) {
    let k = integ.k;
    let tc = gr.tc;
    let d = mm.nrows();
    let mut qq = DenseMat::zeros(d);
    for j in js.rev() {
        // -int Sigma^{R}(t, t-l) G^{<}(t-l, t-j), l = 1, .., tc
        let mut stemp = DenseMat::zeros(d);
        for l in 1..=tc {
            let gtemp = if j >= l {
                DenseMat::from_slice(d, d, gl.mblock(l, j - l))
            } else {
                adj_dm(gl.mblock(j, l - j), d)
            };
            let sm = DenseMat::from_slice(d, d, sr.mblock(0, l));
            let mut p = sm.mat_mul(&gtemp);
            for q in 0..(d * d) {
                p.raw_mut()[q] *= Complex::from(integ.giw.at(tc as usize, l as usize));
            }
            axpy_c(&mut stemp, Complex::ONE, &p);
        }
        qq.set_from(&stemp);
        for z in qq.raw_mut() {
            *z *= Complex::from(hstep);
        }
        // int Sigma^{<}(t, s) G^{A}(s, t-j), s = t - tc + l
        let mut stemp = DenseMat::zeros(d);
        let dl1 = (tc - j).max(k);
        for l in 0..=dl1 {
            let gtemp = if tc - l >= j {
                // adjoint of G^{R}(j, tc - l - j)
                let src = gr.mblock(j, tc - l - j);
                let mut out = DenseMat::zeros(d);
                adjoint_block_into(src, out.raw_mut(), d, d);
                out
            } else {
                let mut t = DenseMat::from_slice(d, d, gr.mblock(tc - l, j - tc + l));
                for q in 0..(d * d) {
                    t.raw_mut()[q] *= -Complex::ONE;
                }
                t
            };
            let sm = DenseMat::from_slice(d, d, sl.mblock(0, tc - l));
            let mut p = sm.mat_mul(&gtemp);
            for q in 0..(d * d) {
                p.raw_mut()[q] *= Complex::from(integ.giw.at((tc - j) as usize, l as usize));
            }
            axpy_c(&mut stemp, Complex::ONE, &p);
        }
        for (zq, zs) in qq.raw_mut().iter_mut().zip(stemp.raw().iter()) {
            *zq += Complex::from(hstep) * *zs;
        }
        // Derivative
        for (p, w) in diffw.iter().enumerate().skip(1).take(k as usize + 1) {
            let pi = p as i64;
            let gtemp = if j >= pi {
                DenseMat::from_slice(d, d, gl.mblock(pi, j - pi))
            } else {
                adj_dm(gl.mblock(j, pi - j), d)
            };
            axpy_c(&mut qq, -*w, &gtemp);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_right_single(&mut xj, mm, &qq, ws1);
        gl.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Lesser blocks `G^{<}(t, t-j)`, `j = k, .., 0`, solved sequentially.
#[allow(clippy::too_many_arguments)]
fn dyson_moving_less_near(
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut MovingDysonWorkspace,
) {
    let k = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;
    ws.diffw.clear();
    ws.diffw.extend(
        (0..=(k + 1) as usize)
            .map(|p| Complex::I / Complex::from(hstep) * Complex::from(integ.bdw.at(p))),
    );
    let diffw = &ws.diffw;
    let id = &ws.id;
    let ws1 = &mut ws.ws1;
    let mut qq = DenseMat::zeros(d);

    // The sequence is important, the step j needs j+1, .., tc.
    for j in (0..=k).rev() {
        // int G^{R}(t, t-l) Sigma^{<}(t-l, t-j), l = 0, .., tc
        let mut stemp = DenseMat::zeros(d);
        for l in 0..=tc {
            let gtemp = if j > l {
                DenseMat::from_slice(d, d, sigma.less.mblock(l, j - l))
            } else {
                adj_dm(sigma.less.mblock(j, l - j), d)
            };
            acc_mm(
                &mut stemp,
                g.ret.mblock(0, l),
                Complex::from(integ.giw.at(tc as usize, l as usize)),
                gtemp.raw(),
                d,
            );
        }
        qq.set_from(&stemp);
        for z in qq.raw_mut() {
            *z *= Complex::from(hstep);
        }
        // int G^{<}(t, t-l) Sigma^{A}(t-l, t-j), l = tc, .., j+1
        let mut stemp = DenseMat::zeros(d);
        for l in ((j + 1)..=tc).rev() {
            // gtemp = Sigma^{R}(j, l - j)^dagger
            let src = sigma.ret.mblock(j, l - j);
            let mut gtemp = DenseMat::zeros(d);
            adjoint_block_into(src, gtemp.raw_mut(), d, d);
            acc_mm(
                &mut stemp,
                g.less.mblock(0, l),
                Complex::from(integ.giw.at((tc - j) as usize, (l - j) as usize)),
                gtemp.raw(),
                d,
            );
        }
        for (zq, zs) in qq.raw_mut().iter_mut().zip(stemp.raw().iter()) {
            *zq += Complex::from(hstep) * *zs;
        }
        // Derivative -i d/dt'
        for (p, w) in diffw.iter().enumerate().skip(1).take(k as usize + 1) {
            let src = DenseMat::from_slice(d, d, g.less.mblock(0, j + p as i64));
            axpy_c(&mut qq, *w, &src);
        }
        // mm2 = (-diffw[1] + mu) Id - h omega[1] Sigma^{R}(j, 0)^dag - H(j)^dag
        let mm2 = &mut ws.mm2;
        mm2.set_from(id);
        axpy_c(mm2, -diffw[0] + Complex::from(mu) - Complex::ONE, id);
        {
            let src = sigma.ret.mblock(j, 0);
            let mut t = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    t.set(
                        a,
                        b,
                        -src[b * d + a].conj() * Complex::from(hstep * integ.giw.omega[0]),
                    );
                }
            }
            axpy_c(mm2, Complex::ONE, &t);
            let hsrc = h.fslot(j);
            let mut ht = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    ht.set(a, b, -hsrc[b * d + a].conj());
                }
            }
            axpy_c(mm2, Complex::ONE, &ht);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_left_single(&mut xj, mm2, &qq, ws1);
        g.less.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Solve the truncated Dyson equation for the leading row (0-based time
/// step `tstp`) of the moving Green's function `g`. `g`, `sigma`, and
/// `h` must be initialized, with the same cut-off
/// `tc >= 2 solve_order + 2` and leading time step `tstp`. The history
/// rows of `g` are not changed. Pass one [`MovingDysonWorkspace`] (and
/// the matching `Integrator`) built before the time-step loop; reuse is
/// valid while the moving `sigma` is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep_moving_ws(
    tstp: i64,
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut MovingDysonWorkspace,
) {
    let k = integ.k;
    check_moving_solver(tstp, k, &[g, sigma]);
    check_hamiltonian_moving(tstp, g, h);
    assert!(hstep > 0.0, "h must be positive");

    dyson_moving_ret(g, mu, h, sigma, hstep, integ, ws);
    dyson_moving_less(g, mu, h, sigma, hstep, integ, ws);
}

/// Check the moving Hamiltonian of the truncated Dyson solver:
/// initialized, `h.tc >= g.tc`, the leading time step, and the matrix size.
fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    assert!(h.initialized, "the moving object is not initialized");
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}

/// `dst += w * (a . b)` for raw `d x d` blocks.
fn acc_mm(dst: &mut DenseMat, a: &[Complex], w: Complex, b: &[Complex], d: usize) {
    let am = DenseMat::from_slice(d, d, a);
    let bm = DenseMat::from_slice(d, d, b);
    let mut p = am.mat_mul(&bm);
    for q in 0..(d * d) {
        p.raw_mut()[q] *= w;
    }
    axpy_c(dst, Complex::ONE, &p);
}

/// The block `-src^dagger` as a `DenseMat` (the hermitian
/// reconstruction of the non-stored triangle).
fn adj_dm(src: &[Complex], d: usize) -> DenseMat {
    let mut out = DenseMat::zeros(d);
    for a in 0..d {
        for b in 0..d {
            out.set(a, b, -src[b * d + a].conj());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::equilibrium::from_ham;
    use crate::linalg::DenseMat;
    use crate::types::{Contour, ContourGreen, Sign, TimeFunction};
    use crate::variants::{forward, init_from_function, init_from_green_herm};

    /// One workspace across the loop equals a fresh workspace per step.
    #[test]
    fn dyson_timestep_moving_ws_matches_fresh_workspace() {
        let (nt, ntau, beta, h, k, tc) = (14i64, 16i64, 4.0f64, 0.05, 3i64, 10i64);
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let h0 = DenseMat::from_rows(&[
            vec![Complex::new(-0.4, 0.0), Complex::new(0.0, 0.1)],
            vec![Complex::new(0.0, -0.1), Complex::new(0.6, 0.0)],
        ]);
        let mut hh = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        for t in -1..=nt {
            for i in 0..2 {
                for j in 0..2 {
                    hh.set(t, i, j, h0.get(i, j));
                }
            }
        }
        let mut s0 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut s0, 0.0, &h0, beta, h);
        let mut hmv = {
            let mut x = MovingTimeFunction::new(tc, 2);
            init_from_function(&mut x, &hh, tc);
            x
        };
        let mut g1 = MovingGreen::new(tc, 2, Sign::Fermi);
        let mut g2 = MovingGreen::new(tc, 2, Sign::Fermi);
        init_from_green_herm(&mut g1, &s0, tc);
        init_from_green_herm(&mut g2, &s0, tc);
        let mut sg = MovingGreen::new(tc, 2, Sign::Fermi);
        init_from_green_herm(&mut sg, &s0, tc);
        let integ = Integrator::new(k);
        let mut mws = MovingDysonWorkspace::new(k, 2);
        for _step in 0..4 {
            let tstp = g1.tstp;
            let fresh_integ = Integrator::new(k);
            let mut fresh_mws = MovingDysonWorkspace::new(k, 2);
            dyson_timestep_moving_ws(
                tstp,
                &mut g1,
                0.0,
                &hmv,
                &sg,
                h,
                &fresh_integ,
                &mut fresh_mws,
            );
            dyson_timestep_moving_ws(tstp, &mut g2, 0.0, &hmv, &sg, h, &integ, &mut mws);
            for g in [&mut g1, &mut g2, &mut sg] {
                forward(g);
            }
            hmv.forward();
            // refill the leading slot of H from the full function
            hmv.fslot_mut(0).copy_from_slice(hh.block(g1.tstp));
        }
        assert_eq!(g1.ret.raw(), g2.ret.raw());
        assert_eq!(g1.less.raw(), g2.less.raw());
    }
}
