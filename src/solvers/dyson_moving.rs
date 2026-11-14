// Project : KadanoffBaym
// File    : src/solvers/dyson_moving.rs
// Purpose : the truncated (moving) Dyson equation at one time step
//           (blueprint julia/src/solvers/dyson_timestep.jl,
//           `_dyson_moving_*`)
//
// All kernels act on the relative window indices `(delti, deltj)`
// directly; the leading row is `delti = 0`. Only the retarded and
// lesser components are stored; the hermitian relations
// `G^{R/<}(t_n, t_m) = -G^{R/<}(t_m, t_n)^dagger` reconstruct the
// blocks with `n < m`.

use crate::complex::c64::C64;
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right_single, BlockSolver,
};
use crate::linalg::mat::{axpy_c, scale_c, DenseMat};
use crate::moving::layout::{check_hamiltonian_moving, check_moving_solver};
use crate::core::moving::{CfMoving, MovingGreen};
use crate::weights::Integrator;

const I: C64 = C64::new(0.0, 1.0);

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn dm(blk: &[C64], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, blk.to_vec())
}

/// `dst += w * (a . b)` for raw `d x d` blocks.
fn acc_mm(dst: &mut DenseMat, a: &[C64], w: C64, b: &[C64], d: usize) {
    let am = dm(a, d);
    let bm = dm(b, d);
    let mut p = am.mat_mul(&bm);
    for q in 0..(d * d) {
        p.raw_mut()[q] *= w;
    }
    axpy_c(dst, C64::ONE, &p);
}

/// The block `-src^dagger` as a `DenseMat` (the hermitian
/// reconstruction of the non-stored triangle).
fn adj_dm(src: &[C64], d: usize) -> DenseMat {
    let mut out = DenseMat::zeros(d);
    for a in 0..d {
        for b in 0..d {
            out.set(a, b, -src[b * d + a].conj());
        }
    }
    out
}

/// Retarded component of the truncated Dyson equation at the leading
/// row (blueprint `_dyson_moving_ret!`).
pub fn dyson_moving_ret(
    g: &mut MovingGreen,
    mu: f64,
    h: &CfMoving,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k as usize;
    let ki = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;

    // Clear the leading row, G^{R}(t, t) = -i
    let gr = &mut g.ret;
    let sr = &sigma.ret;
    for z in gr.mrow_mut(0) {
        *z = C64::ZERO;
    }
    let id = DenseMat::identity(d);
    {
        let g00 = gr.mblock_mut(0, 0);
        for a in 0..d {
            for b in 0..d {
                g00[a * d + b] = if a == b { -I } else { C64::ZERO };
            }
        }
    }
    let g00 = gr.mblock(0, 0).to_vec();

    // Start values G^{R}(t, t-j), j = 1, .., k: sum_q X_q M_{pq} = Q_p
    let mut mblocks = vec![vec![DenseMat::zeros(d); k]; k];
    let mut qblocks = vec![DenseMat::zeros(d); k];
    let mut xblocks = vec![DenseMat::zeros(d); k];
    for j in 1..=ki as usize {
        let jj = j as i64;
        // Derivatives
        for l in 0..=ki {
            let cweight = I / c(hstep, 0.0) * c(integ.pdw.at(j, l as usize), 0.0);
            if l == 0 {
                let src = dm(&g00, d);
                axpy_c(&mut qblocks[j - 1], -cweight, &src);
            } else {
                let mut m = id.clone();
                for q in 0..(d * d) {
                    m.raw_mut()[q] *= cweight;
                }
                axpy_c(&mut mblocks[j - 1][l as usize - 1], C64::ONE, &m);
            }
        }
        // Hamiltonian: M[j, j] -= H(j) - mu Id
        let mut hh = dm(h.fslot(jj), d);
        for a in 0..d {
            let v = hh.get(a, a) - c(mu, 0.0);
            hh.set(a, a, v);
        }
        axpy_c(&mut mblocks[j - 1][j - 1], -C64::ONE, &hh);
        // Integral
        for l in 0..=ki {
            let mut weight = c(hstep * integ.giw.at(j, l as usize), 0.0);
            let sblk: Vec<C64> = if jj >= l {
                sr.mblock(l, jj - l).to_vec()
            } else {
                // S = Sigma^{R}(j, l - j)^dagger, weight = -weight
                let src = sr.mblock(jj, l - jj);
                weight = -weight;
                let mut out = vec![C64::ZERO; d * d];
                for a in 0..d {
                    for b in 0..d {
                        out[a * d + b] = src[b * d + a].conj();
                    }
                }
                out
            };
            if l == 0 {
                // Q[j] += weight * G00 * S
                let sm = dm(&sblk, d);
                let gm = dm(&g00, d);
                let mut p = gm.mat_mul(&sm);
                for q in 0..(d * d) {
                    p.raw_mut()[q] *= weight;
                }
                axpy_c(&mut qblocks[j - 1], C64::ONE, &p);
            } else {
                let mut sm = dm(&sblk, d);
                for q in 0..(d * d) {
                    sm.raw_mut()[q] *= weight;
                }
                axpy_c(&mut mblocks[j - 1][l as usize - 1], -C64::ONE, &sm);
            }
        }
    }
    let mut ws = BlockSolver::new(k, d);
    solve_block_left(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for j in 1..=ki as usize {
        gr.mblock_mut(0, j as i64)
            .copy_from_slice(xblocks[j - 1].raw());
    }

    // The rest, j = k+1, .., tc: independent 1 x 1 problems M X = qq
    let diffw: Vec<C64> = (0..=(ki + 1) as usize)
        .map(|p| I / c(hstep, 0.0) * c(integ.bdw.at(p), 0.0))
        .collect();
    let mut mm = scale_c(&id,I / c(hstep, 0.0) * c(integ.bdw.at(0), 0.0) + c(mu, 0.0));
    {
        let mut t = dm(sr.mblock(0, 0), d);
        for q in 0..(d * d) {
            t.raw_mut()[q] *= c(-hstep * integ.giw.omega[0], 0.0);
        }
        axpy_c(&mut mm, C64::ONE, &t);
        let mut hh = dm(h.fslot(0), d);
        for q in 0..(d * d) {
            hh.raw_mut()[q] *= -C64::ONE;
        }
        axpy_c(&mut mm, C64::ONE, &hh);
    }
    dyson_moving_ret_far((ki + 1)..=tc, gr, sr, &mm, &diffw, hstep, integ);
}

/// Independent retarded blocks `G^{R}(t, t-j)`, `j > k`, of the
/// truncated Dyson equation. Only the history rows are read (blueprint
/// `_dyson_moving_ret_far!`).
fn dyson_moving_ret_far(
    js: std::ops::RangeInclusive<i64>,
    gr: &mut crate::core::moving::MovingComp,
    sr: &crate::core::moving::MovingComp,
    mm: &DenseMat,
    diffw: &[C64],
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = mm.nrows();
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    for j in js {
        let mut stemp = DenseMat::zeros(d);
        if j < 2 * k + 2 {
            for l in 1..=j {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    c(integ.giw.at(j as usize, l as usize), 0.0),
                    gr.mblock(l, j - l),
                    d,
                );
            }
        } else {
            for l in 1..=k {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    c(integ.giw.omega[l as usize], 0.0),
                    gr.mblock(l, j - l),
                    d,
                );
            }
            for l in (k + 1)..=(j - k - 1) {
                acc_mm(&mut stemp, sr.mblock(0, l), C64::ONE, gr.mblock(l, j - l), d);
            }
            for l in (j - k)..=j {
                acc_mm(
                    &mut stemp,
                    sr.mblock(0, l),
                    c(integ.giw.omega[(j - l) as usize], 0.0),
                    gr.mblock(l, j - l),
                    d,
                );
            }
        }
        qq.set_from(&stemp);
        for q in 0..(d * d) {
            qq.raw_mut()[q] *= c(hstep, 0.0);
        }
        for p in 1..=(k + 1) as usize {
            let src = dm(gr.mblock(p as i64, j - p as i64), d);
            axpy_c(&mut qq, -diffw[p], &src);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_right_single(&mut xj, mm, &qq, &mut ws1);
        gr.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Lesser component of the truncated Dyson equation at the leading
/// row. The retarded component of the leading row must be known
/// (blueprint `_dyson_moving_less!`).
pub fn dyson_moving_less(
    g: &mut MovingGreen,
    mu: f64,
    h: &CfMoving,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;
    let diffw: Vec<C64> = (0..=(k + 1) as usize)
        .map(|p| I / c(hstep, 0.0) * c(integ.bdw.at(p), 0.0))
        .collect();
    for z in g.less.mrow_mut(0) {
        *z = C64::ZERO;
    }

    let id = DenseMat::identity(d);
    let mut mm = scale_c(&id,I / c(hstep, 0.0) * c(integ.bdw.at(0), 0.0) + c(mu, 0.0));
    {
        let mut t = dm(sigma.ret.mblock(0, 0), d);
        for q in 0..(d * d) {
            t.raw_mut()[q] *= c(-hstep * integ.giw.omega[0], 0.0);
        }
        axpy_c(&mut mm, C64::ONE, &t);
        let mut hh = dm(h.fslot(0), d);
        for q in 0..(d * d) {
            hh.raw_mut()[q] *= -C64::ONE;
        }
        axpy_c(&mut mm, C64::ONE, &hh);
    }

    // G^{<}(t, t-j), j = tc, .., k+1, from the first time argument.
    // These blocks are independent, only the history rows are read.
    dyson_moving_less_far((k + 1)..=tc, &g.ret, &mut g.less, &sigma.ret, &sigma.less, &mm, &diffw, hstep, integ);

    dyson_moving_less_near(g, mu, h, sigma, hstep, integ, &diffw);
}

/// Independent lesser blocks `G^{<}(t, t-j)`, `j > k` (blueprint
/// `_dyson_moving_less_far!`).
#[allow(clippy::too_many_arguments)]
fn dyson_moving_less_far(
    js: std::ops::RangeInclusive<i64>,
    gr: &crate::core::moving::MovingComp,
    gl: &mut crate::core::moving::MovingComp,
    sr: &crate::core::moving::MovingComp,
    sl: &crate::core::moving::MovingComp,
    mm: &DenseMat,
    diffw: &[C64],
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let tc = gr.tc;
    let d = mm.nrows();
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    for j in js.rev() {
        // -int Sigma^{R}(t, t-l) G^{<}(t-l, t-j), l = 1, .., tc
        let mut stemp = DenseMat::zeros(d);
        for l in 1..=tc {
            let gtemp = if j >= l {
                dm(gl.mblock(l, j - l), d)
            } else {
                adj_dm(gl.mblock(j, l - j), d)
            };
            let sm = dm(sr.mblock(0, l), d);
            let mut p = sm.mat_mul(&gtemp);
            for q in 0..(d * d) {
                p.raw_mut()[q] *= c(integ.giw.at(tc as usize, l as usize), 0.0);
            }
            axpy_c(&mut stemp, C64::ONE, &p);
        }
        qq.set_from(&stemp);
        for q in 0..(d * d) {
            qq.raw_mut()[q] *= c(hstep, 0.0);
        }
        // int Sigma^{<}(t, s) G^{A}(s, t-j), s = t - tc + l
        let mut stemp = DenseMat::zeros(d);
        let dl1 = (tc - j).max(k);
        for l in 0..=dl1 {
            let gtemp = if tc - l >= j {
                // adjoint of G^{R}(j, tc - l - j)
                let src = gr.mblock(j, tc - l - j);
                let mut out = DenseMat::zeros(d);
                for a in 0..d {
                    for b in 0..d {
                        out.set(a, b, src[b * d + a].conj());
                    }
                }
                out
            } else {
                let mut t = dm(gr.mblock(tc - l, j - tc + l), d);
                for q in 0..(d * d) {
                    t.raw_mut()[q] *= -C64::ONE;
                }
                t
            };
            let sm = dm(sl.mblock(0, tc - l), d);
            let mut p = sm.mat_mul(&gtemp);
            for q in 0..(d * d) {
                p.raw_mut()[q] *= c(integ.giw.at((tc - j) as usize, l as usize), 0.0);
            }
            axpy_c(&mut stemp, C64::ONE, &p);
        }
        for q in 0..(d * d) {
            let v = qq.raw()[q] + c(hstep, 0.0) * stemp.raw()[q];
            qq.raw_mut()[q] = v;
        }
        // Derivative
        for p in 1..=(k + 1) as usize {
            let pi = p as i64;
            let gtemp = if j >= pi {
                dm(gl.mblock(pi, j - pi), d)
            } else {
                adj_dm(gl.mblock(j, pi - j), d)
            };
            axpy_c(&mut qq, -diffw[p], &gtemp);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_right_single(&mut xj, mm, &qq, &mut ws1);
        gl.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Lesser blocks `G^{<}(t, t-j)`, `j = k, .., 0`, solved sequentially
/// (blueprint `_dyson_moving_less_near!`).
#[allow(clippy::too_many_arguments)]
fn dyson_moving_less_near(
    g: &mut MovingGreen,
    mu: f64,
    h: &CfMoving,
    sigma: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
    diffw: &[C64],
) {
    let k = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;
    let id = DenseMat::identity(d);
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);

    // The sequence is important, the step j needs j+1, .., tc.
    for j in (0..=k).rev() {
        // int G^{R}(t, t-l) Sigma^{<}(t-l, t-j), l = 0, .., tc
        let mut stemp = DenseMat::zeros(d);
        for l in 0..=tc {
            let gtemp = if j > l {
                dm(sigma.less.mblock(l, j - l), d)
            } else {
                adj_dm(sigma.less.mblock(j, l - j), d)
            };
            acc_mm(
                &mut stemp,
                g.ret.mblock(0, l),
                c(integ.giw.at(tc as usize, l as usize), 0.0),
                gtemp.raw(),
                d,
            );
        }
        qq.set_from(&stemp);
        for q in 0..(d * d) {
            qq.raw_mut()[q] *= c(hstep, 0.0);
        }
        // int G^{<}(t, t-l) Sigma^{A}(t-l, t-j), l = tc, .., j+1
        let mut stemp = DenseMat::zeros(d);
        for l in ((j + 1)..=tc).rev() {
            // gtemp = Sigma^{R}(j, l - j)^dagger
            let src = sigma.ret.mblock(j, l - j);
            let mut gtemp = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    gtemp.set(a, b, src[b * d + a].conj());
                }
            }
            acc_mm(
                &mut stemp,
                g.less.mblock(0, l),
                c(integ.giw.at((tc - j) as usize, (l - j) as usize), 0.0),
                gtemp.raw(),
                d,
            );
        }
        for q in 0..(d * d) {
            let v = qq.raw()[q] + c(hstep, 0.0) * stemp.raw()[q];
            qq.raw_mut()[q] = v;
        }
        // Derivative -i d/dt'
        for p in 1..=(k + 1) as usize {
            let src = dm(g.less.mblock(0, j + p as i64), d);
            axpy_c(&mut qq, diffw[p], &src);
        }
        // mm2 = (-diffw[1] + mu) Id - h omega[1] Sigma^{R}(j, 0)^dag - H(j)^dag
        let mut mm2 = scale_c(&id, -diffw[0] + c(mu, 0.0));
        {
            let src = sigma.ret.mblock(j, 0);
            let mut t = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    t.set(a, b, -src[b * d + a].conj() * c(hstep * integ.giw.omega[0], 0.0));
                }
            }
            axpy_c(&mut mm2, C64::ONE, &t);
            let hsrc = h.fslot(j);
            let mut ht = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    ht.set(a, b, -hsrc[b * d + a].conj());
                }
            }
            axpy_c(&mut mm2, C64::ONE, &ht);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_left_single(&mut xj, &mm2, &qq, &mut ws1);
        g.less.mblock_mut(0, j).copy_from_slice(xj.raw());
    }
}

/// Solve the truncated Dyson equation for the leading row (0-based time
/// step `tstp`) of the moving Green's function `g`. `g`, `sigma`, and
/// `h` must be initialized, with the same cut-off
/// `tc >= 2 solve_order + 2` and leading time step `tstp`. The history
/// rows of `g` are not changed (blueprint `dyson_timestep!` on
/// `ℱᵐᵒᵛ`).
pub fn dyson_timestep_moving(
    tstp: i64,
    g: &mut MovingGreen,
    mu: f64,
    h: &CfMoving,
    sigma: &MovingGreen,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    check_moving_solver(tstp, k, &[g, sigma]);
    check_hamiltonian_moving(tstp, g, h);
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    dyson_moving_ret(g, mu, h, sigma, hstep, &integ);
    dyson_moving_less(g, mu, h, sigma, hstep, &integ);
}
