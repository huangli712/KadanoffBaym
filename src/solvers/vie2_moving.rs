// Project : KadanoffBaym
// File    : src/solvers/vie2_moving.rs
// Purpose : the truncated (moving) VIE2 at one time step (blueprint
//           julia/src/solvers/vie2_timestep.jl, `_vie2_moving_*`)
//
// The kernels act on the window-absolute steps `(a, b)` through the
// `_wret` / `_wles` accessors: `_wret(X, a, b)` is the block
// `X^{R}(t_a, t_b)` inside the window, the leading row is `a = tc`.

use crate::complex::c64::C64;
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    BlockSolver,
};
use crate::linalg::mat::{axpy_c, DenseMat};
use crate::moving::layout::check_moving_solver;
use crate::types::moving::MovingGreen;
use crate::weights::Integrator;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn dm(blk: &[C64], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, blk.to_vec())
}

/// `X^{R}(t_a, t_b)` inside the window (blueprint `_wret`).
fn wret<'a>(x: &'a MovingGreen, a: i64, b: i64) -> &'a [C64] {
    x.ret.mblock(x.tc - a, a - b)
}

/// `X^{<}(t_a, t_b)` inside the window (blueprint `_wles`).
fn wles<'a>(x: &'a MovingGreen, a: i64, b: i64) -> &'a [C64] {
    x.less.mblock(x.tc - a, a - b)
}

/// `src^dagger` of a raw block as a `DenseMat`.
fn adj_of(src: &[C64], d: usize) -> DenseMat {
    let mut out = DenseMat::zeros(d);
    for a in 0..d {
        for b in 0..d {
            out.set(a, b, src[b * d + a].conj());
        }
    }
    out
}

/// Retarded component of the truncated VIE2 at the leading row, the
/// right action equation with the kernel `K = fcc` and its partner
/// `Kcc = f` (blueprint `_vie2_moving_ret!`).
pub fn vie2_moving_ret(
    g: &mut MovingGreen,
    f: &MovingGreen,
    fcc: &MovingGreen,
    q: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k as usize;
    let ki = integ.k;
    let n = g.tc;
    let d = g.ret.size1;
    let id = DenseMat::identity(d);

    // Clear the leading row, G^{R}(n, n) = Q^{R}(n, n)
    for z in g.ret.mrow_mut(0) {
        *z = C64::ZERO;
    }
    g.ret.mblock_mut(0, 0).copy_from_slice(wret(q, n, n));
    let g00 = g.ret.mblock(0, 0).to_vec();

    // Start values G^{R}(n, n-j), j = 1, .., k
    let mut mblocks = vec![vec![DenseMat::zeros(d); k]; k];
    let mut rblocks = vec![DenseMat::zeros(d); k];
    let mut xblocks = vec![DenseMat::zeros(d); k];
    for j in 1..=ki as usize {
        let jj = j as i64;
        rblocks[j - 1] = dm(wret(q, n, n - jj), d);
        axpy_c(&mut mblocks[j - 1][j - 1], C64::ONE, &id);
        for l in 0..=ki {
            let mut weight = c(-hstep * integ.giw.at(j, l as usize), 0.0);
            let sblk: Vec<C64> = if l <= jj {
                // S = K^{R}(n - l, n - j)
                fcc.ret.mblock(l, jj - l).to_vec()
            } else {
                // S = Kcc^{R}(n - j, n - l)^dagger, weight = -weight
                let src = f.ret.mblock(jj, l - jj);
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
                // R[j] += weight * G00 * S
                let p = dm(&g00, d).mat_mul(&dm(&sblk, d));
                for z in 0..(d * d) {
                    let v = p.raw()[z] * weight;
                    let old = rblocks[j - 1].raw()[z];
                    rblocks[j - 1].raw_mut()[z] = old + v;
                }
            } else {
                let mut sm = dm(&sblk, d);
                for z in 0..(d * d) {
                    let v = sm.raw()[z] * -weight;
                    sm.raw_mut()[z] = v;
                }
                axpy_c(&mut mblocks[j - 1][l as usize - 1], C64::ONE, &sm);
            }
        }
    }
    let mut ws = BlockSolver::new(k, d);
    solve_block_left(&mut xblocks, &mblocks, &rblocks, &mut ws);
    for qq in 1..=ki as usize {
        g.ret
            .mblock_mut(0, qq as i64)
            .copy_from_slice(xblocks[qq - 1].raw());
    }

    // Contribution of G^{R}(n, m), m = n-k, .., n, to
    // -int dm G^{R}(n, m) K^{R}(m, j)
    let mut qqv: Vec<DenseMat> = vec![DenseMat::zeros(d); n as usize + 1];
    for m in (n - ki)..=n {
        let gnm = dm(wret(g, n, m), d);
        for j in 0..=(n - k as i64 - 1) {
            let w = c(-integ.giw.at((n - j) as usize, (n - m) as usize), 0.0);
            let p = gnm.mat_mul(&dm(wret(fcc, m, j), d));
            for z in 0..(d * d) {
                let v = p.raw()[z] * w;
                let old = qqv[j as usize].raw()[z];
                qqv[j as usize].raw_mut()[z] = old + v;
            }
        }
    }

    // The rest: j = n-k-1, .., 0
    let mut ws1 = BlockSolver::new(1, d);
    let w0 = c(hstep * integ.giw.omega[0], 0.0);
    for l in (ki + 1)..=n {
        let j = n - l;
        // qq[j] = h * qq[j] + Q^{R}(n, j)
        for z in 0..(d * d) {
            let v = c(hstep, 0.0) * qqv[j as usize].raw()[z] + wret(q, n, j)[z];
            qqv[j as usize].raw_mut()[z] = v;
        }
        // mm = Id + w0 * K^{R}(j, j)
        let mut mm = id.clone();
        axpy_c(&mut mm, w0, &dm(wret(fcc, j, j), d));
        let mut gnj = DenseMat::zeros(d);
        solve_block_left_single(&mut gnj, &mm, &qqv[j as usize], &mut ws1);
        g.ret
            .mblock_mut(0, n - j)
            .copy_from_slice(gnj.raw());
        for j2 in 0..j {
            let w = c(-integ.giw.at((n - j2) as usize, (n - j) as usize), 0.0);
            let p = gnj.mat_mul(&dm(wret(fcc, j, j2), d));
            for z in 0..(d * d) {
                let v = p.raw()[z] * w;
                let old = qqv[j2 as usize].raw()[z];
                qqv[j2 as usize].raw_mut()[z] = old + v;
            }
        }
    }
}

/// Lesser component of the truncated VIE2 at the leading row. The
/// retarded component of the leading row must be known (blueprint
/// `_vie2_moving_less!`).
pub fn vie2_moving_less(
    g: &mut MovingGreen,
    f: &MovingGreen,
    fcc: &MovingGreen,
    q: &MovingGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k as usize;
    let ki = integ.k;
    let n = g.tc;
    let d = g.ret.size1;
    let id = DenseMat::identity(d);

    // F^{<}(m, j) in the window, m >= j from F, otherwise from Fcc
    let fles = |m: i64, j: i64, out: &mut DenseMat| {
        if m >= j {
            for z in 0..(d * d) {
                out.raw_mut()[z] = wles(f, m, j)[z];
            }
        } else {
            let src = wles(fcc, j, m);
            for a in 0..d {
                for b in 0..d {
                    out.set(a, b, -src[b * d + a].conj());
                }
            }
        }
    };

    // Source term -[F^{<} * G^{A}](m, n), m = 0, .., n
    let mut gles: Vec<DenseMat> = vec![DenseMat::zeros(d); (n + 1) as usize];
    let mut ftmp = DenseMat::zeros(d);
    for j in 0..=n {
        let b = adj_of(wret(g, n, j), d);
        let w = c(-hstep * integ.giw.at(n as usize, j as usize), 0.0);
        fles(j, j, &mut ftmp); // placeholder; replaced below per m
        for m in 0..=n {
            fles(m, j, &mut ftmp);
            let p = ftmp.mat_mul(&b);
            for z in 0..(d * d) {
                let v = p.raw()[z] * w;
                let old = gles[m as usize].raw()[z];
                gles[m as usize].raw_mut()[z] = old + v;
            }
        }
    }

    // Q^{<}(j, n) = -Q^{<}(n, j)^dagger
    let qles = |j: i64| -> DenseMat {
        let src = wles(q, n, j);
        let mut out = DenseMat::zeros(d);
        for a in 0..d {
            for b in 0..d {
                out.set(a, b, -src[b * d + a].conj());
            }
        }
        out
    };

    // Start: j = 0, .., k, M X = R
    let mut mblocks = vec![vec![DenseMat::zeros(d); k + 1]; k + 1];
    let mut rblocks = vec![DenseMat::zeros(d); k + 1];
    let mut xblocks = vec![DenseMat::zeros(d); k + 1];
    for j in 0..=ki as usize {
        let ji = j as i64;
        axpy_c(&mut rblocks[j], C64::ONE, &gles[j]);
        axpy_c(&mut rblocks[j], C64::ONE, &qles(ji));
        axpy_c(&mut mblocks[j][j], C64::ONE, &id);
        for m in 0..=ki as usize {
            let mi = m as i64;
            let (sblk, s) = if ji >= mi {
                (wret(f, ji, mi).to_vec(), 1.0f64)
            } else {
                (adj_of(wret(fcc, mi, ji), d).raw().to_vec(), -1.0f64)
            };
            let mut sm = dm(&sblk, d);
            for z in 0..(d * d) {
                let v = sm.raw()[z] * c(s * hstep * integ.giw.at(j, m), 0.0);
                sm.raw_mut()[z] = v;
            }
            axpy_c(&mut mblocks[j][m], C64::ONE, &sm);
        }
    }
    let mut ws = BlockSolver::new(k + 1, d);
    solve_block_right(&mut xblocks, &mblocks, &rblocks, &mut ws);
    for j in 0..=ki as usize {
        gles[j].set_from(&xblocks[j]);
    }

    // Time stepping: j = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    for j in (ki + 1) as usize..=(n as usize) {
        let ji = j as i64;
        let mut qq = DenseMat::zeros(d);
        axpy_c(&mut qq, C64::ONE, &gles[j]);
        axpy_c(&mut qq, C64::ONE, &qles(ji));
        let mut mm = id.clone();
        axpy_c(
            &mut mm,
            c(hstep * integ.giw.at(j, j), 0.0),
            &dm(wret(f, ji, ji), d),
        );
        for m in 0..j {
            let p = dm(wret(f, ji, m as i64), d).mat_mul(&gles[m]);
            for z in 0..(d * d) {
                let v = p.raw()[z] * c(-hstep * integ.giw.at(j, m), 0.0);
                let old = qq.raw()[z];
                qq.raw_mut()[z] = old + v;
            }
        }
        solve_block_right_single(&mut gles[j], &mm, &qq, &mut ws1);
    }

    // Write the leading row, G^{<}(n, j) = -G^{<}(j, n)^dagger for
    // j < n; the diagonal block is stored as it is solved.
    for j in 0..n {
        let src = gles[j as usize].raw();
        let blk = g.less.mblock_mut(0, n - j);
        for a in 0..d {
            for b in 0..d {
                blk[a * d + b] = -src[b * d + a].conj();
            }
        }
    }
    g.less
        .mblock_mut(0, 0)
        .copy_from_slice(gles[n as usize].raw());
}

/// Solve the truncated VIE2 for the leading row (0-based time step
/// `tstp`) of the moving Green's function `g` (blueprint
/// `vie2_timestep!` on `ℱᵐᵒᵛ`).
pub fn vie2_timestep_moving(
    tstp: i64,
    g: &mut MovingGreen,
    f: &MovingGreen,
    fcc: &MovingGreen,
    q: &MovingGreen,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    check_moving_solver(tstp, k, &[g, f, fcc, q]);
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    vie2_moving_ret(g, f, fcc, q, hstep, &integ);
    vie2_moving_less(g, f, fcc, q, hstep, &integ);
}
