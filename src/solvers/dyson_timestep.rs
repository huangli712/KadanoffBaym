// Project : KadanoffBaym
// File    : src/solvers/dyson_timestep.rs
// Purpose : time stepping of the real time Dyson components (blueprint:
//           solvers/dyson_timestep.jl; the moving-window variants are
//           R6 scope)
//
// All time indices follow the NESSi convention (0-based real times).

use crate::complex::c64::C64;
use crate::convolution::less::{
    incr_convolution_less_adv_step, incr_convolution_lmix_rmix_step,
};
use crate::convolution::lmix::incr_convolution_lmix_step;
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    BlockSolver,
};
use crate::linalg::mat::{axpy_c, DenseMat};
use crate::core::contour::Contour;
use crate::core::green::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::weights::Integrator;

const I: C64 = C64::new(0.0, 1.0);

fn dm(v: &[C64], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, v.to_vec())
}

fn blocks_to_vec(x: &[DenseMat]) -> Vec<Vec<C64>> {
    x.iter().map(|m| m.raw().to_vec()).collect()
}

/// `Sigma^R(n, l)` (NESSi indices) and the sign of the weight; for
/// `n < l` the adjoint with the sign `-1` is returned (blueprint
/// `_sigma_ret`).
pub(crate) fn sigma_ret(sigma: &ContourGreen, n: i64, l: i64) -> (Vec<C64>, f64) {
    if n >= l {
        (sigma.ret.raw_block_at(n, l).to_vec(), 1.0)
    } else {
        (
            crate::convolution::adjoint_block(
                sigma.ret.raw_block_at(l, n),
                sigma.ret.size1,
                sigma.ret.size2,
            ),
            -1.0,
        )
    }
}

/// Solve the lesser component `G^<(j, n)` for `j = 0, .., n` (NESSi
/// indices). On entry `gles[j]` contains the source term
/// `[Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)`; on exit the
/// solution (also written into `g`) (blueprint `_dyson_less_core!`).
#[allow(clippy::too_many_arguments)]
pub fn dyson_less_core(
    n: i64,
    gles: &mut [Vec<C64>],
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    assert!(gles.len() >= (n.max(k) + 1) as usize);

    // initial value G^<(0, n) = -G^rangle(n, 0)^dagger
    let g0: Vec<C64> = crate::convolution::adjoint_block(g.lmix.raw_block_at(n, 0), g.size1, g.size2)
        .iter()
        .map(|z| -*z)
        .collect();
    gles[0] = g0.clone();

    // start: j = 1, .., k, solve a k x k problem M X = Q
    let mut ws = BlockSolver::new(k as usize, d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    let id = DenseMat::identity(d);
    for j in 1..=k as i64 {
        let mut qj = dm(&gles[j as usize], d);
        // derivative
        let cw = I / C64::new(hstep, 0.0) * C64::new(integ.pdw.at(j as usize, 0), 0.0);
        axpy_c(&mut qj, -cw, &dm(&g0, d));
        // integral Sigma^R * G^< (j = 0 term)
        let w0 = hstep * integ.giw.at(j as usize, 0);
        let s0 = sigma.ret.raw_block_at(j, 0);
        let prod = dm(s0, d).mat_mul(&dm(&g0, d));
        axpy_c(&mut qj, C64::new(w0, 0.0), &prod);
        qblocks[(j - 1) as usize] = qj;
        for m in 1..=k as i64 {
            let cw =
                I / C64::new(hstep, 0.0) * C64::new(integ.pdw.at(j as usize, m as usize), 0.0);
            let mut mj = crate::linalg::mat::scale_c(&id, cw);
            if m == j {
                // Hamiltonian
                axpy_c(&mut mj, C64::new(mu, 0.0), &id);
                let hj = h.block(j);
                axpy_c(&mut mj, C64::new(-1.0, 0.0), &dm(hj, d));
            }
            // integral Sigma^R * G^<
            let (sblk, s) = sigma_ret(sigma, j, m);
            let wm = s * hstep * integ.giw.at(j as usize, m as usize);
            axpy_c(&mut mj, C64::new(-wm, 0.0), &dm(&sblk, d));
            mblocks[(j - 1) as usize][(m - 1) as usize] = mj;
        }
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    let sols = blocks_to_vec(&xblocks);
    for j in 1..=k as i64 {
        gles[j as usize] = sols[(j - 1) as usize].clone();
    }

    // time stepping: j = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    for j in (k + 1)..=n {
        let mut qq = dm(&gles[j as usize], d);
        for p in 1..=(k + 1) as i64 {
            let w = -I / C64::new(hstep, 0.0) * C64::new(integ.bdw.at(p as usize), 0.0);
            axpy_c(&mut qq, w, &dm(&gles[(j - p) as usize], d));
        }
        // M = i b_0 / h + mu - H(j) - h Omega_{jj} Sigma^R(j,j)
        let mut mm = crate::linalg::mat::scale_c(
        &id,
        I / C64::new(hstep, 0.0) * C64::new(integ.bdw.at(0), 0.0) + C64::new(mu, 0.0),
    );
        axpy_c(&mut mm, C64::new(-1.0, 0.0), &dm(h.block(j), d));
        let wj = hstep * integ.giw.at(j as usize, j as usize);
        let sj = sigma.ret.raw_block_at(j, j);
        axpy_c(&mut mm, C64::new(-wj, 0.0), &dm(sj, d));
        for m in 0..j {
            let w = hstep * integ.giw.at(j as usize, m as usize);
            let sm = sigma.ret.raw_block_at(j, m);
            let prod = dm(sm, d).mat_mul(&dm(&gles[m as usize], d));
            axpy_c(&mut qq, C64::new(w, 0.0), &prod);
        }
        let mut xj = DenseMat::zeros(d);
        solve_block_right_single(&mut xj, &mm, &qq, &mut ws1);
        gles[j as usize] = xj.raw().to_vec();
    }

    // write the stored triangle
    for j in 0..=n {
        g.less
            .raw_block_at_mut(j, n)
            .copy_from_slice(&gles[j as usize]);
    }
}

/// Time stepping of the retarded component `G^R(n, j)` for `0 <= j <= n`
/// with `n > k` (blueprint `_dyson_timestep_ret!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_ret(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let id = DenseMat::identity(d);

    // clear the time step, G^R(n,n) = -i
    for j in 0..=n {
        g.ret.raw_block_at_mut(n, j).fill(C64::ZERO);
    }
    let gnn: Vec<C64> = (0..d * d)
        .map(|q| if q % (d + 1) == 0 { C64::new(0.0, -1.0) } else { C64::ZERO })
        .collect();
    let gnn_dm = dm(&gnn, d);
    g.ret.raw_block_at_mut(n, n).copy_from_slice(&gnn);

    // start values G^R(n, n-q), q = 1, .., k: solve sum_q X_q M_{qj} = Q_j
    let mut wsk = BlockSolver::new(k as usize, d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    for j in 1..=k as i64 {
        let mut qj = DenseMat::zeros(d);
        // derivatives
        for l in 0..=k as i64 {
            let cw = I / C64::new(hstep, 0.0) * C64::new(integ.pdw.at(j as usize, l as usize), 0.0);
            if l == 0 {
                axpy_c(&mut qj, -cw, &gnn_dm);
            } else {
                axpy_c(&mut mblocks[(j - 1) as usize][(l - 1) as usize], cw, &id);
            }
        }
        // Hamiltonian
        axpy_c(&mut mblocks[(j - 1) as usize][(j - 1) as usize], C64::new(mu, 0.0), &id);
        axpy_c(&mut mblocks[(j - 1) as usize][(j - 1) as usize],
            C64::new(-1.0, 0.0),
            &dm(h.block(n - j), d),
        );
        // integral
        for l in 0..=k as i64 {
            let mut weight = hstep * integ.giw.at(j as usize, l as usize);
            let s_blk = if l <= j {
                sigma.ret.raw_block_at(n - l, n - j).to_vec()
            } else {
                weight = -weight;
                crate::convolution::adjoint_block(
                    sigma.ret.raw_block_at(n - j, n - l),
                    sigma.ret.size1,
                    sigma.ret.size2,
                )
            };
            if l == 0 {
                let prod = gnn_dm.mat_mul(&dm(&s_blk, d));
                axpy_c(&mut qj, C64::new(weight, 0.0), &prod);
            } else {
                axpy_c(&mut mblocks[(j - 1) as usize][(l - 1) as usize],
                    C64::new(-weight, 0.0),
                    &dm(&s_blk, d),
                );
            }
        }
        qblocks[(j - 1) as usize] = qj;
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    solve_block_left(&mut xblocks, &mblocks, &qblocks, &mut wsk);
    for q in 1..=k as i64 {
        g.ret
            .raw_block_at_mut(n, n - q)
            .copy_from_slice(xblocks[(q - 1) as usize].raw());
    }
    // contribution of G^R(n, m), m = n-k, .., n, to the convolution,
    // stored into qq(j) without the factor h
    let nel = d * d;
    let mut qq: Vec<Vec<C64>> = vec![vec![C64::ZERO; nel]; (n + 1) as usize];
    for m in (n - k)..=n {
        let gnm = g.ret.raw_block_at(n, m);
        for j in 0..=(n - k - 1) {
            let w = integ.giw.at((n - j) as usize, (n - m) as usize);
            let s = sigma.ret.raw_block_at(m, j);
            let prod = dm(gnm, d).mat_mul(&dm(s, d));
            for q in 0..nel {
                qq[j as usize][q] += C64::new(w, 0.0) * prod.raw()[q];
            }
        }
    }

    // the rest: t' = n - l, l = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    let diffw: Vec<C64> = (0..=(k + 1) as usize)
        .map(|p| I / C64::new(hstep, 0.0) * C64::new(integ.bdw.at(p), 0.0))
        .collect();
    let w0 = hstep * integ.giw.omega[0];
    let mut mm = DenseMat::zeros(d);
    for l in (k + 1)..=n {
        let j = n - l;
        let mut qqj = dm(&qq[j as usize], d);
        qqj = crate::linalg::mat::scale_c(&qqj, C64::new(hstep, 0.0));
        for p in 1..=(k + 1) as i64 {
            axpy_c(&mut qqj, -diffw[p as usize], &dm(g.ret.raw_block_at(n, j + p), d));
        }
        // M = i b_0 / h + mu - H(j)^dagger - h omega_0 Sigma^R(j,j)
        mm.set_from(&crate::linalg::mat::scale_c(&id, diffw[0] + C64::new(mu, 0.0)));
        let hj = h.block(j);
        let hjd = crate::convolution::adjoint_block(hj, d, d);
        axpy_c(&mut mm, C64::new(-1.0, 0.0), &dm(&hjd, d));
        let sj = sigma.ret.raw_block_at(j, j);
        axpy_c(&mut mm, C64::new(-w0, 0.0), &dm(sj, d));
        let mut gnj = dm(g.ret.raw_block_at(n, j), d);
        solve_block_left_single(&mut gnj, &mm, &qqj, &mut ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gnj.raw());

        // contribution of G^R(n, j) to the remaining columns
        for j2 in 0..j {
            let w = integ.giw.at((n - j2) as usize, (n - j) as usize);
            let s = sigma.ret.raw_block_at(j, j2);
            let prod = gnj.mat_mul(&dm(s, d));
            for q in 0..nel {
                qq[j2 as usize][q] += C64::new(w, 0.0) * prod.raw()[q];
            }
        }
    }
}

/// Time stepping of the left-mixing component `G^rangle(n, tau)` with
/// `n > k`; the retarded component at time step `n` must be known
/// (blueprint `_dyson_timestep_lmix!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_lmix(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let ntau = g.ntau;
    let id = DenseMat::identity(d);

    // clear the current row (the implicit term does not enter the source)
    let rowlen = (ntau + 1) * d * d;
    let off = n as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].fill(C64::ZERO);

    // known part of the convolution [Sigma * G]^rangle(n, tau)
    let cshape = Contour::new(n, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, n, g.sign, C64::ZERO);
    incr_convolution_lmix_step(
        n,
        &mut w,
        sigma,
        sigma,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        C64::ONE,
    );

    // backward differentiation, G^rangle(m, tau), m = n-k-1, .., n-1
    let mcols = d * (ntau + 1);
    let mut ws = BlockSolver::new_rect(1, d, mcols);
    let mut q = DenseMat::zeros_shape(d, mcols);
    for m in 0..=ntau as i64 {
        for i in 0..d {
            for jj in 0..d {
                let mut v = w.lmix.raw_block_at(0, m)[i * d + jj];
                for l in (n - k - 1)..=n - 1 {
                    let cw =
                        I / C64::new(hstep, 0.0) * C64::new(integ.bdw.at((n - l) as usize), 0.0);
                    v -= cw * g.lmix.raw_block_at(l, m)[i * d + jj];
                }
                q.set(i, m as usize * d + jj, v);
            }
        }
    }

    // [i b_0 / h - h Omega_{n0} Sigma^R(n,n) + mu - H(n)] G^rangle = Q
    let mut m1 = crate::linalg::mat::scale_c(
        &id,
        I / C64::new(hstep, 0.0) * C64::new(integ.bdw.at(0), 0.0) + C64::new(mu, 0.0),
    );
    let wn = hstep * integ.giw.at(n as usize, 0);
    let sn = sigma.ret.raw_block_at(n, n);
    axpy_c(&mut m1, C64::new(-wn, 0.0), &dm(sn, d));
    axpy_c(&mut m1, C64::new(-1.0, 0.0), &dm(h.block(n), d));
    let mblocks = vec![vec![m1]];
    let qblocks = vec![q];
    let mut xblocks = vec![DenseMat::zeros_shape(d, mcols)];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for m in 0..=ntau as i64 {
        for i in 0..d {
            for jj in 0..d {
                g.lmix.raw_block_at_mut(n, m)[i * d + jj] = xblocks[0].get(i, m as usize * d + jj);
            }
        }
    }
}

/// Time stepping of the lesser component `G^<(j, n)` for `0 <= j <= n`
/// with `n > k`; the retarded and left-mixing components at time step
/// `n` must be known (blueprint `_dyson_timestep_less!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_less(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let n = tstp;
    assert!(n > integ.k);
    let d = g.size1;
    let ntau = g.ntau;

    // source term [Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)
    let cshape = Contour::new(n, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, n, g.sign, C64::ZERO);
    incr_convolution_lmix_rmix_step(
        n,
        &mut w,
        sigma,
        sigma,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        C64::ONE,
    );
    incr_convolution_less_adv_step(
        n,
        &mut w,
        sigma,
        sigma,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        C64::ONE,
    );
    let mut gles: Vec<Vec<C64>> = (0..=n).map(|j| w.less.raw_block_at(j, n).to_vec()).collect();
    dyson_less_core(n, &mut gles, g, mu, h, sigma, hstep, integ);
}

/// Solve the Dyson equation at the real time step `tstp > solve_order`
/// (blueprint `dyson_timestep!` with the KB index `tstp + 1`). The
/// previous time steps of `g` must be known; the Matsubara component is
/// not recalculated.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    assert!(
        tstp > k,
        "dyson_timestep: time step {tstp} must be larger than {k}"
    );
    super::dyson::check_dyson_rt(g, sigma, h, k, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    dyson_timestep_ret(tstp, g, mu, h, sigma, hstep, &integ);
    dyson_timestep_lmix(tstp, g, mu, h, sigma, beta, hstep, &integ);
    dyson_timestep_less(tstp, g, mu, h, sigma, beta, hstep, &integ);
}
