// Project : KadanoffBaym
// File    : src/vie2/timestep.rs
// Purpose : time stepping of the real time VIE2 components (blueprint:
//           solvers/vie2_timestep.jl; the moving-window variants are R6
//           scope)
//
// All time indices follow the NESSi convention (0-based real times).

use crate::dyson::mat::MatsubaraMethod;
use super::full::{check_vie2, check_vie2_order};
use crate::complex::Complex;
use crate::convolution::less::{
    incr_convolution_less_adv_step, incr_convolution_lmix_rmix_step,
};
use crate::convolution::lmix::incr_convolution_lmix_step;
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    BlockSolver,
};
use crate::linalg::mat::{axpy_c, scale_c, DenseMat};
use crate::types::contour::Contour;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::MATSUBARA;
use crate::weights::Integrator;

fn dm(v: &[Complex], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, v.to_vec())
}

/// `F^R(n, l)` (NESSi indices) and the sign `1` if `n >= l`; otherwise
/// `F^{cc,R}(l, n)^dagger` and the sign `-1` (blueprint
/// `_vie2_kernel_ret`).
pub(crate) fn vie2_kernel_ret(f: &ContourGreen, fcc: &ContourGreen, n: i64, l: i64) -> (Vec<Complex>, f64) {
    if n >= l {
        (f.ret.raw_block_at(n, l).to_vec(), 1.0)
    } else {
        (
            crate::convolution::adjoint_block(
                fcc.ret.raw_block_at(l, n),
                fcc.ret.size1,
                fcc.ret.size2,
            ),
            -1.0,
        )
    }
}

/// Solve the lesser component `G^<(j, n)` for `j = 0, .., n` (NESSi
/// indices). On entry `gles[j]` contains the source term
/// `-[F^rangle star G^lceil + F^< * G^A](j, n)`; on exit the solution
/// (also written into `g`) (blueprint `_vie2_less_core!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_less_core(
    n: i64,
    gles: &mut [Vec<Complex>],
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let id = DenseMat::identity(d);
    assert!(gles.len() >= (n.max(k) + 1) as usize);

    // start: j = 0, .., k, a (k+1) x (k+1) problem M X = R
    let dim = (k + 1) as usize;
    let mut ws = BlockSolver::new(dim, d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); dim]; dim];
    let mut rblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];
    for j in 0..=k as i64 {
        let mut rj = dm(&gles[j as usize], d);
        if j <= n {
            axpy_c(&mut rj, Complex::ONE, &dm(q.less.raw_block_at(j, n), d));
        } else {
            let qadj = crate::convolution::adjoint_block(
                q.less.raw_block_at(n, j),
                q.less.size1,
                q.less.size2,
            );
            axpy_c(&mut rj, Complex::new(-1.0, 0.0), &dm(&qadj, d));
        }
        rblocks[j as usize] = rj;
        for m in 0..=k as i64 {
            let (sblk, s) = vie2_kernel_ret(f, fcc, j, m);
            let w = s * hstep * integ.giw.at(j as usize, m as usize);
            let mut mjm = DenseMat::zeros(d);
            if m == j {
                mjm.axpy(1.0, &id);
            }
            axpy_c(&mut mjm, Complex::new(w, 0.0), &dm(&sblk, d));
            mblocks[j as usize][m as usize] = mjm;
        }
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];
    solve_block_right(&mut xblocks, &mblocks, &rblocks, &mut ws);
    for j in 0..=k as i64 {
        gles[j as usize] = xblocks[j as usize].raw().to_vec();
    }

    // time stepping: j = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    for j in (k + 1)..=n {
        let mut qq = dm(&gles[j as usize], d);
        axpy_c(&mut qq, Complex::ONE, &dm(q.less.raw_block_at(j, n), d));
        let mut mm = DenseMat::zeros(d);
        mm.axpy(1.0, &id);
        let wj = hstep * integ.giw.at(j as usize, j as usize);
        axpy_c(&mut mm, Complex::new(wj, 0.0), &dm(f.ret.raw_block_at(j, j), d));
        for m in 0..j {
            let w = hstep * integ.giw.at(j as usize, m as usize);
            let prod = dm(f.ret.raw_block_at(j, m), d).mat_mul(&dm(&gles[m as usize], d));
            axpy_c(&mut qq, Complex::new(-w, 0.0), &prod);
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

/// Time stepping of the retarded component: the right action equation
/// `G^R + G^R * K^R = Q^R` with the kernel `K = Fcc` and its partner
/// `Kcc = F` (blueprint `_vie2_timestep_ret!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_ret(
    tstp: i64,
    g: &mut ContourGreen,
    kk: &ContourGreen,
    kcc: &ContourGreen,
    q: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let id = DenseMat::identity(d);

    // clear the time step, G^R(n,n) = Q^R(n,n)
    for j in 0..=n {
        g.ret.raw_block_at_mut(n, j).fill(Complex::ZERO);
    }
    g.ret
        .raw_block_at_mut(n, n)
        .copy_from_slice(q.ret.raw_block_at(n, n));
    let gnn = dm(g.ret.raw_block_at(n, n), d);

    // start values G^R(n, n-q), q = 1, .., k: solve sum_q X_q M_{qj} = R_j
    let mut wsk = BlockSolver::new(k as usize, d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut rblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    for j in 1..=k as i64 {
        let mut rj = dm(q.ret.raw_block_at(n, n - j), d);
        // M_jj starts from the identity; the l = j integral term is
        // accumulated into the SAME block by the loop below
        mblocks[(j - 1) as usize][(j - 1) as usize].axpy(1.0, &id);
        for l in 0..=k as i64 {
            let mut weight = -hstep * integ.giw.at(j as usize, l as usize);
            let s_blk = if l <= j {
                kk.ret.raw_block_at(n - l, n - j).to_vec()
            } else {
                weight = -weight;
                crate::convolution::adjoint_block(
                    kcc.ret.raw_block_at(n - j, n - l),
                    kcc.ret.size1,
                    kcc.ret.size2,
                )
            };
            if l == 0 {
                let prod = gnn.mat_mul(&dm(&s_blk, d));
                axpy_c(&mut rj, Complex::new(weight, 0.0), &prod);
            } else {
                axpy_c(
                    &mut mblocks[(j - 1) as usize][(l - 1) as usize],
                    Complex::new(-weight, 0.0),
                    &dm(&s_blk, d),
                );
            }
        }
        rblocks[(j - 1) as usize] = rj;
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); k as usize];
    solve_block_left(&mut xblocks, &mblocks, &rblocks, &mut wsk);
    for qq in 1..=k as i64 {
        g.ret
            .raw_block_at_mut(n, n - qq)
            .copy_from_slice(xblocks[(qq - 1) as usize].raw());
    }

    // contribution of G^R(n, m), m = n-k, .., n, to the convolution
    // -int dm G^R(n,m) K^R(m,j), stored into qq(j) without the factor h
    let nel = d * d;
    let mut qqbuf: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; nel]; (n + 1) as usize];
    for m in (n - k)..=n {
        let gnm = g.ret.raw_block_at(n, m);
        for j in 0..=(n - k - 1) {
            let w = -integ.giw.at((n - j) as usize, (n - m) as usize);
            let s = kk.ret.raw_block_at(m, j);
            let prod = dm(gnm, d).mat_mul(&dm(s, d));
            for c in 0..nel {
                qqbuf[j as usize][c] += Complex::new(w, 0.0) * prod.raw()[c];
            }
        }
    }

    // the rest: t' = n-l, l = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    let w0 = hstep * integ.giw.omega[0];
    let mut mm = DenseMat::zeros(d);
    for l in (k + 1)..=n {
        let j = n - l;
        let mut qqj = dm(&qqbuf[j as usize], d);
        qqj = scale_c(&qqj, Complex::new(hstep, 0.0));
        axpy_c(&mut qqj, Complex::ONE, &dm(q.ret.raw_block_at(n, j), d));
        // G^R(n,j) [1 + h omega_0 K^R(j,j)] = qq(j)
        mm.set_from(&id);
        axpy_c(&mut mm, Complex::new(w0, 0.0), &dm(kk.ret.raw_block_at(j, j), d));
        let mut gnj = dm(g.ret.raw_block_at(n, j), d);
        solve_block_left_single(&mut gnj, &mm, &qqj, &mut ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gnj.raw());

        // contribution of G^R(n, j) to the remaining columns
        for j2 in 0..j {
            let w = -integ.giw.at((n - j2) as usize, (n - j) as usize);
            let s = kk.ret.raw_block_at(j, j2);
            let prod = gnj.mat_mul(&dm(s, d));
            for c in 0..nel {
                qqbuf[j2 as usize][c] += Complex::new(w, 0.0) * prod.raw()[c];
            }
        }
    }
}

/// Time stepping of the left-mixing component (blueprint
/// `_vie2_timestep_lmix!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_lmix(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
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

    // clear the current row first, so that its old values do not enter
    // the source term
    let rowlen = (ntau + 1) * d * d;
    let off = n as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].fill(Complex::ZERO);

    // convolution [F * G]^rangle(n, tau) without the implicit term
    let cshape = Contour::new(n, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, n, g.sign, Complex::ZERO);
    incr_convolution_lmix_step(n, &mut w, f, fcc, None, g, g, integ, beta, hstep, Complex::ONE);

    // [1 + h Omega_{n0} F^R(n,n)] G^rangle(n,tau) = Q^rangle - [F*G]^rangle
    let mcols = d * (ntau + 1);
    let mut ws = BlockSolver::new_rect(1, d, mcols);
    let mut qb = DenseMat::zeros_shape(d, mcols);
    for m in 0..=(ntau as i64) {
        for a in 0..d {
            for b in 0..d {
                let v = q.lmix.raw_block_at(n, m)[a * d + b]
                    - w.lmix.raw_block_at(0, m)[a * d + b];
                qb.set(a, m as usize * d + b, v);
            }
        }
    }
    let mut m1 = DenseMat::zeros(d);
    m1.axpy(1.0, &id);
    let wn = hstep * integ.giw.at(n as usize, 0);
    axpy_c(&mut m1, Complex::new(wn, 0.0), &dm(f.ret.raw_block_at(n, n), d));
    let mblocks = vec![vec![m1]];
    let qblocks = vec![qb];
    let mut xblocks = vec![DenseMat::zeros_shape(d, mcols)];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for m in 0..=(ntau as i64) {
        for a in 0..d {
            for b in 0..d {
                g.lmix.raw_block_at_mut(n, m)[a * d + b] =
                    xblocks[0].get(a, m as usize * d + b);
            }
        }
    }
}

/// Time stepping of the lesser component (blueprint
/// `_vie2_timestep_less!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_less(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let n = tstp;
    assert!(n > integ.k);
    let d = g.size1;
    let ntau = g.ntau;

    let cshape = Contour::new(n, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, n, g.sign, Complex::ZERO);
    incr_convolution_lmix_rmix_step(n, &mut w, f, fcc, None, g, g, integ, beta, hstep, -Complex::ONE);
    incr_convolution_less_adv_step(n, &mut w, f, fcc, None, g, g, integ, beta, hstep, -Complex::ONE);
    let mut gles: Vec<Vec<Complex>> = (0..=n).map(|j| w.less.raw_block_at(j, n).to_vec()).collect();
    vie2_less_core(n, &mut gles, g, f, fcc, q, hstep, integ);
}

/// The three time stepping components with the kernel swapped for the
/// right action retarded equation (blueprint `vie2_timestep!`, the
/// `tstp >= k + 2` branch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_timestep_core(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    // the retarded component is solved from the right action equation,
    // so F and Fcc are exchanged
    vie2_timestep_ret(tstp, g, fcc, f, q, h, integ);
    vie2_timestep_lmix(tstp, g, f, fcc, q, beta, h, integ);
    vie2_timestep_less(tstp, g, f, fcc, q, beta, h, integ);
}

/// Solve the VIE2 at the time step `tstp` (blueprint `vie2_timestep!`
/// with the KB convention `tstp + 1`):
///
/// * `MATSUBARA` -> the Matsubara component (`vie2_mat`)
/// * `0` -> the zeroth order start-up at `t = 0`
/// * `1 <= tstp <= k` -> the start-up of order `tstp`, i.e. the time
///   steps `0, .., tstp` are (re)calculated
/// * `tstp >= k + 1` -> the time stepping; the previous time steps must
///   be known
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    method: MatsubaraMethod,
) {
    check_vie2(g, f, fcc, q);
    check_vie2_order(integ);
    assert!(
        MATSUBARA <= tstp && tstp <= g.nt as i64,
        "time step {tstp} out of range"
    );
    assert!(beta > 0.0, "beta must be positive");
    assert!(h > 0.0, "h must be positive");
    let k = integ.k;

    if tstp == MATSUBARA {
        super::mat::vie2_mat(g, f, fcc, q, beta, integ, method, 6, 40, 1e-12);
    } else if tstp == 0 {
        super::start::vie2_bootstrap(g, f, q, beta);
    } else if tstp <= k {
        let start_integ = Integrator::new(tstp);
        super::start::vie2_start_impl(g, f, fcc, q, &start_integ, beta, h);
    } else {
        vie2_timestep_core(tstp, g, f, fcc, q, integ, beta, h);
    }
}
