// Project : KadanoffBaym
// File    : src/vie2/pseudo.rs
// Purpose : pseudo-particle Volterra integral equation (blueprint
//           julia/src/solvers/vie2_pseudo.jl)
//
// The pseudo-particle VIE2 reads `(1 + F) * G = Q`, where the
// convolution is the pseudo-particle one. The Matsubara component is
// a causal Volterra equation with `G^M(0) = Q^M(0)`. The left-mixing
// component in the start-up time steps is solved along the reversed
// imaginary time axis, `X(l) = G^](t, beta - l)`, with
// `X(0) = xi G^{>}(t, 0)`. The greater (stored as `ret`) and lesser
// components in the start-up time steps are solved by the standard
// VIE2 kernels (`vie2_start_ret`, `vie2_start_less`). For the later
// time steps (`tstp >= k + 2`) all the components are obtained by the
// incremental convolutions (the unknown components at the current
// time step are set to zero) and the end-point corrections.
//!
//! *Remarks* : *Pseudo-Particle Volterra Integral Equation*
//!
//! The pseudo-particle VIE2 reads ``(1 + F) \ast G = Q``, where the
//! convolution is the pseudo-particle one (see `pseudo.jl`).
//!
//! * Matsubara component. It is a causal Volterra equation,
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau) + \int^{\tau}_{0} d\bar{\tau} F^{M}(\tau - \bar{\tau})
//! G^{M}(\bar{\tau}) = Q^{M}(\tau), \quad G^{M}(0) = Q^{M}(0).
//! \end{equation}
//! ```
//!
//! * Left-mixing component in the start-up time steps. Let
//! ``X(l) = G^{\rceil}(t,\beta - l)``, then
//!
//! ```math
//! \begin{equation}
//! X(l) = Q^{\rceil}(t,\beta - l)
//!      - \int^{l}_{0} d\bar{l} X(\bar{l}) F^{cc,M}(l - \bar{l})
//!      - \int^{t}_{0} d\bar{t} G^{>}(t,\bar{t}) F^{cc,\rceil}(\bar{t},\beta - l),
//! \end{equation}
//! ```
//!
//! with ``X(0) = \xi G^{>}(t,0)``.
//!
//! * The greater (stored as `ret`) and lesser components in the start-up
//! time steps are solved by the standard VIE2 kernels, see `vie2_start.jl`.
//!
//! * For the later time steps (`tstp ≥ k + 2`), all the components are
//! obtained by the incremental convolutions (the unknown components at the
//! current time step are set to zero) and the end-point corrections, i.e.,
//! a 1 × 1 block problem for each matrix element. The left-mixing component
//! uses the pseudo-particle convolution.
//!
//! The time indices of the private kernels follow the NESSi convention
//! (start from 0), see *Remarks* : *Real Time Dyson Equation* in `dyson_timestep.jl`.
//!
//! *References* :
//!
//! Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`.

use crate::complex::Complex;
use crate::convolution::less::{
    conv_less_adv_kernel, conv_lmix_rmix_kernel, conv_ret_less_kernel, incr_les_diagcc,
};
use crate::convolution::pseudo::pseudo_incr_convolution_lmix_step_c;
use crate::convolution::ret::incr_convolution_ret_step;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    BlockSolver,
};
use crate::linalg::mat::{axpy_c, DenseMat};
use crate::types::ContourGreen;
use crate::basics::indexing::timestep_at;
use crate::types::pseudo::PseudoGreen;
use crate::weights::Integrator;

use crate::dyson::pseudo::gtr_blk;
use crate::vie2::start::{vie2_start_less, vie2_start_ret};

/// Check the arguments of the pseudo-particle VIE2 solvers (blueprint
/// `_check_pseudo_vie2`). `nmax < 0` means that only the Matsubara
/// axis is needed.
fn check_pseudo_vie2(
    g: &PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    nmax: i64,
) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order must be in 1..=5, got {}",
        integ.k
    );
    for x in [f, fcc, q] {
        assert!(
            x.sign == g.sign,
            "all the Green's functions must have the same statistics"
        );
        assert!(
            x.size1 == g.size1 && x.size2 == g.size2,
            "all the Green's functions must have the same size"
        );
        assert!(
            x.ntau == g.ntau,
            "all the Green's functions must have the same imaginary time grid"
        );
        assert!(
            x.nt == g.nt,
            "all the Green's functions must have the same real time grid"
        );
    }
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        g.ntau as i64 > integ.k,
        "ntau must be larger than solve_order + 1"
    );
    if nmax >= 0 {
        assert!(
            g.nt as i64 >= nmax,
            "G must have at least {} real time steps",
            nmax + 1
        );
    }
}

/// Solve the Matsubara component of the pseudo-particle VIE2
/// (blueprint `_pseudo_vie2_mat!`). Internal version on a
/// transferred `ContourGreen`.
pub(crate) fn pseudo_vie2_mat_impl(
    g: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
) {
    let k = integ.k as i64;
    let ntau = g.ntau as i64;
    let d = g.size1;
    let dtau = beta / ntau as f64;
    let id = DenseMat::identity(d);

    // initial value: G^M(0) = Q^M(0)
    g.mat.block_mut(0).copy_from_slice(q.mat.block(0));

    // the first k steps, solve sum_q M[p][q] X[q] = R[p]
    let mut ws = BlockSolver::new(k as usize, d);
    let mut mblocks = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks = vec![DenseMat::zeros(d); k as usize];
    for n in 1..=k {
        qblocks[(n - 1) as usize].set_from(&DenseMat::from_vec(
            d,
            d,
            q.mat.block(n as usize).to_vec(),
        ));
        axpy_c(&mut mblocks[(n - 1) as usize][(n - 1) as usize], Complex::ONE, &id);
        // integral int_0^tau dx F^M(tau - x) G^M(x)
        if n < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight =
                        dtau * integ.bcw.at(n as usize, m as usize, j as usize);
                    if j == 0 {
                        let fm = DenseMat::from_vec(d, d, f.mat.block(m as usize).to_vec());
                        let g0 = DenseMat::from_vec(d, d, g.mat.block(0).to_vec());
                        axpy_c(
                            &mut qblocks[(n - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &fm.mat_mul(&g0),
                        );
                    } else {
                        let fm = DenseMat::from_vec(d, d, f.mat.block(m as usize).to_vec());
                        axpy_c(
                            &mut mblocks[(n - 1) as usize][(j - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &fm,
                        );
                    }
                }
            }
        } else {
            for m in 0..=k {
                let cweight = dtau * integ.giw.at(k as usize, m as usize);
                if m == 0 {
                    let fm = DenseMat::from_vec(d, d, f.mat.block(k as usize).to_vec());
                    let g0 = DenseMat::from_vec(d, d, g.mat.block(0).to_vec());
                    axpy_c(
                        &mut qblocks[(n - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &fm.mat_mul(&g0),
                    );
                } else {
                    let fm = DenseMat::from_vec(
                        d,
                        d,
                        f.mat.block((k - m) as usize).to_vec(),
                    );
                    axpy_c(
                        &mut mblocks[(n - 1) as usize][(m - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &fm,
                    );
                }
            }
        }
    }
    let mut xblocks = vec![DenseMat::zeros(d); k as usize];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for n in 1..=k {
        g.mat
            .block_mut(n as usize)
            .copy_from_slice(xblocks[(n - 1) as usize].raw());
    }

    // the remaining steps: Gregory integration
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    let mut mm = DenseMat::zeros(d);
    for n in (k + 1)..=ntau {
        qq.set_from(&DenseMat::from_vec(d, d, q.mat.block(n as usize).to_vec()));
        for m in 0..n {
            let w = integ.giw.at(n as usize, m as usize);
            let fm = DenseMat::from_vec(d, d, f.mat.block((n - m) as usize).to_vec());
            let gm = DenseMat::from_vec(d, d, g.mat.block(m as usize).to_vec());
            axpy_c(&mut qq, Complex::new(-dtau * w, 0.0), &fm.mat_mul(&gm));
        }
        mm.set_from(&id);
        let f0 = DenseMat::from_vec(d, d, f.mat.block(0).to_vec());
        axpy_c(
            &mut mm,
            Complex::new(dtau * integ.giw.at(n as usize, n as usize), 0.0),
            &f0,
        );
        let mut gn = DenseMat::from_vec(d, d, g.mat.block(n as usize).to_vec());
        solve_block_right_single(&mut gn, &mm, &qq, &mut ws1);
        g.mat.block_mut(n as usize).copy_from_slice(gn.raw());
    }
}

/// Solve the left-mixing component `G^](n, tau)` (n <= k) of the
/// pseudo-particle VIE2 along the reversed imaginary time axis
/// (blueprint `_pseudo_vie2_start_lmix!`). Internal version on a
/// transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_vie2_start_lmix_impl(
    n: i64,
    g: &mut ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    assert!((0..=k).contains(&n), "n must be in 0..={k}");
    let ntau = g.ntau as i64;
    let d = g.size1;
    let xi = g.sign.as_i64() as f64;
    let dtau = beta / ntau as f64;
    let n1 = n.max(k);
    let id = DenseMat::identity(d);

    // X(i) = G^](n, beta - tau_i), i = 0, .., ntau
    let mut xx: Vec<DenseMat> = (0..=ntau).map(|_| DenseMat::zeros(d)).collect();

    // Q^](n, beta - tau_i) - int_0^t dj G^{>}(n, j) Fcc^](j, beta - tau_i)
    for i in 1..=ntau {
        xx[i as usize].set_from(&DenseMat::from_vec(
            d,
            d,
            q.lmix.raw_block_at(n, ntau - i).to_vec(),
        ));
    }
    for j in 0..=n1 {
        let mut gtemp = gtr_blk(g, n, j);
        let w = -integ.giw.at(n as usize, j as usize) * hstep;
        for q0 in 0..(d * d) {
            let v = gtemp.raw()[q0];
            gtemp.raw_mut()[q0] = v * Complex::new(w, 0.0);
        }
        for i in 1..=ntau {
            let fl = DenseMat::from_vec(d, d, fcc.lmix.raw_block_at(j, ntau - i).to_vec());
            axpy_c(&mut xx[i as usize], Complex::ONE, &gtemp.mat_mul(&fl));
        }
    }

    // initial value: X(0) = xi G^{>}(n, 0)
    xx[0] = gtr_blk(g, n, 0);
    for q0 in 0..(d * d) {
        let v = xx[0].raw()[q0];
        xx[0].raw_mut()[q0] = v * Complex::new(xi, 0.0);
    }

    // the first k steps, solve sum_q X[q] M[p][q] = R[p]
    let mut ws = BlockSolver::new(k as usize, d);
    let mut mblocks = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks: Vec<DenseMat> = xx[1..=(k as usize)].to_vec();
    for i in 1..=k {
        axpy_c(&mut mblocks[(i - 1) as usize][(i - 1) as usize], Complex::ONE, &id);
        // integral int_0^l dx X(x) Fcc^M(l - x)
        if i < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight =
                        dtau * integ.bcw.at(i as usize, m as usize, j as usize);
                    if j == 0 {
                        let fm =
                            DenseMat::from_vec(d, d, fcc.mat.block(m as usize).to_vec());
                        axpy_c(
                            &mut qblocks[(i - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &xx[0].mat_mul(&fm),
                        );
                    } else {
                        let fm =
                            DenseMat::from_vec(d, d, fcc.mat.block(m as usize).to_vec());
                        axpy_c(
                            &mut mblocks[(i - 1) as usize][(j - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &fm,
                        );
                    }
                }
            }
        } else {
            for m in 0..=k {
                let cweight = dtau * integ.giw.at(i as usize, m as usize);
                if m == 0 {
                    let fm = DenseMat::from_vec(d, d, fcc.mat.block(k as usize).to_vec());
                    axpy_c(
                        &mut qblocks[(i - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &xx[0].mat_mul(&fm),
                    );
                } else {
                    let fm = DenseMat::from_vec(
                        d,
                        d,
                        fcc.mat.block((k - m) as usize).to_vec(),
                    );
                    axpy_c(
                        &mut mblocks[(i - 1) as usize][(m - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &fm,
                    );
                }
            }
        }
    }
    solve_block_left(&mut xx[1..=(k as usize)], &mblocks, &qblocks, &mut ws);

    // the remaining steps: Gregory integration
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    let mut mm = DenseMat::zeros(d);
    for i in (k + 1)..=ntau {
        qq.set_from(&xx[i as usize]);
        for m in 0..i {
            let w = integ.giw.at(i as usize, m as usize);
            let fm = DenseMat::from_vec(d, d, fcc.mat.block((i - m) as usize).to_vec());
            axpy_c(
                &mut qq,
                Complex::new(-dtau * w, 0.0),
                &xx[m as usize].mat_mul(&fm),
            );
        }
        mm.set_from(&id);
        let f0 = DenseMat::from_vec(d, d, fcc.mat.block(0).to_vec());
        axpy_c(
            &mut mm,
            Complex::new(dtau * integ.giw.at(i as usize, i as usize), 0.0),
            &f0,
        );
        solve_block_left_single(&mut xx[i as usize], &mm, &qq, &mut ws1);
    }

    // G^](n, tau_{ntau - l}) = X(l)
    for l in 0..=ntau {
        g.lmix
            .raw_block_at_mut(n, ntau - l)
            .copy_from_slice(xx[l as usize].raw());
    }
}

/// Solve the pseudo-particle VIE2 at time step `tstp >= k + 2`
/// (blueprint `_pseudo_vie2_timestep!`). Internal version on a
/// transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_vie2_timestep_impl(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    let n = tstp;
    assert!(n > k, "n must be larger than the solve order");
    let d = g.size1;
    let ntau = g.ntau as i64;
    let id = DenseMat::identity(d);
    let mut ws1 = BlockSolver::new(1, d);
    let mut ws = ConvWorkspace::new(g.size1, g.size2, f.size1, f.size2);
    let mut mm = DenseMat::zeros(d);

    // source terms, a copy of Q at the current time step
    let mut w = timestep_at(q, tstp);

    // zeros!(G, tstp)
    for j in 0..=tstp {
        g.ret.raw_block_at_mut(tstp, j).fill(Complex::ZERO);
        g.less.raw_block_at_mut(j, tstp).fill(Complex::ZERO);
    }
    for m in 0..=ntau {
        g.lmix.raw_block_at_mut(tstp, m).fill(Complex::ZERO);
    }

    // h F^{R}(n, n), it enters the end-point terms
    let mtemp = DenseMat::from_vec(d, d, f.ret.raw_block_at(n, n).to_vec());
    let mtemp_c = Complex::new(hstep, 0.0);

    // greater component: Q - F * G, with G^{>}(n, j) = 0
    incr_convolution_ret_step(
        tstp, &mut w, f, fcc, None, g, g, integ, hstep, -Complex::ONE,
    );
    for j in 0..=n {
        mm.set_from(&id);
        axpy_c(
            &mut mm,
            mtemp_c * Complex::new(integ.giw.at((n - j) as usize, 0), 0.0),
            &mtemp,
        );
        let qblk = DenseMat::from_vec(d, d, w.ret.raw_block_at(tstp, j).to_vec());
        let mut gj = DenseMat::from_vec(d, d, g.ret.raw_block_at(n, j).to_vec());
        solve_block_right_single(&mut gj, &mm, &qblk, &mut ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gj.raw());
    }

    // left-mixing component: Q - F star G, with G^](n, tau) = 0.
    // The pseudo-particle kernel with a timestep target.
    pseudo_incr_convolution_lmix_step_c(
        tstp, &mut w, f, fcc, None, g, g, integ, beta, hstep, -Complex::ONE, None,
    );
    mm.set_from(&id);
    axpy_c(
        &mut mm,
        mtemp_c * Complex::new(integ.giw.at(n as usize, n as usize), 0.0),
        &mtemp,
    );
    for m in 0..=ntau {
        let qblk = DenseMat::from_vec(d, d, w.lmix.raw_block_at(0, m).to_vec());
        let mut gm = DenseMat::from_vec(d, d, g.lmix.raw_block_at(n, m).to_vec());
        solve_block_right_single(&mut gm, &mm, &qblk, &mut ws1);
        g.lmix.raw_block_at_mut(n, m).copy_from_slice(gm.raw());
    }

    // lesser component: Q - G * Fcc, with G^{<}(j, n) = 0. The blocks
    // j < n are solved first, then they enter the block j = n.
    let mmt = mm.adjoint();
    let diagcc = incr_les_diagcc(tstp, integ);
    for ms in [(0, n - 1), (n, n)] {
        conv_ret_less_kernel(
            &mut w.less, ms, tstp, &g.ret, &g.ret, FOp::None,
            &fcc.less, &f.less, integ, -Complex::new(hstep, 0.0), &mut ws,
        );
        conv_less_adv_kernel(
            &mut w.less, ms, tstp, &g.less, &g.less, FOp::None,
            &fcc.ret, &f.ret, integ, -Complex::new(hstep, 0.0), &mut ws, diagcc,
        );
        conv_lmix_rmix_kernel(
            &mut w.less, ms, tstp, &g.lmix, None, &f.lmix, integ, g.sign,
            -Complex::new(0.0, -(beta / ntau as f64)), &mut ws,
        );
        for j in ms.0..=ms.1 {
            let qblk = DenseMat::from_vec(d, d, w.less.raw_block_at(j, tstp).to_vec());
            let mut gj = DenseMat::from_vec(d, d, g.less.raw_block_at(j, tstp).to_vec());
            solve_block_left_single(&mut gj, &mmt, &qblk, &mut ws1);
            g.less.raw_block_at_mut(j, tstp).copy_from_slice(gj.raw());
        }
    }
}

/// Solve the Matsubara component of the pseudo-particle VIE2
/// (blueprint `pseudo_vie2_mat!`).
pub fn pseudo_vie2_mat(
    g: &mut PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    beta: f64,
    integ: &Integrator,
) {
    check_pseudo_vie2(g, f, fcc, q, integ, -1);
    assert!(beta > 0.0, "beta must be positive");
    let _ = fcc;
    let mut gc = g.to_contour();
    let (fc, qc) = (f.to_contour(), q.to_contour());
    pseudo_vie2_mat_impl(&mut gc, &fc, &qc, integ, beta);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle VIE2 at the first `k + 1` real time steps
/// (blueprint `pseudo_vie2_start!`).
#[allow(clippy::too_many_arguments)]
pub fn pseudo_vie2_start(
    g: &mut PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    check_pseudo_vie2(g, f, fcc, q, integ, k);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut gc = g.to_contour();
    let (fc, fccc, qc) = (f.to_contour(), fcc.to_contour(), q.to_contour());
    vie2_start_ret(&mut gc, &fc, &fccc, &qc, hstep, integ);
    for n in 0..=k {
        pseudo_vie2_start_lmix_impl(n, &mut gc, &fccc, &qc, integ, beta, hstep);
    }
    vie2_start_less(&mut gc, &fc, &fccc, &qc, beta, hstep, integ);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle VIE2 at time step `tstp >= k + 2`
/// (blueprint `pseudo_vie2_timestep!`).
#[allow(clippy::too_many_arguments)]
pub fn pseudo_vie2_timestep(
    tstp: i64,
    g: &mut PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    assert!(
        tstp >= k + 1,
        "tstp must be larger than the solve order, got {tstp}"
    );
    check_pseudo_vie2(g, f, fcc, q, integ, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut gc = g.to_contour();
    let (fc, fccc, qc) = (f.to_contour(), fcc.to_contour(), q.to_contour());
    pseudo_vie2_timestep_impl(tstp, &mut gc, &fc, &fccc, &qc, integ, beta, hstep);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle VIE2 at the Matsubara axis and at all the
/// real time steps (blueprint `pseudo_vie2!`).
#[allow(clippy::too_many_arguments)]
pub fn pseudo_vie2(
    g: &mut PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    check_pseudo_vie2(g, f, fcc, q, integ, g.nt as i64 - 1);

    let mut gc = g.to_contour();
    let (fc, fccc, qc) = (f.to_contour(), fcc.to_contour(), q.to_contour());
    pseudo_vie2_mat_impl(&mut gc, &fc, &qc, integ, beta);
    vie2_start_ret(&mut gc, &fc, &fccc, &qc, hstep, integ);
    for n in 0..=k {
        pseudo_vie2_start_lmix_impl(n, &mut gc, &fccc, &qc, integ, beta, hstep);
    }
    vie2_start_less(&mut gc, &fc, &fccc, &qc, beta, hstep, integ);
    for tstp in (k + 1)..=gc.nt as i64 {
        pseudo_vie2_timestep_impl(tstp, &mut gc, &fc, &fccc, &qc, integ, beta, hstep);
    }
    g.copy_from_contour(&gc);
}
