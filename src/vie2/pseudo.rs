// Project : KadanoffBaym
// File    : src/vie2/pseudo.rs
// Purpose : pseudo-particle Volterra integral equation

//!
//! *Remarks* : *Pseudo-Particle Volterra Integral Equation*
//!
//! The pseudo-particle VIE2 reads ``(1 + F) \ast G = Q``, where the
//! convolution is the pseudo-particle one (see `crate::convolution::pseudo`).
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
//! **Left-mixing component in the start-up time steps.** Let
//! ``X(l) = G^{\rceil}(t,\beta - l)``. Then
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
//! The greater (stored as `ret`) and lesser components in the start-up
//! time steps are solved by the standard VIE2 kernels, see
//! `crate::vie2::start`.
//!
//! For the later time steps (`tstp ≥ k + 2`), all the components are
//! obtained by the incremental convolutions (the unknown components at the
//! current time step are set to zero) and the end-point corrections, i.e.,
//! a 1 × 1 block problem for each matrix element. The left-mixing component
//! uses the pseudo-particle convolution.
//!
//! The time indices of the private kernels are 0-based, see
//! *Remarks* : *Real Time Dyson Equation*.

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{ContourGreen, PseudoGreen};
use crate::weights::Integrator;

use crate::basics::{gtr_blk, timestep_at};
use crate::convolution::{conv_less_adv_kernel, conv_lmix_rmix_kernel};
use crate::convolution::{conv_ret_less_kernel, incr_less_diagcc};
use crate::convolution::{incr_convolution_ret_step, pseudo_incr_convolution_lmix_step_c};
use crate::linalg::{axpy_c, solve_block_left, solve_block_left_single};
use crate::linalg::{solve_block_right, solve_block_right_single};
use crate::vie2::{vie2_start_less, vie2_start_ret};

/// Solve the pseudo-particle VIE2 at the Matsubara axis and at all the
/// real time steps.
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
    let k = integ.k;
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
        let mut pws0 = PseudoVie2Workspace::new(integ.k, g.size1);
        pseudo_vie2_timestep_impl(
            tstp, &mut gc, &fc, &fccc, &qc, integ, beta, hstep, &mut pws0,
        );
    }
    g.copy_from_contour(&gc);
}

/// Solve the Matsubara component of the pseudo-particle VIE2.
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

/// Solve the pseudo-particle VIE2 at the first `k + 1` real time steps.
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
    let k = integ.k;
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

/// Solve the pseudo-particle VIE2 at time step `tstp >= k + 2`.
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
    let k = integ.k;
    assert!(
        tstp > k,
        "tstp must be larger than the solve order, got {tstp}"
    );
    check_pseudo_vie2(g, f, fcc, q, integ, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut pws = PseudoVie2Workspace::new(integ.k, g.size1);
    pseudo_vie2_timestep_ws(tstp, g, f, fcc, q, integ, beta, hstep, &mut pws);
}

/// Workspace-reusing variant of [`pseudo_vie2_timestep`]: pass one
/// [`PseudoVie2Workspace`] built before the time-step loop.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_vie2_timestep_ws(
    tstp: i64,
    g: &mut PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
    pws: &mut PseudoVie2Workspace,
) {
    let mut gc = g.to_contour();
    let (fc, fccc, qc) = (f.to_contour(), fcc.to_contour(), q.to_contour());
    pseudo_vie2_timestep_impl(tstp, &mut gc, &fc, &fccc, &qc, integ, beta, hstep, pws);
    g.copy_from_contour(&gc);
}

/// Check the arguments of the pseudo-particle VIE2 solvers. `nmax < 0`
/// means that only the Matsubara axis is needed.
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

/// Solve the Matsubara component of the pseudo-particle VIE2. Internal
/// version on a transferred `ContourGreen`.
fn pseudo_vie2_mat_impl(
    g: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
) {
    let k = integ.k;
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
        axpy_c(
            &mut mblocks[(n - 1) as usize][(n - 1) as usize],
            Complex::ONE,
            &id,
        );
        // integral int_0^tau dx F^M(tau - x) G^M(x)
        if n < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight = dtau * integ.bcw.at(n as usize, m as usize, j as usize);
                    if j == 0 {
                        let fm = DenseMat::from_slice(d, d, f.mat.block(m as usize));
                        let g0 = DenseMat::from_slice(d, d, g.mat.block(0));
                        axpy_c(
                            &mut qblocks[(n - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &fm.mat_mul(&g0),
                        );
                    } else {
                        let fm = DenseMat::from_slice(d, d, f.mat.block(m as usize));
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
                    let fm = DenseMat::from_slice(d, d, f.mat.block(k as usize));
                    let g0 = DenseMat::from_slice(d, d, g.mat.block(0));
                    axpy_c(
                        &mut qblocks[(n - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &fm.mat_mul(&g0),
                    );
                } else {
                    let fm = DenseMat::from_slice(d, d, f.mat.block((k - m) as usize));
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
        qq.set_from(&DenseMat::from_slice(d, d, q.mat.block(n as usize)));
        for m in 0..n {
            let w = integ.giw.at(n as usize, m as usize);
            let fm = DenseMat::from_slice(d, d, f.mat.block((n - m) as usize));
            let gm = DenseMat::from_slice(d, d, g.mat.block(m as usize));
            axpy_c(&mut qq, Complex::new(-dtau * w, 0.0), &fm.mat_mul(&gm));
        }
        mm.set_from(id);
        let f0 = DenseMat::from_slice(d, d, f.mat.block(0));
        axpy_c(
            &mut mm,
            Complex::new(dtau * integ.giw.at(n as usize, n as usize), 0.0),
            &f0,
        );
        let mut gn = DenseMat::from_slice(d, d, g.mat.block(n as usize));
        solve_block_right_single(&mut gn, &mm, &qq, &mut ws1);
        g.mat.block_mut(n as usize).copy_from_slice(gn.raw());
    }
}

/// Solve the left-mixing component `G^](n, tau)` (n <= k) of the
/// pseudo-particle VIE2 along the reversed imaginary time axis. Internal
/// version on a transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
fn pseudo_vie2_start_lmix_impl(
    n: i64,
    g: &mut ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k;
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
            let fl = DenseMat::from_slice(d, d, fcc.lmix.raw_block_at(j, ntau - i));
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
        axpy_c(
            &mut mblocks[(i - 1) as usize][(i - 1) as usize],
            Complex::ONE,
            &id,
        );
        // integral int_0^l dx X(x) Fcc^M(l - x)
        if i < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight = dtau * integ.bcw.at(i as usize, m as usize, j as usize);
                    if j == 0 {
                        let fm = DenseMat::from_slice(d, d, fcc.mat.block(m as usize));
                        axpy_c(
                            &mut qblocks[(i - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &xx[0].mat_mul(&fm),
                        );
                    } else {
                        let fm = DenseMat::from_slice(d, d, fcc.mat.block(m as usize));
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
                    let fm = DenseMat::from_slice(d, d, fcc.mat.block(k as usize));
                    axpy_c(
                        &mut qblocks[(i - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &xx[0].mat_mul(&fm),
                    );
                } else {
                    let fm = DenseMat::from_slice(d, d, fcc.mat.block((k - m) as usize));
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
            let fm = DenseMat::from_slice(d, d, fcc.mat.block((i - m) as usize));
            axpy_c(
                &mut qq,
                Complex::new(-dtau * w, 0.0),
                &xx[m as usize].mat_mul(&fm),
            );
        }
        mm.set_from(id);
        let f0 = DenseMat::from_slice(d, d, fcc.mat.block(0));
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

/// Solve the pseudo-particle VIE2 at time step `tstp >= k + 2`. Internal
/// version on a transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
/// Reusable scratch of the pseudo-particle VIE2 time stepping: one
/// instance per time-step loop. All blocks are re-zeroed on every
/// call; the convolution scratch carries no cross-step cache (the
/// incremental kernels use only its buffers).
pub struct PseudoVie2Workspace {
    k: i64,
    id: DenseMat,
    ws1: BlockSolver,
    mm: DenseMat,
    cws: ConvWorkspace,
}

impl PseudoVie2Workspace {
    pub fn new(k: i64, d: usize) -> PseudoVie2Workspace {
        PseudoVie2Workspace {
            k,
            id: DenseMat::identity(d),
            ws1: BlockSolver::new(1, d),
            mm: DenseMat::zeros(d),
            cws: ConvWorkspace::new(d, d, d, d),
        }
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.k
    }

    /// Drop the cached convolution buffers; the incremental kernels
    /// carry no cross-step Matsubara cache, so reuse needs no reset in
    /// the current kernels, but the call keeps the workspace contract
    /// uniform.
    pub fn reset(&mut self) {
        self.cws.reset();
    }
}

#[allow(clippy::too_many_arguments)]
fn pseudo_vie2_timestep_impl(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
    pws: &mut PseudoVie2Workspace,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k, "n must be larger than the solve order");
    let d = g.size1;
    let ntau = g.ntau as i64;
    let id = &pws.id;
    let ws1 = &mut pws.ws1;
    let ws = &mut pws.cws;
    let mm = &mut pws.mm;

    // source terms, a copy of Q at the current time step
    let mut w = timestep_at(q, tstp);

    // zero the current time step of G
    for j in 0..=tstp {
        g.ret.raw_block_at_mut(tstp, j).fill(Complex::ZERO);
        g.less.raw_block_at_mut(j, tstp).fill(Complex::ZERO);
    }
    for m in 0..=ntau {
        g.lmix.raw_block_at_mut(tstp, m).fill(Complex::ZERO);
    }

    // h F^{R}(n, n), it enters the end-point terms
    let mtemp = DenseMat::from_slice(d, d, f.ret.raw_block_at(n, n));
    let mtemp_c = Complex::new(hstep, 0.0);

    // greater component: Q - F * G, with G^{>}(n, j) = 0
    incr_convolution_ret_step(
        tstp,
        &mut w,
        f,
        fcc,
        None,
        g,
        g,
        integ,
        hstep,
        -Complex::ONE,
    );
    for j in 0..=n {
        mm.set_from(id);
        axpy_c(
            mm,
            mtemp_c * Complex::new(integ.giw.at((n - j) as usize, 0), 0.0),
            &mtemp,
        );
        let qblk = DenseMat::from_slice(d, d, w.ret.raw_block_at(tstp, j));
        let mut gj = DenseMat::from_slice(d, d, g.ret.raw_block_at(n, j));
        solve_block_right_single(&mut gj, mm, &qblk, ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gj.raw());
    }

    // left-mixing component: Q - F star G, with G^](n, tau) = 0.
    // The pseudo-particle kernel with a timestep target.
    pseudo_incr_convolution_lmix_step_c(
        tstp,
        &mut w,
        f,
        fcc,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        -Complex::ONE,
        None,
    );
    mm.set_from(&id);
    axpy_c(
        mm,
        mtemp_c * Complex::new(integ.giw.at(n as usize, n as usize), 0.0),
        &mtemp,
    );
    for m in 0..=ntau {
        let qblk = DenseMat::from_slice(d, d, w.lmix.raw_block_at(0, m));
        let mut gm = DenseMat::from_slice(d, d, g.lmix.raw_block_at(n, m));
        solve_block_right_single(&mut gm, &mm, &qblk, ws1);
        g.lmix.raw_block_at_mut(n, m).copy_from_slice(gm.raw());
    }

    // lesser component: Q - G * Fcc, with G^{<}(j, n) = 0. The blocks
    // j < n are solved first, then they enter the block j = n.
    let mmt = mm.adjoint();
    let diagcc = incr_less_diagcc(tstp, integ);
    for ms in [(0, n - 1), (n, n)] {
        conv_ret_less_kernel(
            &mut w.less,
            ms,
            tstp,
            &g.ret,
            &g.ret,
            FOp::None,
            &fcc.less,
            &f.less,
            integ,
            -Complex::new(hstep, 0.0),
            ws,
        );
        conv_less_adv_kernel(
            &mut w.less,
            ms,
            tstp,
            &g.less,
            &g.less,
            FOp::None,
            &fcc.ret,
            &f.ret,
            integ,
            -Complex::new(hstep, 0.0),
            ws,
            diagcc,
        );
        conv_lmix_rmix_kernel(
            &mut w.less,
            ms,
            tstp,
            &g.lmix,
            None,
            &f.lmix,
            integ,
            g.sign,
            -Complex::new(0.0, -(beta / ntau as f64)),
            ws,
        );
        for j in ms.0..=ms.1 {
            let qblk = DenseMat::from_slice(d, d, w.less.raw_block_at(j, tstp));
            let mut gj = DenseMat::from_slice(d, d, g.less.raw_block_at(j, tstp));
            solve_block_left_single(&mut gj, &mmt, &qblk, ws1);
            g.less.raw_block_at_mut(j, tstp).copy_from_slice(gj.raw());
        }
    }
}

#[cfg(test)]
mod pseudo_ws_tests {
    use super::*;

    use crate::equilibrium::from_ham;
    use crate::linalg::DenseMat;
    use crate::types::{Contour, Sign, TimeFunction};

    /// One pseudo workspace across the loop equals the convenience
    /// path (fresh workspace per step).
    #[test]
    fn pseudo_vie2_timestep_ws_matches_fresh_workspace() {
        let (nt, ntau, beta, h, k) = (12i64, 16i64, 4.0f64, 0.05, 3i64);
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
        let mk = |from: bool| {
            let mut g = PseudoGreen::new(&c, Sign::Fermi, Complex::ZERO);
            if from {
                let mut gc = g.to_contour();
                from_ham(&mut gc, 0.0, &h0, beta, h);
                g.copy_from_contour(&gc);
            }
            g
        };
        let f = mk(true);
        let fcc = mk(true);
        let q = mk(true);
        let integ = Integrator::new(k);
        let mut g1 = mk(false);
        let mut g2 = mk(false);
        for g in [&mut g1, &mut g2] {
            pseudo_vie2_mat(g, &f, &fcc, &q, beta, &integ);
            pseudo_vie2_start(g, &f, &fcc, &q, &integ, beta, h);
        }
        let mut pws = PseudoVie2Workspace::new(k, 2);
        for tstp in (k + 1)..=nt {
            pseudo_vie2_timestep(tstp, &mut g1, &f, &fcc, &q, &integ, beta, h);
            pseudo_vie2_timestep_ws(tstp, &mut g2, &f, &fcc, &q, &integ, beta, h, &mut pws);
        }
        assert_eq!(g1.ret.raw(), g2.ret.raw());
        assert_eq!(g1.lmix.raw(), g2.lmix.raw());
        assert_eq!(g1.less.raw(), g2.less.raw());
    }
}
