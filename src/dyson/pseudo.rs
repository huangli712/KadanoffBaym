// Project : KadanoffBaym
// File    : src/dyson/pseudo.rs
// Purpose : pseudo-particle Dyson equation

//!
//! *Remarks* : *Pseudo-Particle Dyson Equation*
//!
//! The pseudo-particle Dyson equation reads
//!
//! ```math
//! \begin{equation}
//! [i\partial_t + \lambda_0 - H(t) - \Sigma] \ast G = 1,
//! \end{equation}
//! ```
//!
//! where ``\lambda_0`` is not included in ``H``. The retarded (greater) and
//! lesser components are solved by the standard real-time Dyson kernels,
//! see `crate::dyson::timestep`. Only the Matsubara and the left-mixing
//! components are special.
//!
//! * Matsubara component. It is a causal initial value problem,
//!
//! ```math
//! \begin{equation}
//! [-\partial_\tau + \lambda_0 - H_0] G^{M}(\tau)
//! - \int^{\tau}_{0} d\bar{\tau} \Sigma^{M}(\tau - \bar{\tau}) G^{M}(\bar{\tau})
//! = 0, \quad G^{M}(0) = -1.
//! \end{equation}
//! ```
//!
//! The first ``k`` points are solved together (polynomial differentiation
//! and boundary corrections), the others by the BDF(k+1) and Gregory rules.
//!
//! * Left-mixing component. Let ``X(l) = G^{\rceil}(t,\beta - l)``, then
//!
//! ```math
//! \begin{equation}
//! [-\partial_l + \lambda_0 - H_0^{\dagger}] X(l)
//! - \int^{l}_{0} d\bar{l} X(\bar{l}) \Sigma^{M}(l - \bar{l})
//! = \int^{t}_{0} d\bar{t} G^{>}(t,\bar{t}) \Sigma^{\rceil}(\bar{t},\beta - l),
//! \end{equation}
//! ```
//!
//! with ``X(0) = \xi G^{>}(t,0)``. The equation is solved from ``l = 0`` to
//! ``l = \beta`` for fixed ``t``, using the same rules as the Matsubara
//! component. Note that ``H_0`` (the Hamiltonian at the Matsubara axis) is
//! used.
//!
//! The time indices of the private kernels are 0-based, see
//! *Remarks* : *Real Time Dyson Equation*.
//!

use crate::complex::Complex;
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{ContourGreen, PseudoGreen, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::linalg::{axpy_c, scale_c};
use crate::linalg::{solve_block_left, solve_block_left_single};
use crate::linalg::{solve_block_right, solve_block_right_single};
use crate::storage::pseudo_gtr_block;

use crate::dyson::workspace::PseudoDysonWorkspace;

use crate::dyson::timestep::{dyson_timestep_less_ws, dyson_timestep_ret_ws};
use crate::dyson::{dyson_start_less, dyson_start_ret};
use crate::equilibrium::check_hamiltonian;

/// Solve the pseudo-particle Dyson equation at the Matsubara axis and
/// at all the real time steps.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_dyson(
    g: &mut PseudoGreen,
    lambda0: f64,
    h: &TimeFunction,
    sigma: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k;
    check_pseudo_dyson(g, sigma, h, integ, g.nt as i64 - 1);

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_slice(g.size1, g.size2, h.block(MATSUBARA));
    pseudo_dyson_mat_impl(&mut gc, lambda0, &h0, &sc, integ, beta);
    dyson_start_ret(&mut gc, lambda0, h, &sc, hstep, integ);
    let mut pws0 = PseudoDysonWorkspace::new(integ.k, g.size1, g.ntau);
    for n in 0..=k {
        pseudo_dyson_lmix_impl(n, &mut gc, lambda0, &h0, &sc, integ, beta, hstep, &mut pws0);
    }
    dyson_start_less(&mut gc, lambda0, h, &sc, beta, hstep, integ);
    let mut pws1 = PseudoDysonWorkspace::new(integ.k, g.size1, g.ntau);
    for tstp in (k + 1)..=gc.nt as i64 {
        pseudo_dyson_timestep_impl(
            tstp, &mut gc, lambda0, h, &h0, &sc, integ, beta, hstep, &mut pws1,
        );
    }
    g.copy_from_contour(&gc);
}

/// Solve the Matsubara component of the pseudo-particle Dyson equation.
pub fn pseudo_dyson_mat(
    g: &mut PseudoGreen,
    lambda0: f64,
    h0: &DenseMat,
    sigma: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
) {
    check_pseudo_dyson_common(g, sigma, integ);
    assert_eq!(h0.nrows(), g.size1, "H0 has a wrong matrix size");
    assert!(beta > 0.0, "beta must be positive");
    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    pseudo_dyson_mat_impl(&mut gc, lambda0, h0, &sc, integ, beta);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle Dyson equation at the first `k + 1` real
/// time steps.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_dyson_start(
    g: &mut PseudoGreen,
    lambda0: f64,
    h: &TimeFunction,
    sigma: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k;
    check_pseudo_dyson(g, sigma, h, integ, k);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_slice(g.size1, g.size2, h.block(MATSUBARA));
    dyson_start_ret(&mut gc, lambda0, h, &sc, hstep, integ);
    let mut pws0 = PseudoDysonWorkspace::new(integ.k, g.size1, g.ntau);
    for n in 0..=k {
        pseudo_dyson_lmix_impl(n, &mut gc, lambda0, &h0, &sc, integ, beta, hstep, &mut pws0);
    }
    dyson_start_less(&mut gc, lambda0, h, &sc, beta, hstep, integ);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle Dyson equation at time step `tstp >= k + 2`.
/// Pass one [`PseudoDysonWorkspace`] built before the time-step loop to
/// reuse the scratch buffers; `None` builds a fresh workspace for the call.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_dyson_timestep(
    tstp: i64,
    g: &mut PseudoGreen,
    lambda0: f64,
    h: &TimeFunction,
    sigma: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
    pws: Option<&mut PseudoDysonWorkspace>,
) {
    let k = integ.k;
    assert!(
        tstp > k,
        "tstp must be larger than the solve order, got {tstp}"
    );
    check_pseudo_dyson(g, sigma, h, integ, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut local;
    let pws = match pws {
        Some(pws) => pws,
        None => {
            local = PseudoDysonWorkspace::new(integ.k, g.size1, g.ntau);
            &mut local
        }
    };

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_slice(g.size1, g.size2, h.block(MATSUBARA));
    pseudo_dyson_timestep_impl(tstp, &mut gc, lambda0, h, &h0, &sc, integ, beta, hstep, pws);
    g.copy_from_contour(&gc);
}

/// Solve the Matsubara component of the pseudo-particle Dyson equation.
/// Internal version on a transferred `ContourGreen`.
fn pseudo_dyson_mat_impl(
    g: &mut ContourGreen,
    lambda0: f64,
    h0: &DenseMat,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
) {
    let k = integ.k;
    let ntau = g.ntau as i64;
    let d = g.size1;
    let dtau = beta / ntau as f64;
    let id = DenseMat::identity(d);

    // initial value: G^M(0) = -1
    for q in 0..(d * d) {
        g.mat.block_mut(0)[q] = -id.raw()[q];
    }

    // the first k steps, solve sum_q M[p][q] X[q] = Q[p]
    let mut ws = BlockSolver::new(k as usize, d);
    let mut mblocks = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks = vec![DenseMat::zeros(d); k as usize];
    for n in 1..=k {
        // derivative -d/dtau G^M(n)
        for m in 0..=k {
            let cweight = -1.0 / dtau * integ.pdw.at(n as usize, m as usize);
            if m == 0 {
                let g0 = DenseMat::from_slice(d, d, g.mat.block(0));
                axpy_c(
                    &mut qblocks[(n - 1) as usize],
                    Complex::new(-cweight, 0.0),
                    &g0,
                );
            } else {
                axpy_c(
                    &mut mblocks[(n - 1) as usize][(m - 1) as usize],
                    Complex::new(cweight, 0.0),
                    &id,
                );
            }
        }
        // Hamiltonian
        axpy_c(
            &mut mblocks[(n - 1) as usize][(n - 1) as usize],
            Complex::ONE,
            &scale_c(h0, Complex::new(-1.0, 0.0)),
        );
        axpy_c(
            &mut mblocks[(n - 1) as usize][(n - 1) as usize],
            Complex::new(lambda0, 0.0),
            &id,
        );
        // integral int_0^tau dx Sigma^M(tau - x) G^M(x)
        if n < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight = dtau * integ.bcw.at(n as usize, m as usize, j as usize);
                    if j == 0 {
                        let sm = DenseMat::from_slice(d, d, sigma.mat.block(m as usize));
                        let g0 = DenseMat::from_slice(d, d, g.mat.block(0));
                        axpy_c(
                            &mut qblocks[(n - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &sm.mat_mul(&g0),
                        );
                    } else {
                        let sm = DenseMat::from_slice(d, d, sigma.mat.block(m as usize));
                        axpy_c(
                            &mut mblocks[(n - 1) as usize][(j - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &sm,
                        );
                    }
                }
            }
        } else {
            for m in 0..=k {
                let cweight = dtau * integ.giw.at(k as usize, m as usize);
                if m == 0 {
                    let sm = DenseMat::from_slice(d, d, sigma.mat.block(k as usize));
                    let g0 = DenseMat::from_slice(d, d, g.mat.block(0));
                    axpy_c(
                        &mut qblocks[(n - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &sm.mat_mul(&g0),
                    );
                } else {
                    let sm = DenseMat::from_slice(d, d, sigma.mat.block((k - m) as usize));
                    axpy_c(
                        &mut mblocks[(n - 1) as usize][(m - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &sm,
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

    // the remaining steps: BDF(k+1) and Gregory integration
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    let mut mm = DenseMat::zeros(d);
    for n in (k + 1)..=ntau {
        qq.set_from(&DenseMat::zeros(d));
        for m in 0..n {
            let w = integ.giw.at(n as usize, m as usize);
            let sm = DenseMat::from_slice(d, d, sigma.mat.block((n - m) as usize));
            let gm = DenseMat::from_slice(d, d, g.mat.block(m as usize));
            axpy_c(&mut qq, Complex::new(w, 0.0), &sm.mat_mul(&gm));
        }
        scale_c_mut(&mut qq, dtau);
        for m in 1..=(k + 1) {
            let gm = DenseMat::from_slice(d, d, g.mat.block((n - m) as usize));
            axpy_c(
                &mut qq,
                Complex::new(integ.bdw.at(m as usize) / dtau, 0.0),
                &gm,
            );
        }
        // mm = (-BDW[0]/dtau + lambda0) Id - H0 - dtau GIW[n,n] Sigma^M(0)
        mm.set_from(&scale_c(
            &id,
            Complex::new(-integ.bdw.at(0) / dtau + lambda0, 0.0),
        ));
        axpy_c(&mut mm, Complex::new(-1.0, 0.0), h0);
        let s0 = DenseMat::from_slice(d, d, sigma.mat.block(0));
        axpy_c(
            &mut mm,
            Complex::new(-dtau * integ.giw.at(n as usize, n as usize), 0.0),
            &s0,
        );
        let mut gn = DenseMat::from_slice(d, d, g.mat.block(n as usize));
        solve_block_right_single(&mut gn, &mm, &qq, &mut ws1);
        g.mat.block_mut(n as usize).copy_from_slice(gn.raw());
    }
}

/// Solve the left-mixing component `G^](n, tau)` of the pseudo-particle
/// Dyson equation. Internal version on a transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
fn pseudo_dyson_lmix_impl(
    n: i64,
    g: &mut ContourGreen,
    lambda0: f64,
    h0: &DenseMat,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
    pws: &mut PseudoDysonWorkspace,
) {
    let k = integ.k;
    let ntau = g.ntau as i64;
    let d = g.size1;
    let xi = g.sign.as_i64() as f64;
    let dtau = beta / ntau as f64;
    let n1 = n.max(k);
    let id = &pws.id;
    let h0d = h0.adjoint();

    // X(i) = G^](n, beta - tau_i), i = 0, .., ntau; accumulated, so
    // re-zeroed every call
    let xx = &mut pws.xx;
    for blk in xx.iter_mut() {
        blk.raw_mut().fill(Complex::ZERO);
    }

    // source term: int_0^t dj G^{>}(n, j) Sigma^](j, beta - tau_i)
    for j in 0..=n1 {
        let mut gtemp = pseudo_gtr_block(g, n, j);
        let w = integ.giw.at(n as usize, j as usize) * hstep;
        scale_c_mut(&mut gtemp, w);
        for i in 1..=ntau {
            let sl = DenseMat::from_slice(d, d, sigma.lmix.raw_block_at(j, ntau - i));
            axpy_c(&mut xx[i as usize], Complex::ONE, &gtemp.mat_mul(&sl));
        }
    }

    // initial value: X(0) = xi G^{>}(n, 0)
    xx[0] = pseudo_gtr_block(g, n, 0);
    scale_c_mut(&mut xx[0], xi);

    // the first k steps, solve sum_q X[q] M[p][q] = Q[p]
    for row in pws.lm_mblocks.iter_mut().flatten() {
        row.raw_mut().fill(Complex::ZERO);
    }
    let mblocks = &mut pws.lm_mblocks;
    let mut qblocks: Vec<DenseMat> = xx[1..=(k as usize)].to_vec();
    for i in 1..=k {
        // derivative -d/dl X(i)
        for m in 0..=k {
            let cweight = -1.0 / dtau * integ.pdw.at(i as usize, m as usize);
            if m == 0 {
                axpy_c(
                    &mut qblocks[(i - 1) as usize],
                    Complex::new(-cweight, 0.0),
                    &xx[0],
                );
            } else {
                axpy_c(
                    &mut mblocks[(i - 1) as usize][(m - 1) as usize],
                    Complex::new(cweight, 0.0),
                    id,
                );
            }
        }
        // Hamiltonian
        axpy_c(
            &mut mblocks[(i - 1) as usize][(i - 1) as usize],
            Complex::ONE,
            &scale_c(&h0d, Complex::new(-1.0, 0.0)),
        );
        axpy_c(
            &mut mblocks[(i - 1) as usize][(i - 1) as usize],
            Complex::new(lambda0, 0.0),
            id,
        );
        // integral int_0^l dx X(x) Sigma^M(l - x)
        if i < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight = dtau * integ.bcw.at(i as usize, m as usize, j as usize);
                    if j == 0 {
                        let sm = DenseMat::from_slice(d, d, sigma.mat.block(m as usize));
                        axpy_c(
                            &mut qblocks[(i - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &xx[0].mat_mul(&sm),
                        );
                    } else {
                        let sm = DenseMat::from_slice(d, d, sigma.mat.block(m as usize));
                        axpy_c(
                            &mut mblocks[(i - 1) as usize][(j - 1) as usize],
                            Complex::new(-cweight, 0.0),
                            &sm,
                        );
                    }
                }
            }
        } else {
            for m in 0..=k {
                let cweight = dtau * integ.giw.at(i as usize, m as usize);
                if m == 0 {
                    let sm = DenseMat::from_slice(d, d, sigma.mat.block(k as usize));
                    axpy_c(
                        &mut qblocks[(i - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &xx[0].mat_mul(&sm),
                    );
                } else {
                    let sm = DenseMat::from_slice(d, d, sigma.mat.block((k - m) as usize));
                    axpy_c(
                        &mut mblocks[(i - 1) as usize][(m - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &sm,
                    );
                }
            }
        }
    }
    solve_block_left(&mut xx[1..=(k as usize)], mblocks, &qblocks, &mut pws.lm_ws);

    // the remaining steps: BDF(k+1) and Gregory integration
    let ws1 = &mut pws.ws1;
    let qq = &mut pws.qq;
    let mm = &mut pws.mm;
    for i in (k + 1)..=ntau {
        qq.raw_mut().fill(Complex::ZERO);
        for m in 0..i {
            let w = integ.giw.at(i as usize, m as usize);
            let sm = DenseMat::from_slice(d, d, sigma.mat.block((i - m) as usize));
            axpy_c(qq, Complex::new(w, 0.0), &xx[m as usize].mat_mul(&sm));
        }
        scale_c_mut(qq, dtau);
        axpy_c(qq, Complex::ONE, &xx[i as usize]);
        for m in 1..=(k + 1) {
            axpy_c(
                qq,
                Complex::new(integ.bdw.at(m as usize) / dtau, 0.0),
                &xx[(i - m) as usize],
            );
        }
        mm.set_from(&scale_c(
            id,
            Complex::new(-integ.bdw.at(0) / dtau + lambda0, 0.0),
        ));
        axpy_c(mm, Complex::new(-1.0, 0.0), &h0d);
        let s0 = DenseMat::from_slice(d, d, sigma.mat.block(0));
        axpy_c(
            mm,
            Complex::new(-dtau * integ.giw.at(i as usize, i as usize), 0.0),
            &s0,
        );
        solve_block_left_single(&mut xx[i as usize], mm, qq, ws1);
    }

    // G^](n, tau_{ntau - l}) = X(l)
    for l in 0..=ntau {
        g.lmix
            .raw_block_at_mut(n, ntau - l)
            .copy_from_slice(xx[l as usize].raw());
    }
}

/// Solve one real time step `tstp > k` of the pseudo-particle Dyson
/// equation on a transferred `ContourGreen`: the retarded, left-mixing,
/// and lesser components are updated in place.
#[allow(clippy::too_many_arguments)]
fn pseudo_dyson_timestep_impl(
    tstp: i64,
    g: &mut ContourGreen,
    lambda0: f64,
    h: &TimeFunction,
    h0: &DenseMat,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
    pws: &mut PseudoDysonWorkspace,
) {
    dyson_timestep_ret_ws(tstp, g, lambda0, h, sigma, hstep, integ, &mut pws.dw);
    pseudo_dyson_lmix_impl(tstp, g, lambda0, h0, sigma, integ, beta, hstep, pws);
    dyson_timestep_less_ws(tstp, g, lambda0, h, sigma, beta, hstep, integ, &mut pws.dw);
}

/// `scale_c` in-place (the local helper of this module).
fn scale_c_mut(m: &mut DenseMat, w: f64) {
    for q in 0..m.raw().len() {
        let v = m.raw()[q];
        m.raw_mut()[q] = v * Complex::new(w, 0.0);
    }
}

/// Check the arguments of the pseudo-particle Dyson solvers.
fn check_pseudo_dyson(
    g: &PseudoGreen,
    sigma: &PseudoGreen,
    h: &TimeFunction,
    integ: &Integrator,
    nmax: i64,
) {
    check_pseudo_dyson_common(g, sigma, integ);
    if nmax >= 0 {
        assert!(
            g.nt as i64 >= nmax,
            "G must have at least {} real time steps",
            nmax + 1
        );
        assert!(
            h.nt as i64 >= nmax,
            "H must have at least {} real time steps",
            nmax + 1
        );
    }
    let d = g.size1;
    // Hamiltonian checks over the needed steps (the
    // Matsubara axis and 0..nmax), including the hermiticity check
    let tstps: Vec<i64> = if nmax >= 0 {
        (MATSUBARA..=nmax).collect()
    } else {
        vec![MATSUBARA]
    };
    check_hamiltonian(h, tstps.into_iter(), d);
}

/// The shared part of `_check_pseudo_dyson` (no Hamiltonian checks).
fn check_pseudo_dyson_common(g: &PseudoGreen, sigma: &PseudoGreen, integ: &Integrator) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order must be in 1..=5, got {}",
        integ.k
    );
    assert!(
        g.sign == sigma.sign,
        "G and Sigma must have the same statistics"
    );
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        sigma.size1 == g.size1 && sigma.size2 == g.size2,
        "G and Sigma have different sizes"
    );
    assert!(
        g.ntau == sigma.ntau,
        "G and Sigma have different imaginary time grids"
    );
    assert!(
        g.nt == sigma.nt,
        "G and Sigma have different real time grids"
    );
    assert!(
        g.ntau as i64 > integ.k,
        "ntau must be larger than solve_order + 1"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

    /// One pseudo workspace across the loop equals the convenience
    /// path (fresh workspace per step).
    #[test]
    fn pseudo_dyson_timestep_ws_matches_fresh_workspace() {
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
        let mut sig = PseudoGreen::new(&c, Sign::Fermi, Complex::ZERO);
        {
            let mut sc = sig.to_contour();
            from_ham(&mut sc, 0.0, &h0, beta, h);
            sig.copy_from_contour(&sc);
        }
        let integ = Integrator::new(k);
        let mut g1 = PseudoGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = PseudoGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for g in [&mut g1, &mut g2] {
            pseudo_dyson_mat(g, 0.0, &h0, &sig, &integ, beta);
            pseudo_dyson_start(g, 0.0, &hh, &sig, &integ, beta, h);
        }
        let mut pws = PseudoDysonWorkspace::new(k, 2, ntau as usize);
        for tstp in (k + 1)..=nt {
            pseudo_dyson_timestep(tstp, &mut g1, 0.0, &hh, &sig, &integ, beta, h, None);
            pseudo_dyson_timestep(
                tstp,
                &mut g2,
                0.0,
                &hh,
                &sig,
                &integ,
                beta,
                h,
                Some(&mut pws),
            );
        }
        assert_eq!(g1.mat.raw(), g2.mat.raw());
        assert_eq!(g1.ret.raw(), g2.ret.raw());
        assert_eq!(g1.lmix.raw(), g2.lmix.raw());
        assert_eq!(g1.less.raw(), g2.less.raw());
    }
}
