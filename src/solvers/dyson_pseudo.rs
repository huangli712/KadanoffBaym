// Project : KadanoffBaym
// File    : src/solvers/dyson_pseudo.rs
// Purpose : pseudo-particle Dyson equation (blueprint
//           julia/src/solvers/dyson_pseudo.jl)
//
// The pseudo-particle Dyson equation reads
// `[i d/dt + lambda_0 - H(t) - Sigma] * G = 1`, where `lambda_0` is
// not included in `H`. The greater (stored as `ret`) and lesser
// components are solved by the standard real-time Dyson kernels
// (`dyson_start_ret`, `dyson_start_less`, `dyson_timestep_ret`,
// `dyson_timestep_less`). Only the Matsubara and the left-mixing
// components are special:
//
// * Matsubara component, a causal initial value problem with
//   `G^M(0) = -1`. The first `k` points are solved together
//   (polynomial differentiation and boundary corrections), the others
//   by the BDF(k+1) and Gregory rules.
//
// * Left-mixing component. Let `X(l) = G^](t, beta - l)`, then
//   `[-d/dl + lambda_0 - H_0^dagger] X(l) - int_0^l dl' X(l')
//   Sigma^M(l - l') = int_0^t dt' G^>(t, t') Sigma^](t', beta - l)`
//   with `X(0) = xi G^>(t, 0)`, solved from `l = 0` to `l = beta` for
//   fixed `t`. Note that `H_0` (the Hamiltonian at the Matsubara axis)
//   is used.

use crate::complex::Complex;
use crate::linalg::linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    BlockSolver,
};
use crate::linalg::mat::{axpy_c, scale_c, DenseMat};
use crate::types::green::{ContourGreen, TimeFunction};
use crate::types::pseudo::PseudoGreen;
use crate::weights::Integrator;

use super::dyson_start::{dyson_start_less, dyson_start_ret};
use super::dyson_timestep::{dyson_timestep_less, dyson_timestep_ret};

/// The greater block `G^{>}(t_n, t_j)` of the pseudo-particle Green's
/// function (the stored `ret` component); for `j > n` the value is
/// reconstructed as `-(G^{>}(t_j, t_n))^dagger`.
pub(crate) fn gtr_blk(g: &ContourGreen, n: i64, j: i64) -> DenseMat {
    if n >= j {
        DenseMat::from_vec(g.size1, g.size2, g.ret.raw_block_at(n, j).to_vec())
    } else {
        let src = g.ret.raw_block_at(j, n);
        let d1 = g.size1;
        let d2 = g.size2;
        let mut out = DenseMat::zeros_shape(d1, d2);
        for a in 0..d1 {
            for b in 0..d2 {
                out.set(a, b, -src[b * d2 + a].conj());
            }
        }
        out
    }
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

/// Check the arguments of the pseudo-particle Dyson solvers (blueprint
/// `_check_pseudo_dyson`).
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
    // the blueprint _check_hamiltonian over the needed steps (the
    // Matsubara axis and 0..nmax), including the hermiticity check
    let tstps: Vec<i64> = if nmax >= 0 {
        (crate::basics::MATSUBARA..=nmax).collect()
    } else {
        vec![crate::basics::MATSUBARA]
    };
    super::dyson::check_hamiltonian(h, tstps.into_iter(), d);
}

/// Solve the Matsubara component of the pseudo-particle Dyson equation
/// (blueprint `_pseudo_dyson_mat!`). Internal version on a
/// transferred `ContourGreen`.
pub(crate) fn pseudo_dyson_mat_impl(
    g: &mut ContourGreen,
    lambda0: f64,
    h0: &DenseMat,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
) {
    let k = integ.k as i64;
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
                let g0 = DenseMat::from_vec(d, d, g.mat.block(0).to_vec());
                axpy_c(&mut qblocks[(n - 1) as usize], Complex::new(-cweight, 0.0), &g0);
            } else {
                axpy_c(
                    &mut mblocks[(n - 1) as usize][(m - 1) as usize],
                    Complex::new(cweight, 0.0),
                    &id,
                );
            }
        }
        // Hamiltonian
        axpy_c(&mut mblocks[(n - 1) as usize][(n - 1) as usize], Complex::ONE, &scale_c(h0, Complex::new(-1.0, 0.0)));
        axpy_c(
            &mut mblocks[(n - 1) as usize][(n - 1) as usize],
            Complex::new(lambda0, 0.0),
            &id,
        );
        // integral int_0^tau dx Sigma^M(tau - x) G^M(x)
        if n < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight =
                        dtau * integ.bcw.at(n as usize, m as usize, j as usize);
                    if j == 0 {
                        let sm = DenseMat::from_vec(d, d, sigma.mat.block(m as usize).to_vec());
                        let g0 = DenseMat::from_vec(d, d, g.mat.block(0).to_vec());
                        axpy_c(
                            &mut qblocks[(n - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &sm.mat_mul(&g0),
                        );
                    } else {
                        let sm = DenseMat::from_vec(d, d, sigma.mat.block(m as usize).to_vec());
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
                    let sm = DenseMat::from_vec(d, d, sigma.mat.block(k as usize).to_vec());
                    let g0 = DenseMat::from_vec(d, d, g.mat.block(0).to_vec());
                    axpy_c(
                        &mut qblocks[(n - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &sm.mat_mul(&g0),
                    );
                } else {
                    let sm = DenseMat::from_vec(
                        d,
                        d,
                        sigma.mat.block((k - m) as usize).to_vec(),
                    );
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
            let sm = DenseMat::from_vec(d, d, sigma.mat.block((n - m) as usize).to_vec());
            let gm = DenseMat::from_vec(d, d, g.mat.block(m as usize).to_vec());
            axpy_c(&mut qq, Complex::new(w, 0.0), &sm.mat_mul(&gm));
        }
        scale_c_mut(&mut qq, dtau);
        for m in 1..=(k + 1) {
            let gm = DenseMat::from_vec(d, d, g.mat.block((n - m) as usize).to_vec());
            axpy_c(&mut qq, Complex::new(integ.bdw.at(m as usize) / dtau, 0.0), &gm);
        }
        // mm = (-BDW[0]/dtau + lambda0) Id - H0 - dtau GIW[n,n] Sigma^M(0)
        mm.set_from(&scale_c(&id, Complex::new(-integ.bdw.at(0) / dtau + lambda0, 0.0)));
        axpy_c(&mut mm, Complex::new(-1.0, 0.0), h0);
        let s0 = DenseMat::from_vec(d, d, sigma.mat.block(0).to_vec());
        axpy_c(
            &mut mm,
            Complex::new(-dtau * integ.giw.at(n as usize, n as usize), 0.0),
            &s0,
        );
        let mut gn = DenseMat::from_vec(d, d, g.mat.block(n as usize).to_vec());
        solve_block_right_single(&mut gn, &mm, &qq, &mut ws1);
        g.mat.block_mut(n as usize).copy_from_slice(gn.raw());
    }
}

/// `scale_c` in-place (the local helper of this module).
fn scale_c_mut(m: &mut DenseMat, w: f64) {
    for q in 0..m.raw().len() {
        let v = m.raw()[q];
        m.raw_mut()[q] = v * Complex::new(w, 0.0);
    }
}

/// Solve the left-mixing component `G^](n, tau)` of the pseudo-particle
/// Dyson equation (blueprint `_pseudo_dyson_lmix!`). Internal version
/// on a transferred `ContourGreen`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_dyson_lmix_impl(
    n: i64,
    g: &mut ContourGreen,
    lambda0: f64,
    h0: &DenseMat,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    hstep: f64,
) {
    let k = integ.k as i64;
    let ntau = g.ntau as i64;
    let d = g.size1;
    let xi = g.sign.as_i64() as f64;
    let dtau = beta / ntau as f64;
    let n1 = n.max(k);
    let id = DenseMat::identity(d);
    let h0d = h0.adjoint();

    // X(i) = G^](n, beta - tau_i), i = 0, .., ntau
    let mut xx: Vec<DenseMat> = (0..=ntau).map(|_| DenseMat::zeros(d)).collect();

    // source term: int_0^t dj G^{>}(n, j) Sigma^](j, beta - tau_i)
    for j in 0..=n1 {
        let mut gtemp = gtr_blk(g, n, j);
        let w = integ.giw.at(n as usize, j as usize) * hstep;
        scale_c_mut(&mut gtemp, w);
        for i in 1..=ntau {
            let sl = DenseMat::from_vec(d, d, sigma.lmix.raw_block_at(j, ntau - i).to_vec());
            axpy_c(&mut xx[i as usize], Complex::ONE, &gtemp.mat_mul(&sl));
        }
    }

    // initial value: X(0) = xi G^{>}(n, 0)
    xx[0] = gtr_blk(g, n, 0);
    scale_c_mut(&mut xx[0], xi);

    // the first k steps, solve sum_q X[q] M[p][q] = Q[p]
    let mut ws = BlockSolver::new(k as usize, d);
    let mut mblocks = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
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
                    &id,
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
            &id,
        );
        // integral int_0^l dx X(x) Sigma^M(l - x)
        if i < k {
            for m in 0..=k {
                for j in 0..=k {
                    let cweight =
                        dtau * integ.bcw.at(i as usize, m as usize, j as usize);
                    if j == 0 {
                        let sm = DenseMat::from_vec(d, d, sigma.mat.block(m as usize).to_vec());
                        axpy_c(
                            &mut qblocks[(i - 1) as usize],
                            Complex::new(cweight, 0.0),
                            &xx[0].mat_mul(&sm),
                        );
                    } else {
                        let sm = DenseMat::from_vec(d, d, sigma.mat.block(m as usize).to_vec());
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
                    let sm = DenseMat::from_vec(d, d, sigma.mat.block(k as usize).to_vec());
                    axpy_c(
                        &mut qblocks[(i - 1) as usize],
                        Complex::new(cweight, 0.0),
                        &xx[0].mat_mul(&sm),
                    );
                } else {
                    let sm = DenseMat::from_vec(
                        d,
                        d,
                        sigma.mat.block((k - m) as usize).to_vec(),
                    );
                    axpy_c(
                        &mut mblocks[(i - 1) as usize][(m - 1) as usize],
                        Complex::new(-cweight, 0.0),
                        &sm,
                    );
                }
            }
        }
    }
    solve_block_left(&mut xx[1..=(k as usize)], &mblocks, &qblocks, &mut ws);

    // the remaining steps: BDF(k+1) and Gregory integration
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    let mut mm = DenseMat::zeros(d);
    for i in (k + 1)..=ntau {
        qq.set_from(&DenseMat::zeros(d));
        for m in 0..i {
            let w = integ.giw.at(i as usize, m as usize);
            let sm = DenseMat::from_vec(d, d, sigma.mat.block((i - m) as usize).to_vec());
            axpy_c(&mut qq, Complex::new(w, 0.0), &xx[m as usize].mat_mul(&sm));
        }
        scale_c_mut(&mut qq, dtau);
        axpy_c(&mut qq, Complex::ONE, &xx[i as usize]);
        for m in 1..=(k + 1) {
            axpy_c(
                &mut qq,
                Complex::new(integ.bdw.at(m as usize) / dtau, 0.0),
                &xx[(i - m) as usize],
            );
        }
        mm.set_from(&scale_c(&id, Complex::new(-integ.bdw.at(0) / dtau + lambda0, 0.0)));
        axpy_c(&mut mm, Complex::new(-1.0, 0.0), &h0d);
        let s0 = DenseMat::from_vec(d, d, sigma.mat.block(0).to_vec());
        axpy_c(
            &mut mm,
            Complex::new(-dtau * integ.giw.at(i as usize, i as usize), 0.0),
            &s0,
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

/// Solve the Matsubara component of the pseudo-particle Dyson equation
/// (blueprint `pseudo_dyson_mat!`).
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
/// time steps (blueprint `pseudo_dyson_start!`).
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
    let k = integ.k as i64;
    check_pseudo_dyson(g, sigma, h, integ, k);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_vec(g.size1, g.size2, h.block(crate::basics::MATSUBARA).to_vec());
    dyson_start_ret(&mut gc, lambda0, h, &sc, hstep, integ);
    for n in 0..=k {
        pseudo_dyson_lmix_impl(n, &mut gc, lambda0, &h0, &sc, integ, beta, hstep);
    }
    dyson_start_less(&mut gc, lambda0, h, &sc, beta, hstep, integ);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle Dyson equation at time step `tstp >= k + 2`
/// (blueprint `pseudo_dyson_timestep!`).
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
) {
    let k = integ.k as i64;
    assert!(
        tstp >= k + 1,
        "tstp must be larger than the solve order, got {tstp}"
    );
    check_pseudo_dyson(g, sigma, h, integ, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_vec(g.size1, g.size2, h.block(crate::basics::MATSUBARA).to_vec());
    dyson_timestep_ret(tstp, &mut gc, lambda0, h, &sc, hstep, integ);
    pseudo_dyson_lmix_impl(tstp, &mut gc, lambda0, &h0, &sc, integ, beta, hstep);
    dyson_timestep_less(tstp, &mut gc, lambda0, h, &sc, beta, hstep, integ);
    g.copy_from_contour(&gc);
}

/// Solve the pseudo-particle Dyson equation at the Matsubara axis and
/// at all the real time steps (blueprint `pseudo_dyson!`).
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
    let k = integ.k as i64;
    check_pseudo_dyson(g, sigma, h, integ, g.nt as i64 - 1);

    let mut gc = g.to_contour();
    let sc = sigma.to_contour();
    let h0 = DenseMat::from_vec(g.size1, g.size2, h.block(crate::basics::MATSUBARA).to_vec());
    pseudo_dyson_mat_impl(&mut gc, lambda0, &h0, &sc, integ, beta);
    dyson_start_ret(&mut gc, lambda0, h, &sc, hstep, integ);
    for n in 0..=k {
        pseudo_dyson_lmix_impl(n, &mut gc, lambda0, &h0, &sc, integ, beta, hstep);
    }
    dyson_start_less(&mut gc, lambda0, h, &sc, beta, hstep, integ);
    for tstp in (k + 1)..=gc.nt as i64 {
        dyson_timestep_ret(tstp, &mut gc, lambda0, h, &sc, hstep, integ);
        pseudo_dyson_lmix_impl(tstp, &mut gc, lambda0, &h0, &sc, integ, beta, hstep);
        dyson_timestep_less(tstp, &mut gc, lambda0, h, &sc, beta, hstep, integ);
    }
    g.copy_from_contour(&gc);
}
