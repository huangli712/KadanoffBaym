// Project : KadanoffBaym
// File    : src/solvers/vie2_mat.rs
// Purpose : Matsubara component of the VIE2: fixpoint, conjugate
//           gradient, and frequency-domain methods (blueprint:
//           solvers/vie2_mat.jl)

use super::dyson_mat::MatsubaraMethod;
use super::vie2::check_vie2;
use crate::complex::Complex;
use crate::convolution::mat::conv_mat;
use crate::fourier::matsubara::matsubara_seed;
use crate::types::green::{ContourGreen, MatComp};
use crate::weights::Integrator;

/// Fixpoint solver (blueprint `_vie2_mat_fixpoint!`).
#[allow(clippy::too_many_arguments)]
fn vie2_mat_fixpoint(
    g: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    integ: &Integrator,
    nfixpoint: i64,
) {
    let sign = g.sign;
    matsubara_seed(&mut g.mat, &f.mat, &q.mat, beta, sign, 5, 3);

    let mut r = MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO);
    let mut dg = MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO);
    for _ in 0..nfixpoint {
        // R = F * G + G - Q
        conv_mat(&mut r, &f.mat, &g.mat, integ, beta, sign);
        for k in 0..r.raw().len() {
            r.raw_mut()[k] += g.mat.raw()[k] - q.mat.raw()[k];
        }
        matsubara_seed(&mut dg, &f.mat, &r, beta, sign, 5, 3);
        for k in 0..g.mat.raw().len() {
            g.mat.raw_mut()[k] -= dg.raw()[k];
        }
    }
}

/// Fletcher-Reeves conjugate gradient solver (blueprint
/// `_vie2_mat_cg!`); returns the number of performed iterations.
#[allow(clippy::too_many_arguments)]
fn vie2_mat_cg(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    integ: &Integrator,
    maxiter: i64,
    tol: f64,
) -> i64 {
    let sign = g.sign;
    let d = g.size1;
    matsubara_seed(&mut g.mat, &f.mat, &q.mat, beta, sign, 3, 3);
    if maxiter <= 0 {
        return 0;
    }

    let mut r = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut p = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut ap = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut c1 = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut c2 = MatComp::new(g.ntau, d, d, Complex::ZERO);

    // A(X) = X + [F * X + X * Fcc] / 2
    let mut apply_a = |ax: &mut MatComp, x: &MatComp| {
        conv_mat(&mut c1, &f.mat, x, integ, beta, sign);
        conv_mat(&mut c2, x, &fcc.mat, integ, beta, sign);
        for k in 0..ax.raw().len() {
            ax.raw_mut()[k] = (c1.raw()[k] + c2.raw()[k]) * Complex::new(0.5, 0.0) + x.raw()[k];
        }
    };

    // initial residual R = Q - A(G)
    apply_a(&mut r, &g.mat);
    for k in 0..r.raw().len() {
        r.raw_mut()[k] = q.mat.raw()[k] - r.raw()[k];
    }
    p.raw_mut().copy_from_slice(r.raw());
    let mut rsold = super::dyson_mat::mat_inner(&r, &r, integ);
    assert!(rsold.is_finite(), "residual is not finite");

    let mut niter = 0i64;
    if rsold > tol {
        for _ in 0..maxiter {
            niter += 1;
            apply_a(&mut ap, &p);
            let c2v = super::dyson_mat::mat_inner(&p, &ap, integ);
            assert!(
                c2v.is_finite() && c2v != 0.0,
                "zero or non-finite denominator in CG"
            );
            let alpha = rsold / c2v;
            for k in 0..g.mat.raw().len() {
                g.mat.raw_mut()[k] += Complex::new(alpha, 0.0) * p.raw()[k];
                r.raw_mut()[k] -= Complex::new(alpha, 0.0) * ap.raw()[k];
            }
            let rsnew = super::dyson_mat::mat_inner(&r, &r, integ);
            assert!(rsnew.is_finite(), "residual is not finite");
            let bcg = rsnew / rsold;
            for k in 0..p.raw().len() {
                p.raw_mut()[k] = r.raw()[k] + Complex::new(bcg, 0.0) * p.raw()[k];
            }
            rsold = rsnew;
            if rsold.sqrt() < tol {
                break;
            }
        }
    }
    niter
}

/// Solve the Matsubara component of the VIE2 `[1 + F] * G = Q`; only
/// the Matsubara component of `g` is modified (blueprint `vie2_mat!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_mat(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    integ: &Integrator,
    method: MatsubaraMethod,
    nfixpoint: i64,
    maxiter: i64,
    tol: f64,
) {
    check_vie2(g, f, fcc, q);
    check_vie2_order(integ);
    assert!(beta > 0.0, "beta must be positive");
    assert!(nfixpoint >= 0, "nfixpoint must be nonnegative");
    assert!(maxiter >= 0, "maxiter must be nonnegative");

    match method {
        MatsubaraMethod::Fixpoint => {
            vie2_mat_fixpoint(g, f, q, beta, integ, nfixpoint);
        }
        MatsubaraMethod::Cg => {
            vie2_mat_cg(g, f, fcc, q, beta, integ, maxiter, tol);
        }
        MatsubaraMethod::Fourier => {
            // the seed IS the frequency-domain solution (pcf = 20)
            let n = g.ntau as i64;
            assert!(
                n % 2 == 0,
                "the Fourier method needs an even number of imaginary time intervals, got {n}"
            );
            matsubara_seed(&mut g.mat, &f.mat, &q.mat, beta, g.sign, 20, 3);
        }
    }
}

use super::vie2::check_vie2_order;

/// The Fourier method of the Matsubara VIE2 (blueprint
/// `_vie2_mat_fourier!`): `X` is the seed with the frequency window
/// `pcf` and the interpolation `order`.
pub fn vie2_mat_fourier(
    x: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    pcf: i64,
    order: i64,
) {
    matsubara_seed(&mut x.mat, &f.mat, &q.mat, beta, x.sign, pcf, order);
}
