// Project : KadanoffBaym
// File    : src/vie2/mat.rs
// Purpose : Matsubara component of the VIE2: fixpoint, conjugate
//           gradient, and frequency-domain methods

use crate::base::MatsubaraMethod;
use crate::types::ContourGreen;
use crate::weights::Integrator;

use crate::base::{matsubara_cg, matsubara_fixpoint, matsubara_seed};

use crate::vie2::check::{check_vie2, check_vie2_order};

///
/// *Remarks* : *Matsubara VIE2*
///
/// The three methods of [`MatsubaraMethod`] solve the same equation:
///
/// * `Fixpoint`: start from the frequency-domain seed (`pcf = 5`), then
///   apply `nfixpoint` preconditioned residual corrections
///   `R = F*G + G - Q`, `G -> G - P^{-1}R`.
///
/// * `Cg`: start from the frequency-domain seed (`pcf = 3`), then apply
///   the Fletcher-Reeves conjugate gradient method to the hermitian
///   operator `A(X) = X + [F*X + X*Fcc]/2` with the Gregory-weighted
///   inner product.
///
/// * `Fourier`: the frequency-domain seed itself (`pcf = 20` and the
///   cubically corrected transform); the number of imaginary time
///   intervals must be even.
///
///
/// Solve the Matsubara component of the VIE2 `[1 + F] * G = Q`; only
/// the Matsubara component of `g` is modified.
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
            vie2_mat_fourier(g, f, q, beta, 20, 3);
        }
    }
}

/// The Fourier method of the Matsubara VIE2: `X` is the seed with the
/// frequency window `pcf` and the interpolation `order`.
pub(crate) fn vie2_mat_fourier(
    x: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    pcf: i64,
    order: i64,
) {
    matsubara_seed(&mut x.mat, &f.mat, &q.mat, beta, x.sign, pcf, order);
}

/// Fixpoint solver.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_mat_fixpoint(
    g: &mut ContourGreen,
    f: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    integ: &Integrator,
    nfixpoint: i64,
) {
    matsubara_fixpoint(&mut g.mat, &f.mat, &q.mat, beta, g.sign, integ, nfixpoint);
}

/// Fletcher-Reeves conjugate gradient solver; returns the number of
/// performed iterations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_mat_cg(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    integ: &Integrator,
    maxiter: i64,
    tol: f64,
) -> i64 {
    matsubara_cg(
        &mut g.mat, &f.mat, &fcc.mat, &q.mat, beta, g.sign, integ, maxiter, tol,
    )
}
