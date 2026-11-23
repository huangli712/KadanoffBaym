// Project : KadanoffBaym
// File    : src/base/mat.rs
// Purpose : shared Matsubara-axis solving components of the Dyson and
//           VIE2 families: the method selector, the Gregory-weighted
//           inner product, and the fixpoint / conjugate-gradient
//           iteration drivers

//!
//! *Remarks* : *Solver Commons*
//!
//! The components sit in a separate layer above the kernel algebra
//! (`convolution::conv_mat`) and the frequency-domain seed
//! (`base::seed::MatsubaraSeed`), and are shared by both solver families:
//!
//! `{dyson, vie2} -> base -> {convolution, fourier, types,
//! weights}`.
//!

use crate::base::MatsubaraSeed;
use crate::complex::Complex;
use crate::types::{MatComp, Sign};
use crate::weights::Integrator;

use crate::convolution::conv_mat;

/// Method for the Matsubara component of the Dyson and VIE2 equations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatsubaraMethod {
    /// Preconditioned residual corrections (default, 6 iterations).
    Fixpoint,

    /// Fletcher-Reeves conjugate gradient.
    Cg,

    /// Frequency-domain solution.
    Fourier,
}

/// Inner product of two Matsubara functions,
/// `sum_r w_r sum_ab Re[(X^dag(tau_r) Y(tau_r))_ab]` (the real part of
/// all the elements of the matrix product), where `w_r` are the Gregory
/// end weights (1 in the interior).
pub fn mat_inner(x: &MatComp, y: &MatComp, integ: &Integrator) -> f64 {
    let n = x.ntau as i64;
    let k = integ.k;
    let d = x.size1;
    let mut s = 0.0f64;
    for r in 0..=n {
        let w = if r <= k {
            integ.giw.omega[r as usize]
        } else if r >= n - k {
            integ.giw.omega[(n - r) as usize]
        } else {
            1.0
        };
        // sum over all the elements of X^dagger Y
        let xb = x.block(r as usize);
        let yb = y.block(r as usize);
        let mut acc = 0.0f64;
        for a in 0..d {
            for b in 0..d {
                let mut z = Complex::ZERO;
                for kk in 0..d {
                    z += xb[kk * d + a].conj() * yb[kk * d + b];
                }
                acc += z.re;
            }
        }
        s += w * acc;
    }
    s
}

/// Fletcher-Reeves conjugate gradient for a hermitian Matsubara
/// operator, applied through the `apply_a` closure (both solvers use
/// `A(X) = X + [F * X + X * Fcc] / 2`); solves `A(G) = Q` starting
/// from the current value of `g_mat` and returns the number of
/// performed iterations.
///
/// The two convergence tests differ in scale, as established by the
/// reference solver: no iteration is performed when the initial
/// residual satisfies `||R||^2 <= tol`, while the iterations stop as
/// soon as `||R|| < tol`. The count is capped by `maxiter`.
pub(crate) fn matsubara_cg(
    g_mat: &mut MatComp,
    q_mat: &MatComp,
    apply_a: &mut dyn FnMut(&mut MatComp, &MatComp),
    integ: &Integrator,
    maxiter: usize,
    tol: f64,
) -> i64 {
    let mut r = MatComp::new(g_mat.ntau, g_mat.size1, g_mat.size2, Complex::ZERO);
    let mut p = MatComp::new(g_mat.ntau, g_mat.size1, g_mat.size2, Complex::ZERO);
    let mut ap = MatComp::new(g_mat.ntau, g_mat.size1, g_mat.size2, Complex::ZERO);

    // initial residual R = Q - A(G)
    apply_a(&mut r, g_mat);
    for q in 0..r.raw().len() {
        r.raw_mut()[q] = q_mat.raw()[q] - r.raw_mut()[q];
    }
    p.raw_mut().copy_from_slice(r.raw());
    let mut rsold = mat_inner(&r, &r, integ);
    assert!(rsold.is_finite(), "residual is not finite");

    let mut niter = 0i64;
    if rsold > tol {
        for _ in 0..maxiter {
            niter += 1;
            apply_a(&mut ap, &p);
            let c2v = mat_inner(&p, &ap, integ);
            assert!(
                c2v.is_finite() && c2v != 0.0,
                "zero or non-finite denominator in CG"
            );
            let alpha = rsold / c2v;
            for q in 0..g_mat.raw().len() {
                g_mat.raw_mut()[q] += Complex::new(alpha, 0.0) * p.raw()[q];
                r.raw_mut()[q] -= Complex::new(alpha, 0.0) * ap.raw()[q];
            }
            let rsnew = mat_inner(&r, &r, integ);
            assert!(rsnew.is_finite(), "residual is not finite");
            let bcg = rsnew / rsold;
            for q in 0..p.raw().len() {
                p.raw_mut()[q] = r.raw()[q] + Complex::new(bcg, 0.0) * p.raw()[q];
            }
            rsold = rsnew;
            if rsold.sqrt() < tol {
                break;
            }
        }
    }
    niter
}

/// Preconditioned residual corrections: starting from the
/// frequency-domain seed solution of `[1 + F] G = Q` (`F` fixed), apply
/// `G <- G - P^{-1} R` with `R = F * G + G - Q` for `maxiter` times.
/// The preconditioner `P^{-1}` is the same seeded solve applied to the
/// residual.
pub(crate) fn matsubara_fixpoint(
    g_mat: &mut MatComp,
    f_mat: &MatComp,
    q_mat: &MatComp,
    beta: f64,
    sign: Sign,
    integ: &Integrator,
    maxiter: i64,
) {
    // the kernel F is fixed across the iterations: its transforms are
    // computed once (MatsubaraSeed) and reused
    let mut seed = MatsubaraSeed::new(f_mat, beta, sign, 5, 3);
    seed.solve(g_mat, q_mat);

    let mut r = MatComp::new(g_mat.ntau, g_mat.size1, g_mat.size2, Complex::ZERO);
    let mut dg = MatComp::new(g_mat.ntau, g_mat.size1, g_mat.size2, Complex::ZERO);
    for _ in 0..maxiter {
        // R = F * G + G - Q
        conv_mat(&mut r, f_mat, g_mat, integ, beta, sign);
        for k in 0..r.raw().len() {
            r.raw_mut()[k] += g_mat.raw()[k] - q_mat.raw()[k];
        }
        seed.solve(&mut dg, &r);
        for k in 0..g_mat.raw().len() {
            g_mat.raw_mut()[k] -= dg.raw()[k];
        }
    }
}
