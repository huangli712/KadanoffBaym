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

/// Inner product of two Matsubara functions,
/// `sum_r w_r sum_ab Re[(X^dag(tau_r) Y(tau_r))_ab]` (the real part of
/// all the elements of the matrix product), where `w_r` are the Gregory
/// end weights (1 in the interior).
fn mat_inner(x: &MatComp, y: &MatComp, integ: &Integrator) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    /// The inner product sums the real parts of all the elements of
    /// `X^dagger Y` (not the trace); the Gregory end weights scale the
    /// first and the last Matsubara points.
    #[test]
    fn inner_product_real_parts_and_end_weights() {
        let (ntau, n) = (80usize, 80usize);
        let integ = Integrator::new(5);
        let mut x = MatComp::new(ntau, 2, 2, Complex::ZERO);
        let mut y = MatComp::new(ntau, 2, 2, Complex::ZERO);
        let xb = [c(1.0, 0.0), c(0.0, 2.0), c(0.5, 0.0), c(-1.0, 0.0)];
        let yb = [c(0.3, 0.0), c(1.0, 0.0), c(0.0, -2.0), c(0.7, 0.0)];
        x.block_mut(39).copy_from_slice(&xb);
        y.block_mut(39).copy_from_slice(&yb);
        // sum of the real parts of all the elements of X^dagger Y
        let prod = [
            xb[0].conj() * yb[0] + xb[2].conj() * yb[2],
            xb[0].conj() * yb[1] + xb[2].conj() * yb[3],
            xb[1].conj() * yb[0] + xb[3].conj() * yb[2],
            xb[1].conj() * yb[1] + xb[3].conj() * yb[3],
        ];
        let sum_reals: f64 = prod.iter().map(|z| z.re).sum();
        let tr = prod[0].re + prod[3].re;
        let got = mat_inner(&x, &y, &integ);
        assert!((got - sum_reals).abs() < 1e-12, "{got} vs {sum_reals}");
        assert!((got - tr).abs() > 1e-12);

        // Gregory end weights
        let e00 = [c(1.0, 0.0), Complex::ZERO, Complex::ZERO, Complex::ZERO];
        let e11 = [Complex::ZERO, Complex::ZERO, Complex::ZERO, c(1.0, 0.0)];
        x.block_mut(0).copy_from_slice(&e00);
        y.block_mut(0).copy_from_slice(&e00);
        x.block_mut(n - 1).copy_from_slice(&e11);
        y.block_mut(n - 1).copy_from_slice(&e11);
        let want = sum_reals + integ.giw.omega[0] + integ.giw.omega[1];
        let got = mat_inner(&x, &y, &integ);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
