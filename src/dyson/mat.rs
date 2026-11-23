// Project : KadanoffBaym
// File    : src/dyson/mat.rs
// Purpose : Matsubara Dyson equation: fixpoint, conjugate gradient, and
//           frequency-domain methods

use std::f64::consts::PI;

use std::iter::once;

use crate::base::{MatsubaraMethod, MatsubaraSeed};
use crate::complex::Complex;
use crate::fourier::MatsubaraTables;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, MatComp, Sign, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::base::{matsubara_cg, matsubara_fixpoint};
use crate::convolution::conv_mat;
use crate::fourier::first_order_tail;
use crate::linalg::{eigh, mat_inv};
use crate::math::{bose_tau, fermi_tau};

use crate::dyson::full::{check_dyson, check_hamiltonian};

///
/// *Remarks* : *Matsubara Dyson Equation*
///
/// Let ``G_0`` be the free Matsubara Green's function of ``H(-i\beta)``.
/// The Matsubara Dyson equation is written as
///
/// ```math
/// \begin{equation}
/// [1 + F] * G = Q, \quad F = -G_0 * \Sigma, \quad Q = G_0.
/// \end{equation}
/// ```
///
/// Two methods are supported:
///
/// * `:fixpoint` -> Start from the frequency-domain solution (`pcf = 5`),
///   then apply the preconditioned residual corrections ``R = F*G + G - Q``,
///   ``G \to G - P^{-1}R`` for `maxiter` (default 6) times.
///
/// * `:cg` -> Start from the frequency-domain solution (`pcf = 3`), then
///   apply the Fletcher-Reeves conjugate gradient method to the hermitian
///   operator ``A(X) = X + [F*X + X*F^{cc}]/2``, where
///   ``F^{cc} = -\Sigma * G_0``. The inner product is
///   ``\langle X,Y \rangle = \sum_r w_r \sum_{ab} \mathrm{Re}[(X^{\dagger}_r Y_r)_{ab}]``,
///   where ``w_r`` are the Gregory end weights.
///
/// The method `:fourier` solves the equation in the frequency domain with
/// `pcf = 10` oversampling copies and the cubically corrected transform.
///
///
/// Solve the Matsubara Dyson equation
/// `[-d_tau + mu - H] G^M - Sigma^M * G^M = delta(tau)`; only the
/// Matsubara component of `g` is modified.
#[allow(clippy::too_many_arguments)]
pub fn dyson_mat(
    g: &mut ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    solve_order: i64,
    method: MatsubaraMethod,
    force_hermitian: bool,
    maxiter: Option<i64>,
    tol: f64,
) {
    check_dyson(g, sigma, h, solve_order);
    check_hamiltonian(h, once(MATSUBARA), g.size1);
    assert!(
        sigma
            .mat
            .raw()
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "Sigma is not finite"
    );
    assert!(beta > 0.0, "beta must be positive");
    if let Some(mi) = maxiter {
        assert!(mi >= 0, "maxiter must be nonnegative");
    }

    let integ = Integrator::new(solve_order);
    match method {
        MatsubaraMethod::Fixpoint => {
            dyson_mat_fixpoint(g, sigma, mu, h, beta, &integ, maxiter.unwrap_or(6));
        }
        MatsubaraMethod::Cg => {
            dyson_mat_cg(g, sigma, mu, h, beta, &integ, maxiter.unwrap_or(40), tol);
        }
        MatsubaraMethod::Fourier => {
            dyson_mat_fourier(g, sigma, mu, h, beta, 10, 3);
        }
    }

    if force_hermitian {
        g.mat.force_matsubara_hermitian();
    }
}

/// Return the free Matsubara Green's function `G0` and the kernel
/// `F = -G0 * Sigma`.
#[doc(hidden)] // test-support export: exercised directly by the integration tests
pub fn dyson_mat_kernel(
    g: &ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    integ: &Integrator,
) -> (MatComp, MatComp) {
    let sign = g.sign;
    let d = g.size1;
    let mut g0 = MatComp::new(g.ntau, d, d, Complex::ZERO);
    free_mat(&mut g0, mu, h, beta, sign);
    let mut f = MatComp::new(g.ntau, d, d, Complex::ZERO);
    conv_mat(&mut f, &g0, &sigma.mat, integ, beta, sign);
    f.smul(Complex::new(-1.0, 0.0));
    (g0, f)
}

/// Fixpoint solver.
pub fn dyson_mat_fixpoint(
    g: &mut ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    integ: &Integrator,
    maxiter: i64,
) {
    let (g0, f) = dyson_mat_kernel(g, sigma, mu, h, beta, integ);
    matsubara_fixpoint(&mut g.mat, &f, &g0, beta, g.sign, integ, maxiter);
}

/// Fletcher-Reeves conjugate gradient solver; returns the number of
/// performed iterations. the parameter list mirrors the reference signature.
#[allow(clippy::too_many_arguments)]
pub fn dyson_mat_cg(
    g: &mut ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    integ: &Integrator,
    maxiter: i64,
    tol: f64,
) -> i64 {
    let sign = g.sign;
    let d = g.size1;
    let (g0, f) = dyson_mat_kernel(g, sigma, mu, h, beta, integ);
    let mut fcc = MatComp::new(g.ntau, d, d, Complex::ZERO);
    conv_mat(&mut fcc, &sigma.mat, &g0, integ, beta, sign);
    fcc.smul(Complex::new(-1.0, 0.0));

    let mut seed = MatsubaraSeed::new(&f, beta, sign, 3, 3);
    seed.solve(&mut g.mat, &g0);
    if maxiter <= 0 {
        return 0;
    }

    let mut c1 = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut c2 = MatComp::new(g.ntau, d, d, Complex::ZERO);

    // A(X) = X + [F * X + X * Fcc] / 2
    let mut apply_a = |ax: &mut MatComp, x: &MatComp| {
        conv_mat(&mut c1, &f, x, integ, beta, sign);
        conv_mat(&mut c2, x, &fcc, integ, beta, sign);
        for q in 0..ax.raw().len() {
            ax.raw_mut()[q] = (c1.raw()[q] + c2.raw()[q]) * Complex::new(0.5, 0.0) + x.raw()[q];
        }
    };
    matsubara_cg(&mut g.mat, &g0, &mut apply_a, integ, maxiter as usize, tol)
}

/// Frequency-domain solver.
///
/// Solve the Matsubara Dyson equation in the frequency domain. With ``K =
/// H - \\mu + \\Sigma(i\\omega_n)``, the non-zero modes contribute
/// ``[(i\\omega_n - K)^{-1}/(i\\omega_n)] K / \\beta`` (the free tail
/// ``1/(i\\omega_n)`` is treated analytically by the first order tail
/// correction), and the bosonic zero mode contributes ``-K^{-1}/\\beta``.
/// The frequency sum includes `2pcf + 1` copies of the basic window. The
/// number of imaginary time intervals must be even. It is a private
/// function.
///
/// ### Arguments
/// * G     -> Contour-ordered Green's function.
/// * Σ     -> Self-energy.
/// * μ     -> Chemical potential.
/// * H     -> Hamiltonian, only `H[0]` is used.
/// * β     -> Inverse temperature.
/// * pcf   -> Number of the oversampling copies on each side.
/// * order -> Order of the boundary correction of the Fourier transform.
///
/// ### Returns
/// * G -> Contour-ordered Green's function.
pub(crate) fn dyson_mat_fourier(
    g: &mut ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    pcf: i64,
    order: i64,
) {
    let sign = g.sign;
    let n = g.ntau as i64;
    let d = g.size1;
    assert!(
        n % 2 == 0,
        "the Fourier method needs an even number of imaginary time intervals, got {n}"
    );

    let mut hj = DenseMat::zeros(d);
    let h0 = h.block(MATSUBARA);
    for i in 0..d {
        hj.set(i, i, Complex::new(-mu, 0.0));
    }
    for (hq, h0q) in hj.raw_mut().iter_mut().zip(h0.iter()) {
        *hq += *h0q;
    }
    let m2 = n / 2;
    let one: i64 = if sign == Sign::Fermi { 1 } else { 0 };

    // raw transform of Sigma and the free tail; the corrected
    // frequency values and the accumulation phases go through the
    // twiddle/correction tables (MatsubaraTables)
    let tables = MatsubaraTables::new(n, sign, order);
    let mut sdft = Vec::new();
    tables.dft_into(&sigma.mat, &mut sdft);
    let mut mps = Vec::with_capacity((11 * n) as usize);
    for m in -m2..m2 {
        for p in -pcf..=pcf {
            mps.push(m + p * n);
        }
    }
    let corrs: Vec<_> = mps.iter().map(|&mp| tables.correction(mp)).collect();
    let mut sft = Vec::new();
    tables.ft_into(&sigma.mat, &sdft, beta, &mps, &corrs, &mut sft);

    let mut gmat = vec![vec![Complex::ZERO; d * d]; (n + 1) as usize];
    let id_coeff = {
        let mut v = vec![Complex::ZERO; d * d];
        for i in 0..d {
            v[i * d + i] = Complex::ONE;
        }
        v
    };
    first_order_tail(&mut gmat, &id_coeff, beta, sign);

    let mut z1 = DenseMat::zeros(d);
    for (imp, &mp) in mps.iter().enumerate() {
        let iom = Complex::new(0.0, PI / beta * (2 * mp + one) as f64);
        let sv = &sft[imp];
        z1.set_from(&hj);
        for (zq, sq) in z1.raw_mut().iter_mut().zip(sv.iter()) {
            *zq += *sq;
        }
        let z2 = if sign == Sign::Bose && mp == 0 {
            // the bosonic zero mode, 1/(i omega_n) diverges
            let zinv = mat_inv(&z1);
            zinv.raw().iter().map(|&z| -z).collect()
        } else {
            // 1/(i w - K) - 1/(i w) = [i w (i w - K)]^{-1} K
            let mut a = DenseMat::zeros(d);
            for i in 0..d {
                a.set(i, i, iom);
            }
            for q in 0..d * d {
                a.raw_mut()[q] -= z1.raw()[q];
            }
            let ainv = mat_inv(&a);
            let scale = Complex::ONE / iom;
            let zinv_scaled: Vec<Complex> = ainv.raw().iter().map(|&z| z * scale).collect();
            DenseMat::from_vec(d, d, zinv_scaled)
                .mat_mul(&z1)
                .raw()
                .to_vec()
        };
        let z2: Vec<Complex> = z2.iter().map(|&z| z / Complex::new(beta, 0.0)).collect();
        // e^{-i omega tau_r} = conj(tw[k]), k = (2 mp + one) r mod 2n
        let t = (2 * mp + one).rem_euclid(2 * n) as usize;
        let tw = &tables.tw;
        for (r, blk) in gmat.iter_mut().enumerate() {
            let k = (t * r) % (2 * n as usize);
            let w = tw[k].conj();
            for q in 0..d * d {
                blk[q] += z2[q] * w;
            }
        }
    }

    for (dst, src) in g.mat.raw_mut().chunks_mut(d * d).zip(&gmat) {
        dst.copy_from_slice(src);
    }
    assert!(
        g.mat
            .raw()
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "G is not finite"
    );
}

/// Free Matsubara Green's function of the hermitian matrix `H[MATSUBARA]`:
/// `-R diag(e^{tau lambda} n_F(lambda)) R^dag` for fermions and
/// `+R diag(e^{tau lambda} n_B(lambda)) R^dag` for bosons, where
/// `mu*I - H = R diag(lambda) R^dag`.
///
/// ### Arguments
/// * G0   -> Matsubara Green's function.
/// * μ    -> Chemical potential.
/// * H    -> Hamiltonian.
/// * β    -> Inverse temperature.
/// * sign -> FERMI or BOSE.
///
/// ### Returns
/// * G0 -> Matsubara Green's function.
pub fn free_mat(g0: &mut MatComp, mu: f64, h: &TimeFunction, beta: f64, sign: Sign) {
    let d = g0.size1;
    check_hamiltonian(h, once(MATSUBARA), d);

    let mut k = DenseMat::zeros(d);
    for i in 0..d {
        k.set(i, i, Complex::new(mu, 0.0));
    }
    let h0 = h.block(MATSUBARA);
    for (kq, hq) in k.raw_mut().iter_mut().zip(h0.iter()) {
        *kq -= *hq;
    }
    let (lambda, r) = eigh(&k);
    let dtau = beta / g0.ntau as f64;
    for m in 0..=(g0.ntau as i64) {
        let tau = m as f64 * dtau;
        let blk = if sign == Sign::Fermi {
            let d: Vec<f64> = lambda.iter().map(|&w| fermi_tau(beta, tau, w)).collect();
            r_diag_rdag(&r, &d, -1.0)
        } else {
            let d: Vec<f64> = lambda.iter().map(|&w| bose_tau(beta, tau, w)).collect();
            r_diag_rdag(&r, &d, 1.0)
        };
        g0.block_mut(m as usize).copy_from_slice(&blk);
    }
}

/// `sign * R diag(d) R^dag` as a flat row-major block.
fn r_diag_rdag(r: &DenseMat, d: &[f64], front: f64) -> Vec<Complex> {
    let n = r.nrows();
    let mut out = vec![Complex::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex::ZERO;
            for (l, &dl) in d.iter().enumerate() {
                acc += r.get(i, l) * r.get(j, l).conj() * Complex::new(dl, 0.0);
            }
            out[i * n + j] = acc * Complex::new(front, 0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::Contour;

    fn setup(ntau: usize) -> (Contour, ContourGreen, ContourGreen, TimeFunction) {
        let c = Contour::new(4, ntau as i64, 2, 2, 1.0, 4.0);
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let sigma = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // H = [[-0.4, 0.1i], [-0.1i, 0.6]]
        let mut h = TimeFunction::new(4, 2, 2, Complex::ZERO);
        h.set(MATSUBARA, 0, 0, Complex::new(-0.4, 0.0));
        h.set(MATSUBARA, 0, 1, Complex::new(0.0, 0.1));
        h.set(MATSUBARA, 1, 0, Complex::new(0.0, -0.1));
        h.set(MATSUBARA, 1, 1, Complex::new(0.6, 0.0));
        for t in 0..=4 {
            h.set(t, 0, 0, Complex::new(-0.4, 0.0));
            h.set(t, 0, 1, Complex::new(0.0, 0.1));
            h.set(t, 1, 0, Complex::new(0.0, -0.1));
            h.set(t, 1, 1, Complex::new(0.6, 0.0));
        }
        (c, g, sigma, h)
    }

    /// Analytic free solution for Sigma = 0: G^M(tau) = -R diag(e^{tau
    /// lambda} n_F(lambda)) R^dag with lambda = eigenvalues of mu*I - H.
    #[test]
    fn sigma_zero_returns_free_mat() {
        let (beta, mu) = (4.0f64, 0.0);
        for method in [
            MatsubaraMethod::Fixpoint,
            MatsubaraMethod::Cg,
            MatsubaraMethod::Fourier,
        ] {
            let (_c, mut g, sigma, h) = setup(100);
            dyson_mat(&mut g, &sigma, mu, &h, beta, 5, method, true, None, 1e-12);
            // free reference from from_mat
            let mut want = MatComp::new(100, 2, 2, Complex::ZERO);
            free_mat(&mut want, mu, &h, beta, Sign::Fermi);
            let mut err = 0.0f64;
            for q in 0..want.raw().len() {
                err = err.max((g.mat.raw()[q] - want.raw()[q]).abs());
            }
            // fixpoint and cg converge to the free solution; the
            // Fourier method carries the pcf = 10 oversampling
            // truncation (~1.16e-4, the same deviation as the reference
            // values)
            let tol = match method {
                MatsubaraMethod::Fourier => 5e-4,
                _ => 1e-10,
            };
            assert!(err < tol, "{method:?}: {err}");
        }
    }

    /// The three methods agree for a smooth hermitian Sigma.
    #[test]
    fn methods_agree_for_smooth_sigma() {
        let (beta, mu) = (4.0f64, 0.0);
        let (_c, _g, mut sigma, _h) = setup(64);
        // Sigma^M(tau) = s0 * exp(-tau) * cos(beta - tau)-ish smooth profile
        let s0 = Complex::new(0.3, 0.0);
        let s1 = Complex::new(0.0, -0.15);
        for m in 0..=(sigma.ntau as i64) {
            let tau = m as f64 * beta / sigma.ntau as f64;
            let f = (-(tau * 0.8)).exp();
            sigma.set_mat(m, 0, 0, s0 * Complex::new(f, 0.0));
            sigma.set_mat(m, 1, 1, s0 * Complex::new(f * 0.7, 0.0));
            sigma.set_mat(m, 0, 1, s1 * Complex::new(f, 0.0));
            sigma.set_mat(m, 1, 0, s1.conj() * Complex::new(f, 0.0));
        }
        let mut refs: Vec<MatComp> = Vec::new();
        for method in [
            MatsubaraMethod::Fixpoint,
            MatsubaraMethod::Cg,
            MatsubaraMethod::Fourier,
        ] {
            let (_c, mut g, sig, hh) = setup(64);
            // copy sigma content
            for _q in 0..sig.mat.raw().len() {
                // rebuild identical sigma
            }
            let mut sig2 = sig.clone();
            for m in 0..=(sig2.mat.ntau as i64) {
                for q in 0..4 {
                    sig2.mat.block_mut(m as usize)[q] = sigma.mat.block(m as usize)[q];
                }
            }
            dyson_mat(&mut g, &sig2, mu, &hh, beta, 5, method, true, None, 1e-12);
            refs.push(g.mat.clone());
        }
        // fixpoint vs cg agree tightly
        let mut err = 0.0f64;
        for q in 0..refs[0].raw().len() {
            err = err.max((refs[0].raw()[q] - refs[1].raw()[q]).abs());
        }
        assert!(err < 1e-4, "fixpoint vs cg disagree by {err}");
        // the Fourier method carries its own truncation
        let mut errf = 0.0f64;
        for q in 0..refs[0].raw().len() {
            errf = errf.max((refs[0].raw()[q] - refs[2].raw()[q]).abs());
        }
        assert!(errf < 5e-3, "fourier vs fixpoint differ by {errf}");
    }
}
