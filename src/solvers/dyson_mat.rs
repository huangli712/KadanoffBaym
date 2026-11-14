// Project : KadanoffBaym
// File    : src/solvers/dyson_mat.rs
// Purpose : Matsubara Dyson equation: fixpoint, conjugate gradient, and
//           frequency-domain methods (blueprint: solvers/dyson_mat.jl)

use crate::complex::Complex;
use crate::convolution::mat::conv_mat;
use crate::linalg::eigh::eigh;
use crate::fourier::matsubara::{
    first_order_tail, matsubara_dft, matsubara_ft_direct, matsubara_seed,
};
use crate::linalg::lu::mat_inv;
use crate::linalg::mat::DenseMat;
use crate::basics::force_matsubara_hermitian;
use crate::types::sign::Sign;
use crate::types::green::{ContourGreen, TimeFunction};
use crate::types::mat::MatComp;
use crate::weights::Integrator;

/// Method for the Matsubara component of the Dyson equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatsubaraMethod {
    /// Preconditioned residual corrections (default, 6 iterations).
    Fixpoint,
    /// Fletcher-Reeves conjugate gradient.
    Cg,
    /// Frequency-domain solution.
    Fourier,
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

/// Free Matsubara Green's function of the hermitian matrix `H[MATSUBARA]`
/// (blueprint `_free_mat!`): `-R diag(e^{tau lambda} n_F(lambda)) R^dag`
/// for fermions and `+R diag(e^{tau lambda} n_B(lambda)) R^dag` for
/// bosons, where `mu*I - H = R diag(lambda) R^dag`.
pub fn free_mat(g0: &mut MatComp, mu: f64, h: &TimeFunction, beta: f64, sign: Sign) {
    let d = g0.size1;
    super::dyson::check_hamiltonian(h, std::iter::once(crate::basics::MATSUBARA), d);

    let mut k = DenseMat::zeros(d);
    for i in 0..d {
        k.set(i, i, Complex::new(mu, 0.0));
    }
    let h0 = h.block(crate::basics::MATSUBARA);
    for q in 0..d * d {
        k.raw_mut()[q] -= h0[q];
    }
    let (lambda, r) = eigh(&k);
    let dtau = beta / g0.ntau as f64;
    for m in 0..=(g0.ntau as i64) {
        let tau = m as f64 * dtau;
        let blk = if sign == Sign::Fermi {
            let d: Vec<f64> = lambda
                .iter()
                .map(|&w| fermi_of(beta, tau, w))
                .collect();
            r_diag_rdag(&r, &d, -1.0)
        } else {
            let d: Vec<f64> = lambda
                .iter()
                .map(|&w| bose_of(beta, tau, w))
                .collect();
            r_diag_rdag(&r, &d, 1.0)
        };
        g0.block_mut(m as usize).copy_from_slice(&blk);
    }
}

use crate::math::{bose_tau as bose_of, fermi_tau as fermi_of};

/// Inner product of two Matsubara functions with the Gregory end
/// weights (blueprint `_mat_inner`).
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
        // sum over ALL elements of the matrix product X^dagger Y
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

/// Return the free Matsubara Green's function `G0` and the kernel
/// `F = -G0 * Sigma` (blueprint `_dyson_mat_kernel`).
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

/// Fixpoint solver (blueprint `_dyson_mat_fixpoint!`).
pub fn dyson_mat_fixpoint(
    g: &mut ContourGreen,
    sigma: &ContourGreen,
    mu: f64,
    h: &TimeFunction,
    beta: f64,
    integ: &Integrator,
    maxiter: i64,
) {
    let sign = g.sign;
    let (g0, f) = dyson_mat_kernel(g, sigma, mu, h, beta, integ);
    matsubara_seed(&mut g.mat, &f, &g0, beta, sign, 5, 3);

    let mut rmat = MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO);
    let mut dg = MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO);
    for _ in 0..maxiter {
        // R = F * G + G - Q
        conv_mat(&mut rmat, &f, &g.mat, integ, beta, sign);
        for q in 0..rmat.raw().len() {
            rmat.raw_mut()[q] += g.mat.raw()[q] - g0.raw()[q];
        }
        matsubara_seed(&mut dg, &f, &rmat, beta, sign, 5, 3);
        for q in 0..g.mat.raw().len() {
            g.mat.raw_mut()[q] -= dg.raw()[q];
        }
    }
}

/// Fletcher-Reeves conjugate gradient solver (blueprint
/// `_dyson_mat_cg!`); returns the number of performed iterations.
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

    matsubara_seed(&mut g.mat, &f, &g0, beta, sign, 3, 3);
    if maxiter <= 0 {
        return 0;
    }

    let mut rmat = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut p = MatComp::new(g.ntau, d, d, Complex::ZERO);
    let mut ap = MatComp::new(g.ntau, d, d, Complex::ZERO);
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

    // initial residual R = Q - A(G)
    apply_a(&mut rmat, &g.mat);
    for q in 0..rmat.raw().len() {
        rmat.raw_mut()[q] = g0.raw()[q] - rmat.raw()[q];
    }
    p.raw_mut().copy_from_slice(rmat.raw());
    let mut rsold = mat_inner(&rmat, &rmat, integ);
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
            for q in 0..g.mat.raw().len() {
                g.mat.raw_mut()[q] += Complex::new(alpha, 0.0) * p.raw()[q];
                rmat.raw_mut()[q] -= Complex::new(alpha, 0.0) * ap.raw()[q];
            }
            let rsnew = mat_inner(&rmat, &rmat, integ);
            assert!(rsnew.is_finite(), "residual is not finite");
            for q in 0..p.raw().len() {
                p.raw_mut()[q] = rmat.raw()[q] + Complex::new(rsnew / rsold, 0.0) * p.raw()[q];
            }
            rsold = rsnew;
            if rsold.sqrt() < tol {
                break;
            }
        }
    }
    niter
}

/// Frequency-domain solver (blueprint `_dyson_mat_fourier!`).
pub fn dyson_mat_fourier(
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
    let h0 = h.block(crate::basics::MATSUBARA);
    for i in 0..d {
        hj.set(i, i, Complex::new(-mu, 0.0));
    }
    for q in 0..d * d {
        hj.raw_mut()[q] += h0[q];
    }
    let m2 = n / 2;
    let one: i64 = if sign == Sign::Fermi { 1 } else { 0 };

    // raw transform of Sigma and the free tail
    let sdft = matsubara_dft(&sigma.mat, sign);
    let mut gmat = vec![vec![Complex::ZERO; d * d]; (n + 1) as usize];
    let id_coeff = {
        let mut v = vec![Complex::ZERO; d * d];
        for i in 0..d {
            v[i * d + i] = Complex::ONE;
        }
        v
    };
    first_order_tail(&mut gmat, &id_coeff, beta, sign);

    for m in -m2..m2 {
        for p in -pcf..=pcf {
            let mp = m + p * n;
            let iom = Complex::new(0.0, std::f64::consts::PI / beta * (2 * mp + one) as f64);
            let sv = matsubara_ft_direct(mp, &sigma.mat, &sdft, sign, beta, order);
            let mut z1 = hj.clone();
            for q in 0..d * d {
                z1.raw_mut()[q] += sv[q];
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
            for (r, blk) in gmat.iter_mut().enumerate() {
                let w = (-Complex::new(r as f64 * (beta / n as f64), 0.0) * iom).exp();
                for q in 0..d * d {
                    blk[q] += z2[q] * w;
                }
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

/// Solve the Matsubara Dyson equation
/// `[-d_tau + mu - H] G^M - Sigma^M * G^M = delta(tau)`; only the
/// Matsubara component of `g` is modified (blueprint `dyson_mat!`).
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
    super::dyson::check_dyson(g, sigma, h, solve_order);
    super::dyson::check_hamiltonian(h, std::iter::once(crate::basics::MATSUBARA), g.size1);
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
        force_matsubara_hermitian(g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::green::ContourGreen;

    fn setup(ntau: usize) -> (Contour, ContourGreen, ContourGreen, TimeFunction) {
        let c = Contour::new(4, ntau as i64, 2, 2, 1.0, 4.0);
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let sigma = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // H = [[-0.4, 0.1i], [-0.1i, 0.6]]
        let mut h = TimeFunction::new(4, 2, 2, Complex::ZERO);
        h.set(crate::basics::MATSUBARA, 0, 0, Complex::new(-0.4, 0.0));
        h.set(crate::basics::MATSUBARA, 0, 1, Complex::new(0.0, 0.1));
        h.set(crate::basics::MATSUBARA, 1, 0, Complex::new(0.0, -0.1));
        h.set(crate::basics::MATSUBARA, 1, 1, Complex::new(0.6, 0.0));
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
            // truncation (~1.16e-4, the same deviation as the blueprint
            // itself)
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

