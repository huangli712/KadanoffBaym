// Project : KadanoffBaym
// File    : src/fourier/matsubara.rs
// Purpose : Matsubara Fourier transforms and the frequency-domain seed
//           (blueprint: fourier/matsubara.jl)

use crate::complex::Complex;
use crate::linalg::lu::{lu_factor, lu_solve_mat};
use crate::linalg::mat::DenseMat;
use crate::types::sign::Sign;
use crate::types::mat::MatComp;

#[inline]
fn cis(x: f64) -> Complex {
    Complex::from_polar(1.0, x)
}

#[inline]
fn sgn(sign: Sign) -> f64 {
    sign.as_i64() as f64
}

/// Imaginary time `tau_r = r beta / N` with `N = ntau_points - 1`
/// intervals (blueprint `get_tau`).
pub fn get_tau(r: i64, beta: f64, ntau_points: i64) -> f64 {
    assert!(beta > 0.0, "beta must be positive");
    assert!(
        ntau_points >= 2,
        "at least 2 imaginary time points are required"
    );
    r as f64 * (beta / (ntau_points - 1) as f64)
}

/// Matsubara frequency `omega_m = pi (2 m + delta) / beta` (blueprint
/// `get_omega`); negative indices are allowed.
pub fn get_omega(m: i64, beta: f64, sign: Sign) -> f64 {
    assert!(beta > 0.0, "beta must be positive");
    let one = if sign == Sign::Fermi { 1 } else { 0 };
    std::f64::consts::PI / beta * (2 * m + one) as f64
}

/// First order tail correction `c_0(tau_r)` written into the `N + 1`
/// blocks (blueprint `_first_order_tail!`): `-a_0 / 2` for fermions,
/// `a_0 (tau/beta - 1/2)` for bosons.
pub fn first_order_tail(blocks: &mut [Vec<Complex>], coeff: &[Complex], beta: f64, sign: Sign) {
    assert!(beta > 0.0, "beta must be positive");
    if let Some(first) = blocks.first() {
        assert_eq!(coeff.len(), first.len(), "coeff must match the block shape");
    }
    let n = blocks.len() - 1;
    let dtau = beta / n as f64;
    for (r, blk) in blocks.iter_mut().enumerate() {
        let tau = r as f64 * dtau;
        let scale = if sign == Sign::Bose {
            Complex::new(tau / beta - 0.5, 0.0)
        } else {
            Complex::new(-0.5, 0.0)
        };
        for (z, c) in blk.iter_mut().zip(coeff) {
            *z = *c * scale;
        }
    }
}

/// Raw discrete Fourier transform (without the factor `delta_tau`) of a
/// Matsubara function, `sum_r e^{i pi (2m + delta) r / N} A(tau_r)` for
/// `m = 0, ..., N` (blueprint `_matsubara_dft`). The result has `N + 1`
/// blocks.
pub fn matsubara_dft(a: &MatComp, sign: Sign) -> Vec<Vec<Complex>> {
    let n = a.ntau as i64;
    let one = if sign == Sign::Fermi { 1 } else { 0 };
    let nel = a.size1 * a.size2;
    let mut mdft = vec![vec![Complex::ZERO; nel]; (n + 1) as usize];
    for (m, out) in mdft.iter_mut().enumerate() {
        for r in 0..=n {
            let arg = ((2 * m as i64 + one) * r) as f64 * std::f64::consts::PI / n as f64;
            let w = cis(arg);
            let blk = a.block(r as usize);
            for (o, &v) in out.iter_mut().zip(blk) {
                *o += w * v;
            }
        }
    }
    mdft
}

/// Corrected Fourier transform of a Matsubara function at the
/// Matsubara frequency with index `m` (any integer), an approximation
/// to `int_0^beta dtau e^{i omega_m tau} A(tau)` (blueprint
/// `_matsubara_ft_direct`). The raw transform `mdft` is periodically
/// mapped. `order` is the order of the boundary correction (0, 1, 3).
pub fn matsubara_ft_direct(
    m: i64,
    a: &MatComp,
    mdft: &[Vec<Complex>],
    sign: Sign,
    beta: f64,
    order: i64,
) -> Vec<Complex> {
    let n = a.ntau as i64;
    let dtau = beta / n as f64;
    let one = if sign == Sign::Fermi { 1 } else { 0 };

    let arg = (2 * m + one) as f64 * std::f64::consts::PI / n as f64;
    let (corfac, bcorr): (f64, Vec<Complex>) = match order {
        0 => (1.0, Vec::new()),
        1 => {
            let (c, b) = crate::fourier::dftcor::dftcorr_linear(arg);
            (c, b.to_vec())
        }
        3 => {
            let (c, b) = crate::fourier::dftcor::dftcorr_cubic(arg);
            (c, b.to_vec())
        }
        _ => panic!("_matsubara_ft_direct: unsupported order {order}"),
    };

    // shift m into the interval [0, N-1]
    let m1 = m.rem_euclid(n);
    let mut res = mdft[m1 as usize]
        .iter()
        .map(|&z| z * Complex::new(corfac, 0.0))
        .collect::<Vec<_>>();
    for (r, b) in bcorr.iter().enumerate() {
        if r as i64 > order.min(bcorr.len() as i64 - 1) {
            break;
        }
        let left = a.block(r);
        let right = a.block((n as usize) - r);
        let sb = Complex::new(sgn(sign), 0.0) * b.conj();
        for q in 0..res.len() {
            res[q] += *b * left[q] + sb * right[q];
        }
    }

    res.iter().map(|&z| z * Complex::new(dtau, 0.0)).collect()
}

/// Corrected Fourier transform `int_0^beta dtau e^{i omega_m tau} G(tau)`
/// for any integer `m` (blueprint `matsubara_ft!`). `raw_dft` is the
/// result of [`matsubara_dft`].
pub fn matsubara_ft(
    m: i64,
    g: &MatComp,
    raw_dft: &[Vec<Complex>],
    sign: Sign,
    beta: f64,
    order: i64,
) -> Vec<Complex> {
    assert!(
        order == 0 || order == 1 || order == 3,
        "order must be 0, 1, or 3, got {order}"
    );
    assert!(beta > 0.0, "beta must be positive");
    let n = g.ntau as i64;
    assert!(
        order == 0 || n >= 2 * order + 1,
        "too few imaginary time points for order {order}"
    );
    assert!(
        raw_dft.len() == (n + 1) as usize,
        "raw_dft must have {} blocks",
        n + 1
    );
    matsubara_ft_direct(m, g, raw_dft, sign, beta, order)
}

/// Solve `[1 + F] X = B` for Matsubara functions in the frequency
/// domain (blueprint `_matsubara_seed!`). The number of imaginary time
/// intervals must be even and at least 4; `pcf` is the number of the
/// oversampling copies on each side.
pub fn matsubara_seed(
    x: &mut MatComp,
    f: &MatComp,
    b: &MatComp,
    beta: f64,
    sign: Sign,
    pcf: i64,
    order: i64,
) {
    assert!(x.is_compatible(f), "X and F are not compatible");
    assert!(x.is_compatible(b), "X and B are not compatible");
    let d = x.size1;
    assert_eq!(d, x.size2, "square matrices are expected");
    let n = x.ntau as i64;
    assert!(
        n % 2 == 0 && n >= 4,
        "the number of imaginary time intervals must be even and >= 4, got {n}"
    );
    assert!(pcf >= 0, "pcf must be nonnegative");
    assert!(beta > 0.0, "beta must be positive");

    let dtau = beta / n as f64;
    let m2 = n / 2;
    let one = if sign == Sign::Fermi { 1i64 } else { 0 };
    let nel = d * d;

    // first order tail correction, q_inf = xi B(beta) - B(0)
    let qmasy: Vec<Complex> = b
        .block(n as usize)
        .iter()
        .zip(b.block(0))
        .map(|(&bb, &b0)| Complex::new(sgn(sign), 0.0) * bb - b0)
        .collect();

    let mut xmat = vec![vec![Complex::ZERO; nel]; (n + 1) as usize];
    first_order_tail(&mut xmat, &qmasy, beta, sign);

    let fmdft = matsubara_dft(f, sign);
    let qmdft = matsubara_dft(b, sign);

    let mut fiwo = DenseMat::zeros(d);
    let mut qiwo = DenseMat::zeros(d);
    for p in -pcf..=pcf {
        // for bosons the middle interval includes the zero frequency
        let (mut m_low, mut m_high) = (0i64, 0i64);
        if sign == Sign::Bose {
            m_high = if p < 0 { 0 } else { 1 };
            m_low = if p > 0 { -1 } else { 0 };
        }
        for m in (-m2 - m_low)..=(m2 - 1 + m_high) {
            let mp = m + p * n;
            let omega = (2 * mp + one) as f64 * std::f64::consts::PI / beta;
            let fv = matsubara_ft_direct(mp, f, &fmdft, sign, beta, order);
            let qv = matsubara_ft_direct(mp, b, &qmdft, sign, beta, order);
            for q in 0..nel {
                fiwo.raw_mut()[q] = fv[q];
                qiwo.raw_mut()[q] = qv[q];
            }

            // tail term in frequency space, q_inf / (i omega)
            let z3: Vec<Complex> = if sign == Sign::Bose && mp == 0 {
                vec![Complex::ZERO; nel]
            } else {
                qmasy
                    .iter()
                    .map(|&q| Complex::new(0.0, -1.0 / omega) * q)
                    .collect()
            };

            // [1 + F]^{-1} [B - (1 + F) q_inf / (i omega)] / beta
            let mut z1 = DenseMat::identity(d);
            for q in 0..nel {
                z1.raw_mut()[q] += fiwo.raw()[q];
            }
            let fz3 = fiwo.mat_mul(&DenseMat::from_vec(d, d, z3.clone()));
            let mut z2 = qiwo.raw().to_vec();
            for q in 0..nel {
                z2[q] -= z3[q] + fz3.raw()[q];
            }
            let mut z1f = z1.clone();
            let piv = lu_factor(&mut z1f);
            let mut sol = lu_solve_mat(&z1f, &piv, &DenseMat::from_vec(d, d, z2));
            for q in sol.raw_mut().iter_mut() {
                *q = *q / Complex::new(beta, 0.0);
            }

            for (r, blk) in xmat.iter_mut().enumerate() {
                let w = cis(-omega * r as f64 * dtau);
                for (z, &sv) in blk.iter_mut().zip(sol.raw()) {
                    *z += w * sv;
                }
            }
        }
    }

    for (dst, src) in x.raw_mut().chunks_mut(nel).zip(&xmat) {
        dst.copy_from_slice(src);
    }
    assert!(
        x.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "Matsubara seed is not finite"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::ContourGreen;

    /// Free fermionic Matsubara function of a single level with the
    /// energy `eps`: `G(tau) = -e^{-eps tau} n_F`, whose Fourier
    /// coefficients are `G(i omega_m) = 1 / (i omega_m - eps)`.
    fn free_mat(n: i64, beta: f64, eps: f64, d: usize) -> MatComp {
        let c = Contour::new(1, n, d as i64, d as i64, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let nf = 1.0 / (1.0 + (-(beta * eps)).exp()); // 1/(1+e^{-beta eps})
        for r in 0..=n {
            let v = -Complex::new((-(eps * r as f64 * beta / n as f64)).exp() * nf, 0.0);
            g.set_mat(r, 0, 0, v);
        }
        g.mat
    }

    #[test]
    fn tau_and_omega_helpers() {
        let tau = get_tau(3, 4.0, 9);
        assert!((tau - 4.0 * 3.0 / 8.0).abs() < 1e-15);
        let w = get_omega(2, 1.0, Sign::Fermi);
        assert!((w - 5.0 * std::f64::consts::PI).abs() < 1e-12);
        let w0 = get_omega(0, 1.0, Sign::Bose);
        assert!(w0.abs() < 1e-15);
    }

    #[test]
    fn corrected_ft_of_free_exponential() {
        // G(i omega_m) = 1 / (i omega_m - eps)
        let (beta, eps) = (10.0f64, 0.7);
        let n = 1000i64;
        let a = free_mat(n, beta, eps, 1);
        let mdft = matsubara_dft(&a, Sign::Fermi);
        for m in [-250i64, -3, 0, 7, 123, 500] {
            let got = matsubara_ft(m, &a, &mdft, Sign::Fermi, beta, 3);
            let om = get_omega(m, beta, Sign::Fermi);
            let want = Complex::ONE / Complex::new(-eps, om);
            let rel = (got[0] - want).abs() / want.abs();
            assert!(rel < 1e-7, "m {m}: {} vs {want} (rel {rel})", got[0]);
        }
    }

    #[test]
    fn raw_dft_matches_direct_sum() {
        let (beta, eps) = (4.0f64, 0.3);
        let n = 16i64;
        let a = free_mat(n, beta, eps, 1);
        let mdft = matsubara_dft(&a, Sign::Fermi);
        for m in 0..=n {
            let mut direct = Complex::ZERO;
            for r in 0..=n {
                let arg = (2 * m + 1) as f64 * r as f64 * std::f64::consts::PI / n as f64;
                direct += cis(arg) * a.get(r as usize, 0, 0);
            }
            assert!((mdft[m as usize][0] - direct).abs() < 1e-12, "m {m}");
        }
    }

    #[test]
    fn seed_identity_kernel_returns_source() {
        // F = 0: the seed must reproduce B
        let (beta, n, pcf) = (4.0f64, 16i64, 1);
        let c = Contour::new(1, n, 1, 1, 1.0, beta);
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for r in 0..=n {
            let tau = r as f64 * beta / n as f64;
            b.set_mat(
                r,
                0,
                0,
                Complex::new((-(0.5 * tau)).exp() * (0.3 * tau).cos(), 0.0) + Complex::new(0.0, 0.11 * tau),
            );
        }
        let f = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        matsubara_seed(&mut x.mat, &f.mat, &b.mat, beta, Sign::Fermi, pcf, 3);
        let mut err = 0.0f64;
        for r in 0..=n {
            err = err.max((x.get_mat(r, 0, 0) - b.get_mat(r, 0, 0)).abs());
        }
        // the frequency window (N = 16 frequencies) truncates the 1/w^2
        // tails of the non-periodic source; ~5e-3 is the expected floor
        assert!(err < 8e-3, "seed differs from the source by {err}");
    }
}
