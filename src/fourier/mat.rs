// Project : KadanoffBaym
// File    : src/fourier/mat.rs
// Purpose : Matsubara Fourier transforms and the correction tables
//

//!
//! *Remarks* : *Matsubara Seed*
//!
//! The frequency-domain seed solver lives in `base::seed`; this module
//! provides the Matsubara transforms and the correction tables it uses.
//!
//! ```math
//! \begin{equation}
//! X(\tau) + \int^{\beta}_0 d\tau'\ F(\tau - \tau') X(\tau') = B(\tau)
//! \end{equation}
//! ```
//!
//! in the Matsubara frequency domain. The first order tail
//! ``q_{\infty} = \xi B(\beta) - B(0)`` is treated analytically, i.e.,
//!
//! ```math
//! \begin{equation}
//! X(\tau) = c_0(\tau) + \frac{1}{\beta} \sum_n e^{-i\omega_n \tau}
//!     [1 + F(i\omega_n)]^{-1}
//!     \left\{
//!         B(i\omega_n) - [1 + F(i\omega_n)]\frac{q_{\infty}}{i\omega_n}
//!     \right\},
//! \end{equation}
//! ```
//!
//! where ``c_0(\tau) = -q_{\infty}/2`` for fermions, and ``c_0(\tau) =
//! q_{\infty}(\tau/\beta - 1/2)`` for bosons. The frequency sum includes
//! `2pcf + 1` copies of the basic frequency window (oversampling). For
//! the bosonic zero frequency the tail term is skipped.
//!
//! This is not a public Fourier solver. It is the preconditioner used by
//! the fixpoint and conjugate gradient Matsubara solvers.
//!

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::types::{MatComp, Sign};

use crate::fourier::{dft_coefficients_cubic, dft_coefficients_linear};

/// Twiddle and correction tables of the frequency-domain Matsubara
/// solvers (internal): `tw[k] = exp(i pi k / n)` for `k = 0, .., 2n-1`
/// covers every phase `e^{i pi (2m+delta) r / n}` of the raw DFT and
/// every accumulation phase `e^{-i omega_m tau_r}` by periodicity, and
/// `corr[m1]` holds the boundary correction of the frequency `m1`
/// (mathematically `2 pi`-periodic in `m`, so the `2pcf+1` oversampling
/// windows share the `n` basic entries).
pub(crate) struct MatsubaraTables {
    pub(crate) n: i64,
    sign: Sign,
    order: i64,
    pub(crate) tw: Vec<Complex>,
}

impl MatsubaraTables {
    pub(crate) fn new(n: i64, sign: Sign, order: i64) -> MatsubaraTables {
        let tw: Vec<Complex> = (0..2 * n)
            .map(|k| Complex::from_polar(1.0, k as f64 * PI / n as f64))
            .collect();
        MatsubaraTables { n, sign, order, tw }
    }

    /// The correction of the actual frequency `mp` (the boundary
    /// correction is NOT periodic in the oversampling windows, so the
    /// full index matters).
    pub(crate) fn corr_of_mp(&self, mp: i64) -> Corr {
        let one = if self.sign == Sign::Fermi { 1i64 } else { 0 };
        let arg = (2 * mp + one) as f64 * PI / self.n as f64;
        corr_of(self.order, arg)
    }

    /// Table-driven [`matsubara_dft`] into a reused buffer (the phases
    /// are reduced to the basic `2n`-periodic twiddle table).
    pub(crate) fn dft_into(&self, a: &MatComp, out: &mut Vec<Vec<Complex>>) {
        let n = self.n;
        let one = if self.sign == Sign::Fermi { 1i64 } else { 0 };
        let nel = a.size1 * a.size2;
        out.clear();
        out.resize((n + 1) as usize, vec![Complex::ZERO; nel]);
        let tw = &self.tw;
        for (m, o) in out.iter_mut().enumerate() {
            let tm = ((2 * m as i64 + one).rem_euclid(2 * n)) as usize;
            for r in 0..=n as usize {
                let k = (tm * r) % (2 * n as usize);
                let w = tw[k];
                let blk = a.block(r);
                for (q, &v) in o.iter_mut().zip(blk) {
                    *q += w * v;
                }
            }
        }
    }

    /// The corrected transforms of all frequencies `mps` into a reused
    /// buffer, aligned with `mps` (the corrections are given per actual
    /// frequency in `corrs`).
    pub(crate) fn ft_list(
        &self,
        a: &MatComp,
        mdft: &[Vec<Complex>],
        beta: f64,
        mps: &[i64],
        corrs: &[Corr],
        out: &mut Vec<Vec<Complex>>,
    ) {
        let n = self.n;
        let dtau = beta / n as f64;
        out.clear();
        out.resize_with(mps.len(), Vec::new);
        for (i, &mp) in mps.iter().enumerate() {
            let m1 = mp.rem_euclid(n) as usize;
            let mut row = std::mem::take(&mut out[i]);
            ft_core_into(m1, a, mdft, self.sign, dtau, &corrs[i], &mut row);
            out[i] = row;
        }
    }
}

/// Boundary-correction data of one frequency: the factor and the
/// correction terms (effective length `nterms`).
pub(crate) type Corr = (f64, [Complex; 4], usize);

/// Matsubara frequency `omega_m = pi (2 m + delta) / beta`; negative
/// indices are allowed.
#[allow(dead_code)]
pub fn get_omega(m: i64, beta: f64, sign: Sign) -> f64 {
    assert!(beta > 0.0, "beta must be positive");
    let one = if sign == Sign::Fermi { 1 } else { 0 };
    PI / beta * (2 * m + one) as f64
}

/// First order tail correction `c_0(tau_r)` written into the `N + 1`
/// blocks, such that `G(tau) + c_0(tau)` is continuous: `-a_0 / 2` for
/// fermions, `a_0 (tau/beta - 1/2)` for bosons. `coeff` is not changed.
///
/// ### Arguments
/// * blocks -> On return, the tail correction `c_0(tau_r)`.
/// * coeff  -> Jump coefficient `a_0`.
/// * beta   -> Inverse temperature.
/// * sign   -> FERMI or BOSE.
pub(crate) fn first_order_tail(
    blocks: &mut [Vec<Complex>],
    coeff: &[Complex],
    beta: f64,
    sign: Sign,
) {
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
/// Matsubara function, ``\\sum^{N}_{r=0} e^{i\\pi(2m+1)r/N} A(\\tau_r)`` for
/// fermions, or ``\\sum^{N}_{r=0} e^{2i\\pi m r/N} A(\\tau_r)`` for bosons,
/// where ``m = 0, \\cdots, N`` and ``N = ntau - 1`` is the number of
/// imaginary time intervals. The result has `N + 1` blocks.
///
/// ### Arguments
/// * a    -> Matsubara function.
/// * sign -> FERMI or BOSE.
///
/// ### Returns
/// * mdft -> mdft[:,:,m+1] is the transform for frequency index m.
pub fn matsubara_dft(a: &MatComp, sign: Sign) -> Vec<Vec<Complex>> {
    let n = a.ntau as i64;
    let one = if sign == Sign::Fermi { 1 } else { 0 };
    let nel = a.size1 * a.size2;
    let mut mdft = vec![vec![Complex::ZERO; nel]; (n + 1) as usize];
    for (m, out) in mdft.iter_mut().enumerate() {
        for r in 0..=n {
            let arg = ((2 * m as i64 + one) * r) as f64 * PI / n as f64;
            let w = Complex::from_polar(1.0, arg);
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
/// to `int_0^beta dtau e^{i omega_m tau} A(tau)`. The raw transform
/// `mdft` is periodically mapped. `order` is the order of the boundary
/// correction (0, 1, 3).
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

    let arg = (2 * m + one) as f64 * PI / n as f64;
    let (corfac, bcorr): (f64, Vec<Complex>) = match order {
        0 => (1.0, Vec::new()),
        1 => {
            let (c, b) = dft_coefficients_linear(arg);
            (c, b.to_vec())
        }
        3 => {
            let (c, b) = dft_coefficients_cubic(arg);
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
        let sb = Complex::new(sign.as_i64() as f64, 0.0) * b.conj();
        for q in 0..res.len() {
            res[q] += *b * left[q] + sb * right[q];
        }
    }

    res.iter().map(|&z| z * Complex::new(dtau, 0.0)).collect()
}

/// Corrected Fourier transform `int_0^beta dtau e^{i omega_m tau} G(tau)`
/// for any integer `m`. `raw_dft` is the result of [`matsubara_dft`].
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
        order == 0 || n > 2 * order,
        "too few imaginary time points for order {order}"
    );
    assert!(
        raw_dft.len() == (n + 1) as usize,
        "raw_dft must have {} blocks",
        n + 1
    );
    matsubara_ft_direct(m, g, raw_dft, sign, beta, order)
}

fn corr_of(order: i64, arg: f64) -> Corr {
    match order {
        0 => (1.0, [Complex::ZERO; 4], 0),
        1 => {
            let (c, b) = dft_coefficients_linear(arg);
            (c, [b[0], b[1], Complex::ZERO, Complex::ZERO], 2)
        }
        3 => {
            let (c, b) = dft_coefficients_cubic(arg);
            (c, b, 4)
        }
        _ => panic!("unsupported correction order {order}"),
    }
}

/// The corrected Fourier transform of one frequency index `m1` in the
/// basic window, evaluated from the raw transform `mdft` with a
/// precomputed correction (the shared core of [`matsubara_ft_direct`]
/// and the solver memo tables). `arg` is the reduced phase of the
/// frequency.
fn ft_core_into(
    m1: usize,
    a: &MatComp,
    mdft: &[Vec<Complex>],
    sign: Sign,
    dtau: f64,
    corr: &Corr,
    res: &mut Vec<Complex>,
) {
    let n = a.ntau as i64;
    let (corfac, bcorr, nterms) = (corr.0, &corr.1, corr.2);
    res.clear();
    res.extend(mdft[m1].iter().map(|&z| z * Complex::new(corfac, 0.0)));
    for (r, b) in bcorr.iter().enumerate() {
        if r >= nterms {
            break;
        }
        let left = a.block(r);
        let right = a.block((n as usize) - r);
        let sb = Complex::new(sign.as_i64() as f64, 0.0) * b.conj();
        for q in 0..res.len() {
            res[q] += *b * left[q] + sb * right[q];
        }
    }
    let dt = Complex::new(dtau, 0.0);
    for z in res.iter_mut() {
        *z *= dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_PI_2;

    use crate::linalg::DenseMat;
    use crate::types::{Contour, ContourGreen};

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

    /// U = [[1, i], [i, 1]] / sqrt(2)
    fn mf_u() -> DenseMat {
        let s = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        DenseMat::from_vec(
            2,
            2,
            vec![s, Complex::new(0.0, 1.0) * s, Complex::new(0.0, 1.0) * s, s],
        )
    }

    /// 2x2 non-diagonal matrix U diag(g(e1), g(e2)) U^dag .
    fn mf_matrix(eps: &[f64], n: i64, beta: f64, sign: Sign) -> MatComp {
        let g1 = mf_pole(eps[0], n, beta, sign);
        let g2 = mf_pole(eps[1], n, beta, sign);
        let u = mf_u();
        let mut g = MatComp::new(n as usize, 2, 2, Complex::ZERO);
        for r in 0..=(n as usize) {
            let mut diag = DenseMat::zeros(2);
            diag.set(0, 0, g1.block(r)[0]);
            diag.set(1, 1, g2.block(r)[0]);
            let m = u.mat_mul(&diag).mat_mul(&u.adjoint());
            let blk = g.block_mut(r);
            blk.copy_from_slice(&m.raw()[..blk.len()]);
        }
        g
    }

    fn mf_pole(eps: f64, n: i64, beta: f64, sign: Sign) -> MatComp {
        let mut g = MatComp::new(n as usize, 1, 1, Complex::ZERO);
        let z = if sign == Sign::Fermi {
            1.0 + (-beta * eps).exp()
        } else {
            1.0 - (-beta * eps).exp()
        };
        for r in 0..=n {
            g.block_mut(r as usize)[0] =
                Complex::new(-(-(eps * r as f64 * beta / n as f64)).exp() / z, 0.0);
        }
        g
    }

    fn mf_err(g: &MatComp, eps: f64, beta: f64, sign: Sign, order: i64) -> f64 {
        let raw = matsubara_dft(g, sign);
        let mut err = 0.0f64;
        for m in [-7i64, -1, 0, 3, 11] {
            let x = matsubara_ft(m, g, &raw, sign, beta, order);
            let om = get_omega(m, beta, sign);
            let ex = Complex::ONE / Complex::new(-eps, om);
            err = err.max((x[0] - ex).abs() / ex.abs());
        }
        err
    }

    #[test]
    fn omega_helper() {
        let w = get_omega(2, 1.0, Sign::Fermi);
        assert!((w - 5.0 * PI).abs() < 1e-12);
        let w0 = get_omega(0, 1.0, Sign::Bose);
        assert!(w0.abs() < 1e-15);
    }

    #[test]
    fn grids() {
        assert!((get_omega(0, 2.0, Sign::Fermi) - FRAC_PI_2).abs() < 1e-15);
        assert!(get_omega(0, 2.0, Sign::Bose).abs() < 1e-15);
        assert!((get_omega(-1, 2.0, Sign::Fermi) + FRAC_PI_2).abs() < 1e-15);
        assert!((get_omega(-3, 2.0, Sign::Bose) + 3.0 * PI).abs() < 1e-14);
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
    fn corrected_transform_poles_and_matrix() {
        let (beta, eps) = (2.0f64, 0.7);
        for sign in [Sign::Fermi, Sign::Bose] {
            // cubic order on a moderate grid, negative and high modes
            assert!(mf_err(&mf_pole(eps, 256, beta, sign), eps, beta, sign, 3) <= 1e-10);
            assert!(mf_err(&mf_pole(eps, 64, beta, sign), eps, beta, sign, 1) <= 1e-4);
            assert!(mf_err(&mf_pole(eps, 64, beta, sign), eps, beta, sign, 0) <= 1.0);
            // periodic mapping of the raw transform: m and m + N share the
            // raw value; with no correction the results are identical
            let n = 32i64;
            let g = mf_pole(eps, n, beta, sign);
            let d = matsubara_dft(&g, sign);
            let x0 = matsubara_ft(2, &g, &d, sign, beta, 0);
            let x1 = matsubara_ft(2 + n, &g, &d, sign, beta, 0);
            assert_eq!(x0, x1);
            // 2x2 non-diagonal matrix
            let n = 256i64;
            let g = mf_matrix(&[0.3, 1.1], n, beta, sign);
            let d = matsubara_dft(&g, sign);
            let u = mf_u();
            for m in [-5i64, 0, 4] {
                let x = matsubara_ft(m, &g, &d, sign, beta, 3);
                let iom = Complex::new(0.0, get_omega(m, beta, sign));
                let mut exd = DenseMat::zeros(2);
                exd.set(0, 0, Complex::ONE / (iom - Complex::new(0.3, 0.0)));
                exd.set(1, 1, Complex::ONE / (iom - Complex::new(1.1, 0.0)));
                let ex = u.mat_mul(&exd).mat_mul(&u.adjoint());
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for (xq, eq) in x.iter().zip(ex.raw().iter()) {
                    let dz = *xq - *eq;
                    num += dz.abs() * dz.abs();
                    den += eq.abs() * eq.abs();
                }
                assert!((num / den).sqrt() <= 1e-10, "m {m}");
            }
        }
    }

    #[test]
    fn tables_match_one_shot_transforms() {
        let (beta, n) = (4.0f64, 16i64);
        let c = Contour::new(1, n, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for r in 0..=n {
            let tau = r as f64 * beta / n as f64;
            g.set_mat(
                r,
                0,
                0,
                Complex::new((-(0.5 * tau)).exp(), 0.0) + Complex::new(0.0, 0.11 * tau),
            );
        }
        let tables = MatsubaraTables::new(n, Sign::Fermi, 3);
        let mut tdft = Vec::new();
        tables.dft_into(&g.mat, &mut tdft);
        let odft = matsubara_dft(&g.mat, Sign::Fermi);
        for m in 0..=n as usize {
            let d = (tdft[m][0] - odft[m][0]).abs();
            assert!(d < 1e-12, "dft m={m}: {d}");
        }
        let mps: Vec<i64> = (-20..20).collect();
        let corrs: Vec<_> = mps.iter().map(|&m| tables.corr_of_mp(m)).collect();
        let mut memo = Vec::new();
        tables.ft_list(&g.mat, &odft, beta, &mps, &corrs, &mut memo);
        for (i, &m) in mps.iter().enumerate() {
            let direct = matsubara_ft_direct(m, &g.mat, &odft, Sign::Fermi, beta, 3);
            let d = (memo[i][0] - direct[0]).abs();
            assert!(d < 1e-12, "ft m={m}: {} vs {} ({d})", memo[i][0], direct[0]);
        }
    }

    #[test]
    fn first_order_tail_fermi_and_bose() {
        let n = 16usize;
        let (beta, eps) = (2.0f64, 0.7);
        for sign in [Sign::Fermi, Sign::Bose] {
            let c = vec![Complex::new(2.0, 0.0)];
            let mut d = vec![vec![Complex::ZERO; 1]; n + 1];
            first_order_tail(&mut d, &c, 4.0, sign);
            assert_eq!(c, vec![Complex::new(2.0, 0.0)], "coeff changed");
            if sign == Sign::Fermi {
                for blk in &d {
                    assert_eq!(blk[0], Complex::new(-1.0, 0.0));
                }
            } else {
                assert_eq!(d[0][0], Complex::new(-1.0, 0.0));
                assert_eq!(d[n][0], Complex::new(1.0, 0.0));
                assert!(d[n / 2][0].abs() < 1e-15);
            }
            // the tail removes the jump: G - c0 is (anti)periodic
            let g = mf_pole(eps, n as i64, beta, sign);
            let sg = Complex::new(sign.as_i64() as f64, 0.0);
            let a0 = sg * g.block(n)[0] - g.block(0)[0];
            let mut t = vec![vec![Complex::ZERO; 1]; n + 1];
            first_order_tail(&mut t, &[a0], beta, sign);
            let lhs = g.block(0)[0] - t[0][0];
            let rhs = sg * (g.block(n)[0] - t[n][0]);
            assert!((lhs - rhs).abs() <= 1e-12, "{sign:?}");

            // the transform of the tail is a0/(i omega), exact for the
            // cubic order
            let mut tt = MatComp::new(n, 1, 1, Complex::ZERO);
            for (r, tr) in t.iter().enumerate().take(n + 1) {
                tt.block_mut(r)[0] = tr[0];
            }
            let raw = matsubara_dft(&tt, sign);
            for m in [-3i64, 1, 5] {
                let x = matsubara_ft(m, &tt, &raw, sign, beta, 3);
                let om = get_omega(m, beta, sign);
                let want = a0 / Complex::new(0.0, om);
                assert!((x[0] - want).abs() <= 1e-12, "m {m}: {} vs {want}", x[0]);
            }
        }
        // matrix coefficients
        for sign in [Sign::Fermi, Sign::Bose] {
            let cm = vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 2.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ];
            let mut dm = vec![vec![Complex::ZERO; 4]; n + 1];
            first_order_tail(&mut dm, &cm, 2.0, sign);
            for q in 0..4 {
                let want = cm[q].scale(-0.5);
                assert!((dm[0][q] - want).abs() < 1e-15, "q {q}");
            }
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
                let arg = (2 * m + 1) as f64 * r as f64 * PI / n as f64;
                direct += Complex::from_polar(1.0, arg) * a.get(r as usize, 0, 0);
            }
            assert!((mdft[m as usize][0] - direct).abs() < 1e-12, "m {m}");
        }
    }

    #[test]
    fn raw_dft_delta_and_matrix() {
        // a delta at tau = 0 transforms to all ones (boson)
        let mut g = MatComp::new(16, 1, 1, Complex::ZERO);
        g.block_mut(0)[0] = Complex::ONE;
        let d = matsubara_dft(&g, Sign::Bose);
        for (m, dm) in d.iter().enumerate() {
            assert!((dm[0] - Complex::ONE).abs() < 1e-14, "m {m}");
        }
        // direct sum for a 2x2 matrix function, m = 0..N
        let (n, beta) = (8i64, 2.0);
        let g = mf_matrix(&[0.3, 1.1], n, beta, Sign::Fermi);
        let d = matsubara_dft(&g, Sign::Fermi);
        for m in 0..=n {
            let mut s = [Complex::ZERO; 4];
            for r in 0..=n {
                let arg = PI * (2 * m + 1) as f64 * r as f64 / n as f64;
                let w = Complex::from_polar(1.0, arg);
                let blk = g.block(r as usize);
                for q in 0..4 {
                    s[q] += w * blk[q];
                }
            }
            for q in 0..4 {
                assert!(
                    (d[m as usize][q] - s[q]).abs() < 1e-13,
                    "m {m} q {q}: {} vs {}",
                    d[m as usize][q],
                    s[q]
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn fail_get_omega_zero_beta() {
        let _ = get_omega(1, 0.0, Sign::Fermi);
    }

    #[test]
    #[should_panic]
    fn fail_tail_coeff_size_mismatch() {
        let mut dm = vec![vec![Complex::ZERO; 4]; 17];
        first_order_tail(&mut dm, &[Complex::ONE], 2.0, Sign::Fermi);
    }

    #[test]
    #[should_panic]
    fn fail_tail_negative_beta() {
        let mut dm = vec![vec![Complex::ZERO; 1]; 17];
        first_order_tail(&mut dm, &[Complex::ONE], -1.0, Sign::Fermi);
    }

    #[test]
    #[should_panic]
    fn fail_ft_order_two() {
        let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
        let d = matsubara_dft(&g, Sign::Fermi);
        let _ = matsubara_ft(0, &g, &d, Sign::Fermi, 2.0, 2);
    }

    #[test]
    #[should_panic]
    fn fail_ft_negative_beta() {
        let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
        let d = matsubara_dft(&g, Sign::Fermi);
        let _ = matsubara_ft(0, &g, &d, Sign::Fermi, -2.0, 3);
    }

    #[test]
    #[should_panic]
    fn fail_ft_raw_len_mismatch() {
        let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
        let mut d = matsubara_dft(&g, Sign::Fermi);
        d.pop();
        let _ = matsubara_ft(0, &g, &d, Sign::Fermi, 2.0, 3);
    }
}
