// Project : KadanoffBaym
// File    : src/fourier/mat.rs
// Purpose : Matsubara Fourier transforms and the frequency-domain seed
//

//!
//! *Remarks* : *Matsubara Seed*
//!
//! The private function `_matsubara_seed!` solves the integral equation
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
//! *References* :
//!
//! Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_mat_fourier_dispatch`.

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{MatComp, Sign};

use crate::linalg::{lu_factor, lu_solve_mat_into};

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
    PI / beta * (2 * m + one) as f64
}

/// First order tail correction `c_0(tau_r)` written into the `N + 1`
/// blocks (blueprint `_first_order_tail!`): `-a_0 / 2` for fermions,
/// `a_0 (tau/beta - 1/2)` for bosons.
///
/// First order tail correction ``c_0(\\tau)`` at the `N + 1` imaginary time
/// points ``\\tau_r = r\\beta/N``, where `N + 1 = size(xmat, 3)`. It is
/// ``-a_0/2`` for fermions, and ``a_0(\\tau/\\beta - 1/2)`` for bosons,
/// where ``a_0`` is `coeff`. It is a private function.
///
/// ### Arguments
/// * xmat  -> On return, xmat[:,:,r+1] = c₀(τᵣ).
/// * coeff -> Coefficient a₀ (jump of the Matsubara function).
/// * β     -> Inverse temperature.
/// * sign  -> FERMI or BOSE.
///
/// ### Returns
/// * xmat -> The tail correction.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `set_first_order_tail`.
///
/// First order tail correction ``c_0(\\tau_r)``, such that ``G(\\tau) +
/// c_0(\\tau)`` is continuous. `dest[r+1,:,:]` is ``c_0(\\tau_r)``, and the
/// number of imaginary time points is `size(dest, 1)`. `coeff` is not
/// changed.
///
/// ### Arguments
/// * dest  -> On return, the tail correction.
/// * coeff -> Jump coefficient ``a_0``.
/// * beta  -> Inverse temperature.
/// * sign  -> FERMI or BOSE.
///
/// ### Returns
/// * dest -> The tail correction.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `set_first_order_tail`.
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
///
/// Raw discrete Fourier transform (without the factor δτ) of a Matsubara
/// function, ``\\sum^{N}_{r=0} e^{i\\pi(2m+1)r/N} A(\\tau_r)`` for fermions,
/// or ``\\sum^{N}_{r=0} e^{2i\\pi m r/N} A(\\tau_r)`` for bosons, where
/// ``m = 0, \\cdots, N``. Here ``N = ntau - 1`` is the number of
/// imaginary time intervals. It is a private function.
///
/// ### Arguments
/// * A    -> Matsubara function.
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

    let arg = (2 * m + one) as f64 * PI / n as f64;
    let (corfac, bcorr): (f64, Vec<Complex>) = match order {
        0 => (1.0, Vec::new()),
        1 => {
            let (c, b) = crate::fourier::dftcorr_linear(arg);
            (c, b.to_vec())
        }
        3 => {
            let (c, b) = crate::fourier::dftcorr_cubic(arg);
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

/// Boundary-correction data of one frequency: the factor and the
/// correction terms (effective length `nterms`).
type Corr = (f64, [Complex; 4], usize);

fn corr_of(order: i64, arg: f64) -> Corr {
    match order {
        0 => (1.0, [Complex::ZERO; 4], 0),
        1 => {
            let (c, b) = crate::fourier::dftcorr_linear(arg);
            (c, [b[0], b[1], Complex::ZERO, Complex::ZERO], 2)
        }
        3 => {
            let (c, b) = crate::fourier::dftcorr_cubic(arg);
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
        let sb = Complex::new(sgn(sign), 0.0) * b.conj();
        for q in 0..res.len() {
            res[q] += *b * left[q] + sb * right[q];
        }
    }
    let dt = Complex::new(dtau, 0.0);
    for z in res.iter_mut() {
        *z = *z * dt;
    }
}

/// Twiddle and correction tables of the frequency-domain Matsubara
/// solvers (internal): `tw[k] = cis(pi k / n)` for `k = 0, .., 2n-1`
/// covers every phase `e^{i pi (2m+delta) r / n}` of the raw DFT and
/// every accumulation phase `e^{-i omega_m tau_r}` by periodicity, and
/// `corr[m1]` holds the boundary correction of the frequency `m1`
/// (mathematically `2 pi`-periodic in `m`, so the `2pcf+1` oversampling
/// windows share the `n` basic entries).
pub(crate) struct MatsubaraTables {
    n: i64,
    sign: Sign,
    order: i64,
    tw: Vec<Complex>,
}

impl MatsubaraTables {
    pub(crate) fn new(n: i64, sign: Sign, order: i64) -> MatsubaraTables {
        let tw: Vec<Complex> = (0..2 * n).map(|k| cis(k as f64 * PI / n as f64)).collect();
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
        out.resize_with(mps.len(), || Vec::new());
        for (i, &mp) in mps.iter().enumerate() {
            let m1 = mp.rem_euclid(n) as usize;
            let mut row = std::mem::take(&mut out[i]);
            ft_core_into(m1, a, mdft, self.sign, dtau, &corrs[i], &mut row);
            out[i] = row;
        }
    }
}

/// The seed solver scratch of the fixpoint and conjugate-gradient
/// Matsubara Dyson solvers (internal): the kernel `F` is fixed across
/// the iterations, so its raw transform and corrected frequency values
/// are computed once and reused (`dyson/mat.rs`).
pub(crate) struct MatsubaraSeed {
    d: usize,
    nel: usize,
    beta: f64,
    sign: Sign,
    tables: MatsubaraTables,
    mps: Vec<i64>,
    corrs: Vec<Corr>,
    f_ft: Vec<Vec<Complex>>,
    b_dft: Vec<Vec<Complex>>,
    b_ft: Vec<Vec<Complex>>,
    xmat: Vec<Vec<Complex>>,
    qmasy: Vec<Complex>,
    z1: DenseMat,
    z1f: DenseMat,
    z3m: DenseMat,
    sol: DenseMat,
    xout: DenseMat,
}

impl MatsubaraSeed {
    /// Prepare the fixed kernel `f` (blueprint `_matsubara_seed!` with
    /// the kernel transforms hoisted out of the iteration loop).
    pub(crate) fn new(f: &MatComp, beta: f64, sign: Sign, pcf: i64, order: i64) -> MatsubaraSeed {
        let d = f.size1;
        assert_eq!(d, f.size2, "square matrices are expected");
        let n = f.ntau as i64;
        assert!(
            n % 2 == 0 && n >= 4,
            "the number of imaginary time intervals must be even and >= 4, got {n}"
        );
        assert!(pcf >= 0, "pcf must be nonnegative");
        assert!(beta > 0.0, "beta must be positive");
        let tables = MatsubaraTables::new(n, sign, order);
        // the (p, m) window enumeration: the corrections depend on the
        // actual frequency, so the lists are aligned with this order
        let mut mps = Vec::new();
        for p in -pcf..=pcf {
            let (mut m_low, mut m_high) = (0i64, 0i64);
            if sign == Sign::Bose {
                m_high = if p < 0 { 0 } else { 1 };
                m_low = if p > 0 { -1 } else { 0 };
            }
            let m2 = n / 2;
            for m in (-m2 - m_low)..=(m2 - 1 + m_high) {
                mps.push(m + p * n);
            }
        }
        let corrs: Vec<Corr> = mps.iter().map(|&mp| tables.corr_of_mp(mp)).collect();
        let mut f_dft = Vec::new();
        tables.dft_into(f, &mut f_dft);
        let mut f_ft = Vec::new();
        tables.ft_list(f, &f_dft, beta, &mps, &corrs, &mut f_ft);
        drop(f_dft);
        MatsubaraSeed {
            d,
            nel: d * d,
            beta,
            sign,
            tables,
            mps,
            corrs,
            f_ft,
            b_dft: Vec::new(),
            b_ft: Vec::new(),
            xmat: vec![vec![Complex::ZERO; d * d]; (n + 1) as usize],
            qmasy: vec![Complex::ZERO; d * d],
            z1: DenseMat::zeros(d),
            z1f: DenseMat::zeros(d),
            z3m: DenseMat::zeros(d),
            sol: DenseMat::zeros(d),
            xout: DenseMat::zeros(d),
        }
    }

    /// Solve `[1 + F] X = B` for the source `b` (the per-iteration part
    /// of `_matsubara_seed!`).
    pub(crate) fn solve(&mut self, x: &mut MatComp, b: &MatComp) {
        assert!(x.is_compatible(b), "X and B are not compatible");
        let n = self.tables.n;
        let d = self.d;
        let nel = self.nel;
        let beta = self.beta;
        let sign = self.sign;
        let one = if sign == Sign::Fermi { 1i64 } else { 0 };
        // first order tail correction, q_inf = xi B(beta) - B(0)
        let qmasy = &mut self.qmasy;
        for q in 0..nel {
            qmasy[q] =
                Complex::new(sgn(sign), 0.0) * b.block(n as usize)[q] - b.block(0)[q];
        }
        first_order_tail(&mut self.xmat, qmasy, beta, sign);

        let mut b_dft = std::mem::take(&mut self.b_dft);
        let mut b_ft = std::mem::take(&mut self.b_ft);
        self.tables.dft_into(b, &mut b_dft);
        self.tables.ft_list(b, &b_dft, beta, &self.mps, &self.corrs, &mut b_ft);

        for (imp, &mp) in self.mps.iter().enumerate() {
            {
                let omega = (2 * mp + one) as f64 * PI / beta;

                // tail term in frequency space, q_inf / (i omega)
                let z3_is_zero = sign == Sign::Bose && mp == 0;
                if !z3_is_zero {
                    let f = Complex::new(0.0, -1.0 / omega);
                    for q in 0..nel {
                        self.z3m.raw_mut()[q] = f * qmasy[q];
                    }
                } else {
                    for q in 0..nel {
                        self.z3m.raw_mut()[q] = Complex::ZERO;
                    }
                }

                // [1 + F]^{-1} [B - (1 + F) q_inf / (i omega)] / beta
                let fiwo = &self.f_ft[imp];
                let qiwo = &b_ft[imp];
                let fim = DenseMat::from_slice(d, d, fiwo);
                for i in 0..d {
                    for j in 0..d {
                        let mut v = Complex::ZERO;
                        if i == j {
                            v = Complex::ONE;
                        }
                        self.z1.set(i, j, v + fiwo[i * d + j]);
                    }
                }
                let fz3 = fim.mat_mul(&self.z3m);
                for q in 0..nel {
                    self.sol.raw_mut()[q] = qiwo[q] - self.z3m.raw()[q] - fz3.raw()[q];
                }
                self.z1f.set_from(&self.z1);
                let piv = lu_factor(&mut self.z1f);
                lu_solve_mat_into(&self.z1f, &piv, &self.sol, &mut self.xout);
                for q in self.xout.raw_mut().iter_mut() {
                    *q /= Complex::new(beta, 0.0);
                }

                // accumulation: e^{-i omega tau_r} = conj(tw[k]),
                // k = (2 mp + one) r mod 2n
                let t = (2 * mp + one).rem_euclid(2 * n) as usize;
                let tw = &self.tables.tw;
                for (r, blk) in self.xmat.iter_mut().enumerate() {
                    let k = (t * r) % (2 * n as usize);
                    let w = tw[k].conj();
                    for (z, &sv) in blk.iter_mut().zip(self.xout.raw()) {
                        *z += w * sv;
                    }
                }
            }
        }

        for (dst, src) in x.raw_mut().chunks_mut(nel).zip(&self.xmat) {
            dst.copy_from_slice(src);
        }
        assert!(
            x.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "Matsubara seed is not finite"
        );
        self.b_dft = b_dft;
        self.b_ft = b_ft;
    }
}

/// Solve `[1 + F] X = B` for Matsubara functions in the frequency
/// domain (blueprint `_matsubara_seed!`). The number of imaginary time
/// intervals must be even and at least 4; `pcf` is the number of the
/// oversampling copies on each side.
///
/// Solve ``[1 + F] * X = B`` for Matsubara functions in the frequency
/// domain, see *Remarks* : *Matsubara Seed*. The number of imaginary time
/// intervals must be even and at least 4. It is a private function.
///
/// ### Arguments
/// * X     -> Matsubara function, the solution.
/// * F     -> Matsubara function, the kernel.
/// * B     -> Matsubara function, the source term.
/// * β     -> Inverse temperature.
/// * sign  -> FERMI or BOSE.
/// * pcf   -> Number of the oversampling copies on each side.
/// * order -> Order of the boundary correction.
///
/// ### Returns
/// * X -> The solution.
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
    let mut solver = MatsubaraSeed::new(f, beta, sign, pcf, order);
    solver.solve(x, b);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn tau_and_omega_helpers() {
        let tau = get_tau(3, 4.0, 9);
        assert!((tau - 4.0 * 3.0 / 8.0).abs() < 1e-15);
        let w = get_omega(2, 1.0, Sign::Fermi);
        assert!((w - 5.0 * PI).abs() < 1e-12);
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
                let arg = (2 * m + 1) as f64 * r as f64 * PI / n as f64;
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
                Complex::new((-(0.5 * tau)).exp() * (0.3 * tau).cos(), 0.0)
                    + Complex::new(0.0, 0.11 * tau),
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

#[cfg(test)]
mod probe_tables {
    use super::*;
    use crate::types::{Contour, ContourGreen};

    #[test]
    fn probe_dft_and_memo() {
        let (beta, n) = (4.0f64, 16i64);
        let c = Contour::new(1, n, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for r in 0..=n {
            let tau = r as f64 * beta / n as f64;
            g.set_mat(r, 0, 0, Complex::new((-(0.5 * tau)).exp(), 0.0) + Complex::new(0.0, 0.11 * tau));
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
}
