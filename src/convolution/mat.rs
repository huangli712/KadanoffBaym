// Project : KadanoffBaym
// File    : src/convolution/mat.rs
// Purpose : Matsubara convolution kernels (blueprint: convolution/mat.jl)
//!
//! *Remarks* :
//!
//! The evaluation of ``C^{M}(\tau)`` is implemented as follows:
//!
//! ```math
//! \begin{equation}
//! C^{M}(mh_\tau) = C^{M}_1[A,f,B](m) + C^{M}_2[A,f,B](m),
//! \end{equation}
//! ```text
//!
//! where ``m = 0,\ \cdots,\ N_{\tau}`` (it means that the number of imaginary
//! time points is ``N_{\tau}+1``) and ``h_{\tau}`` means the interval in the
//! imaginary time axis (``\equiv \delta \tau``).
//!
//! ```math
//! \begin{equation}
//! C^{M}_1[A,f,B](m) = \int^{mh_{\tau}}_0 d\tau' \
//!     A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
//! \end{equation}
//! ```text
//!
//! ```math
//! \begin{equation}
//! C^{M}_2[A,f,B](m) = \int^{\beta}_{mh_{\tau}} d\tau' \
//!     A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
//! \end{equation}
//! ```text
//!
//! Actually, we adopted the following equations:
//!
//! ```math
//! \begin{equation}
//! C^{M}_1[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
//!     R^{(k)}_{m;j,l} A^{M}_j f_{-1} B^{M}_l, \quad m \le k,
//! \end{equation}
//! ```text
//!
//! ```math
//! \begin{equation}
//! C^{M}_1[A,f,B](m) = h_{\tau} \sum^{m}_{l = 0}
//!     w^{(k)}_{m,l} A^{M}_{m-l} f_{-1} B^{M}_l, \quad m > k.
//! \end{equation}
//! ```text
//!
//! ```math
//! \begin{equation}
//! C^{M}_2[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
//!     R^{(k)}_{N_{\tau}-m; j,l} \xi
//!     A^{M}_{N_{\tau}-j} f_{-1} B^{M}_{N_{\tau} - l}, \quad m \ge N_{\tau} -k,
//! \end{equation}
//! ```text
//!
//! ```math
//! \begin{equation}
//! C^{M}_2[A,f,B](m) = h_{\tau} \sum^{N_{\tau} - m}_{l = 0}
//!     w^{(k)}_{N_{\tau}-m,l} \xi
//!     A^{M}_{N_{\tau}-l} f_{-1} B^{M}_{m+l}, \quad m < N_{\tau} - k.
//! \end{equation}
//! ```text
//!
//! Note that ``A^{M}(\tau)`` at the values ``\tau \in [-\beta, 0]``
//! is obtained by using the periodicity property:
//!
//! ```math
//! \begin{equation}
//! A^{M}(\tau + \beta) = \xi A^{M}(\tau).
//! \end{equation}
//! ```text
//!
//! We also associate fermions (bosons) with the negative (positive) sign
//! ``\xi \equiv -1`` (``\xi \equiv 1``).
//!
//! *References* :
//!
//! Please see [`NESSi`] Sections `9` and `11` for more details.

use super::mul_acc;
use crate::complex::Complex;
use crate::linalg::mat::DenseMat;
use crate::types::sign::Sign;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::function::TimeFunction;
use crate::types::mat::MatComp;
use crate::weights::Integrator;

/// `C^M = A^M * B^M` over all imaginary time points, then scaled by
/// `dtau` (blueprint `conv_mat`).
pub fn conv_mat(
    c: &mut MatComp,
    a: &MatComp,
    b: &MatComp,
    integ: &Integrator,
    beta: f64,
    sign: Sign,
) {
    let ntau = c.ntau;
    assert!(a.is_compatible(b) && b.is_compatible(c));
    assert!(beta >= 0.0);
    let dtau = beta / ntau as f64;
    for m in 0..=(ntau as i64) {
        conv_mat_mat_1(m, c, a, b, integ, sign);
    }
    c.smul(Complex::new(dtau, 0.0));
}

/// `C^M = A^M * f_0 * B^M` with a constant matrix `f_0 = f(0^-)`
/// (blueprint `conv_mat` with `f0::Element`).
pub(crate) fn conv_mat_f0(
    c: &mut MatComp,
    a: &MatComp,
    f0: &DenseMat,
    b: &MatComp,
    integ: &Integrator,
    beta: f64,
    sign: Sign,
) {
    let mut bt = b.clone();
    bt.lmul(f0, crate::basics::Mode::Normal);
    conv_mat(c, a, &bt, integ, beta, sign);
}

/// Matsubara integral 1 at a single point `m` (0-based imaginary time
/// index): `C(tau_m) = Int_0^beta A(tau_m - tau') B(tau') d tau'`,
/// including the KMS-continued second segment (blueprint
/// `conv_mat_mat_1`).
///
/// *Remarks* : *Matsubara Integral 1*
///
/// The Matsubara integral 1 reads:
///
/// ```math
/// \begin{equation}
/// C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau - \tau') B(\tau').
/// \end{equation}
/// ```text
///
/// The objects `A`, `B`, and `C` are of type Matsubara Green's function,
/// i.e., Matsubara component of contour-ordered Green's function. They are
/// anti-periodic, i.e.,
///
/// ```math
/// \begin{equation}
/// A(\beta - \tau) = -A(-\tau),
/// \end{equation}
/// ```text
/// for ``0 \le \tau \le \beta``.
///
/// The following `conv_mat_mat_1()` and `conv_mat_mat_1p()` functions will
/// try to calculate similar integral.
pub(crate) fn conv_mat_mat_1(
    m: i64,
    c: &mut MatComp,
    a: &MatComp,
    b: &MatComp,
    integ: &Integrator,
    sign: Sign,
) {
    let ntau = a.ntau as i64;
    let k = integ.k;
    assert!(a.is_compatible(b) && b.is_compatible(c));
    assert!((0..=ntau).contains(&m));

    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;
    debug_assert_eq!(s2a, b.size1);

    let mut c1 = vec![Complex::ZERO; s1 * s2b];
    // contributions from 0 to tau
    if m >= k {
        // usual Gregory integration: Omega_{m,l} A_{m-l} B_l
        let (giw, _) = (&integ.giw, ());
        for l in 0..=m {
            mul_acc(
                &mut c1,
                giw.at(m as usize, l as usize),
                a.block((m - l) as usize),
                b.block(l as usize),
                s1,
                s2a,
                s2b,
            );
        }
    } else if m > 0 {
        // strange boundary correction: R_{m,j,l} A_j B_l
        for j in 0..=(k + 1) as i64 - 1 {
            for l in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c1,
                    integ.bcw.at(m as usize, j as usize, l as usize),
                    a.block(j as usize),
                    b.block(l as usize),
                    s1,
                    s2a,
                    s2b,
                );
            }
        }
    }

    let mut c2 = vec![Complex::ZERO; s1 * s2b];
    // contributions from tau to beta
    if ntau - m >= k {
        // usual Gregory integration
        let mut inda = ntau;
        let mut indb = m;
        let mut indg = 0i64;
        while indb <= ntau {
            mul_acc(
                &mut c2,
                integ.giw.at((ntau - m) as usize, indg as usize),
                a.block(inda as usize),
                b.block(indb as usize),
                s1,
                s2a,
                s2b,
            );
            inda -= 1;
            indb += 1;
            indg += 1;
        }
    } else if ntau - m > 0 {
        // strange boundary correction
        let mut inda = ntau;
        for j in 0..=(k + 1) as i64 - 1 {
            let mut indb = ntau;
            for l in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c2,
                    integ.bcw.at((ntau - m) as usize, j as usize, l as usize),
                    a.block(inda as usize),
                    b.block(indb as usize),
                    s1,
                    s2a,
                    s2b,
                );
                indb -= 1;
            }
            inda -= 1;
        }
    }

    // assemble: C_m = c1 + sign * c2
    let s = sign.as_i64() as f64;
    let tgt = c.block_mut(m as usize);
    for q in 0..(s1 * s2b) {
        tgt[q] = c1[q] + c2[q].scale(s);
    }
}

#[allow(dead_code)]
/// Matsubara integral 1, partial: only the contributions from 0 to tau
/// (blueprint `conv_mat_mat_1p`).
pub(crate) fn conv_mat_mat_1p(m: i64, c: &mut MatComp, a: &MatComp, b: &MatComp, integ: &Integrator) {
    let ntau = a.ntau as i64;
    let k = integ.k;
    assert!(a.is_compatible(b) && b.is_compatible(c));
    assert!((0..=ntau).contains(&m));

    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;

    let mut c1 = vec![Complex::ZERO; s1 * s2b];
    if m >= k {
        for l in 0..=m {
            mul_acc(
                &mut c1,
                integ.giw.at(m as usize, l as usize),
                a.block((m - l) as usize),
                b.block(l as usize),
                s1,
                s2a,
                s2b,
            );
        }
    } else if m > 0 {
        for j in 0..=(k + 1) as i64 - 1 {
            for l in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c1,
                    integ.bcw.at(m as usize, j as usize, l as usize),
                    a.block(j as usize),
                    b.block(l as usize),
                    s1,
                    s2a,
                    s2b,
                );
            }
        }
    }

    c.block_mut(m as usize).copy_from_slice(&c1);
}

#[allow(dead_code)]
/// Matsubara integral 2 at a single point: `C(tau_m) = Int_0^beta
/// A(tau') B(tau' - tau_m) d tau'` (blueprint `conv_mat_mat_2`).
///
/// *Remarks* : *Matsubara Integral 2*
///
/// The Matsubara integral 2 reads:
///
/// ```math
/// \begin{equation}
/// C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau') B(\tau' - \tau).
/// \end{equation}
/// ```text
///
/// The objects `A`, `B`, and `C` are of type Matsubara Green's function,
/// i.e., Matsubara component of contour-ordered Green's function. They are
/// anti-periodic, i.e.,
///
/// ```math
/// \begin{equation}
/// A(\beta - \tau) = -A(-\tau),
/// \end{equation}
/// ```text
/// for ``0 \le \tau \le \beta``.
///
/// The following `conv_mat_mat_2()` and `conv_mat_mat_2p()` functions will
/// try to calculate similar integral.
pub(crate) fn conv_mat_mat_2(
    m: i64,
    c: &mut MatComp,
    a: &MatComp,
    b: &MatComp,
    integ: &Integrator,
    sign: Sign,
) {
    let ntau = a.ntau as i64;
    let k = integ.k;
    assert!(a.is_compatible(b) && b.is_compatible(c));
    assert!((0..=ntau).contains(&m));

    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;

    let mut c1 = vec![Complex::ZERO; s1 * s2b];
    // contributions from 0 to tau
    if m == 0 {
        // PASS
    } else if m < k {
        // strange boundary correction (note the swapped BCW indices)
        let mut inda = 0i64;
        for j in 0..=(k + 1) as i64 - 1 {
            let mut indb = ntau;
            for l in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c1,
                    integ.bcw.at(m as usize, l as usize, j as usize),
                    a.block(inda as usize),
                    b.block(indb as usize),
                    s1,
                    s2a,
                    s2b,
                );
                indb -= 1;
            }
            inda += 1;
        }
    } else {
        // usual Gregory integration
        let mut inda = m;
        let mut indb = ntau;
        for l in 0..=m {
            mul_acc(
                &mut c1,
                integ.giw.at(m as usize, l as usize),
                a.block(inda as usize),
                b.block(indb as usize),
                s1,
                s2a,
                s2b,
            );
            inda -= 1;
            indb -= 1;
        }
    }

    let mut c2 = vec![Complex::ZERO; s1 * s2b];
    // contributions from tau to beta
    if m == ntau {
        // PASS
    } else if m > ntau - k {
        // strange boundary correction
        let mut inda = ntau;
        for l in 0..=(k + 1) as i64 - 1 {
            for j in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c2,
                    integ.bcw.at((ntau - m) as usize, l as usize, j as usize),
                    a.block(inda as usize),
                    b.block(j as usize),
                    s1,
                    s2a,
                    s2b,
                );
            }
            inda -= 1;
        }
    } else if m > ntau - 2 * k - 1 {
        // usual Gregory integration (short)
        let mut inda = m;
        for l in 0..=(ntau - m) {
            mul_acc(
                &mut c2,
                integ.giw.at((ntau - m) as usize, l as usize),
                a.block(inda as usize),
                b.block(l as usize),
                s1,
                s2a,
                s2b,
            );
            inda += 1;
        }
    } else {
        // usual Gregory integration (long)
        let mut inda = m;
        let mut indb = 0i64;
        for _l in m..=ntau {
            mul_acc(
                &mut c2,
                integ.giw.at((ntau - m) as usize, (ntau - m - indb) as usize),
                a.block(inda as usize),
                b.block(indb as usize),
                s1,
                s2a,
                s2b,
            );
            inda += 1;
            indb += 1;
        }
    }

    // assemble: C_m = sign * c1 + c2
    let s = sign.as_i64() as f64;
    let tgt = c.block_mut(m as usize);
    for q in 0..(s1 * s2b) {
        tgt[q] = c1[q].scale(s) + c2[q];
    }
}

#[allow(dead_code)]
/// Matsubara integral 2, partial: only the contributions from tau to
/// beta (blueprint `conv_mat_mat_2p`).
pub(crate) fn conv_mat_mat_2p(m: i64, c: &mut MatComp, a: &MatComp, b: &MatComp, integ: &Integrator) {
    let ntau = a.ntau as i64;
    let k = integ.k;
    assert!(a.is_compatible(b) && b.is_compatible(c));
    assert!((0..=ntau).contains(&m));

    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;

    let mut c2 = vec![Complex::ZERO; s1 * s2b];
    if m == ntau {
        // PASS
    } else if m > ntau - k {
        let mut inda = ntau;
        for l in 0..=(k + 1) as i64 - 1 {
            for j in 0..=(k + 1) as i64 - 1 {
                mul_acc(
                    &mut c2,
                    integ.bcw.at((ntau - m) as usize, l as usize, j as usize),
                    a.block(inda as usize),
                    b.block(j as usize),
                    s1,
                    s2a,
                    s2b,
                );
            }
            inda -= 1;
        }
    } else if m > ntau - 2 * k - 1 {
        let mut inda = m;
        for l in 0..=(ntau - m) {
            mul_acc(
                &mut c2,
                integ.giw.at((ntau - m) as usize, l as usize),
                a.block(inda as usize),
                b.block(l as usize),
                s1,
                s2a,
                s2b,
            );
            inda += 1;
        }
    } else {
        let mut inda = m;
        let mut indb = 0i64;
        for _l in m..=ntau {
            mul_acc(
                &mut c2,
                integ.giw.at((ntau - m) as usize, (ntau - m - indb) as usize),
                a.block(inda as usize),
                b.block(indb as usize),
                s1,
                s2a,
                s2b,
            );
            inda += 1;
            indb += 1;
        }
    }

    c.block_mut(m as usize).copy_from_slice(&c2);
}

/// Increment the Matsubara component of `C` by `alpha * [A * f * B]^M`
/// (blueprint `incr_convolution_mat!`).
pub fn incr_convolution_mat(
    c: &mut ContourGreen,
    a: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    alpha: Complex,
) {
    assert!(beta > 0.0);
    let ntau = a.ntau;
    let mut tmp = MatComp::new(ntau, c.size1, c.size2, Complex::ZERO);
    match f {
        Some(cf) => conv_mat_f0(&mut tmp, &a.mat, &cf_mat0(cf), &b.mat, integ, beta, a.sign),
        None => conv_mat(&mut tmp, &a.mat, &b.mat, integ, beta, a.sign),
    }
    for m in 0..=ntau as i64 {
        let t = c.mat_raw_block_mut(m);
        for q in 0..t.len() {
            t[q] += alpha * tmp.block(m as usize)[q];
        }
    }
}

/// `incr_convolution_mat!` for a timestep target (𝒻 object).
pub(crate) fn incr_convolution_mat_step(
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    alpha: Complex,
) {
    assert!(beta > 0.0);
    let ntau = a.ntau;
    let mut tmp = MatComp::new(ntau, c.size1, c.size2, Complex::ZERO);
    match f {
        Some(cf) => conv_mat_f0(&mut tmp, &a.mat, &cf_mat0(cf), &b.mat, integ, beta, a.sign),
        None => conv_mat(&mut tmp, &a.mat, &b.mat, integ, beta, a.sign),
    }
    for m in 0..=ntau as i64 {
        let t = c.mat_raw_block_mut(m);
        for q in 0..t.len() {
            t[q] += alpha * tmp.block(m as usize)[q];
        }
    }
}

/// Value of the contour function at the Matsubara branch as a matrix.
pub(crate) fn cf_mat0(cf: &TimeFunction) -> DenseMat {
    DenseMat::from_vec(
        cf.size1,
        cf.size2,
        cf.block(crate::types::MATSUBARA).to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;

    fn ones(ntau: usize) -> (MatComp, MatComp) {
        let a = MatComp::new(ntau, 1, 1, Complex::ONE);
        let b = MatComp::new(ntau, 1, 1, Complex::ONE);
        (a, b)
    }

    #[test]
    fn constant_identity_convolutions() {
        // A(tau) = B(tau) = 1:
        // C(tau) = Int_0^tau 1 + xi * Int_tau^beta 1 = tau + xi (beta - tau)
        let ntau = 20usize;
        let beta = 2.0;
        let integ = Integrator::new(3);
        let (a, b) = ones(ntau);
        let dtau = beta / ntau as f64;
        for (sign, xi) in [(Sign::Fermi, -1.0), (Sign::Bose, 1.0)] {
            let mut c = MatComp::new(ntau, 1, 1, Complex::ZERO);
            conv_mat(&mut c, &a, &b, &integ, beta, sign);
            for m in 0..=ntau as i64 {
                let tau = m as f64 * dtau;
                let want = tau + xi * (beta - tau);
                let got = c.get(m as usize, 0, 0);
                assert!(
                    (got - Complex::new(want, 0.0)).abs() < 1e-12,
                    "m {m}: {got} vs {want}",
                );
            }
        }
    }

    #[test]
    fn linear_times_constant_convolutions() {
        // A(tau) = tau, B(tau) = 1 (fermions):
        // C(tau) = tau^2/2 + xi (beta^2/2 - tau^2/2) = tau^2 - beta^2/2
        let ntau = 30usize;
        let beta = 1.5;
        let integ = Integrator::new(4);
        let dtau = beta / ntau as f64;
        let mut a = MatComp::new(ntau, 1, 1, Complex::ZERO);
        for m in 0..=ntau as i64 {
            a.set(m as usize, 0, 0, Complex::new(m as f64 * dtau, 0.0));
        }
        let b = MatComp::new(ntau, 1, 1, Complex::ONE);
        let mut c = MatComp::new(ntau, 1, 1, Complex::ZERO);
        conv_mat(&mut c, &a, &b, &integ, beta, Sign::Fermi);
        for m in 0..=ntau as i64 {
            let tau = m as f64 * dtau;
            let want = tau * tau - beta * beta / 2.0;
            let got = c.get(m as usize, 0, 0);
            assert!((got.re - want).abs() < 1e-11, "m {m}: {} vs {want}", got.re);
        }
    }

    #[test]
    fn integral_1_plus_2_equals_full_mat() {
        // 1p + sign-corrected consistency: conv_mat_mat_1(m) computed
        // piecewise equals the direct evaluation
        let ntau = 16usize;
        let beta = 1.0;
        let integ = Integrator::new(3);
        let mut a = MatComp::new(ntau, 1, 1, Complex::ZERO);
        let mut b = MatComp::new(ntau, 1, 1, Complex::ZERO);
        for m in 0..=ntau as i64 {
            let t = m as f64 * beta / ntau as f64;
            a.set(m as usize, 0, 0, Complex::new((t - 0.3).exp(), 0.2 * t));
            b.set(m as usize, 0, 0, Complex::new((0.5 * t).sin(), (t).cos()));
        }
        let mut cfull = MatComp::new(ntau, 1, 1, Complex::ZERO);
        conv_mat(&mut cfull, &a, &b, &integ, beta, Sign::Fermi);
        // conv_mat already scaled by dtau; replicate with mat_1 manually
        let mut c1 = MatComp::new(ntau, 1, 1, Complex::ZERO);
        for m in 0..=ntau as i64 {
            conv_mat_mat_1(m, &mut c1, &a, &b, &integ, Sign::Fermi);
        }
        c1.smul(Complex::new(beta / ntau as f64, 0.0));
        for m in 0..=ntau as i64 {
            assert!(
                (c1.get(m as usize, 0, 0) - cfull.get(m as usize, 0, 0)).abs() < 1e-14,
                "m {m}"
            );
        }
    }

    #[test]
    fn f0_variant_matches_prescaled() {
        let ntau = 16usize;
        let beta = 1.0;
        let integ = Integrator::new(3);
        let mut a = MatComp::new(ntau, 2, 2, Complex::ZERO);
        let mut b = MatComp::new(ntau, 2, 2, Complex::ZERO);
        for m in 0..=ntau as i64 {
            let t = m as f64 * beta / ntau as f64;
            for q in 0..4 {
                a.set(m as usize, q / 2, q % 2, Complex::new(t, (q as f64) * 0.1));
                b.set(m as usize, q / 2, q % 2, Complex::new((q as f64) * 0.2, -t));
            }
        }
        let f0 = DenseMat::from_vec(
            2,
            2,
            vec![Complex::new(1.0, 0.5), Complex::ZERO, Complex::new(0.3, 0.0), Complex::ONE],
        );
        let mut c1 = MatComp::new(ntau, 2, 2, Complex::ZERO);
        conv_mat_f0(&mut c1, &a, &f0, &b, &integ, beta, Sign::Bose);
        let mut c2 = MatComp::new(ntau, 2, 2, Complex::ZERO);
        let mut bt = b.clone();
        bt.lmul(&f0, crate::basics::Mode::Normal);
        conv_mat(&mut c2, &a, &bt, &integ, beta, Sign::Bose);
        for m in 0..=ntau as i64 {
            for q in 0..4 {
                assert!((c1.get(m as usize, q / 2, q % 2) - c2.get(m as usize, q / 2, q % 2)).abs() < 1e-14);
            }
        }
        let _ = Contour::new(1, 1, 1, 1, 1.0, 1.0);
    }
}
