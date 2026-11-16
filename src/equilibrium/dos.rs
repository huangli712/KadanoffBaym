// Project : KadanoffBaym
// File    : src/equilibrium/dos.rs
// Purpose : equilibrium propagators from a density of states
//
// Split from the former standard.rs

use crate::complex::Complex;
use crate::math::BetheDos;
use crate::types::{ContourGreen, Sign};

use crate::fourier::{adft_eval, adft_sample};
use crate::math::{bose, bose_tau, fermi, fermi_tau};

/// The component selector of the DOS integrand.
pub(crate) enum DosComp {
    Ret,
    Les,
    Tv,
    Mat,
}

fn check_equilibrium(g: &ContourGreen, beta: f64, limit: i64, nn: i64) {
    assert!(beta > 0.0, "beta must be positive");
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        nn % 2 == 0 && nn >= 20,
        "nn must be even and >= 20, got {nn}"
    );
    assert!(limit >= 1, "limit must be >= 1");
}

/// Integrand of the equilibrium Green's function; the DOS must be finite
/// and nonnegative, and the bosonic zero mode is rejected.
fn dos_integrand(
    dos: &dyn Fn(f64) -> f64,
    eps: f64,
    comp: DosComp,
    sign: Sign,
    mu: f64,
    beta: f64,
    tau: f64,
) -> Complex {
    let a = dos(eps);
    assert!(a.is_finite(), "DOS is not finite at eps = {eps}");
    assert!(a >= 0.0, "DOS is negative at eps = {eps}");
    let x = eps - mu;
    let fermion = sign == Sign::Fermi;
    if !fermion && !matches!(comp, DosComp::Ret) && x == 0.0 {
        panic!("zero mode of the Bose function at eps = {eps}");
    }
    let nxi = |y: f64| {
        if fermion {
            fermi(beta, y)
        } else {
            bose(beta, y)
        }
    };
    let xi = sign.as_i64() as f64;
    let fm = match comp {
        DosComp::Ret => 1.0,
        DosComp::Les => xi * nxi(x),
        DosComp::Tv => {
            xi * if fermion {
                fermi_tau(beta, tau, x)
            } else {
                bose_tau(beta, tau, x)
            }
        }
        DosComp::Mat => {
            if x > 0.0 {
                xi * (-tau * x).exp() * nxi(-x)
            } else {
                -((beta - tau) * x).exp() * nxi(x)
            }
        }
    };
    Complex::new(a * fm, 0.0)
}

#[allow(dead_code)]
/// All the components of the equilibrium Green's function of the Bethe
/// lattice `rho(eps) = sqrt(4V^2 - (eps - E0)^2) / (2 pi V^2)`.
#[allow(clippy::too_many_arguments)]
fn from_dos_bethe(
    g: &mut ContourGreen,
    beta: f64,
    hstep: f64,
    v: f64,
    e0: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    assert!(v > 0.0, "V must be positive");
    let rho = BetheDos::new(e0 - 2.0 * v, e0 + 2.0 * v);
    let lo = rho.lo;
    let hi = rho.hi;
    from_dos(
        g,
        &|eps: f64| rho.at(eps),
        lo,
        hi,
        beta,
        hstep,
        mu,
        limit,
        nn,
    );
}

/// Matsubara component of the equilibrium Green's function for the
/// density of states `dos` with the support `[lo, hi]`.
#[allow(clippy::too_many_arguments)]
pub fn from_dos_mat(
    g: &mut ContourGreen,
    dos: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    beta: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    check_equilibrium(g, beta, limit, nn);
    let sign = g.sign;
    let d = g.size1;
    let dtau = beta / g.ntau as f64;
    for m in 0..=(g.ntau as i64) {
        let tau = m as f64 * dtau;
        let fz = |eps: f64| dos_integrand(dos, eps, DosComp::Mat, sign, mu, beta, tau);
        let p = adft_sample(fz, lo, hi, nn, limit, 0.0, 20);
        let (res, _) = adft_eval(&p, 0.0);
        for i in 0..d {
            for j in 0..d {
                let v = if i == j { res } else { Complex::ZERO };
                g.set_mat(m, i, j, v);
            }
        }
    }
}

/// All the components of the equilibrium Green's function for the
/// density of states `dos`.
#[allow(clippy::too_many_arguments)]
///
/// *Remarks* : *Equilibrium Green's Functions From DOS*
///
/// For a density of states ``\rho(\epsilon)`` with support ``[lo, hi]``,
/// ``x = \epsilon - \mu``, and the statistics ``\xi = -1`` (``+1``) for
/// fermions (bosons), the equilibrium Green's function reads
///
/// ```math
/// \begin{equation}
/// G^{R}(t,t') = -i \int d\epsilon\ \rho(\epsilon) e^{-i\epsilon(t-t')},
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// G^{<}(t,t') = -i\xi \int d\epsilon\ \rho(\epsilon) n_{\xi}(x)
///     e^{-i\epsilon(t-t')},
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// G^{\rceil}(t,\tau) = -i\xi \int d\epsilon\ \rho(\epsilon)
///     e^{\tau x} n_{\xi}(x) e^{-i\epsilon t}.
/// \end{equation}
/// ```
///
/// The Matsubara component is evaluated by the overflow-safe expressions
/// ``\xi \rho(\epsilon) e^{-\tau x} n_{\xi}(-x)`` (``x > 0``) and
/// ``-\rho(\epsilon) e^{(\beta-\tau) x} n_{\xi}(x)`` (``x \le 0``). For
/// matrix-valued functions, the result is ``G = g\ I_d``.
///
/// The Fourier integrals are evaluated by the adaptive, cubically corrected
/// discrete Fourier transform. The integration interval is divided into
/// panels of `nn` subintervals. The panels are determined at ω = 0: the
/// first panel is always bisected (if `limit > 1`), then the panel with the
/// largest error estimate is bisected, until there are `limit - 1` panels.
/// There is no tolerance and no early termination. The error estimate is
/// given by the difference between the results with `nn` and `nn/2`
/// subintervals.
pub fn from_dos(
    g: &mut ContourGreen,
    dos: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    beta: f64,
    hstep: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    check_equilibrium(g, beta, limit, nn);
    assert!(hstep > 0.0, "h must be positive");
    let sign = g.sign;
    let d = g.size1;
    let nt = g.nt as i64;
    let ntau = g.ntau as i64;
    let dtau = beta / g.ntau as f64;
    let i = Complex::new(0.0, 1.0);

    // Matsubara component
    from_dos_mat(g, dos, lo, hi, beta, mu, limit, nn);

    // retarded component, G^R(t,t') with l = t - t'
    let fz_ret = |eps: f64| dos_integrand(dos, eps, DosComp::Ret, sign, mu, beta, 0.0);
    let p = adft_sample(fz_ret, lo, hi, nn, limit, 0.0, 20);
    for l in 0..=nt {
        let (res, _) = adft_eval(&p, -hstep * l as f64);
        let res = -i * res;
        for row in l..=nt {
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { Complex::ZERO };
                    g.set_ret(row, row - l, a, b, v);
                }
            }
        }
    }

    // left-mixing component
    for m in 0..=ntau {
        let tau = m as f64 * dtau;
        let fz_tv = |eps: f64| dos_integrand(dos, eps, DosComp::Tv, sign, mu, beta, tau);
        let p = adft_sample(fz_tv, lo, hi, nn, limit, 0.0, 20);
        for n in 0..=nt {
            let (res, _) = adft_eval(&p, -n as f64 * hstep);
            let res = -i * res;
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { Complex::ZERO };
                    g.set_lmix(n, m, a, b, v);
                }
            }
        }
    }

    // lesser component, G^<(t',t) with l = t - t'
    let fz_les = |eps: f64| dos_integrand(dos, eps, DosComp::Les, sign, mu, beta, 0.0);
    let p = adft_sample(fz_les, lo, hi, nn, limit, 0.0, 20);
    for l in 0..=nt {
        let (res, _) = adft_eval(&p, hstep * l as f64);
        let res = -i * res;
        for col in l..=nt {
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { Complex::ZERO };
                    g.set_less(col - l, col, a, b, v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::Contour;

    /// Bethe lattice: G^R(t,t) = -i (the DOS is normalized), and the
    /// fermionic relation mat(0) + mat(beta) = -integral rho = -1 links
    /// the Matsubara endpoints; less(0,0) = i (mat(0) + 1).
    #[test]
    fn from_dos_bethe_normalization() {
        let (beta, hstep, nt, ntau) = (2.0f64, 0.05, 10usize, 100usize);
        let c = Contour::new(nt as i64, ntau as i64, 2, 2, nt as f64 * hstep, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_dos_bethe(&mut g, beta, hstep, 1.0, 0.0, 0.0, 100, 20);

        let i = Complex::new(0.0, 1.0);
        for n in [0i64, 5, 10] {
            assert!(
                (g.get_ret(n, n, 0, 0) + i).abs() < 1e-6,
                "ret({n},{n}) = {:?}",
                g.get_ret(n, n, 0, 0)
            );
            assert!((g.get_ret(n, n, 1, 1) + i).abs() < 1e-6);
        }
        let m0 = g.get_mat(0, 0, 0);
        let mb = g.get_mat(ntau as i64, 0, 0);
        assert!(
            (m0 + mb - Complex::new(-1.0, 0.0)).abs() < 1e-6,
            "mat(0) + mat(beta) = {:?}",
            m0 + mb
        );
        let l00 = g.get_less(0, 0, 0, 0);
        assert!(
            (l00 - i * (m0 + Complex::new(1.0, 0.0))).abs() < 1e-6,
            "less(0,0) = {l00:?} vs i(mat(0)+1)"
        );
    }
}
