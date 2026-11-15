// Project : KadanoffBaym
// File    : src/equilibrium/single_pole.rs
// Purpose : the bosonic single pole (blueprint
//           julia/src/equilibrium/single_pole.jl)
//
// Split from the former standard.rs (one file per blueprint source).

use crate::complex::Complex;
use crate::types::{ContourGreen, Sign};

use crate::equilibrium::from_ham::bose_exp;
use crate::math::bose;

/// Matsubara component of the bosonic single pole Green's function
/// (blueprint `green_single_pole_mat!`); only the (0,0) element is
/// written.
fn green_single_pole_mat(d0: &mut ContourGreen, w: f64, beta: f64) {
    assert!(d0.sign == Sign::Bose, "D0 must be bosonic");
    assert!(w * beta > 0.0, "w*beta must be positive");
    let dtau = beta / d0.ntau as f64;
    for m in 0..=(d0.ntau as i64) {
        let tau = m as f64 * dtau;
        let ewv = bose_exp(beta, tau, w);
        let emwv = -bose_exp(beta, tau, -w);
        d0.set_mat(m, 0, 0, Complex::new(-0.5 * (ewv + emwv), 0.0));
    }
}

/// Real time components of the bosonic single pole Green's function at
/// the time step `tstp` (0-based; blueprint `green_single_pole_timestep!`).
fn green_single_pole_timestep(tstp: i64, d0: &mut ContourGreen, w: f64, beta: f64, hstep: f64) {
    assert!(d0.sign == Sign::Bose, "D0 must be bosonic");
    assert!(w * beta > 0.0, "w*beta must be positive");
    assert!(0 <= tstp && tstp <= d0.nt as i64, "wrong time step {tstp}");
    let ntau = d0.ntau as i64;
    let dtau = beta / d0.ntau as f64;
    let bw = bose(beta, w);
    let i = Complex::new(0.0, 1.0);

    for tp in 0..=tstp {
        let c = (w * (tstp - tp) as f64 * hstep).cos();
        let s = (w * (tstp - tp) as f64 * hstep).sin();
        let d_les = -i * (Complex::new(c, -s) + Complex::new(2.0 * c * bw, 0.0));
        let d_gtr = d_les;
        let d_ret = d_gtr + d_les.conj();
        d0.set_less(tp, tstp, 0, 0, d_les * Complex::new(0.5, 0.0));
        d0.set_ret(tstp, tp, 0, 0, d_ret * Complex::new(0.5, 0.0));
    }

    let c = (w * tstp as f64 * hstep).cos();
    let s = (w * tstp as f64 * hstep).sin();
    for m in 0..=ntau {
        let ewv = (w * m as f64 * dtau).exp();
        let emwv = (-w * m as f64 * dtau).exp();
        let d_tv = -i
            * (Complex::new(c, s) * emwv
                + (Complex::new(c, s) * emwv + Complex::new(c, -s) * ewv) * bw);
        d0.set_lmix(tstp, m, 0, 0, d_tv * Complex::new(0.5, 0.0));
    }
}

/// All the components of the bosonic single pole Green's function
/// (blueprint `green_single_pole!`).
///
/// *Remarks* : *Bosonic Single Pole*
///
/// For the displacement ``X = (b + b^{\dagger})/\sqrt{2}`` of a harmonic
/// oscillator with frequency ``w``, ``n_B = n_B(w)``,
///
/// ```math
/// \begin{equation}
/// D^{M}(\tau) = -\frac{1}{2}[e^{\tau w} n_B(w) - e^{-\tau w} n_B(-w)],
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// D^{<}(t',t) = -\frac{i}{2}[e^{-iw(t-t')} + 2n_B\cos w(t-t')],
/// \quad
/// D^{R}(t,t') = -\sin w(t-t'),
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// D^{\rceil}(t,\tau) = -\frac{i}{2}\{e^{iwt}e^{-w\tau}
///     + n_B[e^{iwt}e^{-w\tau} + e^{-iwt}e^{w\tau}]\}.
/// \end{equation}
/// ```
///
/// Only the (1,1) element is written.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_single_pole`.
pub fn green_single_pole(d0: &mut ContourGreen, w: f64, beta: f64, hstep: f64) {
    green_single_pole_mat(d0, w, beta);
    for tstp in 0..=(d0.nt as i64) {
        green_single_pole_timestep(tstp, d0, w, beta, hstep);
    }
}

// ---------------------------------------------------------------------------
// Equilibrium Green's functions from a DOS (blueprint from_dos.jl)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Contour;

    /// The bosonic single pole against the closed forms of the
    /// blueprint Remarks, including the periodicity of the Matsubara
    /// component.
    #[test]
    fn single_pole_matches_closed_forms() {
        let (w, beta, hstep, nt, ntau) = (1.3f64, 2.0, 0.05, 20usize, 50usize);
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * hstep, beta);
        let mut d0 = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        green_single_pole(&mut d0, w, beta, hstep);

        let i = Complex::new(0.0, 1.0);
        let bw = bose(beta, w);
        let dtau = beta / ntau as f64;

        // D^M(tau) = -0.5 [e^{tau w} n_B(w) - e^{-tau w} n_B(-w)]
        for m in [0usize, 13, 25, 50] {
            let tau = m as f64 * dtau;
            let want =
                -0.5 * ((w * tau).exp() * bw - (-w * tau).exp() * crate::math::bose(beta, -w));
            assert!((d0.get_mat(m as i64, 0, 0) - Complex::new(want, 0.0)).abs() < 1e-12);
        }
        // bosonic periodicity D^M(beta - tau) = D^M(tau)
        for m in [1usize, 10, 25] {
            let d = d0.get_mat(m as i64, 0, 0) - d0.get_mat((ntau - m) as i64, 0, 0);
            assert!(d.abs() < 1e-12, "periodicity at m = {m}");
        }
        // D^R(t,t') = -sin w (t - t'), D^<(t',t) = -i/2 [e^{-i w dt} + 2 n_B cos w dt]
        for (tstp, tp) in [(0i64, 0), (5, 0), (20, 7), (20, 20)] {
            let dt = (tstp - tp) as f64 * hstep;
            let cw = (w * dt).cos();
            let sw = (w * dt).sin();
            let ret_want = -sw;
            let les_want = -0.5 * i * (Complex::new(cw, -sw) + Complex::new(2.0 * cw * bw, 0.0));
            assert!((d0.get_ret(tstp, tp, 0, 0) - Complex::new(ret_want, 0.0)).abs() < 1e-12);
            assert!((d0.get_less(tp, tstp, 0, 0) - les_want).abs() < 1e-12);
        }
        // D^rangle(t,tau) closed form
        for (tstp, m) in [(0i64, 0usize), (7, 13), (20, 50)] {
            let t = tstp as f64 * hstep;
            let tau = m as f64 * dtau;
            let c = (w * t).cos();
            let s = (w * t).sin();
            let want = -0.5
                * i
                * (Complex::new(c, s) * (-w * tau).exp()
                    + (Complex::new(c, s) * (-w * tau).exp()
                        + Complex::new(c, -s) * (w * tau).exp())
                        * bw);
            assert!((d0.get_lmix(tstp, m as i64, 0, 0) - want).abs() < 1e-12);
        }
    }
}
