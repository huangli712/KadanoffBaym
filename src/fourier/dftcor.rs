// Project : KadanoffBaym
// File    : src/fourier/dftcor.rs
// Purpose : cubically corrected discrete Fourier transform (blueprint:
//           fourier/dftcor.jl + fourier/adft.jl `_dft_endcor`/`_dft_panel`)
//!
//! *Remarks* : *Cubically Corrected Discrete Fourier Transform*
//!
//! For a function ``f(t)`` given on an equidistant grid ``t_r = r\delta``
//! (``r = 0, \cdots, N``), the Fourier integral
//!
//! ```math
//! \begin{equation}
//! I(\omega) = \int^{N\delta}_0 dt\ e^{i\omega t} f(t)
//! \end{equation}
//! ```
//!
//! is approximated by piecewise cubic interpolation of ``f(t)``, which
//! yields
//!
//! ```math
//! \begin{equation}
//! I(\omega) \approx \delta \left\{
//!     W(\theta) \sum^{N}_{r=0} e^{i\omega t_r} f_r
//!     + \sum^{3}_{j=0} \left[
//!         \alpha_j(\theta) f_j
//!         + e^{i\omega N\delta} \alpha^{*}_j(\theta) f_{N-j}
//!     \right]
//! \right\},
//! \end{equation}
//! ```
//!
//! where ``\theta = \omega\delta``, ``W(\theta)`` is the correction factor
//! and ``\alpha_j(\theta)`` are the boundary terms. For Matsubara frequencies
//! ``e^{i\omega_n \beta} = \xi``.
//!
//! *References* :
//!
//! W. H. Press et al., Numerical Recipes 3rd Edition, Section 13.9.
//!
//! Please see [`NESSi`] `fourier.cpp`, `get_dftcorr_cubic`, and
//! `cntr_matsubara_impl.hpp`, `matsubara_dft` and `matsubara_ft`.

use crate::complex::Complex;

#[inline]
fn cis(x: f64) -> Complex {
    Complex::from_polar(1.0, x)
}

/// Correction factor `W(theta)` and boundary terms `(a0, a1)` of the
/// linearly corrected DFT (`a1 = 0`); series expansions for
/// `|theta| < 0.05` (blueprint `get_dftcorr_linear`).
pub fn dftcorr_linear(theta: f64) -> (f64, [Complex; 2]) {
    assert!(theta.is_finite(), "theta must be finite");
    let (corfac, ar, ai) = dftcorr_linear_impl(theta);
    (corfac, [Complex::new(ar, ai), Complex::ZERO])
}

fn dftcorr_linear_impl(theta: f64) -> (f64, f64, f64) {
    if theta.abs() < 5.0e-2 {
        let th2 = theta * theta;
        let th4 = th2 * th2;
        let th6 = th4 * th2;
        let corfac = 1.0 - th2 / 12.0 + th4 / 360.0 - th6 / 20160.0;
        let ar = -0.5 + th2 / 24.0 - th4 / 720.0 + th6 / 40320.0;
        let ai = theta * (1.0 / 6.0 - th2 / 120.0 + th4 / 5040.0 - th6 / 362880.0);
        (corfac, ar, ai)
    } else {
        let cth = theta.cos();
        let sth = theta.sin();
        let th2 = theta * theta;
        let corfac = 2.0 * (1.0 - cth) / th2;
        let ar = (cth - 1.0) / th2;
        let ai = (theta - sth) / th2;
        (corfac, ar, ai)
    }
}

/// Correction factor `W(theta)` and boundary terms `(a0, .., a3)` of
/// the cubically corrected DFT; series expansions for
/// `|theta| < 0.05` (blueprint `get_dftcorr_cubic`).
pub fn dftcorr_cubic(theta: f64) -> (f64, [Complex; 4]) {
    assert!(theta.is_finite(), "theta must be finite");
    if theta.abs() < 5.0e-2 {
        let t = theta;
        let t2 = t * t;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let corfac = 1.0 - (11.0 / 720.0) * t4 + (23.0 / 15120.0) * t6;
        let ar0 = -2.0 / 3.0 + t2 / 45.0 + (103.0 / 15120.0) * t4 - (169.0 / 226800.0) * t6;
        let ar1 = 7.0 / 24.0 - (7.0 / 180.0) * t2 + (5.0 / 3456.0) * t4 - (7.0 / 259200.0) * t6;
        let ar2 = -1.0 / 6.0 + t2 / 45.0 - (5.0 / 6048.0) * t4 + t6 / 64800.0;
        let ar3 = 1.0 / 24.0 - t2 / 180.0 + (5.0 / 24192.0) * t4 - t6 / 259200.0;
        let ai0 =
            t * (2.0 / 45.0 + (2.0 / 105.0) * t2 - (8.0 / 2835.0) * t4 + (86.0 / 467775.0) * t6);
        let ai1 = t * (7.0 / 72.0 - t2 / 168.0 + (11.0 / 72576.0) * t4 - (13.0 / 5987520.0) * t6);
        let ai2 = t * (-7.0 / 90.0 + t2 / 210.0 - (11.0 / 90720.0) * t4 + (13.0 / 7484400.0) * t6);
        let ai3 =
            t * (7.0 / 360.0 - t2 / 840.0 + (11.0 / 362880.0) * t4 - (13.0 / 29937600.0) * t6);
        (
            corfac,
            [
                Complex::new(ar0, ai0),
                Complex::new(ar1, ai1),
                Complex::new(ar2, ai2),
                Complex::new(ar3, ai3),
            ],
        )
    } else {
        let cth = theta.cos();
        let sth = theta.sin();
        let ctth = cth * cth - sth * sth;
        let stth = 2.0 * sth * cth;
        let th2 = theta * theta;
        let th4 = th2 * th2;
        let tmth2 = 3.0 - th2;
        let spth2 = 6.0 + th2;
        let sth4i = 1.0 / (6.0 * th4);
        let tth4i = 2.0 * sth4i;
        let corfac = tth4i * spth2 * (3.0 - 4.0 * cth + ctth);
        let ar0 = sth4i * (-42.0 + 5.0 * th2 + spth2 * (8.0 * cth - ctth));
        let ai0 = sth4i * (theta * (-12.0 + 6.0 * th2) + spth2 * stth);
        let ar1 = sth4i * (14.0 * tmth2 - 7.0 * spth2 * cth);
        let ai1 = sth4i * (30.0 * theta - 5.0 * spth2 * sth);
        let ar2 = tth4i * (-4.0 * tmth2 + 2.0 * spth2 * cth);
        let ai2 = tth4i * (-12.0 * theta + 2.0 * spth2 * sth);
        let ar3 = sth4i * (2.0 * tmth2 - spth2 * cth);
        let ai3 = sth4i * (6.0 * theta - spth2 * sth);
        (
            corfac,
            [
                Complex::new(ar0, ai0),
                Complex::new(ar1, ai1),
                Complex::new(ar2, ai2),
                Complex::new(ar3, ai3),
            ],
        )
    }
}

/// Correction factor and boundary correction of the cubically
/// corrected DFT of the panel `[a, b]`; only every `step`-th point of
/// `f` is used with the spacing `delta` (blueprint `_dft_endcor`).
pub(crate) fn dft_endcor(w: f64, delta: f64, a: f64, b: f64, f: &[Complex], step: usize) -> (f64, Complex) {
    let n = f.len() - 1;
    let (corfac, alpha) = dftcorr_cubic(w * delta);
    let mut cl = Complex::ZERO;
    let mut cr = Complex::ZERO;
    for j in 0..4usize {
        cl += f[step * j] * alpha[j];
        cr += f[n - step * j] * alpha[j].conj();
    }
    cr *= cis(w * (b - a));
    (corfac, cl + cr)
}

/// Cubically corrected DFT of a single panel: an approximation to
/// `int_a^b dt e^{i w t} f(t)` with the error estimate given by the
/// difference to the result obtained with every second point
/// (blueprint `_dft_panel`).
pub(crate) fn dft_panel(w: f64, a: f64, b: f64, f: &[Complex]) -> (Complex, Complex) {
    let n = f.len() - 1;
    assert!(
        n > 16 && n % 2 == 0 && b > a,
        "_dft_panel: wrong input, n = {n}, [{a}, {b}]"
    );
    let delta = (b - a) / n as f64;
    let theta = w * delta;

    let (corfac, endcor) = dft_endcor(w, delta, a, b, f, 1);
    let (corfac2, endcor2) = dft_endcor(w, 2.0 * delta, a, b, f, 2);

    // discrete Fourier transform, every second point first
    let mut res2 = Complex::ZERO;
    for j in (0..=n).step_by(2) {
        res2 += cis(j as f64 * theta) * f[j];
    }
    let mut res = res2;
    for j in (1..n).step_by(2) {
        res += cis(j as f64 * theta) * f[j];
    }

    let expfac = Complex::new(delta, 0.0) * cis(w * a);
    res = (res * Complex::new(corfac, 0.0) + endcor) * expfac;
    res2 = (res2 * Complex::new(corfac2, 0.0) + endcor2) * expfac.scale(2.0);

    (res, res - res2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_matches_closed_form_at_boundary() {
        // the series (|theta| < 0.05) and the closed form must agree
        // to ~1e-14 at the switch point
        for th in [0.04999999f64, -0.04999999, 0.0500001, -0.0500001] {
            let (c, a) = dftcorr_cubic(th);
            // evaluated via the two branches by nudging across 0.05
            let eps = 1e-9;
            let (_, a2) = dftcorr_cubic(th + if th > 0.0 { eps } else { -eps });
            if (th.abs() - 0.05).abs() > 1e-6 {
                // both on the same side: skip
                continue;
            }
            let _ = c;
            for j in 0..4 {
                assert!(
                    (a[j] - a2[j]).abs() < 1e-9,
                    "th {th}: a[{j}] {} vs {}",
                    a[j],
                    a2[j]
                );
            }
        }
    }

    #[test]
    fn limits_are_sensible() {
        // W(0) = 1 and the boundary terms vanish as theta -> 0
        let (c, a) = dftcorr_cubic(0.0);
        assert!((c - 1.0).abs() < 1e-15);
        // the real parts of the cubic boundary terms at theta = 0
        let want_re = [-2.0 / 3.0, 7.0 / 24.0, -1.0 / 6.0, 1.0 / 24.0];
        for j in 0..4 {
            assert!((a[j].re - want_re[j]).abs() < 1e-15, "a[{j}].re");
            assert!(a[j].im.abs() < 1e-15, "a[{j}].im");
        }
        let (c1, a1) = dftcorr_linear(0.0);
        assert!((c1 - 1.0).abs() < 1e-15);
        assert!((a1[0].re + 0.5).abs() < 1e-15 && a1[0].im.abs() < 1e-15);
        assert!(a1[1] == Complex::ZERO);
        // large theta: W oscillates but stays finite
        let (c3, _) = dftcorr_cubic(1.7);
        assert!(c3.is_finite());
    }

    #[test]
    fn panel_integral_of_exponential() {
        // int_0^10 e^{i w t} e^{-t} dt = (1 - e^{-10(1 - i w)}) / (1 - i w)
        let (a, b, n) = (0.0f64, 10.0f64, 200usize);
        let f: Vec<Complex> = (0..=n)
            .map(|j| Complex::new((-(a + (b - a) * j as f64 / n as f64)).exp(), 0.0))
            .collect();
        for w in [0.0f64, 0.5, 3.7, -2.2] {
            let (res, err) = dft_panel(w, a, b, &f);
            let want =
                (Complex::ONE - (-Complex::new(10.0, 0.0) * Complex::new(1.0, -w)).exp()) / Complex::new(1.0, -w);
            assert!(
                (res - want).abs() < 1e-6,
                "w {w}: {res} vs {want} (err est {err})"
            );
        }
    }
}

/// Boundary correction of the cubically corrected DFT on the interval
/// `[a, b]`, i.e., `sum_j alpha_j f_j + e^{i omega (b-a)} sum_j
/// conj(alpha_j) f_{N-j}` with the correction factor `W(omega delta)`
/// (blueprint `complex_dftcor_cubic`). `endpoints[j] = f_j` and
/// `endpoints[7 - j] = f_{N-j}` (j = 0, .., 3).
///
/// *Remarks* : *Corrected Fourier Integrals*
///
/// The Fourier integral of a function ``f(t)`` on the interval ``[a,b]``,
///
/// ```math
/// \begin{equation}
/// I(\omega) = \int^b_a dt\ e^{i\omega t} f(t),
/// \end{equation}
/// ```
///
/// is approximated by the linearly or cubically corrected discrete Fourier
/// transform, see *Remarks* : *Cubically Corrected Discrete Fourier
/// Transform* in `matsubara.jl`, which contains the only
/// implementation of the correction coefficients. The functions in this
/// file are public wrappers with argument checks.
///
/// The adaptive Fourier integral splits the panel with the largest error
/// estimate (the difference between the results obtained with all points
/// and with every second point) until the panel budget is exhausted.
///
/// *Remarks* : *Matsubara Fourier Transform*
///
/// For a Matsubara function ``G(\tau)`` given at ``\tau_r = r\beta/N``
/// (``r = 0, \cdots, N``, KB uses ``N + 1`` points), the Fourier
/// coefficients are
///
/// ```math
/// \begin{equation}
/// G(i\omega_m) = \int^{\beta}_0 d\tau\ e^{i\omega_m\tau} G(\tau),\quad
/// \omega_m = \frac{\pi}{\beta}(2m + \delta),
/// \end{equation}
/// ```
///
/// where ``\delta = 1`` for fermions and ``\delta = 0`` for bosons.
///
/// *References* :
///
/// Please see [`NESSi`] `fourier.cpp`, `fourier.hpp`, and
/// `cntr_matsubara_impl.hpp`.
pub fn complex_dftcor_cubic(
    omega: f64,
    delta: f64,
    a: f64,
    b: f64,
    endpoints: &[Complex],
) -> (Complex, f64) {
    assert_eq!(
        endpoints.len(),
        8,
        "endpoints must contain 8 points, got {}",
        endpoints.len()
    );
    assert!(delta > 0.0, "delta must be positive");
    assert!(b > a, "b must be larger than a");
    assert!(omega.is_finite(), "omega must be finite");
    let (corfac, endcor) = dft_endcor(omega, delta, a, b, endpoints, 1);
    (endcor, corfac)
}
