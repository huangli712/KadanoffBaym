// Project : KadanoffBaym
// File    : tests/fourier.rs
// Purpose : port of julia/test/t_fourier.jl — corrected Fourier
//           integrals (correction coefficients, complex_dftcor_cubic,
//           dft_cplx, adft_func) against analytic results

mod common;

use common::prelude::*;
use kadanoffbaym::fourier::{
    adft_func, complex_dftcor_cubic, dft_cplx, dftcorr_cubic, dftcorr_linear, fft, FftDirection,
};

fn cis(x: f64) -> Complex {
    Complex::from_polar(1.0, x)
}

/// Linearly corrected DFT built from the public correction
/// coefficients (blueprint `fo_linear`).
fn fo_linear(omega: f64, a: f64, b: f64, v: &[Complex]) -> Complex {
    let n = v.len() - 1;
    let delta = (b - a) / n as f64;
    let (w, alpha) = dftcorr_linear(omega * delta);
    let s: Complex = (0..=n).fold(Complex::ZERO, |acc, j| acc + cis(j as f64 * omega * delta) * v[j]);
    Complex::new(delta, 0.0)
        * cis(omega * a)
        * (s * Complex::new(w, 0.0) + alpha[0] * v[0] + cis(omega * (b - a)) * alpha[0].conj() * v[n])
}

/// Cubically corrected DFT without the endpoint correction (mutant,
/// blueprint `fo_mutant`).
fn fo_mutant(omega: f64, a: f64, b: f64, v: &[Complex]) -> Complex {
    let n = v.len() - 1;
    let delta = (b - a) / n as f64;
    let (w, _) = dftcorr_cubic(omega * delta);
    let s: Complex = (0..=n).fold(Complex::ZERO, |acc, j| acc + cis(j as f64 * omega * delta) * v[j]);
    Complex::new(delta, 0.0) * cis(omega * a) * s * Complex::new(w, 0.0)
}

fn fo_exact_exp(omega: f64, a: f64, b: f64) -> Complex {
    ((Complex::new(-1.0, omega) * Complex::new(b, 0.0)).exp()
        - (Complex::new(-1.0, omega) * Complex::new(a, 0.0)).exp())
        / Complex::new(-1.0, omega)
}

fn fo_exact_cexp(omega: f64, nu: f64, a: f64, b: f64) -> Complex {
    if omega + nu == 0.0 {
        Complex::new(b - a, 0.0)
    } else {
        (cis((omega + nu) * b) - cis((omega + nu) * a)) / Complex::new(0.0, omega + nu)
    }
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn correction_coefficients_finite_and_continuous() {
    for t in [0.0f64, 1e-8, -1e-8, 0.049999, 0.050001, -0.049999, -0.050001, 0.7, -0.7, 3.0] {
        let (w, a) = dftcorr_cubic(t);
        assert!(w.is_finite() && a.len() == 4 && a.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let (w1, a1) = dftcorr_linear(t);
        assert!(w1.is_finite() && a1.len() == 2 && a1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        assert_eq!(a1[1], Complex::ZERO);
    }
    // continuity at |theta| = 0.05
    for s in [1.0f64, -1.0] {
        let (wm, am) = dftcorr_cubic(s * (0.05 - 1e-12));
        let (wp, ap) = dftcorr_cubic(s * (0.05 + 1e-12));
        assert!((wm - wp).abs() < 1e-9);
        for j in 0..4 {
            assert!((am[j] - ap[j]).abs() < 1e-9, "cubic s {s} a[{j}]");
        }
        let (wm, am) = dftcorr_linear(s * (0.05 - 1e-12));
        let (wp, ap) = dftcorr_linear(s * (0.05 + 1e-12));
        assert!((wm - wp).abs() < 1e-9);
        assert!((am[0] - ap[0]).abs() < 1e-9, "linear s {s}");
    }
    // theta = 0: W = 1, trapezoidal rule for the linear correction
    assert_eq!(dftcorr_cubic(0.0).0, 1.0);
    let (w1, a1) = dftcorr_linear(0.0);
    assert_eq!(w1, 1.0);
    assert_eq!(a1[0], Complex::new(-0.5, 0.0));
    assert_eq!(a1[1], Complex::ZERO);
}

#[test]
#[should_panic]
fn fail_dftcorr_cubic_nan() {
    let _ = dftcorr_cubic(f64::NAN);
}

#[test]
#[should_panic]
fn fail_dftcorr_linear_inf() {
    let _ = dftcorr_linear(f64::INFINITY);
}

#[test]
fn complex_dftcor_cubic_omega_zero() {
    let e: Vec<Complex> = (1..=8).map(|i| Complex::new(i as f64, 0.0)).collect();
    let (c, w) = complex_dftcor_cubic(0.0, 0.1, 0.0, 0.7, &e);
    assert!(c.re.is_finite() && c.im.is_finite() && w.is_finite());
    // at omega = 0: alpha = (-2/3, 7/24, -1/6, 1/24), both ends real
    let alpha = [-2.0 / 3.0, 7.0 / 24.0, -1.0 / 6.0, 1.0 / 24.0];
    let mut want = Complex::ZERO;
    for j in 0..4usize {
        want += Complex::new(alpha[j], 0.0) * (e[j] + e[7 - j]);
    }
    assert!((c - want).abs() < 1e-14, "{c} vs {want}");
}

#[test]
#[should_panic]
fn fail_complex_dftcor_seven_endpoints() {
    let e: Vec<Complex> = (1..=7).map(|i| Complex::new(i as f64, 0.0)).collect();
    let _ = complex_dftcor_cubic(0.0, 0.1, 0.0, 0.7, &e);
}

#[test]
#[should_panic]
fn fail_complex_dftcor_zero_delta() {
    let e: Vec<Complex> = (1..=8).map(|i| Complex::new(i as f64, 0.0)).collect();
    let _ = complex_dftcor_cubic(0.0, 0.0, 0.0, 0.7, &e);
}

#[test]
#[should_panic]
fn fail_complex_dftcor_equal_bounds() {
    let e: Vec<Complex> = (1..=8).map(|i| Complex::new(i as f64, 0.0)).collect();
    let _ = complex_dftcor_cubic(0.0, 0.1, 0.7, 0.7, &e);
}

#[test]
fn dft_cplx_analytic_integrals() {
    let (a, b) = (0.0f64, 4.0);
    // exponential
    let t = linspace(a, b, 65);
    for omega in [0.0f64, 0.7, -2.3, 10.0] {
        let v: Vec<Complex> = t.iter().map(|&x| Complex::new((-x).exp(), 0.0)).collect();
        let (r, e) = dft_cplx(omega, a, b, &v);
        let ex = fo_exact_exp(omega, a, b);
        assert!((r - ex).abs() < 1e-6, "omega {omega}: {r} vs {ex}");
        assert!((r - ex).abs() <= e.abs(), "estimate not conservative, omega {omega}");
    }
    // constant is exact up to rounding
    let ones = vec![Complex::ONE; 65];
    assert!((dft_cplx(0.0, a, b, &ones).0 - Complex::new(4.0, 0.0)).abs() < 1e-13);
    assert!((dft_cplx(1.3, a, b, &ones).0 - fo_exact_cexp(1.3, 0.0, a, b)).abs() < 1e-6);
    // two frequencies
    let t = linspace(a, b, 129);
    let v: Vec<Complex> = t
        .iter()
        .map(|&x| cis(2.0 * x) + cis(-3.0 * x).scale(0.5))
        .collect();
    let ex = fo_exact_cexp(0.4, 2.0, a, b) + fo_exact_cexp(0.4, -3.0, a, b).scale(0.5);
    assert!((dft_cplx(0.4, a, b, &v).0 - ex).abs() < 1e-5);
}

#[test]
#[should_panic]
fn fail_dft_cplx_seventeen() {
    let _ = dft_cplx(0.0, 0.0, 4.0, &vec![Complex::ONE; 17]);
}

#[test]
#[should_panic]
fn fail_dft_cplx_odd() {
    let _ = dft_cplx(0.0, 0.0, 4.0, &vec![Complex::ONE; 18]);
}

#[test]
#[should_panic]
fn fail_dft_cplx_reversed_bounds() {
    let _ = dft_cplx(0.0, 4.0, 0.0, &vec![Complex::ONE; 19]);
}

#[test]
#[should_panic]
fn fail_dft_cplx_nan_omega() {
    let _ = dft_cplx(f64::NAN, 0.0, 4.0, &vec![Complex::ONE; 19]);
}

#[test]
fn dft_cplx_nineteen_works() {
    let (a, b) = (0.0f64, 4.0);
    assert!((dft_cplx(0.0, a, b, &vec![Complex::ONE; 19]).0 - Complex::new(4.0, 0.0)).abs() < 1e-12);
}

#[test]
fn cubic_beats_linear() {
    let (a, b, omega) = (0.0f64, 1.0, 0.7);
    let ex = fo_exact_exp(omega, a, b);
    for n in [32usize, 64] {
        let t = linspace(a, b, n + 1);
        let v: Vec<Complex> = t.iter().map(|&x| Complex::new((-x).exp(), 0.0)).collect();
        let ec = (dft_cplx(omega, a, b, &v).0 - ex).abs();
        let el = (fo_linear(omega, a, b, &v) - ex).abs();
        assert!(ec < 1e-3 * el, "n {n}: cubic {ec} vs linear {el}");
    }
}

#[test]
fn plain_dft_comparison_with_fft() {
    // at the grid frequencies omega = 2 pi k / (N delta) the plain sum
    // is a DFT
    let (n, a, b) = (32usize, 0.0f64, 2.0);
    let delta = (b - a) / n as f64;
    let t = linspace(a, b, n + 1);
    let v: Vec<Complex> = t
        .iter()
        .map(|&x| {
            Complex::new((-x).exp() * (3.0 * x).cos(), 0.0) + Complex::new(0.0, x)
        })
        .collect();
    let first: Vec<Complex> = v[..n].to_vec();
    let y = fft(&first, FftDirection::Backward);
    for k in [0usize, 3, 17] {
        let omega = 2.0 * std::f64::consts::PI * k as f64 / (n as f64 * delta);
        let (r, _) = dft_cplx(omega, a, b, &v);
        let mut endpoints = Vec::new();
        endpoints.extend_from_slice(&v[0..4]);
        endpoints.extend_from_slice(&v[n - 3..=n]);
        let (c, w) = complex_dftcor_cubic(omega, delta, a, b, &endpoints);
        let plain = (r / (Complex::new(delta, 0.0) * cis(omega * a)) - c) / Complex::new(w, 0.0);
        assert!(
            (plain - (y[k] + v[n])).abs() < 1e-11 * n as f64,
            "k {k}: {plain} vs {}",
            y[k] + v[n]
        );
    }
}

#[test]
fn adaptive_fourier_integral() {
    let ex = Complex::new(std::f64::consts::PI.sqrt() * (-0.7f64 * 0.7 / 4.0).exp(), 0.0);
    let f = |x: f64| (-x * x).exp();
    let (r, e) = adft_func(f, 0.7, -5.0, 5.0, 32, 64);
    assert!((r - ex).abs() < 1e-10, "{r} vs {ex}");
    assert!(e.re.is_finite() && e.im.is_finite());
    // more panels are more accurate
    let (r1, _) = adft_func(f, 0.7, -5.0, 5.0, 18, 3);
    let (r2, _) = adft_func(f, 0.7, -5.0, 5.0, 18, 40);
    assert!((r2 - ex).abs() < (r1 - ex).abs());
    // a single panel is dft_cplx on the whole interval
    let t = linspace(-5.0, 5.0, 19);
    let v: Vec<Complex> = t.iter().map(|&x| Complex::new((-x * x).exp(), 0.0)).collect();
    let (r_single, _) = adft_func(f, 0.7, -5.0, 5.0, 18, 1);
    assert!((r_single - dft_cplx(0.7, -5.0, 5.0, &v).0).abs() < 1e-14);
    // the default arguments are executable
    let (r, _) = adft_func(f, 0.7, -5.0, 5.0, 18, 100);
    assert!((r - ex).abs() < 1e-10);
}

#[test]
#[should_panic]
fn fail_adft_reversed_bounds() {
    let _ = adft_func(|x| (-x * x).exp(), 0.7, 5.0, -5.0, 18, 10);
}

#[test]
#[should_panic]
fn fail_adft_n_too_small() {
    let _ = adft_func(|x| (-x * x).exp(), 0.7, -5.0, 5.0, 14, 10);
}

#[test]
#[should_panic]
fn fail_adft_n_odd() {
    let _ = adft_func(|x| (-x * x).exp(), 0.7, -5.0, 5.0, 19, 10);
}

#[test]
#[should_panic]
fn fail_adft_zero_limit() {
    let _ = adft_func(|x| (-x * x).exp(), 0.7, -5.0, 5.0, 20, 0);
}

#[test]
#[should_panic]
fn fail_adft_n16_needs_correction() {
    // n = 16 passes the sampler but the corrected DFT requires n > 16
    let _ = adft_func(|x| (-x * x).exp(), 0.7, -5.0, 5.0, 16, 4);
}

#[test]
fn release_convergence_order_gates() {
    if !common::is_release() {
        return;
    }
    let (a, b) = (0.0f64, 1.0);
    let ns: Vec<i64> = vec![32, 64, 128, 256];
    for omega in [0.7f64, 5.0] {
        let ex = fo_exact_exp(omega, a, b);
        let grid = |n: usize| -> Vec<Complex> {
            let t = linspace(a, b, n + 1);
            t.iter().map(|&x| Complex::new((-x).exp(), 0.0)).collect()
        };
        let ec: Vec<f64> = ns
            .iter()
            .map(|&n| (dft_cplx(omega, a, b, &grid(n as usize)).0 - ex).abs())
            .collect();
        let el: Vec<f64> = ns
            .iter()
            .map(|&n| (fo_linear(omega, a, b, &grid(n as usize)) - ex).abs())
            .collect();
        let em: Vec<f64> = ns
            .iter()
            .map(|&n| (fo_mutant(omega, a, b, &grid(n as usize)) - ex).abs())
            .collect();
        assert!(common::check_order_gate(&ns, &ec, 4.0, 1e-10, 1e-13, 0.3).ok, "cubic {omega}");
        assert!(common::check_order_gate(&ns, &el, 2.0, 1e-5, 1e-13, 0.3).ok, "linear {omega}");
        // the mutant without the endpoint correction is rejected
        assert!(!common::check_order_gate(&ns, &em, 4.0, 1e-10, 1e-13, 0.3).ok, "mutant {omega}");
        // the error estimate decreases with the grid
        let ee: Vec<f64> = ns
            .iter()
            .map(|&n| dft_cplx(omega, a, b, &grid(n as usize)).1.abs())
            .collect();
        for w in ee.windows(2) {
            assert!(w[1] < w[0], "estimate not decreasing: {ee:?}");
        }
    }
}
