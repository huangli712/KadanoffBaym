// Project : KadanoffBaym
// File    : src/fourier/adft.rs
// Purpose : adaptive Fourier integral (blueprint: fourier/adft.jl and
//           `dft_cplx`/`adft_func` of fourier/matsubara.jl)

use crate::complex::Complex;
use crate::fourier::dftcor::dft_panel;

/// Panels of the adaptive Fourier integral: `a[i]` and `b[i]` are the
/// bounds of the i-th panel and `f[i]` contains the function values at
/// the `nn + 1` equidistant points of the i-th panel (blueprint
/// `_AdftPanels`).
pub(crate) struct AdftPanels {
    /// Lower bounds of the panels.
    pub a: Vec<f64>,
    /// Upper bounds of the panels.
    pub b: Vec<f64>,
    /// Function values of the panels.
    pub f: Vec<Vec<Complex>>,
}

/// Fourier integral `int_a^b dt e^{i omega t} f(t)` by the cubically
/// corrected DFT (blueprint `dft_cplx`). The function is given at
/// `N + 1` equidistant points, `N` must be even and larger than 16.
/// The error estimate is the difference to the result obtained with
/// every second point.
pub fn dft_cplx(omega: f64, a: f64, b: f64, values: &[Complex]) -> (Complex, Complex) {
    let n = values.len() - 1;
    assert!(
        n % 2 == 0 && n > 16,
        "N = length - 1 must be even and > 16, got {n}"
    );
    assert!(b > a, "b must be larger than a");
    assert!(omega.is_finite(), "omega must be finite");
    dft_panel(omega, a, b, values)
}

/// Bisect the i-th panel (blueprint `_adft_split!`). The function
/// values are reused at the old grid points and evaluated at the new
/// ones.
fn adft_split(p: &mut AdftPanels, i: usize, fz: &impl Fn(f64) -> Complex) {
    let (a0, b0) = (p.a[i], p.b[i]);
    let f0 = std::mem::replace(&mut p.f[i], Vec::new());
    let n = f0.len() - 1;
    let mid = (a0 + b0) / 2.0;
    let delta = (b0 - a0) / (2.0 * n as f64);
    let m = n / 2;

    let mut fl = vec![Complex::ZERO; n + 1];
    let mut fr = vec![Complex::ZERO; n + 1];
    for j in 0..=n {
        if j % 2 == 0 {
            fl[j] = f0[j / 2];
            fr[j] = f0[m + j / 2];
        } else {
            fl[j] = fz(a0 + delta * j as f64);
            fr[j] = fz(mid + delta * j as f64);
        }
    }

    p.b[i] = mid;
    p.f[i] = fl;
    p.a.insert(i + 1, mid);
    p.b.insert(i + 1, b0);
    p.f.insert(i + 1, fr);
}

/// Determine the panels of the adaptive Fourier integral of `fz` on
/// `[lo, hi]`; the panels are selected according to the error estimates
/// at the Fourier variable `w` (blueprint `_adft_sample`).
pub(crate) fn adft_sample(
    fz: impl Fn(f64) -> Complex,
    lo: f64,
    hi: f64,
    nn: i64,
    limit: i64,
    w: f64,
    nmin: i64,
) -> AdftPanels {
    assert!(lo < hi, "lo must be smaller than hi");
    assert!(
        nn % 2 == 0 && nn >= nmin,
        "nn must be even and >= {nmin}, got {nn}"
    );
    assert!(limit >= 1, "limit must be >= 1, got {limit}");

    // sample the first panel
    let delta = (hi - lo) / nn as f64;
    let f: Vec<Complex> = (0..=nn).map(|i| fz(lo + i as f64 * delta)).collect();
    let mut p = AdftPanels {
        a: vec![lo],
        b: vec![hi],
        f: vec![f],
    };
    if limit == 1 {
        return p;
    }

    // split the first panel
    adft_split(&mut p, 0, &fz);
    let mut errs: Vec<Complex> = (0..2)
        .map(|i| dft_panel(w, p.a[i], p.b[i], &p.f[i]).1)
        .collect();

    // split the panel with the largest error estimate
    while p.a.len() + 1 < limit as usize {
        let mut isplit = 0usize;
        let mut emax = -1.0f64;
        for (i, e) in errs.iter().enumerate() {
            if e.abs() > emax {
                isplit = i;
                emax = e.abs();
            }
        }
        adft_split(&mut p, isplit, &fz);
        errs[isplit] = dft_panel(w, p.a[isplit], p.b[isplit], &p.f[isplit]).1;
        let j = isplit + 1;
        let e_new = dft_panel(w, p.a[j], p.b[j], &p.f[j]).1;
        errs.insert(j, e_new);
    }

    p
}

/// Fourier integral `int d eps e^{i omega eps} f(eps)` on the panels
/// (blueprint `_adft_eval`); the error estimate is a diagnostic only.
pub(crate) fn adft_eval(p: &AdftPanels, w: f64) -> (Complex, Complex) {
    let mut res = Complex::ZERO;
    let mut err = Complex::ZERO;
    for i in 0..p.a.len() {
        let (ri, ei) = dft_panel(w, p.a[i], p.b[i], &p.f[i]);
        res += ri;
        err += ei;
    }
    assert!(
        res.re.is_finite() && res.im.is_finite(),
        "the Fourier integral is not finite"
    );
    (res, err)
}

/// Adaptive Fourier integral `int_a^b dt e^{i omega t} f(t)` of a real
/// function (blueprint `adft_func`). Each panel contains `n`
/// subintervals (even and >= 18 in practice); the panel budget is
/// `limit`.
pub fn adft_func(
    f: impl Fn(f64) -> f64,
    omega: f64,
    a: f64,
    b: f64,
    n: i64,
    limit: i64,
) -> (Complex, Complex) {
    assert!(a < b, "a must be smaller than b");
    assert!(limit >= 1, "limit must be >= 1, got {limit}");
    assert!(omega.is_finite(), "omega must be finite");
    let fz = |x: f64| Complex::new(f(x), 0.0);
    let p = adft_sample(fz, a, b, n, limit, omega, 16);
    adft_eval(&p, omega)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_cplx_exponential_panel() {
        // int_0^10 e^{i w t} e^{-t} dt = (1 - e^{-10(1 - i w)}) / (1 - i w)
        let (a, b, n) = (0.0f64, 10.0, 256usize);
        let values: Vec<Complex> = (0..=n)
            .map(|j| Complex::new((-(a + (b - a) * j as f64 / n as f64)).exp(), 0.0))
            .collect();
        for w in [0.0f64, 1.3, -4.1] {
            let (res, err) = dft_cplx(w, a, b, &values);
            let want =
                (Complex::ONE - (-Complex::new(10.0, 0.0) * Complex::new(1.0, -w)).exp()) / Complex::new(1.0, -w);
            assert!(
                (res - want).abs() < 1e-5,
                "w {w}: {res} vs {want} (err {err})"
            );
        }
    }

    #[test]
    fn adft_func_gaussian() {
        // int_{-5}^{5} e^{i w t} e^{-t^2} dt = sqrt(pi) e^{-w^2 / 4}
        // (the truncation error of the gaussian tails is ~1e-11)
        for w in [0.0f64, 1.0, 2.7] {
            let (res, err) = adft_func(|t| (-(t * t)).exp(), w, -5.0, 5.0, 18, 40);
            let want = Complex::new(std::f64::consts::PI.sqrt() * (-w * w / 4.0).exp(), 0.0);
            assert!(
                (res - want).abs() < 1e-7,
                "w {w}: {res} vs {want} (err {err})"
            );
        }
    }

    #[test]
    fn sample_panels_cover_the_interval() {
        let fz = |x: f64| Complex::new((-(x * x)).exp(), 0.0);
        let p = adft_sample(fz, -1.0, 3.0, 18, 6, 0.0, 16);
        // the split loop stops once len + 1 == limit, i.e. limit - 1 panels
        assert_eq!(p.a.len(), 5);
        assert!((p.a[0] - (-1.0)).abs() < 1e-15);
        assert!((p.b[4] - 3.0).abs() < 1e-15);
        for i in 0..4 {
            assert!(
                (p.b[i] - p.a[i + 1]).abs() < 1e-15,
                "panel {i} not contiguous"
            );
        }
        // limit = 1: a single panel
        let p1 = adft_sample(fz, -1.0, 3.0, 18, 1, 0.0, 16);
        assert_eq!(p1.a.len(), 1);
    }
}
