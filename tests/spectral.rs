// Project : KadanoffBaym
// File    : tests/spectral.rs
// Purpose : blueprint t_spectral.jl: the Wigner slices and the
//           windowed spectral functions
//
// Known deviations: the invalid method name (`:spline`) and the
// scalar-returning window of the blueprint are unrepresentable with
// the enum/closure signatures of the Rust API.

mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::observables::{evaluate_spectrum_windowed, wigner_slices, SpectrumMethod};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::ContourGreen;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn sp_contour(ntime: i64, d: i64, h: f64) -> Contour {
    Contour::new(ntime - 1, 4, d, d, (ntime - 1) as f64 * h, 1.0)
}

/// Damped single poles `G^R_ii(t) = -i e^{-i eps_i t - eta t}` for
/// `t >= 0`, lesser `0.3i e^{i eps Delta}`.
fn sp_pole_gf(ntime: i64, h: f64, eps: &[f64], eta: f64) -> ContourGreen {
    let d = eps.len();
    let mut g = ContourGreen::new(&sp_contour(ntime, d as i64, h), Sign::Fermi, Complex::ZERO);
    for t1 in 0..ntime {
        for t2 in 0..=t1 {
            let delta = (t1 - t2) as f64 * h;
            let mut r = vec![Complex::ZERO; d * d];
            let mut l = vec![Complex::ZERO; d * d];
            for (i, e) in eps.iter().enumerate() {
                r[i * d + i] = c(0.0, -1.0)
                    * (-(eta * delta)).exp()
                    * Complex::from_polar(1.0, -e * delta);
                l[i * d + i] = c(0.0, 0.3) * Complex::from_polar(1.0, e * delta);
            }
            g.ret.raw_block_at_mut(t1, t2).copy_from_slice(&r);
            g.less.raw_block_at_mut(t2, t1).copy_from_slice(&l);
        }
    }
    g
}

fn lorentz(w: f64, eps: f64, eta: f64) -> f64 {
    eta / (std::f64::consts::PI * ((w - eps) * (w - eps) + eta * eta))
}

fn ones_window() -> impl Fn(&[f64]) -> Vec<Complex> {
    |t: &[f64]| vec![Complex::ONE; t.len()]
}

#[test]
fn spectral_wigner_slices() {
    let (ntime, h) = (21i64, 0.1f64);
    let mut g = ContourGreen::new(&sp_contour(ntime, 2, h), Sign::Fermi, Complex::ZERO);
    for t1 in 0..ntime {
        for t2 in 0..=t1 {
            g.ret.raw_block_at_mut(t1, t2).fill(c((t1 - t2) as f64, 0.0));
            g.less.raw_block_at_mut(t2, t1).fill(c(0.0, (t1 + t2 + 2) as f64));
        }
    }
    for tav_kb in [1i64, 5, 11, 15, 21] {
        let tav = tav_kb - 1;
        let (t_rel, ret, les) = wigner_slices(&g, tav, h);
        let nj = (tav.min(ntime - 1 - tav) + 1) as usize;
        assert_eq!(t_rel.len(), nj);
        assert_eq!(ret.len(), nj);
        assert_eq!(les.len(), nj);
        for j in 0..nj {
            assert_eq!(t_rel[j], 2.0 * j as f64 * h);
            assert_eq!(ret[j][0 * 2 + 1], c(2.0 * j as f64, 0.0));
            if j >= 1 {
                // G^<(t1,t2) = -G^<(t2,t1)^dagger
                assert_eq!(les[j][1 * 2 + 0], -c(0.0, (2 * tav_kb) as f64).conj());
            }
        }
        assert_eq!(les[0][0], c(0.0, (2 * tav_kb) as f64));
    }
    assert!(panics(|| wigner_slices(&g, -1, h)));
    assert!(panics(|| wigner_slices(&g, ntime, h)));
    assert!(panics(|| wigner_slices(&g, 3, 0.0)));
    let cont2 = Contour::new(ntime - 1, 4, 2, 3, (ntime - 1) as f64 * h, 1.0);
    let gr = ContourGreen::new(&cont2, Sign::Fermi, Complex::ZERO);
    assert!(panics(|| wigner_slices(&gr, 3, h)));
}

#[test]
fn spectral_damped_poles_identity_window() {
    let (eta, h) = (0.5f64, 0.02f64);
    let eps = [0.5f64, -1.0];
    let ntime = 2001i64;
    let g = sp_pole_gf(ntime, h, &eps, eta);
    let tav = 1000; // KB 1001
    let (t_rel, ret, _) = wigner_slices(&g, tav, h);
    assert_eq!(t_rel.len(), 1001);
    let dw = 0.01f64;
    let _ = dw;
    // the blueprint's `collect(-2.0:0.01:2.0)`: the midpoint is exactly 0
    let omega: Vec<f64> = (0..401).map(|k| -2.0 + k as f64 * dw).collect();
    for &method in &[SpectrumMethod::Cubic, SpectrumMethod::Linear] {
        let a = evaluate_spectrum_windowed(&ret, &t_rel, &omega, method, &|t: &[f64]| vec![Complex::ONE; t.len()]);
        for i in 0..2usize {
            // peak position
            let k = a.iter().enumerate().max_by(|x, y| x.1[i].total_cmp(&y.1[i])).unwrap().0;
            assert!((omega[k] - eps[i]).abs() <= dw);
            let kc = omega
                .iter()
                .enumerate()
                .min_by(|x, y| (x.1 - eps[i]).abs().total_cmp(&(y.1 - eps[i]).abs()))
                .unwrap()
                .0;
            let want = lorentz(omega[kc], eps[i], eta);
            assert!((a[kc][i] - want).abs() / want < 1e-2);
            let mut mx = 0.0f64;
            for (kk, w) in omega.iter().enumerate() {
                mx = mx.max((a[kk][i] - lorentz(*w, eps[i], eta)).abs());
            }
            assert!(mx * std::f64::consts::PI * eta < 1e-2);
        }
    }
}

#[test]
fn spectral_windows_converge() {
    let (eta, h) = (0.3f64, 0.02f64);
    let eps = [0.3f64];
    let ntime = 4001i64;
    let g = sp_pole_gf(ntime, h, &eps, eta);
    let (t_rel, ret, _) = wigner_slices(&g, 2000, h);
    let tmax = t_rel[t_rel.len() - 1];
    // the blueprint's `collect(-0.5:0.01:1.1)` (161 points)
    let omega: Vec<f64> = (0..161).map(|k| -0.5 + k as f64 * 0.01).collect();
    let hann = |ell: f64| -> Box<dyn Fn(&[f64]) -> Vec<Complex>> {
        Box::new(move |t: &[f64]| {
            t.iter()
                .map(|x| {
                    if *x <= ell {
                        c(0.5 * (1.0 + (std::f64::consts::PI * x / ell).cos()), 0.0)
                    } else {
                        Complex::ZERO
                    }
                })
                .collect()
        })
    };
    let gauss = |sigma: f64| -> Box<dyn Fn(&[f64]) -> Vec<Complex>> {
        Box::new(move |t: &[f64]| {
            t.iter()
                .map(|x| {
                    let y = x / sigma;
                    c((-(y * y)).exp(), 0.0)
                })
                .collect()
        })
    };
    let makers: Vec<Box<dyn Fn(f64) -> Box<dyn Fn(&[f64]) -> Vec<Complex>>>> =
        vec![Box::new(hann), Box::new(gauss)];
    for make_win in &makers {
        let mut errs = Vec::new();
        for &ell in &[tmax / 8.0, tmax / 4.0, tmax / 2.0] {
            let w = make_win(ell);
            let a = evaluate_spectrum_windowed(&ret, &t_rel, &omega, SpectrumMethod::Linear, w.as_ref());
            let k = a.iter().enumerate().max_by(|x, y| x.1[0].total_cmp(&y.1[0])).unwrap().0;
            assert!((omega[k] - eps[0]).abs() <= 0.01);
            let kc = omega
                .iter()
                .enumerate()
                .min_by(|x, y| (x.1 - eps[0]).abs().total_cmp(&(y.1 - eps[0]).abs()))
                .unwrap()
                .0;
            errs.push((a[kc][0] - lorentz(omega[kc], eps[0], eta)).abs());
        }
        assert!(errs[2] < errs[1] && errs[1] < errs[0], "{errs:?}");
    }
    // complex window: a phase e^{i dt} shifts the peak by -0.2
    let a = evaluate_spectrum_windowed(
        &ret,
        &t_rel,
        &omega,
        SpectrumMethod::Cubic,
        &|t: &[f64]| t.iter().map(|x| Complex::from_polar(1.0, 0.2 * x)).collect(),
    );
    let k = a.iter().enumerate().max_by(|x, y| x.1[0].total_cmp(&y.1[0])).unwrap().0;
    assert!((omega[k] - (eps[0] - 0.2)).abs() <= 0.01);
}

#[test]
fn spectral_boundary_slices() {
    let (eta, h) = (0.5f64, 0.05f64);
    let eps = [0.4f64];
    let ntime = 41i64;
    let g = sp_pole_gf(ntime, h, &eps, eta);
    let omega = [-1.0f64, 0.0, 0.4, 1.0];
    let one = ones_window();
    // the slice closest to the boundary with two points (KB tav 2)
    let (t_rel, ret, _) = wigner_slices(&g, 1, h);
    assert_eq!(t_rel.len(), 2);
    let a = evaluate_spectrum_windowed(&ret, &t_rel, &omega, SpectrumMethod::Linear, &one);
    assert!(a.iter().flatten().all(|v| v.is_finite()));
    assert!(panics(|| evaluate_spectrum_windowed(&ret, &t_rel, &omega, SpectrumMethod::Cubic, &one)));
    // omega = 0 uses the trapezoidal rule
    let a0 = evaluate_spectrum_windowed(&ret, &t_rel, &[0.0], SpectrumMethod::Linear, &one);
    let v0 = ret[0][0];
    let v1 = ret[1][0];
    let want = -(0.5 * (v0 + v1) * c(2.0 * h, 0.0)).im / std::f64::consts::PI;
    assert!((a0[0][0] - want).abs() < 1e-14);
    // a single point is not enough
    let (t1, r1, _) = wigner_slices(&g, 0, h);
    assert!(panics(|| evaluate_spectrum_windowed(&r1, &t1, &omega, SpectrumMethod::Linear, &one)));
    // N = 16 is rejected, N = 18 (19 points) is the smallest cubic grid
    let (t17, r17, _) = wigner_slices(&g, 16, h);
    assert_eq!(t17.len(), 17);
    assert!(panics(|| evaluate_spectrum_windowed(&r17, &t17, &omega, SpectrumMethod::Cubic, &one)));
    let (t19, r19, _) = wigner_slices(&g, 18, h);
    assert_eq!(t19.len(), 19);
    let a = evaluate_spectrum_windowed(&r19, &t19, &omega, SpectrumMethod::Cubic, &one);
    assert!(a.iter().flatten().all(|v| v.is_finite()));
    // odd number of intervals
    let r18 = r19[..18].to_vec();
    let t18 = t19[..18].to_vec();
    assert!(panics(|| evaluate_spectrum_windowed(&r18, &t18, &omega, SpectrumMethod::Cubic, &one)));
    // cubic and linear agree on a smooth integrand
    let al = evaluate_spectrum_windowed(&r19, &t19, &omega, SpectrumMethod::Linear, &one);
    let mut maxd = 0.0f64;
    for (rowa, rowl) in a.iter().zip(al.iter()) {
        for (x, y) in rowa.iter().zip(rowl.iter()) {
            maxd = maxd.max((x - y).abs());
        }
    }
    assert!(maxd < 1e-2);
}

#[test]
fn spectral_invalid_input() {
    let t: Vec<f64> = (0..21).map(|k| k as f64 * 0.1).collect();
    let g = vec![vec![Complex::ZERO; 4]; 21];
    let omega = [0.0f64, 1.0];
    let one = ones_window();
    // (the invalid method `:spline` is unrepresentable with the enum)
    let g23 = vec![vec![Complex::ZERO; 6]; 21];
    assert!(panics(|| evaluate_spectrum_windowed(&g23, &t, &omega, SpectrumMethod::Linear, &one)));
    let t20 = t[..20].to_vec();
    assert!(panics(|| evaluate_spectrum_windowed(&g, &t20, &omega, SpectrumMethod::Linear, &one)));
    assert!(panics(|| {
        evaluate_spectrum_windowed(&g, &t, &omega, SpectrumMethod::Linear, &|_: &[f64]| vec![Complex::ZERO; 3])
    }));
    let mut tr: Vec<f64> = t.clone();
    tr.reverse();
    assert!(panics(|| evaluate_spectrum_windowed(&g, &tr, &omega, SpectrumMethod::Linear, &one)));
    let mut tb = t.clone();
    tb[4] += 0.03;
    assert!(panics(|| evaluate_spectrum_windowed(&g, &tb, &omega, SpectrumMethod::Linear, &one)));
    assert!(panics(|| evaluate_spectrum_windowed(&g, &t, &[f64::NAN], SpectrumMethod::Linear, &one)));
    let a = evaluate_spectrum_windowed(&g, &t, &omega, SpectrumMethod::Cubic, &one);
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].len(), 2);
    assert!(a.iter().flatten().all(|v| *v == 0.0));
}
