// Project : KadanoffBaym
// File    : src/fourier/steady.rs
// Purpose : Fourier transforms of the steady-state functions
//
// The Fourier convention is
//     G(w) = integral dt e^{+i w t} G(t),
//     G(t) = integral dw/(2 pi) e^{-i w t} G(w).
// The bare transforms `fft_to_freq_array` (BACKWARD, positive exponent) and
// `fft_to_time_array` (FORWARD, negative exponent) are unnormalized. The
// physical integral transforms multiply by `h` or `dw/(2 pi)` and
// apply the trapezoidal or the cubically corrected end corrections.

//!
//! *Remarks* : *Steady-State Green's Functions*
//!
//! In a nonequilibrium steady state the Green's functions depend on the
//! relative time only, ``G(t, t') = G(t - t')``. The retarded and lesser
//! components are stored on the periodic grid of `Nft` points (see *Remarks* :
//! *Steady-State Grid*), both in the time and in the frequency domain. The Fourier
//! convention is
//!
//! ```math
//! \begin{equation}
//! G(\omega) = \int dt\ e^{i\omega t} G(t),\quad
//! G(t) = \int \frac{d\omega}{2\pi} e^{-i\omega t} G(\omega).
//! \end{equation}
//! ```
//!
//! The bare transforms `fft_to_freq_array` (BACKWARD, positive exponent) and
//! `fft_to_time_array` (FORWARD, negative exponent) are unnormalized. The
//! physical integral transforms multiply by ``h`` or ``\Delta\omega/(2\pi)``:
//!
//! * `integral_transform_to_freq`: the retarded component is integrated over
//!   ``t \ge 0`` (trapezoidal end correction ``-G^R(0)/2`` or the cubic end
//!   corrections), the lesser component over all times (the cubic method
//!   only multiplies by the correction factor ``W(\omega h)``). The Nyquist
//!   point is set to zero.
//! * `integral_transform_to_time`: the retarded component is reconstructed
//!   from ``G^R(\omega) - G^R(\omega)^\dagger`` and kept at ``t \ge 0`` only.
//!   The lesser component satisfies ``G^<(-t) = -G^<(t)^\dagger``. The
//!   Nyquist point is set to zero.
//!

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::fourier::FftDirection;
use crate::types::{FftDomain, SteadyFunction, SteadyGreen};

use crate::fourier::{dft_boundary_correction, dft_coefficients_cubic, fft_strided, plan_fft};

/// Method of the integral transforms of the steady-state data.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FftIntegralMethod {
    /// Trapezoidal rule.
    Trapez,

    /// Cubically corrected.
    Cubic,

    /// Adaptive Fourier integral (only for the equilibrium Green's
    /// functions).
    Adaptive,
}

/// Only `Trapez` and `Cubic` are supported by the integral transforms.
fn check_steady_method(method: FftIntegralMethod) {
    assert!(
        method == FftIntegralMethod::Trapez || method == FftIntegralMethod::Cubic,
        "only FftIntegralMethod::Trapez and FftIntegralMethod::Cubic are supported"
    );
}

/// Bare (unnormalized) transform of the time data to the frequency
/// data with the positive exponent.
fn fft_to_freq_array(a: &mut SteadyFunction) {
    let n = a.nft as usize;
    let d1 = a.size1;
    let d2 = a.size2;
    let p = plan_fft(n, FftDirection::Backward);
    let mut scratch = vec![Complex::ZERO; p.scratch_len + n];
    fft_strided(
        &mut a.freq,
        &a.time,
        (d1, d2, n),
        2,
        FftDirection::Backward,
        &p,
        &mut scratch,
    );
}

/// Bare (unnormalized) transform of the frequency data to the time
/// data with the negative exponent.
pub(crate) fn fft_to_time_array(a: &mut SteadyFunction) {
    let n = a.nft as usize;
    let d1 = a.size1;
    let d2 = a.size2;
    let p = plan_fft(n, FftDirection::Forward);
    let mut scratch = vec![Complex::ZERO; p.scratch_len + n];
    fft_strided(
        &mut a.time,
        &a.freq,
        (d1, d2, n),
        2,
        FftDirection::Forward,
        &p,
        &mut scratch,
    );
}

/// Physical Fourier integral of the retarded and lesser components. The
/// retarded component is integrated over `t >= 0` (trapezoidal end
/// correction `-G^R(0)/2` or
/// the cubic end corrections), the lesser component over all times
/// (the cubic method only multiplies by the correction factor). The
/// Nyquist mode is set to zero.
pub fn integral_transform_to_freq(g: &mut SteadyGreen, method: FftIntegralMethod) {
    check_steady_method(method);
    let grid = g.grid;
    let nft = grid.nft;
    let h = grid.h;
    let nhalf = (nft / 2) as usize;
    let nftu = nft as usize;
    let d = g.ret.size1;
    let b = g.ret.size1 * g.ret.size2;

    // Retarded component, integrated over t >= 0
    fft_to_freq_array(&mut g.ret);
    if method == FftIntegralMethod::Cubic {
        let a = 0.0f64;
        let bb = (nhalf as f64 - 1.0) * h;
        let t = g.ret.time.clone();
        let mut f8 = [Complex::ZERO; 8];
        let mut endcor = vec![Complex::ZERO; b];
        for k in 0..nftu {
            let w = grid.freq_mode(k as i64);
            let mut corfac = 0.0f64;
            for j in 0..d {
                for i in 0..d {
                    for m in 0..4usize {
                        f8[m] = t[m * b + i * d + j];
                        f8[7 - m] = t[(nhalf - 1 - m) * b + i * d + j];
                    }
                    let (ec, cf) = dft_boundary_correction(w, h, a, bb, &f8);
                    corfac = cf;
                    endcor[i * d + j] = ec;
                }
            }
            let z = g.ret.block_mut(FftDomain::Freq, k as i64);
            for q in 0..b {
                z[q] = z[q] * corfac + endcor[q];
            }
        }
    } else {
        let g0 = g.ret.block(FftDomain::Time, 0).to_vec();
        for k in 0..nftu {
            let z = g.ret.block_mut(FftDomain::Freq, k as i64);
            for q in 0..b {
                z[q] -= 0.5 * g0[q];
            }
        }
    }
    let sh = Complex::new(h, 0.0);
    for z in g.ret.freq.iter_mut() {
        *z *= sh;
    }
    g.ret
        .block_mut(FftDomain::Freq, nft / 2)
        .fill(Complex::ZERO);

    // Lesser component, integrated over all times
    fft_to_freq_array(&mut g.les);
    if method == FftIntegralMethod::Cubic {
        for k in 0..nftu {
            let (corfac, _) = dft_coefficients_cubic(grid.freq_mode(k as i64) * h);
            let z = g.les.block_mut(FftDomain::Freq, k as i64);
            let f = Complex::new(corfac, 0.0);
            for zq in z[..b].iter_mut() {
                *zq *= f;
            }
        }
    }
    let sh = Complex::new(h, 0.0);
    for z in g.les.freq.iter_mut() {
        *z *= sh;
    }
    g.les
        .block_mut(FftDomain::Freq, nft / 2)
        .fill(Complex::ZERO);
}

/// Physical inverse Fourier integral of the retarded and lesser
/// components. The retarded component is reconstructed from its
/// anti-hermitian (spectral) part and kept at `t >= 0`, and
/// `G^<(-t) = -G^<(t)^dagger` is imposed.
pub fn integral_transform_to_time(g: &mut SteadyGreen, method: FftIntegralMethod) {
    check_steady_method(method);
    let grid = g.grid;
    let nft = grid.nft;
    let nhalf = (nft / 2) as usize;
    let nftu = nft as usize;
    let d = g.ret.size1;
    let b = g.ret.size1 * g.ret.size2;
    let dw = grid.dw;

    // Retarded component from G^R(w) - G^R(w)^dagger
    let rfreq = g.ret.freq.clone();
    let mut tf = vec![Complex::ZERO; nftu * b];
    for k in 0..nftu {
        let src = &rfreq[k * b..(k + 1) * b];
        let dst = &mut tf[k * b..(k + 1) * b];
        for i in 0..d {
            for j in 0..d {
                dst[i * d + j] = src[i * d + j] - src[j * d + i].conj();
            }
        }
    }
    let p = plan_fft(nftu, FftDirection::Forward);
    let mut scratch = vec![Complex::ZERO; p.scratch_len + nftu];
    let mut tt = vec![Complex::ZERO; nftu * b];
    fft_strided(
        &mut tt,
        &tf,
        (d, d, nftu),
        2,
        FftDirection::Forward,
        &p,
        &mut scratch,
    );
    if method == FftIntegralMethod::Cubic {
        for n in 0..nhalf {
            let (corfac, _) = dft_coefficients_cubic(-grid.time_mode(n as i64) * dw);
            for q in 0..b {
                tt[n * b + q] *= Complex::new(corfac, 0.0);
            }
        }
    }
    g.ret.time.fill(Complex::ZERO);
    for n in 0..nhalf {
        let dst = g.ret.block_mut(FftDomain::Time, n as i64);
        for q in 0..b {
            dst[q] = tt[n * b + q] * (dw * 0.5 / PI);
        }
    }

    // Lesser component
    fft_to_time_array(&mut g.les);
    if method == FftIntegralMethod::Cubic {
        for n in 0..nhalf {
            let (corfac, _) = dft_coefficients_cubic(-grid.time_mode(n as i64) * dw);
            let z = g.les.block_mut(FftDomain::Time, n as i64);
            let f = Complex::new(corfac, 0.0);
            for zq in z[..b].iter_mut() {
                *zq *= f;
            }
        }
    }
    let s = Complex::new(dw * 0.5 / PI, 0.0);
    for z in g.les.time.iter_mut() {
        *z *= s;
    }
    for n in 1..nhalf {
        let src = g.les.block(FftDomain::Time, n as i64).to_vec();
        let dst = g.les.block_mut(FftDomain::Time, nft - n as i64);
        for i in 0..d {
            for j in 0..d {
                dst[i * d + j] = -src[j * d + i].conj();
            }
        }
    }
    g.les
        .block_mut(FftDomain::Time, nft / 2)
        .fill(Complex::ZERO);
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    use crate::types::SteadyGrid;

    fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    #[test]
    fn steady_fft_round_trip() {
        // round trip of the bare transforms, unnormalized (1/nft)
        let nft = 8i64;
        let mut a = SteadyFunction::new(nft, 2);
        for t in 0..nft {
            for i in 0..2usize {
                for j in 0..2usize {
                    let v = Complex::new(t as f64 + i as f64 * 0.1, j as f64 * 0.2);
                    a.block_mut(FftDomain::Time, t)[i * 2 + j] = v;
                }
            }
        }
        let orig = a.time.clone();
        fft_to_freq_array(&mut a);
        assert_eq!(a.time, orig);
        fft_to_time_array(&mut a);
        for (x, y) in a.time.iter().zip(orig.iter()) {
            assert!((*x / 8.0 - *y).abs() < 1e-10);
        }
        // a pulse at t = h gives exp(+i w h) in frequency
        let mut b = SteadyFunction::new(nft, 1);
        b.block_mut(FftDomain::Time, 1)[0] = Complex::ONE;
        fft_to_freq_array(&mut b);
        for k in 0..nft as usize {
            let want = Complex::from_polar(1.0, 2.0 * PI * k as f64 / nft as f64);
            assert!((b.freq[k] - want).abs() < 1e-12);
        }
    }

    #[test]
    fn steady_transforms_analytic() {
        for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
            // G^R(t) = -i e^{-(eta + i eps) t}, G^R(w) = 1/(w - eps + i eta)
            let (eps, eta) = (0.3f64, 1.0f64);
            let mut errs = Vec::new();
            for &h in &[0.1f64, 0.05, 0.025] {
                let nft = (102.4 / h).round() as i64;
                let mut g = SteadyGreen::new(SteadyGrid::new(nft, h), 1);
                for n in 0..(nft / 2) {
                    let t = n as f64 * h;
                    let decay = (-(eta * t)).exp();
                    g.set_ret(
                        n,
                        FftDomain::Time,
                        &[Complex::new(0.0, -decay) * Complex::from_polar(1.0, -eps * t)],
                    );
                }
                integral_transform_to_freq(&mut g, method);
                // Nyquist mode
                assert_eq!(g.ret.freq[(nft / 2) as usize], Complex::ZERO);
                let grid = g.grid;
                let mut e = 0.0f64;
                for &n in &[0i64, 1, 4, 10, nft - 1] {
                    let want = Complex::ONE / (Complex::new(grid.freq_at(n) - eps, eta));
                    e = e.max((g.ret.freq[n as usize] - want).abs());
                }
                errs.push(e);
            }
            if method == FftIntegralMethod::Trapez {
                assert!(
                    (errs[0] / errs[1]) > 3.5 && (errs[1] / errs[2]) > 3.5,
                    "{errs:?}"
                );
            } else {
                assert!(
                    (errs[0] / errs[1]) > 12.0 && (errs[1] / errs[2]) > 12.0,
                    "{errs:?}"
                );
                assert!(errs[2] < 1e-8, "{errs:?}");
            }
        }
    }

    #[test]
    fn steady_transforms_physical_round_trip() {
        for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
            let (h, nft) = (0.1f64, 512i64);
            let mut g = SteadyGreen::new(SteadyGrid::new(nft, h), 2);
            let u = [
                Complex::new(1.0 / 2f64.sqrt(), 0.0),
                Complex::new(0.0, 1.0 / 2f64.sqrt()),
                Complex::new(0.0, 1.0 / 2f64.sqrt()),
                Complex::new(1.0 / 2f64.sqrt(), 0.0),
            ];
            for n in 0..nft {
                let t = g.grid.time_at(n);
                let d0 = (-t * t / 2.0).exp();
                let d1 = 0.5 * (-t * t / 3.0).exp();
                // L = i U diag(d0, d1) U'
                // U diag(d0, d1) U^dagger
                let ud = [
                    u[0] * d0 * u[0].conj() + u[1] * d1 * u[1].conj(),
                    u[0] * d0 * u[2].conj() + u[1] * d1 * u[3].conj(),
                    u[2] * d0 * u[0].conj() + u[3] * d1 * u[2].conj(),
                    u[2] * d0 * u[2].conj() + u[3] * d1 * u[3].conj(),
                ];
                let mut l = [Complex::ZERO; 4];
                for q in 0..4 {
                    l[q] = Complex::new(0.0, 1.0) * ud[q];
                }
                g.set_les(n, FftDomain::Time, &l);
                if t >= 0.0 && n < nft / 2 {
                    let ph = Complex::from_polar(1.0, -0.3 * t);
                    let mut r = [Complex::ZERO; 4];
                    for q in 0..4 {
                        r[q] = Complex::new(0.0, -1.0) * ud[q] * ph;
                    }
                    g.set_ret(n, FftDomain::Time, &r);
                }
            }
            let g0 = g.clone();
            integral_transform_to_freq(&mut g, method);
            integral_transform_to_time(&mut g, method);
            let tol = if method == FftIntegralMethod::Trapez {
                1e-10
            } else {
                1e-6
            };
            assert!(
                (g.ret.distance(&g0.ret, FftDomain::Time) / nft as f64) < tol,
                "ret"
            );
            assert!(
                (g.les.distance(&g0.les, FftDomain::Time) / nft as f64) < tol,
                "les"
            );
            // lesser hermiticity G^<(-t) = -G^<(t)'
            for m in 1..(nft / 2) as usize {
                let a = g.les.block(FftDomain::Time, m as i64).to_vec();
                let b = g.les.block(FftDomain::Time, nft - m as i64).to_vec();
                for q in 0..4 {
                    assert!((b[q] + a[(q % 2) * 2 + q / 2].conj()).abs() < 1e-10);
                }
            }
            assert!(
                g.les
                    .block(FftDomain::Time, nft / 2)
                    .iter()
                    .all(|z| *z == Complex::ZERO)
            );
            // retarded at negative times is zero
            for n in (nft / 2)..nft {
                assert!(
                    g.ret
                        .block(FftDomain::Time, n)
                        .iter()
                        .all(|z| *z == Complex::ZERO)
                );
            }
            assert!(panics(|| integral_transform_to_freq(
                &mut g,
                FftIntegralMethod::Adaptive
            )));
            assert!(panics(|| integral_transform_to_time(
                &mut g,
                FftIntegralMethod::Adaptive
            )));
        }
    }
}
