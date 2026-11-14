// Project : KadanoffBaym
// File    : src/fourier/transforms.rs
// Purpose : Fourier transforms of the steady-state (NESS2) functions
//           (blueprint julia/src/fourier/transforms.jl)
//
// The Fourier convention is
//     G(w) = integral dt e^{+i w t} G(t),
//     G(t) = integral dw/(2 pi) e^{-i w t} G(w).
// The bare transforms `fft_to_freq` (BACKWARD, positive exponent) and
// `fft_to_time` (FORWARD, negative exponent) are unnormalized. The
// physical integral transforms multiply by `h` or `dw/(2 pi)` and
// apply the trapezoidal or the cubically corrected end corrections.

use crate::complex::Complex;
use crate::fourier::dftcor::{dft_endcor, dftcorr_cubic};
use crate::fourier::fft::{fft_strided, plan_fft, FftDirection};
use crate::types::ness::{FftDomain, NessArray, NessGrid, NessHermMat};

/// Method of the integral transforms of the NESS2 data (blueprint
/// `FFTIntegralMethod`).
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

/// Only `Trapez` and `Cubic` are supported by the integral transforms
/// (blueprint `_check_ness_method`).
pub(crate) fn check_ness_method(method: FftIntegralMethod) {
    assert!(
        method == FftIntegralMethod::Trapez || method == FftIntegralMethod::Cubic,
        "only FFT_TRAPEZ and FFT_CUBIC are supported"
    );
}

/// Bare (unnormalized) transform of the time data to the frequency
/// data with the positive exponent (blueprint `fft_to_freq!(A)`).
pub fn fft_to_freq_array(a: &mut NessArray) {
    let n = a.nft as usize;
    let d1 = a.size1;
    let d2 = a.size2;
    let p = plan_fft(n, FftDirection::Backward);
    let mut scratch = vec![Complex::ZERO; p.scratch_len];
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
/// data with the negative exponent (blueprint `fft_to_time!(A)`).
pub fn fft_to_time_array(a: &mut NessArray) {
    let n = a.nft as usize;
    let d1 = a.size1;
    let d2 = a.size2;
    let p = plan_fft(n, FftDirection::Forward);
    let mut scratch = vec![Complex::ZERO; p.scratch_len];
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

#[allow(dead_code)]
/// Bare transform of the retarded and lesser components to the
/// frequency domain (blueprint `fft_to_freq!(G)`).
pub(crate) fn fft_to_freq(g: &mut NessHermMat) {
    fft_to_freq_array(&mut g.ret);
    fft_to_freq_array(&mut g.les);
}

#[allow(dead_code)]
/// Bare transform of the retarded and lesser components to the time
/// domain (blueprint `fft_to_time!(G)`).
pub(crate) fn fft_to_time(g: &mut NessHermMat) {
    fft_to_time_array(&mut g.ret);
    fft_to_time_array(&mut g.les);
}

/// Physical Fourier integral of the retarded and lesser components
/// (blueprint `integral_transform_to_freq!`). The retarded component is
/// integrated over `t >= 0` (trapezoidal end correction `-G^R(0)/2` or
/// the cubic end corrections), the lesser component over all times
/// (the cubic method only multiplies by the correction factor). The
/// Nyquist mode is set to zero.
pub fn integral_transform_to_freq(g: &mut NessHermMat, method: FftIntegralMethod) {
    check_ness_method(method);
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
                    let (cf, ec) = dft_endcor(w, h, a, bb, &f8, 1);
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
    g.ret.block_mut(FftDomain::Freq, nft / 2).fill(Complex::ZERO);

    // Lesser component, integrated over all times
    fft_to_freq_array(&mut g.les);
    if method == FftIntegralMethod::Cubic {
        for k in 0..nftu {
            let (corfac, _) = dftcorr_cubic(grid.freq_mode(k as i64) * h);
            let z = g.les.block_mut(FftDomain::Freq, k as i64);
            for q in 0..b {
                z[q] *= Complex::new(corfac, 0.0);
            }
        }
    }
    let sh = Complex::new(h, 0.0);
    for z in g.les.freq.iter_mut() {
        *z *= sh;
    }
    g.les.block_mut(FftDomain::Freq, nft / 2).fill(Complex::ZERO);
}

/// Physical inverse Fourier integral of the retarded and lesser
/// components (blueprint `integral_transform_to_time!`). The retarded
/// component is reconstructed from its anti-hermitian (spectral) part
/// and kept at `t >= 0`, and `G^<(-t) = -G^<(t)^dagger` is imposed.
pub fn integral_transform_to_time(g: &mut NessHermMat, method: FftIntegralMethod) {
    check_ness_method(method);
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
    let mut scratch = vec![Complex::ZERO; p.scratch_len];
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
            let (corfac, _) = dftcorr_cubic(-grid.time_mode(n as i64) * dw);
            for q in 0..b {
                tt[n * b + q] *= Complex::new(corfac, 0.0);
            }
        }
    }
    g.ret.time.fill(Complex::ZERO);
    for n in 0..nhalf {
        let dst = g.ret.block_mut(FftDomain::Time, n as i64);
        for q in 0..b {
            dst[q] = tt[n * b + q] * (dw * 0.5 / std::f64::consts::PI);
        }
    }

    // Lesser component
    fft_to_time_array(&mut g.les);
    if method == FftIntegralMethod::Cubic {
        for n in 0..nhalf {
            let (corfac, _) = dftcorr_cubic(-grid.time_mode(n as i64) * dw);
            let z = g.les.block_mut(FftDomain::Time, n as i64);
            for q in 0..b {
                z[q] *= Complex::new(corfac, 0.0);
            }
        }
    }
    let s = Complex::new(dw * 0.5 / std::f64::consts::PI, 0.0);
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
    g.les.block_mut(FftDomain::Time, nft / 2).fill(Complex::ZERO);
}

#[allow(dead_code)]
/// The number of the positive time points (used by the equilibrium
/// constructors).
pub(crate) fn ness_positive_half(g: &NessGrid) -> usize {
    (g.nft / 2) as usize
}
