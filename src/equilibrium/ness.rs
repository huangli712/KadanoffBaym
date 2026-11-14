// Project : KadanoffBaym
// File    : src/equilibrium/ness.rs
// Purpose : the equilibrium steady-state Green's functions on the NESS2
//           grid (blueprint julia/src/equilibrium/ness.jl and the
//           private helpers of julia/src/math/dos.jl)

use crate::complex::Complex;
use crate::fourier::adft::{adft_eval, adft_sample};
use crate::fourier::transforms::{
    fft_to_time_array, integral_transform_to_freq, integral_transform_to_time, FftIntegralMethod,
};
use crate::solvers::dyson_ness::fermi_cut;
use crate::types::contour::Sign;
use crate::types::ness::{freq_mode, FftDomain, NessArray, NessHermMat};

const NEG_I: Complex = Complex::new(0.0, -1.0);

/// `A(w) n_B(w - mu)` with the cut-offs `+-100` and the DOS-slope
/// limit for `|beta (w - mu)| < 1e-8` (blueprint `_bose_integrand`).
pub(crate) fn bose_integrand(dos: &dyn Fn(f64) -> f64, w: f64, beta: f64, mu: f64) -> f64 {
    let arg = beta * (w - mu);
    if arg > 100.0 {
        return 0.0;
    }
    if arg < -100.0 {
        return -dos(w);
    }
    if arg.abs() < 1.0e-8 {
        let delta = 1.0e-6;
        let slope = (dos(delta + mu) - dos(mu)) / delta;
        return slope / beta;
    }
    dos(w) / ((arg).exp_m1())
}

/// Retarded component `G^R(t) = -i integral A(w) e^{-i w t}` at
/// `t >= 0` by the adaptive Fourier integral, zero at negative times
/// (blueprint `_from_dos_ness_adaptive!`).
fn from_dos_ness_adaptive(
    g: &mut NessHermMat,
    dos: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    nn: i64,
    limit: i64,
) {
    let nt = (g.grid.nft / 2) as usize;
    let h = g.grid.h;
    let d = g.ret.size1;
    let p = adft_sample(|x| Complex::new(dos(x), 0.0), lo, hi, nn, limit, 0.0, 16);
    g.ret.time.fill(Complex::ZERO);
    for n in 0..nt {
        let (res, _) = adft_eval(&p, -h * n as f64);
        let res = res * NEG_I;
        let b = n * d * d;
        for i in 0..d {
            g.ret.time[b + i * d + i] = res;
        }
    }
}

/// Lesser component `-i integral f(w) e^{-i w t}` at
/// `0 <= t < Nft/2`, `G^<(-t) = -G^<(t)^dagger`, and zero at the
/// Nyquist mode (blueprint `_compute_lesser_adaptive!`). The panels
/// are selected at the Fourier variable `10 h`, as in NESSi.
#[allow(clippy::too_many_arguments)]
fn compute_lesser_adaptive(
    a: &mut NessArray,
    nt: usize,
    integrand: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    h: f64,
    nn: i64,
    limit: i64,
) {
    let nft = a.nft as usize;
    let d = a.size1;
    assert!(nt <= nft / 2);
    a.time.fill(Complex::ZERO);
    let p = adft_sample(
        |x| Complex::new(integrand(x), 0.0),
        lo,
        hi,
        nn,
        limit,
        10.0 * h,
        16,
    );
    for n in 0..nt {
        let (res, _) = adft_eval(&p, -(n as f64) * h);
        let res = res * NEG_I;
        for i in 0..d {
            a.time[n * d * d + i * d + i] = res;
            if n > 0 {
                a.time[(nft - n) * d * d + i * d + i] = -res.conj();
            }
        }
    }
    a.block_mut(FftDomain::Time, (nft / 2) as i64).fill(Complex::ZERO);
}

/// Equilibrium steady-state Green's function of the density of states
/// `dos`, diagonal in the orbital space (blueprint `from_dos_ness!`),
///
/// `G^R(t) = -i integral dw A(w) e^{-i w t}`,
/// `G^<(t) = -xi i integral dw A(w) f(w - mu) e^{-i w t}`.
///
/// `Trapez` sets the frequency data and transforms back;
/// `Adaptive` computes the time data by the adaptive Fourier integral
/// on `[lo, hi]` (`lo`/`hi` must be passed for `Adaptive`, a plain
/// closure carries no bounds).
#[allow(clippy::too_many_arguments)]
pub fn from_dos_ness(
    g: &mut NessHermMat,
    sign: Sign,
    dos: &dyn Fn(f64) -> f64,
    beta: f64,
    mu: f64,
    method: FftIntegralMethod,
    lo: Option<f64>,
    hi: Option<f64>,
    limit: i64,
    nn: i64,
) {
    assert!(beta > 0.0, "beta must be positive, got {beta}");
    assert!(
        method == FftIntegralMethod::Trapez || method == FftIntegralMethod::Adaptive,
        "method must be FFT_TRAPEZ or FFT_ADAPTIVE"
    );
    let grid = g.grid;
    let nft = grid.nft;
    let d = g.ret.size1;

    if method == FftIntegralMethod::Adaptive {
        let l = lo.expect("from_dos_ness: pass `lo` for FFT_ADAPTIVE (a closure carries no bounds)");
        let u = hi.expect("from_dos_ness: pass `hi` for FFT_ADAPTIVE (a closure carries no bounds)");
        assert!(l < u, "lo must be smaller than hi");
        from_dos_ness_adaptive(g, dos, l, u, nn, limit);
        if sign == Sign::Fermi {
            let integrand = |x: f64| -dos(x) * fermi_cut(beta * (x - mu));
            compute_lesser_adaptive(&mut g.les, (nft / 2) as usize, &integrand, l, u, grid.h, nn, limit);
        } else {
            let integrand = |x: f64| bose_integrand(dos, x, beta, mu);
            compute_lesser_adaptive(&mut g.les, (nft / 2) as usize, &integrand, l, u, grid.h, nn, limit);
        }
        return;
    }

    // FFT_TRAPEZ
    g.ret.freq.fill(Complex::ZERO);
    g.les.freq.fill(Complex::ZERO);
    let nfreq = nft / 2;
    for w in (-nfreq + 1)..=(nfreq - 1) {
        let n = w.rem_euclid(nft);
        let omega = freq_mode(&grid, w);
        let aw = dos(omega);
        let r = Complex::new(0.0, -std::f64::consts::PI * aw);
        let l = if sign == Sign::Fermi {
            -2.0 * fermi_cut(beta * (omega - mu)) * r
        } else {
            Complex::new(0.0, -bose_integrand(dos, omega, beta, mu) * 2.0 * std::f64::consts::PI)
        };
        let b = n as usize * d * d;
        for i in 0..d {
            g.ret.freq[b + i * d + i] = r;
            g.les.freq[b + i * d + i] = l;
        }
    }
    integral_transform_to_time(g, FftIntegralMethod::Trapez);
}

/// Replace the lesser component by the equilibrium one obtained from
/// the retarded component by the fluctuation-dissipation relation
/// (blueprint `equilibrate_lesser!`),
///
/// `G^<(w) = 2 xi f(w - mu) (G^R(w) - G^R(w)^dagger) / 2`.
///
/// For bosons and a frequency mode at `w = mu`
/// (`|beta (w - mu)| < 1e-8`) the limit `A'(mu)/beta` of `A n_B` is
/// used when the spectral function vanishes linearly there; otherwise
/// the function panics.
pub fn equilibrate_lesser(g: &mut NessHermMat, sign: Sign, beta: f64, mu: f64) {
    assert!(beta > 0.0, "beta must be positive, got {beta}");
    let grid = g.grid;
    let nft = grid.nft;
    let d = g.ret.size1;
    let b = d * d;

    integral_transform_to_freq(g, FftIntegralMethod::Trapez);
    let rfreq = g.ret.freq.clone();
    g.les.freq.fill(Complex::ZERO);

    // spectral matrix A = -(R - R')/(2 pi i)
    let spec = |n: usize| -> Vec<Complex> {
        let src = &rfreq[n * b..(n + 1) * b];
        (0..b)
            .map(|q| {
                let i = q / d;
                let j = q % d;
                -(src[i * d + j] - src[j * d + i].conj())
                    / (2.0 * std::f64::consts::PI * Complex::new(0.0, 1.0))
            })
            .collect()
    };
    let spec_norm = |n: usize| -> f64 {
        let s = spec(n);
        s.iter().map(|z| z.abs() * z.abs()).sum::<f64>().sqrt()
    };
    let mut scale = 0.0f64;
    for n in 0..nft as usize {
        scale = scale.max(spec_norm(n));
    }

    let xi = sign.as_i64() as f64;
    for k in 0..nft {
        let n = k as usize;
        let omega = freq_mode(&grid, k);
        let src = &rfreq[n * b..(n + 1) * b];
        let arg = beta * (omega - mu);
        if sign == Sign::Bose && arg.abs() < 1.0e-8 {
            let a0 = spec(n);
            let al = spec((k - 1).rem_euclid(nft) as usize);
            let ar = spec(((k + 1) % nft) as usize);
            let sl: Vec<Complex> = a0.iter().zip(al.iter()).map(|(x, y)| (*x - *y) / grid.dw).collect();
            let sr: Vec<Complex> = ar.iter().zip(a0.iter()).map(|(x, y)| (*x - *y) / grid.dw).collect();
            let nl = sl.iter().map(|z| z.abs() * z.abs()).sum::<f64>().sqrt();
            let nr = sr.iter().map(|z| z.abs() * z.abs()).sum::<f64>().sqrt();
            let diff: f64 = sr
                .iter()
                .zip(sl.iter())
                .map(|(x, y)| (*x - *y).abs() * (*x - *y).abs())
                .sum::<f64>()
                .sqrt();
            let ok = spec_norm(n) <= 1.0e-12 * scale
                && sl.iter().chain(sr.iter()).all(|z| z.re.is_finite() && z.im.is_finite())
                && diff <= 1.0e-6 * nl.max(nr) + 1.0e-12 * scale;
            assert!(
                ok,
                "equilibrate_lesser: the spectral function does not vanish linearly at omega = mu"
            );
            let slope: Vec<Complex> = sl.iter().zip(sr.iter()).map(|(x, y)| 0.5 * (*x + *y)).collect();
            // A n_B -> A'(mu)/beta, G^< = -2 pi i A n_B
            let out = g.les.block_mut(FftDomain::Freq, k);
            for q in 0..b {
                out[q] = (-2.0 * std::f64::consts::PI * Complex::new(0.0, 1.0) / beta) * slope[q];
            }
            continue;
        }
        let f = if sign == Sign::Bose {
            if arg > 100.0 {
                0.0
            } else if arg < -100.0 {
                -1.0
            } else {
                1.0 / ((arg).exp_m1())
            }
        } else {
            fermi_cut(arg)
        };
        // ImG = 0.5 (M - M'), elementwise in the matrix sense
        let out = g.les.block_mut(FftDomain::Freq, k);
        for i in 0..d {
            for j in 0..d {
                let img = 0.5 * (src[i * d + j] - src[j * d + i].conj());
                out[i * d + j] = (2.0 * xi * f) * img;
            }
        }
    }

    fft_to_time_array(&mut g.les);
    let s = Complex::new(grid.dw / (2.0 * std::f64::consts::PI), 0.0);
    for z in g.les.time.iter_mut() {
        *z *= s;
    }
    for n in 1..(nft / 2) {
        let src = g.les.block(FftDomain::Time, n).to_vec();
        let dst = g.les.block_mut(FftDomain::Time, nft - n);
        for i in 0..d {
            for j in 0..d {
                dst[i * d + j] = -src[j * d + i].conj();
            }
        }
    }
    g.les.block_mut(FftDomain::Time, nft / 2).fill(Complex::ZERO);
}
