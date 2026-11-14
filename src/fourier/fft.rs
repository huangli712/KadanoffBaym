// Project : KadanoffBaym
// File    : src/fourier/fft.rs
// Purpose : arbitrary-length FFT (blueprint: fourier/fft.jl)
//
// Mixed-radix self-sorting (Stockham) Cooley-Tukey transform with radix
// 4 preferred, then 2, 3, 5, 7, 11, 13, and a Bluestein (chirp-z)
// reduction for lengths containing larger prime factors. Both
// directions are unnormalized (FFTW convention).
//!
//! *Remarks* : *Fast Fourier Transform*
//!
//! The discrete Fourier transform of a sequence ``x_n`` (``n = 0, \cdots,
//! N-1``) is
//!
//! ```math
//! \begin{equation}
//! X_k = \sum^{N-1}_{n=0} x_n e^{2\pi i s n k / N},
//! \end{equation}
//! ```
//!
//! where ``s = -1`` (`FFT_FORWARD`) or ``s = +1`` (`FFT_BACKWARD`). Both
//! directions are unnormalized, so that a forward transform followed by a
//! backward transform multiplies the data by ``N``. This is the FFTW
//! convention used by NESSi.
//!
//! If all prime factors of ``N`` are not larger than 13, a mixed-radix
//! self-sorting (Stockham) Cooley-Tukey algorithm is used. A stage with
//! radix ``p``, current length ``n``, ``m = n/p``, and stride ``s'`` (the
//! product of the previous radices) reads
//!
//! ```math
//! \begin{equation}
//! y_{q + s'(pk + u)} = W^{uk}_{n} \sum^{p-1}_{t=0} W^{ut}_{p}
//!     x_{q + s'(k + tm)},
//! \end{equation}
//! ```
//!
//! where ``W_n = e^{2\pi i s/n}``. Otherwise the transform is reduced to a
//! cyclic convolution of length ``M = 2^{\lceil \log_2(2N-1) \rceil}`` by
//! the Bluestein (chirp-z) algorithm,
//!
//! ```math
//! \begin{equation}
//! X_k = c_k \sum^{N-1}_{n=0} (x_n c_n) b_{k-n},\quad
//! c_n = e^{i\pi s n^2/N},\quad
//! b_m = e^{-i\pi s m^2/N}.
//! \end{equation}
//! ```
//!
//! *References* :
//!
//! Please see [`NESSi`] `ness2_fft_decl.hpp` and `ness2_fft_impl.hpp`. NESSi
//! delegates the transforms to FFTW, here they are implemented with Julia
//! Base only.

use crate::complex::Complex;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Direction of the discrete Fourier transform: `Forward` uses
/// `e^{-2 pi i nk/N}`, `Backward` uses `e^{+2 pi i nk/N}`. Both are
/// unnormalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FftDirection {
    Forward,
    Backward,
}

impl FftDirection {
    /// Sign of the exponent (`-1` forward, `+1` backward).
    #[inline]
    pub fn sign(self) -> i64 {
        match self {
            FftDirection::Forward => -1,
            FftDirection::Backward => 1,
        }
    }
}

/// Immutable plan of the discrete Fourier transform of length `n`
/// (blueprint `FFTPlan`).
#[derive(Debug)]
pub struct FftPlan {
    /// Length of the transform.
    pub n: usize,
    /// Direction.
    pub direction: FftDirection,
    /// Radices of the mixed-radix stages (of the inner transform for a
    /// Bluestein plan).
    pub factors: Vec<usize>,
    /// Twiddle factors and roots of unity of the stages.
    pub twiddles: Vec<Vec<Complex>>,
    /// Length of the Bluestein cyclic convolution, 0 for mixed-radix
    /// plans.
    pub bluestein_n: usize,
    /// Chirp `c_n`, empty for mixed-radix plans.
    pub chirp: Vec<Complex>,
    /// Forward transform of the convolution kernel, empty for
    /// mixed-radix plans.
    pub kernel_fft: Vec<Complex>,
    /// Required length of the scratch vector.
    pub scratch_len: usize,
}

/// `cispi(x) = exp(i pi x)`.
#[inline]
fn cispi(x: f64) -> Complex {
    Complex::from_polar(1.0, std::f64::consts::PI * x)
}

/// Factorize `n` into the radices of the mixed-radix stages: radix 4
/// preferred, then 2, 3, 5, 7, 11, and 13. Returns `None` if `n` has a
/// prime factor larger than 13 (blueprint `_fft_factorize`).
fn fft_factorize(n: usize) -> Option<Vec<usize>> {
    let mut factors = Vec::new();
    let mut r = n;
    while r % 4 == 0 {
        factors.push(4);
        r /= 4;
    }
    for p in [2usize, 3, 5, 7, 11, 13] {
        while r % p == 0 {
            factors.push(p);
            r /= p;
        }
    }
    if r == 1 {
        Some(factors)
    } else {
        None
    }
}

/// Twiddle factors and roots of unity of the stages (blueprint
/// `_fft_twiddles`).
fn fft_twiddles(n: usize, factors: &[usize], s: i64) -> Vec<Vec<Complex>> {
    let mut twiddles = Vec::new();
    let mut nc = n as i64;
    for &p in factors {
        let p = p as i64;
        let m = nc / p;
        let mut tw = vec![Complex::ZERO; ((p - 1) * m + p) as usize];
        for k in 0..m {
            for u in 1..p {
                let arg = 2.0 * s as f64 * ((u * k) % nc) as f64 / nc as f64;
                tw[(k * (p - 1) + u - 1) as usize] = cispi(arg);
            }
        }
        for j in 0..p {
            tw[((p - 1) * m + j) as usize] = cispi(2.0 * s as f64 * j as f64 / p as f64);
        }
        twiddles.push(tw);
        nc = m;
    }
    twiddles
}

/// One stage of the self-sorting mixed-radix transform (blueprint
/// `_fft_stage!`). `st` is the stride; the sign of the exponent is
/// encoded in the roots of unity at the end of `tw`.
#[allow(clippy::too_many_arguments)]
fn fft_stage(
    dst: &mut [Complex],
    doff: usize,
    src: &[Complex],
    soff: usize,
    p: usize,
    m: usize,
    st: usize,
    tw: &[Complex],
) {
    let nr = (p - 1) * m;
    match p {
        2 => {
            for k in 0..m {
                let w1 = tw[k];
                for q in 0..st {
                    let a0 = src[soff + q + st * k];
                    let a1 = src[soff + q + st * (k + m)];
                    dst[doff + q + st * (2 * k)] = a0 + a1;
                    dst[doff + q + st * (2 * k + 1)] = w1 * (a0 - a1);
                }
            }
        }
        3 => {
            // the imaginary part of the root W_3 is s sqrt(3)/2
            let r3 = Complex::new(0.0, tw[nr + 1].im);
            for k in 0..m {
                let w1 = tw[2 * k];
                let w2 = tw[2 * k + 1];
                for q in 0..st {
                    let a0 = src[soff + q + st * k];
                    let a1 = src[soff + q + st * (k + m)];
                    let a2 = src[soff + q + st * (k + 2 * m)];
                    let t1 = a1 + a2;
                    let t2 = r3 * (a1 - a2);
                    let t3 = a0 - t1.scale(0.5);
                    dst[doff + q + st * (3 * k)] = a0 + t1;
                    dst[doff + q + st * (3 * k + 1)] = w1 * (t3 + t2);
                    dst[doff + q + st * (3 * k + 2)] = w2 * (t3 - t2);
                }
            }
        }
        4 => {
            // the root W_4 is s i
            let r4 = tw[nr + 1];
            for k in 0..m {
                let w1 = tw[3 * k];
                let w2 = tw[3 * k + 1];
                let w3 = tw[3 * k + 2];
                for q in 0..st {
                    let a0 = src[soff + q + st * k];
                    let a1 = src[soff + q + st * (k + m)];
                    let a2 = src[soff + q + st * (k + 2 * m)];
                    let a3 = src[soff + q + st * (k + 3 * m)];
                    let t0 = a0 + a2;
                    let t1 = a0 - a2;
                    let t2 = a1 + a3;
                    let t3 = r4 * (a1 - a3);
                    dst[doff + q + st * (4 * k)] = t0 + t2;
                    dst[doff + q + st * (4 * k + 1)] = w1 * (t1 + t3);
                    dst[doff + q + st * (4 * k + 2)] = w2 * (t0 - t2);
                    dst[doff + q + st * (4 * k + 3)] = w3 * (t1 - t3);
                }
            }
        }
        _ => {
            // direct small DFT (radices 5, 7, 11, 13)
            for k in 0..m {
                for u in 0..p {
                    let w = if u == 0 {
                        Complex::ONE
                    } else {
                        tw[k * (p - 1) + u - 1]
                    };
                    for q in 0..st {
                        let mut acc = Complex::ZERO;
                        for t in 0..p {
                            acc += tw[nr + (u * t) % p] * src[soff + q + st * (k + t * m)];
                        }
                        dst[doff + q + st * (p * k + u)] = w * acc;
                    }
                }
            }
        }
    }
}

/// Mixed-radix transform of the `n` elements of `x` starting at
/// `xoff`; the `n` elements of `w` starting at `woff` are used as work
/// space. On return the result is stored in `x` (blueprint
/// `_fft_mixed_radix!`).
fn fft_mixed_radix(
    x: &mut [Complex],
    xoff: usize,
    w: &mut [Complex],
    woff: usize,
    n: usize,
    factors: &[usize],
    twiddles: &[Vec<Complex>],
) {
    let mut st = 1usize;
    let mut nc = n;
    for (i, &p) in factors.iter().enumerate() {
        let m = nc / p;
        if i % 2 == 0 {
            // Julia: isodd(i) with 1-based i -> 0-based even
            fft_stage(w, woff, x, xoff, p, m, st, &twiddles[i]);
        } else {
            fft_stage(x, xoff, w, woff, p, m, st, &twiddles[i]);
        }
        st *= p;
        nc = m;
    }
    if factors.len() % 2 == 1 {
        for j in 0..n {
            x[xoff + j] = w[woff + j];
        }
    }
}

/// Bluestein transform of `x[..n]` in place (blueprint
/// `_fft_bluestein!`). `scratch[0..M]` holds the chirp-modulated and
/// zero-padded data, `scratch[M..2M]` the work space of the inner
/// transforms.
fn fft_bluestein(x: &mut [Complex], plan: &FftPlan, scratch: &mut [Complex]) {
    let n = plan.n;
    let m = plan.bluestein_n;
    let chirp = &plan.chirp;
    let kernel = &plan.kernel_fft;

    for j in 0..n {
        scratch[j] = x[j] * chirp[j];
    }
    for j in n..m {
        scratch[j] = Complex::ZERO;
    }
    let (data, work) = scratch.split_at_mut(m);
    fft_mixed_radix(data, 0, work, 0, m, &plan.factors, &plan.twiddles);
    for j in 0..m {
        data[j] = (data[j] * kernel[j]).conj();
    }
    fft_mixed_radix(data, 0, work, 0, m, &plan.factors, &plan.twiddles);
    for j in 0..n {
        x[j] = chirp[j] * data[j].conj() / Complex::new(m as f64, 0.0);
    }
}

/// Build a new FFT plan (blueprint `_fft_make_plan`).
fn fft_make_plan(n: usize, direction: FftDirection) -> FftPlan {
    let s = direction.sign();
    let factors = fft_factorize(n);
    if let Some(factors) = factors {
        let twiddles = fft_twiddles(n, &factors, s);
        return FftPlan {
            n,
            direction,
            factors,
            twiddles,
            bluestein_n: 0,
            chirp: Vec::new(),
            kernel_fft: Vec::new(),
            scratch_len: n,
        };
    }

    // Bluestein plan; the inner transform of length M is mixed-radix
    let m = (2 * n - 1).next_power_of_two();
    let mfactors = fft_factorize(m).expect("M is a power of two");
    let mtwiddles = fft_twiddles(m, &mfactors, -1);

    let mut chirp = vec![Complex::ZERO; n];
    for j in 0..n {
        let jj = j as i64;
        chirp[j] = cispi(s as f64 * ((jj * jj) % (2 * n as i64)) as f64 / n as f64);
    }

    let mut kernel = vec![Complex::ZERO; m];
    kernel[0] = chirp[0].conj();
    for k in 1..n {
        kernel[k] = chirp[k].conj();
        kernel[m - k] = chirp[k].conj();
    }
    let mut work = vec![Complex::ZERO; m];
    fft_mixed_radix(&mut kernel, 0, &mut work, 0, m, &mfactors, &mtwiddles);

    FftPlan {
        n,
        direction,
        factors: mfactors,
        twiddles: mtwiddles,
        bluestein_n: m,
        chirp,
        kernel_fft: kernel,
        scratch_len: 2 * m,
    }
}

fn plan_cache() -> &'static Mutex<HashMap<(usize, i8), std::sync::Arc<FftPlan>>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, i8), std::sync::Arc<FftPlan>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Return the cached plan of the discrete Fourier transform of length
/// `n` (blueprint `plan_fft`). A new plan is created and cached on the
/// first call.
pub fn plan_fft(n: usize, direction: FftDirection) -> std::sync::Arc<FftPlan> {
    assert!(n >= 1, "the length of the transform must be >= 1, got {n}");
    let key = (n, direction.sign() as i8);
    let mut cache = plan_cache().lock().unwrap();
    cache
        .entry(key)
        .or_insert_with(|| std::sync::Arc::new(fft_make_plan(n, direction)))
        .clone()
}

/// Allocate a scratch vector for `plan`.
pub fn allocate_fft_scratch(plan: &FftPlan) -> Vec<Complex> {
    vec![Complex::ZERO; plan.scratch_len]
}

/// Remove all cached FFT plans.
pub fn clear_fft_plan_cache() {
    plan_cache().lock().unwrap().clear();
}

/// Transform `x` in place (blueprint `fft!`). The scratch vector must
/// have at least `plan.scratch_len` elements.
pub fn fft_exec(x: &mut [Complex], plan: &FftPlan, scratch: &mut [Complex]) {
    assert_eq!(
        x.len(),
        plan.n,
        "length of the data ({}) does not match the plan ({})",
        x.len(),
        plan.n
    );
    assert!(scratch.len() >= plan.scratch_len, "scratch is too short");
    if plan.bluestein_n > 0 {
        fft_bluestein(&mut x[..plan.n], plan, scratch);
    } else if plan.n > 1 {
        let n = plan.n;
        let work = &mut scratch[..n];
        fft_mixed_radix(&mut x[..n], 0, work, 0, n, &plan.factors, &plan.twiddles);
    }
}

/// In-place unnormalized transform of `x` (blueprint `fft!` without a
/// preallocated plan).
pub fn fft_inplace(x: &mut [Complex], direction: FftDirection) {
    let plan = plan_fft(x.len(), direction);
    let mut scratch = allocate_fft_scratch(&plan);
    fft_exec(x, &plan, &mut scratch);
}

/// Unnormalized transform of `x` returning a new vector (blueprint
/// `fft`).
pub fn fft(x: &[Complex], direction: FftDirection) -> Vec<Complex> {
    let mut y = x.to_vec();
    fft_inplace(&mut y, direction);
    y
}

/// Direct DFT of `x` (test oracle, O(n^2)).
pub fn dft_direct(x: &[Complex], direction: FftDirection) -> Vec<Complex> {
    let n = x.len();
    let s = direction.sign() as f64;
    let mut out = vec![Complex::ZERO; n];
    for (k, o) in out.iter_mut().enumerate() {
        for (j, &xj) in x.iter().enumerate() {
            *o += xj * cispi(2.0 * s * (j * k) as f64 / n as f64);
        }
    }
    out
}

/// Batched unnormalized discrete Fourier transform along the dimension
/// `dim` (0-based) of a 3-dimensional array with the row-major shape
/// `dims` (blueprint `fft_strided!`). Every line of `src` along `dim`
/// is transformed into the corresponding line of `dest`; `dest` and
/// `src` may be the same slice.
pub fn fft_strided(
    dest: &mut [Complex],
    src: &[Complex],
    dims: (usize, usize, usize),
    dim: usize,
    direction: FftDirection,
    plan: &FftPlan,
    scratch: &mut [Complex],
) {
    assert!(dim < 3, "dim must be in 0..3, got {dim}");
    let total = dims.0 * dims.1 * dims.2;
    assert_eq!(dest.len(), total, "dest and src must have the same size");
    assert_eq!(src.len(), total, "dest and src must have the same size");
    let n = match dim {
        0 => dims.0,
        1 => dims.1,
        _ => dims.2,
    };
    assert_eq!(
        plan.n, n,
        "length of the data ({n}) does not match the plan ({})",
        plan.n
    );
    assert_eq!(plan.direction, direction, "direction of the plan does not match");
    assert!(scratch.len() >= plan.scratch_len, "scratch is too short");

    // column-major (Fortran) addressing as in the blueprint: the
    // element (i, j, k) sits at `i + j*d0 + k*d0*d1`, so the stride of
    // the transformed axis is the product of the faster axes
    let (d0, d1, d2) = dims;
    let mut line = vec![Complex::ZERO; n];
    let mut do_line = |base: usize, stride: usize, line: &mut Vec<Complex>| {
        for (k, l) in line.iter_mut().enumerate() {
            *l = src[base + k * stride];
        }
        fft_exec(line, plan, scratch);
        for k in 0..n {
            dest[base + k * stride] = line[k];
        }
    };
    match dim {
        0 => {
            for k in 0..d2 {
                for j in 0..d1 {
                    do_line(j * d0 + k * d0 * d1, 1, &mut line);
                }
            }
        }
        1 => {
            for k in 0..d2 {
                for i in 0..d0 {
                    do_line(i + k * d0 * d1, d0, &mut line);
                }
            }
        }
        _ => {
            for ch in 0..(d0 * d1) {
                do_line(ch, d0 * d1, &mut line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn rand_vec(n: usize, state: &mut u64) -> Vec<Complex> {
        (0..n)
            .map(|_| {
                let re = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let im = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                Complex::new(re, im)
            })
            .collect()
    }

    #[test]
    fn mixed_radix_matches_direct() {
        // mixed-radix paths: powers of two (radix 4/2), 3, 5, 7, 11,
        // 13, and products
        let mut rng = 0xABCDEF01u64;
        for n in [
            1usize, 2, 3, 4, 5, 7, 8, 11, 13, 16, 25, 27, 32, 35, 48, 64, 121, 128, 169, 243, 256,
            315, 512, 1024,
        ] {
            let x = rand_vec(n, &mut rng);
            for dir in [FftDirection::Forward, FftDirection::Backward] {
                let got = fft(&x, dir);
                let want = dft_direct(&x, dir);
                let mut err = 0.0f64;
                for (g, w) in got.iter().zip(&want) {
                    err = err.max((*g - *w).abs());
                }
                assert!(err < 1e-13 * n as f64, "n {n} dir {dir:?}: {err}");
            }
        }
    }

    #[test]
    fn bluestein_matches_direct() {
        // prime and large-prime lengths take the Bluestein path
        let mut rng = 0x12349999u64;
        for n in [17usize, 19, 23, 29, 31, 37, 41, 97, 101, 127, 211, 997] {
            let x = rand_vec(n, &mut rng);
            for dir in [FftDirection::Forward, FftDirection::Backward] {
                let got = fft(&x, dir);
                let want = dft_direct(&x, dir);
                let mut err = 0.0f64;
                for (g, w) in got.iter().zip(&want) {
                    err = err.max((*g - *w).abs());
                }
                assert!(err < 1e-12 * n as f64, "n {n} dir {dir:?}: {err}");
            }
        }
    }

    #[test]
    fn round_trip_scales_by_n() {
        let mut rng = 0x55AA55AAu64;
        for n in [8usize, 12, 17, 60, 128] {
            let x = rand_vec(n, &mut rng);
            let mut y = x.clone();
            fft_inplace(&mut y, FftDirection::Forward);
            fft_inplace(&mut y, FftDirection::Backward);
            for j in 0..n {
                let want = x[j] * Complex::new(n as f64, 0.0);
                assert!((y[j] - want).abs() < 1e-9 * n as f64, "n {n} j {j}");
            }
        }
    }

    #[test]
    fn plan_cache_reuses() {
        let p1 = plan_fft(64, FftDirection::Forward);
        let p2 = plan_fft(64, FftDirection::Forward);
        assert!(std::sync::Arc::ptr_eq(&p1, &p2));
        let p3 = plan_fft(64, FftDirection::Backward);
        assert!(!std::sync::Arc::ptr_eq(&p1, &p3));
    }
}

/// In-place batched transform along the dimension `dim` of a
/// 3-dimensional array with the row-major shape `dims` (the
/// `dest === src` case of the blueprint `fft_strided!`): each line is
/// gathered into a scratch line before any write.
#[allow(clippy::too_many_arguments)]
pub fn fft_strided_in_place(
    x: &mut [Complex],
    dims: (usize, usize, usize),
    dim: usize,
    direction: FftDirection,
    plan: &FftPlan,
    scratch: &mut [Complex],
) {
    assert!(dim < 3, "dim must be in 0..3, got {dim}");
    let total = dims.0 * dims.1 * dims.2;
    assert_eq!(x.len(), total, "size mismatch");
    let n = match dim {
        0 => dims.0,
        1 => dims.1,
        _ => dims.2,
    };
    assert_eq!(plan.n, n, "length of the data ({n}) does not match the plan ({})", plan.n);
    assert_eq!(plan.direction, direction, "direction of the plan does not match");
    assert!(scratch.len() >= plan.scratch_len, "scratch is too short");
    // column-major (Fortran) addressing as in the blueprint: the
    // element (i, j, k) sits at `i + j*d0 + k*d0*d1`, so the stride of
    // the transformed axis is the product of the faster axes
    let (d0, d1, d2) = dims;
    let mut line = vec![Complex::ZERO; n];
    let mut do_line = |base: usize, stride: usize, line: &mut Vec<Complex>| {
        for (k, l) in line.iter_mut().enumerate() {
            *l = x[base + k * stride];
        }
        fft_exec(line, plan, scratch);
        for k in 0..n {
            x[base + k * stride] = line[k];
        }
    };
    match dim {
        0 => {
            for k in 0..d2 {
                for j in 0..d1 {
                    do_line(j * d0 + k * d0 * d1, 1, &mut line);
                }
            }
        }
        1 => {
            for k in 0..d2 {
                for i in 0..d0 {
                    do_line(i + k * d0 * d1, d0, &mut line);
                }
            }
        }
        _ => {
            for ch in 0..(d0 * d1) {
                do_line(ch, d0 * d1, &mut line);
            }
        }
    }
}
