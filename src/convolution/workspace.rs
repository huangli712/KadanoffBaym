// Project : KadanoffBaym
// File    : src/convolution/workspace.rs
// Purpose : shared convolution workspace and block helpers
//           (blueprint julia/src/convolution/workspace.jl)

use crate::complex::Complex;
use crate::linalg::mat::DenseMat;
use crate::types::function::TimeFunction;

/// Accumulate `acc += w * (a * b)` for row-major blocks `a` (m x k) and
/// `b` (k x n); `acc` is m x n. This is the workhorse of all kernels.
#[inline]
pub(crate) fn mul_acc(acc: &mut [Complex], w: f64, a: &[Complex], b: &[Complex], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = Complex::ZERO;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            acc[i * n + j] += s.scale(w);
        }
    }
}

/// Accumulate `acc += w * (a * b)` with a complex weight.
#[inline]
pub(crate) fn mul_acc_c(acc: &mut [Complex], w: Complex, a: &[Complex], b: &[Complex], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = Complex::ZERO;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            acc[i * n + j] += w * s;
        }
    }
}

/// Adjoint (conjugate transpose) of a row-major block with `r` rows and
/// `c` columns, returned as a row-major block with `c` rows and `r`
/// columns.
#[inline]
pub(crate) fn adjoint_block(src: &[Complex], r: usize, c: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; r * c];
    for a in 0..c {
        for b in 0..r {
            out[a * r + b] = src[b * c + a].conj();
        }
    }
    out
}

/// The optional middle function `f` of the convolution kernels.
#[derive(Clone, Copy)]
pub enum FOp<'a> {
    None,
    /// Blocks indexed by Rust time (blueprint `VecArray` `f_t`).
    T(&'a [Vec<Complex>]),
    /// A contour function.
    Cf(&'a TimeFunction),
}

/// `a * f[j]` into the workspace (blueprint `_af!`). Returns a slice of
/// `ws.af` (dims `s1 x s2b`).
pub(crate) fn af<'a>(
    ws: &'a mut ConvWorkspace,
    a: &[Complex],
    f: &FOp,
    j: i64,
    s1: usize,
    s2a: usize,
) -> &'a [Complex] {
    match f {
        FOp::None => {
            ws.af.raw_mut()[..a.len()].copy_from_slice(a);
            &ws.af.raw()[..a.len()]
        }
        FOp::T(v) => {
            let fb = &v[j as usize];
            let s2b = fb.len() / s2a;
            let dst = ws.af.raw_mut();
            for q in 0..s1 * s2b {
                dst[q] = Complex::ZERO;
            }
            mul_acc(&mut dst[..s1 * s2b], 1.0, a, fb, s1, s2a, s2b);
            &ws.af.raw()[..s1 * s2b]
        }
        FOp::Cf(cf) => {
            let fb = cf.block(j);
            let s2b = fb.len() / s2a;
            let dst = ws.af.raw_mut();
            for q in 0..s1 * s2b {
                dst[q] = Complex::ZERO;
            }
            mul_acc(&mut dst[..s1 * s2b], 1.0, a, fb, s1, s2a, s2b);
            &ws.af.raw()[..s1 * s2b]
        }
    }
}

/// `f[j] * b` into the workspace (blueprint `_fb!`). Returns a slice of
/// `ws.fb` (dims `s2a x s2b`).
pub(crate) fn fb<'a>(
    ws: &'a mut ConvWorkspace,
    f: &FOp,
    j: i64,
    b: &[Complex],
    s1b: usize,
    s2b: usize,
) -> &'a [Complex] {
    match f {
        FOp::None => {
            ws.fb.raw_mut()[..b.len()].copy_from_slice(b);
            &ws.fb.raw()[..b.len()]
        }
        FOp::T(v) => {
            let fbk = &v[j as usize];
            let s2a = fbk.len() / s1b;
            let dst = ws.fb.raw_mut();
            for q in 0..s2a * s2b {
                dst[q] = Complex::ZERO;
            }
            mul_acc(&mut dst[..s2a * s2b], 1.0, fbk, b, s2a, s1b, s2b);
            &ws.fb.raw()[..s2a * s2b]
        }
        FOp::Cf(cf) => {
            let fbk = cf.block(j);
            let s2a = fbk.len() / s1b;
            let dst = ws.fb.raw_mut();
            for q in 0..s2a * s2b {
                dst[q] = Complex::ZERO;
            }
            mul_acc(&mut dst[..s2a * s2b], 1.0, fbk, b, s2a, s1b, s2b);
            &ws.fb.raw()[..s2a * s2b]
        }
    }
}

/// Workspace for the private convolution kernels (blueprint
/// `_ConvWorkspace`): `af` stores `A f_j`, `fb` stores `f_j B`, and
/// `fbm` stores `f_0 B^M(tau_m)` for all imaginary time points.
pub struct ConvWorkspace {
    pub af: DenseMat,
    pub fb: DenseMat,
    pub fbm: Vec<DenseMat>,
}

impl ConvWorkspace {
    pub fn new(s1a: usize, s2a: usize, s1b: usize, s2b: usize) -> Self {
        Self {
            af: DenseMat::zeros_shape(s1a, s1b),
            fb: DenseMat::zeros_shape(s2a, s2b),
            fbm: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_acc_matches_dense_product() {
        let a = [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0), Complex::new(0.0, -1.0), Complex::new(1.0, 2.0)];
        let b = [Complex::new(0.5, 0.0), Complex::new(1.0, 1.0), Complex::new(-1.0, 0.0), Complex::new(0.0, 2.0)];
        let mut acc = vec![Complex::ZERO; 4];
        mul_acc(&mut acc, 2.0, &a, &b, 2, 2, 2);
        let am = DenseMat::from_vec(2, 2, a.to_vec());
        let bm = DenseMat::from_vec(2, 2, b.to_vec());
        let want = am.mat_mul(&bm);
        for q in 0..4 {
            assert_eq!(acc[q], want.get(q / 2, q % 2).scale(2.0));
        }
    }
}
