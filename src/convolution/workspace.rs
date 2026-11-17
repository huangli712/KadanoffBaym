// Project : KadanoffBaym
// File    : src/convolution/workspace.rs
// Purpose : shared convolution workspace and block helpers

//!
//! *Remarks* : *Block-Target Kernels*
//!
//! The following private kernels implement the integration loops of the
//! retarded, left-mixing, and lesser convolutions. They accumulate the
//! results directly into the target blocks, i.e., `tgt[m] += s * (...)`,
//! where `tgt` is any object whose `getindex` returns a mutable matrix
//! block, such as a time slice of a contour-ordered Green's function, a
//! time slice of a `𝒻` object, or a scratch vector of matrices. All matrix
//! products are evaluated by `mul!`, and all intermediate results are kept
//! in a preallocated workspace `ws`. Thus, the kernels do not allocate
//! memory which grows with the number of time steps.
//!
//! The target blocks must not be aliased with the input blocks.
//!
//! The formulas, weights, and summation domains are the same as those in
//! the public `conv_*` functions. Please see [`NESSi`] Eqs. (110) - (122).

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::TimeFunction;

/// The optional middle function `f` of the convolution kernels.
#[derive(Clone, Copy)]
pub enum FOp<'a> {
    None,

    /// Blocks indexed by Rust time.
    T(&'a [Vec<Complex>]),

    /// A contour function.
    Cf(&'a TimeFunction),
}

/// Workspace for the private convolution kernels: `af` stores `A f_j`,
/// `fb` stores `f_j B`, and `fbm` stores `f_0 B^M(tau_m)` for all
/// imaginary time points.
pub struct ConvWorkspace {
    pub af: DenseMat,
    pub fb: DenseMat,
    /// Cached `f_0 * B^M(tau_m)` blocks (or the plain `B^M` blocks when
    /// `f` is absent), flat and contiguous, built once per workspace
    /// lifetime by the full-contour driver.
    pub fbm: Vec<Complex>,
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

/// Accumulate `acc += w * (a * b)` for row-major blocks `a` (m x k) and
/// `b` (k x n); `acc` is m x n. This is the workhorse of all kernels.
#[inline]
pub(crate) fn mul_acc(
    acc: &mut [Complex],
    w: f64,
    a: &[Complex],
    b: &[Complex],
    m: usize,
    k: usize,
    n: usize,
) {
    // row-axpy form: identical to the k-sum form for k = 1 and free of
    // bounds-checked index arithmetic otherwise; the real weight is
    // applied to the product's re/im parts separately (saves half of
    // the flops of a full complex multiply)
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let orow = &mut acc[i * n..(i + 1) * n];
        for (l, &al) in arow.iter().enumerate() {
            let brow = &b[l * n..(l + 1) * n];
            for (o, &bv) in orow.iter_mut().zip(brow) {
                let p = al * bv;
                o.re += w * p.re;
                o.im += w * p.im;
            }
        }
    }
}

/// Accumulate `acc += w * (a * b)` with a complex weight.
#[inline]
pub(crate) fn mul_acc_c(
    acc: &mut [Complex],
    w: Complex,
    a: &[Complex],
    b: &[Complex],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let orow = &mut acc[i * n..(i + 1) * n];
        for (l, &al) in arow.iter().enumerate() {
            let brow = &b[l * n..(l + 1) * n];
            for (o, &bv) in orow.iter_mut().zip(brow) {
                *o += w * (al * bv);
            }
        }
    }
}

/// `a * f[j]`: for the identity `f` the input block is borrowed
/// directly (zero copy), otherwise the product goes into the
/// workspace slice `ws.af` (dims `s1 x s2b`). The result lifetime is
/// tied to both `ws` and `a`.
pub(crate) fn af<'a>(
    ws: &'a mut ConvWorkspace,
    a: &'a [Complex],
    f: &FOp,
    j: i64,
    s1: usize,
    s2a: usize,
) -> &'a [Complex] {
    match f {
        FOp::None => a,
        FOp::T(v) => {
            let fb = &v[j as usize];
            let s2b = fb.len() / s2a;
            let dst = ws.af.raw_mut();
            dst[..s1 * s2b].fill(Complex::ZERO);
            mul_acc(&mut dst[..s1 * s2b], 1.0, a, fb, s1, s2a, s2b);
            &ws.af.raw()[..s1 * s2b]
        }
        FOp::Cf(cf) => {
            let fb = cf.block(j);
            let s2b = fb.len() / s2a;
            let dst = ws.af.raw_mut();
            dst[..s1 * s2b].fill(Complex::ZERO);
            mul_acc(&mut dst[..s1 * s2b], 1.0, a, fb, s1, s2a, s2b);
            &ws.af.raw()[..s1 * s2b]
        }
    }
}

/// `f[j] * b` into the workspace. Returns a slice of `ws.fb` (dims
/// `s2a x s2b`);
/// for `FOp::None` the input block is borrowed directly (zero copy).
pub fn fb<'a>(
    ws: &'a mut ConvWorkspace,
    f: &FOp,
    j: i64,
    b: &'a [Complex],
    s1b: usize,
    s2b: usize,
) -> &'a [Complex] {
    match f {
        FOp::None => b,
        FOp::T(v) => {
            let fbk = &v[j as usize];
            let s2a = fbk.len() / s1b;
            let dst = ws.fb.raw_mut();
            dst[..s2a * s2b].fill(Complex::ZERO);
            mul_acc(&mut dst[..s2a * s2b], 1.0, fbk, b, s2a, s1b, s2b);
            &ws.fb.raw()[..s2a * s2b]
        }
        FOp::Cf(cf) => {
            let fbk = cf.block(j);
            let s2a = fbk.len() / s1b;
            let dst = ws.fb.raw_mut();
            dst[..s2a * s2b].fill(Complex::ZERO);
            mul_acc(&mut dst[..s2a * s2b], 1.0, fbk, b, s2a, s1b, s2b);
            &ws.fb.raw()[..s2a * s2b]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_acc_matches_dense_product() {
        let a = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, -1.0),
            Complex::new(1.0, 2.0),
        ];
        let b = [
            Complex::new(0.5, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 2.0),
        ];
        let mut acc = vec![Complex::ZERO; 4];
        mul_acc(&mut acc, 2.0, &a, &b, 2, 2, 2);
        let am = DenseMat::from_slice(2, 2, &a);
        let bm = DenseMat::from_slice(2, 2, &b);
        let want = am.mat_mul(&bm);
        for (q, a) in acc.iter().enumerate() {
            assert_eq!(*a, want.get(q / 2, q % 2).scale(2.0));
        }
    }
}
