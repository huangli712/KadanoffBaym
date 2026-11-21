// Project : KadanoffBaym
// File    : src/convolution/workspace.rs
// Purpose : shared convolution workspace and block helpers

//!
//! *Remarks* : *Block-Target Kernels*
//!
//! The following private kernels implement the integration loops of the
//! retarded, left-mixing, and lesser convolutions. They accumulate the
//! results directly into the target blocks, i.e., `tgt[m] += s * (...)`,
//! where `tgt` is any object whose block accessor returns a mutable
//! matrix block, such as a time slice of a contour-ordered Green's
//! function, a `GreenTimeStep`, or a scratch vector of matrices. All
//! matrix products are evaluated by the `linalg` multiply-accumulate
//! kernels, and all intermediate results are kept in a preallocated
//! workspace `ws`. Thus, the kernels do not allocate memory which grows
//! with the number of time steps.
//!
//! The target blocks must not be aliased with the input blocks.
//!
//! The formulas, weights, and summation domains are the same as those in
//! the public `conv_*` functions.

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::TimeFunction;

use crate::linalg::mul_acc;

/// The optional middle function `f` of the convolution kernels.
#[derive(Clone, Copy)]
pub enum FOp<'a> {
    None,

    /// One block per time step.
    T(&'a [Vec<Complex>]),

    /// A contour function.
    Cf(&'a TimeFunction),
}

/// Workspace for the private convolution kernels: `af` stores `A f_j`,
/// `fb` stores `f_j B`, and `fbm` stores `f_0 B^M(tau_m)` for all
/// imaginary time points.
pub struct ConvWorkspace {
    pub(crate) af: DenseMat,
    pub(crate) fb: DenseMat,
    /// Cached `f_0 * B^M(tau_m)` blocks (or the plain `B^M` blocks when
    /// `f` is absent), flat and contiguous, built once per workspace
    /// lifetime by the full-contour driver.
    pub(crate) fbm: Vec<Complex>,
}

impl ConvWorkspace {
    pub fn new(s1a: usize, s2a: usize, s1b: usize, s2b: usize) -> Self {
        Self {
            af: DenseMat::zeros_shape(s1a, s1b),
            fb: DenseMat::zeros_shape(s2a, s2b),
            fbm: Vec::new(),
        }
    }

    /// Drop the cached `f_0 * B^M(tau_m)` blocks. The cache is valid
    /// only while the Matsubara blocks of the inputs (`f`, `B`) stay
    /// unchanged: call `reset` when a new convolution with different
    /// inputs starts on a reused workspace.
    pub fn reset(&mut self) {
        self.fbm.clear();
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

/// `f[j] * b`: for the identity `f` the input block is borrowed
/// directly (zero copy), otherwise the product goes into the workspace
/// slice `ws.fb` (dims `s2a x s2b`). The result lifetime is tied to both
/// `ws` and `b`.
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
