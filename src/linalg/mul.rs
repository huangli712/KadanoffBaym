// Project : KadanoffBaym
// File    : src/linalg/mul.rs
// Purpose : matrix multiply-accumulate kernels on flat slices

//!
//! *Remarks* : *Sunk from `convolution/workspace.rs`*
//!
//! The kernels are pure matrix algebra primitives (the GEMM-accumulate
//! family next to `axpy_c`/`scale_c`), so they live in `linalg` and every
//! higher module — `convolution`, `dyson`, `vie2`, `fourier` — reaches
//! them downward.
//!
//! `acc += w * (a * b)` for row-major blocks; the dimension-specialized
//! leaves (`mul_acc_c2`/`c4` via `mul_acc_d`, the `_u` unweighted
//! variants via `unit2`/`unit4`/`unit8`) are bit-identical to the
//! generic kernel in accumulation order `(i, l, j)`.

use crate::complex::Complex;
use crate::linalg::DenseMat;

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
    // NOTE: dimension-specialized leaves (1x1x1 scalar, fully unrolled
    // 2x2x2) were tried inside this function and rejected by A/B
    // measurement: the generic row loop auto-vectorizes for n = 2, and
    // every extra entry branch cost the d = 2 kernels 3-30% wall time.
    // The callers' explicit 2x2 specialization lives in [`mul_acc_c2`].
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

/// Accumulate `acc += w * (a * b)` for fixed `D x D` blocks: the
/// compile-time-dimension specialization behind [`mul_acc_c2`] and
/// [`mul_acc_c4`]. Array references carry their length in the type, so
/// the block body has no length checks at all, and each `D` is
/// monomorphized into straight-line code. The accumulation order
/// (i, l, j) matches [`mul_acc_c`], keeping results bit-identical.
#[inline(always)]
fn mul_acc_d<const D: usize, const N: usize>(
    acc: &mut [Complex; N],
    w: Complex,
    a: &[Complex; N],
    b: &[Complex; N],
) {
    debug_assert_eq!(N, D * D);
    for i in 0..D {
        let mut o = [Complex::ZERO; D];
        for j in 0..D {
            o[j] = acc[i * D + j];
        }
        for l in 0..D {
            let al = a[i * D + l];
            for j in 0..D {
                o[j] += w * (al * b[l * D + j]);
            }
        }
        for j in 0..D {
            acc[i * D + j] = o[j];
        }
    }
}

/// Accumulate `acc += w * (a * b)` for fixed 2x2 blocks (the `d = 2`
/// specialization of [`mul_acc_c`]).
#[inline(always)]
pub(crate) fn mul_acc_c2(acc: &mut [Complex], w: Complex, a: &[Complex], b: &[Complex]) {
    mul_acc_d::<2, 4>(
        (&mut acc[..4]).try_into().unwrap(),
        w,
        (&a[..4]).try_into().unwrap(),
        (&b[..4]).try_into().unwrap(),
    );
}

/// Accumulate `acc += w * (a * b)` for fixed 4x4 blocks (the `d = 4`
/// specialization of [`mul_acc_c`]).
#[inline(always)]
pub(crate) fn mul_acc_c4(acc: &mut [Complex], w: Complex, a: &[Complex], b: &[Complex]) {
    mul_acc_d::<4, 16>(
        (&mut acc[..16]).try_into().unwrap(),
        w,
        (&a[..16]).try_into().unwrap(),
        (&b[..16]).try_into().unwrap(),
    );
}

/// Accumulate `acc += a * b` for fixed 2x2 blocks: the unweighted
/// variant used when the caller has pre-scaled the source row.
///
/// The caller's arithmetic becomes `acc += (alpha * a) * b` where the
/// weighted form is `acc += alpha * (a * b)`; the association differs,
/// so results can move in the last bits.
#[inline(always)]
pub(crate) fn mul_acc_c2_u(acc: &mut [Complex], a: &[Complex], b: &[Complex]) {
    unit2(
        (&mut acc[..4]).try_into().unwrap(),
        (&a[..4]).try_into().unwrap(),
        (&b[..4]).try_into().unwrap(),
    );
}

/// Accumulate `acc += a * b` for fixed 4x4 blocks (see
/// [`mul_acc_c2_u`]).
#[inline(always)]
pub(crate) fn mul_acc_c4_u(acc: &mut [Complex], a: &[Complex], b: &[Complex]) {
    unit4(
        (&mut acc[..16]).try_into().unwrap(),
        (&a[..16]).try_into().unwrap(),
        (&b[..16]).try_into().unwrap(),
    );
}

/// Accumulate `acc += a * b` for fixed 8x8 blocks: the `d = 8`
/// unweighted variant (see [`mul_acc_c2_u`]).
#[inline(always)]
pub(crate) fn mul_acc_c8_u(acc: &mut [Complex], a: &[Complex], b: &[Complex]) {
    unit8(
        (&mut acc[..64]).try_into().unwrap(),
        (&a[..64]).try_into().unwrap(),
        (&b[..64]).try_into().unwrap(),
    );
}

/// `out = a * b` for square `d x d` flat blocks, zero-allocation path
/// (the [`mul_acc`] workhorse on a preallocated target).
pub(crate) fn mul_sq_into(out: &mut DenseMat, a: &[Complex], b: &[Complex], d: usize) {
    let dst = out.raw_mut();
    dst.fill(Complex::ZERO);
    mul_acc(dst, 1.0, a, b, d, d, d);
}

/// [`mul_acc_c2_u`] body, same accumulation order (i, l, j).
#[inline(always)]
fn unit2(dst: &mut [Complex; 4], a: &[Complex; 4], b: &[Complex; 4]) {
    for i in 0..2 {
        let mut o = [Complex::ZERO; 2];
        for j in 0..2 {
            o[j] = dst[i * 2 + j];
        }
        for l in 0..2 {
            let al = a[i * 2 + l];
            for j in 0..2 {
                o[j] += al * b[l * 2 + j];
            }
        }
        for j in 0..2 {
            dst[i * 2 + j] = o[j];
        }
    }
}

/// [`mul_acc_c4_u`] body, same accumulation order (i, l, j).
#[inline(always)]
fn unit4(dst: &mut [Complex; 16], a: &[Complex; 16], b: &[Complex; 16]) {
    for i in 0..4 {
        let mut o = [Complex::ZERO; 4];
        for j in 0..4 {
            o[j] = dst[i * 4 + j];
        }
        for l in 0..4 {
            let al = a[i * 4 + l];
            for j in 0..4 {
                o[j] += al * b[l * 4 + j];
            }
        }
        for j in 0..4 {
            dst[i * 4 + j] = o[j];
        }
    }
}

/// [`mul_acc_c8_u`] body, same accumulation order (i, l, j).
#[inline(always)]
fn unit8(dst: &mut [Complex; 64], a: &[Complex; 64], b: &[Complex; 64]) {
    for i in 0..8 {
        let mut o = [Complex::ZERO; 8];
        for j in 0..8 {
            o[j] = dst[i * 8 + j];
        }
        for l in 0..8 {
            let al = a[i * 8 + l];
            for j in 0..8 {
                o[j] += al * b[l * 8 + j];
            }
        }
        for j in 0..8 {
            dst[i * 8 + j] = o[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::DenseMat;

    /// Deterministic splitmix64 stream of complex values in `[-1, 1)`.
    fn rand_c64() -> impl FnMut() -> Complex {
        let mut s = 0x243F6A8885A308D3u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        move || {
            Complex::new(
                (next() as f64 / u64::MAX as f64) * 2.0 - 1.0,
                (next() as f64 / u64::MAX as f64) * 2.0 - 1.0,
            )
        }
    }

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

    #[test]
    fn mul_acc_c2_matches_generic_bitwise() {
        let mut rc = rand_c64();
        for _ in 0..64 {
            let a: Vec<Complex> = (0..4).map(|_| rc()).collect();
            let b: Vec<Complex> = (0..4).map(|_| rc()).collect();
            let w = rc();
            let mut x = vec![Complex::ZERO; 4];
            let mut y = vec![Complex::ZERO; 4];
            mul_acc_c2(&mut x, w, &a, &b);
            mul_acc_c(&mut y, w, &a, &b, 2, 2, 2);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn mul_acc_c4_matches_generic_bitwise() {
        let mut rc = rand_c64();
        for _ in 0..64 {
            let a: Vec<Complex> = (0..16).map(|_| rc()).collect();
            let b: Vec<Complex> = (0..16).map(|_| rc()).collect();
            let w = rc();
            let mut x = vec![Complex::ZERO; 16];
            let mut y = vec![Complex::ZERO; 16];
            mul_acc_c4(&mut x, w, &a, &b);
            mul_acc_c(&mut y, w, &a, &b, 4, 4, 4);
            assert_eq!(x, y);
        }
    }
}
