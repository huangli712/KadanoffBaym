// Project : KadanoffBaym
// File    : src/convolution/ret.rs
// Purpose : retarded convolution kernels

//!
//! ### *Convolution* : ``G^{R}`` *Component*
//!
//! *Remarks* : *Retarded Convolution*
//!
//! The evaluation of ``C^{R}(t,t')`` at given time slice ``n`` is implemented
//! as follows:
//!
//! ```math
//! \begin{equation}
//! C^{R}(nh,mh) = C^{R}_1[A,f,B](n,m),
//! \end{equation}
//! ```
//!
//! where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
//! time axis (``\equiv \delta t``).
//!
//! ```math
//! \begin{equation}
//! C^{R}_1[A,f,B](n,m) = \int^{nh}_{mh} d\bar{t} \
//!     A^{R}(nh,\bar{t}) f(\bar{t}) B^{R}(\bar{t},mh).
//! \end{equation}
//! ```
//!
//! Actually, we implement the following equations:
//!
//! ```math
//! \begin{equation}
//! C^{R}_1[A,f,B](n,m) = h \sum^{n}_{j = m}
//!     w^{(k)}_{n-m,j-m} A^{R}_{n,j} f_j B^{R}_{j,m},
//!     \quad n > k,\ n - m > k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
//!     w^{(k)}_{n-m,j} A^{R}_{n,n-j} f_{n-j} \tilde{B}^{R}_{n-j,m},
//!     \quad n > k,\ n - m \le k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
//!     I^{(k)}_{m,n;j} \tilde{A}^{R}_{n,j} f_j \tilde{B}^{R}_{j,m},
//!     \quad n \le k.
//! \end{equation}
//! ```
//!
//! The ``\tilde{B}^{R}_{j,m}`` in the above equations denotes
//! ``B^{R}_{j,m}`` evaluated outside the stored domain ``j \ge m``,
//! reconstructed from the adjoint partner ``B^{\ddagger}``:
//!
//! ```math
//! \begin{equation}
//! \tilde{B}^{R}_{j,m} = -\left([B^{\ddagger}]^{R}_{m,j}\right)^{\dagger},
//!     \quad j < m.
//! \end{equation}
//! ```
//!
//! Analogous definitions also hold for ``\tilde{A}^{R}_{n,j}`` and
//! ``\tilde{B}^{R}_{n-j,m}`` that appear in the above equations.

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::types::{ContourGreen, GreenTimeStep, RetComp, TimeFunction};
use crate::weights::Integrator;

use crate::convolution::af;
use crate::linalg::{mul_acc_c, mul_acc_c2, mul_acc_c4};
use crate::types::{adjoint_block_into, triangular_row_offset};

/// Accumulate `s * [A^R * f * B^R](t_n, t_m)` into the target's ret row
/// `(n, m)` for `m = lo..=hi`. All block accesses stay inside the stored
/// triangles.
#[allow(clippy::too_many_arguments)]
fn conv_ret_kernel(
    tgt: &mut RetComp,
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
    lo: i64,
    hi: i64,
) {
    let k = integ.k;
    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;
    let mut adjbuf = vec![Complex::ZERO; bcc.size1 * bcc.size2];

    if n >= k {
        if s1 == 1 && s2a == 1 && s2b == 1 {
            conv_ret_kernel_d1(tgt, n, a, f, b, bcc, integ, s, ws, lo, hi);
            return;
        }
        if s1 == 2 && s2a == 2 && s2b == 2 && bcc.size1 == 2 && bcc.size2 == 2 {
            // d = 2 specialization (see mul_acc_c2)
            let mut adj2 = [Complex::ZERO; 4];
            for j in lo..=n {
                let aj = af(ws, a.raw_block_unchecked(n, j), &f, j, 2, 2);
                let mhi = j.min(hi);
                let mint = if j < n - k {
                    mhi.min(n - 2 * k - 2).min(j - k - 1)
                } else {
                    lo - 1
                };
                if mint >= lo {
                    for m in lo..=mint {
                        mul_acc_c2(
                            tgt.raw_block_unchecked_mut(n, m),
                            s,
                            aj,
                            b.raw_block_unchecked(j, m),
                        );
                    }
                }
                for m in (mint + 1).max(lo)..=mhi {
                    let w = integ.giw.at((n - m) as usize, (n - j) as usize);
                    mul_acc_c2(
                        tgt.raw_block_unchecked_mut(n, m),
                        s.scale(w),
                        aj,
                        b.raw_block_unchecked(j, m),
                    );
                }
            }
            for j in 1..=k {
                let mlo = (n - j + 1).max(lo);
                if mlo > hi {
                    continue;
                }
                let aj = af(ws, a.raw_block_unchecked(n, n - j), &f, n - j, 2, 2);
                for m in mlo..=hi {
                    let w = integ.giw.at((n - m) as usize, j as usize);
                    adjoint_block_into(
                        bcc.raw_block_unchecked(m, n - j),
                        &mut adj2,
                        bcc.size2,
                        bcc.size1,
                    );
                    mul_acc_c2(tgt.raw_block_unchecked_mut(n, m), -s.scale(w), aj, &adj2);
                }
            }
            return;
        }
        if s1 == 4 && s2a == 4 && s2b == 4 && bcc.size1 == 4 && bcc.size2 == 4 {
            // d = 4 specialization (see mul_acc_c4)
            let mut adj4 = [Complex::ZERO; 16];
            for j in lo..=n {
                let aj = af(ws, a.raw_block_unchecked(n, j), &f, j, 4, 4);
                let mhi = j.min(hi);
                let mint = if j < n - k {
                    mhi.min(n - 2 * k - 2).min(j - k - 1)
                } else {
                    lo - 1
                };
                if mint >= lo {
                    for m in lo..=mint {
                        mul_acc_c4(
                            tgt.raw_block_unchecked_mut(n, m),
                            s,
                            aj,
                            b.raw_block_unchecked(j, m),
                        );
                    }
                }
                for m in (mint + 1).max(lo)..=mhi {
                    let w = integ.giw.at((n - m) as usize, (n - j) as usize);
                    mul_acc_c4(
                        tgt.raw_block_unchecked_mut(n, m),
                        s.scale(w),
                        aj,
                        b.raw_block_unchecked(j, m),
                    );
                }
            }
            for j in 1..=k {
                let mlo = (n - j + 1).max(lo);
                if mlo > hi {
                    continue;
                }
                let aj = af(ws, a.raw_block_unchecked(n, n - j), &f, n - j, 4, 4);
                for m in mlo..=hi {
                    let w = integ.giw.at((n - m) as usize, j as usize);
                    adjoint_block_into(
                        bcc.raw_block_unchecked(m, n - j),
                        &mut adj4,
                        bcc.size2,
                        bcc.size1,
                    );
                    mul_acc_c4(tgt.raw_block_unchecked_mut(n, m), -s.scale(w), aj, &adj4);
                }
            }
            return;
        }
        // Case n - m > k: the stored triangles cover the whole range
        for j in lo..=n {
            let aj = af(ws, a.raw_block_unchecked(n, j), &f, j, s1, s2a);
            let mhi = j.min(hi);
            // interior columns: Omega[n-m][n-j] is exactly 1 for
            // m <= n-2k-2 and k < n-j < n-m-k, the weight multiply is
            // skipped there (bit-identical, ascending order kept)
            let mint = if j < n - k {
                mhi.min(n - 2 * k - 2).min(j - k - 1)
            } else {
                lo - 1
            };
            if mint >= lo {
                for m in lo..=mint {
                    let blk = tgt.raw_block_unchecked_mut(n, m);
                    mul_acc_c(blk, s, aj, b.raw_block_unchecked(j, m), s1, s2a, s2b);
                }
            }
            for m in (mint + 1).max(lo)..=mhi {
                let w = integ.giw.at((n - m) as usize, (n - j) as usize);
                let blk = tgt.raw_block_unchecked_mut(n, m);
                mul_acc_c(
                    blk,
                    s.scale(w),
                    aj,
                    b.raw_block_unchecked(j, m),
                    s1,
                    s2a,
                    s2b,
                );
            }
        }
        // Case n - m <= k: B^R is reconstructed outside the stored triangle
        for j in 1..=k {
            let mlo = (n - j + 1).max(lo);
            if mlo > hi {
                continue;
            }
            let aj = af(ws, a.raw_block_unchecked(n, n - j), &f, n - j, s1, s2a);
            for m in mlo..=hi {
                let w = integ.giw.at((n - m) as usize, j as usize);
                adjoint_block_into(
                    bcc.raw_block_unchecked(m, n - j),
                    &mut adjbuf,
                    bcc.size2,
                    bcc.size1,
                );
                let blk = tgt.raw_block_unchecked_mut(n, m);
                mul_acc_c(blk, -s.scale(w), aj, &adjbuf, s1, s2a, s2b);
            }
        }
    } else {
        // Case n <= k: A^R is reconstructed outside the stored triangle
        for j in 0..=k {
            if n >= j {
                let aj = af(ws, a.raw_block_unchecked(n, j), &f, j, s1, s2a);
                conv_ret_start(tgt, n, j, aj, s, b, bcc, integ, lo, hi, &mut adjbuf);
            } else {
                adjoint_block_into(
                    acc.raw_block_unchecked(j, n),
                    &mut adjbuf,
                    acc.size2,
                    acc.size1,
                );
                // the zero-copy view aliases its source, while
                // conv_ret_start uses adjbuf as its own scratch:
                // materialize through a local copy in this cold start-up
                // branch only (k + 1 calls per row)
                let asrc = adjbuf.clone();
                let aj = af(ws, &asrc, &f, j, s1, s2a);
                conv_ret_start(tgt, n, j, aj, -s, b, bcc, integ, lo, hi, &mut adjbuf);
            }
        }
    }
}

/// Scalar (`d = 1`) variant of [`conv_ret_kernel`] for `n >= k`.
///
/// The inner loops walk the contiguous triangle rows with running
/// offsets (cursor stepping) instead of re-deriving the block offset
/// from `(i, j)` on every block: at `d = 1` the addressing otherwise
/// dominates the single complex multiply-add per block.
#[allow(clippy::too_many_arguments)]
fn conv_ret_kernel_d1(
    tgt: &mut RetComp,
    n: i64,
    a: &RetComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
    lo: i64,
    hi: i64,
) {
    let k = integ.k;
    let om = &integ.giw.omega;
    let sg = &integ.giw.sigma;
    let bs = &integ.giw.big_s;
    let sk = (k + 1) as usize;
    let trow = tgt.raw_mut();
    let arow = a.raw();
    let brow = b.raw();
    let bccr = bcc.raw();
    // element offset of row n (d = 1: one element per block)
    let tn = triangular_row_offset(n as usize);
    for j in lo..=n {
        let ju = j as usize;
        let aj = af(ws, &arow[tn + ju..tn + ju + 1], &f, j, 1, 1);
        let mhi = j.min(hi);
        let mint = if j < n - k {
            mhi.min(n - 2 * k - 2).min(j - k - 1)
        } else {
            lo - 1
        };
        if mint >= lo {
            // interior: weights are one, so `tgt[n, .]` and `b[j, .]`
            // advance in lockstep over contiguous row segments
            let cnt = (mint - lo + 1) as usize;
            let tseg = &mut trow[tn + lo as usize..tn + lo as usize + cnt];
            let bseg = &brow[triangular_row_offset(ju) + lo as usize
                ..triangular_row_offset(ju) + lo as usize + cnt];
            if s.im == 0.0 {
                // real kernel factor (alpha*h): the complex multiply by
                // `s` degenerates to a real scale -- same rounding
                let c = s.re;
                let a0 = aj[0];
                for (tb, &bv) in tseg.iter_mut().zip(bseg) {
                    *tb += (a0 * bv).scale(c);
                }
            } else {
                for (tb, &bv) in tseg.iter_mut().zip(bseg) {
                    *tb += s * (aj[0] * bv);
                }
            }
        }
        let bj = triangular_row_offset(ju);
        // boundary stripe, weights Omega[n-m][n-j] with column
        // c = n-j: for c <= k (j >= n-k) the regimes are constant
        // omega[c] (m <= n-2k-2), the big_s window, then sigma; for
        // c > k (j < n-k) the whole stripe reads omega[j-m], because
        // the stripe starts at m >= j-k and thereby excludes the
        // unit-weight region m < j-k
        let c = (n - j) as usize;
        if c > k as usize {
            for m in (mint + 1).max(lo)..=mhi {
                let w = om[(j - m) as usize];
                let mu = m as usize;
                trow[tn + mu] += s.scale(w) * (aj[0] * brow[bj + mu]);
            }
        } else {
            let a_hi = mhi.min(n - 2 * k - 2);
            if a_hi >= (mint + 1).max(lo) {
                let m0 = (mint + 1).max(lo);
                let cnt = (a_hi - m0 + 1) as usize;
                let tseg = &mut trow[tn + m0 as usize..tn + m0 as usize + cnt];
                let bseg = &brow[bj + m0 as usize..bj + m0 as usize + cnt];
                let wc = s.scale(om[c]);
                for (tb, &bv) in tseg.iter_mut().zip(bseg) {
                    *tb += wc * (aj[0] * bv);
                }
            }
            for m in (mint + 1).max(lo).max(n - 2 * k - 1)..=mhi.min(n - k - 1) {
                let w = bs[((n - m - k - 1) as usize) * sk + c];
                let mu = m as usize;
                trow[tn + mu] += s.scale(w) * (aj[0] * brow[bj + mu]);
            }
            for m in (mint + 1).max(lo).max(n - k)..=mhi {
                let w = sg[((n - m) as usize) * sk + c];
                let mu = m as usize;
                trow[tn + mu] += s.scale(w) * (aj[0] * brow[bj + mu]);
            }
        }
    }
    for j in 1..=k {
        let mlo = (n - j + 1).max(lo);
        if mlo > hi {
            continue;
        }
        let aj = af(
            ws,
            &arow[tn + (n - j) as usize..tn + (n - j) as usize + 1],
            &f,
            n - j,
            1,
            1,
        );
        // bcc column (n-j): offset(m+1, n-j) - offset(m, n-j) = m + 1;
        // the weight Omega[n-m][j] has n-m <= k-1 here (m >= n-j+1),
        // so it always reads the sigma table
        let mut boff = triangular_row_offset(mlo as usize) + (n - j) as usize;
        for m in mlo..=hi {
            let w = sg[((n - m) as usize) * sk + j as usize];
            let sb = bccr[boff].conj();
            trow[tn + m as usize] += -s.scale(w) * (aj[0] * sb);
            boff += m as usize + 1;
        }
    }
}

/// The `j`-th term of the start-up case (`n <= k`).
/// `adjbuf` is the caller's scratch for the adjoint blocks.
#[allow(clippy::too_many_arguments)]
fn conv_ret_start(
    tgt: &mut RetComp,
    n: i64,
    j: i64,
    a: &[Complex],
    s: Complex,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    lo: i64,
    hi: i64,
    adjbuf: &mut [Complex],
) {
    let s1 = b.size1;
    let s2a = a.len() / s1; // dims of a: (s1, s2a)
    let s2b = b.size2;
    for m in lo..=hi {
        let w = integ.xiw.at(m as usize, n as usize, j as usize);
        if j >= m {
            let blk = tgt.raw_block_unchecked_mut(n, m);
            mul_acc_c(
                blk,
                s.scale(w),
                a,
                b.raw_block_unchecked(j, m),
                s1,
                s2a,
                s2b,
            );
        } else {
            adjoint_block_into(bcc.raw_block_unchecked(m, j), adjbuf, bcc.size2, bcc.size1);
            let blk = tgt.raw_block_unchecked_mut(n, m);
            mul_acc_c(blk, -s.scale(w), a, adjbuf, s1, s2a, s2b);
        }
    }
}

/// Evaluate `C^R(t_n, t_m) = [A^R * f * B^R](t_n, t_m)` for the whole
/// row `n`, overwriting old values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_ret_row(
    c_ret: &mut RetComp,
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    h: f64,
    ws: &mut ConvWorkspace,
) {
    for m in 0..=n {
        let blk = c_ret.block_mut(c_ret.idx(n, m));
        blk.fill(Complex::ZERO);
    }
    conv_ret_kernel(
        c_ret,
        n,
        a,
        acc,
        f,
        b,
        bcc,
        integ,
        Complex::new(h, 0.0),
        ws,
        0,
        n,
    );
}

/// [`conv_ret_row`] with an optional time-dependent `f_t` (one block per
/// time step).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn conv_ret(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f_t: Option<&[Vec<Complex>]>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
) {
    assert!((0..=(c.nt as i64)).contains(&n));
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let f = match f_t {
        Some(v) => FOp::T(v),
        None => FOp::None,
    };
    conv_ret_row(
        &mut c.ret, n, &a.ret, &acc.ret, f, &b.ret, &bcc.ret, integ, h, &mut ws,
    );
}

/// Increment the retarded row `tstp` of `C` by `alpha * [A * f * B]^R`.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_ret(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_ret_impl(&mut c.ret, tstp, a, acc, f, b, bcc, integ, h, alpha);
}

/// [`incr_convolution_ret`] with a `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_ret_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_ret_impl(&mut c.ret, tstp, a, acc, f, b, bcc, integ, h, alpha);
}

/// Shared body of `incr_convolution_ret` / `incr_convolution_ret_step`.
#[allow(clippy::too_many_arguments)]
fn incr_convolution_ret_impl(
    comp: &mut RetComp,
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    conv_ret_kernel(
        comp,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        0,
        tstp,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    fn dagger_partner(g: &ContourGreen) -> ContourGreen {
        // G-dagger partner: (G‡)^R(i,j) = -G^R(i,j) in the stored
        // triangle (tilde anti-hermiticity)
        let mut acc = g.clone();
        for i in 0..=(g.nt as i64) {
            for j in 0..=i {
                let v = g.get_ret(i, j, 0, 0);
                acc.set_ret(i, j, 0, 0, -v);
            }
        }
        acc
    }

    #[test]
    fn constant_retarded_kernel_exact() {
        // A^R = B^R = 1 (constant): C^R(t_n, t_m) = Int_{t_m}^{t_n} = (n-m) h
        let nt = 8;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let integ = Integrator::new(3);
        let h = 0.1;
        let acc = dagger_partner(&g);
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for n in 1..=nt as i64 {
            conv_ret_row(
                &mut cgreen.ret,
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.ret,
                &acc.ret,
                &integ,
                h,
                &mut ws,
            );
        }
        for n in 1..=nt as i64 {
            for m in 0..=n {
                let want = Complex::new((n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-13, "({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn start_up_matches_gregory() {
        // for n <= k both branches must agree with the n > k formula
        // continued on small grids: use k = 2, compare n = 1, 2 rows
        // against direct quadrature of constants (exact anyway)
        let nt = 6;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let integ = Integrator::new(2);
        let h = 0.1;
        let acc = dagger_partner(&g);
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for n in 1..=2 {
            conv_ret_row(
                &mut cgreen.ret,
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.ret,
                &acc.ret,
                &integ,
                h,
                &mut ws,
            );
        }
        for n in 1..=2 {
            for m in 0..=n {
                let want = Complex::new((n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-13, "({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn with_f_t_weight() {
        // f(t) = 2 constant: C = 2 * (n-m) h
        let nt = 8;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let fblocks: Vec<Vec<Complex>> = (0..=nt as i64 + 1)
            .map(|_| vec![Complex::new(2.0, 0.0)])
            .collect();
        let integ = Integrator::new(3);
        let h = 0.1;
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        let acc = dagger_partner(&g);
        conv_ret(
            5,
            &mut cgreen,
            &g,
            &acc,
            Some(&fblocks),
            &g,
            &acc,
            &integ,
            h,
        );
        for m in 0..=5 {
            let want = Complex::new(2.0 * (5 - m) as f64 * h, 0.0);
            let got = cgreen.get_ret(5, m, 0, 0);
            assert!((got - want).abs() < 1e-13);
        }
    }
}
