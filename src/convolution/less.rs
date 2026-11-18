// Project : KadanoffBaym
// File    : src/convolution/less.rs
// Purpose : lesser convolution kernels

//!
//! ### *Convolution* : ``G^{<}`` *Component*
//!
//! *Remarks* : *Lesser Convolution*
//!
//! The evaluation of ``C^{<}(t,t')`` at given time slice ``n`` is
//! implemented as follows:
//!
//! ```math
//! \begin{equation}
//! C^{<}(mh,nh) = C^{<}_{1}[A,f,B](m,n) +
//!                C^{<}_{2}[A,f,B](m,n) +
//!                C^{<}_{3}[A,f,B](m,n).
//! \end{equation}
//! ```
//!
//! where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
//! time axis (``\equiv \delta t``).
//!
//! ```math
//! \begin{equation}
//! C^{<}_{1}[A,f,B](m,n) = \int^{mh}_{0} d\bar{t}~
//!     A^{R}(mh,\bar{t}) f(\bar{t}) B^{<}(\bar{t},nh).
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{2}[A,f,B](m,n) = \int^{nh}_{0} d\bar{t}~
//!     A^{<}(mh,\bar{t}) f(\bar{t}) B^{A}(\bar{t},nh).
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{3}[A,f,B](m,n) = -i \int^{\beta}_{0} d\tau~
//!     A^{\rceil}(mh,\tau) f(0^{-}) B^{\lceil}(\tau,nh).
//! \end{equation}
//! ```
//!
//! Actually, we implement the following equations:
//!
//! ```math
//! \begin{equation}
//! C^{<}_{1}[A,f,B](m,n) = h\sum^{m}_{j=0}~
//!     w^{(k)}_{m,j} A^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m > k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{1}[A,f,B](m,n) = h\sum^{k}_{j=0}~
//!     w^{(k)}_{m,j} \tilde{A}^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m \le k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{2}[A,f,B](m,n) = h\sum^{n}_{j=0}~
//!     w^{(k)}_{n,j} A^{<}_{m,j} f_{j} B^{A}_{j,n}, \quad n > k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{2}[A,f,B](m,n) = h\sum^{k}_{j=0}~
//!     w^{(k)}_{n,j} A^{<}_{m,j} f_{j} \tilde{B}^{A}_{j,n}, \quad n \le k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{<}_{3}[A,f,B](m,n) = -i h_{\tau} \sum^{N_{\tau}}_{j=0}~
//!     w^{(k)}_{N_{\tau},j} A^{\rceil}_{m,j} f_{-1} B^{\lceil}_{j,n}.
//! \end{equation}
//! ```
//!
//! Because the advanced and the right-mixing components of contour-ordered
//! Green's functions are not stored, these quantities must be reconstructed
//! from the adjoint. For example:
//!
//! ```math
//! \begin{equation}
//! C^{\lceil}(\tau,t) =
//!     -\xi \left([C^{\ddagger}]^{\rceil}(t,\beta-\tau) \right)^{\dagger},
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{R}(t,t') = \left([C^{\ddagger}]^{A}(t',t)\right)^{\dagger}.
//! \end{equation}
//! ```

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, RetComp, Sign};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::{fb, mul_acc, mul_acc_c, mul_acc_c2, mul_acc_c4};
use crate::types::{adjoint_block_into, tri_off};

// ------------------------------------------------------------------
// lesser convolution kernels
// ------------------------------------------------------------------

/// The `C_1` part: accumulate `s * [A^R * f * B^<](t_m, t_n)` into
/// `tgt.less(m, n)` for `m` in `ms`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_ret_less_kernel(
    tgt: &mut LessComp,
    ms: (i64, i64),
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &LessComp,
    bcc: &LessComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
) {
    let n1r = n.max(integ.k);
    let mut adjbuf = vec![Complex::ZERO; bcc.size1 * bcc.size2];
    let mut adjbuf2 = vec![Complex::ZERO; acc.size1 * acc.size2];
    for j in 0..=n1r {
        if j <= n {
            let bj = fb(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
            conv_ret_less_col(
                tgt,
                n,
                ms,
                j,
                bj,
                s,
                a,
                acc,
                integ,
                b.size1,
                b.size2,
                &mut adjbuf2,
            );
        } else {
            adjoint_block_into(bcc.raw_block_at(n, j), &mut adjbuf, bcc.size2, bcc.size1);
            let bj = fb(ws, &f, j, &adjbuf, bcc.size2, bcc.size1);
            conv_ret_less_col(
                tgt,
                n,
                ms,
                j,
                bj,
                -s,
                a,
                acc,
                integ,
                bcc.size2,
                bcc.size1,
                &mut adjbuf2,
            );
        }
    }
}

/// The `j`-th term of the C_1 part.
/// `adjbuf` is the caller's scratch for the adjoint blocks.
#[allow(clippy::too_many_arguments)]
fn conv_ret_less_col(
    tgt: &mut LessComp,
    col: i64,
    ms: (i64, i64),
    j: i64,
    b: &[Complex],
    s: Complex,
    a: &RetComp,
    acc: &RetComp,
    integ: &Integrator,
    s1b: usize,
    _s2b: usize,
    adjbuf: &mut [Complex],
) {
    let k = integ.k;
    let s1 = a.size1;
    let s2a = a.size2;
    if s1 == 1 && s2a == 1 && s1b == 1 {
        conv_ret_less_col_d1(tgt, col, ms, j, b, s, a, acc, integ);
        return;
    }
    if s1 == 4 && s2a == 4 && s1b == 4 && acc.size1 == 4 && acc.size2 == 4 {
        // d = 4 specialization (see mul_acc_c4)
        let m_int = if j > k {
            (2 * k + 2).max(j + k + 1).max(ms.0)
        } else {
            ms.1 + 1
        };
        for m in ms.0..=ms.1.min(m_int - 1) {
            if m > k {
                if j > m {
                    continue;
                }
                let w = integ.giw.at(m as usize, j as usize);
                mul_acc_c4(
                    tgt.raw_block_at_mut(m, col),
                    s.scale(w),
                    a.raw_block_at(m, j),
                    b,
                );
            } else {
                if j > k + 1 {
                    continue;
                }
                let w = integ.giw.at(m as usize, j as usize);
                if j > m {
                    adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
                    let nb = -s.scale(w);
                    mul_acc_c4(tgt.raw_block_at_mut(m, col), nb, adjbuf, b);
                } else {
                    mul_acc_c4(
                        tgt.raw_block_at_mut(m, col),
                        s.scale(w),
                        a.raw_block_at(m, j),
                        b,
                    );
                }
            }
        }
        for m in m_int.max(ms.0)..=ms.1 {
            mul_acc_c4(tgt.raw_block_at_mut(m, col), s, a.raw_block_at(m, j), b);
        }
        return;
    }
    // interior rows: Omega[m][j] is exactly 1 for m >= 2k+2 and
    // k < j < m-k, the weight multiply is skipped there (ascending
    // order kept, bit-identical)
    let m_int = if j > k {
        (2 * k + 2).max(j + k + 1).max(ms.0)
    } else {
        ms.1 + 1
    };
    for m in ms.0..=ms.1.min(m_int - 1) {
        if m > k {
            if j > m {
                continue;
            }
            let w = integ.giw.at(m as usize, j as usize);
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, s.scale(w), a.raw_block_at(m, j), b, s1, s2a, s1b);
        } else {
            if j > k + 1 {
                continue;
            }
            let w = integ.giw.at(m as usize, j as usize);
            if j > m {
                adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, -s.scale(w), adjbuf, b, acc.size2, acc.size1, s1b);
            } else {
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, s.scale(w), a.raw_block_at(m, j), b, s1, s2a, s1b);
            }
        }
    }
    for m in m_int.max(ms.0)..=ms.1 {
        let blk = tgt.raw_block_at_mut(m, col);
        mul_acc_c(blk, s, a.raw_block_at(m, j), b, s1, s2a, s1b);
    }
}

/// Scalar (`d = 1`) variant of [`conv_ret_less_col`].
///
/// Loop form tuned by instruction-level measurement: the target less
/// column is sliced once per call (unit stride), the `A^R(m, j)` walk
/// uses a running offset cursor (row stride `m + 1`), and the five-case
/// `Omega` lookup is hoisted to the per-`j` case analysis instead of
/// the branchy `at()` per element.
#[allow(clippy::too_many_arguments)]
fn conv_ret_less_col_d1(
    tgt: &mut LessComp,
    col: i64,
    ms: (i64, i64),
    j: i64,
    b: &[Complex],
    s: Complex,
    a: &RetComp,
    acc: &RetComp,
    integ: &Integrator,
) {
    let k = integ.k;
    let m_int = if j > k {
        (2 * k + 2).max(j + k + 1).max(ms.0)
    } else {
        ms.1 + 1
    };
    let om = &integ.giw.omega;
    let sg = &integ.giw.sigma;
    let bs = &integ.giw.big_s;
    let sk = (k + 1) as usize;
    let b0 = b[0];
    // target column `col`: less rows are keyed by the later index, so
    // `(m, col)` for m = ms.0..=ms.1 is one contiguous segment
    let t0 = tri_off(col as usize) + ms.0 as usize;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize];
    let arow = a.raw();
    let accr = acc.raw();
    let accrow_j = tri_off(j as usize);
    // A^R(m, j) cursor: offset(m, j) = tri_off(m) + j, stride m + 1
    let mut aoff = tri_off(ms.0 as usize) + j as usize;
    for (ti, m) in (ms.0..=ms.1.min(m_int - 1)).enumerate() {
        let ti = ti as i64;
        if m > k {
            if j <= m {
                // hoisted five-case Omega for fixed j, varying m:
                // j <= k reads big_s in the m <= 2k+1 window, omega[j]
                // beyond it; k < j <= m always reads omega[m - j]
                let w = if j <= k {
                    if m <= 2 * k + 1 {
                        bs[(m - k - 1) as usize * sk + j as usize]
                    } else {
                        om[j as usize]
                    }
                } else {
                    om[(m - j) as usize]
                };
                tseg[ti as usize] += s.scale(w) * (arow[aoff] * b0);
            }
        } else if j <= k + 1 {
            // j == k+1 falls through every Omega case (weight 0), so
            // only j <= k reads the sigma table
            let w = if j <= k {
                sg[m as usize * sk + j as usize]
            } else {
                0.0
            };
            if j > m {
                let sb = accr[accrow_j + m as usize].conj();
                tseg[ti as usize] += -s.scale(w) * (sb * b0);
            } else {
                tseg[ti as usize] += s.scale(w) * (arow[aoff] * b0);
            }
        }
        aoff += m as usize + 1;
    }
    // interior: unit weights
    for m in m_int.max(ms.0)..=ms.1 {
        tseg[(m - ms.0) as usize] += s * (arow[aoff] * b0);
        aoff += m as usize + 1;
    }
}

/// The `C_2` part: accumulate `s * [A^< * f * B^A](t_m, t_n)` into
/// `tgt.less(m, n)`. With `diagcc = true` the diagonal block
/// `A^<(t_m, t_m)` is replaced by `-Acc^<(t_m, t_m)^dagger` (start-up
/// convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_less_adv_kernel(
    tgt: &mut LessComp,
    ms: (i64, i64),
    n: i64,
    a: &LessComp,
    acc: &LessComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
    diagcc: bool,
) {
    let n1r = n.max(integ.k);
    let mut adjbuf = vec![Complex::ZERO; bcc.size1 * bcc.size2];
    let mut adjbuf2 = vec![Complex::ZERO; acc.size1 * acc.size2];
    for j in 0..=n1r {
        let w = integ.giw.at(n as usize, j as usize);
        if j <= n {
            adjoint_block_into(bcc.raw_block_at(n, j), &mut adjbuf, bcc.size2, bcc.size1);
            let bj = fb(ws, &f, j, &adjbuf, bcc.size2, bcc.size1);
            conv_less_adv_col(
                tgt,
                n,
                diagcc,
                ms,
                j,
                bj,
                s.scale(w),
                a,
                acc,
                bcc.size2,
                bcc.size1,
                &mut adjbuf2,
            );
        } else {
            let bj = fb(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
            conv_less_adv_col(
                tgt,
                n,
                diagcc,
                ms,
                j,
                bj,
                -s.scale(w),
                a,
                acc,
                b.size1,
                b.size2,
                &mut adjbuf2,
            );
        }
    }
}

/// The `j`-th term of the C_2 part.
/// `adjbuf` is the caller's scratch for the adjoint blocks.
#[allow(clippy::too_many_arguments)]
fn conv_less_adv_col(
    tgt: &mut LessComp,
    col: i64,
    diagcc: bool,
    ms: (i64, i64),
    j: i64,
    b: &[Complex],
    alpha: Complex,
    a: &LessComp,
    acc: &LessComp,
    s1b: usize,
    _s2b: usize,
    adjbuf: &mut [Complex],
) {
    let s1 = a.size1;
    let s2a = a.size2;
    if s1 == 1 && s2a == 1 && s1b == 1 {
        conv_less_adv_col_d1(tgt, col, diagcc, ms, j, b, alpha, a, acc);
        return;
    }
    if s1 == 2 && s2a == 2 && s1b == 2 && acc.size1 == 2 && acc.size2 == 2 {
        // d = 2 specialization (see mul_acc_c2)
        for m in ms.0..=ms.1 {
            let blk = tgt.raw_block_at_mut(m, col);
            if j < m || (j == m && diagcc) {
                adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
                let nb = -alpha;
                mul_acc_c2(blk, nb, adjbuf, b);
            } else {
                mul_acc_c2(blk, alpha, a.raw_block_at(m, j), b);
            }
        }
        return;
    }
    if s1 == 4 && s2a == 4 && s1b == 4 && acc.size1 == 4 && acc.size2 == 4 {
        // d = 4 specialization (see mul_acc_c4)
        for m in ms.0..=ms.1 {
            let blk = tgt.raw_block_at_mut(m, col);
            if j < m || (j == m && diagcc) {
                adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
                let nb = -alpha;
                mul_acc_c4(blk, nb, adjbuf, b);
            } else {
                mul_acc_c4(blk, alpha, a.raw_block_at(m, j), b);
            }
        }
        return;
    }
    for m in ms.0..=ms.1 {
        if j < m || (j == m && diagcc) {
            adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, -alpha, adjbuf, b, acc.size2, acc.size1, s1b);
        } else {
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), b, s1, s2a, s1b);
        }
    }
}

/// Scalar (`d = 1`) variant of [`conv_less_adv_col`]: the target column
/// and the source column `A^<(m, j)` are sliced once (unit stride), the
/// adjoint walk uses a running offset cursor.
#[allow(clippy::too_many_arguments)]
fn conv_less_adv_col_d1(
    tgt: &mut LessComp,
    col: i64,
    diagcc: bool,
    ms: (i64, i64),
    j: i64,
    b: &[Complex],
    alpha: Complex,
    a: &LessComp,
    acc: &LessComp,
) {
    let b0 = b[0];
    let t0 = tri_off(col as usize) + ms.0 as usize;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize];
    let ar = a.raw();
    let acr = acc.raw();
    let ja = tri_off(j as usize);
    // acc^<(j, m): row key m varies, stride m + 1
    let mut accoff = tri_off(ms.0 as usize) + j as usize;
    for (ti, m) in (ms.0..=ms.1).enumerate() {
        if j < m || (j == m && diagcc) {
            let sb = acr[accoff].conj();
            tseg[ti] += -alpha * (sb * b0);
        } else {
            tseg[ti] += alpha * (ar[ja + m as usize] * b0);
        }
        accoff += m as usize + 1;
    }
}

/// The `C_3` part: accumulate `s * [A^⌉ * f_0 * B^⌈](t_m, t_n) / (-i)`
/// into `tgt.less(m, n)`; the factor `-i*dtau` must be included in `s`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_lmix_rmix_kernel(
    tgt: &mut LessComp,
    ms: (i64, i64),
    n: i64,
    a: &LmixComp,
    f0: Option<&[Complex]>,
    bcc: &LmixComp,
    integ: &Integrator,
    sign: Sign,
    s: Complex,
    _ws: &mut ConvWorkspace,
) {
    let ntau = a.ntau as i64;
    if a.size1 == 1 && a.size2 == 1 && bcc.size1 == 1 && bcc.size2 == 1 {
        conv_lmix_rmix_kernel_d1(tgt, ms, n, a, f0, bcc, integ, sign, s);
        return;
    }
    if a.size1 == 2 && a.size2 == 2 && bcc.size1 == 2 && bcc.size2 == 2 {
        // d = 2 specialization (see mul_acc_c2)
        let mut adjbuf = vec![Complex::ZERO; 4];
        let mut fbbuf: Vec<Complex> = Vec::new();
        for j in 0..=ntau {
            let w = integ.giw.at(ntau as usize, j as usize);
            adjoint_block_into(
                bcc.raw_block_at(n, ntau - j),
                &mut adjbuf,
                bcc.size2,
                bcc.size1,
            );
            let bj: &[Complex] = match f0 {
                Some(f0blk) => {
                    let s1f = f0blk.len() / bcc.size2;
                    fbbuf.clear();
                    fbbuf.resize(s1f * bcc.size1, Complex::ZERO);
                    mul_acc(&mut fbbuf, 1.0, f0blk, &adjbuf, s1f, bcc.size2, bcc.size1);
                    &fbbuf
                }
                None => &adjbuf,
            };
            let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
            for m in ms.0..=ms.1 {
                if bj.len() == 4 {
                    mul_acc_c2(tgt.raw_block_at_mut(m, n), alpha, a.raw_block_at(m, j), bj);
                } else {
                    // rectangular f_0 product: fall back to the generic
                    // block multiply with the generic path's `n`
                    let s1b = bj.len() / bcc.size1;
                    mul_acc_c(
                        tgt.raw_block_at_mut(m, n),
                        alpha,
                        a.raw_block_at(m, j),
                        bj,
                        2,
                        2,
                        s1b,
                    );
                }
            }
        }
        return;
    }
    if a.size1 == 4 && a.size2 == 4 && bcc.size1 == 4 && bcc.size2 == 4 {
        // d = 4 specialization (see mul_acc_c4)
        let mut adjbuf = vec![Complex::ZERO; 16];
        let mut fbbuf: Vec<Complex> = Vec::new();
        for j in 0..=ntau {
            let w = integ.giw.at(ntau as usize, j as usize);
            adjoint_block_into(
                bcc.raw_block_at(n, ntau - j),
                &mut adjbuf,
                bcc.size2,
                bcc.size1,
            );
            let bj: &[Complex] = match f0 {
                Some(f0blk) => {
                    let s1f = f0blk.len() / bcc.size2;
                    fbbuf.clear();
                    fbbuf.resize(s1f * bcc.size1, Complex::ZERO);
                    mul_acc(&mut fbbuf, 1.0, f0blk, &adjbuf, s1f, bcc.size2, bcc.size1);
                    &fbbuf
                }
                None => &adjbuf,
            };
            let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
            for m in ms.0..=ms.1 {
                if bj.len() == 16 {
                    mul_acc_c4(tgt.raw_block_at_mut(m, n), alpha, a.raw_block_at(m, j), bj);
                } else {
                    // rectangular f_0 product: fall back to the generic
                    // block multiply with the generic path's `n`
                    let s1b = bj.len() / bcc.size1;
                    mul_acc_c(
                        tgt.raw_block_at_mut(m, n),
                        alpha,
                        a.raw_block_at(m, j),
                        bj,
                        4,
                        4,
                        s1b,
                    );
                }
            }
        }
        return;
    }
    let mut adjbuf = vec![Complex::ZERO; bcc.size1 * bcc.size2];
    let mut fbbuf: Vec<Complex> = Vec::new();
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        adjoint_block_into(
            bcc.raw_block_at(n, ntau - j),
            &mut adjbuf,
            bcc.size2,
            bcc.size1,
        );
        let bj: &[Complex] = match f0 {
            Some(f0blk) => {
                let s1f = f0blk.len() / bcc.size2;
                fbbuf.clear();
                fbbuf.resize(s1f * bcc.size1, Complex::ZERO);
                mul_acc(&mut fbbuf, 1.0, f0blk, &adjbuf, s1f, bcc.size2, bcc.size1);
                &fbbuf
            }
            None => &adjbuf,
        };
        let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        let s1 = a.size1;
        let s2a = a.size2;
        let s1b = bj.len() / bcc.size1;
        for m in ms.0..=ms.1 {
            let blk = tgt.raw_block_at_mut(m, n);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), bj, s1, s2a, s1b);
        }
    }
}

/// Scalar (`d = 1`) variant of [`conv_lmix_rmix_kernel`].
#[allow(clippy::too_many_arguments)]
fn conv_lmix_rmix_kernel_d1(
    tgt: &mut LessComp,
    ms: (i64, i64),
    n: i64,
    a: &LmixComp,
    f0: Option<&[Complex]>,
    bcc: &LmixComp,
    integ: &Integrator,
    sign: Sign,
    s: Complex,
) {
    let ntau = a.ntau as i64;
    let wstride = a.ntau + 1;
    let t0 = tri_off(n as usize) + ms.0 as usize;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize];
    let ar = a.raw();
    let bcr = bcc.raw();
    let bcrow = n as usize * wstride;
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        let sb = bcr[bcrow + (ntau - j) as usize].conj();
        let bj = match f0 {
            Some(f0blk) => f0blk[0] * sb,
            None => sb,
        };
        let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        // A^⌉(t_m, tau_j): stride ntau + 1 in m
        let mut aoff = ms.0 as usize * wstride + j as usize;
        for tb in tseg.iter_mut() {
            *tb += alpha * (ar[aoff] * bj);
            aoff += wstride;
        }
    }
}

/// Start-up convention: `diagcc = true` iff `tstp <= 2k + 2` (0-based
/// `tstp`). In `incr_convolution_less` the diagonal block `A^<(t,t)` of the
/// term `A^< * f * B^A` is taken as `-Acc^<(t,t)^dagger` up to that time
/// step. The two choices agree if `Acc` is the adjoint partner of `A`.
pub(crate) fn incr_less_diagcc(n: i64, integ: &Integrator) -> bool {
    n <= 2 * integ.k + 2
}

// ------------------------------------------------------------------
// green- and timestep-target drivers
// ------------------------------------------------------------------

/// Increment `C^<(t', t)` (`t' <= t = tstp`) by
/// `alpha * [A * f * B]^<`, the sum of the three partial lesser
/// convolutions.
///
/// *Remarks* : *Partial Lesser Convolutions*
///
/// The lesser component of `A ∗ f ∗ B` consists of three terms:
///
/// ```math
/// \begin{equation}
/// C^{<} = A^{R} \ast f \ast B^{<} + A^{<} \ast f \ast B^{A}
///       + A^{\rceil} \ast f \ast B^{\lceil}.
/// \end{equation}
/// ```
///
/// They are evaluated by `incr_convolution_ret_less`,
/// `incr_convolution_less_adv`, and `incr_convolution_lmix_rmix`,
/// respectively. In the time stepping of the Dyson and VIE2 equations, only
/// the latter two terms enter the source term, while the first one belongs
/// to the implicit linear system. None of them reads the target `C`.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_less(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_less_impl(&mut c.less, tstp, a, acc, f, b, bcc, integ, beta, h, alpha);
}

/// [`incr_convolution_less`] with a `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_less_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_less_impl(&mut c.less, tstp, a, acc, f, b, bcc, integ, beta, h, alpha);
}

/// Shared body of [`incr_convolution_less`] / [`incr_convolution_less_step`].
#[allow(clippy::too_many_arguments)]
fn incr_convolution_less_impl(
    comp: &mut LessComp,
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    let diagcc = incr_less_diagcc(tstp, integ);
    conv_ret_less_kernel(
        comp,
        (0, tstp),
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.less,
        &bcc.less,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
    );
    conv_less_adv_kernel(
        comp,
        (0, tstp),
        tstp,
        &a.less,
        &acc.less,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        diagcc,
    );
    let f0 = f.map(|cf| cf.block(MATSUBARA));
    conv_lmix_rmix_kernel(
        comp,
        (0, tstp),
        tstp,
        &a.lmix,
        f0,
        &bcc.lmix,
        integ,
        a.sign,
        alpha * Complex::new(0.0, -(beta / a.ntau as f64)),
        &mut ws,
    );
}

/// Partial increment: [`incr_convolution_ret_less`].
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_ret_less(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    _beta: f64,
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
    conv_ret_less_kernel(
        &mut c.less,
        (0, tstp),
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.less,
        &bcc.less,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
    );
}

/// Partial increment: [`incr_convolution_less_adv`].
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_less_adv(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    _beta: f64,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_less_adv_impl(&mut c.less, tstp, a, acc, f, b, bcc, integ, h, alpha);
}

/// Partial slice-target increment: [`incr_convolution_less_adv`] with a
/// `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_less_adv_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    _beta: f64,
    h: f64,
    alpha: Complex,
) {
    incr_convolution_less_adv_impl(&mut c.less, tstp, a, acc, f, b, bcc, integ, h, alpha);
}

/// Shared body of [`incr_convolution_less_adv`] /
/// [`incr_convolution_less_adv_step`].
#[allow(clippy::too_many_arguments)]
fn incr_convolution_less_adv_impl(
    comp: &mut LessComp,
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
    // The *partial* less_adv kernel defaults to
    // `diagcc = false` and the standard Dyson/VIE2 solvers use it without
    // the start-up substitution (only the combined `incr_convolution_less`
    // and the pseudo-particle VIE2 time step apply the start-up rule).
    conv_less_adv_kernel(
        comp,
        (0, tstp),
        tstp,
        &a.less,
        &acc.less,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        false,
    );
}

/// Partial increment: [`incr_convolution_lmix_rmix`].
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_lmix_rmix(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    _acc: &ContourGreen,
    f: Option<&TimeFunction>,
    _b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    _h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, bcc.size1, bcc.size2);
    let f0 = f.map(|cf| cf.block(MATSUBARA));
    conv_lmix_rmix_kernel(
        &mut c.less,
        (0, tstp),
        tstp,
        &a.lmix,
        f0,
        &bcc.lmix,
        integ,
        a.sign,
        alpha * Complex::new(0.0, -(beta / a.ntau as f64)),
        &mut ws,
    );
}

/// Partial slice-target increment: [`incr_convolution_lmix_rmix`] with a
/// `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_lmix_rmix_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    _h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let f0 = f.map(|cf| cf.block(MATSUBARA));
    let _ = acc;
    conv_lmix_rmix_kernel(
        &mut c.less,
        (0, tstp),
        tstp,
        &a.lmix,
        f0,
        &bcc.lmix,
        integ,
        a.sign,
        alpha * Complex::new(0.0, -(beta / a.ntau as f64)),
        &mut ws,
    );
}

// ------------------------------------------------------------------
// Scratch-buffer start-up kernels: the end-point solves accumulate into
// a private row buffer instead of a stored Green's function.
// ------------------------------------------------------------------

/// Buffer-target variant of [`incr_convolution_lmix_rmix`]: for every
/// `m` in `0..=tstp`, `buf[m] += alpha * [A^⌉ * B^⌈]`.
#[allow(clippy::too_many_arguments)]
pub fn incr_lmix_rmix_buf(
    buf: &mut [Vec<Complex>],
    tstp: i64,
    a: &ContourGreen,
    f0: Option<&[Complex]>,
    bcc: &ContourGreen,
    integ: &Integrator,
    sign: Sign,
    beta: f64,
    _h: f64,
    alpha: Complex,
    _ws: &mut ConvWorkspace,
) {
    // s = alpha * (-i * delta_tau), as in the public green-target version
    let dtau = beta / a.ntau as f64;
    let s = alpha * Complex::new(0.0, -dtau);
    let ntau = a.ntau as i64;
    if a.lmix.size1 == 1 && a.lmix.size2 == 1 && bcc.lmix.size1 == 1 && bcc.lmix.size2 == 1 {
        incr_lmix_rmix_buf_d1(buf, tstp, a, f0, bcc, integ, sign, s);
        return;
    }
    if a.lmix.size1 == 2 && a.lmix.size2 == 2 && bcc.lmix.size1 == 2 && bcc.lmix.size2 == 2 {
        // d = 2 specialization (see mul_acc_c2)
        let ntau = a.ntau as i64;
        let mut adjbuf = vec![Complex::ZERO; 4];
        let mut fbbuf: Vec<Complex> = Vec::new();
        for j in 0..=ntau {
            let w = integ.giw.at(ntau as usize, j as usize);
            // B^ceil(tau_j, t_n) = B^rangle(t_n, tau_{N-j})^dagger
            adjoint_block_into(
                bcc.lmix.raw_block_at(tstp, ntau - j),
                &mut adjbuf,
                bcc.lmix.size2,
                bcc.lmix.size1,
            );
            let bj: &[Complex] = match f0 {
                Some(f0blk) => {
                    let s1f = f0blk.len() / bcc.lmix.size2;
                    fbbuf.clear();
                    fbbuf.resize(s1f * bcc.lmix.size1, Complex::ZERO);
                    mul_acc(
                        &mut fbbuf,
                        1.0,
                        f0blk,
                        &adjbuf,
                        s1f,
                        bcc.lmix.size2,
                        bcc.lmix.size1,
                    );
                    &fbbuf
                }
                None => &adjbuf,
            };
            let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
            let rect = bj.len() != 4;
            let s1b = if rect { bj.len() / 2 } else { 2 };
            let s2b = if rect { bj.len() / s1b } else { 2 };
            for (m, bufm) in buf.iter_mut().enumerate() {
                let ablk = a.lmix.raw_block_at(m as i64, j);
                if rect {
                    mul_acc_c(bufm, alpha, ablk, bj, 2, 2, s2b);
                } else {
                    mul_acc_c2(bufm, alpha, ablk, bj);
                }
            }
        }
        return;
    }
    if a.lmix.size1 == 4 && a.lmix.size2 == 4 && bcc.lmix.size1 == 4 && bcc.lmix.size2 == 4 {
        // d = 4 specialization (see mul_acc_c4)
        let ntau = a.ntau as i64;
        let mut adjbuf = vec![Complex::ZERO; 16];
        let mut fbbuf: Vec<Complex> = Vec::new();
        for j in 0..=ntau {
            let w = integ.giw.at(ntau as usize, j as usize);
            adjoint_block_into(
                bcc.lmix.raw_block_at(tstp, ntau - j),
                &mut adjbuf,
                bcc.lmix.size2,
                bcc.lmix.size1,
            );
            let bj: &[Complex] = match f0 {
                Some(f0blk) => {
                    let s1f = f0blk.len() / bcc.lmix.size2;
                    fbbuf.clear();
                    fbbuf.resize(s1f * bcc.lmix.size1, Complex::ZERO);
                    mul_acc(
                        &mut fbbuf,
                        1.0,
                        f0blk,
                        &adjbuf,
                        s1f,
                        bcc.lmix.size2,
                        bcc.lmix.size1,
                    );
                    &fbbuf
                }
                None => &adjbuf,
            };
            let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
            let rect = bj.len() != 16;
            let s1b = if rect { bj.len() / 4 } else { 4 };
            let s2b = if rect { bj.len() / s1b } else { 4 };
            for (m, bufm) in buf.iter_mut().enumerate() {
                let ablk = a.lmix.raw_block_at(m as i64, j);
                if rect {
                    mul_acc_c(bufm, alpha, ablk, bj, 4, 4, s2b);
                } else {
                    mul_acc_c4(bufm, alpha, ablk, bj);
                }
            }
        }
        return;
    }
    let mut adjbuf = vec![Complex::ZERO; bcc.lmix.size1 * bcc.lmix.size2];
    let mut fbbuf: Vec<Complex> = Vec::new();
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        adjoint_block_into(
            bcc.lmix.raw_block_at(tstp, ntau - j),
            &mut adjbuf,
            bcc.lmix.size2,
            bcc.lmix.size1,
        );
        let bj: &[Complex] = match f0 {
            Some(f0blk) => {
                let s1f = f0blk.len() / bcc.lmix.size2;
                fbbuf.clear();
                fbbuf.resize(s1f * bcc.lmix.size1, Complex::ZERO);
                mul_acc(
                    &mut fbbuf,
                    1.0,
                    f0blk,
                    &adjbuf,
                    s1f,
                    bcc.lmix.size2,
                    bcc.lmix.size1,
                );
                &fbbuf
            }
            None => &adjbuf,
        };
        let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        let s1 = a.lmix.size1;
        let s2a = a.lmix.size2;
        let s1b = bj.len() / bcc.lmix.size1;
        let s2b = bj.len() / s1b;
        for (m, bufm) in buf.iter_mut().enumerate() {
            mul_acc_c(
                bufm,
                alpha,
                a.lmix.raw_block_at(m as i64, j),
                bj,
                s1,
                s2a,
                s2b,
            );
        }
    }
}

/// Scalar (`d = 1`) variant of [`incr_lmix_rmix_buf`]: no adjoint block or
/// product scratch is needed, the adjoint of a scalar is its conjugate.
#[allow(clippy::too_many_arguments)]
fn incr_lmix_rmix_buf_d1(
    buf: &mut [Vec<Complex>],
    tstp: i64,
    a: &ContourGreen,
    f0: Option<&[Complex]>,
    bcc: &ContourGreen,
    integ: &Integrator,
    sign: Sign,
    s: Complex,
) {
    let ntau = a.ntau as i64;
    let wstride = a.ntau + 1;
    let ar = a.lmix.raw();
    let bcr = bcc.lmix.raw();
    let bcrow = tstp as usize * wstride;
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        let sb = bcr[bcrow + (ntau - j) as usize].conj();
        let bj = match f0 {
            Some(f0blk) => f0blk[0] * sb,
            None => sb,
        };
        let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        // A^⌉(t_m, tau_j): stride ntau + 1 in m
        let mut aoff = j as usize;
        for bufm in buf.iter_mut() {
            bufm[0] += alpha * (ar[aoff] * bj);
            aoff += wstride;
        }
    }
}

/// Buffer-target variant of [`incr_convolution_less_adv`]:
/// `buf[m] += alpha * [Sigma^< * G^A](m, tstp)`.
#[allow(clippy::too_many_arguments)]
pub fn incr_less_adv_buf(
    buf: &mut [Vec<Complex>],
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: &FOp,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    _beta: f64,
    h: f64,
    alpha: Complex,
    ws: &mut ConvWorkspace,
    diagcc: bool,
) {
    // s = alpha * h, as in the public green-target version
    let s = alpha * Complex::new(h, 0.0);
    let n1r = tstp.max(integ.k);
    let mut adjbuf = vec![Complex::ZERO; bcc.ret.size1 * bcc.ret.size2];
    let mut adjbuf2 = vec![Complex::ZERO; acc.less.size1 * acc.less.size2];
    for j in 0..=n1r {
        let w = integ.giw.at(tstp as usize, j as usize);
        if j <= tstp {
            adjoint_block_into(
                bcc.ret.raw_block_at(tstp, j),
                &mut adjbuf,
                bcc.ret.size2,
                bcc.ret.size1,
            );
            let bj = fb(ws, f, j, &adjbuf, bcc.ret.size2, bcc.ret.size1);
            let (s1b, s2b) = (bcc.ret.size2, bcc.ret.size1);
            conv_less_adv_col_buf(
                buf,
                tstp,
                j,
                bj,
                s * Complex::new(w, 0.0),
                a,
                acc,
                diagcc,
                s1b,
                s2b,
                &mut adjbuf2,
            );
        } else {
            let braw = b.ret.raw_block_at(j, tstp);
            let bj = fb(ws, f, j, braw, b.ret.size1, b.ret.size2);
            let (s1b, s2b) = (b.ret.size1, b.ret.size2);
            conv_less_adv_col_buf(
                buf,
                tstp,
                j,
                bj,
                -s * Complex::new(w, 0.0),
                a,
                acc,
                diagcc,
                s1b,
                s2b,
                &mut adjbuf,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_less_adv_col_buf(
    buf: &mut [Vec<Complex>],
    _n: i64,
    j: i64,
    b: &[Complex],
    alpha: Complex,
    a: &ContourGreen,
    acc: &ContourGreen,
    diagcc: bool,
    _s1b: usize,
    s2b: usize,
    adjbuf: &mut [Complex],
) {
    let s1 = a.less.size1;
    let s2a = a.less.size2;
    if s1 == 1 && s2a == 1 && s2b == 1 {
        conv_less_adv_col_buf_d1(buf, j, b, alpha, a, acc, diagcc);
        return;
    }
    if s1 == 2 && s2a == 2 && s2b == 2 && acc.less.size1 == 2 && acc.less.size2 == 2 {
        // d = 2 specialization (see mul_acc_c2)
        for (m, bufm) in buf.iter_mut().enumerate() {
            let m = m as i64;
            if j < m || (j == m && diagcc) {
                adjoint_block_into(
                    acc.less.raw_block_at(j, m),
                    adjbuf,
                    acc.less.size2,
                    acc.less.size1,
                );
                let nb = -alpha;
                mul_acc_c2(bufm, nb, adjbuf, b);
            } else {
                mul_acc_c2(bufm, alpha, a.less.raw_block_at(m, j), b);
            }
        }
        return;
    }
    if s1 == 4 && s2a == 4 && s2b == 4 && acc.less.size1 == 4 && acc.less.size2 == 4 {
        // d = 4 specialization (see mul_acc_c4)
        for (m, bufm) in buf.iter_mut().enumerate() {
            let m = m as i64;
            if j < m || (j == m && diagcc) {
                adjoint_block_into(
                    acc.less.raw_block_at(j, m),
                    adjbuf,
                    acc.less.size2,
                    acc.less.size1,
                );
                let nb = -alpha;
                mul_acc_c4(bufm, nb, adjbuf, b);
            } else {
                mul_acc_c4(bufm, alpha, a.less.raw_block_at(m, j), b);
            }
        }
        return;
    }
    for (m, bufm) in buf.iter_mut().enumerate() {
        let m = m as i64;
        if j < m || (j == m && diagcc) {
            adjoint_block_into(
                acc.less.raw_block_at(j, m),
                adjbuf,
                acc.less.size2,
                acc.less.size1,
            );
            mul_acc_c(bufm, -alpha, adjbuf, b, acc.less.size2, acc.less.size1, s2b);
        } else {
            let adir = a.less.raw_block_at(m, j);
            mul_acc_c(bufm, alpha, adir, b, s1, s2a, s2b);
        }
    }
}

/// Scalar (`d = 1`) variant of [`conv_less_adv_col_buf`].
fn conv_less_adv_col_buf_d1(
    buf: &mut [Vec<Complex>],
    j: i64,
    b: &[Complex],
    alpha: Complex,
    a: &ContourGreen,
    acc: &ContourGreen,
    diagcc: bool,
) {
    let b0 = b[0];
    let ar = a.less.raw();
    let acr = acc.less.raw();
    let ja = tri_off(j as usize);
    // acc^<(j, m): row key m varies, stride m + 1; buf covers m = 0..
    let mut accoff = j as usize;
    for (mu, bufm) in buf.iter_mut().enumerate() {
        let m = mu as i64;
        if j < m || (j == m && diagcc) {
            let sb = acr[accoff].conj();
            bufm[0] += -alpha * (sb * b0);
        } else {
            bufm[0] += alpha * (ar[ja + mu] * b0);
        }
        accoff += mu + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::Contour;

    fn setup(nt: i64, ntau: i64) -> (Contour, ContourGreen, ContourGreen) {
        let c = Contour::new(nt, ntau, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
            for j in i..=nt {
                g.set_less(i, j, 0, 0, Complex::ONE);
            }
            for m in 0..=ntau {
                g.set_lmix(i, m, 0, 0, Complex::ONE);
            }
        }
        let mut acc = g.clone();
        for i in 0..=nt {
            for j in 0..=i {
                let v = g.get_ret(i, j, 0, 0);
                acc.set_ret(i, j, 0, 0, -v);
            }
            for j in i..=nt {
                let v = g.get_less(i, j, 0, 0);
                acc.set_less(i, j, 0, 0, -v);
            }
        }
        (c, g, acc)
    }

    #[test]
    fn zero_b_gives_zero() {
        let (c, g, acc) = setup(6, 4);
        let zb = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ = Integrator::new(3);
        let mut cg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        incr_convolution_less(
            4,
            &mut cg,
            &g,
            &acc,
            None,
            &zb,
            &zb,
            &integ,
            1.0,
            0.1,
            Complex::ONE,
        );
        for i in 0..=4 {
            assert_eq!(cg.get_less(i, 4, 0, 0), Complex::ZERO);
        }
    }

    #[test]
    fn c1_part_constant_exact() {
        let (c, g, acc) = setup(8, 4);
        let integ = Integrator::new(3);
        let h = 0.1;
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for n in 1..=8 {
            conv_ret_less_kernel(
                &mut cg.less,
                (0, n),
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.less,
                &acc.less,
                &integ,
                Complex::new(h, 0.0),
                &mut ws,
            );
        }
        for n in 1..=8 {
            for m in 0..=n {
                // the C_1 integral runs to t_m (row-m weights)
                let want = Complex::new(m as f64 * h, 0.0);
                let got = cg.get_less(m, n, 0, 0);
                assert!((got - want).abs() < 1e-12, "({m},{n}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn c2_part_constant_exact() {
        let (c, g, acc) = setup(8, 4);
        let integ = Integrator::new(3);
        let h = 0.1;
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for n in 1..=8 {
            conv_less_adv_kernel(
                &mut cg.less,
                (0, n),
                n,
                &g.less,
                &acc.less,
                FOp::None,
                &g.ret,
                &acc.ret,
                &integ,
                Complex::new(h, 0.0),
                &mut ws,
                false,
            );
        }
        // hand-traced values of the discretized formula for the toy
        // +-1 fill (dagger partners negate)
        assert!((cg.get_less(0, 1, 0, 0) - Complex::new(-h, 0.0)).abs() < 1e-12);
        assert!((cg.get_less(1, 1, 0, 0) - Complex::new(-h, 0.0)).abs() < 1e-12);
        assert!(
            (cg.get_less(3, 5, 0, 0) - Complex::new(-5.0 * h, 0.0)).abs() < 1e-12,
            "{}",
            cg.get_less(3, 5, 0, 0)
        );
    }

    #[test]
    fn c3_part_is_pure_imaginary_structure() {
        let (c, g, _acc) = setup(4, 20);
        let integ = Integrator::new(3);
        let beta = 1.0;
        let dtau = beta / 20.0;
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        conv_lmix_rmix_kernel(
            &mut cg.less,
            (0, 2),
            2,
            &g.lmix,
            None,
            &g.lmix,
            &integ,
            Sign::Fermi,
            Complex::new(0.0, -dtau),
            &mut ws,
        );
        let got = cg.get_less(1, 2, 0, 0);
        assert!(got.re.abs() < 1e-12);
        // -i * dtau * (-sign) * sum(Omega[N]) = -i * beta for fermions
        assert!((got.im + beta).abs() < 1e-12, "got {got}");
    }
}
