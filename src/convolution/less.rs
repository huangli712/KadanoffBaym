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
//!

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, RetComp, Sign};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::f_times_b;
use crate::convolution::full::{check_conv_args, check_conv_shapes};
use crate::linalg::{mul_acc, mul_acc_c, mul_acc_c2_u, mul_acc_c4, mul_acc_c4_u, mul_acc_cd};
use crate::types::{adjoint_block_into, triangular_row_offset};

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
            let bj = f_times_b(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
            conv_ret_less_col(tgt, n, ms, j, bj, s, a, acc, integ, b.size2, &mut adjbuf2);
        } else {
            adjoint_block_into(bcc.raw_block_at(n, j), &mut adjbuf, bcc.size2, bcc.size1);
            let bj = f_times_b(ws, &f, j, &adjbuf, bcc.size2, bcc.size1);
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
    s2b: usize,
    adjbuf: &mut [Complex],
) {
    let k = integ.k;
    let s1 = a.size1;
    let s2a = a.size2;
    if s1 == 1 && s2a == 1 && s2b == 1 {
        conv_ret_less_col_d1(tgt, col, ms, j, b, s, a, acc, integ);
        return;
    }
    if s1 == 4 && s2a == 4 && s2b == 4 && acc.size1 == 4 && acc.size2 == 4 {
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
            mul_acc_c(blk, s.scale(w), a.raw_block_at(m, j), b, s1, s2a, s2b);
        } else {
            if j > k + 1 {
                continue;
            }
            let w = integ.giw.at(m as usize, j as usize);
            if j > m {
                adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, -s.scale(w), adjbuf, b, acc.size2, acc.size1, s2b);
            } else {
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, s.scale(w), a.raw_block_at(m, j), b, s1, s2a, s2b);
            }
        }
    }
    for m in m_int.max(ms.0)..=ms.1 {
        let blk = tgt.raw_block_at_mut(m, col);
        mul_acc_c(blk, s, a.raw_block_at(m, j), b, s1, s2a, s2b);
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
    let t0 = triangular_row_offset(col as usize) + ms.0 as usize;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize];
    let arow = a.raw();
    let accr = acc.raw();
    let accrow_j = triangular_row_offset(j as usize);
    // A^R(m, j) cursor: offset(m, j) = triangular_row_offset(m) + j, stride m + 1
    let mut aoff = triangular_row_offset(ms.0 as usize) + j as usize;
    // branch-tree split by m regime (ascending order, bit-identical):
    // m < j <= k reads the adjoint sigma branch, m in [j, k] the direct
    // sigma one, the (k, 2k+1] window big_s (omega[m-j] beyond k), and
    // m >= 2k+2 the constant omega[j]; the j == k+1 zero-weight terms
    // are kept so the accumulation stays bit-identical
    let m1 = ms.1.min(m_int - 1);
    let m0 = ms.0;
    let seg = |r0: i64, r1: i64| (r0.max(m0), r1.min(m1));
    let mut run = |aoff: &mut usize, r0: i64, r1: i64, adjoint: bool, wsel: i32| {
        let (r0, r1) = seg(r0, r1);
        if r0 > r1 {
            return;
        }
        for m in r0..=r1 {
            let ti = (m - m0) as usize;
            let w = match wsel {
                0 => sg[m as usize * sk + j as usize],
                1 => bs[(m - k - 1) as usize * sk + j as usize],
                2 => om[j as usize],
                3 => om[(m - j) as usize],
                _ => 0.0,
            };
            if adjoint {
                let sb = accr[accrow_j + m as usize].conj();
                if s.im == 0.0 {
                    tseg[ti] += (sb * b0).scale(-(s.re) * w);
                } else {
                    tseg[ti] += -s.scale(w) * (sb * b0);
                }
            } else if s.im == 0.0 {
                tseg[ti] += (arow[*aoff] * b0).scale(s.re * w);
            } else {
                tseg[ti] += s.scale(w) * (arow[*aoff] * b0);
            }
            *aoff += m as usize + 1;
        }
    };
    if j <= k {
        run(&mut aoff, m0, j - 1, true, 0);
        run(&mut aoff, j, k, false, 0);
        run(&mut aoff, k + 1, 2 * k + 1, false, 1);
        run(&mut aoff, 2 * k + 2, m1, false, 2);
    } else if j == k + 1 {
        // zero-weight adjoint terms (kept for bit-identity)
        run(&mut aoff, m0, k, true, 4);
        run(&mut aoff, k + 1, m1, false, 3);
    } else {
        // the whole [m0, j-1] range accumulates nothing but still
        // advances the A^R cursor
        let (r0, r1) = seg(m0, j - 1);
        if r0 <= r1 {
            aoff += triangular_row_offset((r1 + 1) as usize) - triangular_row_offset(r0 as usize);
        }
        run(&mut aoff, j, m1, false, 3);
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
            let bj = f_times_b(ws, &f, j, &adjbuf, bcc.size2, bcc.size1);
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
                bcc.size1,
                &mut adjbuf2,
            );
        } else {
            let bj = f_times_b(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
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
    s2b: usize,
    adjbuf: &mut [Complex],
) {
    let s1 = a.size1;
    let s2a = a.size2;
    if s1 == 1 && s2a == 1 && s2b == 1 {
        conv_less_adv_col_d1(tgt, col, diagcc, ms, j, b, alpha, a, acc);
        return;
    }
    if s1 == 2 && s2a == 2 && s2b == 2 && acc.size1 == 2 && acc.size2 == 2 {
        conv_less_adv_col_d::<2, 4>(tgt, col, diagcc, ms, j, b, alpha, a, acc);
        return;
    }
    if s1 == 4 && s2a == 4 && s2b == 4 && acc.size1 == 4 && acc.size2 == 4 {
        conv_less_adv_col_d::<4, 16>(tgt, col, diagcc, ms, j, b, alpha, a, acc);
        return;
    }
    for m in ms.0..=ms.1 {
        if j < m || (j == m && diagcc) {
            adjoint_block_into(acc.raw_block_at(j, m), adjbuf, acc.size2, acc.size1);
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, -alpha, adjbuf, b, acc.size2, acc.size1, s2b);
        } else {
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), b, s1, s2a, s2b);
        }
    }
}

/// Fixed-size (`D x D`) form of the square branch of
/// [`conv_less_adv_col`] (see `mul_acc_c2` / `mul_acc_c4`): the stored
/// `A^<(m, j)` blocks for the direct branch (`m <= j`) are one contiguous
/// row, the adjoint branch (`j < m`, plus the diagonal when `diagcc`)
/// walks the `Acc^<(j, m)` column with a running cursor; each target
/// block receives exactly one term, so the split is bit-identical.
#[allow(clippy::too_many_arguments)]
fn conv_less_adv_col_d<const D: usize, const N: usize>(
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
    let nel = N;
    let t0 = (triangular_row_offset(col as usize) + ms.0 as usize) * nel;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize * nel];
    let arow = a.raw();
    let accr = acc.raw();
    let jrow = triangular_row_offset(j as usize) * nel;
    let nb = -alpha;
    // direct part: m in [ms.0, j] (diagonal excluded under diagcc)
    let dend = if diagcc { j - 1 } else { j }.min(ms.1);
    if ms.0 <= dend {
        let cnt = (dend - ms.0 + 1) as usize;
        let base = jrow + ms.0 as usize * nel;
        let aseg = &arow[base..base + cnt * nel];
        for (i, blk) in aseg.chunks_exact(nel).enumerate() {
            mul_acc_cd::<D, N>(&mut tseg[i * nel..(i + 1) * nel], alpha, blk, b);
        }
    }
    // adjoint part: m in [j or j+1, ms.1]
    let astart = if diagcc { j } else { j + 1 }.max(ms.0);
    if astart <= ms.1 {
        let mut coff = (triangular_row_offset(astart as usize) + j as usize) * nel;
        for m in astart..=ms.1 {
            let ti = (m - ms.0) as usize;
            let sv = &accr[coff..coff + nel];
            // adjoint of a D x D block: transpose + conjugate
            let mut adj = [Complex::ZERO; N];
            for p in 0..D {
                for q in 0..D {
                    adj[p * D + q] = sv[q * D + p].conj();
                }
            }
            mul_acc_cd::<D, N>(&mut tseg[ti * nel..ti * nel + nel], nb, &adj, b);
            coff += (m as usize + 1) * nel;
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
    let t0 = triangular_row_offset(col as usize) + ms.0 as usize;
    let tseg = &mut tgt.raw_mut()[t0..t0 + (ms.1 - ms.0 + 1) as usize];
    let ar = a.raw();
    let acr = acc.raw();
    let ja = triangular_row_offset(j as usize);
    // acc^<(j, m): row key m varies, stride m + 1
    let mut accoff = triangular_row_offset(ms.0 as usize) + j as usize;
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
) {
    let ntau = a.ntau as i64;
    if a.size1 == 1 && a.size2 == 1 && bcc.size1 == 1 && bcc.size2 == 1 {
        conv_lmix_rmix_kernel_d1(tgt, ms, n, a, f0, bcc, integ, sign, s);
        return;
    }
    if a.size1 == 2
        && a.size2 == 2
        && bcc.size1 == 2
        && bcc.size2 == 2
        && f0.is_none_or(|f0blk| f0blk.len() == 4)
    {
        conv_lmix_rmix_kernel_d::<2, 4>(tgt, ms, n, a, f0, bcc, integ, sign, s);
        return;
    }
    if a.size1 == 4
        && a.size2 == 4
        && bcc.size1 == 4
        && bcc.size2 == 4
        && f0.is_none_or(|f0blk| f0blk.len() == 16)
    {
        conv_lmix_rmix_kernel_d::<4, 16>(tgt, ms, n, a, f0, bcc, integ, sign, s);
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
        // number of columns of `bj`
        let s2b = bcc.size1;
        for m in ms.0..=ms.1 {
            let blk = tgt.raw_block_at_mut(m, n);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), bj, s1, s2a, s2b);
        }
    }
}

/// Fixed-size (`D x D`) form of the square-f0 branch of
/// [`conv_lmix_rmix_kernel`] (see `mul_acc_c2` / `mul_acc_c4`): rhs
/// precompute + contiguous `A^rangle` row walk per target block (m
/// outer, j inner). Each target block still accumulates in ascending j
/// with the same products, so the arithmetic is identical.
#[allow(clippy::too_many_arguments)]
fn conv_lmix_rmix_kernel_d<const D: usize, const N: usize>(
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
    let nel = N;
    let nj = (ntau + 1) as usize;
    let wstride = a.ntau + 1;
    let bccr = bcc.raw();
    let arow = a.raw();
    let jbase = n as usize * wstride;
    let mut rhs = vec![Complex::ZERO; nj * nel];
    let mut alphas = vec![Complex::ZERO; nj];
    let mut fbbuf: Vec<Complex> = Vec::new();
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        let bsrc = &bccr[(jbase + (ntau - j) as usize) * nel..][..nel];
        // adjoint of a D x D block: transpose + conjugate
        let mut adj = [Complex::ZERO; N];
        for p in 0..D {
            for q in 0..D {
                adj[p * D + q] = bsrc[q * D + p].conj();
            }
        }
        let bj: &[Complex] = match f0 {
            Some(f0blk) => {
                let s1f = f0blk.len() / bcc.size2;
                fbbuf.clear();
                fbbuf.resize(s1f * bcc.size1, Complex::ZERO);
                mul_acc(&mut fbbuf, 1.0, f0blk, &adj, s1f, bcc.size2, bcc.size1);
                &fbbuf
            }
            None => &adj,
        };
        alphas[j as usize] = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        rhs[j as usize * nel..(j as usize + 1) * nel].copy_from_slice(bj);
    }
    // A^rangle(m, .) is contiguous in j: walk each target row
    for m in ms.0..=ms.1 {
        let blk = tgt.raw_block_at_mut(m, n);
        let row = &arow[m as usize * wstride * nel..(m as usize * wstride + nj) * nel];
        for j in 0..nj {
            mul_acc_cd::<D, N>(
                blk,
                alphas[j],
                &row[j * nel..][..nel],
                &rhs[j * nel..][..nel],
            );
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
    let t0 = triangular_row_offset(n as usize) + ms.0 as usize;
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
///
///
/// Increment `C^<(t', t)` (`t' <= t = tstp`) by
/// `alpha * [A * f * B]^<`, the sum of the three partial lesser
/// convolutions.
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
    check_conv_args((comp.size1, comp.size2), a, acc, f, b, bcc);
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
    check_conv_args((comp.size1, comp.size2), a, acc, f, b, bcc);
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

/// Partial increment of the left-mixing/right-mixing term,
/// `C^< += alpha * [A^⌉ * B^⌈]`.
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
    // Bcc holds the transposed blocks: its column count is the row count of B
    check_conv_shapes(
        (c.size1, c.size2),
        (a.size1, a.size2),
        f.map(|cf| (cf.size1, cf.size2)),
        (bcc.size2, bcc.size1),
    );
    incr_convolution_lmix_rmix_impl(&mut c.less, tstp, a, f, bcc, integ, beta, alpha);
}

/// Slice-target variant of [`incr_convolution_lmix_rmix`] with a
/// `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_lmix_rmix_step(
    tstp: i64,
    c: &mut GreenTimeStep,
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
    // Bcc holds the transposed blocks: its column count is the row count of B
    check_conv_shapes(
        (c.size1, c.size2),
        (a.size1, a.size2),
        f.map(|cf| (cf.size1, cf.size2)),
        (bcc.size2, bcc.size1),
    );
    incr_convolution_lmix_rmix_impl(&mut c.less, tstp, a, f, bcc, integ, beta, alpha);
}

/// Shared body of [`incr_convolution_lmix_rmix`] /
/// [`incr_convolution_lmix_rmix_step`].
#[allow(clippy::too_many_arguments)]
fn incr_convolution_lmix_rmix_impl(
    comp: &mut LessComp,
    tstp: i64,
    a: &ContourGreen,
    f: Option<&TimeFunction>,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
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
    check_conv_args((c.size1, c.size2), a, acc, f, b, bcc);
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
    alpha: Complex,
) {
    // the buffer gets (a.size1 x bcc.size1) blocks: the columns of A match
    // the rows of f_0, whose columns match the column count of Bcc
    assert!(
        match f0 {
            Some(f0blk) => f0blk.len() == a.size2 * bcc.size2,
            None => a.size2 == bcc.size2,
        },
        "incr_lmix_rmix_buf: inconsistent shapes: A {}x{}, f0 {:?}, Bcc {}x{}",
        a.size1,
        a.size2,
        f0.map(<[Complex]>::len),
        bcc.size1,
        bcc.size2
    );
    // s = alpha * (-i * delta_tau), as in the public green-target version
    let dtau = beta / a.ntau as f64;
    let s = alpha * Complex::new(0.0, -dtau);
    let ntau = a.ntau as i64;
    if a.lmix.size1 == 1 && a.lmix.size2 == 1 && bcc.lmix.size1 == 1 && bcc.lmix.size2 == 1 {
        incr_lmix_rmix_buf_d1(buf, tstp, a, f0, bcc, integ, sign, s);
        return;
    }
    if a.lmix.size1 == 2 && a.lmix.size2 == 2 && bcc.lmix.size1 == 2 && bcc.lmix.size2 == 2 {
        // d = 2 specialization: the per-j shared right-hand blocks are
        // precomputed once, then each buffer block walks its contiguous
        // A^rangle row (m outer, j inner). Every buffer still
        // accumulates in ascending j with the same products, so the
        // arithmetic is identical; only the memory order changes.
        // Rectangular f0 blocks are rare (f0 has the Green's shape) and
        // fall through to the generic path below.
        let square = match f0 {
            None => true,
            Some(f0blk) => f0blk.len() == 4,
        };
        if square {
            let ntau = a.ntau as i64;
            let nel = 4usize;
            let nj = (ntau + 1) as usize;
            let bccr = bcc.lmix.raw();
            let arow = a.lmix.raw();
            let wstride = a.ntau + 1;
            let jbase = tstp as usize * wstride;
            // rhs[j] = (B^ceil block at j) * alpha_j
            let mut rhs = vec![Complex::ZERO; nj * nel];
            let mut fbbuf: Vec<Complex> = Vec::new();
            for j in 0..=ntau {
                let w = integ.giw.at(ntau as usize, j as usize);
                // B^ceil(tau_j, t_n) = B^rangle(t_n, tau_{N-j})^dagger
                let bsrc = &bccr[(jbase + (ntau - j) as usize) * nel..][..nel];
                let mut adj = [Complex::ZERO; 4];
                for (o, &z) in adj.iter_mut().zip(bsrc) {
                    *o = z.conj();
                }
                adj.swap(1, 2);
                let bj: &[Complex] = match f0 {
                    Some(f0blk) => {
                        let s1f = f0blk.len() / bcc.lmix.size2;
                        fbbuf.clear();
                        fbbuf.resize(s1f * bcc.lmix.size1, Complex::ZERO);
                        mul_acc(
                            &mut fbbuf,
                            1.0,
                            f0blk,
                            &adj,
                            s1f,
                            bcc.lmix.size2,
                            bcc.lmix.size1,
                        );
                        &fbbuf
                    }
                    None => &adj,
                };
                let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
                let dst = &mut rhs[j as usize * nel..(j as usize + 1) * nel];
                for (o, v) in dst.iter_mut().zip(bj) {
                    *o = *v * alpha;
                }
            }
            // A^rangle(m, .) is contiguous in j: walk each buffer's row
            for (m, bufm) in buf.iter_mut().enumerate() {
                let row = &arow[m * wstride * nel..(m * wstride + nj) * nel];
                for j in 0..nj {
                    mul_acc_c2_u(bufm, &row[j * nel..][..nel], &rhs[j * nel..][..nel]);
                }
            }
            return;
        }
    }
    if a.lmix.size1 == 4 && a.lmix.size2 == 4 && bcc.lmix.size1 == 4 && bcc.lmix.size2 == 4 {
        // d = 4 specialization: rhs precompute + contiguous A^rangle
        // row walk per buffer (m outer, j inner), as in the d = 2
        // branch. The products and the per-buffer accumulation order
        // are those of the column walk, so the arithmetic is identical.
        // Rectangular f0 blocks fall through to the generic path.
        let square = match f0 {
            None => true,
            Some(f0blk) => f0blk.len() == 16,
        };
        if square {
            let ntau = a.ntau as i64;
            let nel = 16usize;
            let nj = (ntau + 1) as usize;
            let bccr = bcc.lmix.raw();
            let arow = a.lmix.raw();
            let wstride = a.ntau + 1;
            let jbase = tstp as usize * wstride;
            // rhs[j] = B^ceil block at j, alphas[j] = its row weight
            let mut rhs = vec![Complex::ZERO; nj * nel];
            let mut alphas = vec![Complex::ZERO; nj];
            let mut fbbuf: Vec<Complex> = Vec::new();
            for j in 0..=ntau {
                let w = integ.giw.at(ntau as usize, j as usize);
                let bsrc = &bccr[(jbase + (ntau - j) as usize) * nel..][..nel];
                let mut adj = [Complex::ZERO; 16];
                for p in 0..4 {
                    for q in 0..4 {
                        adj[p * 4 + q] = bsrc[q * 4 + p].conj();
                    }
                }
                let bj: &[Complex] = match f0 {
                    Some(f0blk) => {
                        let s1f = f0blk.len() / bcc.lmix.size2;
                        fbbuf.clear();
                        fbbuf.resize(s1f * bcc.lmix.size1, Complex::ZERO);
                        mul_acc(
                            &mut fbbuf,
                            1.0,
                            f0blk,
                            &adj,
                            s1f,
                            bcc.lmix.size2,
                            bcc.lmix.size1,
                        );
                        &fbbuf
                    }
                    None => &adj,
                };
                alphas[j as usize] = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
                rhs[j as usize * nel..(j as usize + 1) * nel].copy_from_slice(bj);
            }
            // A^rangle(m, .) is contiguous in j: walk each buffer's row
            for (m, bufm) in buf.iter_mut().enumerate() {
                let row = &arow[m * wstride * nel..(m * wstride + nj) * nel];
                for j in 0..nj {
                    mul_acc_c4(
                        bufm,
                        alphas[j],
                        &row[j * nel..][..nel],
                        &rhs[j * nel..][..nel],
                    );
                }
            }
            return;
        }
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
        if s1 == 4 && s2a == 4 && bj.len() == 16 {
            // fold the row weight into the shared right-hand block
            // once, then accumulate the whole target walk unweighted
            let mut bj_scaled = [Complex::ZERO; 16];
            for (o, v) in bj_scaled.iter_mut().zip(bj) {
                *o = *v * alpha;
            }
            for (m, bufm) in buf.iter_mut().enumerate() {
                mul_acc_c4_u(bufm, a.lmix.raw_block_at(m as i64, j), &bj_scaled);
            }
        } else {
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
    let nj = (ntau + 1) as usize;
    let ar = a.lmix.raw();
    let bcr = bcc.lmix.raw();
    let bcrow = tstp as usize * wstride;
    // per-j right-hand value and weight, the same expressions the
    // column walk used per (j, m) pair
    let mut rhs = vec![Complex::ZERO; nj];
    let mut alphas = vec![Complex::ZERO; nj];
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        let sb = bcr[bcrow + (ntau - j) as usize].conj();
        let bj = match f0 {
            Some(f0blk) => f0blk[0] * sb,
            None => sb,
        };
        rhs[j as usize] = bj;
        alphas[j as usize] = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
    }
    // A^⌉(t_m, tau_j) is contiguous in j within the tau row of m:
    // each buffer accumulates in ascending j, as in the column walk
    for (m, bufm) in buf.iter_mut().enumerate() {
        let row = &ar[m * wstride..m * wstride + nj];
        for j in 0..nj {
            bufm[0] += alphas[j] * (row[j] * rhs[j]);
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
    // the shape contract of the lesser convolution, `f` given as blocks
    assert!(
        match f {
            FOp::None => a.size2 == b.size1,
            FOp::Cf(cf) => a.size2 == cf.size1 && cf.size2 == b.size1,
            FOp::T(v) => v.iter().all(|blk| blk.len() == a.size2 * b.size1),
        } && acc.size1 == a.size2
            && acc.size2 == a.size1
            && bcc.size1 == b.size2
            && bcc.size2 == b.size1,
        "incr_less_adv_buf: inconsistent shapes: A {}x{}, Acc {}x{}, B {}x{}, Bcc {}x{}",
        a.size1,
        a.size2,
        acc.size1,
        acc.size2,
        b.size1,
        b.size2,
        bcc.size1,
        bcc.size2
    );
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
            let bj = f_times_b(ws, f, j, &adjbuf, bcc.ret.size2, bcc.ret.size1);
            let s2b = bcc.ret.size1;
            conv_less_adv_col_buf(
                buf,
                j,
                bj,
                s * Complex::new(w, 0.0),
                a,
                acc,
                diagcc,
                s2b,
                &mut adjbuf2,
            );
        } else {
            let braw = b.ret.raw_block_at(j, tstp);
            let bj = f_times_b(ws, f, j, braw, b.ret.size1, b.ret.size2);
            let s2b = b.ret.size2;
            conv_less_adv_col_buf(
                buf,
                j,
                bj,
                -s * Complex::new(w, 0.0),
                a,
                acc,
                diagcc,
                s2b,
                &mut adjbuf,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_less_adv_col_buf(
    buf: &mut [Vec<Complex>],
    j: i64,
    b: &[Complex],
    alpha: Complex,
    a: &ContourGreen,
    acc: &ContourGreen,
    diagcc: bool,
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
        // d = 2 specialization: contiguous A^<(m, j) row for the
        // direct part, cursor-walked Acc^<(j, m) column with the
        // inlined 2x2 adjoint for the other part (bit-identical split)
        let nel = 4usize;
        let arow = a.less.raw();
        let accr = acc.less.raw();
        let jrow = triangular_row_offset(j as usize) * nel;
        let nb = -alpha;
        let dend = if diagcc { j - 1 } else { j };
        // direct part: A^<(m, j) column is contiguous in m -> one
        // pre-sliced segment zipped with the buffer
        let dcnt = ((dend + 1).min(buf.len() as i64)).max(0) as usize;
        let src = &arow[jrow..jrow + dcnt * nel];
        // the right-hand block is shared by the whole column walk:
        // fold the column weight in once and accumulate unweighted
        let mut bs = [Complex::ZERO; 4];
        let mut bns = [Complex::ZERO; 4];
        for (o, &v) in bs.iter_mut().zip(b) {
            *o = v * alpha;
        }
        for (o, &v) in bns.iter_mut().zip(b) {
            *o = v * nb;
        }
        for (bufm, sblk) in buf[..dcnt].iter_mut().zip(src.chunks_exact(nel)) {
            mul_acc_c2_u(bufm, sblk, &bs);
        }
        // adjoint part: cursor-walked Acc^<(j, m) column
        let mut coff = (triangular_row_offset(dcnt) + j as usize) * 4;
        for (mi, bufm) in buf.iter_mut().enumerate().skip(dcnt) {
            let sv = &accr[coff..coff + 4];
            let mut adj = [Complex::ZERO; 4];
            for (o, &z) in adj.iter_mut().zip(sv) {
                *o = z.conj();
            }
            adj.swap(1, 2);
            mul_acc_c2_u(bufm, &adj, &bns);
            coff += (mi + 1) * 4;
        }
        return;
    }
    if s1 == 4 && s2a == 4 && s2b == 4 && acc.less.size1 == 4 && acc.less.size2 == 4 {
        // d = 4 specialization: same split as the d = 2 buf branch
        let nel = 16usize;
        let arow = a.less.raw();
        let accr = acc.less.raw();
        let jrow = triangular_row_offset(j as usize) * nel;
        let nb = -alpha;
        let dend = if diagcc { j - 1 } else { j };
        let dcnt = ((dend + 1).min(buf.len() as i64)).max(0) as usize;
        let src = &arow[jrow..jrow + dcnt * nel];
        let mut bs = [Complex::ZERO; 16];
        let mut bns = [Complex::ZERO; 16];
        for (o, &v) in bs.iter_mut().zip(b) {
            *o = v * alpha;
        }
        for (o, &v) in bns.iter_mut().zip(b) {
            *o = v * nb;
        }
        for (bufm, sblk) in buf[..dcnt].iter_mut().zip(src.chunks_exact(nel)) {
            mul_acc_c4_u(bufm, sblk, &bs);
        }
        let mut coff = (triangular_row_offset(dcnt) + j as usize) * nel;
        for (mi, bufm) in buf.iter_mut().enumerate().skip(dcnt) {
            let sv = &accr[coff..coff + nel];
            let mut adj = [Complex::ZERO; 16];
            for i2 in 0..4 {
                for j2 in 0..4 {
                    adj[i2 * 4 + j2] = sv[j2 * 4 + i2].conj();
                }
            }
            mul_acc_c4_u(bufm, &adj, &bns);
            coff += (mi + 1) * nel;
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
    let ja = triangular_row_offset(j as usize);
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
        );
        let got = cg.get_less(1, 2, 0, 0);
        assert!(got.re.abs() < 1e-12);
        // -i * dtau * (-sign) * sum(Omega[N]) = -i * beta for fermions
        assert!((got.im + beta).abs() < 1e-12, "got {got}");
    }
}
