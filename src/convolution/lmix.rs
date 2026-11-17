// Project : KadanoffBaym
// File    : src/convolution/lmix.rs
// Purpose : left-mixing convolution kernels

//!
//! ### *Convolution* : ``G^{⌉}`` *Component*
//!
//! *Remarks* : *Left-Mixing Convolution*
//!
//! The evaluation of ``C^{\rceil}(t,\tau)`` at given time slice ``n`` is
//! implemented as follows:
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}(nh,mh_{\tau}) =
//!     C^{\rceil}_1[A,f,B](n,m) +
//!     C^{\rceil}_2[A,f,B](n,m) +
//!     C^{\rceil}_3[A,f,B](n,m),
//! \end{equation}
//! ```
//!
//! where ``m = 0,\ \cdots,\ N_{\tau}``. In the above equation, ``h`` means
//! the interval in the real time axis (``\equiv \delta t``) and ``h_{\tau}``
//! means the interval in the imaginary time axis (``\equiv \delta \tau``).
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_1[A,f,B](n,m) = \int^{nh}_{0} d\bar{t}~
//!     A^{R}(nh,\bar{t}) f(\bar{t}) B^{\rceil}(\bar{t},mh_{\tau}).
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_2[A,f,B](n,m) = \int^{mh_{\tau}}_{0} d\tau~
//!     A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau'-mh_{\tau}).
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_3[A,f,B](n,m) = \int^{\beta}_{mh_{\tau}} d\tau~
//!     A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau' - mh_{\tau}).
//! \end{equation}
//! ```
//!
//! Actually, we implement the following equations:
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_1[A,f,B](n,m) = h \sum^{n}_{j = 0}
//!     w^{(k)}_{n,j} A^{R}_{n,j} f_j B^{\rceil}_{j,m},
//!     \quad n > k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
//!     w^{(k)}_{n,j} \tilde{A}^{R}_{n,j} f_j B^{\rceil}_{j,m},
//!     \quad n \le k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
//!     R^{(k)}_{m;j,l} A^{\rceil}_l f_{-1} \xi B^{M}_{N_{\tau}-j},
//!     \quad m \le k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{m}_{l = 0}
//!     w^{(k)}_{m,l} A^{\rceil}_{m-l} f_{-1} \xi B^{M}_{N_{\tau}-l},
//!     \quad m > k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
//!     R^{(k)}_{N_{\tau}-m;j,l} A^{\rceil}_{N_{\tau}-l} f_{-1} B^{M}_{j},
//!     \quad m \ge N_{\tau} - k.
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{N_{\tau}-m}_{l = 0}
//!     w^{(k)}_{N_{\tau}-m,l} A^{\rceil}_{m+l} f_{-1} B^{M}_{l},
//!     \quad m < N_{\tau} - k.
//! \end{equation}
//! ```

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, Sign, TimeFunction};
use crate::types::{LmixComp, MatComp, RetComp};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::{af, mul_acc, mul_acc_c};
use crate::types::adjoint_block_into;

/// The `C_1` part: accumulate `s * [A^R * f * B^⌉](t_n, tau_m)` into
/// the lmix row `n`.
#[allow(clippy::too_many_arguments)]
pub fn conv_ret_lmix_kernel(
    tgt: &mut LmixComp,
    row: i64,
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &LmixComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
    ms: (i64, i64),
) {
    // n1 = max(n_j - 1, k) + 1 gives the 0-based
    // inclusive j bound max(n_r, k)
    let n1r = n.max(integ.k);
    let mut adjbuf = vec![Complex::ZERO; acc.size1 * acc.size2];
    for j in 0..=n1r {
        let w = integ.giw.at(n as usize, j as usize);
        if n < j {
            adjoint_block_into(acc.raw_block_at(j, n), &mut adjbuf, acc.size2, acc.size1);
            let aj = af(ws, &adjbuf, &f, j, a.size1, a.size2);
            conv_lmix_row(tgt, row, aj, -s.scale(w), b, j, ms);
        } else {
            let aj = af(ws, a.raw_block_at(n, j), &f, j, a.size1, a.size2);
            conv_lmix_row(tgt, row, aj, s.scale(w), b, j, ms);
        }
    }
}

/// Accumulate `alpha * a * B^⌉(t_j, tau_m)` into `tgt[row, m]` for
/// `m` in `ms`.
#[allow(clippy::too_many_arguments)]
fn conv_lmix_row(
    tgt: &mut LmixComp,
    row: i64,
    a: &[Complex],
    alpha: Complex,
    b: &LmixComp,
    j: i64,
    ms: (i64, i64),
) {
    let s1 = b.size1;
    let s2a = a.len() / s1;
    let s2b = b.size2;
    // scalar fast path for d = 1
    if s1 == 1 && s2a == 1 && s2b == 1 {
        for m in ms.0..=ms.1 {
            let bb = b.raw_block_at(j, m);
            tgt.raw_block_at_mut(row, m)[0] += alpha * (a[0] * bb[0]);
        }
        return;
    }
    for m in ms.0..=ms.1 {
        let blk = tgt.raw_block_at_mut(row, m);
        mul_acc_c(blk, alpha, a, b.raw_block_at(j, m), s1, s2a, s2b);
    }
}

/// The `C_2 + C_3` parts: accumulate `s * [A^⌉ * f_0 * B^M](t_n, tau_m)`
/// into the lmix row `n`. `bt` holds the `(a.ntau + 1)` contiguous
/// `f_0 B^M(tau_m)` blocks of equal length.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_lmix_mat_loop(
    tgt_row: &mut [Complex],
    n: i64,
    a: &LmixComp,
    bt: &[Complex],
    integ: &Integrator,
    sign: Sign,
    s: Complex,
    ms: (i64, i64),
) {
    let k = integ.k;
    let ntau = a.ntau as i64; // Rust max tau index (N)
    let (s1, s2a) = (a.size1, a.size2);
    // bt holds the (ntau + 1) contiguous (s2a x s2b) blocks
    let nelb = bt.len() / (a.ntau + 1);
    let s2b = nelb.checked_div(s2a).unwrap_or(0);
    if s1 == 1 && s2a == 1 && s2b == 1 {
        conv_lmix_mat_loop_d1(tgt_row, n, a, bt, integ, sign, s, ms);
        return;
    }

    for m in ms.0..=ms.1 {
        let blk = &mut tgt_row[(m as usize) * s1 * s2b..(m as usize + 1) * s1 * s2b];

        // contributions from 0 to tau
        if m == 0 {
            // PASS
        } else if m < k {
            // strange boundary correction
            for j in 0..=k {
                let mut indb = ntau;
                for l in 0..=k {
                    let w = integ.bcw.at(m as usize, l as usize, j as usize);
                    mul_acc_c(
                        blk,
                        sign.as_i64() as f64 * s * Complex::new(w, 0.0),
                        a.raw_block_at(n, j),
                        &bt[indb as usize * nelb..(indb as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                    indb -= 1;
                }
            }
        } else {
            // usual Gregory integration
            if m >= 2 * k + 2 {
                // interior l: Omega[m][l] is exactly 1 for k < l < m-k,
                // the weight multiply is skipped there (ascending
                // order kept, bit-identical)
                for l in 0..=k {
                    let w = integ.giw.at(m as usize, l as usize);
                    mul_acc_c(
                        blk,
                        sign.as_i64() as f64 * s * Complex::new(w, 0.0),
                        a.raw_block_at(n, m - l),
                        &bt[(ntau - l) as usize * nelb..((ntau - l) as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                }
                for l in (k + 1)..=(m - k - 1) {
                    mul_acc_c(
                        blk,
                        sign.as_i64() as f64 * s,
                        a.raw_block_at(n, m - l),
                        &bt[(ntau - l) as usize * nelb..((ntau - l) as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                }
                for l in (m - k)..=m {
                    let w = integ.giw.at(m as usize, l as usize);
                    mul_acc_c(
                        blk,
                        sign.as_i64() as f64 * s * Complex::new(w, 0.0),
                        a.raw_block_at(n, m - l),
                        &bt[(ntau - l) as usize * nelb..((ntau - l) as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                }
            } else {
                for l in 0..=m {
                    let w = integ.giw.at(m as usize, l as usize);
                    mul_acc_c(
                        blk,
                        sign.as_i64() as f64 * s * Complex::new(w, 0.0),
                        a.raw_block_at(n, m - l),
                        &bt[(ntau - l) as usize * nelb..((ntau - l) as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                }
            }
        }

        // contributions from tau to beta
        if m == ntau {
            // PASS
        } else if m > ntau - k {
            // strange boundary correction
            let mut inda = ntau;
            for l in 0..=k {
                for j in 0..=k {
                    let w = integ.bcw.at((ntau - m) as usize, l as usize, j as usize);
                    mul_acc_c(
                        blk,
                        s * Complex::new(w, 0.0),
                        a.raw_block_at(n, inda),
                        &bt[j as usize * nelb..(j as usize + 1) * nelb],
                        s1,
                        s2a,
                        s2b,
                    );
                }
                inda -= 1;
            }
        } else if m > ntau - 2 * k - 1 {
            // usual Gregory integration (short)
            for l in 0..=(ntau - m) {
                let w = integ.giw.at((ntau - m) as usize, l as usize);
                mul_acc_c(
                    blk,
                    s * Complex::new(w, 0.0),
                    a.raw_block_at(n, m + l),
                    &bt[l as usize * nelb..(l as usize + 1) * nelb],
                    s1,
                    s2a,
                    s2b,
                );
            }
        } else {
            // usual Gregory integration (long); Omega[N-m][N-l] is
            // exactly 1 for m+k < l < N-k, the weight multiply is
            // skipped there (ascending order kept, bit-identical)
            for l in m..=(m + k) {
                let w = integ.giw.at((ntau - m) as usize, (ntau - l) as usize);
                mul_acc_c(
                    blk,
                    s * Complex::new(w, 0.0),
                    a.raw_block_at(n, l),
                    &bt[(l - m) as usize * nelb..((l - m) as usize + 1) * nelb],
                    s1,
                    s2a,
                    s2b,
                );
            }
            for l in (m + k + 1)..=(ntau - k - 1) {
                mul_acc_c(
                    blk,
                    s,
                    a.raw_block_at(n, l),
                    &bt[(l - m) as usize * nelb..((l - m) as usize + 1) * nelb],
                    s1,
                    s2a,
                    s2b,
                );
            }
            for l in (ntau - k).max(m + k + 1)..=ntau {
                let w = integ.giw.at((ntau - m) as usize, (ntau - l) as usize);
                mul_acc_c(
                    blk,
                    s * Complex::new(w, 0.0),
                    a.raw_block_at(n, l),
                    &bt[(l - m) as usize * nelb..((l - m) as usize + 1) * nelb],
                    s1,
                    s2a,
                    s2b,
                );
            }
        }
    }
}

/// Scalar (`d = 1`) variant of [`conv_lmix_mat_loop`].
#[allow(clippy::too_many_arguments)]
fn conv_lmix_mat_loop_d1(
    tgt_row: &mut [Complex],
    n: i64,
    a: &LmixComp,
    bt: &[Complex],
    integ: &Integrator,
    sign: Sign,
    s: Complex,
    ms: (i64, i64),
) {
    let k = integ.k;
    let ntau = a.ntau as i64;
    // scalar blocks: one complex number per tau point
    let nelb = 1;

    let pref = sign.as_i64() as f64 * s;
    for m in ms.0..=ms.1 {
        let blk = &mut tgt_row[m as usize];

        // contributions from 0 to tau
        if m == 0 {
            // PASS
        } else if m < k {
            // strange boundary correction
            for j in 0..=k {
                let mut indb = ntau;
                for l in 0..=k {
                    let w = integ.bcw.at(m as usize, l as usize, j as usize);
                    let ab = a.raw_block_at(n, j)[0];
                    *blk += pref * Complex::new(w, 0.0) * (ab * bt[indb as usize * nelb]);
                    indb -= 1;
                }
            }
        } else if m >= 2 * k + 2 {
            // usual Gregory integration, interior weight 1
            for l in 0..=k {
                let w = integ.giw.at(m as usize, l as usize);
                let ab = a.raw_block_at(n, m - l)[0];
                *blk += pref * Complex::new(w, 0.0) * (ab * bt[(ntau - l) as usize * nelb]);
            }
            for l in (k + 1)..=(m - k - 1) {
                let ab = a.raw_block_at(n, m - l)[0];
                *blk += pref * (ab * bt[(ntau - l) as usize * nelb]);
            }
            for l in (m - k)..=m {
                let w = integ.giw.at(m as usize, l as usize);
                let ab = a.raw_block_at(n, m - l)[0];
                *blk += pref * Complex::new(w, 0.0) * (ab * bt[(ntau - l) as usize * nelb]);
            }
        } else {
            for l in 0..=m {
                let w = integ.giw.at(m as usize, l as usize);
                let ab = a.raw_block_at(n, m - l)[0];
                *blk += pref * Complex::new(w, 0.0) * (ab * bt[(ntau - l) as usize * nelb]);
            }
        }

        // contributions from tau to beta
        if m == ntau {
            // PASS
        } else if m > ntau - k {
            // strange boundary correction
            let mut inda = ntau;
            for l in 0..=k {
                for j in 0..=k {
                    let w = integ.bcw.at((ntau - m) as usize, l as usize, j as usize);
                    let ab = a.raw_block_at(n, inda)[0];
                    *blk += s * Complex::new(w, 0.0) * (ab * bt[j as usize * nelb]);
                }
                inda -= 1;
            }
        } else if m > ntau - 2 * k - 1 {
            // usual Gregory integration (short)
            for l in 0..=(ntau - m) {
                let w = integ.giw.at((ntau - m) as usize, l as usize);
                let ab = a.raw_block_at(n, m + l)[0];
                *blk += s * Complex::new(w, 0.0) * (ab * bt[l as usize * nelb]);
            }
        } else {
            // usual Gregory integration (long), interior weight 1
            for l in m..=(m + k) {
                let w = integ.giw.at((ntau - m) as usize, (ntau - l) as usize);
                let ab = a.raw_block_at(n, l)[0];
                *blk += s * Complex::new(w, 0.0) * (ab * bt[(l - m) as usize * nelb]);
            }
            for l in (m + k + 1)..=(ntau - k - 1) {
                let ab = a.raw_block_at(n, l)[0];
                *blk += s * (ab * bt[(l - m) as usize * nelb]);
            }
            for l in (ntau - k).max(m + k + 1)..=ntau {
                let w = integ.giw.at((ntau - m) as usize, (ntau - l) as usize);
                let ab = a.raw_block_at(n, l)[0];
                *blk += s * Complex::new(w, 0.0) * (ab * bt[(l - m) as usize * nelb]);
            }
        }
    }
}

/// Evaluate the `C_2 + C_3` parts into the lmix row `n` of `c`.
#[allow(clippy::too_many_arguments)]
pub fn conv_lmix_mat(
    n: i64,
    c: &mut LmixComp,
    a: &LmixComp,
    b: &MatComp,
    integ: &Integrator,
    beta: f64,
    sign: Sign,
) {
    assert!(beta > 0.0);
    let dtau = beta / a.ntau as f64;
    let bt: &[Complex] = b.raw();
    let rowlen = (a.ntau + 1) * a.size1 * b.size2;
    let off = n as usize * rowlen;
    let row = &mut c.raw_mut()[off..off + rowlen];
    for m in 0..=a.ntau as i64 {
        let blk = &mut row[(m as usize) * a.size1 * b.size2..(m as usize + 1) * a.size1 * b.size2];
        blk.fill(Complex::ZERO);
    }
    conv_lmix_mat_loop(
        row,
        n,
        a,
        bt,
        integ,
        sign,
        Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

/// `conv_lmix_mat` with a constant matrix `f_0`.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn conv_lmix_mat_f0(
    n: i64,
    c: &mut LmixComp,
    a: &LmixComp,
    f0: &DenseMat,
    b: &MatComp,
    integ: &Integrator,
    beta: f64,
    sign: Sign,
) {
    assert!(beta > 0.0);
    let dtau = beta / a.ntau as f64;
    let nelb = f0.nrows() * b.size2;
    let mut ws_mat: Vec<Complex> = vec![Complex::ZERO; (a.ntau + 1) * nelb];
    for m in 0..=(a.ntau as i64) {
        mul_acc(
            &mut ws_mat[m as usize * nelb..(m as usize + 1) * nelb],
            1.0,
            f0.raw(),
            b.block(m as usize),
            f0.nrows(),
            f0.ncols(),
            b.size2,
        );
    }
    let rowlen = (a.ntau + 1) * a.size1 * f0.ncols();
    let off = n as usize * rowlen;
    let row = &mut c.raw_mut()[off..off + rowlen];
    for m in 0..=a.ntau as i64 {
        let blk =
            &mut row[(m as usize) * a.size1 * f0.ncols()..(m as usize + 1) * a.size1 * f0.ncols()];
        blk.fill(Complex::ZERO);
    }
    conv_lmix_mat_loop(
        row,
        n,
        a,
        &ws_mat,
        integ,
        sign,
        Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

/// Increment the lmix row `tstp` of `C` by `alpha * [A * f * B]^⌉`: the
/// real-time part uses `f(t'')`, the imaginary-time part uses `f(0^-)`.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_lmix(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    _bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let dtau = beta / a.ntau as f64;
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    // C_1: real-time part
    conv_ret_lmix_kernel(
        &mut c.lmix,
        tstp,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.lmix,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        (0, a.ntau as i64),
    );
    // C_2 + C_3: imaginary-time part with f_0 = f(0^-); the blocks
    // f_0 B^M(tau_m) are flat and contiguous
    let mut btbuf: Vec<Complex>;
    let bt: &[Complex] = match f {
        Some(cf) => {
            let f0 = cf.block(MATSUBARA);
            let kdim = f0.len() / a.size2;
            let nelb = kdim * b.size2;
            btbuf = vec![Complex::ZERO; (a.ntau + 1) * nelb];
            for m in 0..=(a.ntau as i64) {
                mul_acc(
                    &mut btbuf[m as usize * nelb..(m as usize + 1) * nelb],
                    1.0,
                    f0,
                    b.mat.block(m as usize),
                    a.size2,
                    kdim,
                    b.size2,
                );
            }
            &btbuf
        }
        None => b.mat.raw(),
    };
    let rowlen = (a.ntau + 1) * a.size1 * b.size2;
    let off = tstp as usize * rowlen;
    let row = &mut c.lmix.raw_mut()[off..off + rowlen];
    conv_lmix_mat_loop(
        row,
        tstp,
        &a.lmix,
        bt,
        integ,
        a.sign,
        alpha * Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

/// [`incr_convolution_lmix`] with a single-row `LmixComp` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_lmix_step(
    tstp: i64,
    c: &mut LmixComp,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    _bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let dtau = beta / a.ntau as f64;
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    // the timestep lmix is a single-row component: row index 0
    conv_ret_lmix_kernel(
        c,
        0,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.lmix,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        (0, a.ntau as i64),
    );
    // C_2 + C_3: imaginary-time part with f_0 = f(0^-); the blocks
    // f_0 B^M(tau_m) are flat and contiguous
    let mut btbuf: Vec<Complex>;
    let bt: &[Complex] = match f {
        Some(cf) => {
            let f0 = cf.block(MATSUBARA);
            let kdim = f0.len() / a.size2;
            let nelb = kdim * b.size2;
            btbuf = vec![Complex::ZERO; (a.ntau + 1) * nelb];
            for m in 0..=(a.ntau as i64) {
                mul_acc(
                    &mut btbuf[m as usize * nelb..(m as usize + 1) * nelb],
                    1.0,
                    f0,
                    b.mat.block(m as usize),
                    a.size2,
                    kdim,
                    b.size2,
                );
            }
            &btbuf
        }
        None => b.mat.raw(),
    };
    let row = c.raw_mut();
    conv_lmix_mat_loop(
        row,
        tstp,
        &a.lmix,
        bt,
        integ,
        a.sign,
        alpha * Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::Contour;

    #[test]
    fn c1_part_constant_exact() {
        // A^R = 1, B^⌉ = 1: C_1(t_n, tau) = Int_0^{t_n} 1 = n h
        let nt = 8;
        let ntau = 6;
        let c0 = Contour::new(nt, ntau, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        for i in 0..=nt {
            for m in 0..=ntau {
                g.set_lmix(i, m, 0, 0, Complex::ONE);
            }
        }
        // dagger partner: ret negated
        let mut acc = g.clone();
        for i in 0..=nt {
            for j in 0..=i {
                let v = g.get_ret(i, j, 0, 0);
                acc.set_ret(i, j, 0, 0, -v);
            }
        }
        let integ = Integrator::new(3);
        let h = 0.1;
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for n in 1..=nt {
            conv_ret_lmix_kernel(
                &mut cgreen.lmix,
                n,
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.lmix,
                &integ,
                Complex::new(h, 0.0),
                &mut ws,
                (0, ntau),
            );
        }
        for n in 1..=nt {
            for m in 0..=ntau {
                let want = Complex::new(n as f64 * h, 0.0);
                let got = cgreen.get_lmix(n, m, 0, 0);
                assert!((got - want).abs() < 1e-12, "({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn c2_c3_part_constant_exact() {
        // A^⌉ = 1, B^M = 1: C_23(tau) = xi * tau + (beta - tau)
        let nt = 4;
        let ntau = 20usize;
        let beta = 2.0;
        let c0 = Contour::new(nt, ntau as i64, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt {
            for m in 0..=ntau as i64 {
                g.set_lmix(i, m, 0, 0, Complex::ONE);
            }
        }
        for m in 0..=ntau as i64 {
            g.set_mat(m, 0, 0, Complex::ONE);
        }
        let integ = Integrator::new(3);
        let dtau = beta / ntau as f64;
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for (sign, xi) in [(Sign::Fermi, -1.0), (Sign::Bose, 1.0)] {
            conv_lmix_mat(2, &mut cgreen.lmix, &g.lmix, &g.mat, &integ, beta, sign);
            for m in 0..=ntau as i64 {
                let tau = m as f64 * dtau;
                let want = xi * tau + (beta - tau);
                let got = cgreen.get_lmix(2, m, 0, 0);
                assert!(
                    (got - Complex::new(want, 0.0)).abs() < 1e-12,
                    "{sign:?} m {m}: {got} vs {want}"
                );
            }
        }
    }
}
