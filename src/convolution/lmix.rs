// Project : KadanoffBaym
// File    : src/convolution/lmix.rs
// Purpose : left-mixing convolution kernels (blueprint: convolution/lmix.jl)
//!
//! ### *Convolution* : ``G^{⌉}`` *Component*
//!
//! *Remarks* :
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
//!
//! *References* :
//!
//! Please see [`NESSi`] Sections `9` and `11` for more details.

use crate::convolution::{adjoint_block, af, mul_acc_c, ConvWorkspace, FOp};
use crate::complex::Complex;
use crate::linalg::mat::DenseMat;
use crate::types::sign::Sign;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::function::TimeFunction;
use crate::types::lmix::LmixComp;
use crate::types::mat::MatComp;
use crate::types::ret::RetComp;
use crate::weights::Integrator;

/// The `C_1` part: accumulate `s * [A^R * f * B^⌉](t_n, tau_m)` into
/// the lmix row `n` (blueprint `_conv_ret_lmix_kernel!`, NESSi
/// Eqs. (111)-(112)).
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
    // Julia n1 = max(n_j - 1, k) + 1 -> 0-based inclusive j bound
    // max(n_r, k)
    let n1r = n.max(integ.k);
    for j in 0..=n1r {
        let w = integ.giw.at(n as usize, j as usize);
        if n < j {
            let aadj = adjoint_block(acc.raw_block_at(j, n), acc.size1, acc.size2);
            let aj = af(ws, &aadj, &f, j, a.size1, a.size2);
            conv_lmix_row(tgt, row, aj, -s.scale(w), b, j, ms);
        } else {
            let aj = af(ws, a.raw_block_at(n, j), &f, j, a.size1, a.size2);
            conv_lmix_row(tgt, row, aj, s.scale(w), b, j, ms);
        }
    }
}

/// Accumulate `alpha * a * B^⌉(t_j, tau_m)` into `tgt[row, m]` for
/// `m = lo..=hi` (blueprint `_conv_lmix_row!`).
#[allow(clippy::too_many_arguments)]
fn conv_lmix_row(tgt: &mut LmixComp, row: i64, a: &[Complex], alpha: Complex, b: &LmixComp, j: i64, ms: (i64, i64)) {
    let s1 = b.size1;
    let s2a = a.len() / s1;
    let s2b = b.size2;
    for m in ms.0..=ms.1 {
        let blk = tgt.raw_block_at_mut(row, m);
        mul_acc_c(blk, alpha, a, b.raw_block_at(j, m), s1, s2a, s2b);
    }
}

/// The `C_2 + C_3` parts: accumulate `s * [A^⌉ * f_0 * B^M](t_n, tau_m)`
/// into the lmix row `n` (blueprint `_conv_lmix_mat_loop!`, NESSi
/// Eqs. (113)-(116)). `bt[m]` is the block `f_0 B^M(tau_m)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_lmix_mat_loop(
    tgt_row: &mut [Complex],
    n: i64,
    a: &LmixComp,
    bt: &[Vec<Complex>],
    integ: &Integrator,
    sign: Sign,
    s: Complex,
    ms: (i64, i64),
) {
    let k = integ.k;
    let ntau = a.ntau as i64; // Rust max tau index (N)
    let (s1, s2a) = (a.size1, a.size2);
    let s2b = bt.first().map(|v| v.len() / s2a).unwrap_or(0);

    for m in ms.0..=ms.1 {
        let blk = &mut tgt_row[(m as usize) * s1 * s2b..(m as usize + 1) * s1 * s2b];

        // contributions from 0 to tau (NESSi Eqs. (113)-(114))
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
                        &bt[indb as usize],
                        s1,
                        s2a,
                        s2b,
                    );
                    indb -= 1;
                }
            }
        } else {
            // usual Gregory integration
            for l in 0..=m {
                let w = integ.giw.at(m as usize, l as usize);
                mul_acc_c(
                    blk,
                    sign.as_i64() as f64 * s * Complex::new(w, 0.0),
                    a.raw_block_at(n, m - l),
                    &bt[(ntau - l) as usize],
                    s1,
                    s2a,
                    s2b,
                );
            }
        }

        // contributions from tau to beta (NESSi Eqs. (115)-(116))
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
                        &bt[j as usize],
                        s1,
                        s2a,
                        s2b,
                    );
                }
                inda -= 1;
            }
        } else if m > ntau - 2 * k - 1 {
            // usual Gregory integration (short)
            let mut inda = m;
            for l in 0..=(ntau - m) {
                let w = integ.giw.at((ntau - m) as usize, l as usize);
                mul_acc_c(
                    blk,
                    s * Complex::new(w, 0.0),
                    a.raw_block_at(n, inda),
                    &bt[l as usize],
                    s1,
                    s2a,
                    s2b,
                );
                inda += 1;
            }
        } else {
            // usual Gregory integration (long)
            let mut inda = m;
            let mut indb = 0i64;
            for _l in m..=ntau {
                let w = integ.giw.at((ntau - m) as usize, (ntau - m - indb) as usize);
                mul_acc_c(
                    blk,
                    s * Complex::new(w, 0.0),
                    a.raw_block_at(n, inda),
                    &bt[indb as usize],
                    s1,
                    s2a,
                    s2b,
                );
                inda += 1;
                indb += 1;
            }
        }
    }
}

/// Evaluate the `C_2 + C_3` parts into the lmix row `n` of `c`
/// (blueprint `conv_lmix_mat`).
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
    let ws_mat: Vec<Vec<Complex>> = (0..=a.ntau as i64)
        .map(|m| b.block(m as usize).to_vec())
        .collect();
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
        &ws_mat,
        integ,
        sign,
        Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

#[allow(dead_code)]
/// `conv_lmix_mat` with a constant matrix `f_0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_lmix_mat_f0(
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
    let ws_mat: Vec<Vec<Complex>> = (0..=a.ntau as i64)
        .map(|m| {
            let mut out = vec![Complex::ZERO; f0.nrows() * b.size2];
            super::mul_acc(
                &mut out,
                1.0,
                f0.raw(),
                b.block(m as usize),
                f0.nrows(),
                f0.ncols(),
                b.size2,
            );
            out
        })
        .collect();
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

/// Increment the lmix row `tstp` of `C` by `alpha * [A * f * B]^⌉`
/// (blueprint `incr_convolution_lmix!`): the real-time part uses
/// `f(t'')`, the imaginary-time part uses `f[0]`.
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
    // C_2 + C_3: imaginary-time part with f_0 = f(0^-)
    let bt: Vec<Vec<Complex>> = match f {
        Some(cf) => {
            let f0 = cf.block(crate::types::MATSUBARA);
            (0..=a.ntau as i64)
                .map(|m| {
                    let mut out = vec![Complex::ZERO; f0.len() / a.size2 * b.size2];
                    let kdim = f0.len() / a.size2;
                    super::mul_acc(
                        &mut out,
                        1.0,
                        f0,
                        b.mat.block(m as usize),
                        a.size2,
                        kdim,
                        b.size2,
                    );
                    out
                })
                .collect()
        }
        None => (0..=a.ntau as i64)
            .map(|m| b.mat.block(m as usize).to_vec())
            .collect(),
    };
    let rowlen = (a.ntau + 1) * a.size1 * b.size2;
    let off = tstp as usize * rowlen;
    let row = &mut c.lmix.raw_mut()[off..off + rowlen];
    conv_lmix_mat_loop(
        row,
        tstp,
        &a.lmix,
        &bt,
        integ,
        a.sign,
        alpha * Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

/// `incr_convolution_lmix!` with a timestep (𝒻) target: the timestep
/// lmix component stores a SINGLE row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_lmix_step(
    tstp: i64,
    c: &mut GreenTimeStep,
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
        &mut c.lmix,
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
    let bt: Vec<Vec<Complex>> = match f {
        Some(cf) => {
            let f0 = cf.block(crate::types::MATSUBARA);
            (0..=a.ntau as i64)
                .map(|m| {
                    let kdim = f0.len() / a.size2;
                    let mut out =
                        vec![Complex::ZERO; a.size1 * b.size2.min(f0.len() / a.size2 * b.size2)];
                    out = vec![Complex::ZERO; a.size1 * b.size2];
                    super::mul_acc(
                        &mut out,
                        1.0,
                        f0,
                        b.mat.block(m as usize),
                        a.size2,
                        kdim,
                        b.size2,
                    );
                    out
                })
                .collect()
        }
        None => (0..=a.ntau as i64)
            .map(|m| b.mat.block(m as usize).to_vec())
            .collect(),
    };
    let row = c.lmix.raw_mut();
    conv_lmix_mat_loop(
        row,
        tstp,
        &a.lmix,
        &bt,
        integ,
        a.sign,
        alpha * Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::MATSUBARA;

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
            for m in 0..=ntau as i64 {
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
        for n in 1..=nt as i64 {
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
                (0, ntau as i64),
            );
        }
        for n in 1..=nt as i64 {
            for m in 0..=ntau as i64 {
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
        for i in 0..=nt as i64 {
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
        let _ = MATSUBARA;
    }
}
