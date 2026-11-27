// Project : KadanoffBaym
// File    : src/convolution/pseudo.rs
// Purpose : pseudo-particle convolution

//!
//! *Remarks* : *Pseudo-Particle Convolution*
//!
//! For the pseudo-particle Green's functions, only the Matsubara and the
//! left-mixing components of the convolution ``C = A \ast f \ast B`` differ
//! from the standard ones. The Matsubara component is a causal (restricted)
//! integral,
//!
//! ```math
//! \begin{equation}
//! C^{M}(\tau) = \int^{\tau}_{0} d\bar{\tau}
//!               A^{M}(\tau - \bar{\tau}) f_0 B^{M}(\bar{\tau}),
//! \end{equation}
//! ```
//!
//! without the periodic (or anti-periodic) contribution from ``\tau`` to
//! ``\beta``. In the left-mixing component, the imaginary time integral is
//! restricted to ``\tau \cdots \beta``:
//!
//! ```math
//! \begin{equation}
//! C^{\rceil}(t,\tau) = \int^{t}_{0} d\bar{t} A^{>}(t,\bar{t}) f(\bar{t})
//!                      B^{\rceil}(\bar{t},\tau)
//!                    + \int^{\beta}_{\tau} d\bar{\tau} A^{\rceil}(t,\bar{\tau})
//!                      f_0 B^{M}(\bar{\tau} - \tau).
//! \end{equation}
//! ```
//!
//! Here the stored `ret` component of `A` (i.e., ``A^{>}``) is used, and the
//! weights are ``h`` and ``\delta\tau``, respectively. No extra factor ``-i``
//! or statistical sign is involved. The retarded and lesser components are
//! evaluated by the standard kernels (see `crate::convolution`).
//!

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::types::{ContourGreen, GreenTimeStep, PseudoGreen, TimeFunction};
use crate::types::{LmixComp, MatComp};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::check::{check_conv_shapes, check_partner};
use crate::convolution::conv_ret_lmix_kernel;
use crate::convolution::less::incr_convolution_less_step;
use crate::convolution::ret::incr_convolution_ret_step;
use crate::convolution::{incr_convolution_less, incr_convolution_ret};
use crate::linalg::{mul_acc, mul_acc_c};

/// The restricted Matsubara integral `c = \int_0^tau dx A^M(tau - x) Bt(x)`
/// at the imaginary time point `m` (my index, `tau = m * dtau`) without
/// the factor `dtau`.
fn pseudo_mat_integral(
    c: &mut [Complex],
    m: i64,
    amat: &MatComp,
    bt: &[Vec<Complex>],
    integ: &Integrator,
) {
    let k = integ.k;
    let n = m;
    let d = amat.size1;
    c.fill(Complex::ZERO);
    if n > 2 * k {
        // usual Gregory integration with end corrections
        for j in 0..=k {
            mul_acc(
                c,
                integ.giw.omega[j as usize],
                amat.block((n - j) as usize),
                &bt[j as usize],
                d,
                d,
                d,
            );
        }
        for j in (k + 1)..=(n - k - 1) {
            mul_acc(
                c,
                1.0,
                amat.block((n - j) as usize),
                &bt[j as usize],
                d,
                d,
                d,
            );
        }
        for j in (n - k)..=n {
            mul_acc(
                c,
                integ.giw.omega[(n - j) as usize],
                amat.block((n - j) as usize),
                &bt[j as usize],
                d,
                d,
                d,
            );
        }
    } else if n >= k {
        // Gregory integration
        for j in 0..=n {
            mul_acc(
                c,
                integ.giw.at(n as usize, j as usize),
                amat.block((n - j) as usize),
                &bt[j as usize],
                d,
                d,
                d,
            );
        }
    } else if n > 0 {
        // strange boundary correction
        for l in 0..=k {
            for j in 0..=k {
                mul_acc(
                    c,
                    integ.bcw.at(n as usize, l as usize, j as usize),
                    amat.block(l as usize),
                    &bt[j as usize],
                    d,
                    d,
                    d,
                );
            }
        }
    }
}

/// Accumulate `s * \int_tau^beta dx A^] (t_n, x) Bt(x - tau)` into
/// `tgt[m]` for all `m`. Here
/// `tgt[m]` is the block of the scratch row at the imaginary time
/// point `m` (my index).
fn pseudo_lmix_mat_loop(
    tgt: &mut [Vec<Complex>],
    n: i64,
    a: &LmixComp,
    bt: &[Vec<Complex>],
    integ: &Integrator,
    s: Complex,
) {
    let k = integ.k;
    let ntau = a.ntau as i64;
    let d1 = a.size1;
    let d2 = a.size2;
    for m in 0..=ntau {
        let c = &mut tgt[m as usize];
        if m == ntau {
            // PASS
        } else if m > ntau - k {
            // strange boundary correction
            for l in 0..=k {
                for j in 0..=k {
                    let w = integ.bcw.at((ntau - m) as usize, l as usize, j as usize);
                    mul_acc_c(
                        c,
                        s.scale(w),
                        a.raw_block_at(n, ntau - l),
                        &bt[j as usize],
                        d1,
                        d2,
                        d2,
                    );
                }
            }
        } else if m > ntau - 2 * k - 1 {
            // usual Gregory integration (short range)
            for l in 0..=(ntau - m) {
                let w = integ.giw.at((ntau - m) as usize, l as usize);
                mul_acc_c(
                    c,
                    s.scale(w),
                    a.raw_block_at(n, m + l),
                    &bt[l as usize],
                    d1,
                    d2,
                    d2,
                );
            }
        } else {
            // usual Gregory integration
            for l in m..=ntau {
                let w = integ.giw.at((ntau - m) as usize, (ntau - l) as usize);
                mul_acc_c(
                    c,
                    s.scale(w),
                    a.raw_block_at(n, l),
                    &bt[(l - m) as usize],
                    d1,
                    d2,
                    d2,
                );
            }
        }
    }
}

/// Fill the scratch `row` (ntau + 1 blocks) with
/// `alpha * [A * f * B]^](t_tstp, tau_m)` (both parts). It is the
/// core of `pseudo_incr_convolution_lmix`.
#[allow(clippy::too_many_arguments)]
fn pseudo_lmix_row(
    row: &mut [Vec<Complex>],
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
    ws: &mut ConvWorkspace,
) {
    let ntau = a.ntau as i64;
    let dtau = beta / a.ntau as f64;
    for blk in row.iter_mut() {
        blk.fill(Complex::ZERO);
    }
    // imaginary time part, A^] * f_0 B^M restricted to tau .. beta
    let bt = f0_bmat_blocks(f, &b.mat, a.size2);
    pseudo_lmix_mat_loop(
        row,
        tstp,
        &a.lmix,
        &bt,
        integ,
        alpha * Complex::new(dtau, 0.0),
    );
    // real time part, A^{>} * f * B^], the same as the standard one
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    let mut scratch = LmixComp::new(1, a.ntau, a.size1, b.size2, Complex::ZERO);
    conv_ret_lmix_kernel(
        &mut scratch,
        0,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.lmix,
        integ,
        alpha * Complex::new(h, 0.0),
        ws,
        (0, ntau),
    );
    for (m, rowm) in row.iter_mut().enumerate() {
        let src = scratch.raw_block_at(0, m as i64);
        for (rq, sq) in rowm.iter_mut().zip(src.iter()) {
            *rq += *sq;
        }
    }
}

/// Increment the Matsubara component of `C` by
/// `alpha * dtau * \int_0^tau dx A^M(tau - x) f_0 B^M(x)`.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_incr_convolution_mat(
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    alpha: Complex,
    mask: Option<&[bool]>,
) {
    assert!(beta > 0.0);
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    check_pseudo_mask(mask, a.ntau + 1);
    let ac = a.to_contour();
    let bc = b.to_contour();
    let ntau = a.ntau as i64;
    let dtau = beta / a.ntau as f64;
    let bt = f0_bmat_blocks(f, &bc.mat, a.size2);
    let mut cblk = vec![Complex::ZERO; a.size1 * b.size1];
    let w = alpha * Complex::new(dtau, 0.0);
    for m in 0..=ntau as usize {
        if !is_selected(mask, m) {
            continue;
        }
        pseudo_mat_integral(&mut cblk, m as i64, &ac.mat, &bt, integ);
        let tgt = c.mat.block_mut(m);
        for q in 0..tgt.len() {
            tgt[q] += w * cblk[q];
        }
    }
}

/// `pseudo_incr_convolution_lmix` with a `PseudoGreen` target.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_incr_convolution_lmix(
    tstp: i64,
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    acc: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    _bcc: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
    mask: Option<&[bool]>,
) {
    assert!(tstp >= 0, "tstp must be nonnegative");
    assert!(beta > 0.0);
    assert!(h > 0.0);
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    let (ac, accc, bc) = (a.to_contour(), acc.to_contour(), b.to_contour());
    let mut cc = c.to_contour();
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    pseudo_incr_convolution_lmix_impl(
        &mut cc.lmix,
        tstp,
        tstp,
        &ac,
        &accc,
        f,
        &bc,
        integ,
        beta,
        h,
        alpha,
        mask,
        &mut ws,
    );
    c.copy_from_contour(&cc);
}

/// [`pseudo_incr_convolution_lmix`] with a `GreenTimeStep` target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_incr_convolution_lmix_step(
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
    mask: Option<&[bool]>,
) {
    assert!(tstp >= 0, "tstp must be nonnegative");
    assert!(beta > 0.0);
    assert!(h > 0.0);
    check_conv_shapes(
        (c.size1, c.size2),
        (a.size1, a.size2),
        f.map(|cf| (cf.size1, cf.size2)),
        (b.size1, b.size2),
    );
    check_partner("Acc", (acc.size1, acc.size2), (a.size1, a.size2));
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    pseudo_incr_convolution_lmix_impl(
        &mut c.lmix,
        0,
        tstp,
        a,
        acc,
        f,
        b,
        integ,
        beta,
        h,
        alpha,
        mask,
        &mut ws,
    );
}

/// Compute the masked left-mixing scratch row of
/// `alpha * [A * f * B]^]` at time step `tstp` and accumulate it into
/// row `target_row` of `target`.
#[allow(clippy::too_many_arguments)]
fn pseudo_incr_convolution_lmix_impl(
    target: &mut LmixComp,
    target_row: i64,
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
    mask: Option<&[bool]>,
    ws: &mut ConvWorkspace,
) {
    check_pseudo_mask(mask, a.ntau + 1);
    let mut row: Vec<Vec<Complex>> = (0..=a.ntau)
        .map(|_| vec![Complex::ZERO; a.size1 * b.size2])
        .collect();
    pseudo_lmix_row(&mut row, tstp, a, acc, f, b, integ, beta, h, alpha, ws);
    for (m, rowm) in row.iter().enumerate() {
        if !is_selected(mask, m) {
            continue;
        }
        let tgt = target.raw_block_at_mut(target_row, m as i64);
        for (tq, rq) in tgt.iter_mut().zip(rowm.iter()) {
            *tq += *rq;
        }
    }
}

/// Increment the pseudo-particle Green's function `C` at time step
/// `tstp` by `alpha * [A * f * B]`.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_incr_convolution(
    tstp: i64,
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    acc: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    bcc: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    if tstp == MATSUBARA {
        pseudo_incr_convolution_mat(c, a, f, b, integ, beta, alpha, None);
    } else {
        assert!(beta > 0.0);
        assert!(h > 0.0);
        let (ac, accc, bc, bccc) = (
            a.to_contour(),
            acc.to_contour(),
            b.to_contour(),
            bcc.to_contour(),
        );
        let mut cc = c.to_contour();
        let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
        incr_convolution_ret(tstp, &mut cc, &ac, &accc, f, &bc, &bccc, integ, h, alpha);
        pseudo_incr_convolution_lmix_impl(
            &mut cc.lmix,
            tstp,
            tstp,
            &ac,
            &accc,
            f,
            &bc,
            integ,
            beta,
            h,
            alpha,
            None,
            &mut ws,
        );
        incr_convolution_less(
            tstp, &mut cc, &ac, &accc, f, &bc, &bccc, integ, beta, h, alpha,
        );
        c.copy_from_contour(&cc);
    }
}

/// `pseudo_incr_convolution` with a timestep (`f^{ps}`) target: the
/// greater and lesser parts by the standard step kernels, the
/// left-mixing part by the pseudo-particle kernel.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_incr_convolution_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &PseudoGreen,
    acc: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    bcc: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    if tstp == MATSUBARA {
        // restricted Matsubara component into the timestep's mat
        let ac = a.to_contour();
        let bc = b.to_contour();
        let ntau = a.ntau as i64;
        let dtau = beta / a.ntau as f64;
        let bt = f0_bmat_blocks(f, &bc.mat, a.size2);
        let mut cblk = vec![Complex::ZERO; a.size1 * b.size1];
        let w = alpha * Complex::new(dtau, 0.0);
        for m in 0..=ntau as usize {
            pseudo_mat_integral(&mut cblk, m as i64, &ac.mat, &bt, integ);
            let tgt = c.mat.block_mut(m);
            for q in 0..tgt.len() {
                tgt[q] += w * cblk[q];
            }
        }
    } else {
        assert!(beta > 0.0);
        assert!(h > 0.0);
        let (ac, accc, bc, bccc) = (
            a.to_contour(),
            acc.to_contour(),
            b.to_contour(),
            bcc.to_contour(),
        );
        incr_convolution_ret_step(tstp, c, &ac, &accc, f, &bc, &bccc, integ, h, alpha);
        pseudo_incr_convolution_lmix_step(
            tstp, c, &ac, &accc, f, &bc, &bccc, integ, beta, h, alpha, None,
        );
        incr_convolution_less_step(tstp, c, &ac, &accc, f, &bc, &bccc, integ, beta, h, alpha);
    }
}

/// Calculate the Matsubara component of the pseudo-particle convolution
/// `C = A * f * B`.
pub fn pseudo_convolution_mat(
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
) {
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    c.zeros_step(MATSUBARA);
    pseudo_incr_convolution_mat(c, a, f, b, integ, beta, Complex::ONE, None);
}

/// Calculate the pseudo-particle convolution `C = A * f * B` at time
/// step `tstp`.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_convolution_time_step(
    tstp: i64,
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    acc: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    bcc: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    assert!(tstp == MATSUBARA || (0..=c.nt as i64).contains(&tstp));
    assert!((1..=5).contains(&integ.k));
    if tstp != MATSUBARA {
        // the start-up time steps need the data at the first k + 1 steps
        let nmin = tstp.max(integ.k + 1);
        assert!(a.nt as i64 >= nmin && acc.nt as i64 >= nmin);
        assert!(b.nt as i64 >= nmin && bcc.nt as i64 >= nmin);
        if let Some(cf) = f {
            assert!(cf.nt as i64 >= nmin);
        }
    }
    c.zeros_step(tstp);
    pseudo_incr_convolution(tstp, c, a, acc, f, b, bcc, integ, beta, h, Complex::ONE);
}

/// Calculate the pseudo-particle convolution `C = A * f * B` at all the
/// time steps.
#[allow(clippy::too_many_arguments)]
pub fn pseudo_convolution(
    c: &mut PseudoGreen,
    a: &PseudoGreen,
    acc: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
    bcc: &PseudoGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    assert!(beta > 0.0);
    assert!(h > 0.0);
    check_pseudo_shapes((c.size1, c.size2), a, f, b);
    let (ac, accc, bc, bccc) = (
        a.to_contour(),
        acc.to_contour(),
        b.to_contour(),
        bcc.to_contour(),
    );
    let mut cc = c.to_contour();
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    // Matsubara component
    cc.mat.raw_mut().fill(Complex::ZERO);
    let ntau = a.ntau as i64;
    let dtau = beta / a.ntau as f64;
    let bt = f0_bmat_blocks(f, &bc.mat, a.size2);
    let mut cblk = vec![Complex::ZERO; a.size1 * b.size1];
    for m in 0..=ntau as usize {
        pseudo_mat_integral(&mut cblk, m as i64, &ac.mat, &bt, integ);
        let tgt = cc.mat.block_mut(m);
        for q in 0..tgt.len() {
            tgt[q] += Complex::new(dtau, 0.0) * cblk[q];
        }
    }
    // real time components
    for tstp in 0..=cc.nt as i64 {
        for j in 0..=tstp {
            cc.ret.raw_block_at_mut(tstp, j).fill(Complex::ZERO);
            cc.less.raw_block_at_mut(j, tstp).fill(Complex::ZERO);
        }
        for m in 0..=ntau {
            cc.lmix.raw_block_at_mut(tstp, m).fill(Complex::ZERO);
        }
        incr_convolution_ret(
            tstp,
            &mut cc,
            &ac,
            &accc,
            f,
            &bc,
            &bccc,
            integ,
            h,
            Complex::ONE,
        );
        pseudo_incr_convolution_lmix_impl(
            &mut cc.lmix,
            tstp,
            tstp,
            &ac,
            &accc,
            f,
            &bc,
            integ,
            beta,
            h,
            Complex::ONE,
            None,
            &mut ws,
        );
        incr_convolution_less(
            tstp,
            &mut cc,
            &ac,
            &accc,
            f,
            &bc,
            &bccc,
            integ,
            beta,
            h,
            Complex::ONE,
        );
    }
    c.copy_from_contour(&cc);
}

/// The blocks `f_0 B^{M}(tau_m)`; `f = None` means `f = 1`.
fn f0_bmat_blocks(f: Option<&TimeFunction>, bmat: &MatComp, s2a: usize) -> Vec<Vec<Complex>> {
    match f {
        Some(cf) => {
            let f0 = cf.block(MATSUBARA);
            let kdim = f0.len() / s2a;
            (0..=bmat.ntau as i64)
                .map(|m| {
                    let mut out = vec![Complex::ZERO; s2a * bmat.size2];
                    mul_acc(
                        &mut out,
                        1.0,
                        f0,
                        bmat.block(m as usize),
                        s2a,
                        kdim,
                        bmat.size2,
                    );
                    out
                })
                .collect()
        }
        None => (0..=bmat.ntau as i64)
            .map(|m| bmat.block(m as usize).to_vec())
            .collect(),
    }
}

/// Shape contract of the pseudo-particle convolution `C = A * f * B`:
/// the objects are square of one size, and so is `f`.
fn check_pseudo_shapes(
    c: (usize, usize),
    a: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
) {
    let d = a.size1;
    let ok = a.size2 == d
        && b.size1 == d
        && b.size2 == d
        && c.0 == d
        && c.1 == d
        && f.is_none_or(|cf| cf.size1 == d && cf.size2 == d);
    assert!(
        ok,
        "pseudo convolution: inconsistent shapes: A {}x{}, f {:?}, B {}x{}, C {}x{} (one square size is needed)",
        a.size1,
        a.size2,
        f.map(|cf| (cf.size1, cf.size2)),
        b.size1,
        b.size2,
        c.0,
        c.1
    );
}

/// Check the optional mask.
fn check_pseudo_mask(mask: Option<&[bool]>, n: usize) {
    if let Some(mk) = mask {
        assert_eq!(mk.len(), n, "the mask must have {n} entries");
    }
}

fn is_selected(mask: Option<&[bool]>, m: usize) -> bool {
    match mask {
        Some(mk) => mk[m],
        None => true,
    }
}
