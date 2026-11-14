// Project : KadanoffBaym
// File    : src/convolution/less.rs
// Purpose : lesser convolution kernels (blueprint: convolution/less.jl)
//!
//! ### *Convolution* : ``G^{<}`` *Component*
//!
//! *Remarks* :
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
//! Green's functions are not really stored in ℱ., these quantities must be
//! reconstructed from the hermitian conjugate. For examples:
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
//! *References* :
//!
//! Please see [`NESSi`] Sections `3.1`, `9` and `11` for more details.

use super::{adjoint_block, mul_acc_c, ConvWorkspace, FOp};
use crate::complex::Complex;
use crate::types::sign::Sign;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::function::TimeFunction;
use crate::types::less::LessComp;
use crate::types::lmix::LmixComp;
use crate::types::ret::RetComp;
use crate::weights::Integrator;

/// The `C_1` part: accumulate `s * [A^R * f * B^<](t_m, t_n)` into
/// `tgt.less(m, n)` for `m` in `ms` (blueprint `_conv_ret_less_kernel!`,
/// NESSi Eqs. (117)-(118)).
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
    for j in 0..=n1r {
        if j <= n {
            let bj = super::fb(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
            conv_ret_less_col(tgt, n, ms, j, bj, s, a, acc, integ, b.size1, b.size2);
        } else {
            let badj = adjoint_block(bcc.raw_block_at(n, j), bcc.size1, bcc.size2);
            let bj = super::fb(ws, &f, j, &badj, bcc.size2, bcc.size1);
            conv_ret_less_col(tgt, n, ms, j, bj, -s, a, acc, integ, bcc.size2, bcc.size1);
        }
    }
}

/// The `j`-th term of NESSi Eq. (118) (blueprint `_conv_ret_less_col!`).
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
) {
    let _ = s1b;
    let k = integ.k;
    let s1 = a.size1;
    let s2a = a.size2;
    for m in ms.0..=ms.1 {
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
                let aadj = adjoint_block(acc.raw_block_at(j, m), acc.size1, acc.size2);
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, -s.scale(w), &aadj, b, acc.size2, acc.size1, s1b);
            } else {
                let blk = tgt.raw_block_at_mut(m, col);
                mul_acc_c(blk, s.scale(w), a.raw_block_at(m, j), b, s1, s2a, s1b);
            }
        }
    }
}

/// The `C_2` part: accumulate `s * [A^< * f * B^A](t_m, t_n)` into
/// `tgt.less(m, n)` (blueprint `_conv_less_adv_kernel!`, NESSi
/// Eqs. (119)-(120)). With `diagcc = true` the diagonal block
/// `A^<(t_m, t_m)` is replaced by `-Acc^<(t_m, t_m)^dagger` (NESSi
/// start-up convention).
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
    for j in 0..=n1r {
        let w = integ.giw.at(n as usize, j as usize);
        if j <= n {
            let badj = adjoint_block(bcc.raw_block_at(n, j), bcc.size1, bcc.size2);
            let bj = super::fb(ws, &f, j, &badj, bcc.size2, bcc.size1);
            conv_less_adv_col(tgt, n, diagcc, ms, j, bj, s.scale(w), a, acc, bcc.size2, bcc.size1);
        } else {
            let bj = super::fb(ws, &f, j, b.raw_block_at(j, n), b.size1, b.size2);
            conv_less_adv_col(tgt, n, diagcc, ms, j, bj, -s.scale(w), a, acc, b.size1, b.size2);
        }
    }
}

/// The `j`-th term of NESSi Eq. (120) (blueprint `_conv_less_adv_col!`).
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
) {
    let _ = s1b;
    let s1 = a.size1;
    let s2a = a.size2;
    for m in ms.0..=ms.1 {
        if j < m || (j == m && diagcc) {
            let aadj = adjoint_block(acc.raw_block_at(j, m), acc.size1, acc.size2);
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, -alpha, &aadj, b, acc.size2, acc.size1, s1b);
        } else {
            let blk = tgt.raw_block_at_mut(m, col);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), b, s1, s2a, s1b);
        }
    }
}

/// The `C_3` part: accumulate `s * [A^⌉ * f_0 * B^⌈](t_m, t_n) / (-i)`
/// into `tgt.less(m, n)` (blueprint `_conv_lmix_rmix_kernel!`, NESSi
/// Eqs. (121)-(122)); the factor `-i*dtau` must be included in `s`.
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
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^⌈(tau_j, t_n) = B^⌉(t_n, tau_{N-j})^dagger
        let badj = adjoint_block(bcc.raw_block_at(n, ntau - j), bcc.size1, bcc.size2);
        let bj = match f0 {
            Some(f0blk) => {
                let s1f = f0blk.len() / bcc.size2;
                let mut out = vec![Complex::ZERO; s1f * bcc.size1];
                super::mul_acc(&mut out, 1.0, f0blk, &badj, s1f, bcc.size2, bcc.size1);
                out
            }
            None => badj,
        };
        let alpha = -s * Complex::new(sign.as_i64() as f64 * w, 0.0);
        let s1 = a.size1;
        let s2a = a.size2;
        let s1b = bj.len() / bcc.size1;
        for m in ms.0..=ms.1 {
            let blk = tgt.raw_block_at_mut(m, n);
            mul_acc_c(blk, alpha, a.raw_block_at(m, j), &bj, s1, s2a, s1b);
        }
    }
}

/// NESSi start-up convention: `diagcc = true` iff `tstp <= 2k + 2`
/// (0-based `tstp`, blueprint `_incr_les_diagcc`).
///
/// NESSi startup convention (blueprint `_incr_les_diagcc`):
/// In incr_convolution_les of NESSi, the diagonal block A^<(t,t) of the
/// term A^< ∗ f ∗ Bᴬ is taken as -Acc^<(t,t)† if tstp ≤ 2k + 2 (NESSi
/// index). The two choices agree if Acc is the hermitian conjugate partner
/// of A. See [NESSi] cntr_convolution_impl.hpp, incr_convolution_les.
pub(crate) fn incr_les_diagcc(n: i64, integ: &Integrator) -> bool {
    n <= 2 * integ.k + 2
}

#[allow(dead_code)]
/// Public `conv_ret_less` (blueprint `conv_ret_less`).
#[allow(clippy::too_many_arguments)]
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
/// They are evaluated by `incr_convolution_ret_less!`,
/// `incr_convolution_less_adv!`, and `incr_convolution_lmix_rmix!`,
/// respectively. In the time stepping of the Dyson and VIE2 equations, only
/// the latter two terms enter the source term, while the first one belongs
/// to the implicit linear system. None of them reads the target `C`.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_convolution_impl.hpp`,
/// `convolution_timestep_les_retles`, `convolution_timestep_les_lesadv`, and
/// `convolution_timestep_les_tvvt`.
pub(crate) fn conv_ret_less(
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
    for m in 0..=n {
        c.less.raw_block_at_mut(m, n).fill(Complex::ZERO);
    }
    conv_ret_less_kernel(&mut c.less, (0, n), n, &a.ret, &acc.ret, f, &b.less, &bcc.less, integ, Complex::new(h, 0.0), &mut ws);
}

#[allow(dead_code)]
/// Public `conv_less_adv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_less_adv(
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
    for m in 0..=n {
        c.less.raw_block_at_mut(m, n).fill(Complex::ZERO);
    }
    conv_less_adv_kernel(&mut c.less, (0, n), n, &a.less, &acc.less, f, &b.ret, &bcc.ret, integ, Complex::new(h, 0.0), &mut ws, false);
}

/// Increment `C^<(t', t)` (`t' <= t = tstp`) by
/// `alpha * [A * f * B]^<` (blueprint `incr_convolution_les!`): the sum
/// of the three partial lesser convolutions.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_les(
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
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    let diagcc = incr_les_diagcc(tstp, integ);
    conv_ret_less_kernel(&mut c.less, (0, tstp), tstp, &a.ret, &acc.ret, fop, &b.less, &bcc.less, integ, alpha * Complex::new(h, 0.0), &mut ws);
    conv_less_adv_kernel(&mut c.less, (0, tstp), tstp, &a.less, &acc.less, fop, &b.ret, &bcc.ret, integ, alpha * Complex::new(h, 0.0), &mut ws, diagcc);
    let f0 = f.map(|cf| cf.block(crate::types::MATSUBARA));
    conv_lmix_rmix_kernel(&mut c.less, (0, tstp), tstp, &a.lmix, f0, &bcc.lmix, integ, a.sign, alpha * Complex::new(0.0, -(beta / a.ntau as f64)), &mut ws);
}

/// `incr_convolution_les!` with a timestep (𝒻) target.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_les_step(
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
    assert!(tstp >= 0);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    let diagcc = incr_les_diagcc(tstp, integ);
    conv_ret_less_kernel(&mut c.less, (0, tstp), tstp, &a.ret, &acc.ret, fop, &b.less, &bcc.less, integ, alpha * Complex::new(h, 0.0), &mut ws);
    conv_less_adv_kernel(&mut c.less, (0, tstp), tstp, &a.less, &acc.less, fop, &b.ret, &bcc.ret, integ, alpha * Complex::new(h, 0.0), &mut ws, diagcc);
    let f0 = f.map(|cf| cf.block(crate::types::MATSUBARA));
    conv_lmix_rmix_kernel(&mut c.less, (0, tstp), tstp, &a.lmix, f0, &bcc.lmix, integ, a.sign, alpha * Complex::new(0.0, -(beta / a.ntau as f64)), &mut ws);
}

/// Partial increment: `incr_convolution_ret_less!`.
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
    conv_ret_less_kernel(&mut c.less, (0, tstp), tstp, &a.ret, &acc.ret, fop, &b.less, &bcc.less, integ, alpha * Complex::new(h, 0.0), &mut ws);
}

/// Partial increment: `incr_convolution_less_adv!`.
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
    assert!(tstp >= 0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    conv_less_adv_kernel(&mut c.less, (0, tstp), tstp, &a.less, &acc.less, fop, &b.ret, &bcc.ret, integ, alpha * Complex::new(h, 0.0), &mut ws, false);
}

/// Partial increment: `incr_convolution_lmix_rmix!`.
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
    let f0 = f.map(|cf| cf.block(crate::types::MATSUBARA));
    conv_lmix_rmix_kernel(&mut c.less, (0, tstp), tstp, &a.lmix, f0, &bcc.lmix, integ, a.sign, alpha * Complex::new(0.0, -(beta / a.ntau as f64)), &mut ws);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;

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
        incr_convolution_les(4, &mut cg, &g, &acc, None, &zb, &zb, &integ, 1.0, 0.1, Complex::ONE);
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
            conv_ret_less_kernel(&mut cg.less, (0, n), n, &g.ret, &acc.ret, FOp::None, &g.less, &acc.less, &integ, Complex::new(h, 0.0), &mut ws);
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
            conv_less_adv_kernel(&mut cg.less, (0, n), n, &g.less, &acc.less, FOp::None, &g.ret, &acc.ret, &integ, Complex::new(h, 0.0), &mut ws, false);
        }
        // hand-traced values of the discretized formula for the toy
        // +-1 fill (dagger partners negate)
        assert!((cg.get_less(0, 1, 0, 0) - Complex::new(-h, 0.0)).abs() < 1e-12);
        assert!((cg.get_less(1, 1, 0, 0) - Complex::new(-h, 0.0)).abs() < 1e-12);
        assert!((cg.get_less(3, 5, 0, 0) - Complex::new(-5.0 * h, 0.0)).abs() < 1e-12, "{}", cg.get_less(3, 5, 0, 0));
    }

    #[test]
    fn c3_part_is_pure_imaginary_structure() {
        let (c, g, _acc) = setup(4, 20);
        let integ = Integrator::new(3);
        let beta = 1.0;
        let dtau = beta / 20.0;
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        conv_lmix_rmix_kernel(&mut cg.less, (0, 2), 2, &g.lmix, None, &g.lmix, &integ, Sign::Fermi, Complex::new(0.0, -dtau), &mut ws);
        let got = cg.get_less(1, 2, 0, 0);
        assert!(got.re.abs() < 1e-12);
        // -i * dtau * (-sign) * sum(Omega[N]) = -i * beta for fermions
        assert!((got.im + beta).abs() < 1e-12, "got {got}");
    }

}


/// Partial slice-target increment: `incr_convolution_less_adv!` with a
/// timestep object target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_less_adv_step(
    tstp: i64,
    c: &mut crate::types::step::GreenTimeStep,
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
    // The blueprint's *partial* `_incr_convolution_less_adv!` defaults to
    // `diagcc = false` and the standard Dyson/VIE2 solvers use it without
    // the start-up substitution (only the combined `incr_convolution_les`
    // and the pseudo-particle VIE2 time step apply the NESSi rule).
    conv_less_adv_kernel(
        &mut c.less,
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

/// Partial slice-target increment: `incr_convolution_lmix_rmix!` with a
/// timestep object target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_lmix_rmix_step(
    tstp: i64,
    c: &mut crate::types::step::GreenTimeStep,
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
    let f0 = f.map(|cf| cf.block(crate::types::MATSUBARA));
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
// Scratch-buffer startup kernels (blueprint `_incr_lmix_rmix!` and
// `_incr_less_adv!` with `Vector{Matrix}` targets): the dyson_start
// lesser solve writes into a private buffer instead of a green.
// ------------------------------------------------------------------

/// Buffer-target variant of [`incr_convolution_lmix_rmix`]: for every
/// `m` in `0..=tstp`, `buf[m] += alpha * [A^| * Bcc^ceil]`.
#[allow(clippy::too_many_arguments)]
pub fn incr_lmix_rmix_buf(
    buf: &mut [Vec<Complex>],
    tstp: i64,
    a: &crate::types::ContourGreen,
    f0: Option<&[Complex]>,
    bcc: &crate::types::ContourGreen,
    integ: &crate::weights::Integrator,
    sign: crate::types::sign::Sign,
    beta: f64,
    _h: f64,
    alpha: Complex,
    _ws: &mut crate::convolution::ConvWorkspace,
) {
    // s = alpha * (-i * delta_tau), as in the public green-target version
    let dtau = beta / a.ntau as f64;
    let s = alpha * Complex::new(0.0, -dtau);
    let ntau = a.ntau as i64;
    for j in 0..=ntau {
        let w = integ.giw.at(ntau as usize, j as usize);
        // B^ceil(tau_j, t_n) = B^rangle(t_n, tau_{N-j})^dagger
        let badj = adjoint_block(
            bcc.lmix.raw_block_at(tstp, ntau - j),
            bcc.lmix.size1,
            bcc.lmix.size2,
        );
        let bj = match f0 {
            Some(f0blk) => {
                let s1f = f0blk.len() / bcc.lmix.size2;
                let mut out = vec![Complex::ZERO; s1f * bcc.lmix.size1];
                super::mul_acc(&mut out, 1.0, f0blk, &badj, s1f, bcc.lmix.size2, bcc.lmix.size1);
                out
            }
            None => badj,
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
                &bj,
                s1,
                s2a,
                s2b,
            );
        }
    }
}

/// Buffer-target variant of [`incr_convolution_less_adv`]:
/// `buf[m] += alpha * [Sigma^< * G^A](m, tstp)`.
#[allow(clippy::too_many_arguments)]
pub fn incr_less_adv_buf(
    buf: &mut [Vec<Complex>],
    tstp: i64,
    a: &crate::types::ContourGreen,
    acc: &crate::types::ContourGreen,
    f: &crate::convolution::FOp,
    b: &crate::types::ContourGreen,
    bcc: &crate::types::ContourGreen,
    integ: &crate::weights::Integrator,
    _beta: f64,
    h: f64,
    alpha: Complex,
    ws: &mut crate::convolution::ConvWorkspace,
    diagcc: bool,
) {
    // s = alpha * h, as in the public green-target version
    let s = alpha * Complex::new(h, 0.0);
    let n1r = tstp.max(integ.k);
    for j in 0..=n1r {
        let w = integ.giw.at(tstp as usize, j as usize);
        if j <= tstp {
            let badj = adjoint_block(bcc.ret.raw_block_at(tstp, j), bcc.ret.size1, bcc.ret.size2);
            let bj = super::fb(ws, f, j, &badj, bcc.ret.size2, bcc.ret.size1);
            let (s1b, s2b) = (bcc.ret.size2, bcc.ret.size1);
            conv_less_adv_col_buf(buf, tstp, j, bj, s * Complex::new(w, 0.0), a, acc, diagcc, s1b, s2b);
        } else {
            let braw = b.ret.raw_block_at(j, tstp);
            let bj = super::fb(ws, f, j, braw, b.ret.size1, b.ret.size2);
            let (s1b, s2b) = (b.ret.size1, b.ret.size2);
            conv_less_adv_col_buf(buf, tstp, j, bj, -s * Complex::new(w, 0.0), a, acc, diagcc, s1b, s2b);
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
    a: &crate::types::ContourGreen,
    acc: &crate::types::ContourGreen,
    diagcc: bool,
    s1b: usize,
    s2b: usize,
) {
    let s1 = a.less.size1;
    let s2a = a.less.size2;
    for (m, bufm) in buf.iter_mut().enumerate() {
        let m = m as i64;
        if j < m || (j == m && diagcc) {
            let aadj = adjoint_block(acc.less.raw_block_at(j, m), acc.less.size1, acc.less.size2);
            mul_acc_c(bufm, -alpha, &aadj, b, acc.less.size2, acc.less.size1, s2b);
        } else {
            let adir = a.less.raw_block_at(m, j);
            mul_acc_c(bufm, alpha, adir, b, s1, s2a, s2b);
        }
        let _ = s1b;
    }
}
