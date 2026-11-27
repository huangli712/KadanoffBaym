// Project : KadanoffBaym
// File    : src/convolution/bubble.rs
// Purpose : bubble diagrams

//!
//! *Remarks* : *Bubble Diagrams*
//!
//! Two types of bubble diagrams are supported. They are evaluated for given
//! orbital indices and at a given time step.
//!
//! **Bubble1**:
//!
//! ```math
//! \begin{equation}
//! C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t') B_{b_2b_1}(t',t).
//! \end{equation}
//! ```
//!
//! A typical example is the polarization, ``\Pi = -i G G``. The components
//! are
//!
//! ```math
//! \begin{equation}
//! C^{M}(\tau) = -\xi_B A^{M}_{a_1a_2}(\tau) B^{M}_{b_2b_1}(\beta - \tau),
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{>}(t,t') = i A^{>}_{a_1a_2}(t,t') B^{<}_{b_2b_1}(t',t), \quad
//! C^{<}(t,t') = i A^{<}_{a_1a_2}(t,t') B^{>}_{b_2b_1}(t',t),
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{R} = C^{>} - C^{<}, \quad
//! C^{\rceil}(t,\tau) = i A^{\rceil}_{a_1a_2}(t,\tau)
//!     B^{\lceil}_{b_2b_1}(\tau,t),
//! \end{equation}
//! ```
//!
//! where ``B^{\lceil}_{b_2b_1}(\tau,t) = -\xi_B [B^{\rceil}_{b_1b_2}(t,\beta -
//! \tau)]^{*}``.
//!
//! **Bubble2**:
//!
//! ```math
//! \begin{equation}
//! C_{c_1c_2}(t,t') = i A_{a_1a_2}(t,t') B_{b_1b_2}(t,t').
//! \end{equation}
//! ```
//!
//! A typical example is the GW self-energy, ``\Sigma = i G W``. The
//! components are
//!
//! ```math
//! \begin{equation}
//! C^{M}(\tau) = -A^{M}_{a_1a_2}(\tau) B^{M}_{b_1b_2}(\tau), \quad
//! C^{\gtrless}(t,t') = i A^{\gtrless}_{a_1a_2}(t,t') B^{\gtrless}_{b_1b_2}(t,t'),
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! C^{R} = C^{>} - C^{<}, \quad
//! C^{\rceil}(t,\tau) = i A^{\rceil}_{a_1a_2}(t,\tau) B^{\rceil}_{b_1b_2}(t,\tau).
//! \end{equation}
//! ```
//!
//! The values at ``t' > t`` are obtained from the hermitian conjugate
//! partners, e.g., ``A^{<}_{12}(t,t') = -[A^{cc,<}_{21}(t',t)]^{*}``. At time
//! step `tstp`, only ``C^{R}(t,t')``, ``C^{<}(t',t)`` (``t' \le t``), and
//! ``C^{\rceil}(t,\tau)`` are written. At `tstp = 0`, only ``C^{M}`` is
//! written. The other orbital elements of `C` are not changed.
//!

use crate::complex::Complex;
use crate::convolution::check::{check_bubble, check_bubble_orbital, check_moving_bubble};
use crate::types::{ContourGreen, FftDomain, MovingGreen, SteadyGreen};

use crate::types::MATSUBARA;

/// Bubble1 diagram `C_{c1c2}(t,t') = i A_{a1a2}(t,t') B_{b2b1}(t',t)`
/// at time step `tstp`. All indices are 0-based. Only the ret row, less
/// column, and lmix row at `tstp` are written (mat only at `MATSUBARA`).
#[allow(clippy::too_many_arguments)]
pub fn bubble1(
    tstp: i64,
    c: &mut ContourGreen,
    c1: usize,
    c2: usize,
    a: &ContourGreen,
    acc: &ContourGreen,
    a1: usize,
    a2: usize,
    b: &ContourGreen,
    bcc: &ContourGreen,
    b1: usize,
    b2: usize,
) {
    check_bubble(tstp, c, a, acc, b, bcc);
    check_bubble_orbital(&[
        (c1, c.size1, "c1"),
        (c2, c.size2, "c2"),
        (a1, a.size1, "a1"),
        (a2, a.size2, "a2"),
        // B enters with its orbital indices swapped
        (b1, b.size2, "b1"),
        (b2, b.size1, "b2"),
    ]);
    let i_c = Complex::new(0.0, 1.0);
    if tstp == MATSUBARA {
        let ntau = c.ntau as i64;
        let sig = -(b.sign.as_i64() as f64);
        for m in 0..=ntau {
            let am = block_elem(a.mat.block(a.mat.check_tau(m)), a.size2, a1, a2);
            let bm = block_elem(b.mat.block(b.mat.check_tau(ntau - m)), b.size2, b2, b1);
            set_block_elem(
                c.mat.block_mut(c.mat.check_tau(m)),
                c.size2,
                c1,
                c2,
                sig * am * bm,
            );
        }
    } else {
        let ntau = c.ntau as i64;
        let msigb = -(b.sign.as_i64() as f64);
        for m in 0..=tstp {
            // the stored blocks of the pairs (tstp, m) and (m, tstp)
            let bcc_les = bcc.less.block(bcc.less.idx(m, tstp));
            let bcc_ret = bcc.ret.block(bcc.ret.idx(tstp, m));
            let b_les = b.less.block(b.less.idx(m, tstp));
            let b_ret = b.ret.block(b.ret.idx(tstp, m));
            let acc_les = acc.less.block(acc.less.idx(m, tstp));
            let a_les = a.less.block(a.less.idx(m, tstp));
            let a_ret = a.ret.block(a.ret.idx(tstp, m));
            let bles21_tt1 = -block_elem(bcc_les, bcc.size2, b1, b2).conj();
            let bgtr21_tt1 = block_elem(b_ret, b.size2, b2, b1) + bles21_tt1;
            let bles21_t1t = block_elem(b_les, b.size2, b2, b1);
            let bgtr21_t1t = -block_elem(bcc_ret, bcc.size2, b1, b2).conj() + bles21_t1t;
            let ales12_tt1 = -block_elem(acc_les, acc.size2, a2, a1).conj();
            let agtr12_tt1 = block_elem(a_ret, a.size2, a1, a2) + ales12_tt1;
            let cgtr_tt1 = i_c * agtr12_tt1 * bles21_t1t;
            let cles_tt1 = i_c * ales12_tt1 * bgtr21_t1t;
            set_block_elem(
                c.ret.block_mut(c.ret.idx(tstp, m)),
                c.size2,
                c1,
                c2,
                cgtr_tt1 - cles_tt1,
            );
            set_block_elem(
                c.less.block_mut(c.less.idx(m, tstp)),
                c.size2,
                c1,
                c2,
                i_c * block_elem(a_les, a.size2, a1, a2) * bgtr21_tt1,
            );
        }
        for m in 0..=ntau {
            let bvt21 = msigb
                * block_elem(
                    bcc.lmix.block(bcc.lmix.idx(tstp, ntau - m)),
                    bcc.size2,
                    b1,
                    b2,
                )
                .conj();
            let avt = block_elem(a.lmix.block(a.lmix.idx(tstp, m)), a.size2, a1, a2);
            set_block_elem(
                c.lmix.block_mut(c.lmix.idx(tstp, m)),
                c.size2,
                c1,
                c2,
                i_c * avt * bvt21,
            );
        }
    }
}

/// Hermitian convenience of `bubble1`.
#[allow(clippy::too_many_arguments)]
pub fn bubble1_herm(
    tstp: i64,
    c: &mut ContourGreen,
    c1: usize,
    c2: usize,
    a: &ContourGreen,
    a1: usize,
    a2: usize,
    b: &ContourGreen,
    b1: usize,
    b2: usize,
) {
    bubble1(tstp, c, c1, c2, a, a, a1, a2, b, b, b1, b2);
}

/// Bubble2 diagram `C_{c1c2}(t,t') = i A_{a1a2}(t,t') B_{b1b2}(t,t')`
/// at time step `tstp`.
#[allow(clippy::too_many_arguments)]
pub fn bubble2(
    tstp: i64,
    c: &mut ContourGreen,
    c1: usize,
    c2: usize,
    a: &ContourGreen,
    acc: &ContourGreen,
    a1: usize,
    a2: usize,
    b: &ContourGreen,
    bcc: &ContourGreen,
    b1: usize,
    b2: usize,
) {
    check_bubble(tstp, c, a, acc, b, bcc);
    check_bubble_orbital(&[
        (c1, c.size1, "c1"),
        (c2, c.size2, "c2"),
        (a1, a.size1, "a1"),
        (a2, a.size2, "a2"),
        (b1, b.size1, "b1"),
        (b2, b.size2, "b2"),
    ]);
    let i_c = Complex::new(0.0, 1.0);
    if tstp == MATSUBARA {
        let ntau = c.ntau as i64;
        for m in 0..=ntau {
            let am = block_elem(a.mat.block(a.mat.check_tau(m)), a.size2, a1, a2);
            let bm = block_elem(b.mat.block(b.mat.check_tau(m)), b.size2, b1, b2);
            set_block_elem(
                c.mat.block_mut(c.mat.check_tau(m)),
                c.size2,
                c1,
                c2,
                -(am * bm),
            );
        }
    } else {
        let ntau = c.ntau as i64;
        for m in 0..=tstp {
            // the stored blocks of the pairs (tstp, m) and (m, tstp)
            let acc_les = acc.less.block(acc.less.idx(m, tstp));
            let bcc_les = bcc.less.block(bcc.less.idx(m, tstp));
            let a_les = a.less.block(a.less.idx(m, tstp));
            let b_les = b.less.block(b.less.idx(m, tstp));
            let a_ret = a.ret.block(a.ret.idx(tstp, m));
            let b_ret = b.ret.block(b.ret.idx(tstp, m));
            let accles21 = block_elem(acc_les, acc.size2, a2, a1);
            let bccles21 = block_elem(bcc_les, bcc.size2, b2, b1);
            let bgtr12_tt1 = block_elem(b_ret, b.size2, b1, b2) - bccles21.conj();
            let agtr12_tt1 = block_elem(a_ret, a.size2, a1, a2) - accles21.conj();
            let cgtr_tt1 = i_c * agtr12_tt1 * bgtr12_tt1;
            let cles_tt1 = i_c * accles21.conj() * bccles21.conj();
            set_block_elem(
                c.ret.block_mut(c.ret.idx(tstp, m)),
                c.size2,
                c1,
                c2,
                cgtr_tt1 - cles_tt1,
            );
            set_block_elem(
                c.less.block_mut(c.less.idx(m, tstp)),
                c.size2,
                c1,
                c2,
                i_c * block_elem(a_les, a.size2, a1, a2) * block_elem(b_les, b.size2, b1, b2),
            );
        }
        for m in 0..=ntau {
            let avt = block_elem(a.lmix.block(a.lmix.idx(tstp, m)), a.size2, a1, a2);
            let bvt = block_elem(b.lmix.block(b.lmix.idx(tstp, m)), b.size2, b1, b2);
            set_block_elem(
                c.lmix.block_mut(c.lmix.idx(tstp, m)),
                c.size2,
                c1,
                c2,
                i_c * avt * bvt,
            );
        }
    }
}

/// Hermitian convenience of `bubble2`.
#[allow(clippy::too_many_arguments)]
pub fn bubble2_herm(
    tstp: i64,
    c: &mut ContourGreen,
    c1: usize,
    c2: usize,
    a: &ContourGreen,
    a1: usize,
    a2: usize,
    b: &ContourGreen,
    b1: usize,
    b2: usize,
) {
    bubble2(tstp, c, c1, c2, a, a, a1, a2, b, b, b1, b2);
}

// ---- moving (truncated) bubbles -------------------------------------------

///
/// *Remarks* : *Moving Bubbles*
///
/// The moving bubbles evaluate the leading time step (`delti = 0`) of the
/// result, ``C(t, t - \delta_j)`` for ``\delta_j = 0, \cdots, t_c``, from the
/// leading time steps of `A`, `Acc`, `B`, and `Bcc`. Only the retarded and
/// lesser components exist. With ``A^{>} = A^{R} + A^{<}``,
///
/// Bubble1: ``C_{12}(t,t') = i A_{a_1a_2}(t,t') B_{b_2b_1}(t',t)``,
///
/// ```math
/// C^{>}(t,t') = i A^{>}(t,t') B^{<}(t',t),\quad
/// C^{<}(t,t') = i A^{<}(t,t') B^{>}(t',t),
/// ```
///
/// where ``B^{<}_{21}(t',t) = -B^{cc,<}_{12}(t,t')^{*}`` and
/// ``B^{>}_{21}(t',t) = B^{<}_{21}(t',t) - B^{cc,R}_{12}(t,t')^{*}``.
///
/// Bubble2: ``C_{12}(t,t') = i A_{a_1a_2}(t,t') B_{b_1b_2}(t,t')``,
///
/// ```math
/// C^{>}(t,t') = i A^{>}(t,t') B^{>}(t,t'),\quad
/// C^{<}(t,t') = i A^{<}(t,t') B^{<}(t,t').
/// ```
///
/// In both cases ``C^{R} = C^{>} - C^{<}`` for ``t \ge t'``. The
/// alternative convention takes the lesser elements of ``A^{>}`` and of
/// the intermediate ``C^{<}`` with the transposed orbital indices (`a21`,
/// `b21`), which is inconsistent with the stored lesser element
/// ``A^{<}_{12}(t,t')`` and with the full contour Bubble2. Here the
/// elements `(a1,a2)` and `(b1,b2)` are used, so that the moving Bubble2
/// agrees with the full contour Bubble2.
///
/// Moving Bubble1 diagram
/// `C_{c1 c2}(t, t') = i A_{a1 a2}(t, t') B_{b2 b1}(t', t)` for the
/// leading time step `tstp` of the windows.
#[allow(clippy::too_many_arguments)]
pub fn bubble1_moving(
    tstp: i64,
    c: &mut MovingGreen,
    c1: usize,
    c2: usize,
    a: &MovingGreen,
    acc: &MovingGreen,
    a1: usize,
    a2: usize,
    b: &MovingGreen,
    bcc: &MovingGreen,
    b1: usize,
    b2: usize,
) {
    check_moving_bubble(tstp, &[c, a, acc, b, bcc], (c1, c2, a1, a2, b1, b2));
    let ii = Complex::new(0.0, 1.0);
    for m in 0..=c.tc {
        let ales12 = a.less.mblock(0, m)[a1 * a.ret.size2 + a2];
        let agtr12 = ales12 + a.ret.mblock(0, m)[a1 * a.ret.size2 + a2];
        let bles21 = -bcc.less.mblock(0, m)[b1 * bcc.ret.size2 + b2].conj();
        let bgtr21 = bles21 - bcc.ret.mblock(0, m)[b1 * bcc.ret.size2 + b2].conj();
        let cgtr = ii * agtr12 * bles21;
        let cles = ii * ales12 * bgtr21;
        let cs2 = c.ret.size2;
        c.ret.mblock_mut(0, m)[c1 * cs2 + c2] = cgtr - cles;
        c.less.mblock_mut(0, m)[c1 * cs2 + c2] = cles;
    }
}

/// Hermitian version (`acc = a`, `bcc = b`) of the moving Bubble1.
#[allow(clippy::too_many_arguments)]
pub fn bubble1_moving_herm(
    tstp: i64,
    c: &mut MovingGreen,
    c1: usize,
    c2: usize,
    a: &MovingGreen,
    a1: usize,
    a2: usize,
    b: &MovingGreen,
    b1: usize,
    b2: usize,
) {
    bubble1_moving(tstp, c, c1, c2, a, a, a1, a2, b, b, b1, b2);
}

/// Moving Bubble2 diagram
/// `C_{c1 c2}(t, t') = i A_{a1 a2}(t, t') B_{b1 b2}(t, t')` for the
/// leading time step `tstp` of the windows. `acc` and `bcc` are not
/// needed by the formula, they are accepted for the symmetry with the full
/// contour API.
#[allow(clippy::too_many_arguments)]
pub fn bubble2_moving(
    tstp: i64,
    c: &mut MovingGreen,
    c1: usize,
    c2: usize,
    a: &MovingGreen,
    _acc: &MovingGreen,
    a1: usize,
    a2: usize,
    b: &MovingGreen,
    _bcc: &MovingGreen,
    b1: usize,
    b2: usize,
) {
    check_moving_bubble(tstp, &[c, a, _acc, b, _bcc], (c1, c2, a1, a2, b1, b2));
    let ii = Complex::new(0.0, 1.0);
    for m in 0..=c.tc {
        let ales12 = a.less.mblock(0, m)[a1 * a.ret.size2 + a2];
        let bles12 = b.less.mblock(0, m)[b1 * b.ret.size2 + b2];
        let agtr12 = a.ret.mblock(0, m)[a1 * a.ret.size2 + a2] + ales12;
        let bgtr12 = b.ret.mblock(0, m)[b1 * b.ret.size2 + b2] + bles12;
        let cgtr = ii * agtr12 * bgtr12;
        let cles = ii * ales12 * bles12;
        let cs2 = c.ret.size2;
        c.ret.mblock_mut(0, m)[c1 * cs2 + c2] = cgtr - cles;
        c.less.mblock_mut(0, m)[c1 * cs2 + c2] = cles;
    }
}

/// Hermitian version of the moving Bubble2.
#[allow(clippy::too_many_arguments)]
pub fn bubble2_moving_herm(
    tstp: i64,
    c: &mut MovingGreen,
    c1: usize,
    c2: usize,
    a: &MovingGreen,
    a1: usize,
    a2: usize,
    b: &MovingGreen,
    b1: usize,
    b2: usize,
) {
    bubble2_moving(tstp, c, c1, c2, a, a, a1, a2, b, b, b1, b2);
}

// ------------------------------------------------------------------
// steady-state (NESS2) bubbles
// ------------------------------------------------------------------

/// Common driver of `bubble1_steady` and `bubble2_steady`. The matrix
/// element `(c1, c2)` of `C` is written at
/// the times `0 <= t < Nft/2`, the lesser element `(c2, c1)` at the
/// negative times. The retarded element at the negative times and the
/// lesser elements at the Nyquist mode are set to zero. The orbital
/// indices are 0-based.
#[allow(clippy::too_many_arguments)]
fn bubble_steady(
    c: &mut SteadyGreen,
    c1: usize,
    c2: usize,
    a: &SteadyGreen,
    a1: usize,
    a2: usize,
    b: &SteadyGreen,
    b1: usize,
    b2: usize,
    kind: i32,
) {
    let i_c = Complex::new(0.0, 1.0);
    let nft = c.grid.nft;
    assert!(
        a.grid.nft == nft && b.grid.nft == nft,
        "A, B, and C must have the same number of grid points"
    );
    let dc = c.ret.size1;
    let da = a.ret.size1;
    let db = b.ret.size1;
    assert!(c1 < dc && c2 < dc, "orbital index of C out of range");
    assert!(a1 < da && a2 < da, "orbital index of A out of range");
    assert!(b1 < db && b2 < db, "orbital index of B out of range");

    for t in 0..(nft / 2) {
        let aret = a.ret.block(FftDomain::Time, t)[a1 * da + a2];
        let ales = a.les.block(FftDomain::Time, t)[a1 * da + a2];
        let bret = b.ret.block(FftDomain::Time, t)[b1 * db + b2];
        let bles = b.les.block(FftDomain::Time, t)[b1 * db + b2];
        let (ret, les) = if kind == 0 {
            let bgtr = bret + bles;
            let les = i_c * ales * (-bgtr.conj());
            let ret = i_c * aret * (-bles.conj()) + i_c * ales * bret.conj();
            (ret, les)
        } else {
            let les = i_c * ales * bles;
            let ret = i_c * aret * bret + i_c * ales * bret + i_c * aret * bles;
            (ret, les)
        };
        c.ret.block_mut(FftDomain::Time, t)[c1 * dc + c2] = ret;
        c.les.block_mut(FftDomain::Time, t)[c1 * dc + c2] = les;
        if t > 0 {
            c.les.block_mut(FftDomain::Time, nft - t)[c2 * dc + c1] = -les.conj();
        }
    }
    for n in (nft / 2)..nft {
        c.ret.block_mut(FftDomain::Time, n)[c1 * dc + c2] = Complex::ZERO;
    }
    c.les.block_mut(FftDomain::Time, nft / 2)[c1 * dc + c2] = Complex::ZERO;
    c.les.block_mut(FftDomain::Time, nft / 2)[c2 * dc + c1] = Complex::ZERO;
}

/// Steady-state particle-hole bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
/// B_{b_2b_1}(-t)``, for ``0 \\le t < N_{ft}/2`` in the time domain:
///
/// ```math
/// C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) [-B^>(t)^\\dagger]_{b_2b_1},\\quad
/// C^R_{c_1c_2}(t) = i A^R_{a_1a_2}(t) [-B^<(t)^\\dagger]_{b_2b_1}
///             + i A^<_{a_1a_2}(t) [B^R(t)^\\dagger]_{b_2b_1},
/// ```
///
/// with ``C^<_{c_2c_1}(-t) = -C^<_{c_1c_2}(t)^*``. The orbital indices are
/// 0-based.
#[allow(clippy::too_many_arguments)]
pub fn bubble1_steady(
    c: &mut SteadyGreen,
    c1: usize,
    c2: usize,
    a: &SteadyGreen,
    a1: usize,
    a2: usize,
    b: &SteadyGreen,
    b1: usize,
    b2: usize,
) {
    bubble_steady(c, c1, c2, a, a1, a2, b, b1, b2, 0);
}

/// Scalar version of `bubble1_steady`, all matrices are 1 x 1. The time
/// data of `C` is reset to zero first.
pub fn bubble1_steady_scalar(c: &mut SteadyGreen, a: &SteadyGreen, b: &SteadyGreen) {
    c.zeros_domain(FftDomain::Time);
    bubble_steady(c, 0, 0, a, 0, 0, b, 0, 0, 0);
}

/// Steady-state particle-particle bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
/// B_{b_1b_2}(t)``, for ``0 \\le t < N_{ft}/2`` in the time domain:
///
/// ```math
/// C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) B^<_{b_1b_2}(t),\\quad
/// C^R_{c_1c_2}(t) = i (A^R B^R + A^< B^R + A^R B^<),
/// ```
///
/// with ``C^<_{c_2c_1}(-t) = -C^<_{c_1c_2}(t)^*``. The orbital indices are
/// 0-based.
#[allow(clippy::too_many_arguments)]
pub fn bubble2_steady(
    c: &mut SteadyGreen,
    c1: usize,
    c2: usize,
    a: &SteadyGreen,
    a1: usize,
    a2: usize,
    b: &SteadyGreen,
    b1: usize,
    b2: usize,
) {
    bubble_steady(c, c1, c2, a, a1, a2, b, b1, b2, 1);
}

/// Scalar version of `bubble2_steady`.
pub fn bubble2_steady_scalar(c: &mut SteadyGreen, a: &SteadyGreen, b: &SteadyGreen) {
    c.zeros_domain(FftDomain::Time);
    bubble_steady(c, 0, 0, a, 0, 0, b, 0, 0, 1);
}

// Block element access. The bubble loops take each block once and index
// it directly: the orbital indices (0-based) have been validated by
// `check_bubble_orbital`, and aliasing of `C` with an input is rejected
// by `check_bubble`.

/// Element `(i, j)` of a row-major block with `cols` columns.
#[inline]
fn block_elem(block: &[Complex], cols: usize, i: usize, j: usize) -> Complex {
    block[i * cols + j]
}

/// Write element `(i, j)` of a row-major block with `cols` columns.
#[inline]
fn set_block_elem(block: &mut [Complex], cols: usize, i: usize, j: usize, v: Complex) {
    block[i * cols + j] = v;
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::DenseMat;
    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

    #[test]
    fn bubble1_free_mat() {
        let (nt, ntau) = (2i64, 10i64);
        let (beta, h) = (1.0, 0.1);
        let cc = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        let mut c = ContourGreen::new(&cc, Sign::Bose, Complex::ZERO);
        bubble1_herm(MATSUBARA, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        let ntau_r = c.ntau as i64;
        for m in 0..=ntau_r {
            // -sign_B = +1 (B fermionic)
            let want = g.get_mat(m, 0, 0) * g.get_mat(ntau_r - m, 0, 0);
            let got = c.get_mat(m, 0, 0);
            assert!((got - want).abs() < 1e-14, "m {m}: {got} vs {want}");
        }
    }

    #[test]
    fn bubble2_free_components() {
        // H = 0 free GF, bubble2 at (0,0): C^M = -A^M B^M etc.
        let (nt, ntau) = (4i64, 10i64);
        let (beta, h) = (1.0, 0.1);
        let cc = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        let mut c = ContourGreen::new(&cc, Sign::Bose, Complex::ZERO);
        bubble2_herm(MATSUBARA, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        // C^M = -G^M G^M = -(1/4) (real)
        for m in 0..=ntau {
            let want = -(g.get_mat(m, 0, 0) * g.get_mat(m, 0, 0));
            let got = c.get_mat(m, 0, 0);
            assert!((got - want).abs() < 1e-14, "m {m}: {got} vs {want}");
        }
        bubble2_herm(3, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        for m in 0..=3 {
            let want_les = Complex::new(0.0, 1.0) * g.get_less(m, 3, 0, 0) * g.get_less(m, 3, 0, 0);
            let got = c.get_less(m, 3, 0, 0);
            assert!((got - want_les).abs() < 1e-14);
            let want_lmix =
                Complex::new(0.0, 1.0) * g.get_lmix(3, m, 0, 0) * g.get_lmix(3, m, 0, 0);
            let got = c.get_lmix(3, m, 0, 0);
            assert!((got - want_lmix).abs() < 1e-14);
        }
    }
}
