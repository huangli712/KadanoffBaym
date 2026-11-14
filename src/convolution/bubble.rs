// Project : KadanoffBaym
// File    : src/convolution/bubble.rs
// Purpose : bubble diagrams (blueprint: convolution/bubble.jl)

use crate::complex::c64::C64;
use crate::types::green::ContourGreen;
use crate::types::MATSUBARA;

#[allow(clippy::too_many_arguments)]
fn check_bubble(
    tstp: i64,
    c: &ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    b: &ContourGreen,
    bcc: &ContourGreen,
    c1: usize,
    c2: usize,
    a1: usize,
    a2: usize,
    b1: usize,
    b2: usize,
) {
    assert!(
        !std::ptr::eq(c, a)
            && !std::ptr::eq(c, acc)
            && !std::ptr::eq(c, b)
            && !std::ptr::eq(c, bcc)
    );
    assert_eq!(a.ntau, c.ntau, "imaginary time grids are not compatible");
    assert!(tstp == MATSUBARA || (0..=c.nt as i64).contains(&tstp));
    assert!(acc.size1 == a.size1 && acc.size2 == a.size2);
    assert!(bcc.size1 == b.size1 && bcc.size2 == b.size2);
    assert!(acc.sign == a.sign);
    assert!(bcc.sign == b.sign);
    assert_eq!(c.sign.as_i64(), a.sign.as_i64() * b.sign.as_i64());
    for (x, d) in [
        (c1, c.size1),
        (c2, c.size1),
        (a1, a.size1),
        (a2, a.size1),
        (b1, b.size1),
        (b2, b.size1),
    ] {
        assert!(
            x < d.min(c.size2.max(a.size2.max(b.size2))),
            "orbital index out of bounds"
        );
    }
}

/// Bubble1 diagram `C_{c1c2}(t,t') = i A_{a1a2}(t,t') B_{b2b1}(t',t)`
/// at time step `tstp` (blueprint `bubble1!`). All indices are
/// 0-based. Only the ret row, less column, and lmix row at `tstp` are
/// written (mat only at `MATSUBARA`).
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
    check_bubble(tstp, c, a, acc, b, bcc, c1, c2, a1, a2, b1, b2);
    let i_c = C64::new(0.0, 1.0);
    if tstp == MATSUBARA {
        let ntau = c.ntau as i64;
        let sig = -(b.sign.as_i64() as f64);
        for m in 0..=ntau {
            let v = sig * a.get_mat(m, a1, a2) * b.get_mat(ntau - m, b2, b1);
            c.set_mat(m, c1, c2, v);
        }
    } else {
        let ntau = c.ntau as i64;
        let msigb = -(b.sign.as_i64() as f64);
        for m in 0..=tstp {
            let bles21_tt1 = -bcc.get_less(m, tstp, b1, b2).conj();
            let bgtr21_tt1 = b.get_ret(tstp, m, b2, b1) + bles21_tt1;
            let bles21_t1t = b.get_less(m, tstp, b2, b1);
            let bgtr21_t1t = -bcc.get_ret(tstp, m, b1, b2).conj() + bles21_t1t;
            let ales12_tt1 = -acc.get_less(m, tstp, a2, a1).conj();
            let agtr12_tt1 = a.get_ret(tstp, m, a1, a2) + ales12_tt1;
            let cgtr_tt1 = i_c * agtr12_tt1 * bles21_t1t;
            let cles_tt1 = i_c * ales12_tt1 * bgtr21_t1t;
            c.set_ret(tstp, m, c1, c2, cgtr_tt1 - cles_tt1);
            c.set_less(
                m,
                tstp,
                c1,
                c2,
                i_c * a.get_less(m, tstp, a1, a2) * bgtr21_tt1,
            );
        }
        for m in 0..=ntau {
            let bvt21 = msigb * bcc.get_lmix(tstp, ntau - m, b1, b2).conj();
            c.set_lmix(tstp, m, c1, c2, i_c * a.get_lmix(tstp, m, a1, a2) * bvt21);
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
/// at time step `tstp` (blueprint `bubble2!`).
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
    check_bubble(tstp, c, a, acc, b, bcc, c1, c2, a1, a2, b1, b2);
    let i_c = C64::new(0.0, 1.0);
    if tstp == MATSUBARA {
        let ntau = c.ntau as i64;
        for m in 0..=ntau {
            let v = -(a.get_mat(m, a1, a2) * b.get_mat(m, b1, b2));
            c.set_mat(m, c1, c2, v);
        }
    } else {
        let ntau = c.ntau as i64;
        for m in 0..=tstp {
            let accles21 = acc.get_less(m, tstp, a2, a1);
            let bccles21 = bcc.get_less(m, tstp, b2, b1);
            let bgtr12_tt1 = b.get_ret(tstp, m, b1, b2) - bccles21.conj();
            let agtr12_tt1 = a.get_ret(tstp, m, a1, a2) - accles21.conj();
            let cgtr_tt1 = i_c * agtr12_tt1 * bgtr12_tt1;
            let cles_tt1 = i_c * accles21.conj() * bccles21.conj();
            c.set_ret(tstp, m, c1, c2, cgtr_tt1 - cles_tt1);
            c.set_less(
                m,
                tstp,
                c1,
                c2,
                i_c * a.get_less(m, tstp, a1, a2) * b.get_less(m, tstp, b1, b2),
            );
        }
        for m in 0..=ntau {
            c.set_lmix(
                tstp,
                m,
                c1,
                c2,
                i_c * a.get_lmix(tstp, m, a1, a2) * b.get_lmix(tstp, m, b1, b2),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equilibrium::from_ham;
    use crate::linalg::mat::DenseMat;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn bubble2_free_components() {
        // H = 0 free GF, bubble2 at (0,0): C^M = -A^M B^M etc.
        let (nt, ntau) = (4i64, 10i64);
        let (beta, h) = (1.0, 0.1);
        let cc = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        let mut c = ContourGreen::new(&cc, Sign::Bose, C64::ZERO);
        bubble2_herm(MATSUBARA, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        // C^M = -G^M G^M = -(1/4) (real)
        for m in 0..=ntau as i64 {
            let want = -(g.get_mat(m, 0, 0) * g.get_mat(m, 0, 0));
            let got = c.get_mat(m, 0, 0);
            assert!((got - want).abs() < 1e-14, "m {m}: {got} vs {want}");
        }
        bubble2_herm(3, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        for m in 0..=3 {
            let want_les = C64::new(0.0, 1.0) * g.get_less(m, 3, 0, 0) * g.get_less(m, 3, 0, 0);
            let got = c.get_less(m, 3, 0, 0);
            assert!((got - want_les).abs() < 1e-14);
            let want_lmix = C64::new(0.0, 1.0) * g.get_lmix(3, m, 0, 0) * g.get_lmix(3, m, 0, 0);
            let got = c.get_lmix(3, m, 0, 0);
            assert!((got - want_lmix).abs() < 1e-14);
        }
    }

    #[test]
    fn bubble1_free_mat() {
        let (nt, ntau) = (2i64, 10i64);
        let (beta, h) = (1.0, 0.1);
        let cc = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        let mut c = ContourGreen::new(&cc, Sign::Bose, C64::ZERO);
        bubble1_herm(MATSUBARA, &mut c, 0, 0, &g, 0, 0, &g, 0, 0);
        let ntau_r = c.ntau as i64;
        for m in 0..=ntau_r {
            // -sign_B = +1 (B fermionic)
            let want = g.get_mat(m, 0, 0) * g.get_mat(ntau_r - m, 0, 0);
            let got = c.get_mat(m, 0, 0);
            assert!((got - want).abs() < 1e-14, "m {m}: {got} vs {want}");
        }
    }
}

// ---- moving (truncated) bubbles -------------------------------------------
//
// Blueprint julia/src/convolution/bubble.jl, the `ℱᵐᵒᵛ` overloads. The
// orbital indices are 0-based. The alias checks of the blueprint are
// subsumed by the Rust borrow checker.

use crate::types::moving::MovingGreen;

/// Sanity check of the moving bubble diagrams (blueprint
/// `_check_bubble_moving`).
#[allow(clippy::too_many_arguments)]
fn check_bubble_moving(
    tstp: i64,
    objs: &[&MovingGreen],
    idx: (usize, usize, usize, usize, usize, usize),
) {
    for x in objs {
        assert!(x.initialized, "the moving object is not initialized");
        assert!(x.tc == objs[0].tc, "the cut-offs differ");
        assert!(
            x.tstp == tstp,
            "tstp = {tstp} is not the leading time step {}",
            x.tstp
        );
    }
    let c = objs[0];
    let (a, acc, b, bcc) = (objs[1], objs[2], objs[3], objs[4]);
    assert!(
        acc.ret.size1 == a.ret.size1 && acc.ret.size2 == a.ret.size2,
        "A and Acc have different sizes"
    );
    assert!(
        bcc.ret.size1 == b.ret.size1 && bcc.ret.size2 == b.ret.size2,
        "B and Bcc have different sizes"
    );
    assert!(
        c.sign.as_i64() == a.sign.as_i64() * b.sign.as_i64(),
        "the statistics of C must be sign(A) * sign(B)"
    );
    let (c1, c2, a1, a2, b1, b2) = idx;
    for (x, i) in [(c, c1), (c, c2), (a, a1), (a, a2), (b, b1), (b, b2)] {
        assert!(i < x.ret.size1, "orbital index {i} out of range");
    }
}

/// Moving Bubble1 diagram
/// `C_{c1 c2}(t, t') = i A_{a1 a2}(t, t') B_{b2 b1}(t', t)` for the
/// leading time step `tstp` of the windows (blueprint `bubble1!` on
/// `ℱᵐᵒᵛ`, the element version).
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
    check_bubble_moving(tstp, &[c, a, acc, b, bcc], (c1, c2, a1, a2, b1, b2));
    let ii = C64::new(0.0, 1.0);
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
/// leading time step `tstp` of the windows (blueprint `bubble2!` on
/// `ℱᵐᵒᵛ`, the element version). `acc` and `bcc` are not needed by
/// the formula, they are accepted for the symmetry with the full
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
    check_bubble_moving(tstp, &[c, a, _acc, b, _bcc], (c1, c2, a1, a2, b1, b2));
    let ii = C64::new(0.0, 1.0);
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

use crate::types::ness::{FftDomain, NessHermMat};

/// Common driver of `bubble1_ness` and `bubble2_ness` (blueprint
/// `_bubble_ness!`). The matrix element `(c1, c2)` of `C` is written at
/// the times `0 <= t < Nft/2`, the lesser element `(c2, c1)` at the
/// negative times. The retarded element at the negative times and the
/// lesser elements at the Nyquist mode are set to zero. The orbital
/// indices are 0-based.
#[allow(clippy::too_many_arguments)]
fn bubble_ness(
    c: &mut NessHermMat,
    c1: usize,
    c2: usize,
    a: &NessHermMat,
    a1: usize,
    a2: usize,
    b: &NessHermMat,
    b1: usize,
    b2: usize,
    kind: i32,
) {
    let i_c = C64::new(0.0, 1.0);
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
        c.ret.block_mut(FftDomain::Time, n)[c1 * dc + c2] = C64::ZERO;
    }
    c.les.block_mut(FftDomain::Time, nft / 2)[c1 * dc + c2] = C64::ZERO;
    c.les.block_mut(FftDomain::Time, nft / 2)[c2 * dc + c1] = C64::ZERO;
}

/// Steady-state particle-hole bubble `C_{c1c2}(t) = i A_{a1a2}(t)
/// B_{b2b1}(-t)` (blueprint `bubble1_ness!`), for
/// `0 <= t < Nft/2` in the time domain. The orbital indices are
/// 0-based.
#[allow(clippy::too_many_arguments)]
pub fn bubble1_ness(
    c: &mut NessHermMat,
    c1: usize,
    c2: usize,
    a: &NessHermMat,
    a1: usize,
    a2: usize,
    b: &NessHermMat,
    b1: usize,
    b2: usize,
) {
    bubble_ness(c, c1, c2, a, a1, a2, b, b1, b2, 0);
}

/// Scalar version of `bubble1_ness`, all matrices are 1 x 1. The time
/// data of `C` is reset to zero first (blueprint
/// `bubble1_ness!(C, A, B)`).
pub fn bubble1_ness_scalar(c: &mut NessHermMat, a: &NessHermMat, b: &NessHermMat) {
    c.zeros_domain(FftDomain::Time);
    bubble_ness(c, 0, 0, a, 0, 0, b, 0, 0, 0);
}

/// Steady-state particle-particle bubble `C_{c1c2}(t) = i A_{a1a2}(t)
/// B_{b1b2}(t)` (blueprint `bubble2_ness!`). The orbital indices are
/// 0-based.
#[allow(clippy::too_many_arguments)]
pub fn bubble2_ness(
    c: &mut NessHermMat,
    c1: usize,
    c2: usize,
    a: &NessHermMat,
    a1: usize,
    a2: usize,
    b: &NessHermMat,
    b1: usize,
    b2: usize,
) {
    bubble_ness(c, c1, c2, a, a1, a2, b, b1, b2, 1);
}

/// Scalar version of `bubble2_ness` (blueprint
/// `bubble2_ness!(C, A, B)`).
pub fn bubble2_ness_scalar(c: &mut NessHermMat, a: &NessHermMat, b: &NessHermMat) {
    c.zeros_domain(FftDomain::Time);
    bubble_ness(c, 0, 0, a, 0, 0, b, 0, 0, 1);
}
