//
// Project : KadanoffBaym
// File    : src/basics/blocks/elemcpy.rs
// Purpose : orbital sub-block copies driven by zones or index maps
//
// Blueprint: julia/src/blocks/elemcpy.jl.

use crate::Complex;
use crate::types::{ContourGreen, LessComp, LmixComp, MatComp, RetComp};

use super::copyzone::CopyZone;

#[inline]
fn copy_zone_pair(
    cz1: CopyZone,
    src: &[Complex],
    src_cols: usize,
    cz2: CopyZone,
    dst: &mut [Complex],
    dst_cols: usize,
) {
    for dx in 0..cz2.nrows() {
        for dy in 0..cz2.ncols() {
            let sx = (cz1.x1 + dx) as usize;
            let sy = (cz1.y1 + dy) as usize;
            let v = src[sx * src_cols + sy];
            let dxx = (cz2.x1 + dx) as usize;
            let dyy = (cz2.y1 + dy) as usize;
            dst[dxx * dst_cols + dyy] = v;
        }
    }
}

/// Copy the orbital sub-blocks selected by the zones between two
/// components, iterating the stored time points.
pub(crate) fn elemcpy_mat(cz1: CopyZone, src: &MatComp, cz2: CopyZone, dst: &mut MatComp) {
    assert!(cz1.is_valid() && cz2.is_valid(), "invalid copy zone");
    assert!(CopyZone::compatible(cz1, cz2), "incompatible copy zones");
    assert!(
        cz1.x2 < src.size1 as i64 && cz1.y2 < src.size2 as i64,
        "zone out of source bounds"
    );
    assert!(
        cz2.x2 < dst.size1 as i64 && cz2.y2 < dst.size2 as i64,
        "zone out of destination bounds"
    );
    for k in 0..src.nblocks() {
        let s = src.block(k);
        let dst_cols = dst.size2;
        let d = dst.block_mut(k);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// Copy the orbital sub-blocks selected by the zones between two
/// components, iterating the stored time points.
pub(crate) fn elemcpy_lmix(cz1: CopyZone, src: &LmixComp, cz2: CopyZone, dst: &mut LmixComp) {
    assert!(cz1.is_valid() && cz2.is_valid(), "invalid copy zone");
    assert!(CopyZone::compatible(cz1, cz2), "incompatible copy zones");
    assert!(
        cz1.x2 < src.size1 as i64 && cz1.y2 < src.size2 as i64,
        "zone out of source bounds"
    );
    assert!(
        cz2.x2 < dst.size1 as i64 && cz2.y2 < dst.size2 as i64,
        "zone out of destination bounds"
    );
    for k in 0..src.nblocks() {
        let s = src.block(k);
        let dst_cols = dst.size2;
        let d = dst.block_mut(k);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// ret-component copy restricted to the row `tstp` (columns `0..=tstp`).
pub(crate) fn elemcpy_ret_at(tstp: i64, cz1: CopyZone, src: &RetComp, cz2: CopyZone, dst: &mut RetComp) {
    assert!(cz1.is_valid() && cz2.is_valid() && CopyZone::compatible(cz1, cz2));
    assert!(
        cz1.x2 < src.size1 as i64 && cz1.y2 < src.size2 as i64,
        "zone out of source bounds"
    );
    assert!(
        cz2.x2 < dst.size1 as i64 && cz2.y2 < dst.size2 as i64,
        "zone out of destination bounds"
    );
    for j in 0..=tstp {
        let k = src.idx(tstp, j);
        let kd = dst.idx(tstp, j);
        let s = src.block(k as usize);
        let dst_cols = dst.size2;
        let d = dst.block_mut(kd as usize);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// less-component copy restricted to the row `tstp` (columns `0..=tstp`).
pub(crate) fn elemcpy_less_at(
    tstp: i64,
    cz1: CopyZone,
    src: &LessComp,
    cz2: CopyZone,
    dst: &mut LessComp,
) {
    assert!(cz1.is_valid() && cz2.is_valid() && CopyZone::compatible(cz1, cz2));
    assert!(
        cz1.x2 < src.size1 as i64 && cz1.y2 < src.size2 as i64,
        "zone out of source bounds"
    );
    assert!(
        cz2.x2 < dst.size1 as i64 && cz2.y2 < dst.size2 as i64,
        "zone out of destination bounds"
    );
    for i in 0..=tstp {
        let k = src.idx(i, tstp);
        let kd = dst.idx(i, tstp);
        let s = src.block(k as usize);
        let dst_cols = dst.size2;
        let d = dst.block_mut(kd as usize);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// Copy orbital zones between two `ContourGreen`s over all components
/// (blueprint `elemcpy!(cz1, src, cz2, dst)`).
pub(crate) fn elemcpy_green(cz1: CopyZone, src: &ContourGreen, cz2: CopyZone, dst: &mut ContourGreen) {
    assert_eq!((src.nt, src.ntau), (dst.nt, dst.ntau), "grid mismatch");
    assert!(CopyZone::compatible(cz1, cz2), "incompatible copy zones");
    elemcpy_mat(cz1, &src.mat, cz2, &mut dst.mat);
    for tstp in 0..=src.nt as i64 {
        elemcpy_ret_at(tstp, cz1, &src.ret, cz2, &mut dst.ret);
        elemcpy_lmix(cz1, &src.lmix, cz2, &mut dst.lmix);
        elemcpy_less_at(tstp, cz1, &src.less, cz2, &mut dst.less);
    }
}

/// Copy orbital zones at one time step (blueprint `elemcpy!(tstp, ...)`);
/// `tstp = MATSUBARA` copies the Matsubara component only, otherwise the
/// ret/lmix/less rows of `tstp`.
pub fn elemcpy_green_at(
    tstp: i64,
    cz1: CopyZone,
    src: &ContourGreen,
    cz2: CopyZone,
    dst: &mut ContourGreen,
) {
    assert_eq!((src.nt, src.ntau), (dst.nt, dst.ntau), "grid mismatch");
    assert!(CopyZone::compatible(cz1, cz2), "incompatible copy zones");
    if tstp == crate::types::MATSUBARA {
        elemcpy_mat(cz1, &src.mat, cz2, &mut dst.mat);
    } else {
        assert!((0..=src.nt as i64).contains(&tstp));
        elemcpy_ret_at(tstp, cz1, &src.ret, cz2, &mut dst.ret);
        elemcpy_lmix(cz1, &src.lmix, cz2, &mut dst.lmix);
        elemcpy_less_at(tstp, cz1, &src.less, cz2, &mut dst.less);
    }
}

/// Copy by orbital index maps
/// `dst[rows2[k], cols2[k]] = src[rows1[k], cols1[k]]` over all
/// components (blueprint `elemcpy!(rows1, cols1, src, rows2, cols2, dst)`).
pub fn elemcpy_green_map(
    rows1: &[i64],
    cols1: &[i64],
    src: &ContourGreen,
    rows2: &[i64],
    cols2: &[i64],
    dst: &mut ContourGreen,
) {
    assert_eq!(
        (rows1.len(), cols1.len(), rows2.len(), cols2.len()),
        (rows2.len(), cols2.len(), rows1.len(), cols1.len()),
        "index map length mismatch"
    );
    for k in 0..rows1.len() {
        let cz1 = CopyZone::point(rows1[k], cols1[k]);
        let cz2 = CopyZone::point(rows2[k], cols2[k]);
        elemcpy_green(cz1, src, cz2, dst);
    }
}

/// Copy by orbital index maps at one time step (`MATSUBARA` = Matsubara).
pub fn elemcpy_green_map_at(
    tstp: i64,
    rows1: &[i64],
    cols1: &[i64],
    src: &ContourGreen,
    rows2: &[i64],
    cols2: &[i64],
    dst: &mut ContourGreen,
) {
    assert_eq!(
        (rows1.len(), cols1.len()),
        (rows2.len(), cols2.len()),
        "index map length mismatch"
    );
    for k in 0..rows1.len() {
        let cz1 = CopyZone::point(rows1[k], cols1[k]);
        let cz2 = CopyZone::point(rows2[k], cols2[k]);
        elemcpy_green_at(tstp, cz1, src, cz2, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MATSUBARA;
    use super::super::testutil::two_greens;

    #[test]
    fn zone_copy_between_greens() {
        let (src, mut dst) = two_greens();
        let cz = CopyZone::point(0, 0);
        elemcpy_green(cz, &src, cz, &mut dst);
        assert_eq!(dst.get_mat(1, 0, 0), Complex::new(101.0, 0.0));
        assert_eq!(dst.get_ret(2, 1, 0, 0), Complex::new(21.0, 0.5));
        assert_eq!(dst.get_less(1, 2, 0, 0), Complex::new(21.0, 0.5));
        assert_eq!(dst.get_lmix(1, 2, 0, 0), Complex::new(1012.0, 0.0));
        // untouched element stays zero
        assert_eq!(dst.get_mat(1, 1, 1), Complex::new(0.0, 0.0));
    }

    #[test]
    fn zone_copy_at_timestep() {
        let (src, mut dst) = two_greens();
        let cz = CopyZone::point(1, 1);
        elemcpy_green_at(2, cz, &src, cz, &mut dst);
        // only the ret row 2 and less row 2 were copied (and lmix row 2)
        assert_eq!(dst.get_mat(1, 1, 1), Complex::ZERO);
        // ret(2,j) blocks' (1,1) element: fill only set (0,0); stays zero
        assert_eq!(dst.get_ret(2, 0, 1, 1), Complex::ZERO);
        // Matsubara-only copy
        let mut dst2 = dst.clone();
        elemcpy_green_at(MATSUBARA, cz, &src, cz, &mut dst2);
        assert_eq!(dst2.get_mat(2, 1, 1), Complex::new(103.0, 0.0));
    }

    #[test]
    fn index_map_copy() {
        let (src, mut dst) = two_greens();
        elemcpy_green_map(&[0, 1], &[0, 1], &src, &[1, 0], &[1, 0], &mut dst);
        // dst(1,1) = src(0,0), dst(0,0) = src(1,1)
        assert_eq!(dst.get_mat(2, 1, 1), Complex::new(102.0, 0.0));
        assert_eq!(dst.get_mat(2, 0, 0), Complex::new(103.0, 0.0)); // src(1,1) at n=2
    }
}
