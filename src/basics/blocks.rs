// Project : KadanoffBaym
// File    : src/basics/blocks.rs
// Purpose : orbital sub-block copy operations
//
// Blueprint: julia/src/blocks/{copyzone,elemcpy,submatrix}.jl. The zone
// coordinates are ORBITAL indices (0-based here, 1-based in Julia).
// set_matrixelement (blocks/element.jl) is NESS-only and lands in R7.

use crate::complex::c64::C64;
use crate::types::green::{ContourGreen, LessComp, LmixComp, MatComp, RetComp};

/// Rectangular orbital region `[x1..=x2, y1..=y2]` for copying
/// (blueprint `CopyZone`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyZone {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
}

impl CopyZone {
    /// Single-cell zone.
    pub fn point(x: i64, y: i64) -> CopyZone {
        CopyZone {
            x1: x,
            y1: y,
            x2: x,
            y2: y,
        }
    }

    /// Square zone `[x..x+delta-1, y..y+delta-1]` (`delta >= 1`).
    pub fn square(x: i64, y: i64, delta: i64) -> CopyZone {
        assert!(delta >= 1, "CopyZone::square: delta >= 1, got {delta}");
        CopyZone {
            x1: x,
            y1: y,
            x2: x + delta - 1,
            y2: y + delta - 1,
        }
    }

    /// Validity: `x2 >= x1 >= 0` and `y2 >= y1 >= 0`.
    pub fn is_valid(&self) -> bool {
        self.x2 >= self.x1 && self.x1 >= 0 && self.y2 >= self.y1 && self.y1 >= 0
    }

    fn nrows(&self) -> i64 {
        self.x2 - self.x1 + 1
    }

    fn ncols(&self) -> i64 {
        self.y2 - self.y1 + 1
    }

    /// Two zones are compatible when their shapes match.
    pub fn compatible(a: CopyZone, b: CopyZone) -> bool {
        a.nrows() == b.nrows() && a.ncols() == b.ncols()
    }
}

macro_rules! zone_copy {
    ($fn_name:ident, $comp:ty, $iter:expr) => {
        /// Copy the orbital sub-blocks selected by the zones between two
        /// components, iterating the stored time points.
        pub fn $fn_name(cz1: CopyZone, src: &$comp, cz2: CopyZone, dst: &mut $comp) {
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
            for k in $iter(src) {
                let s = src.block(k);
                let dst_cols = dst.size2;
                let d = dst.block_mut(k);
                copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
            }
        }
    };
}

#[inline]
fn copy_zone_pair(
    cz1: CopyZone,
    src: &[C64],
    src_cols: usize,
    cz2: CopyZone,
    dst: &mut [C64],
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

zone_copy!(elemcpy_mat, MatComp, |s: &MatComp| 0..s.nblocks());
zone_copy!(elemcpy_lmix, LmixComp, |s: &LmixComp| 0..s.nblocks());

/// ret-component copy restricted to the row `tstp` (columns `0..=tstp`).
pub fn elemcpy_ret_at(tstp: i64, cz1: CopyZone, src: &RetComp, cz2: CopyZone, dst: &mut RetComp) {
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
pub fn elemcpy_less_at(
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
pub fn elemcpy_green(cz1: CopyZone, src: &ContourGreen, cz2: CopyZone, dst: &mut ContourGreen) {
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

/// Build a new `ContourGreen` containing the selected orbital rows and
/// columns of `G` (blueprint `submatrix`): `out[k, l] = G[rows[k], cols[l]]`.
pub fn submatrix(g: &ContourGreen, rows: &[i64], cols: &[i64]) -> ContourGreen {
    let c = crate::types::Contour::new(
        g.nt as i64,
        g.ntau as i64,
        rows.len() as i64,
        cols.len() as i64,
        1.0,
        1.0,
    );
    let mut out = ContourGreen::new(&c, g.sign, C64::ZERO);
    let mut rows1 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols1 = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &cl in cols {
            rows1.push(r);
            cols1.push(cl);
        }
    }
    let mut rows2 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols2 = Vec::with_capacity(rows.len() * cols.len());
    for r in 0..rows.len() as i64 {
        for cl in 0..cols.len() as i64 {
            rows2.push(r);
            cols2.push(cl);
        }
    }
    elemcpy_green_map(&rows1, &cols1, g, &rows2, &cols2, &mut out);
    out
}

/// Write the selected orbital rows and columns of `X` into `G`
/// (blueprint `set_submatrix!`; `dst[rows[k], cols[l]] = src[k, l]`).
pub fn set_submatrix(g: &mut ContourGreen, rows: &[i64], cols: &[i64], x: &ContourGreen) {
    assert_eq!(x.size1, rows.len(), "row count mismatch");
    assert_eq!(x.size2, cols.len(), "col count mismatch");
    // dst index maps enumerate the cross product rows x cols
    let mut rows2 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols2 = Vec::with_capacity(rows.len() * cols.len());
    let mut rows1 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols1 = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &c in cols {
            rows2.push(r);
            cols2.push(c);
        }
    }
    for r in 0..rows.len() as i64 {
        for c in 0..cols.len() as i64 {
            rows1.push(r);
            cols1.push(c);
        }
    }
    elemcpy_green_map(&rows1, &cols1, x, &rows2, &cols2, g);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};

    fn two_greens() -> (ContourGreen, ContourGreen) {
        let c = Contour::new(2, 2, 2, 2, 1.0, 1.0);
        let mut src = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        // deterministic fill: value = component tag + 10*i + j etc.
        for n in 0..=2 {
            src.set_mat(n, 0, 0, C64::new(n as f64 + 100.0, 0.0));
            src.set_mat(n, 1, 1, C64::new(n as f64 + 101.0, 0.0));
        }
        for tstp in 0..=2 {
            for j in 0..=tstp {
                let v = C64::new(10.0 * tstp as f64 + j as f64, 0.5);
                src.set_ret(tstp, j, 0, 0, v);
                src.set_less(j, tstp, 0, 0, v);
            }
            for n in 0..=2 {
                src.set_lmix(
                    tstp,
                    n,
                    0,
                    0,
                    C64::new(1000.0 + 10.0 * tstp as f64 + n as f64, 0.0),
                );
            }
        }
        let dst = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        (src, dst)
    }

    #[test]
    fn zone_validity_and_shapes() {
        assert!(CopyZone::point(0, 0).is_valid());
        assert!(CopyZone::square(1, 2, 3).is_valid());
        assert_eq!(CopyZone::square(1, 2, 3).x2, 3);
        assert!(!CopyZone::point(-1, 0).is_valid());
        assert!(CopyZone::compatible(
            CopyZone::square(0, 0, 2),
            CopyZone::square(1, 1, 2)
        ));
        assert!(!CopyZone::compatible(
            CopyZone::square(0, 0, 2),
            CopyZone::square(1, 1, 3)
        ));
    }

    #[test]
    fn zone_copy_between_greens() {
        let (src, mut dst) = two_greens();
        let cz = CopyZone::point(0, 0);
        elemcpy_green(cz, &src, cz, &mut dst);
        assert_eq!(dst.get_mat(1, 0, 0), C64::new(101.0, 0.0));
        assert_eq!(dst.get_ret(2, 1, 0, 0), C64::new(21.0, 0.5));
        assert_eq!(dst.get_less(1, 2, 0, 0), C64::new(21.0, 0.5));
        assert_eq!(dst.get_lmix(1, 2, 0, 0), C64::new(1012.0, 0.0));
        // untouched element stays zero
        assert_eq!(dst.get_mat(1, 1, 1), C64::new(0.0, 0.0));
    }

    #[test]
    fn zone_copy_at_timestep() {
        let (src, mut dst) = two_greens();
        let cz = CopyZone::point(1, 1);
        elemcpy_green_at(2, cz, &src, cz, &mut dst);
        // only the ret row 2 and less row 2 were copied (and lmix row 2)
        assert_eq!(dst.get_mat(1, 1, 1), C64::ZERO);
        // ret(2,j) blocks' (1,1) element: fill only set (0,0); stays zero
        assert_eq!(dst.get_ret(2, 0, 1, 1), C64::ZERO);
        // Matsubara-only copy
        let mut dst2 = dst.clone();
        elemcpy_green_at(crate::types::MATSUBARA, cz, &src, cz, &mut dst2);
        assert_eq!(dst2.get_mat(2, 1, 1), C64::new(103.0, 0.0));
    }

    #[test]
    fn index_map_copy() {
        let (src, mut dst) = two_greens();
        elemcpy_green_map(&[0, 1], &[0, 1], &src, &[1, 0], &[1, 0], &mut dst);
        // dst(1,1) = src(0,0), dst(0,0) = src(1,1)
        assert_eq!(dst.get_mat(2, 1, 1), C64::new(102.0, 0.0));
        assert_eq!(dst.get_mat(2, 0, 0), C64::new(103.0, 0.0)); // src(1,1) at n=2
    }

    #[test]
    fn submatrix_selection() {
        let (src, _) = two_greens();
        // additionally fill (1,0) so the selection is visible
        let mut src = src;
        for n in 0..=2 {
            src.set_mat(n, 1, 0, C64::new(-(n as f64), 0.0));
        }
        let sub = submatrix(&src, &[1, 0], &[0]);
        assert_eq!((sub.size1, sub.size2), (2, 1));
        // row order follows `rows`: row 0 of sub = orbital 1, row 1 = orbital 0
        assert_eq!(sub.get_mat(2, 0, 0), C64::new(-2.0, 0.0));
        assert_eq!(sub.get_mat(2, 1, 0), C64::new(102.0, 0.0));
        // set_submatrix back into a bigger object
        let mut back =
            ContourGreen::new(&Contour::new(2, 2, 2, 2, 1.0, 1.0), Sign::Fermi, C64::ZERO);
        set_submatrix(&mut back, &[1, 0], &[0], &sub);
        assert_eq!(back.get_mat(2, 1, 0), C64::new(-2.0, 0.0));
        assert_eq!(back.get_mat(2, 0, 0), C64::new(102.0, 0.0));
    }
}
