//
// Project : KadanoffBaym
// File    : src/storage/elemcpy.rs
// Purpose : orbital sub-block copies driven by zones or index maps
//

use super::copyzone::CopyZone;
use crate::complex::Complex;
use crate::types::{ContourGreen, LessComp, LmixComp, MatComp, RetComp};

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
fn elemcpy_mat(cz1: CopyZone, src: &MatComp, cz2: CopyZone, dst: &mut MatComp) {
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
fn elemcpy_lmix(cz1: CopyZone, src: &LmixComp, cz2: CopyZone, dst: &mut LmixComp) {
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
fn elemcpy_ret_at(tstp: i64, cz1: CopyZone, src: &RetComp, cz2: CopyZone, dst: &mut RetComp) {
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
        let s = src.block(k);
        let dst_cols = dst.size2;
        let d = dst.block_mut(kd);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// less-component copy restricted to the row `tstp` (columns `0..=tstp`).
fn elemcpy_less_at(tstp: i64, cz1: CopyZone, src: &LessComp, cz2: CopyZone, dst: &mut LessComp) {
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
        let s = src.block(k);
        let dst_cols = dst.size2;
        let d = dst.block_mut(kd);
        copy_zone_pair(cz1, s, src.size2, cz2, d, dst_cols);
    }
}

/// Copy orbital zones between two `ContourGreen`s over all components.
fn elemcpy_green(cz1: CopyZone, src: &ContourGreen, cz2: CopyZone, dst: &mut ContourGreen) {
    assert_eq!((src.nt, src.ntau), (dst.nt, dst.ntau), "grid mismatch");
    assert!(CopyZone::compatible(cz1, cz2), "incompatible copy zones");
    elemcpy_mat(cz1, &src.mat, cz2, &mut dst.mat);
    for tstp in 0..=src.nt as i64 {
        elemcpy_ret_at(tstp, cz1, &src.ret, cz2, &mut dst.ret);
        elemcpy_lmix(cz1, &src.lmix, cz2, &mut dst.lmix);
        elemcpy_less_at(tstp, cz1, &src.less, cz2, &mut dst.less);
    }
}

/// Copy orbital zones at one time step;
/// `tstp = MATSUBARA` copies the Matsubara component only, otherwise the
/// ret/lmix/less rows of `tstp`.
#[cfg(test)]
pub(super) fn elemcpy_green_at(
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
/// `dst[rows2[k], cols2[k]] = src[rows1[k], cols1[k]]` over all components.
pub(super) fn elemcpy_green_map(
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
#[cfg(test)]
pub(super) fn elemcpy_green_map_at(
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

    use crate::types::{Contour, Sign};

    use crate::types::MATSUBARA;

    struct Rng(u64);

    impl Rng {
        fn c64(&mut self) -> Complex {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let f = || ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            let re = f();
            let im = f();
            Complex::new(re, im)
        }

        fn block(&mut self, nel: usize) -> Vec<Complex> {
            (0..nel).map(|_| self.c64()).collect()
        }
    }

    fn two_greens() -> (ContourGreen, ContourGreen) {
        let c = Contour::new(2, 2, 2, 2, 1.0, 1.0);
        let mut src = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // deterministic fill: value = component tag + 10*i + j etc.
        for n in 0..=2 {
            src.set_mat(n, 0, 0, Complex::new(n as f64 + 100.0, 0.0));
            src.set_mat(n, 1, 1, Complex::new(n as f64 + 101.0, 0.0));
        }
        for tstp in 0..=2 {
            for j in 0..=tstp {
                let v = Complex::new(10.0 * tstp as f64 + j as f64, 0.5);
                src.set_ret(tstp, j, 0, 0, v);
                src.set_less(j, tstp, 0, 0, v);
            }
            for n in 0..=2 {
                src.set_lmix(
                    tstp,
                    n,
                    0,
                    0,
                    Complex::new(1000.0 + 10.0 * tstp as f64 + n as f64, 0.0),
                );
            }
        }
        let dst = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        (src, dst)
    }

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

    #[test]
    fn elemcpy_with_index_maps() {
        // 3x3 source, 2x2 destination
        let cs = Contour::new(5, 4, 3, 3, 0.5, 1.0);
        let cd = Contour::new(5, 4, 2, 2, 0.5, 1.0);
        let mut src = ContourGreen::new(&cs, Sign::Fermi, Complex::ZERO);
        let mut rng = Rng(0xC0FFEE);
        for m in 0..=4 {
            src.mat_raw_block_mut(m).copy_from_slice(&rng.block(9));
        }
        for i in 0..=5 {
            for j in 0..=i {
                src.ret_raw_block_mut(i, j).copy_from_slice(&rng.block(9));
            }
            for m in 0..=4 {
                src.lmix_raw_block_mut(i, m).copy_from_slice(&rng.block(9));
            }
            for j in i..=5 {
                src.less_raw_block_mut(i, j).copy_from_slice(&rng.block(9));
            }
        }
        let sentinel = Complex::new(7.0, -7.0);
        // non-rectangular map (1-based -> 0-based)
        let r1 = [2i64, 0, 1];
        let c1 = [0i64, 2, 1];
        let r2 = [0i64, 1, 1];
        let c2 = [0i64, 0, 1];
        let mut dst = ContourGreen::new(&cd, Sign::Fermi, sentinel);
        elemcpy_green_map(&r1, &c1, &src, &r2, &c2, &mut dst);
        for m in 0..=4 {
            for k in 0..3 {
                let s = src.mat_raw_block(m)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.mat_raw_block(m)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s, "mat {m} pair {k}");
            }
            assert_eq!(dst.mat_raw_block(m)[1], sentinel);
        }
        for i in 0..=5 {
            for j in 0..=i {
                for k in 0..3 {
                    let s = src.ret_raw_block(i, j)[(r1[k] * 3 + c1[k]) as usize];
                    let d = dst.ret_raw_block(i, j)[(r2[k] * 2 + c2[k]) as usize];
                    assert_eq!(d, s, "ret ({i},{j}) pair {k}");
                }
            }
            for m in 0..=4 {
                for k in 0..3 {
                    let s = src.lmix_raw_block(i, m)[(r1[k] * 3 + c1[k]) as usize];
                    let d = dst.lmix_raw_block(i, m)[(r2[k] * 2 + c2[k]) as usize];
                    assert_eq!(d, s);
                }
            }
            for j in i..=5 {
                for k in 0..3 {
                    let s = src.less_raw_block(i, j)[(r1[k] * 3 + c1[k]) as usize];
                    let d = dst.less_raw_block(i, j)[(r2[k] * 2 + c2[k]) as usize];
                    assert_eq!(d, s);
                }
            }
            assert_eq!(dst.ret_raw_block(i, 0)[1], sentinel);
            assert_eq!(dst.lmix_raw_block(i, 0)[1], sentinel);
            assert_eq!(dst.less_raw_block(0, i)[1], sentinel);
        }
        // single time step version
        let mut dst = ContourGreen::new(&cd, Sign::Fermi, sentinel);
        let tstp = 3i64;
        elemcpy_green_map_at(tstp, &r1, &c1, &src, &r2, &c2, &mut dst);
        for j in 0..=tstp {
            for k in 0..3 {
                let s = src.ret_raw_block(tstp, j)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.ret_raw_block(tstp, j)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s);
                let s = src.less_raw_block(j, tstp)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.less_raw_block(j, tstp)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s);
            }
        }
        assert!(
            dst.ret_raw_block(tstp - 1, 0)
                .iter()
                .all(|z| *z == sentinel)
        );
        assert!(dst.mat_raw_block(0).iter().all(|z| *z == sentinel));
        // Matsubara version
        elemcpy_green_map_at(MATSUBARA, &r1, &c1, &src, &r2, &c2, &mut dst);
        for k in 0..3 {
            let s = src.mat_raw_block(1)[(r1[k] * 3 + c1[k]) as usize];
            let d = dst.mat_raw_block(1)[(r2[k] * 2 + c2[k]) as usize];
            assert_eq!(d, s);
        }
        // mismatched map lengths are rejected
        let result = std::panic::catch_unwind(|| {
            let mut d = ContourGreen::new(&cd, Sign::Fermi, sentinel);
            elemcpy_green_map(&r1, &c1[..2], &src, &r2, &c2, &mut d);
        });
        assert!(result.is_err());
    }
}
