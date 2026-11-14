// Project : KadanoffBaym
// File    : src/ops.rs
// Purpose : element-wise operations (memcpy/memset/zeros/incr/smul)
//
// Blueprint: julia/src/traits/{memcpy,memset,zeros,incr,smul}.jl. The
// Julia argument order encodes the multiplication side
// (`smul!(x, G)` = x * G, `smul!(G, x)` = G * x); here the side is part
// of the method/function name (`lmul` / `rmul`).

use crate::complex::c64::C64;
use crate::linalg::mat::DenseMat;
use crate::types::green::{
    ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction,
};
use crate::types::MATSUBARA;

/// Adjoint switch for matrix multiplications (blueprint `Mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Adjoint,
}

#[inline]
fn left_mul_block(dst: &mut [C64], m: &DenseMat, size1: usize, size2: usize, mode: Mode) {
    // dst = M(†) * B
    let mut out = vec![C64::ZERO; dst.len()];
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = C64::ZERO;
            for k in 0..size1 {
                let mv = match mode {
                    Mode::Normal => m.get(i, k),
                    Mode::Adjoint => m.get(k, i).conj(),
                };
                acc += mv * dst[k * size2 + j];
            }
            out[i * size2 + j] = acc;
        }
    }
    dst.copy_from_slice(&out);
}

#[inline]
fn right_mul_block(dst: &mut [C64], m: &DenseMat, size1: usize, size2: usize, mode: Mode) {
    // dst = B * M(†)
    let mut out = vec![C64::ZERO; dst.len()];
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = C64::ZERO;
            for k in 0..size2 {
                let mv = match mode {
                    Mode::Normal => m.get(k, j),
                    Mode::Adjoint => m.get(j, k).conj(),
                };
                acc += dst[i * size2 + k] * mv;
            }
            out[i * size2 + j] = acc;
        }
    }
    dst.copy_from_slice(&out);
}

macro_rules! component_ops {
    ($ty:ident) => {
        impl $ty {
            /// Whole-component copy `self <- src`.
            pub fn memcpy_from(&mut self, src: &$ty) {
                assert_eq!(self.raw().len(), src.raw().len(), "memcpy: length mismatch");
                self.raw_mut().copy_from_slice(src.raw());
            }

            /// Whole-component fill.
            pub fn memset(&mut self, v: C64) {
                self.raw_mut().fill(v);
            }

            /// Whole-component zero.
            pub fn set_zero(&mut self) {
                self.memset(C64::ZERO);
            }

            /// Whole-component accumulation `self += src`.
            pub fn incr_by(&mut self, src: &$ty) {
                assert_eq!(self.raw().len(), src.raw().len(), "incr: length mismatch");
                for (d, s) in self.raw_mut().iter_mut().zip(src.raw()) {
                    *d += *s;
                }
            }

            /// Scaled accumulation `self += alpha * src`.
            pub fn incr_scaled(&mut self, src: &$ty, alpha: C64) {
                assert_eq!(self.raw().len(), src.raw().len(), "incr: length mismatch");
                for (d, s) in self.raw_mut().iter_mut().zip(src.raw()) {
                    *d += *s * alpha;
                }
            }

            /// Scale by a complex scalar.
            pub fn smul(&mut self, a: C64) {
                for z in self.raw_mut() {
                    *z = *z * a;
                }
            }

            /// Left-multiply every block by `M(†)`.
            pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
                let (s1, s2) = (self.size1, self.size2);
                for k in 0..self.nblocks() {
                    left_mul_block(self.block_mut(k), m, s1, s2, mode);
                }
            }

            /// Right-multiply every block by `M(†)`.
            pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
                let (s1, s2) = (self.size1, self.size2);
                for k in 0..self.nblocks() {
                    right_mul_block(self.block_mut(k), m, s1, s2, mode);
                }
            }
        }
    };
}

component_ops!(MatComp);
component_ops!(RetComp);
component_ops!(LmixComp);
component_ops!(LessComp);

// ------------------------------------------------------------------
// TimeFunction ops
// ------------------------------------------------------------------

impl TimeFunction {
    /// Copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &TimeFunction) {
        assert_eq!(self.raw().len(), src.raw().len(), "memcpy: length mismatch");
        self.raw_mut().copy_from_slice(src.raw());
    }

    /// Fill every element (including the Matsubara slot).
    pub fn memset(&mut self, v: C64) {
        self.raw_mut().fill(v);
    }

    /// Zero every element.
    pub fn set_zero(&mut self) {
        self.memset(C64::ZERO);
    }

    /// `self += src`.
    pub fn incr_by(&mut self, src: &TimeFunction) {
        for (d, s) in self.raw_mut().iter_mut().zip(src.raw()) {
            *d += *s;
        }
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: C64) {
        for z in self.raw_mut() {
            *z = *z * a;
        }
    }

    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        for tstp in -1..=(self.nt as i64) {
            left_mul_block(self.block_mut(tstp), m, s1, s2, mode);
        }
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        for tstp in -1..=(self.nt as i64) {
            right_mul_block(self.block_mut(tstp), m, s1, s2, mode);
        }
    }
}

// ------------------------------------------------------------------
// Green-level ops (whole object / single time step)
// ------------------------------------------------------------------

macro_rules! green_ops {
    ($ty:ident) => {
        impl $ty {
            /// Copy all components `self <- src` (grids must match).
            pub fn memcpy_from(&mut self, src: &$ty) {
                self.mat.memcpy_from(&src.mat);
                self.ret.memcpy_from(&src.ret);
                self.lmix.memcpy_from(&src.lmix);
                self.less.memcpy_from(&src.less);
            }

            /// Fill all components with `v`.
            pub fn memset(&mut self, v: C64) {
                self.mat.memset(v);
                self.ret.memset(v);
                self.lmix.memset(v);
                self.less.memset(v);
            }

            /// Zero all components.
            pub fn set_zero(&mut self) {
                self.memset(C64::ZERO);
            }

            /// `self += src` component-wise.
            pub fn incr_by(&mut self, src: &$ty) {
                self.mat.incr_by(&src.mat);
                self.ret.incr_by(&src.ret);
                self.lmix.incr_by(&src.lmix);
                self.less.incr_by(&src.less);
            }

            /// `self += alpha * src` component-wise.
            pub fn incr_scaled(&mut self, src: &$ty, alpha: C64) {
                self.mat.incr_scaled(&src.mat, alpha);
                self.ret.incr_scaled(&src.ret, alpha);
                self.lmix.incr_scaled(&src.lmix, alpha);
                self.less.incr_scaled(&src.less, alpha);
            }

            /// Scale all components by a complex scalar.
            pub fn smul(&mut self, a: C64) {
                self.mat.smul(a);
                self.ret.smul(a);
                self.lmix.smul(a);
                self.less.smul(a);
            }

            /// Scale all components by a real scalar.
            pub fn smul_f64(&mut self, a: f64) {
                self.smul(C64::new(a, 0.0));
            }

            /// Left-multiply every block of every component by `M(†)`.
            pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
                self.mat.lmul(m, mode);
                self.ret.lmul(m, mode);
                self.lmix.lmul(m, mode);
                self.less.lmul(m, mode);
            }

            /// Right-multiply every block of every component by `M(†)`.
            pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
                self.mat.rmul(m, mode);
                self.ret.rmul(m, mode);
                self.lmix.rmul(m, mode);
                self.less.rmul(m, mode);
            }
        }
    };
}

green_ops!(ContourGreen);
green_ops!(GreenTimeStep);

impl ContourGreen {
    /// `G[row tstp] += w * f` (blueprint `incr!(G, g, tstp, w)`):
    /// row-scoped accumulation of a single-timestep object; at
    /// `MATSUBARA` only the mat component is updated.
    pub fn incr_by_step(&mut self, f: &GreenTimeStep, tstp: i64, w: C64) {
        if tstp == MATSUBARA {
            for q in 0..=(self.ntau as i64) {
                let upd: Vec<C64> = self
                    .mat_raw_block(q)
                    .iter()
                    .zip(f.mat_raw_block(q))
                    .map(|(x, y)| *x + w * *y)
                    .collect();
                self.mat_raw_block_mut(q).copy_from_slice(&upd);
            }
            return;
        }
        assert!((0..=self.nt as i64).contains(&tstp));
        for j in 0..=tstp {
            let upd: Vec<C64> = self
                .ret_raw_block(tstp, j)
                .iter()
                .zip(f.ret_raw_block(tstp, j))
                .map(|(x, y)| *x + w * *y)
                .collect();
            self.ret_raw_block_mut(tstp, j).copy_from_slice(&upd);
            let upd: Vec<C64> = self
                .less_raw_block(j, tstp)
                .iter()
                .zip(f.less_raw_block(j, tstp))
                .map(|(x, y)| *x + w * *y)
                .collect();
            self.less_raw_block_mut(j, tstp).copy_from_slice(&upd);
        }
        for m in 0..=(self.ntau as i64) {
            let upd: Vec<C64> = self
                .lmix_raw_block(tstp, m)
                .iter()
                .zip(f.lmix_raw_block(tstp, m))
                .map(|(x, y)| *x + w * *y)
                .collect();
            self.lmix_raw_block_mut(tstp, m).copy_from_slice(&upd);
        }
    }
}

impl GreenTimeStep {
    /// Left-multiply the timestep blocks by the time-dependent matrix
    /// `cf(†)` at each block's FIRST time argument (blueprint
    /// `smul!(x::Cf, g, tstp)`): the ret and lmix rows carry the row
    /// time `tstp`, the less blocks `(j, tstp)` carry their own `j`,
    /// and the (shared) mat component the Matsubara slot.
    pub fn lmul_cf(&mut self, cf: &TimeFunction, mode: Mode) {
        assert!(self.size1 == self.size2, "Cf multiplication requires square blocks");
        let (s1, s2) = (self.size1, self.size2);
        let b = s1 * s2;
        let tstp = self.tstp as i64;
        let m = cf_block(cf, MATSUBARA, mode);
        self.mat.lmul(&m, Mode::Normal);
        let m = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let ntau = self.ntau;
        for k in 0..=ntau {
            left_mul_block(
                &mut self.lmix.raw_mut()[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
            );
        }
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            left_mul_block(
                &mut self.less.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
            );
        }
    }

    /// Right-multiply the timestep blocks by `cf(†)` at each block's
    /// SECOND time argument (blueprint `smul!(g, x::Cf, tstp)`): the
    /// ret blocks `(tstp, j)` carry their column `j`, the less blocks
    /// `(j, tstp)` carry `tstp`, the lmix blocks and the (shared) mat
    /// component the Matsubara slot.
    pub fn rmul_cf(&mut self, cf: &TimeFunction, mode: Mode) {
        assert!(self.size1 == self.size2, "Cf multiplication requires square blocks");
        let (s1, s2) = (self.size1, self.size2);
        let b = s1 * s2;
        let tstp = self.tstp as i64;
        let mm = cf_block(cf, MATSUBARA, mode);
        self.mat.rmul(&mm, Mode::Normal);
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let ntau = self.ntau;
        for k in 0..=ntau {
            right_mul_block(
                &mut self.lmix.raw_mut()[k * b..(k + 1) * b],
                &mm,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let mt = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                &mt,
                s1,
                s2,
                Mode::Normal,
            );
        }
    }
}

impl ContourGreen {
    /// Copy the components at `tstp` (`MATSUBARA` = mat) from `src`.
    pub fn memcpy_from_at(&mut self, tstp: i64, src: &ContourGreen) {
        if tstp == MATSUBARA {
            self.mat.memcpy_from(&src.mat);
            return;
        }
        self.check_tstp(tstp);
        self.ret.row_mut(tstp).copy_from_slice(src.ret.row(tstp));
        self.lmix_row_mut(tstp).copy_from_slice(src.lmix_row(tstp));
        self.less.row_mut(tstp).copy_from_slice(src.less.row(tstp));
    }

    /// Zero the components at `tstp`.
    pub fn set_zero_at(&mut self, tstp: i64) {
        self.memset_at(tstp, C64::ZERO);
    }

    /// Fill the components at `tstp` with `v`.
    pub fn memset_at(&mut self, tstp: i64, v: C64) {
        if tstp == MATSUBARA {
            self.mat.memset(v);
            return;
        }
        self.check_tstp(tstp);
        self.ret.row_mut(tstp).fill(v);
        self.lmix_row_mut(tstp).fill(v);
        self.less.row_mut(tstp).fill(v);
    }

    /// `self[components at tstp] += src[...]`.
    pub fn incr_by_at(&mut self, tstp: i64, src: &ContourGreen) {
        if tstp == MATSUBARA {
            self.mat.incr_by(&src.mat);
            return;
        }
        self.check_tstp(tstp);
        for (d, s) in self.ret.row_mut(tstp).iter_mut().zip(src.ret.row(tstp)) {
            *d += *s;
        }
        for (d, s) in self.lmix_row_mut(tstp).iter_mut().zip(src.lmix_row(tstp)) {
            *d += *s;
        }
        for (d, s) in self.less.row_mut(tstp).iter_mut().zip(src.less.row(tstp)) {
            *d += *s;
        }
    }

    /// `self[components at tstp] += alpha * src[...]`.
    pub fn incr_scaled_at(&mut self, tstp: i64, src: &ContourGreen, alpha: C64) {
        if tstp == MATSUBARA {
            self.mat.incr_scaled(&src.mat, alpha);
            return;
        }
        self.check_tstp(tstp);
        for (d, s) in self.ret.row_mut(tstp).iter_mut().zip(src.ret.row(tstp)) {
            *d += *s * alpha;
        }
        for (d, s) in self.lmix_row_mut(tstp).iter_mut().zip(src.lmix_row(tstp)) {
            *d += *s * alpha;
        }
        for (d, s) in self.less.row_mut(tstp).iter_mut().zip(src.less.row(tstp)) {
            *d += *s * alpha;
        }
    }

    /// Scale the components at `tstp` by a complex scalar.
    pub fn smul_at(&mut self, tstp: i64, a: C64) {
        if tstp == MATSUBARA {
            self.mat.smul(a);
            return;
        }
        self.check_tstp(tstp);
        for z in self.ret.row_mut(tstp) {
            *z = *z * a;
        }
        for z in self.lmix_row_mut(tstp) {
            *z = *z * a;
        }
        for z in self.less.row_mut(tstp) {
            *z = *z * a;
        }
    }

    #[inline]
    fn check_tstp(&self, tstp: i64) {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "tstp {tstp} out of range 0..={}",
            self.nt
        );
    }

    #[inline]
    fn lmix_row_mut(&mut self, tstp: i64) -> &mut [C64] {
        let n = (self.ntau + 1) * self.size1 * self.size2;
        let off = tstp as usize * n;
        &mut self.lmix.raw_mut()[off..off + n]
    }

    #[inline]
    fn lmix_row(&self, tstp: i64) -> &[C64] {
        let n = (self.ntau + 1) * self.size1 * self.size2;
        let off = tstp as usize * n;
        &self.lmix.raw()[off..off + n]
    }

    /// Left-multiply the blocks at `tstp` by `M(†)`.
    pub fn lmul_at(&mut self, tstp: i64, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let b = s1 * s2;
        if tstp == MATSUBARA {
            self.mat.lmul(m, mode);
            return;
        }
        self.check_tstp(tstp);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix_row_mut(tstp);
        for k in 0..=ntau {
            left_mul_block(&mut row[k * b..(k + 1) * b], m, s1, s2, mode);
        }
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `M(†)`.
    pub fn rmul_at(&mut self, tstp: i64, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let b = s1 * s2;
        if tstp == MATSUBARA {
            self.mat.rmul(m, mode);
            return;
        }
        self.check_tstp(tstp);
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix_row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(&mut row[k * b..(k + 1) * b], m, s1, s2, mode);
        }
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
            );
        }
    }

    /// Left-multiply the blocks at `tstp` by the time-dependent matrix
    /// `cf[tstp](†)` (blueprint `smul!(x::Cf, ...)`).
    /// Left-multiply the blocks at `tstp` by the time-dependent matrix
    /// `cf(†)` evaluated at each block's FIRST time argument (blueprint
    /// `smul!(x::Cf, G, tstp)`): the ret and lmix rows carry the row
    /// time `tstp`, the less blocks `(j, tstp)` carry their own `j`,
    /// and the mat component the Matsubara slot.
    pub fn lmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, mode: Mode) {
        assert!(self.size1 == self.size2, "Cf multiplication requires square blocks");
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, mode);
            self.mat.lmul(&m, Mode::Normal);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let m = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix_row_mut(tstp);
        for k in 0..=ntau {
            left_mul_block(
                &mut row[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
            );
        }
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            left_mul_block(
                &mut self.less.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `cf(†)` evaluated at each
    /// block's SECOND time argument (blueprint `smul!(G, x::Cf, tstp)`):
    /// the ret blocks `(tstp, j)` carry their column `j`, the less
    /// blocks `(j, tstp)` carry `tstp`, and the lmix blocks and the mat
    /// component carry the Matsubara slot.
    pub fn rmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, mode: Mode) {
        assert!(self.size1 == self.size2, "Cf multiplication requires square blocks");
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, mode);
            self.mat.rmul(&m, Mode::Normal);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let mm = cf_block(cf, MATSUBARA, mode);
        let ntau = self.ntau;
        let row = self.lmix_row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(
                &mut row[k * b..(k + 1) * b],
                &mm,
                s1,
                s2,
                Mode::Normal,
            );
        }
        let mt = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                &mt,
                s1,
                s2,
                Mode::Normal,
            );
        }
    }
}

fn cf_block(cf: &TimeFunction, tstp: i64, mode: Mode) -> DenseMat {
    let b = cf.block(tstp);
    let mut m = DenseMat::from_vec(cf.size1, cf.size2, b.to_vec());
    if mode == Mode::Adjoint {
        m = m.adjoint();
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 2, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, C64::ZERO)
    }

    fn fill(g: &mut ContourGreen) {
        for n in 0..=2 {
            g.set_mat(n, 0, 0, C64::new(n as f64, 1.0));
        }
        g.set_ret(2, 1, 0, 1, C64::new(3.0, 0.0));
        g.set_lmix(1, 2, 1, 0, C64::new(4.0, 0.0));
        g.set_less(0, 1, 0, 0, C64::new(5.0, 0.0));
    }

    #[test]
    fn whole_green_ops() {
        let mut a = mk(2);
        fill(&mut a);
        let mut b = mk(2);
        b.memcpy_from(&a);
        assert_eq!(b.get_mat(2, 0, 0), C64::new(2.0, 1.0));
        assert_eq!(b.get_ret(2, 1, 0, 1), C64::new(3.0, 0.0));
        b.incr_by(&a);
        assert_eq!(b.get_mat(2, 0, 0), C64::new(4.0, 2.0));
        b.smul(C64::new(0.0, 1.0));
        assert_eq!(b.get_mat(2, 0, 0), C64::new(-2.0, 4.0));
        b.set_zero();
        assert_eq!(b.get_less(0, 1, 0, 0), C64::ZERO);
    }

    #[test]
    fn timestep_scoped_ops() {
        let mut a = mk(2);
        fill(&mut a);
        let mut b = mk(2);
        b.memcpy_from_at(2, &a);
        assert_eq!(b.get_ret(2, 1, 0, 1), C64::new(3.0, 0.0));
        assert_eq!(b.get_mat(2, 0, 0), C64::ZERO); // mat untouched
        b.set_zero_at(2);
        assert_eq!(b.get_ret(2, 1, 0, 1), C64::ZERO);
        // Matsubara scope
        let mut c = mk(2);
        c.memcpy_from_at(MATSUBARA, &a);
        assert_eq!(c.get_mat(1, 0, 0), C64::new(1.0, 1.0));
        assert_eq!(c.get_ret(2, 1, 0, 1), C64::ZERO);
        // incr and smul at one step
        let mut d = mk(2);
        d.incr_by_at(2, &a);
        d.incr_by_at(2, &a);
        assert_eq!(d.get_ret(2, 1, 0, 1), C64::new(6.0, 0.0));
        d.smul_at(2, C64::new(0.5, 0.0));
        assert_eq!(d.get_ret(2, 1, 0, 1), C64::new(3.0, 0.0));
    }

    #[test]
    fn left_right_matrix_multiply() {
        // B = [[2,0],[0,3]], M = [[1,1],[0,1]]
        let mut a = mk(1);
        let set_b = |a: &mut ContourGreen| {
            a.set_mat(0, 0, 0, C64::new(2.0, 0.0));
            a.set_mat(0, 0, 1, C64::ZERO);
            a.set_mat(0, 1, 0, C64::ZERO);
            a.set_mat(0, 1, 1, C64::new(3.0, 0.0));
        };
        let mut m = DenseMat::identity(2);
        m.set(0, 1, C64::new(1.0, 0.0));
        // lmul Normal: M*B -> (0,1) = M(0,0)B(0,1) + M(0,1)B(1,1) = 3
        set_b(&mut a);
        a.mat.lmul(&m, Mode::Normal);
        assert_eq!(a.get_mat(0, 0, 1), C64::new(3.0, 0.0));
        assert_eq!(a.get_mat(0, 0, 0), C64::new(2.0, 0.0));
        // lmul Adjoint: M† = [[1,0],[1,1]] -> (1,0) = M†(1,0)B(0,0) = 2
        set_b(&mut a);
        a.mat.lmul(&m, Mode::Adjoint);
        assert_eq!(a.get_mat(0, 1, 0), C64::new(2.0, 0.0));
        assert_eq!(a.get_mat(0, 1, 1), C64::new(3.0, 0.0));
        // rmul Normal: B*M -> (0,1) = B(0,0)M(0,1) = 2
        set_b(&mut a);
        a.mat.rmul(&m, Mode::Normal);
        assert_eq!(a.get_mat(0, 0, 1), C64::new(2.0, 0.0));
        assert_eq!(a.get_mat(0, 1, 1), C64::new(3.0, 0.0));
    }

    #[test]
    fn row_scoped_matrix_multiply() {
        let mut a = mk(2);
        a.set_ret(2, 0, 0, 0, C64::new(2.0, 0.0));
        a.set_mat(0, 0, 0, C64::new(9.0, 0.0));
        let mut m = DenseMat::identity(2);
        m.set(1, 0, C64::new(1.0, 0.0));
        a.lmul_at(2, &m, Mode::Normal);
        // ret(2,0) block row 1 gets old row 0: (1,0) = 2
        assert_eq!(a.get_ret(2, 0, 1, 0), C64::new(2.0, 0.0));
        // mat untouched by tstp=2 scope
        assert_eq!(a.get_mat(0, 0, 0), C64::new(9.0, 0.0));
    }

    #[test]
    fn time_function_weight_multiply() {
        let mut a = mk(1);
        a.set_mat(0, 0, 0, C64::new(2.0, 0.0));
        let mut cf = TimeFunction::new(1, 2, 2, C64::ZERO);
        cf.set(MATSUBARA, 0, 0, C64::new(1.0, 1.0));
        a.lmul_cf_at(MATSUBARA, &cf, Mode::Normal);
        // mat[0] = M * B: (0,0) = M(0,0)*B(0,0) = (1+i)*2
        assert_eq!(a.get_mat(0, 0, 0), C64::new(2.0, 2.0));
    }

    #[test]
    fn time_function_ops() {
        let mut cf = TimeFunction::new(2, 1, 1, C64::ZERO);
        cf.set(1, 0, 0, C64::new(1.0, 0.0));
        cf.set(MATSUBARA, 0, 0, C64::new(2.0, 0.0));
        let mut d = TimeFunction::new(2, 1, 1, C64::ZERO);
        d.memcpy_from(&cf);
        assert_eq!(d.get(1, 0, 0), C64::ONE);
        d.incr_by(&cf);
        d.smul(C64::new(2.0, 0.0));
        assert_eq!(d.get(1, 0, 0), C64::new(4.0, 0.0));
        assert_eq!(d.get(MATSUBARA, 0, 0), C64::new(8.0, 0.0));
        d.set_zero();
        assert_eq!(d.get(MATSUBARA, 0, 0), C64::ZERO);
    }
}

/// Hermitize the Matsubara component in place: every block becomes
/// `(B + B†) / 2` (blueprint `force_matsubara_hermitian!`); only square
/// blocks are supported.
pub fn force_matsubara_hermitian<T: MatsubaraHermitian>(g: &mut T) {
    g.force_matsubara_hermitian();
}

/// Common implementation over the green types that own a mat component.
pub trait MatsubaraHermitian {
    fn force_matsubara_hermitian(&mut self);
}

impl MatsubaraHermitian for crate::types::green::ContourGreen {
    fn force_matsubara_hermitian(&mut self) {
        assert!(
            self.size1 == self.size2,
            "force_matsubara_hermitian requires square blocks"
        );
        let n = self.size1;
        for q in 0..self.mat.nblocks() {
            let blk = self.mat.block(q).to_vec();
            for a in 0..n {
                for b in 0..n {
                    let avg = (blk[a * n + b] + blk[b * n + a].conj()) * 0.5;
                    self.mat.set(q, a, b, avg);
                }
            }
        }
    }
}

impl MatsubaraHermitian for crate::types::green::GreenTimeStep {
    fn force_matsubara_hermitian(&mut self) {
        assert!(
            self.size1 == self.size2,
            "force_matsubara_hermitian requires square blocks"
        );
        let n = self.size1;
        for q in 0..self.mat.nblocks() {
            let blk = self.mat.block(q).to_vec();
            for a in 0..n {
                for b in 0..n {
                    let avg = (blk[a * n + b] + blk[b * n + a].conj()) * 0.5;
                    self.mat.set(q, a, b, avg);
                }
            }
        }
    }
}
