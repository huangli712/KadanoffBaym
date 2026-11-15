//
// Project : KadanoffBaym
// File    : src/basics/mul.rs
// Purpose : matrix multiplication operations (lmul/rmul and the
//           time-dependent Cf variants)
//
// encodes the multiplication side (`smul!(x, G)` = x * G, `smul!(G,
// x)` = G * x); here the side is part of the method/function name
// (`lmul` / `rmul`).

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

/// Adjoint switch for matrix multiplications (blueprint `Mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Adjoint,
}

// ------------------------------------------------------------------
// component ops
// ------------------------------------------------------------------

impl MatComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }
}

impl RetComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }
}

impl LmixComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }
}

impl LessComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, mode, &mut scratch);
    }
}

impl TimeFunction {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        for tstp in -1..=(self.nt as i64) {
            left_mul_block(self.block_mut(tstp), m, s1, s2, mode, &mut scratch);
        }
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        for tstp in -1..=(self.nt as i64) {
            right_mul_block(self.block_mut(tstp), m, s1, s2, mode, &mut scratch);
        }
    }
}

// ------------------------------------------------------------------
// green-level ops (whole object)
// ------------------------------------------------------------------

impl ContourGreen {
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

impl GreenTimeStep {
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

// ------------------------------------------------------------------
// green-level ops (row-scoped at one time step)
// ------------------------------------------------------------------

impl ContourGreen {
    /// Left-multiply the blocks at `tstp` by `M(†)`.
    pub fn lmul_at(&mut self, tstp: i64, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            self.mat.lmul(m, mode);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            left_mul_block(&mut row[k * b..(k + 1) * b], m, s1, s2, mode, &mut scratch);
        }
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
                &mut scratch,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `M(†)`.
    pub fn rmul_at(&mut self, tstp: i64, m: &DenseMat, mode: Mode) {
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            self.mat.rmul(m, mode);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(&mut row[k * b..(k + 1) * b], m, s1, s2, mode, &mut scratch);
        }
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                mode,
                &mut scratch,
            );
        }
    }

    /// Left-multiply the blocks at `tstp` by `cf(†)` evaluated at each
    /// block's FIRST time argument (blueprint `smul!(x::Cf, G, tstp)`):
    /// the ret and lmix rows carry the row time `tstp`, the less blocks
    /// `(j, tstp)` carry their own `j`, and the mat component the
    /// Matsubara slot.
    pub fn lmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, mode: Mode) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, mode);
            self.mat.lmul(&m, Mode::Normal);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        let m = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            left_mul_block(
                &mut row[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
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
                &mut scratch,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `cf(†)` evaluated at each
    /// block's SECOND time argument (blueprint `smul!(G, x::Cf, tstp)`):
    /// the ret blocks `(tstp, j)` carry their column `j`, the less
    /// blocks `(j, tstp)` carry `tstp`, and the lmix blocks and the mat
    /// component carry the Matsubara slot.
    pub fn rmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, mode: Mode) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, mode);
            self.mat.rmul(&m, Mode::Normal);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
            );
        }
        let mm = cf_block(cf, MATSUBARA, mode);
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(
                &mut row[k * b..(k + 1) * b],
                &mm,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
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
                &mut scratch,
            );
        }
    }
}

// ------------------------------------------------------------------
// timestep Cf multiplication
// ------------------------------------------------------------------

impl GreenTimeStep {
    /// Left-multiply the timestep blocks by the time-dependent matrix
    /// `cf(†)` at each block's FIRST time argument (blueprint
    /// `smul!(x::Cf, g, tstp)`): the ret and lmix rows carry the row
    /// time `tstp`, the less blocks `(j, tstp)` carry their own `j`,
    /// and the (shared) mat component the Matsubara slot.
    pub fn lmul_cf(&mut self, cf: &TimeFunction, mode: Mode) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        let tstp = self.tstp as i64;
        let m = cf_block(cf, MATSUBARA, mode);
        self.mat.lmul(&m, Mode::Normal);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        let m = cf_block(cf, tstp, mode);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        for k in 0..=ntau {
            left_mul_block(
                self.lmix.block_mut(k),
                &m,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
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
                &mut scratch,
            );
        }
    }

    /// Right-multiply the timestep blocks by `cf(†)` at each block's
    /// SECOND time argument (blueprint `smul!(g, x::Cf, tstp)`): the
    /// ret blocks `(tstp, j)` carry their column `j`, the less blocks
    /// `(j, tstp)` carry `tstp`, the lmix blocks and the (shared) mat
    /// component the Matsubara slot.
    pub fn rmul_cf(&mut self, cf: &TimeFunction, mode: Mode) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        let tstp = self.tstp as i64;
        let mm = cf_block(cf, MATSUBARA, mode);
        self.mat.rmul(&mm, Mode::Normal);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for j in 0..=tstp {
            let mj = cf_block(cf, j, mode);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        for k in 0..=ntau {
            right_mul_block(
                self.lmix.block_mut(k),
                &mm,
                s1,
                s2,
                Mode::Normal,
                &mut scratch,
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
                &mut scratch,
            );
        }
    }
}

// ------------------------------------------------------------------
// block-level kernels (each loop written once)
// ------------------------------------------------------------------

#[inline]
fn left_mul_block(
    dst: &mut [Complex],
    m: &DenseMat,
    size1: usize,
    size2: usize,
    mode: Mode,
    scratch: &mut [Complex],
) {
    // dst = M(†) * B; `scratch` is dst-sized workspace (fully overwritten)
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = Complex::ZERO;
            for k in 0..size1 {
                let mv = match mode {
                    Mode::Normal => m.get(i, k),
                    Mode::Adjoint => m.get(k, i).conj(),
                };
                acc += mv * dst[k * size2 + j];
            }
            scratch[i * size2 + j] = acc;
        }
    }
    dst.copy_from_slice(scratch);
}

#[inline]
fn right_mul_block(
    dst: &mut [Complex],
    m: &DenseMat,
    size1: usize,
    size2: usize,
    mode: Mode,
    scratch: &mut [Complex],
) {
    // dst = B * M(†); `scratch` is dst-sized workspace (fully overwritten)
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = Complex::ZERO;
            for k in 0..size2 {
                let mv = match mode {
                    Mode::Normal => m.get(k, j),
                    Mode::Adjoint => m.get(j, k).conj(),
                };
                acc += dst[i * size2 + k] * mv;
            }
            scratch[i * size2 + j] = acc;
        }
    }
    dst.copy_from_slice(scratch);
}

fn cf_block(cf: &TimeFunction, tstp: i64, mode: Mode) -> DenseMat {
    let b = cf.block(tstp);
    let mut m = DenseMat::from_slice(cf.size1, cf.size2, b);
    if mode == Mode::Adjoint {
        m = m.adjoint();
    }
    m
}

/// Left-multiply every `b`-sized block of `raw` by `M(†)`.
fn slice_lmul(
    raw: &mut [Complex],
    nblocks: usize,
    m: &DenseMat,
    s1: usize,
    s2: usize,
    mode: Mode,
    scratch: &mut [Complex],
) {
    let b = s1 * s2;
    for k in 0..nblocks {
        left_mul_block(&mut raw[k * b..(k + 1) * b], m, s1, s2, mode, scratch);
    }
}

/// Right-multiply every `b`-sized block of `raw` by `M(†)`.
fn slice_rmul(
    raw: &mut [Complex],
    nblocks: usize,
    m: &DenseMat,
    s1: usize,
    s2: usize,
    mode: Mode,
    scratch: &mut [Complex],
) {
    let b = s1 * s2;
    for k in 0..nblocks {
        right_mul_block(&mut raw[k * b..(k + 1) * b], m, s1, s2, mode, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 2, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    #[test]
    fn left_right_matrix_multiply() {
        // B = [[2,0],[0,3]], M = [[1,1],[0,1]]
        let mut a = mk(1);
        let set_b = |a: &mut ContourGreen| {
            a.set_mat(0, 0, 0, Complex::new(2.0, 0.0));
            a.set_mat(0, 0, 1, Complex::ZERO);
            a.set_mat(0, 1, 0, Complex::ZERO);
            a.set_mat(0, 1, 1, Complex::new(3.0, 0.0));
        };
        let mut m = DenseMat::identity(2);
        m.set(0, 1, Complex::new(1.0, 0.0));
        // lmul Normal: M*B -> (0,1) = M(0,0)B(0,1) + M(0,1)B(1,1) = 3
        set_b(&mut a);
        a.mat.lmul(&m, Mode::Normal);
        assert_eq!(a.get_mat(0, 0, 1), Complex::new(3.0, 0.0));
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(2.0, 0.0));
        // lmul Adjoint: M† = [[1,0],[1,1]] -> (1,0) = M†(1,0)B(0,0) = 2
        set_b(&mut a);
        a.mat.lmul(&m, Mode::Adjoint);
        assert_eq!(a.get_mat(0, 1, 0), Complex::new(2.0, 0.0));
        assert_eq!(a.get_mat(0, 1, 1), Complex::new(3.0, 0.0));
        // rmul Normal: B*M -> (0,1) = B(0,0)M(0,1) = 2
        set_b(&mut a);
        a.mat.rmul(&m, Mode::Normal);
        assert_eq!(a.get_mat(0, 0, 1), Complex::new(2.0, 0.0));
        assert_eq!(a.get_mat(0, 1, 1), Complex::new(3.0, 0.0));
    }

    #[test]
    fn row_scoped_matrix_multiply() {
        let mut a = mk(2);
        a.set_ret(2, 0, 0, 0, Complex::new(2.0, 0.0));
        a.set_mat(0, 0, 0, Complex::new(9.0, 0.0));
        let mut m = DenseMat::identity(2);
        m.set(1, 0, Complex::new(1.0, 0.0));
        a.lmul_at(2, &m, Mode::Normal);
        // ret(2,0) block row 1 gets old row 0: (1,0) = 2
        assert_eq!(a.get_ret(2, 0, 1, 0), Complex::new(2.0, 0.0));
        // mat untouched by tstp=2 scope
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(9.0, 0.0));
    }

    #[test]
    fn time_function_weight_multiply() {
        let mut a = mk(1);
        a.set_mat(0, 0, 0, Complex::new(2.0, 0.0));
        let mut cf = TimeFunction::new(1, 2, 2, Complex::ZERO);
        cf.set(MATSUBARA, 0, 0, Complex::new(1.0, 1.0));
        a.lmul_cf_at(MATSUBARA, &cf, Mode::Normal);
        // mat[0] = M * B: (0,0) = M(0,0)*B(0,0) = (1+i)*2
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(2.0, 2.0));
    }
}
