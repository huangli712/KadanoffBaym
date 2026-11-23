//
// Project : KadanoffBaym
// File    : src/algebra/mul.rs
// Purpose : matrix multiplication operations (lmul/rmul and the
//           time-dependent Cf variants)
//
// encodes the multiplication side (`smul!(x, G)` = x * G, `smul!(G,
// x)` = G * x); here the side is part of the method/function name
// (`lmul` / `rmul`).

use crate::complex::Complex;
use crate::linalg::{Adjoint, DenseMat};
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

// ------------------------------------------------------------------
// green-level ops
// ------------------------------------------------------------------

impl ContourGreen {
    /// Left-multiply every block of every component by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        self.mat.lmul(m, adjoint);
        self.ret.lmul(m, adjoint);
        self.lmix.lmul(m, adjoint);
        self.less.lmul(m, adjoint);
    }

    /// Right-multiply every block of every component by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        self.mat.rmul(m, adjoint);
        self.ret.rmul(m, adjoint);
        self.lmix.rmul(m, adjoint);
        self.less.rmul(m, adjoint);
    }

    /// Left-multiply the blocks at `tstp` by `M(†)`.
    pub fn lmul_at(&mut self, tstp: i64, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            self.mat.lmul(m, adjoint);
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
                adjoint,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            left_mul_block(
                &mut row[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                adjoint,
                &mut scratch,
            );
        }
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                adjoint,
                &mut scratch,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `M(†)`.
    pub fn rmul_at(&mut self, tstp: i64, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            self.mat.rmul(m, adjoint);
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
                adjoint,
                &mut scratch,
            );
        }
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(
                &mut row[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                adjoint,
                &mut scratch,
            );
        }
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                m,
                s1,
                s2,
                adjoint,
                &mut scratch,
            );
        }
    }

    /// Left-multiply the blocks at `tstp` by `cf(†)` evaluated at each
    /// block's FIRST time argument: the ret and lmix rows carry the row
    /// time `tstp`, the less blocks `(j, tstp)` carry their own `j`, and
    /// the mat component the Matsubara slot.
    pub fn lmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, adjoint: Adjoint) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, adjoint);
            self.mat.lmul(&m, Adjoint::No);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        let m = cf_block(cf, tstp, adjoint);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Adjoint::No,
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
                Adjoint::No,
                &mut scratch,
            );
        }
        for j in 0..=tstp {
            let mj = cf_block(cf, j, adjoint);
            left_mul_block(
                &mut self.less.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
    }

    /// Right-multiply the blocks at `tstp` by `cf(†)` evaluated at each
    /// block's SECOND time argument: the ret blocks `(tstp, j)` carry
    /// their column `j`, the less blocks `(j, tstp)` carry `tstp`, and the
    /// lmix blocks and the mat component carry the Matsubara slot.
    pub fn rmul_cf_at(&mut self, tstp: i64, cf: &TimeFunction, adjoint: Adjoint) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        if tstp == MATSUBARA {
            let m = cf_block(cf, MATSUBARA, adjoint);
            self.mat.rmul(&m, Adjoint::No);
            return;
        }
        self.check_tstp(tstp);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for j in 0..=tstp {
            let mj = cf_block(cf, j, adjoint);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
        let mm = cf_block(cf, MATSUBARA, adjoint);
        let ntau = self.ntau;
        let row = self.lmix.row_mut(tstp);
        for k in 0..=ntau {
            right_mul_block(
                &mut row[k * b..(k + 1) * b],
                &mm,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
        let mt = cf_block(cf, tstp, adjoint);
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                &mt,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
    }
}

impl GreenTimeStep {
    /// Left-multiply every block of every component by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        self.mat.lmul(m, adjoint);
        self.ret.lmul(m, adjoint);
        self.lmix.lmul(m, adjoint);
        self.less.lmul(m, adjoint);
    }

    /// Right-multiply every block of every component by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        self.mat.rmul(m, adjoint);
        self.ret.rmul(m, adjoint);
        self.lmix.rmul(m, adjoint);
        self.less.rmul(m, adjoint);
    }

    /// Left-multiply the timestep blocks by the time-dependent matrix
    /// `cf(†)` at each block's FIRST time argument: the ret and lmix rows
    /// carry the row time `tstp`, the less blocks `(j, tstp)` carry their
    /// own `j`, and the (shared) mat component the Matsubara slot.
    pub fn lmul_cf(&mut self, cf: &TimeFunction, adjoint: Adjoint) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        let tstp = self.tstp as i64;
        let m = cf_block(cf, MATSUBARA, adjoint);
        self.mat.lmul(&m, Adjoint::No);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        let m = cf_block(cf, tstp, adjoint);
        for k in 0..=(tstp as usize) {
            left_mul_block(
                &mut self.ret.row_mut(tstp)[k * b..(k + 1) * b],
                &m,
                s1,
                s2,
                Adjoint::No,
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
                Adjoint::No,
                &mut scratch,
            );
        }
        for j in 0..=tstp {
            let mj = cf_block(cf, j, adjoint);
            left_mul_block(
                &mut self.less.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
    }

    /// Right-multiply the timestep blocks by `cf(†)` at each block's
    /// SECOND time argument: the ret blocks `(tstp, j)` carry their column
    /// `j`, the less blocks `(j, tstp)` carry `tstp`, the lmix blocks and
    /// the (shared) mat component the Matsubara slot.
    pub fn rmul_cf(&mut self, cf: &TimeFunction, adjoint: Adjoint) {
        assert!(
            self.size1 == self.size2,
            "Cf multiplication requires square blocks"
        );
        let (s1, s2) = (self.size1, self.size2);
        let tstp = self.tstp as i64;
        let mm = cf_block(cf, MATSUBARA, adjoint);
        self.mat.rmul(&mm, Adjoint::No);
        let b = s1 * s2;
        let mut scratch = vec![Complex::ZERO; b];
        for j in 0..=tstp {
            let mj = cf_block(cf, j, adjoint);
            right_mul_block(
                &mut self.ret.row_mut(tstp)[(j as usize) * b..(j as usize + 1) * b],
                &mj,
                s1,
                s2,
                Adjoint::No,
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
                Adjoint::No,
                &mut scratch,
            );
        }
        let mt = cf_block(cf, tstp, adjoint);
        for k in 0..=(tstp as usize) {
            right_mul_block(
                &mut self.less.row_mut(tstp)[k * b..(k + 1) * b],
                &mt,
                s1,
                s2,
                Adjoint::No,
                &mut scratch,
            );
        }
    }
}

impl TimeFunction {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        for tstp in -1..=(self.nt as i64) {
            left_mul_block(self.block_mut(tstp), m, s1, s2, adjoint, &mut scratch);
        }
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        for tstp in -1..=(self.nt as i64) {
            right_mul_block(self.block_mut(tstp), m, s1, s2, adjoint, &mut scratch);
        }
    }
}

// ------------------------------------------------------------------
// component ops
// ------------------------------------------------------------------

impl MatComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }
}

impl RetComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }
}

impl LmixComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }
}

impl LessComp {
    /// Left-multiply every block by `M(†)`.
    pub fn lmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_lmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
    }

    /// Right-multiply every block by `M(†)`.
    pub fn rmul(&mut self, m: &DenseMat, adjoint: Adjoint) {
        let (s1, s2) = (self.size1, self.size2);
        let n = self.nblocks();
        let mut scratch = vec![Complex::ZERO; s1 * s2];
        slice_rmul(self.raw_mut(), n, m, s1, s2, adjoint, &mut scratch);
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
    adjoint: Adjoint,
    scratch: &mut [Complex],
) {
    // dst = M(†) * B; `scratch` is dst-sized workspace (fully overwritten)
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = Complex::ZERO;
            for k in 0..size1 {
                let mv = match adjoint {
                    Adjoint::No => m.get(i, k),
                    Adjoint::Yes => m.get(k, i).conj(),
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
    adjoint: Adjoint,
    scratch: &mut [Complex],
) {
    // dst = B * M(†); `scratch` is dst-sized workspace (fully overwritten)
    for i in 0..size1 {
        for j in 0..size2 {
            let mut acc = Complex::ZERO;
            for k in 0..size2 {
                let mv = match adjoint {
                    Adjoint::No => m.get(k, j),
                    Adjoint::Yes => m.get(j, k).conj(),
                };
                acc += dst[i * size2 + k] * mv;
            }
            scratch[i * size2 + j] = acc;
        }
    }
    dst.copy_from_slice(scratch);
}

fn cf_block(cf: &TimeFunction, tstp: i64, adjoint: Adjoint) -> DenseMat {
    let b = cf.block(tstp);
    let mut m = DenseMat::from_slice(cf.size1, cf.size2, b);
    if adjoint == Adjoint::Yes {
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
    adjoint: Adjoint,
    scratch: &mut [Complex],
) {
    let b = s1 * s2;
    for k in 0..nblocks {
        left_mul_block(&mut raw[k * b..(k + 1) * b], m, s1, s2, adjoint, scratch);
    }
}

/// Right-multiply every `b`-sized block of `raw` by `M(†)`.
fn slice_rmul(
    raw: &mut [Complex],
    nblocks: usize,
    m: &DenseMat,
    s1: usize,
    s2: usize,
    adjoint: Adjoint,
    scratch: &mut [Complex],
) {
    let b = s1 * s2;
    for k in 0..nblocks {
        right_mul_block(&mut raw[k * b..(k + 1) * b], m, s1, s2, adjoint, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::catch_unwind;

    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;
    use crate::storage::timestep_at;

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

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 2, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    fn adjoint2(b: &[Complex]) -> DenseMat {
        DenseMat::from_vec(2, 2, b.to_vec()).adjoint()
    }

    fn dm(m: &DenseMat, q: usize) -> Complex {
        m.get(q / 2, q % 2)
    }

    fn blkd(a: &[Complex], b: &[Complex]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0, |e, (x, y)| e + (*x - *y).abs() * (*x - *y).abs())
    }

    fn check_left_row(
        a: &ContourGreen,
        g: &ContourGreen,
        f: &TimeFunction,
        w: f64,
        tstp: i64,
        j: i64,
    ) {
        // ret: w * f[tstp]' * ret
        let want = DenseMat::from_vec(2, 2, g.ret_raw_block(tstp, j).to_vec());
        let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
        let wantm = adjoint2(f.block(tstp)).mat_mul(&want);
        for q in 0..4 {
            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
            assert!(d < 1e-13, "ret l ({tstp},{j})");
        }
        // less: w * f[j]' * less
        let want = DenseMat::from_vec(2, 2, g.less_raw_block(j, tstp).to_vec());
        let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
        let wantm = adjoint2(f.block(j)).mat_mul(&want);
        for q in 0..4 {
            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
            assert!(d < 1e-13, "less l ({j},{tstp})");
        }
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
        a.mat.lmul(&m, Adjoint::No);
        assert_eq!(a.get_mat(0, 0, 1), Complex::new(3.0, 0.0));
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(2.0, 0.0));
        // lmul Adjoint: M† = [[1,0],[1,1]] -> (1,0) = M†(1,0)B(0,0) = 2
        set_b(&mut a);
        a.mat.lmul(&m, Adjoint::Yes);
        assert_eq!(a.get_mat(0, 1, 0), Complex::new(2.0, 0.0));
        assert_eq!(a.get_mat(0, 1, 1), Complex::new(3.0, 0.0));
        // rmul Normal: B*M -> (0,1) = B(0,0)M(0,1) = 2
        set_b(&mut a);
        a.mat.rmul(&m, Adjoint::No);
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
        a.lmul_at(2, &m, Adjoint::No);
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
        a.lmul_cf_at(MATSUBARA, &cf, Adjoint::No);
        // mat[0] = M * B: (0,0) = M(0,0)*B(0,0) = (1+i)*2
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(2.0, 2.0));
    }

    #[test]
    fn smul_with_val_adjoint() {
        // grid: 7 x 6 points
        let c = Contour::new(6, 5, 2, 2, 0.6, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut h = DenseMat::zeros(2);
        h.set(0, 0, Complex::new(-0.3, 0.0));
        h.set(1, 1, Complex::new(0.5, 0.0));
        h.set(0, 1, Complex::new(0.0, 0.2));
        h.set(1, 0, Complex::new(0.0, -0.2));
        from_ham(&mut g, 0.0, &h, 1.0, 0.1);
        let mut rng = Rng(0xABCD);
        let mut f = TimeFunction::new(6, 2, 2, Complex::ZERO);
        for t in -1..=6 {
            f.block_mut(t).copy_from_slice(&rng.block(4));
        }
        let w = 0.7f64;
        let mut fw = f.clone();
        fw.smul(Complex::new(w, 0.0));
        for tstp in -1..=6 {
            // left: A <- w * f(†) * A
            let mut a = g.clone();
            a.lmul_cf_at(tstp, &fw, Adjoint::Yes);
            let mut gslice = timestep_at(&g, if tstp == MATSUBARA { 0 } else { tstp });
            gslice.lmul_cf(&fw, Adjoint::Yes);
            match tstp {
                MATSUBARA => {
                    for m in 0..=5 {
                        let want = DenseMat::from_vec(2, 2, g.mat_raw_block(m).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.mat_raw_block(m).to_vec());
                        let wl = f
                            .block(MATSUBARA)
                            .iter()
                            .zip(f.block(MATSUBARA))
                            .map(|_| Complex::ZERO)
                            .collect::<Vec<_>>();
                        let _ = wl;
                        // w * f[0]' * G.mat
                        let fadj = adjoint2(f.block(MATSUBARA));
                        let wantm = fadj.mat_mul(&want);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13, "mat {m} {d}");
                        }
                    }
                }
                _ => {
                    for j in 0..=tstp {
                        check_left_row(&a, &g, &f, w, tstp, j);
                    }
                    for m in 0..=5 {
                        let want = DenseMat::from_vec(2, 2, g.lmix_raw_block(tstp, m).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.lmix_raw_block(tstp, m).to_vec());
                        let fadj = adjoint2(f.block(tstp));
                        let wantm = fadj.mat_mul(&want);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13);
                        }
                    }
                }
            }
            // slice agrees
            let row = timestep_at(&a, if tstp == MATSUBARA { 0 } else { tstp });
            {
                // compare only the components the distance covers
                let mut e = 0.0f64;
                if tstp == MATSUBARA {
                    for q in 0..=5 {
                        e += blkd(row.mat_raw_block(q), gslice.mat_raw_block(q));
                    }
                } else {
                    for j in 0..=tstp {
                        e += blkd(row.ret_raw_block(tstp, j), gslice.ret_raw_block(tstp, j));
                        e += blkd(row.less_raw_block(j, tstp), gslice.less_raw_block(j, tstp));
                    }
                    for m in 0..=5 {
                        e += blkd(row.lmix_raw_block(tstp, m), gslice.lmix_raw_block(tstp, m));
                    }
                }
                assert!(e <= 1e-28, "slice mismatch at tstp {tstp}: {e}");
            }
            // other time steps are untouched
            for t in -1..=6 {
                if t == tstp {
                    continue;
                }
                assert_eq!(a.distance_at(&g, t), 0.0);
            }
            // right: A <- w * A * f(†)
            let mut a = g.clone();
            a.rmul_cf_at(tstp, &fw, Adjoint::Yes);
            let mut gslice = timestep_at(&g, if tstp == MATSUBARA { 0 } else { tstp });
            gslice.rmul_cf(&fw, Adjoint::Yes);
            match tstp {
                MATSUBARA => {
                    for m in 0..=5 {
                        let want = DenseMat::from_vec(2, 2, g.mat_raw_block(m).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.mat_raw_block(m).to_vec());
                        let fadj = adjoint2(f.block(MATSUBARA));
                        let wantm = want.mat_mul(&fadj);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13);
                        }
                    }
                }
                _ => {
                    for j in 0..=tstp {
                        // ret * f[j]'
                        let want = DenseMat::from_vec(2, 2, g.ret_raw_block(tstp, j).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
                        let fadj = adjoint2(f.block(j));
                        let wantm = want.mat_mul(&fadj);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13, "ret r ({tstp},{j})");
                        }
                        // less * f[tstp]'
                        let want = DenseMat::from_vec(2, 2, g.less_raw_block(j, tstp).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
                        let fadj = adjoint2(f.block(tstp));
                        let wantm = want.mat_mul(&fadj);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13, "less r ({j},{tstp})");
                        }
                    }
                    for m in 0..=5 {
                        let want = DenseMat::from_vec(2, 2, g.lmix_raw_block(tstp, m).to_vec());
                        let got = DenseMat::from_vec(2, 2, a.lmix_raw_block(tstp, m).to_vec());
                        let fadj = adjoint2(f.block(MATSUBARA));
                        let wantm = want.mat_mul(&fadj);
                        for q in 0..4 {
                            let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                            assert!(d < 1e-13);
                        }
                    }
                }
            }
        }
        // default weight is one: smul with and without weight agree
        let mut a = g.clone();
        a.lmul_cf_at(3, &f, Adjoint::Yes);
        let mut b = g.clone();
        b.lmul_cf_at(3, &fw, Adjoint::Yes);
        // not equal (w = 0.7); check scaling relation instead on one block
        let q = (a.get_ret(3, 1, 0, 0) - b.get_ret(3, 1, 0, 0) / Complex::new(w, 0.0)).abs();
        assert!(q < 1e-13);
        // rectangular functions are rejected
        let cr = Contour::new(6, 5, 2, 3, 0.6, 1.0);
        let cfr = TimeFunction::new(6, 2, 3, Complex::ZERO);
        let _gr = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
        let r1 = {
            let mut gr2 = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
            let cfl = cfr.clone();
            catch_unwind(move || {
                gr2.lmul_cf_at(2, &cfl, Adjoint::Yes);
            })
        };
        assert!(r1.is_err());
        let r2 = {
            let mut gr3 = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
            catch_unwind(move || {
                gr3.rmul_cf_at(2, &cfr, Adjoint::Yes);
            })
        };
        assert!(r2.is_err());
    }
}
