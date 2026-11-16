//
// Project : KadanoffBaym
// File    : src/basics/ops.rs
// Purpose : element-wise operations (memcpy/memset/zeros/incr/smul)
//
// element-wise loops live once as private slice-level kernels; the
// per-type methods are one-line delegations.

//!
//! *Remarks* :
//!
//! The following functions act on the four stored components only. Since the
//! `ret` component of `ℱᵖˢ` stores the greater component, no derived
//! quantity (such as ``G^{>} = G^{<} + G^{R}``) is evaluated here.

use crate::complex::Complex;
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

// ------------------------------------------------------------------
// component ops
// ------------------------------------------------------------------

impl MatComp {
    /// Whole-component copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &MatComp) {
        slice_memcpy(self.raw_mut(), src.raw());
    }

    /// Whole-component fill.
    pub fn memset(&mut self, v: Complex) {
        self.raw_mut().fill(v);
    }

    /// Whole-component zero.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// Whole-component accumulation `self += src`.
    pub fn incr_by(&mut self, src: &MatComp) {
        slice_incr_by(self.raw_mut(), src.raw());
    }

    /// Scaled accumulation `self += alpha * src`.
    pub fn incr_scaled(&mut self, src: &MatComp, alpha: Complex) {
        slice_incr_scaled(self.raw_mut(), src.raw(), alpha);
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        slice_smul(self.raw_mut(), a);
    }

    /// Hermitize every block in place: each block becomes
    /// `(B + B†) / 2` (blueprint `force_matsubara_hermitian!`); only
    /// square blocks are supported.
    pub fn force_matsubara_hermitian(&mut self) {
        assert!(
            self.size1 == self.size2,
            "force_matsubara_hermitian requires square blocks"
        );
        let n = self.size1;
        for q in 0..self.nblocks() {
            let blk = self.block(q).to_vec();
            for a in 0..n {
                for b in 0..n {
                    let avg = (blk[a * n + b] + blk[b * n + a].conj()) * 0.5;
                    self.set(q, a, b, avg);
                }
            }
        }
    }
}

impl RetComp {
    /// Whole-component copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &RetComp) {
        slice_memcpy(self.raw_mut(), src.raw());
    }

    /// Whole-component fill.
    pub fn memset(&mut self, v: Complex) {
        self.raw_mut().fill(v);
    }

    /// Whole-component zero.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// Whole-component accumulation `self += src`.
    pub fn incr_by(&mut self, src: &RetComp) {
        slice_incr_by(self.raw_mut(), src.raw());
    }

    /// Scaled accumulation `self += alpha * src`.
    pub fn incr_scaled(&mut self, src: &RetComp, alpha: Complex) {
        slice_incr_scaled(self.raw_mut(), src.raw(), alpha);
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        slice_smul(self.raw_mut(), a);
    }
}

impl LmixComp {
    /// Whole-component copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &LmixComp) {
        slice_memcpy(self.raw_mut(), src.raw());
    }

    /// Whole-component fill.
    pub fn memset(&mut self, v: Complex) {
        self.raw_mut().fill(v);
    }

    /// Whole-component zero.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// Whole-component accumulation `self += src`.
    pub fn incr_by(&mut self, src: &LmixComp) {
        slice_incr_by(self.raw_mut(), src.raw());
    }

    /// Scaled accumulation `self += alpha * src`.
    pub fn incr_scaled(&mut self, src: &LmixComp, alpha: Complex) {
        slice_incr_scaled(self.raw_mut(), src.raw(), alpha);
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        slice_smul(self.raw_mut(), a);
    }
}

impl LessComp {
    /// Whole-component copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &LessComp) {
        slice_memcpy(self.raw_mut(), src.raw());
    }

    /// Whole-component fill.
    pub fn memset(&mut self, v: Complex) {
        self.raw_mut().fill(v);
    }

    /// Whole-component zero.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// Whole-component accumulation `self += src`.
    pub fn incr_by(&mut self, src: &LessComp) {
        slice_incr_by(self.raw_mut(), src.raw());
    }

    /// Scaled accumulation `self += alpha * src`.
    pub fn incr_scaled(&mut self, src: &LessComp, alpha: Complex) {
        slice_incr_scaled(self.raw_mut(), src.raw(), alpha);
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        slice_smul(self.raw_mut(), a);
    }
}

impl TimeFunction {
    /// Copy `self <- src`.
    pub fn memcpy_from(&mut self, src: &TimeFunction) {
        slice_memcpy(self.raw_mut(), src.raw());
    }

    /// Fill every element (including the Matsubara slot).
    pub fn memset(&mut self, v: Complex) {
        self.raw_mut().fill(v);
    }

    /// Zero every element.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// `self += src`.
    pub fn incr_by(&mut self, src: &TimeFunction) {
        slice_incr_by(self.raw_mut(), src.raw());
    }

    /// Scale by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        slice_smul(self.raw_mut(), a);
    }
}

// ------------------------------------------------------------------
// green-level ops
// ------------------------------------------------------------------

impl ContourGreen {
    /// Copy all components `self <- src` (grids must match).
    pub fn memcpy_from(&mut self, src: &ContourGreen) {
        self.mat.memcpy_from(&src.mat);
        self.ret.memcpy_from(&src.ret);
        self.lmix.memcpy_from(&src.lmix);
        self.less.memcpy_from(&src.less);
    }

    /// Fill all components with `v`.
    pub fn memset(&mut self, v: Complex) {
        self.mat.memset(v);
        self.ret.memset(v);
        self.lmix.memset(v);
        self.less.memset(v);
    }

    /// Zero all components.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// `self += src` component-wise.
    pub fn incr_by(&mut self, src: &ContourGreen) {
        self.mat.incr_by(&src.mat);
        self.ret.incr_by(&src.ret);
        self.lmix.incr_by(&src.lmix);
        self.less.incr_by(&src.less);
    }

    /// `self += alpha * src` component-wise.
    pub fn incr_scaled(&mut self, src: &ContourGreen, alpha: Complex) {
        self.mat.incr_scaled(&src.mat, alpha);
        self.ret.incr_scaled(&src.ret, alpha);
        self.lmix.incr_scaled(&src.lmix, alpha);
        self.less.incr_scaled(&src.less, alpha);
    }

    /// Scale all components by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        self.mat.smul(a);
        self.ret.smul(a);
        self.lmix.smul(a);
        self.less.smul(a);
    }

    /// Scale all components by a real scalar.
    pub fn smul_f64(&mut self, a: f64) {
        self.smul(Complex::new(a, 0.0));
    }

    // ---- row-scoped at one time step ----

    /// Copy the components at `tstp` (`MATSUBARA` = mat) from `src`.
    pub fn memcpy_from_at(&mut self, tstp: i64, src: &ContourGreen) {
        if tstp == MATSUBARA {
            self.mat.memcpy_from(&src.mat);
            return;
        }
        self.check_tstp(tstp);
        self.ret.row_mut(tstp).copy_from_slice(src.ret.row(tstp));
        self.lmix.row_mut(tstp).copy_from_slice(src.lmix.row(tstp));
        self.less.row_mut(tstp).copy_from_slice(src.less.row(tstp));
    }

    /// Fill the components at `tstp` with `v`.
    pub fn memset_at(&mut self, tstp: i64, v: Complex) {
        if tstp == MATSUBARA {
            self.mat.memset(v);
            return;
        }
        self.check_tstp(tstp);
        self.ret.row_mut(tstp).fill(v);
        self.lmix.row_mut(tstp).fill(v);
        self.less.row_mut(tstp).fill(v);
    }

    /// Zero the components at `tstp`.
    pub fn set_zero_at(&mut self, tstp: i64) {
        self.memset_at(tstp, Complex::ZERO);
    }

    /// `self[components at tstp] += src[...]`.
    // Retained API: currently unused, kept pending a removal decision.
    #[allow(dead_code)]
    fn incr_by_at(&mut self, tstp: i64, src: &ContourGreen) {
        if tstp == MATSUBARA {
            self.mat.incr_by(&src.mat);
            return;
        }
        self.check_tstp(tstp);
        for (d, s) in self.ret.row_mut(tstp).iter_mut().zip(src.ret.row(tstp)) {
            *d += *s;
        }
        for (d, s) in self.lmix.row_mut(tstp).iter_mut().zip(src.lmix.row(tstp)) {
            *d += *s;
        }
        for (d, s) in self.less.row_mut(tstp).iter_mut().zip(src.less.row(tstp)) {
            *d += *s;
        }
    }

    /// `self[components at tstp] += alpha * src[...]`.
    pub fn incr_scaled_at(&mut self, tstp: i64, src: &ContourGreen, alpha: Complex) {
        if tstp == MATSUBARA {
            self.mat.incr_scaled(&src.mat, alpha);
            return;
        }
        self.check_tstp(tstp);
        for (d, s) in self.ret.row_mut(tstp).iter_mut().zip(src.ret.row(tstp)) {
            *d += *s * alpha;
        }
        for (d, s) in self.lmix.row_mut(tstp).iter_mut().zip(src.lmix.row(tstp)) {
            *d += *s * alpha;
        }
        for (d, s) in self.less.row_mut(tstp).iter_mut().zip(src.less.row(tstp)) {
            *d += *s * alpha;
        }
    }

    /// `G[row tstp] += w * f` (blueprint `incr!(G, g, tstp, w)`):
    /// row-scoped accumulation of a single-timestep object; at
    /// `MATSUBARA` only the mat component is updated.
    pub fn incr_by_step(&mut self, f: &GreenTimeStep, tstp: i64, w: Complex) {
        if tstp == MATSUBARA {
            for q in 0..=(self.ntau as i64) {
                for (d, s) in self.mat_raw_block_mut(q).iter_mut().zip(f.mat_raw_block(q)) {
                    *d += w * *s;
                }
            }
            return;
        }
        assert!((0..=self.nt as i64).contains(&tstp));
        for j in 0..=tstp {
            for (d, s) in self
                .ret_raw_block_mut(tstp, j)
                .iter_mut()
                .zip(f.ret_raw_block(tstp, j))
            {
                *d += w * *s;
            }
            for (d, s) in self
                .less_raw_block_mut(j, tstp)
                .iter_mut()
                .zip(f.less_raw_block(j, tstp))
            {
                *d += w * *s;
            }
        }
        for m in 0..=(self.ntau as i64) {
            for (d, s) in self
                .lmix_raw_block_mut(tstp, m)
                .iter_mut()
                .zip(f.lmix_raw_block(tstp, m))
            {
                *d += w * *s;
            }
        }
    }

    /// Scale the components at `tstp` by a complex scalar.
    pub fn smul_at(&mut self, tstp: i64, a: Complex) {
        if tstp == MATSUBARA {
            self.mat.smul(a);
            return;
        }
        self.check_tstp(tstp);
        for z in self.ret.row_mut(tstp) {
            *z *= a;
        }
        for z in self.lmix.row_mut(tstp) {
            *z *= a;
        }
        for z in self.less.row_mut(tstp) {
            *z *= a;
        }
    }

    /// Range check of a real-time step shared by the `_at` operations.
    #[inline]
    pub(crate) fn check_tstp(&self, tstp: i64) {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "tstp {tstp} out of range 0..={}",
            self.nt
        );
    }
}

impl GreenTimeStep {
    /// Copy all components `self <- src` (grids must match).
    pub fn memcpy_from(&mut self, src: &GreenTimeStep) {
        self.mat.memcpy_from(&src.mat);
        self.ret.memcpy_from(&src.ret);
        self.lmix.memcpy_from(&src.lmix);
        self.less.memcpy_from(&src.less);
    }

    /// Fill all components with `v`.
    pub fn memset(&mut self, v: Complex) {
        self.mat.memset(v);
        self.ret.memset(v);
        self.lmix.memset(v);
        self.less.memset(v);
    }

    /// Zero all components.
    pub fn set_zero(&mut self) {
        self.memset(Complex::ZERO);
    }

    /// `self += src` component-wise.
    pub fn incr_by(&mut self, src: &GreenTimeStep) {
        self.mat.incr_by(&src.mat);
        self.ret.incr_by(&src.ret);
        self.lmix.incr_by(&src.lmix);
        self.less.incr_by(&src.less);
    }

    /// `self += alpha * src` component-wise.
    pub fn incr_scaled(&mut self, src: &GreenTimeStep, alpha: Complex) {
        self.mat.incr_scaled(&src.mat, alpha);
        self.ret.incr_scaled(&src.ret, alpha);
        self.lmix.incr_scaled(&src.lmix, alpha);
        self.less.incr_scaled(&src.less, alpha);
    }

    /// Scale all components by a complex scalar.
    pub fn smul(&mut self, a: Complex) {
        self.mat.smul(a);
        self.ret.smul(a);
        self.lmix.smul(a);
        self.less.smul(a);
    }

    /// Scale all components by a real scalar.
    pub fn smul_f64(&mut self, a: f64) {
        self.smul(Complex::new(a, 0.0));
    }
}

// ------------------------------------------------------------------
// t = 0 initialization from the Matsubara component
// ------------------------------------------------------------------

/// Initialize the `t = 0` row (ret/les/lmix) of a green from its
/// Matsubara component (blueprint NESSi `set_t0_from_mat(herm_matrix)`):
/// `lmix(0,m) = i*sig*mat(ntau-m)`, `less(0,0) = lmix(0,0)`, and
/// `ret(0,0) = i*mat(0) - less(0,0)`.
pub fn set_t0_from_mat(g: &mut ContourGreen) {
    let ntau = g.ntau;
    let s = Complex::new(0.0, g.sign.as_i64() as f64);
    let b = g.size1 * g.size2;
    let row = g.lmix.row_mut(0);
    for (m, chunk) in row.chunks_exact_mut(b).enumerate() {
        let src = g.mat.block(ntau - m).to_vec();
        for (d, x) in chunk.iter_mut().zip(src) {
            *d = s * x;
        }
    }
    let l00 = g.lmix.row(0)[..b].to_vec();
    g.less.raw_block_at_mut(0, 0).copy_from_slice(&l00);
    let m0 = g.mat.block(0).to_vec();
    for (d, (x, l)) in g
        .ret
        .raw_block_at_mut(0, 0)
        .iter_mut()
        .zip(m0.iter().zip(&l00))
    {
        *d = Complex::I * *x - *l;
    }
}

// ------------------------------------------------------------------
// slice-level kernels (each loop written once)
// ------------------------------------------------------------------

/// `dst <- src` with a length check.
fn slice_memcpy(dst: &mut [Complex], src: &[Complex]) {
    assert_eq!(dst.len(), src.len(), "memcpy: length mismatch");
    dst.copy_from_slice(src);
}

/// `dst += src` element-wise.
fn slice_incr_by(dst: &mut [Complex], src: &[Complex]) {
    assert_eq!(dst.len(), src.len(), "incr: length mismatch");
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// `dst += alpha * src` element-wise.
fn slice_incr_scaled(dst: &mut [Complex], src: &[Complex], alpha: Complex) {
    assert_eq!(dst.len(), src.len(), "incr: length mismatch");
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s * alpha;
    }
}

/// `dst *= a` element-wise.
fn slice_smul(dst: &mut [Complex], a: Complex) {
    for z in dst.iter_mut() {
        *z *= a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::catch_unwind;

    use crate::types::{Contour, Sign};

    use crate::basics::timestep_at;

    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }

        fn block(&mut self, nel: usize) -> Vec<Complex> {
            (0..nel)
                .map(|_| Complex::new(self.next_f64(), self.next_f64()))
                .collect()
        }
    }

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 2, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    fn fill(g: &mut ContourGreen) {
        for n in 0..=2 {
            g.set_mat(n, 0, 0, Complex::new(n as f64, 1.0));
        }
        g.set_ret(2, 1, 0, 1, Complex::new(3.0, 0.0));
        g.set_lmix(1, 2, 1, 0, Complex::new(4.0, 0.0));
        g.set_less(0, 1, 0, 0, Complex::new(5.0, 0.0));
    }

    #[test]
    fn whole_green_ops() {
        let mut a = mk(2);
        fill(&mut a);
        let mut b = mk(2);
        b.memcpy_from(&a);
        assert_eq!(b.get_mat(2, 0, 0), Complex::new(2.0, 1.0));
        assert_eq!(b.get_ret(2, 1, 0, 1), Complex::new(3.0, 0.0));
        b.incr_by(&a);
        assert_eq!(b.get_mat(2, 0, 0), Complex::new(4.0, 2.0));
        b.smul(Complex::new(0.0, 1.0));
        assert_eq!(b.get_mat(2, 0, 0), Complex::new(-2.0, 4.0));
        b.set_zero();
        assert_eq!(b.get_less(0, 1, 0, 0), Complex::ZERO);
    }

    #[test]
    fn timestep_scoped_ops() {
        let mut a = mk(2);
        fill(&mut a);
        let mut b = mk(2);
        b.memcpy_from_at(2, &a);
        assert_eq!(b.get_ret(2, 1, 0, 1), Complex::new(3.0, 0.0));
        assert_eq!(b.get_mat(2, 0, 0), Complex::ZERO); // mat untouched
        b.set_zero_at(2);
        assert_eq!(b.get_ret(2, 1, 0, 1), Complex::ZERO);
        // Matsubara scope
        let mut c = mk(2);
        c.memcpy_from_at(MATSUBARA, &a);
        assert_eq!(c.get_mat(1, 0, 0), Complex::new(1.0, 1.0));
        assert_eq!(c.get_ret(2, 1, 0, 1), Complex::ZERO);
        // incr and smul at one step
        let mut d = mk(2);
        d.incr_by_at(2, &a);
        d.incr_by_at(2, &a);
        assert_eq!(d.get_ret(2, 1, 0, 1), Complex::new(6.0, 0.0));
        d.smul_at(2, Complex::new(0.5, 0.0));
        assert_eq!(d.get_ret(2, 1, 0, 1), Complex::new(3.0, 0.0));
    }

    #[test]
    fn set_t0_from_mat_standard() {
        let mut g = mk(2); // nt = 2, ntau = 2, 2x2, Fermi (sig = -1)
        for q in 0..=2 {
            for a in 0..2 {
                for b in 0..2 {
                    g.set_mat(q, a, b, Complex::new(q as f64 + a as f64, b as f64 + 1.0));
                }
            }
        }
        set_t0_from_mat(&mut g);
        let s = Complex::new(0.0, -1.0);
        for m in 0..=2i64 {
            // mat(2-m)[0,0] = (2-m, 1)
            let want = s * Complex::new((2 - m) as f64, 1.0);
            assert_eq!(g.get_lmix(0, m, 0, 0), want);
        }
        let l00 = g.get_lmix(0, 0, 0, 0);
        assert_eq!(g.get_less(0, 0, 0, 0), l00);
        assert_eq!(g.get_ret(0, 0, 0, 0), Complex::I * g.get_mat(0, 0, 0) - l00);
    }

    #[test]
    fn set_t0_from_mat_free_fn_dispatches() {
        // free-function form on a single set block
        let mut g = mk(2);
        g.set_mat(1, 0, 0, Complex::new(1.0, 2.0));
        set_t0_from_mat(&mut g);
        assert_eq!(
            g.get_lmix(0, 1, 0, 0),
            Complex::new(0.0, -1.0) * Complex::new(1.0, 2.0)
        );
    }

    #[test]
    fn time_function_ops() {
        let mut cf = TimeFunction::new(2, 1, 1, Complex::ZERO);
        cf.set(1, 0, 0, Complex::new(1.0, 0.0));
        cf.set(MATSUBARA, 0, 0, Complex::new(2.0, 0.0));
        let mut d = TimeFunction::new(2, 1, 1, Complex::ZERO);
        d.memcpy_from(&cf);
        assert_eq!(d.get(1, 0, 0), Complex::ONE);
        d.incr_by(&cf);
        d.smul(Complex::new(2.0, 0.0));
        assert_eq!(d.get(1, 0, 0), Complex::new(4.0, 0.0));
        assert_eq!(d.get(MATSUBARA, 0, 0), Complex::new(8.0, 0.0));
        d.set_zero();
        assert_eq!(d.get(MATSUBARA, 0, 0), Complex::ZERO);
    }

    #[test]
    fn force_matsubara_hermitian_test() {
        let c = Contour::new(4, 5, 2, 2, 0.4, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut rng = Rng(0xF00D);
        for m in 0..=5 {
            g.mat_raw_block_mut(m).copy_from_slice(&rng.block(4));
        }
        for i in 0..=4 {
            for j in 0..=i {
                g.ret_raw_block_mut(i, j).copy_from_slice(&rng.block(4));
            }
        }
        let gref = g.clone();
        g.mat.force_matsubara_hermitian();
        for m in 0..=5 {
            // hermitian
            let blk = g.mat_raw_block(m).to_vec();
            let mut e = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    e += (blk[a * 2 + b] - blk[b * 2 + a].conj()).abs();
                }
            }
            assert!(e <= 1e-14);
            // equals (ref + ref†)/2
            let rf = gref.mat_raw_block(m);
            for a in 0..2 {
                for b in 0..2 {
                    let want = (rf[a * 2 + b] + rf[b * 2 + a].conj()) * 0.5;
                    assert!((blk[a * 2 + b] - want).abs() < 1e-15);
                }
            }
        }
        for t in 0..=4 {
            assert_eq!(g.distance_at(&gref, t), 0.0);
        }
        // timestep variant (equilibrium slice)
        {
            let mut s = timestep_at(&gref, 0);
            s.mat.force_matsubara_hermitian();
            for m in 0..=5 {
                let blk = s.mat_raw_block(m).to_vec();
                let mut e = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        e += (blk[a * 2 + b] - blk[b * 2 + a].conj()).abs();
                    }
                }
                assert!(e <= 1e-14);
                assert_eq!(blk, g.mat_raw_block(m).to_vec());
            }
        }
        // rectangular rejection
        let cr = Contour::new(4, 5, 2, 3, 0.4, 1.0);
        let mut gr = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
        let r = catch_unwind(move || {
            gr.mat.force_matsubara_hermitian();
        });
        assert!(r.is_err());
    }
}
