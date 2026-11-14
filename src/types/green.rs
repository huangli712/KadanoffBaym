// Project : KadanoffBaym
// File    : src/types/green.rs
// Purpose : ContourGreen, GreenTimeStep, TimeFunction (the component
//           containers live in mat.rs, ret.rs, less.rs, lmix.rs)
//
// Blueprint: julia/src/types/{green,function,mat,ret,less,lmix}.jl.
// Julia components store their own arrays (G* owned by ℱ, g* owned by
// 𝒻); the same layout is used here. Indices are 0-based (D6):
// tstp = 0..nt, tau = 0..ntau, Matsubara via tstp = MATSUBARA (-1).

use crate::basics::layout::{Layout, MATSUBARA};
use crate::types::{Contour, LessComp, LmixComp, MatComp, RetComp, Sign};
use crate::Complex;

/// Full contour-ordered Green's function (blueprint `ℱ`): four owned
/// components on an L-shaped contour.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourGreen {
    pub sign: Sign,
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    pub mat: MatComp,
    pub ret: RetComp,
    pub lmix: LmixComp,
    pub less: LessComp,
}

impl ContourGreen {
    /// Standard constructor (blueprint `ℱ(C, v, sign)`): every element
    /// initialized to `v`.
    pub fn new(c: &Contour, sign: Sign, v: Complex) -> ContourGreen {
        let (nt, ntau) = (c.nt as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        ContourGreen {
            sign,
            nt,
            ntau,
            size1: s1,
            size2: s2,
            mat: MatComp::new(ntau, s1, s2, v),
            ret: RetComp::new(nt, s1, s2, v),
            lmix: LmixComp::new(nt, ntau, s1, s2, v),
            less: LessComp::new(nt, s1, s2, v),
        }
    }

    /// Zero-initialized constructor with default sign (FERMI).
    pub fn zeros(c: &Contour) -> ContourGreen {
        ContourGreen::new(c, Sign::Fermi, Complex::ZERO)
    }

    /// The underlying `Layout`.
    #[inline]
    pub fn layout(&self) -> Layout {
        Layout::new(self.nt, self.ntau, self.size1, self.size2)
    }

    /// Total number of stored f64 words (memory gate check).
    pub fn nwords(&self) -> usize {
        (self.mat.raw().len()
            + self.ret.raw().len()
            + self.lmix.raw().len()
            + self.less.raw().len())
            * 2
    }
}

/// Single-timestep Green's function (blueprint `𝒻`): mat (`ntau + 1`),
/// ret and less triangles up to `tstp`, and the lmix row of `tstp`.
#[derive(Clone, Debug, PartialEq)]
pub struct GreenTimeStep {
    pub sign: Sign,
    pub tstp: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    pub mat: MatComp,
    pub ret: RetComp,
    pub lmix: LmixComp,
    pub less: LessComp,
}

impl GreenTimeStep {
    /// Standard constructor (blueprint `𝒻(C, tstp, v, sign)`).
    pub fn new(c: &Contour, tstp: i64, sign: Sign, v: Complex) -> GreenTimeStep {
        assert!(
            (0..=c.nt).contains(&tstp),
            "GreenTimeStep: tstp {tstp} out of range 0..={}",
            c.nt
        );
        let (tstp, ntau) = (tstp as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        GreenTimeStep {
            sign,
            tstp,
            ntau,
            size1: s1,
            size2: s2,
            mat: MatComp::new(ntau, s1, s2, v),
            ret: RetComp::new(tstp, s1, s2, v),
            lmix: LmixComp::new(0, ntau, s1, s2, v), // single row
            less: LessComp::new(tstp, s1, s2, v),
        }
    }
}

/// Single-time function (blueprint `Cf`): one matrix block per real-time
/// step `0..=nt` plus one Matsubara slot.
///
/// Storage: `data[0..=nt]` are the time rows, `data[nt + 1]` is the
/// Matsubara value (Julia `cf[0]` <-> `get(MATSUBARA)`).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeFunction {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl TimeFunction {
    /// Constructor (blueprint `Cf(ntime, ndim1, ndim2, v)` with the -1
    /// translation): `nt >= 0`, all elements set to `v`.
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> TimeFunction {
        let block = size1 * size2;
        TimeFunction {
            nt,
            size1,
            size2,
            block,
            data: vec![v; (nt + 2) * block],
        }
    }

    /// Zero-initialized constructor.
    pub fn zeros(nt: usize, size1: usize, size2: usize) -> TimeFunction {
        TimeFunction::new(nt, size1, size2, Complex::ZERO)
    }

    /// Block index for timestep `tstp` (`0..=nt` or `MATSUBARA`).
    #[inline]
    pub fn idx(&self, tstp: i64) -> usize {
        if tstp == MATSUBARA {
            self.nt + 1
        } else {
            assert!(
                (0..=self.nt as i64).contains(&tstp),
                "TimeFunction: tstp {tstp} out of range 0..={}",
                self.nt
            );
            tstp as usize
        }
    }

    /// Element access at `(tstp, a, b)` (0-based; `MATSUBARA` for the
    /// Matsubara slot).
    #[inline]
    pub fn get(&self, tstp: i64, a: usize, b: usize) -> Complex {
        self.data[self.idx(tstp) * self.block + a * self.size2 + b]
    }

    /// Element write at `(tstp, a, b)`.
    #[inline]
    pub fn set(&mut self, tstp: i64, a: usize, b: usize, v: Complex) {
        let i = self.idx(tstp) * self.block + a * self.size2 + b;
        self.data[i] = v;
    }

    /// Borrow the block of `tstp` as a row-major slice.
    #[inline]
    pub fn block(&self, tstp: i64) -> &[Complex] {
        let k = self.idx(tstp);
        &self.data[k * self.block..(k + 1) * self.block]
    }

    /// Mutably borrow the block of `tstp`.
    #[inline]
    pub fn block_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = self.idx(tstp);
        &mut self.data[k * self.block..(k + 1) * self.block]
    }

    /// Borrow the whole raw storage (time rows then Matsubara slot).
    #[inline]
    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    /// Mutably borrow the whole raw storage.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_contour() -> Contour {
        Contour::new(3, 4, 2, 2, 1.0, 2.0)
    }

    #[test]
    fn component_sizes() {
        let c = small_contour();
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        assert_eq!(g.mat.nblocks(), 5);
        assert_eq!(g.ret.nblocks(), 10); // 1+2+3+4
        assert_eq!(g.less.nblocks(), 10);
        assert_eq!(g.lmix.nblocks(), 4 * 5);
        // Each block is 2x2 = 4 elements.
        assert_eq!(g.ret.raw().len(), 40);
        // mat 20 + ret 40 + lmix 80 + less 40 = 180 elements = 360 words
        assert_eq!(g.nwords(), 360);
    }

    #[test]
    fn triangle_index_matches_offsets() {
        let r = RetComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(r.idx(0, 0), 0);
        assert_eq!(r.idx(1, 0), 1);
        assert_eq!(r.idx(1, 1), 2);
        assert_eq!(r.idx(4, 4), 14);
        let l = LessComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(l.idx(1, 2), 4);
        assert_eq!(l.idx(0, 4), 10);
    }

    #[test]
    fn row_slices_span_the_row() {
        let mut r = RetComp::new(3, 1, 2, Complex::ZERO);
        assert_eq!(r.row(0).len(), 2);
        assert_eq!(r.row(3).len(), 8);
        r.row_mut(2)[0] = Complex::ONE;
        assert_eq!(r.get(r.idx(2, 0), 0, 0), Complex::ONE);
    }

    #[test]
    fn lmix_row_slices_span_the_row() {
        let mut l = LmixComp::new(3, 4, 1, 2, Complex::ZERO);
        assert_eq!(l.row(0).len(), 10);
        assert_eq!(l.row(3).len(), 10);
        l.row_mut(2)[0] = Complex::ONE;
        assert_eq!(l.get(l.idx(2, 0), 0, 0), Complex::ONE);
    }

    #[test]
    fn timestep_object_sizes() {
        let c = small_contour();
        let f = GreenTimeStep::new(&c, 2, Sign::Bose, Complex::ZERO);
        assert_eq!(f.tstp, 2);
        assert_eq!(f.mat.nblocks(), 5);
        assert_eq!(f.ret.nblocks(), 6); // 1+2+3
        assert_eq!(f.less.nblocks(), 6);
        assert_eq!(f.lmix.nblocks(), 5); // single row
    }

    #[test]
    fn time_function_matsubara_slot() {
        let mut cf = TimeFunction::new(3, 1, 1, Complex::ZERO);
        assert_eq!(cf.block(0).len(), 1);
        cf.set(MATSUBARA, 0, 0, Complex::new(9.0, 0.0));
        cf.set(2, 0, 0, Complex::new(2.0, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), Complex::new(9.0, 0.0));
        assert_eq!(cf.get(2, 0, 0), Complex::new(2.0, 0.0));
        assert_eq!(cf.block(MATSUBARA)[0], Complex::new(9.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn ret_rejects_upper_triangle() {
        let r = RetComp::new(2, 1, 1, Complex::ZERO);
        let _ = r.idx(0, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn lmix_rejects_bad_tau() {
        let l = LmixComp::new(2, 3, 1, 1, Complex::ZERO);
        let _ = l.idx(0, 4);
    }
}
