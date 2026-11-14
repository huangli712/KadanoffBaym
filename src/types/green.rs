// Project : KadanoffBaym
// File    : src/types/green.rs
// Purpose : ContourGreen (GreenTimeStep and TimeFunction live in
//           step.rs and function.rs; the component containers in
//           mat.rs, ret.rs, less.rs, lmix.rs)
//
// Blueprint: julia/src/types/{green,function,mat,ret,less,lmix}.jl.
// Julia components store their own arrays (G* owned by ℱ, g* owned by
// 𝒻); the same layout is used here. Indices are 0-based (D6):
// tstp = 0..nt, tau = 0..ntau, Matsubara via tstp = MATSUBARA (-1).

use crate::basics::layout::Layout;
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

// GreenTimeStep moved to step.rs, TimeFunction to function.rs.

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
