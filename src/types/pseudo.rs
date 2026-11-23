// Project : KadanoffBaym
// File    : src/types/pseudo.rs
// Purpose : pseudo-particle Green's functions
//
// The pseudo-particle Green's function has the same storage layout as
// the standard one (`mat`, `ret`, `lmix`, `less`). However, the real
// time triangle `ret` stores the greater component G^{>}(t, t') (with
// t >= t'), instead of G^{R} = G^{>} - G^{<}. There is no advanced
// component. `PseudoGreen` is separated from `ContourGreen`, so that a
// pseudo-particle object can not be passed to the standard Matsubara
// or left-mixing algorithms by mistake.
//
// The standard real-time start kernels (`dyson_start_ret`,
// `vie2_start_ret`, ...) are convention agnostic: the pseudo solvers
// transfer the object into a scratch `ContourGreen` once per public
// driver call (the pseudo character is purely semantic, the storage is
// identical), run all the kernels, and copy back.

//!
//! *Remarks* : *Pseudo-Particle Green's Functions*
//!
//! The pseudo-particle Green's function has the same storage layout as the
//! standard one, i.e., `mat`, `ret`, `lmix`, and `less`. However, the real
//! time triangle `ret` stores the greater component ``G^{>}(t,t')`` (with
//! ``t \ge t'``), instead of ``G^{R} = G^{>} - G^{<}``, and the Matsubara
//! and left-mixing algorithms differ.
//!
//! `PseudoGreen` is therefore separated from `ContourGreen`, so that
//! a pseudo-particle object can not be passed to the standard Matsubara
//! or left-mixing algorithms by mistake. The single-timestep views
//! reuse the standard `GreenTimeStep` (the storage is identical).
//!

use crate::complex::Complex;
use crate::types::{Contour, ContourGreen};
use crate::types::{LessComp, LmixComp, MatComp, RetComp, Sign};

/// Pseudo-particle contour-ordered Green's function `F^{ps}`. The `ret`
/// component stores the **greater** Green's function.
#[derive(Clone)]
pub struct PseudoGreen {
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

impl PseudoGreen {
    /// Standard constructor.
    pub fn new(c: &Contour, sign: Sign, v: Complex) -> PseudoGreen {
        assert!(c.size1 == c.size2, "the Green's function must be square");
        let (nt, ntau) = (c.nt as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        PseudoGreen {
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

    /// Transfer into a scratch `ContourGreen` (one clone per driver
    /// call, see the module remarks).
    pub fn to_contour(&self) -> ContourGreen {
        ContourGreen {
            sign: self.sign,
            nt: self.nt,
            ntau: self.ntau,
            size1: self.size1,
            size2: self.size2,
            mat: self.mat.clone(),
            ret: self.ret.clone(),
            lmix: self.lmix.clone(),
            less: self.less.clone(),
        }
    }

    /// Copy the components back from a scratch `ContourGreen`.
    pub(crate) fn copy_from_contour(&mut self, g: &ContourGreen) {
        self.mat = g.mat.clone();
        self.ret = g.ret.clone();
        self.lmix = g.lmix.clone();
        self.less = g.less.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    /// Exact storage equality of two pseudo greens.
    fn eq_pseudo(a: &PseudoGreen, b: &PseudoGreen) -> bool {
        a.sign == b.sign
            && a.mat.raw() == b.mat.raw()
            && a.ret.raw() == b.ret.raw()
            && a.lmix.raw() == b.lmix.raw()
            && a.less.raw() == b.less.raw()
    }

    #[test]
    fn pseudo_construction_and_equality() {
        // 6 real-time points and 7 tau points -> nt = 5, ntau = 6, 2x2
        let cc = Contour::new(5, 6, 2, 2, 0.5, 2.0);
        let g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        assert_eq!((g.size1, g.size2), (2, 2));
        assert_eq!(g.nt, 5); // 6 points
        assert_eq!(g.ntau, 6); // 7 points
        assert!(g.sign == Sign::Fermi);
        assert!(PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO).sign == Sign::Bose);
        let gv = PseudoGreen::new(&cc, Sign::Fermi, Complex::new(0.0, 0.5));
        assert!(gv.mat.block(2).iter().all(|z| *z == Complex::new(0.0, 0.5)));
        assert!(eq_pseudo(
            &gv,
            &PseudoGreen::new(&cc, Sign::Fermi, Complex::new(0.0, 0.5))
        ));
        assert!(!eq_pseudo(
            &gv,
            &PseudoGreen::new(&cc, Sign::Fermi, Complex::new(0.0, 0.4))
        ));
        // non-square matrices are rejected
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                PseudoGreen::new(
                    &Contour::new(5, 6, 2, 3, 0.5, 2.0),
                    Sign::Fermi,
                    Complex::ZERO,
                )
            }))
            .is_err()
        );
        // the storage is the same as that of the standard Green
        let f = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        assert_eq!(g.ret.raw().len(), f.ret.raw().len());
        assert_eq!(g.less.raw().len(), f.less.raw().len());
        assert_eq!(g.lmix.raw().len(), f.lmix.raw().len());
        assert_eq!(g.mat.raw().len(), f.mat.raw().len());
    }
}
