// Project : KadanoffBaym
// File    : src/types/step.rs
// Purpose : single-timestep Green's function (blueprint: types/green.jl `𝒻`)
//!
//! *Remarks : Full Contour-Ordered Green's Functions at Given Time Step `tstp`*
//!
//! In general, it can be viewed as a slice of the contour-ordered Green's
//! function at time axis. It includes four independent components.
//!
//! * ``G^{M}(\tau)``
//! * ``G^{R}(t_i \equiv tstp, t_j)``, where ``t_j \le tstp``
//! * ``G^{⌉}(t_i \equiv tstp, \tau_j)``
//! * ``G^{<}(t_i, t_j \equiv tstp)``, where ``t_i \le tstp``
//!
//! We also name them as `mat`, `ret`, `lmix`, and `less`, respectively.

use crate::types::{Contour, LessComp, LmixComp, MatComp, RetComp, Sign};
use crate::complex::Complex;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_object_sizes() {
        let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
        let f = GreenTimeStep::new(&c, 2, Sign::Bose, Complex::ZERO);
        assert_eq!(f.tstp, 2);
        assert_eq!(f.mat.nblocks(), 5);
        assert_eq!(f.ret.nblocks(), 6); // 1+2+3
        assert_eq!(f.less.nblocks(), 6);
        assert_eq!(f.lmix.nblocks(), 5); // single row
    }
}
