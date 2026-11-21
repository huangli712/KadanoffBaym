// Project : KadanoffBaym
// File    : src/types/step.rs
// Purpose : single-timestep Green's function

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

use crate::complex::Complex;
use crate::types::{Contour, LessComp, LmixComp, MatComp, RetComp, Sign};

/// Single-timestep Green's function : mat (`ntau + 1`),
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
    /// Standard constructor.
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

    use crate::types::ContourGreen;

    use crate::storage::timestep_at;

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

    #[test]
    fn timestep_struct_constructors() {
        // v1..v3 equal at tstp = 200, e5/e6 equal at 0, and the two
        // groups differ
        let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
        let v1 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::ZERO);
        let v2 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::ZERO);
        let c_direct = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
        let v3 = GreenTimeStep::new(&c_direct, 200, Sign::Fermi, Complex::ZERO);
        assert!(v1.equal(&v2));
        assert!(v1.equal(&v3));
        let e5 = GreenTimeStep::new(&c, 0, Sign::Fermi, Complex::ZERO);
        let e6 = GreenTimeStep::new(&c, 0, Sign::Fermi, Complex::ZERO);
        assert!(e5.equal(&e6));
        assert!(!v1.equal(&e5));
    }

    #[test]
    fn timestep_properties() {
        // 1-based tstp = 101 is the 0-based row tstp = 100
        let c = Contour::new(100, 50, 2, 3, 5.0, 4.0);
        let f1 = GreenTimeStep::new(&c, 100, Sign::Fermi, Complex::ZERO);
        let f2 = GreenTimeStep::new(&c, 100, Sign::Fermi, Complex::ZERO);
        assert_eq!(f1.sign, Sign::Fermi);
        assert_eq!(f1.tstp, 100);
        assert_eq!(f1.ntau + 1, 51);
        assert_eq!((f1.size1, f1.size2), (2, 3));
        assert_ne!(f1.size1, f1.size2);
        assert!(f1.is_compatible(&f2));
        // timestep of a zero green agrees with a zero timestep object
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let row = timestep_at(&g, 100);
        assert!(row.equal(&f1));
        // derived views of the timestep read its own storage
        let mut f = GreenTimeStep::new(&c, 100, Sign::Fermi, Complex::ZERO);
        f.set_mat(3, 1, 2, Complex::new(0.5, 0.0));
        assert_eq!(f.get_matminus(50 - 3, 1, 2), Complex::new(-0.5, 0.0));
        f.set_ret(100, 7, 0, 2, Complex::new(1.0, 2.0));
        let adv = f.adv_block(7, 100);
        assert_eq!(adv.get(2, 0), Complex::new(1.0, -2.0));
    }
}
