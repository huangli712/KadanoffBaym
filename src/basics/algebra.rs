// Project : KadanoffBaym
// File    : src/basics/algebra.rs
// Purpose : non-in-place algebra (blueprint: julia/src/algebra/*.jl)
//
// The in-place kernels live in `ops`; these are the owned counterparts
// (`add`, `sub`, `smul` returning new objects) for every type that
// implements `Algebra`.

use crate::complex::c64::C64;
use crate::types::green::{
    ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction,
};

/// In-place kernels required by the owned algebra.
pub trait Algebra: Clone {
    /// `self += other`.
    fn incr_by(&mut self, other: &Self);
    /// `self *= alpha`.
    fn smul_c(&mut self, alpha: C64);
}

macro_rules! impl_algebra {
    ($ty:ident) => {
        impl Algebra for $ty {
            fn incr_by(&mut self, other: &Self) {
                $ty::incr_by(self, other);
            }
            fn smul_c(&mut self, alpha: C64) {
                $ty::smul(self, alpha);
            }
        }
    };
}

impl_algebra!(ContourGreen);
impl_algebra!(GreenTimeStep);
impl_algebra!(TimeFunction);
impl_algebra!(MatComp);
impl_algebra!(RetComp);
impl_algebra!(LmixComp);
impl_algebra!(LessComp);

/// `a + b` (new object).
pub fn add<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = a.clone();
    out.incr_by(b);
    out
}

/// `a - b` (new object).
pub fn sub<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = b.clone();
    out.smul_c(C64::new(-1.0, 0.0));
    out.incr_by(a);
    out
}

/// `alpha * a` (new object), complex scalar.
pub fn smul<T: Algebra>(a: &T, alpha: C64) -> T {
    let mut out = a.clone();
    out.smul_c(alpha);
    out
}

/// `alpha * a` (new object), real scalar.
pub fn smul_f64<T: Algebra>(a: &T, alpha: f64) -> T {
    smul(a, C64::new(alpha, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn green_add_sub_scale() {
        let c = Contour::new(2, 2, 1, 1, 1.0, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        a.set_mat(0, 0, 0, C64::new(1.0, 0.0));
        let mut b = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        b.set_mat(0, 0, 0, C64::new(0.0, 1.0));
        let s = add(&a, &b);
        assert_eq!(s.get_mat(0, 0, 0), C64::new(1.0, 1.0));
        let d = sub(&a, &b);
        assert_eq!(d.get_mat(0, 0, 0), C64::new(1.0, -1.0));
        let m = smul(&a, C64::new(0.0, 2.0));
        assert_eq!(m.get_mat(0, 0, 0), C64::new(0.0, 2.0));
        let m2 = smul_f64(&a, 3.0);
        assert_eq!(m2.get_mat(0, 0, 0), C64::new(3.0, 0.0));
        // originals untouched
        assert_eq!(a.get_mat(0, 0, 0), C64::new(1.0, 0.0));
        assert_eq!(b.get_mat(0, 0, 0), C64::new(0.0, 1.0));
    }

    #[test]
    fn time_function_algebra() {
        let mut a = TimeFunction::new(1, 1, 1, C64::ZERO);
        a.set(0, 0, 0, C64::new(2.0, 0.0));
        let b = add(&a, &a);
        assert_eq!(b.get(0, 0, 0), C64::new(4.0, 0.0));
        let d = sub(&a, &a);
        assert_eq!(d.get(0, 0, 0), C64::ZERO);
    }
}
