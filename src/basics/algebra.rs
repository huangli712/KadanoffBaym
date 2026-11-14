// Project : KadanoffBaym
// File    : src/basics/algebra.rs
// Purpose : non-in-place algebra (blueprint: julia/src/algebra/*.jl)
//
// The in-place kernels live in `ops`; these are the owned counterparts
// (`add`, `sub`, `smul` returning new objects) for every type that
// implements `Algebra`.

use crate::complex::Complex;
use crate::types::green::{
    ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction,
};

/// In-place kernels required by the owned algebra.
pub trait Algebra: Clone {
    /// `self += other`.
    fn incr_by(&mut self, other: &Self);
    /// `self *= alpha`.
    fn smul_c(&mut self, alpha: Complex);
}

impl Algebra for ContourGreen {
    fn incr_by(&mut self, other: &Self) {
        ContourGreen::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        ContourGreen::smul(self, alpha);
    }
}

impl Algebra for GreenTimeStep {
    fn incr_by(&mut self, other: &Self) {
        GreenTimeStep::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        GreenTimeStep::smul(self, alpha);
    }
}

impl Algebra for TimeFunction {
    fn incr_by(&mut self, other: &Self) {
        TimeFunction::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        TimeFunction::smul(self, alpha);
    }
}

impl Algebra for MatComp {
    fn incr_by(&mut self, other: &Self) {
        MatComp::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        MatComp::smul(self, alpha);
    }
}

impl Algebra for RetComp {
    fn incr_by(&mut self, other: &Self) {
        RetComp::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        RetComp::smul(self, alpha);
    }
}

impl Algebra for LmixComp {
    fn incr_by(&mut self, other: &Self) {
        LmixComp::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        LmixComp::smul(self, alpha);
    }
}

impl Algebra for LessComp {
    fn incr_by(&mut self, other: &Self) {
        LessComp::incr_by(self, other);
    }
    fn smul_c(&mut self, alpha: Complex) {
        LessComp::smul(self, alpha);
    }
}

/// `a + b` (new object).
pub fn add<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = a.clone();
    out.incr_by(b);
    out
}

/// `a - b` (new object).
pub fn sub<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = b.clone();
    out.smul_c(Complex::new(-1.0, 0.0));
    out.incr_by(a);
    out
}

/// `alpha * a` (new object), complex scalar.
pub fn smul<T: Algebra>(a: &T, alpha: Complex) -> T {
    let mut out = a.clone();
    out.smul_c(alpha);
    out
}

/// `alpha * a` (new object), real scalar.
pub fn smul_f64<T: Algebra>(a: &T, alpha: f64) -> T {
    smul(a, Complex::new(alpha, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn green_add_sub_scale() {
        let c = Contour::new(2, 2, 1, 1, 1.0, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        a.set_mat(0, 0, 0, Complex::new(1.0, 0.0));
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        b.set_mat(0, 0, 0, Complex::new(0.0, 1.0));
        let s = add(&a, &b);
        assert_eq!(s.get_mat(0, 0, 0), Complex::new(1.0, 1.0));
        let d = sub(&a, &b);
        assert_eq!(d.get_mat(0, 0, 0), Complex::new(1.0, -1.0));
        let m = smul(&a, Complex::new(0.0, 2.0));
        assert_eq!(m.get_mat(0, 0, 0), Complex::new(0.0, 2.0));
        let m2 = smul_f64(&a, 3.0);
        assert_eq!(m2.get_mat(0, 0, 0), Complex::new(3.0, 0.0));
        // originals untouched
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(1.0, 0.0));
        assert_eq!(b.get_mat(0, 0, 0), Complex::new(0.0, 1.0));
    }

    #[test]
    fn time_function_algebra() {
        let mut a = TimeFunction::new(1, 1, 1, Complex::ZERO);
        a.set(0, 0, 0, Complex::new(2.0, 0.0));
        let b = add(&a, &a);
        assert_eq!(b.get(0, 0, 0), Complex::new(4.0, 0.0));
        let d = sub(&a, &a);
        assert_eq!(d.get(0, 0, 0), Complex::ZERO);
    }
}
