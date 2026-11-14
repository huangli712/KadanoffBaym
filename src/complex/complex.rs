// Project : KadanoffBaym
// File    : src/complex/complex.rs
// Purpose : double-precision complex numbers (hand-written, no crates)

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Double-precision complex number `{ re, im }` (the `Complex` of the
/// Julia blueprint, hand-written instead of a `num-complex` dependency).
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// The additive identity.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    /// The multiplicative identity.
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    /// The imaginary unit.
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    /// Construct from real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// Complex conjugate.
    #[inline]
    pub const fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus `re*re + im*im`.
    #[inline]
    pub fn abs_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `|z|`, overflow-safe scaled form.
    #[inline]
    pub fn abs(self) -> f64 {
        let (a, b) = (self.re.abs(), self.im.abs());
        if a == 0.0 && b == 0.0 {
            0.0
        } else if a >= b {
            let r = b / a;
            a * (1.0 + r * r).sqrt()
        } else {
            let r = a / b;
            b * (1.0 + r * r).sqrt()
        }
    }

    /// Argument `atan2(im, re)`.
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiplicative inverse `1/z` (scaled for overflow safety).
    #[inline]
    pub fn inv(self) -> Complex {
        let (a, b) = (self.re, self.im);
        if a.abs() >= b.abs() {
            let r = b / a;
            let d = a + b * r;
            Complex::new(1.0 / d, -r / d)
        } else {
            let r = a / b;
            let d = a * r + b;
            Complex::new(r / d, -1.0 / d)
        }
    }

    /// Complex exponential `exp(z)` (real part modulated, imaginary rotated).
    #[inline]
    pub fn exp(self) -> Complex {
        let m = self.re.exp();
        Complex::new(m * self.im.cos(), m * self.im.sin())
    }

    /// Construct from polar coordinates `(r, theta)`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Complex {
        Complex::new(r * theta.cos(), r * theta.sin())
    }

    /// Scale by a real factor.
    #[inline]
    pub fn scale(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}

impl From<f64> for Complex {
    #[inline]
    fn from(x: f64) -> Complex {
        Complex::new(x, 0.0)
    }
}

impl From<(f64, f64)> for Complex {
    #[inline]
    fn from((re, im): (f64, f64)) -> Complex {
        Complex::new(re, im)
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: Complex) -> Complex {
        self * rhs.inv()
    }
}

impl Neg for Complex {
    type Output = Complex;
    #[inline]
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Complex) {
        *self = *self + rhs;
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Complex) {
        *self = *self - rhs;
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: Complex) {
        *self = *self * rhs;
    }
}

impl DivAssign for Complex {
    #[inline]
    fn div_assign(&mut self, rhs: Complex) {
        *self = *self / rhs;
    }
}

/// `Complex + f64`, `f64 + Complex`, ... mixed real arithmetic.
impl Add<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, rhs: f64) -> Complex {
        Complex::new(self.re + rhs, self.im)
    }
}

impl Add<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self + rhs.re, rhs.im)
    }
}

impl Sub<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, rhs: f64) -> Complex {
        Complex::new(self.re - rhs, self.im)
    }
}

impl Sub<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self - rhs.re, -rhs.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: f64) -> Complex {
        self.scale(rhs)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        rhs.scale(self)
    }
}

impl Div<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: f64) -> Complex {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Complex>>(iter: I) -> Complex {
        iter.fold(Complex::ZERO, |a, b| a + b)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.im < 0.0 || (self.im == 0.0 && self.im.is_sign_negative()) {
            "-"
        } else {
            "+"
        };
        write!(f, "{}{}{}i", self.re, sign, self.im.abs())
    }
}

impl fmt::Debug for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Complex({}, {})", self.re, self.im)
    }
}

/// Assert two complex numbers are close within `tol` (absolute).
#[cfg(test)]
pub(crate) fn assert_close(actual: Complex, expected: Complex, tol: f64, what: &str) {
    let d = (actual - expected).abs();
    assert!(
        d <= tol,
        "{what}: got {actual}, expected {expected}, |diff| = {d} > {tol}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_roundtrip() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        assert_close(a + b, Complex::new(4.0, 2.0), 1e-15, "add");
        assert_close(a - b, Complex::new(2.0, 6.0), 1e-15, "sub");
        assert_close(a * b, Complex::new(11.0, -2.0), 1e-15, "mul");
        assert_close(a / b, a * b.inv(), 1e-15, "div");
        assert_close((a / b) * b, a, 1e-14, "div-mul roundtrip");
        assert_close(-a, Complex::new(-3.0, -4.0), 0.0, "neg");
    }

    #[test]
    fn mixed_real_ops() {
        let a = Complex::new(1.5, 2.5);
        assert_close(a + 1.0, Complex::new(2.5, 2.5), 0.0, "c + r");
        assert_close(1.0 + a, Complex::new(2.5, 2.5), 0.0, "r + c");
        assert_close(a - 1.0, Complex::new(0.5, 2.5), 0.0, "c - r");
        assert_close(1.0 - a, Complex::new(-0.5, -2.5), 0.0, "r - c");
        assert_close(a * 2.0, Complex::new(3.0, 5.0), 0.0, "c * r");
        assert_close(2.0 * a, Complex::new(3.0, 5.0), 0.0, "r * c");
        assert_close(a / 2.0, Complex::new(0.75, 1.25), 0.0, "c / r");
    }

    #[test]
    fn modulus_and_inverse() {
        let a = Complex::new(3.0, 4.0);
        assert_eq!(a.abs_sqr(), 25.0);
        assert_eq!(a.abs(), 5.0);
        assert_close(a.inv() * a, Complex::ONE, 1e-16, "inv * self");
        // Overflow-safety: huge components.
        let h = Complex::new(1e300, 1e300);
        assert!((h.abs() - 1.4142135623730951e300).abs() < 1e287);
        assert_close(h.inv() * h, Complex::ONE, 1e-16, "huge inv roundtrip");
        // Division by zero yields non-finite values, never panics.
        let z = Complex::ZERO;
        let q = a / z;
        assert!(!q.re.is_finite() || !q.im.is_finite());
    }

    #[test]
    fn exp_and_polar() {
        let z = Complex::new(0.3, 0.7);
        assert_close(z.exp().conj(), z.conj().exp(), 1e-15, "conj exp");
        let r = 2.0;
        let th = 0.9;
        let p = Complex::from_polar(r, th);
        assert!((p.abs() - r).abs() < 1e-15);
        assert!((p.arg() - th).abs() < 1e-15);
        assert_close(Complex::I * Complex::I, -Complex::ONE, 0.0, "i*i = -1");
        // exp(i*pi) = -1
        assert_close(
            (Complex::I * std::f64::consts::PI).exp() + Complex::ONE,
            Complex::ZERO,
            1e-15,
            "euler",
        );
    }

    #[test]
    fn display_and_from() {
        assert_eq!(Complex::from(2.0).to_string(), "2+0i");
        assert_eq!(Complex::from((1.0, -1.0)).to_string(), "1-1i");
        assert_eq!(format!("{:?}", Complex::new(0.5, 0.25)), "Complex(0.5, 0.25)");
    }
}
