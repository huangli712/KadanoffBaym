// Project : KadanoffBaym
// File    : src/c64.rs
// Purpose : double-precision complex numbers (hand-written, no crates)

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Double-precision complex number `{ re, im }`, the `C64` of the Julia
/// blueprint (hand-written instead of a `num-complex` dependency).
#[derive(Clone, Copy, PartialEq, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    /// The additive identity.
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    /// The multiplicative identity.
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    /// The imaginary unit.
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    /// Construct from real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    /// Complex conjugate.
    #[inline]
    pub const fn conj(self) -> C64 {
        C64 {
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
    pub fn inv(self) -> C64 {
        let (a, b) = (self.re, self.im);
        if a.abs() >= b.abs() {
            let r = b / a;
            let d = a + b * r;
            C64::new(1.0 / d, -r / d)
        } else {
            let r = a / b;
            let d = a * r + b;
            C64::new(r / d, -1.0 / d)
        }
    }

    /// Complex exponential `exp(z)` (real part modulated, imaginary rotated).
    #[inline]
    pub fn exp(self) -> C64 {
        let m = self.re.exp();
        C64::new(m * self.im.cos(), m * self.im.sin())
    }

    /// Construct from polar coordinates `(r, theta)`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> C64 {
        C64::new(r * theta.cos(), r * theta.sin())
    }

    /// Scale by a real factor.
    #[inline]
    pub fn scale(self, s: f64) -> C64 {
        C64::new(self.re * s, self.im * s)
    }
}

impl From<f64> for C64 {
    #[inline]
    fn from(x: f64) -> C64 {
        C64::new(x, 0.0)
    }
}

impl From<(f64, f64)> for C64 {
    #[inline]
    fn from((re, im): (f64, f64)) -> C64 {
        C64::new(re, im)
    }
}

impl Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, rhs: C64) -> C64 {
        C64::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, rhs: C64) -> C64 {
        C64::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, rhs: C64) -> C64 {
        C64::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for C64 {
    type Output = C64;
    #[inline]
    fn div(self, rhs: C64) -> C64 {
        self * rhs.inv()
    }
}

impl Neg for C64 {
    type Output = C64;
    #[inline]
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

impl AddAssign for C64 {
    #[inline]
    fn add_assign(&mut self, rhs: C64) {
        *self = *self + rhs;
    }
}

impl SubAssign for C64 {
    #[inline]
    fn sub_assign(&mut self, rhs: C64) {
        *self = *self - rhs;
    }
}

impl MulAssign for C64 {
    #[inline]
    fn mul_assign(&mut self, rhs: C64) {
        *self = *self * rhs;
    }
}

impl DivAssign for C64 {
    #[inline]
    fn div_assign(&mut self, rhs: C64) {
        *self = *self / rhs;
    }
}

/// `C64 + f64`, `f64 + C64`, ... mixed real arithmetic.
impl Add<f64> for C64 {
    type Output = C64;
    #[inline]
    fn add(self, rhs: f64) -> C64 {
        C64::new(self.re + rhs, self.im)
    }
}

impl Add<C64> for f64 {
    type Output = C64;
    #[inline]
    fn add(self, rhs: C64) -> C64 {
        C64::new(self + rhs.re, rhs.im)
    }
}

impl Sub<f64> for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, rhs: f64) -> C64 {
        C64::new(self.re - rhs, self.im)
    }
}

impl Sub<C64> for f64 {
    type Output = C64;
    #[inline]
    fn sub(self, rhs: C64) -> C64 {
        C64::new(self - rhs.re, -rhs.im)
    }
}

impl Mul<f64> for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, rhs: f64) -> C64 {
        self.scale(rhs)
    }
}

impl Mul<C64> for f64 {
    type Output = C64;
    #[inline]
    fn mul(self, rhs: C64) -> C64 {
        rhs.scale(self)
    }
}

impl Div<f64> for C64 {
    type Output = C64;
    #[inline]
    fn div(self, rhs: f64) -> C64 {
        C64::new(self.re / rhs, self.im / rhs)
    }
}

impl Sum for C64 {
    fn sum<I: Iterator<Item = C64>>(iter: I) -> C64 {
        iter.fold(C64::ZERO, |a, b| a + b)
    }
}

impl fmt::Display for C64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.im < 0.0 || (self.im == 0.0 && self.im.is_sign_negative()) {
            "-"
        } else {
            "+"
        };
        write!(f, "{}{}{}i", self.re, sign, self.im.abs())
    }
}

impl fmt::Debug for C64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C64({}, {})", self.re, self.im)
    }
}

/// Assert two complex numbers are close within `tol` (absolute).
#[cfg(test)]
pub(crate) fn assert_close(actual: C64, expected: C64, tol: f64, what: &str) {
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
        let a = C64::new(3.0, 4.0);
        let b = C64::new(1.0, -2.0);
        assert_close(a + b, C64::new(4.0, 2.0), 1e-15, "add");
        assert_close(a - b, C64::new(2.0, 6.0), 1e-15, "sub");
        assert_close(a * b, C64::new(11.0, -2.0), 1e-15, "mul");
        assert_close(a / b, a * b.inv(), 1e-15, "div");
        assert_close((a / b) * b, a, 1e-14, "div-mul roundtrip");
        assert_close(-a, C64::new(-3.0, -4.0), 0.0, "neg");
    }

    #[test]
    fn mixed_real_ops() {
        let a = C64::new(1.5, 2.5);
        assert_close(a + 1.0, C64::new(2.5, 2.5), 0.0, "c + r");
        assert_close(1.0 + a, C64::new(2.5, 2.5), 0.0, "r + c");
        assert_close(a - 1.0, C64::new(0.5, 2.5), 0.0, "c - r");
        assert_close(1.0 - a, C64::new(-0.5, -2.5), 0.0, "r - c");
        assert_close(a * 2.0, C64::new(3.0, 5.0), 0.0, "c * r");
        assert_close(2.0 * a, C64::new(3.0, 5.0), 0.0, "r * c");
        assert_close(a / 2.0, C64::new(0.75, 1.25), 0.0, "c / r");
    }

    #[test]
    fn modulus_and_inverse() {
        let a = C64::new(3.0, 4.0);
        assert_eq!(a.abs_sqr(), 25.0);
        assert_eq!(a.abs(), 5.0);
        assert_close(a.inv() * a, C64::ONE, 1e-16, "inv * self");
        // Overflow-safety: huge components.
        let h = C64::new(1e300, 1e300);
        assert!((h.abs() - 1.4142135623730951e300).abs() < 1e287);
        assert_close(h.inv() * h, C64::ONE, 1e-16, "huge inv roundtrip");
        // Division by zero yields non-finite values, never panics.
        let z = C64::ZERO;
        let q = a / z;
        assert!(!q.re.is_finite() || !q.im.is_finite());
    }

    #[test]
    fn exp_and_polar() {
        let z = C64::new(0.3, 0.7);
        assert_close(z.exp().conj(), z.conj().exp(), 1e-15, "conj exp");
        let r = 2.0;
        let th = 0.9;
        let p = C64::from_polar(r, th);
        assert!((p.abs() - r).abs() < 1e-15);
        assert!((p.arg() - th).abs() < 1e-15);
        assert_close(C64::I * C64::I, -C64::ONE, 0.0, "i*i = -1");
        // exp(i*pi) = -1
        assert_close(
            (C64::I * std::f64::consts::PI).exp() + C64::ONE,
            C64::ZERO,
            1e-15,
            "euler",
        );
    }

    #[test]
    fn display_and_from() {
        assert_eq!(C64::from(2.0).to_string(), "2+0i");
        assert_eq!(C64::from((1.0, -1.0)).to_string(), "1-1i");
        assert_eq!(format!("{:?}", C64::new(0.5, 0.25)), "C64(0.5, 0.25)");
    }
}
