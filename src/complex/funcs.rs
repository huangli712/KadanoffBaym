//
// Project : KadanoffBaym
// File    : src/complex/funcs.rs
// Purpose : elementary functions and derived traits of Complex:
//           conjugate, modulus, argument, exponential, summation,
//           Display and Debug
//

use std::fmt::{Debug, Display, Formatter, Result};
use std::iter::Sum;

use crate::complex::Complex;

impl Complex {
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

    /// Complex exponential `exp(z)` (real part modulated, imaginary rotated).
    #[inline]
    pub fn exp(self) -> Complex {
        let m = self.re.exp();
        Complex::new(m * self.im.cos(), m * self.im.sin())
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Complex>>(iter: I) -> Complex {
        iter.fold(Complex::ZERO, |a, b| a + b)
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let sign = if self.im < 0.0 || (self.im == 0.0 && self.im.is_sign_negative()) {
            "-"
        } else {
            "+"
        };
        write!(f, "{}{}{}i", self.re, sign, self.im.abs())
    }
}

impl Debug for Complex {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Complex({}, {})", self.re, self.im)
    }
}
