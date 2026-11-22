//
// Project : KadanoffBaym
// File    : src/complex/types.rs
// Purpose : the Complex type: definition, constants, constructors and
//           conversions
//

/// Double-precision complex number `{ re, im }`, hand-written instead of
/// relying on a `num-complex` dependency.
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

    /// Construct from polar coordinates `(r, theta)`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Complex {
        Complex::new(r * theta.cos(), r * theta.sin())
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
