//
// Project : KadanoffBaym
// File    : src/weights/rational.rs
// Purpose : exact rational arithmetic (i128) of the weight calculators
//
// Always reduced, denominator > 0; converted to `f64` only at the end
// of the weight recursions.

/// Exact rational number (i128 numerator / denominator, always reduced,
/// denominator > 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Rational {
    pub(super) num: i128,
    pub(super) den: i128,
}

impl Rational {
    pub(super) fn new(num: i128, den: i128) -> Rational {
        assert!(den != 0, "Rational: zero denominator");
        let (mut n, mut d) = (num, den);
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = Self::gcd(n.abs(), d);
        Rational {
            num: n / g,
            den: d / g,
        }
    }

    pub(super) fn from_int(n: i128) -> Rational {
        Rational { num: n, den: 1 }
    }

    pub(super) fn add(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }

    pub(super) fn sub(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den - o.num * self.den, self.den * o.den)
    }

    pub(super) fn div(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den, self.den * o.num)
    }

    pub(super) fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn gcd(a: i128, b: i128) -> i128 {
        if b == 0 { a } else { Self::gcd(b, a % b) }
    }
}
