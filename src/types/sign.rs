// Project : KadanoffBaym
// File    : src/types/sign.rs
// Purpose : particle statistics sign (blueprint: types/contour.jl)

/// Particle statistics sign (NESSi convention: FERMI = -1, BOSE = +1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Fermi,
    Bose,
}

impl Sign {
    /// Integer code (`-1` fermions, `+1` bosons).
    #[inline]
    pub fn as_i64(self) -> i64 {
        match self {
            Sign::Fermi => -1,
            Sign::Bose => 1,
        }
    }

    /// From an integer code.
    pub fn from_i64(s: i64) -> Sign {
        match s {
            -1 => Sign::Fermi,
            1 => Sign::Bose,
            other => panic!("Sign::from_i64: expected -1 or 1, got {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_codes() {
        assert_eq!(Sign::Fermi.as_i64(), -1);
        assert_eq!(Sign::Bose.as_i64(), 1);
        assert_eq!(Sign::from_i64(-1), Sign::Fermi);
        assert_eq!(Sign::from_i64(1), Sign::Bose);
    }
}
