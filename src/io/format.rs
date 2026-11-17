// Project : KadanoffBaym
// File    : src/io/format.rs
// Purpose : C-style formatting shared by the writers (%4i, %16.12f,
//           %.16e, %.17g)

/// C-style `%4i` (width 4, space padded).
pub(crate) fn fmt_i4(x: i64) -> String {
    format!("{:4}", x)
}

/// C-style `%16.12f` (width 16, precision 12, space padded).
pub(crate) fn fmt_f12(x: f64) -> String {
    format!("{:16.12}", x)
}

/// C-style `%.16e` (scientific notation, 16 fractional digits).
pub(crate) fn fmt_e16(x: f64) -> String {
    format!("{x:.16e}")
}

/// C `%.17g` formatting of a float: the shortest representation
/// with 17 significant digits, fixed notation for exponents in
/// `-4..17`, scientific notation otherwise (matches the NESSi text
/// writers digit for digit).
pub(crate) fn fmt_g17(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    let ax = x.abs();
    let exp = ax.log10().floor() as i64;

    if !(-4..17).contains(&exp) {
        // scientific with 17 significant digits
        let raw = format!("{:.16e}", x);
        let (mant, e) = raw.split_once('e').unwrap();
        let mant = if mant.contains('.') {
            mant.trim_end_matches('0').trim_end_matches('.')
        } else {
            mant
        };
        let ev: i64 = e.parse().unwrap();
        let sign = if ev < 0 { '-' } else { '+' };
        let ea = ev.abs();
        format!("{mant}e{sign}{ea:02}")
    } else {
        // fixed with the remaining decimals; only the fractional
        // portion is trimmed (C `%g` never strips integer zeros —
        // without the guard `1e16` would come out as "1")
        let prec = (16 - exp).max(0) as usize;
        let raw = format!("{x:.prec$}");
        if raw.contains('.') {
            raw.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_i4_width_and_padding() {
        assert_eq!(fmt_i4(5), "   5");
        assert_eq!(fmt_i4(-123), "-123");
        assert_eq!(fmt_i4(0), "   0");
        assert_eq!(fmt_i4(12345), "12345"); // wider than 4: no truncation
    }

    #[test]
    fn fmt_f12_width_and_precision() {
        assert_eq!(fmt_f12(1.0), "  1.000000000000");
        assert_eq!(fmt_f12(-0.5), " -0.500000000000");
        assert_eq!(fmt_f12(0.0), "  0.000000000000");
    }

    #[test]
    fn fmt_e16_rust_scientific() {
        // Rust `%.16e` semantics: single-digit exponent, no plus sign
        assert_eq!(fmt_e16(1.0), "1.0000000000000000e0");
        assert_eq!(fmt_e16(0.1), "1.0000000000000001e-1");
        assert_eq!(fmt_e16(-2.5), "-2.5000000000000000e0");
    }

    #[test]
    fn fmt_g17_notational_boundaries() {
        // zero and non-finite specials
        assert_eq!(fmt_g17(0.0), "0");
        assert_eq!(fmt_g17(f64::NAN), "NaN");
        assert_eq!(fmt_g17(f64::INFINITY), "inf");
        // fixed branch: trailing zeros trimmed
        assert_eq!(fmt_g17(1.0), "1");
        assert_eq!(fmt_g17(-2.5), "-2.5");
        assert_eq!(fmt_g17(1.0 / 3.0), "0.33333333333333331");
        // exp = 16 stays fixed, exp = 17 switches to scientific;
        // integer zeros survive (C probe verified)
        assert_eq!(fmt_g17(1e16), "10000000000000000");
        assert_eq!(fmt_g17(1.5e16), "15000000000000000");
        assert_eq!(fmt_g17(9999999999999998.0), "9999999999999998");
        assert_eq!(fmt_g17(1234567890123456.0), "1234567890123456");
        assert_eq!(fmt_g17(1e17), "1e+17");
        // scientific branch keeps the 17th significant digit
        assert_eq!(fmt_g17(2e-5), "2.0000000000000002e-05");
        // exp = -5 switches to scientific with a 2-digit exponent
        assert_eq!(fmt_g17(1e-5), "1.0000000000000001e-05");
    }
}
