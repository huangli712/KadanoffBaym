// Project : KadanoffBaym
// File    : src/io/format.rs
// Purpose : C-style formatting shared by the writers (%4i, %16.12f,
//           %.16e, %.17g)

/// C-style `%4i` (width 4, space padded).
pub(super) fn fmt_i4(x: i64) -> String {
    format!("{:4}", x)
}

/// C-style `%16.12f` (width 16, precision 12, space padded).
pub(super) fn fmt_f12(x: f64) -> String {
    format!("{:16.12}", x)
}

/// C-style `%.16e` (scientific notation, 16 fractional digits).
pub(super) fn fmt_e16(x: f64) -> String {
    format!("{x:.16e}")
}

/// C `%.17g` formatting of a float: the shortest representation
/// with 17 significant digits, fixed notation for exponents in
/// `-4..17`, scientific notation otherwise (C `%g` semantics); NaN and
/// the infinities are written as C printf does. The decimal exponent is
/// taken from the `%.16e` form, so a value just below a power of ten is
/// not rounded onto the power by `log10`.
pub fn fmt_g17(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // the decimal exponent from the %.16e form
    let e = format!("{x:.16e}");
    let exp: i32 = e
        .split('e')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if !(-4..17).contains(&exp) {
        // %e style, trailing zeros of the mantissa stripped
        let mut mantissa = e.clone();
        let epos = mantissa.find('e').unwrap();
        let suffix = mantissa.split_off(epos);
        if mantissa.contains('.') {
            while mantissa.ends_with('0') {
                mantissa.pop();
            }
            if mantissa.ends_with('.') {
                mantissa.pop();
            }
        }
        // C prints at least two exponent digits
        let (_, digits) = suffix.split_once('e').unwrap();
        let (sign, mag) = if let Some(m) = digits.strip_prefix('-') {
            ('-', m)
        } else {
            ('+', digits.strip_prefix('+').unwrap_or(digits))
        };
        if mag.len() < 2 {
            format!("{mantissa}e{sign}0{mag}")
        } else {
            format!("{mantissa}e{sign}{mag}")
        }
    } else {
        // %f style with 16 - exp decimals, trailing zeros stripped
        let prec = (16 - exp).max(0) as usize;
        let mut s = format!("{x:.prec$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
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
        // zero and non-finite specials, written as C printf does
        assert_eq!(fmt_g17(0.0), "0");
        assert_eq!(fmt_g17(f64::NAN), "nan");
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
        // a value just below a power of ten keeps the exponent of the
        // power minus one (C probe verified)
        assert_eq!(fmt_g17(9.999999999999999e-5), "9.9999999999999991e-05");
        assert_eq!(fmt_g17(9.999999999999998e16), "99999999999999984");
    }
}
