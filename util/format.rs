// Project : KadanoffBaym
// File    : util/format.rs
// Purpose : the C-printf emulation layer of the shared utilities
//           (%.12e / %.17g / %.17e with C exponent shapes)
//

/// C-style `%.17g`.
pub fn fmt_17g(x: f64) -> String {
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

/// C-style `%.17e` formatting (two-digit zero-padded exponent with sign,
/// e.g. `4.16666666666666685e-01`). Rust's `{:e}` formatting is exact
/// and correctly rounded like C's printf; only the exponent shape is
/// adapted (`e0` -> `e+00`, `e-1` -> `e-01`).
pub fn fmt_c_e17(x: f64) -> String {
    if !x.is_finite() {
        // C printf style; Rust's f64 parser accepts these tokens back
        if x.is_nan() {
            return "nan".to_string();
        }
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let s = format!("{:.17e}", x);
    let (mant, exp) = s.split_once('e').expect("{:e} always contains 'e'");
    let e: i32 = exp.parse().expect("{:e} exponent is an integer");
    let esign = if e < 0 { '-' } else { '+' };
    format!("{mant}e{esign}{:02}", e.abs())
}

/// `%.12e` formatting of a float (C style two-digit exponent).
pub fn fmt_e12(x: f64) -> String {
    let mut s = format!("{x:.12e}");
    // Rust writes e-4 where C writes e-04: pad the exponent to two digits
    if let Some(epos) = s.rfind(['e', 'E']) {
        let (mant, exp) = s.split_at(epos);
        let (sign, digits) = exp[1..].split_at(1);
        if digits.len() < 2 {
            s = format!("{mant}e{sign}0{digits}");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn fmt_c_e17_matches_c_printout() {
        // Values and their C printf("%.17e") outputs (gcc probe)
        assert_eq!(fmt_c_e17(0.0), "0.00000000000000000e+00");
        assert_eq!(fmt_c_e17(-0.0), "-0.00000000000000000e+00");
        assert_eq!(fmt_c_e17(0.5), "5.00000000000000000e-01");
        assert_eq!(
            fmt_c_e17(4.166_666_666_666_667e-1),
            "4.16666666666666685e-01"
        );
        assert_eq!(fmt_c_e17(1.0), "1.00000000000000000e+00");
        assert_eq!(fmt_c_e17(-2.5), "-2.50000000000000000e+00");
        assert_eq!(fmt_c_e17(10.0), "1.00000000000000000e+01");
        assert_eq!(fmt_c_e17(0.1), "1.00000000000000006e-01");
        assert_eq!(fmt_c_e17(1.0e-7), "9.99999999999999955e-08");
        assert_eq!(fmt_c_e17(5.0e-324), "4.94065645841246544e-324");
        assert_eq!(
            fmt_c_e17(1.7976931348623157e308),
            "1.79769313486231571e+308"
        );
        assert_eq!(fmt_c_e17(6.02e23), "6.01999999999999996e+23");
        assert_eq!(fmt_c_e17(PI), "3.14159265358979312e+00");
        assert_eq!(fmt_c_e17(1.0e100), "1.00000000000000002e+100");
        // roundtrip
        for x in [
            0.1f64,
            PI,
            -6.02e23,
            5.0e-324,
            1.7976931348623157e308,
        ] {
            let s = fmt_c_e17(x);
            let back: f64 = s.parse().unwrap();
            assert_eq!(back, x);
        }
    }
}
