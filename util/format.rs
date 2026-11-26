// Project : KadanoffBaym
// File    : util/format.rs
// Purpose : the printf-style float formatting of the shared utilities
//

/// printf-style `%.17g`.
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

/// printf-style `%.17e` formatting (two-digit zero-padded exponent with sign,
/// e.g. `4.16666666666666685e-01`). Rust's `{:e}` formatting is exact
/// and correctly rounded like C's printf; only the exponent shape is
/// adapted (`e0` -> `e+00`, `e-1` -> `e-01`).
pub fn fmt_e17(x: f64) -> String {
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

/// `%.12e` formatting of a float (C style two-digit exponent with sign).
pub fn fmt_e12(x: f64) -> String {
    let s = format!("{x:.12e}");
    // Rust writes e1 where C writes e+01, and e-4 where C writes e-04
    let Some((mant, exp)) = s.split_once('e') else {
        // NaN and the infinities carry no exponent
        return s;
    };
    let e: i32 = exp.parse().expect("{:e} exponent is an integer");
    let esign = if e < 0 { '-' } else { '+' };
    format!("{mant}e{esign}{:02}", e.abs())
}
