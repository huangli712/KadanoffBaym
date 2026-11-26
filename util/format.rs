// Project : KadanoffBaym
// File    : util/format.rs
// Purpose : the printf-style float formatting of the shared utilities
//

/// printf-style `%.17g` (the library implementation).
pub use kadanoffbaym::io::fmt_g17;

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
