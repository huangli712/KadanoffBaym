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
        // fixed with the remaining decimals
        let prec = (16 - exp).max(0) as usize;
        let raw = format!("{x:.prec$}");
        raw.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}
