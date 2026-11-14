// Project : KadanoffBaym
// File    : src/io/format.rs
// Purpose : C-style float formatting shared by the writers

/// C-style `%16.12f` (width 16, precision 12, space padded).
pub(crate) fn fmt_f12(x: f64) -> String {
    format!("{:16.12}", x)
}
