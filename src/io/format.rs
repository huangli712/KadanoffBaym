// Project : KadanoffBaym
// File    : src/io/format.rs
// Purpose : C-style formatting shared by the writers (%4i, %16.12f)

/// C-style `%4i` (width 4, space padded).
pub(crate) fn fmt_i4(x: i64) -> String {
    format!("{:4}", x)
}

/// C-style `%16.12f` (width 16, precision 12, space padded).
pub(crate) fn fmt_f12(x: f64) -> String {
    format!("{:16.12}", x)
}
