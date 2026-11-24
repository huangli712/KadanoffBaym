// Project : KadanoffBaym
// File    : util/kbref/write.rs
// Purpose : the kbref-v1 writer (schema round-trip tests only)
//

use std::fs::File;
use std::io::{Result as IoResult, Write};

use super::super::format::fmt_e17;
use super::read::KbrefRow;
use super::schema::KBREF_COLUMNS;

/// Write a kbref-v1 file (schema round-trip tests only).
pub fn write_kbref(
    path: &str,
    data_kind: &str,
    meta: &[(String, String)],
    rows: &[KbrefRow],
) -> IoResult<()> {
    let mut out = String::new();
    out.push_str("# kbref 1\n");
    out.push_str(&format!("# data_kind {data_kind}\n"));
    for (k, v) in meta {
        out.push_str(&format!("# {k} {v}\n"));
    }
    out.push_str(&format!("# columns {KBREF_COLUMNS}\n"));
    for r in rows {
        out.push_str(&format!(
            "{} {} {} {} {} {} {}\n",
            r.component,
            r.i,
            r.j,
            r.a,
            r.b,
            fmt_e17(r.value.re),
            fmt_e17(r.value.im)
        ));
    }
    File::create(path)?.write_all(out.as_bytes())
}
