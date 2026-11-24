// Project : KadanoffBaym
// File    : tests/io/kbref_schema/mod.rs
// Purpose : port of t_kbref_schema; contract tests of the kbref-v1
//           plain-text reference format
//

use std::fs;
use std::path::{Path, PathBuf};

use kadanoffbaym::complex::Complex;

use super::util;
use super::util::fmt_e17;
use super::util::{
    KBREF_CG_SCALARS, KBREF_OBSERVABLE_META, KBREF_SELECTED_GREEN_KEYS, KBREF_SERIES_CASE_SETS,
    KbrefData, KbrefRow, OBSERVABLE_MODELS, approximations, observable_components, read_kbref,
    series_components, series_generator, write_kbref,
};

mod legacy;
mod moving_series;
mod observable;
mod reference;
mod structure;

fn row(comp: &str, i: i64, j: i64, a: i64, b: i64, re: f64, im: f64) -> KbrefRow {
    KbrefRow {
        component: comp.to_string(),
        i,
        j,
        a,
        b,
        value: Complex::new(re, im),
    }
}

fn tempdir_path(tag: &str) -> PathBuf {
    PathBuf::from(super::util::tmpdir(tag))
}

/// Write a file, optionally edit its text, and expect that read_kbref
/// rejects it.
fn bad(dir: &Path, name: &str, kind: &str, meta: &[(&str, &str)], rows: &[KbrefRow]) {
    bad_edit(dir, name, kind, meta, rows, None)
}

fn bad_edit(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[KbrefRow],
    edit: Option<fn(String) -> String>,
) {
    let q = dir.join(name);
    let meta_owned: Vec<(String, String)> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_kbref(q.to_str().unwrap(), kind, &meta_owned, rows).expect("write");
    if let Some(f) = edit {
        let text = fs::read_to_string(&q).unwrap();
        fs::write(&q, f(text)).unwrap();
    }
    let r = read_kbref(q.to_str().unwrap());
    assert!(r.is_err(), "{name}: read_kbref accepted a bad file ({r:?})");
}

/// Write/read/rewrite round trip; the two files must be byte identical.
fn roundtrip(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[KbrefRow],
) -> KbrefData {
    let p = dir.join(name);
    let q = dir.join(format!("re_{name}"));
    let meta_owned: Vec<(String, String)> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_kbref(p.to_str().unwrap(), kind, &meta_owned, rows).expect("write");
    let d = read_kbref(p.to_str().unwrap()).expect("read");
    assert_eq!(d.metadata["data_kind"], kind);
    for (k, v) in meta {
        assert_eq!(&d.metadata[*k], v, "{name}: metadata {k}");
    }
    assert_eq!(d.rows.len(), rows.len(), "{name}: row count");
    for (r, w) in d.rows.iter().zip(rows) {
        assert_eq!(
            (r.component.as_str(), r.i, r.j, r.a, r.b, r.value),
            (w.component.as_str(), w.i, w.j, w.a, w.b, w.value),
            "{name}: row"
        );
    }
    write_kbref(q.to_str().unwrap(), kind, &meta_owned, &d.rows).expect("rewrite");
    assert_eq!(
        fs::read(&p).unwrap(),
        fs::read(&q).unwrap(),
        "{name}: rewrite not byte identical"
    );
    d
}

fn without<'a>(meta: &[(&'a str, &'a str)], key: &str) -> Vec<(&'a str, &'a str)> {
    meta.iter()
        .filter(|(k, _)| *k != key)
        .map(|(k, v)| (*k, *v))
        .collect()
}

// add one more column to the first data line
fn ncol7(s: String) -> String {
    let mut lines = s.lines();
    let mut out = String::new();
    let mut done = false;
    for line in lines.by_ref() {
        if !done && !line.starts_with('#') && !line.trim().is_empty() {
            out.push_str(line);
            out.push_str(" 0.0\n");
            done = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Label lookup by case id (port of common: labels_by_case).
struct Labels {
    order: Option<i64>,
    alpha: Option<Complex>,
}

fn labels_by_case(
    d: &KbrefData,
    cases: &[i64],
    want_order: bool,
    want_alpha: bool,
) -> Result<std::collections::HashMap<i64, Labels>, String> {
    if !want_order && !want_alpha {
        return Err("nothing requested".into());
    }
    let mut out = std::collections::HashMap::new();
    for r in &d.rows {
        if !cases.contains(&r.i) {
            return Err(format!("label refers to unknown case {}", r.i));
        }
        let e = out.entry(r.i).or_insert(Labels {
            order: None,
            alpha: None,
        });
        if r.component == "order" {
            if !want_order {
                return Err(format!("unexpected order label for case {}", r.i));
            }
            e.order = Some(r.value.re as i64);
        } else if r.component == "alpha" {
            if !want_alpha {
                return Err(format!("unexpected alpha label for case {}", r.i));
            }
            e.alpha = Some(r.value);
        }
    }
    for &c in cases {
        let e = out
            .get(&c)
            .ok_or_else(|| format!("no labels for case {c}"))?;
        if want_order && e.order.is_none() {
            return Err(format!("missing order for case {c}"));
        }
        if want_alpha && e.alpha.is_none() {
            return Err(format!("missing alpha for case {c}"));
        }
    }
    Ok(out)
}

fn roundtrip7(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[KbrefRow],
) -> KbrefData {
    roundtrip(dir, name, kind, meta, rows)
}
