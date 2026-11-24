// Project : KadanoffBaym
// File    : tests/schema/support.rs
// Purpose : shared helpers of the ref-v1 reference-format tests
//

pub(crate) use std::collections::HashMap;
pub(crate) use std::path::{Path, PathBuf};

pub(crate) use std::fs::{read, read_to_string, write};

pub(crate) use super::util::{RefData, RefRow};

pub(crate) use super::util::{
    OBSERVABLE_MODELS, REF_CG_SCALARS, REF_OBSERVABLE_META, REF_SELECTED_GREEN_KEYS,
    REF_SERIES_CASE_SETS,
};

pub(crate) use super::util::{
    approximations, fmt_e17, observable_components, read_ref, series_components, series_generator,
    tmpdir, write_ref,
};

pub(crate) use super::util::prelude::*;

// ------------------------------------------------------------------
// helper types
// ------------------------------------------------------------------

/// Label lookup by case id (common: labels_by_case).
pub(crate) struct CaseLabels {
    pub(crate) order: Option<i64>,
    pub(crate) alpha: Option<Complex>,
}

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Write a file, optionally edit its text, and expect that read_ref
/// rejects it.
pub(crate) fn expect_rejected(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[RefRow],
) {
    expect_rejected_after_edit(dir, name, kind, meta, rows, None)
}

pub(crate) fn expect_rejected_after_edit(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[RefRow],
    edit: Option<fn(String) -> String>,
) {
    let q = dir.join(name);
    let meta_owned: Vec<(String, String)> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_ref(q.to_str().unwrap(), kind, &meta_owned, rows).expect("write");
    if let Some(f) = edit {
        let text = read_to_string(&q).unwrap();
        write(&q, f(text)).unwrap();
    }
    let r = read_ref(q.to_str().unwrap());
    assert!(
        r.is_err(),
        "{name}: read_ref accepted a expect_rejected file ({r:?})"
    );
}

pub(crate) fn labels_by_case(
    d: &RefData,
    cases: &[i64],
    want_order: bool,
    want_alpha: bool,
) -> Result<HashMap<i64, CaseLabels>, String> {
    if !want_order && !want_alpha {
        return Err("nothing requested".into());
    }
    let mut out = HashMap::new();
    for r in &d.rows {
        if !cases.contains(&r.i) {
            return Err(format!("label refers to unknown case {}", r.i));
        }
        let e = out.entry(r.i).or_insert(CaseLabels {
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

pub(crate) fn reference_row(
    comp: &str,
    i: i64,
    j: i64,
    a: i64,
    b: i64,
    re: f64,
    im: f64,
) -> RefRow {
    RefRow {
        component: comp.to_string(),
        i,
        j,
        a,
        b,
        value: Complex::new(re, im),
    }
}

/// Write/read/rewrite round trip; the two files must be byte identical.
pub(crate) fn roundtrip(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[RefRow],
) -> RefData {
    let p = dir.join(name);
    let q = dir.join(format!("re_{name}"));
    let meta_owned: Vec<(String, String)> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_ref(p.to_str().unwrap(), kind, &meta_owned, rows).expect("write");
    let d = read_ref(p.to_str().unwrap()).expect("read");
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
    write_ref(q.to_str().unwrap(), kind, &meta_owned, &d.rows).expect("rewrite");
    assert_eq!(
        read(&p).unwrap(),
        read(&q).unwrap(),
        "{name}: rewrite not byte identical"
    );
    d
}

pub(crate) fn roundtrip_with_extra_column(
    dir: &Path,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[RefRow],
) -> RefData {
    roundtrip(dir, name, kind, meta, rows)
}

pub(crate) fn tempdir_path(tag: &str) -> PathBuf {
    PathBuf::from(tmpdir(tag))
}

// add one more column to the first data line
pub(crate) fn with_seventh_column(s: String) -> String {
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

pub(crate) fn without_key<'a>(meta: &[(&'a str, &'a str)], key: &str) -> Vec<(&'a str, &'a str)> {
    meta.iter()
        .filter(|(k, _)| *k != key)
        .map(|(k, v)| (*k, *v))
        .collect()
}
