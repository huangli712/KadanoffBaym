//! Port of julia/test/t_kbref_schema.jl (13 testsets).
//!
//! Contract tests of the kbref-v1 plain-text reference format.

mod common;

use std::fs;
use std::path::PathBuf;

use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::kbref::{
    fmt_c_e17, kbref_observable_components, kbref_series_components, p5_approximations, read_kbref,
    series_generator, write_kbref, KbrefData, KbrefRow, KBREF_CG_SCALARS, KBREF_P5_META,
    KBREF_SELECTED_GREEN_KEYS, KBREF_SERIES_CASE_SETS, P5_MODELS,
};

fn row(comp: &str, i: i64, j: i64, a: i64, b: i64, re: f64, im: f64) -> KbrefRow {
    KbrefRow {
        component: comp.to_string(),
        i,
        j,
        a,
        b,
        value: C64::new(re, im),
    }
}

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kb-schema-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("tempdir");
    dir
}

/// Write a file, optionally edit its text, and expect that read_kbref
/// rejects it.
fn bad(dir: &PathBuf, name: &str, kind: &str, meta: &[(&str, &str)], rows: &[KbrefRow]) {
    bad_edit(dir, name, kind, meta, rows, None)
}

fn bad_edit(
    dir: &PathBuf,
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
    dir: &PathBuf,
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

#[test]
fn p1_herm_matrix_and_weights() {
    let dir = tempdir("p1");
    let hmeta = [
        ("case", "selftest"),
        ("nt", "1"),
        ("ntau", "1"),
        ("size1", "1"),
        ("size2", "1"),
    ];
    let hrows = vec![
        row("mat", 0, -1, 0, 0, 1.0, 2.0),
        row("mat", 1, -1, 0, 0, -1.0 / 3.0, 0.0),
        row("ret", 0, 0, 0, 0, 0.0, -1.0),
        row("ret", 1, 0, 0, 0, 0.1, 0.2),
        row("ret", 1, 1, 0, 0, 0.0, -1.0),
        row("tv", 0, 0, 0, 0, 0.5, 0.5),
        row("tv", 0, 1, 0, 0, std::f64::consts::PI, 0.0),
        row("tv", 1, 0, 0, 0, 1e-300, 0.0),
        row("tv", 1, 1, 0, 0, -1e300, 0.0),
        row("les", 0, 0, 0, 0, 0.0, 0.3),
        row("les", 0, 1, 0, 0, 0.7, -0.1),
        row("les", 1, 1, 0, 0, 0.0, 0.3),
    ];
    let d = roundtrip(&dir, "ok_herm.dat", "herm_matrix", &hmeta, &hrows);
    assert_eq!(d.metadata["case"], "selftest");

    let wrows = vec![
        row("greg", 1, 0, 0, -1, 5.0 / 12.0, 0.0),
        row("bd", 1, 2, -1, -1, 0.5, 0.0),
        row("pint", 1, 1, 0, 1, 0.5, 0.0),
        row("pdiff", 1, 1, 0, -1, -1.0, 0.0),
        row("rcorr", 2, 1, 2, 0, 0.1, 0.0),
    ];
    roundtrip(
        &dir,
        "ok_weights.dat",
        "weights",
        &[("case", "selftest")],
        &wrows,
    );

    // Bad inputs
    bad_edit(
        &dir,
        "kind.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| s.replace("data_kind herm_matrix", "data_kind unknown")),
    );
    bad_edit(
        &dir,
        "version.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| s.replace("# kbref 1", "# kbref 2")),
    );
    bad_edit(
        &dir,
        "columns.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| s.replace("re im", "re im extra")),
    );
    let mut r = hrows.clone();
    r.pop();
    r.push(row("gtr", 1, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "comp.dat", "herm_matrix", &hmeta, &r);
    bad_edit(
        &dir,
        "ncol6.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| {
            // drop the last column of the "les 1 1 0 0" line
            let mut out = String::new();
            for line in s.lines() {
                if line.starts_with("les 1 1 0 0") {
                    let f: Vec<&str> = line.split_whitespace().collect();
                    out.push_str(&f[..6].join(" "));
                    out.push('\n');
                } else {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out
        }),
    );
    bad_edit(
        &dir,
        "ncol8.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| {
            s.replace("\nles 1 1 0 0 ", "\nles 1 1 0 0 ")
                .replace("les 1 1 0 0 0.0", "les 1 1 0 0 0.0 0")
        }),
    );
    let mut r = hrows.clone();
    r.push(row("les", 1, 1, 0, 0, 0.0, 0.3));
    bad(&dir, "dup.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[2] = row("ret", 0, 1, 0, 0, 0.0, 0.0);
    bad(&dir, "ret_upper.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[11] = row("les", 1, 0, 0, 0, 0.0, 0.0);
    bad(&dir, "les_lower.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[11] = row("les", 1, 1, 1, 0, 0.0, 0.0);
    bad(&dir, "orbital.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[0] = row("mat", 0, 0, 0, 0, 0.0, 0.0);
    bad(&dir, "mat_j.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[11] = row("les", 1, 1, 0, 0, f64::NAN, 0.0);
    bad(&dir, "nan.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r[11] = row("les", 1, 1, 0, 0, 0.0, f64::INFINITY);
    bad(&dir, "inf.dat", "herm_matrix", &hmeta, &r);
    let mut r = hrows.clone();
    r.pop();
    bad(&dir, "incomplete.dat", "herm_matrix", &hmeta, &r);
    bad_edit(
        &dir,
        "noheader.dat",
        "herm_matrix",
        &hmeta,
        &hrows,
        Some(|s| s.replace("# kbref 1\n", "")),
    );
    bad(
        &dir,
        "w_comp.dat",
        "weights",
        &[("case", "x")],
        &[row("mat", 1, 0, -1, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "w_rcorr_k1.dat",
        "weights",
        &[("case", "x")],
        &[row("rcorr", 1, 0, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "w_rcorr_m0.dat",
        "weights",
        &[("case", "x")],
        &[row("rcorr", 3, 0, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "w_k6.dat",
        "weights",
        &[("case", "x")],
        &[row("bd", 6, 0, -1, -1, 1.0, 0.0)],
    );
    bad(
        &dir,
        "w_imag.dat",
        "weights",
        &[("case", "x")],
        &[row("bd", 1, 0, -1, -1, 1.0, 1.0)],
    );

    // P2 herm_matrix files must carry a non-negative case_id
    let p2meta: Vec<(&str, &str)> = [
        hmeta.as_slice(),
        &[("generator", "p2_selftest"), ("case_set", "x")],
    ]
    .concat();
    let p = dir.join("ok_herm_p2.dat");
    let mut m = p2meta.clone();
    m.push(("case_id", "3"));
    let m_owned: Vec<(String, String)> = m
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_kbref(p.to_str().unwrap(), "herm_matrix", &m_owned, &hrows).unwrap();
    assert_eq!(
        read_kbref(p.to_str().unwrap()).unwrap().metadata["case_id"],
        "3"
    );
    bad(&dir, "herm_p2_nocase.dat", "herm_matrix", &p2meta, &hrows);
    let mut m = p2meta.clone();
    m.push(("case_id", "-1"));
    bad(&dir, "herm_p2_negcase.dat", "herm_matrix", &m, &hrows);
    let _ = fs::remove_dir_all(&dir);
}

/// Label lookup by case id (port of common.jl:labels_by_case).
struct Labels {
    order: Option<i64>,
    alpha: Option<C64>,
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

#[test]
fn p2_scalar_function_element_labels() {
    let dir = tempdir("p2");
    let base = [("generator", "p2_selftest"), ("case_set", "selftest")];
    let srows = vec![
        row("distance_ret", 0, -1, -1, -1, 1.5, 0.0),
        row("distance_tv", 0, -1, -1, -1, 0.0, 0.0),
        row("distance_les", 0, -1, -1, -1, 2.5e-17, 0.0),
        row("distance_total", 0, -1, -1, -1, 4.0, 0.0),
        row("response", 1, -1, -1, -1, -0.3, 0.7),
    ];
    let fmeta: Vec<(&str, &str)> = [
        base.as_slice(),
        &[("nt", "3"), ("size1", "2"), ("size2", "2")],
    ]
    .concat();
    let mut frows = Vec::new();
    for j in -1i64..=3 {
        for a in 0..2 {
            for b in 0..2 {
                frows.push(row("extrapolate", 2, j, a, b, j as f64, (a + b) as f64));
            }
        }
    }
    let imeta: Vec<(&str, &str)> = [fmeta.as_slice(), &[("tinter", "1.5")]].concat();
    let mut irows = Vec::new();
    for a in 0..2 {
        for b in 0..2 {
            irows.push(row("interpolate", 0, -1, a, b, a as f64, b as f64));
        }
    }
    let mut erows = Vec::new();
    for c in ["les_timediag", "density_matrix"] {
        for t in -1i64..=3 {
            for a in 0..2 {
                for b in 0..2 {
                    erows.push(row(c, 4, t, a, b, t as f64, 0.5));
                }
            }
        }
    }
    for c in ["startup_less_adv", "startup_lmix_rmix"] {
        for j in 0i64..=3 {
            for a in 0..2 {
                for b in 0..2 {
                    erows.push(row(c, 5, j, a, b, j as f64, -0.5));
                }
            }
        }
    }
    let lrows = vec![
        row("order", 0, -1, -1, -1, 0.0, 0.0),
        row("order", 1, -1, -1, -1, 5.0, 0.0),
        row("alpha", 1, -1, -1, -1, 0.5, -0.25),
    ];
    let cases: Vec<(&str, &str, Vec<KbrefRow>)> = vec![
        ("ok_scalar.dat", "scalar", srows.clone()),
        ("ok_function.dat", "function", frows.clone()),
        ("ok_interp.dat", "function", irows.clone()),
        ("ok_element.dat", "element", erows.clone()),
        ("ok_labels.dat", "labels", lrows.clone()),
    ];
    for (name, kind, rows) in &cases {
        roundtrip(
            &dir,
            name,
            kind,
            if *kind == "function" || *kind == "element" {
                if name.contains("interp") {
                    &imeta
                } else {
                    &fmeta
                }
            } else {
                &base
            },
            rows,
        );
    }
    // Labels joined with cases
    let lab = read_kbref(dir.join("ok_labels.dat").to_str().unwrap()).unwrap();
    assert!(labels_by_case(&lab, &[0, 1], true, false).is_err());
    let lab2 = KbrefData {
        path: lab.path.clone(),
        metadata: lab.metadata.clone(),
        rows: lab.rows[..2].to_vec(),
    };
    let m = labels_by_case(&lab2, &[0, 1], true, false).unwrap();
    assert_eq!(m[&0].order, Some(0));
    assert_eq!(m[&1].order, Some(5));
    assert!(m[&0].alpha.is_none());
    let lab3 = KbrefData {
        path: lab.path.clone(),
        metadata: lab.metadata.clone(),
        rows: lab.rows[1..3].to_vec(),
    };
    let m = labels_by_case(&lab3, &[1], true, true).unwrap();
    assert_eq!(m[&1].alpha, Some(C64::new(0.5, -0.25)));
    assert!(labels_by_case(&lab2, &[0, 1, 2], true, false).is_err());
    assert!(labels_by_case(&lab2, &[0, 1], true, true).is_err());
    assert!(labels_by_case(&lab2, &[0], true, false).is_err());
    assert!(labels_by_case(&lab2, &[0, 1], false, false).is_err());

    // Bad inputs
    bad(
        &dir,
        "s_comp.dat",
        "scalar",
        &base,
        &[row("distance", 0, -1, -1, -1, 1.0, 0.0)],
    );
    bad(
        &dir,
        "s_imag.dat",
        "scalar",
        &base,
        &[row("distance_ret", 0, -1, -1, -1, 1.0, 1.0)],
    );
    bad(
        &dir,
        "s_neg.dat",
        "scalar",
        &base,
        &[row("response", -1, -1, -1, -1, 1.0, 0.0)],
    );
    bad(
        &dir,
        "s_j.dat",
        "scalar",
        &base,
        &[row("response", 0, 0, -1, -1, 1.0, 0.0)],
    );
    let mut r = srows.clone();
    r.push(row("response", 1, -1, -1, -1, -0.3, 0.7));
    bad(&dir, "s_dup.dat", "scalar", &base, &r);
    bad(
        &dir,
        "s_nan.dat",
        "scalar",
        &base,
        &[row("response", 0, -1, -1, -1, f64::NAN, 0.0)],
    );
    bad(
        &dir,
        "s_inf.dat",
        "scalar",
        &base,
        &[row("response", 0, -1, -1, -1, 0.0, f64::NEG_INFINITY)],
    );
    bad_edit(
        &dir,
        "s_ncol.dat",
        "scalar",
        &base,
        &srows,
        Some(|s| s.replace("\nresponse 1 -1 -1 -1 ", "\nresponse 1 -1 -1 -1 1 ")),
    );
    bad(&dir, "s_nogen.dat", "scalar", &[("case_set", "x")], &srows);
    bad(
        &dir,
        "s_nocaseset.dat",
        "scalar",
        &[("generator", "p2_x")],
        &srows,
    );
    bad(
        &dir,
        "f_orb.dat",
        "function",
        &fmeta,
        &[row("extrapolate", 0, 0, 2, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "f_t.dat",
        "function",
        &fmeta,
        &[row("extrapolate", 0, 4, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "f_t2.dat",
        "function",
        &fmeta,
        &[row("extrapolate", 0, -2, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "f_nont.dat",
        "function",
        &base,
        &[row("extrapolate", 0, 0, 0, 0, 0.0, 0.0)],
    );
    bad(&dir, "f_notinter.dat", "function", &fmeta, &irows);
    bad(
        &dir,
        "f_interp_j.dat",
        "function",
        &imeta,
        &[row("interpolate", 0, 0, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "f_comp.dat",
        "function",
        &fmeta,
        &[row("ret", 0, 0, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "e_comp.dat",
        "element",
        &fmeta,
        &[row("density", 0, 0, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "e_orb.dat",
        "element",
        &fmeta,
        &[row("les_timediag", 0, 0, 0, 2, 0.0, 0.0)],
    );
    bad(
        &dir,
        "e_t.dat",
        "element",
        &fmeta,
        &[row("les_timediag", 0, 4, 0, 0, 0.0, 0.0)],
    );
    let mut r = erows.clone();
    r.push(erows[0].clone());
    bad(&dir, "e_dup.dat", "element", &fmeta, &r);
    bad(
        &dir,
        "e_startup_neg.dat",
        "element",
        &fmeta,
        &[row("startup_less_adv", 0, -1, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "e_startup_big.dat",
        "element",
        &fmeta,
        &[row("startup_lmix_rmix", 0, 4, 0, 0, 0.0, 0.0)],
    );
    bad(
        &dir,
        "l_order_frac.dat",
        "labels",
        &base,
        &[row("order", 0, -1, -1, -1, 2.5, 0.0)],
    );
    bad(
        &dir,
        "l_order_big.dat",
        "labels",
        &base,
        &[row("order", 0, -1, -1, -1, 6.0, 0.0)],
    );
    bad(
        &dir,
        "l_order_neg.dat",
        "labels",
        &base,
        &[row("order", 0, -1, -1, -1, -1.0, 0.0)],
    );
    bad(
        &dir,
        "l_order_imag.dat",
        "labels",
        &base,
        &[row("order", 0, -1, -1, -1, 1.0, 1.0)],
    );
    let mut r = lrows.clone();
    r.push(lrows[0].clone());
    bad(&dir, "l_dup.dat", "labels", &base, &r);
    bad(
        &dir,
        "l_comp.dat",
        "labels",
        &base,
        &[row("beta", 0, -1, -1, -1, 1.0, 0.0)],
    );
    bad(
        &dir,
        "l_idx.dat",
        "labels",
        &base,
        &[row("order", 0, 0, -1, -1, 1.0, 0.0)],
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p4_iteration_and_single_case_files() {
    let dir = tempdir("p4");
    let fmeta = [
        ("generator", "phase4_vie2"),
        ("case_set", "selftest"),
        ("case", "fixpoint_selftest"),
        ("sig", "-1"),
        ("algorithm", "fixpoint"),
        ("niter", "1"),
        ("ntau", "1"),
        ("size1", "1"),
        ("size2", "1"),
    ];
    let mut frows = Vec::new();
    for it in 0i64..=1 {
        for r in 0i64..=1 {
            frows.push(row("g", it, r, 0, 0, 0.1 * it as f64, -0.2 * r as f64));
        }
    }
    let cmeta = [
        ("generator", "phase4_vie2"),
        ("case_set", "selftest"),
        ("case", "cg_selftest"),
        ("sig", "1"),
        ("algorithm", "cg"),
        ("niter", "2"),
        ("ntau", "1"),
        ("size1", "1"),
        ("size2", "1"),
    ];
    let mut crows = Vec::new();
    for it in 1i64..=2 {
        for r in 0i64..=1 {
            crows.push(row("g", it, r, 0, 0, it as f64 / 3.0, r as f64 / 7.0));
        }
        for (n, c) in KBREF_CG_SCALARS.iter().enumerate() {
            crows.push(row(
                c,
                it,
                -1,
                -1,
                -1,
                (n + 1) as f64 * it as f64 / 11.0,
                0.0,
            ));
        }
    }
    roundtrip(&dir, "ok_fix.dat", "iteration", &fmeta, &frows);
    roundtrip(&dir, "ok_cg.dat", "iteration", &cmeta, &crows);

    // Reused herm_matrix and element/density_matrix, one case per file
    let hmeta = [
        ("generator", "phase4_bubble"),
        ("case", "bubble1_ff_size1"),
        ("sig", "-1"),
        ("nt", "0"),
        ("ntau", "0"),
        ("size1", "1"),
        ("size2", "1"),
    ];
    let hrows = vec![
        row("mat", 0, -1, 0, 0, 0.5, 0.0),
        row("ret", 0, 0, 0, 0, 0.0, -1.0),
        row("tv", 0, 0, 0, 0, 0.0, 0.2),
        row("les", 0, 0, 0, 0, 0.0, 0.3),
    ];
    let d = roundtrip(&dir, "ok_p4_herm.dat", "herm_matrix", &hmeta, &hrows);
    assert_eq!(d.metadata["case"], "bubble1_ff_size1");
    let emeta = [
        ("generator", "phase4_observables"),
        ("case_set", "selftest"),
        ("case", "density_equilibrium"),
        ("sig", "1"),
        ("nt", "2"),
        ("size1", "1"),
        ("size2", "1"),
    ];
    let erows = vec![row("density_matrix", 0, -1, 0, 0, 0.4, 0.0)];
    let d = roundtrip(&dir, "ok_p4_elem.dat", "element", &emeta, &erows);
    assert_eq!(d.rows[0].component, "density_matrix");

    // Bad inputs
    bad_edit(
        &dir,
        "i_kind.dat",
        "iteration",
        &fmeta,
        &frows,
        Some(|s| s.replace("data_kind iteration", "data_kind trajectory")),
    );
    let mut r = frows.clone();
    r.push(row("residual", 1, -1, -1, -1, 0.0, 0.0));
    bad(&dir, "i_comp.dat", "iteration", &fmeta, &r);
    let mut r = hrows.clone();
    r[3] = row("les_bubble1_ff", 0, 0, 0, 0, 0.0, 0.3);
    bad(&dir, "i_casecomp.dat", "herm_matrix", &hmeta, &r);
    bad(
        &dir,
        "i_densitycomp.dat",
        "element",
        &emeta,
        &[row("density_state", 0, -1, 0, 0, 0.4, 0.0)],
    );
    let mut m = hmeta.to_vec();
    m.push(("case", "bubble2_ff_size1"));
    bad(&dir, "i_twocases.dat", "herm_matrix", &m, &hrows);
    bad(
        &dir,
        "i_nocase.dat",
        "herm_matrix",
        &without(&hmeta, "case"),
        &hrows,
    );
    bad(
        &dir,
        "i_nosig.dat",
        "iteration",
        &without(&cmeta, "sig"),
        &crows,
    );
    let mut m = without(&cmeta, "sig");
    m.push(("sig", "0"));
    bad(&dir, "i_badsig.dat", "iteration", &m, &crows);
    bad(
        &dir,
        "i_noalg.dat",
        "iteration",
        &without(&cmeta, "algorithm"),
        &crows,
    );
    bad_edit(
        &dir,
        "i_ncol.dat",
        "iteration",
        &fmeta,
        &frows,
        Some(|s| s.replace("\ng 1 1 0 0 ", "\ng 1 1 0 0 1 ")),
    );
    let mut r = frows.clone();
    r.push(frows[0].clone());
    bad(&dir, "i_dup.dat", "iteration", &fmeta, &r);
    let mut r = frows.clone();
    r.pop();
    r.push(row("g", 1, 1, 0, 0, f64::NAN, 0.0));
    bad(&dir, "i_nan.dat", "iteration", &fmeta, &r);
    let mut r = crows.clone();
    r.pop();
    r.push(row("beta_cg", 2, -1, -1, -1, f64::INFINITY, 0.0));
    bad(&dir, "i_inf.dat", "iteration", &cmeta, &r);
    let mut r = frows.clone();
    r.push(row("g", -1, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "i_negiter.dat", "iteration", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("g", 2, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "i_bigiter.dat", "iteration", &fmeta, &r);
    let mut r = crows.clone();
    r.push(row("g", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "i_cgiter0.dat", "iteration", &cmeta, &r);
    let mut r = frows.clone();
    r.push(row("g", 1, 2, 0, 0, 0.0, 0.0));
    bad(&dir, "i_gtau.dat", "iteration", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("g", 1, 0, 1, 0, 0.0, 0.0));
    bad(&dir, "i_gorb.dat", "iteration", &fmeta, &r);
    let mut r = crows.clone();
    r.pop();
    r.push(row("beta_cg", 2, 0, -1, -1, 0.1, 0.0));
    bad(&dir, "i_scalar_idx.dat", "iteration", &cmeta, &r);
    let mut r = crows.clone();
    r.pop();
    r.push(row("beta_cg", 2, -1, -1, -1, 0.1, 0.1));
    bad(&dir, "i_scalar_imag.dat", "iteration", &cmeta, &r);
    let mut r = frows.clone();
    r.push(row("alpha", 1, -1, -1, -1, 0.1, 0.0));
    bad(&dir, "i_scalar_fix.dat", "iteration", &fmeta, &r);
    let r: Vec<KbrefRow> = crows
        .iter()
        .filter(|x| !(x.component == "c2" && x.i == 2))
        .cloned()
        .collect();
    bad(&dir, "i_missing_scalar.dat", "iteration", &cmeta, &r);
    let mut r = crows.clone();
    r.push(row("rsnew", 1, -1, -1, -1, 0.5, 0.0));
    bad(&dir, "i_dup_scalar.dat", "iteration", &cmeta, &r);
    let mut r = frows.clone();
    r.pop();
    bad(&dir, "i_missing_g.dat", "iteration", &fmeta, &r);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p6_pseudo() {
    // NOTE: the `kbref_to_pseudo` rebuild assertions of the blueprint
    // test need PseudoGreen and land with R8; the schema contract is
    // fully covered here.
    let dir = tempdir("p6");
    let pmeta = [
        ("generator", "pseudo_reference"),
        ("case", "selftest"),
        ("sig", "-1"),
        ("nt", "1"),
        ("ntau", "1"),
        ("size1", "2"),
        ("size2", "2"),
        ("k", "2"),
    ];
    let mut prows = Vec::new();
    let mut n = 0i64;
    let mut v = |n: &mut i64| {
        *n += 1;
        C64::new(*n as f64 / 7.0, -(*n as f64) / 11.0)
    };
    for q in 0..2i64 {
        for a in 0..2 {
            for b in 0..2 {
                let val = v(&mut n);
                prows.push(row("mat", q, -1, a, b, val.re, val.im));
            }
        }
    }
    for i in 0..2i64 {
        for j in 0..=i {
            for a in 0..2 {
                for b in 0..2 {
                    let val = v(&mut n);
                    prows.push(row("gtr", i, j, a, b, val.re, val.im));
                }
            }
        }
    }
    for i in 0..2i64 {
        for q in 0..2i64 {
            for a in 0..2 {
                for b in 0..2 {
                    let val = v(&mut n);
                    prows.push(row("tv", i, q, a, b, val.re, val.im));
                }
            }
        }
    }
    for i in 0..2i64 {
        for j in i..2i64 {
            for a in 0..2 {
                for b in 0..2 {
                    let val = v(&mut n);
                    prows.push(row("les", i, j, a, b, val.re, val.im));
                }
            }
        }
    }
    for i in -1i64..=1 {
        for a in 0..2 {
            for b in 0..2 {
                let val = v(&mut n);
                prows.push(row("density", i, -1, a, b, val.re, val.im));
            }
        }
    }
    for i in 1..=2i64 {
        for j in 0..i {
            for a in 0..2 {
                for b in 0..2 {
                    let val = v(&mut n);
                    prows.push(row("les_start_source", i, j, a, b, val.re, val.im));
                }
            }
        }
    }
    roundtrip(&dir, "ok_pseudo.dat", "pseudo", &pmeta, &prows);
    // Storage rows are optional, but complete if present
    let srows: Vec<KbrefRow> = prows
        .iter()
        .filter(|r| r.component == "density" || r.component == "les_start_source")
        .cloned()
        .collect();
    let p = dir.join("ok_pseudo_src.dat");
    let m: Vec<(String, String)> = pmeta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    write_kbref(p.to_str().unwrap(), "pseudo", &m, &srows).unwrap();
    assert_eq!(
        read_kbref(p.to_str().unwrap()).unwrap().rows.len(),
        srows.len()
    );

    // Bad inputs
    bad_edit(
        &dir,
        "p_ncol.dat",
        "pseudo",
        &pmeta,
        &prows,
        Some(|s| s.replace("\nmat 0 -1 0 0 ", "\nmat 0 -1 0 0 1 ")),
    );
    let mut r = prows.clone();
    r.push(row("ret", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p_comp.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("gtr", 0, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p_gtr_tri.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("les", 1, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p_les_tri.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("density", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p_dens_j.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("les_start_source", 1, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p_src_low.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("les_start_source", 3, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p_src_high.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(row("density", 0, -1, 2, 0, 0.0, 0.0));
    bad(&dir, "p_orb.dat", "pseudo", &pmeta, &r);
    let mut r = prows.clone();
    r.push(prows[0].clone());
    bad(&dir, "p_dup.dat", "pseudo", &pmeta, &r);
    let mut r: Vec<KbrefRow> = prows.clone();
    r.pop();
    r.push(row("les_start_source", 2, 1, 1, 1, f64::NAN, 0.0));
    bad(&dir, "p_nan.dat", "pseudo", &pmeta, &r);
    let r: Vec<KbrefRow> = prows
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != 2)
        .map(|(_, x)| x.clone())
        .collect();
    bad(&dir, "p_incomplete.dat", "pseudo", &pmeta, &r);
    bad(&dir, "p_nok.dat", "pseudo", &without(&pmeta, "k"), &prows);
    bad(
        &dir,
        "p_nocase.dat",
        "pseudo",
        &without(&pmeta, "case"),
        &prows,
    );
    let mut m = without(&pmeta, "sig");
    m.push(("sig", "0"));
    bad(&dir, "p_badsig.dat", "pseudo", &m, &prows);
    let _ = fs::remove_dir_all(&dir);
}

fn roundtrip7(
    dir: &PathBuf,
    name: &str,
    kind: &str,
    meta: &[(&str, &str)],
    rows: &[KbrefRow],
) -> KbrefData {
    roundtrip(dir, name, kind, meta, rows)
}

#[test]
fn phase7_fourier_schema() {
    let dir = tempdir("p7f");
    let fmeta = [
        ("generator", "phase7_fourier_ref"),
        ("case_set", "selftest"),
        ("size_max", "2"),
    ];
    let v = |n: f64| C64::new(n / 3.0, -n / 5.0);
    let mut frows = Vec::new();
    for c in ["fft_input", "fft_forward", "fft_backward"] {
        for j in 0..3i64 {
            frows.push(row(c, 3, j, 0, 0, v(j as f64).re, v(j as f64).im));
        }
    }
    frows.push(row("dftcorr", 0, 1, 0, 0, 0.7, 0.0));
    frows.push(row("dftcorr", 0, 1, 1, 0, 0.9, 0.0));
    frows.push(row("dftcorr", 0, 1, 3, 0, v(1.0).re, v(1.0).im));
    frows.push(row("dftcorr", 0, 3, 5, 0, v(2.0).re, v(2.0).im));
    frows.push(row("adaptive", 2, 4, 0, 0, -1.5, 0.0));
    frows.push(row("adaptive", 2, 4, 2, 0, v(3.0).re, v(3.0).im));
    frows.push(row(
        "matsubara_dft_result",
        1,
        5,
        1,
        0,
        v(4.0).re,
        v(4.0).im,
    ));
    frows.push(row(
        "matsubara_ft_result",
        13,
        -25,
        0,
        1,
        v(5.0).re,
        v(5.0).im,
    ));
    frows.push(row("vie2_pcf20", 0, 7, 1, 1, v(6.0).re, v(6.0).im));
    frows.push(row("vie2_pcf10", 0, 7, 1, 1, v(7.0).re, v(7.0).im));
    let d = roundtrip7(&dir, "ok_p7f.dat", "phase7_fourier", &fmeta, &frows);
    let comps: std::collections::BTreeSet<&str> =
        d.rows.iter().map(|r| r.component.as_str()).collect();
    assert_eq!(
        comps,
        [
            "adaptive",
            "dftcorr",
            "fft_backward",
            "fft_forward",
            "fft_input",
            "matsubara_dft_result",
            "matsubara_ft_result",
            "vie2_pcf10",
            "vie2_pcf20"
        ]
        .into_iter()
        .collect()
    );
    // Bad inputs
    bad(&dir, "p7f_kind.dat", "phase7_fft", &fmeta, &frows);
    let mut r = frows.clone();
    r.push(row("fft", 3, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7f_comp.dat", "phase7_fourier", &fmeta, &r);
    bad_edit(
        &dir,
        "p7f_ncol.dat",
        "phase7_fourier",
        &fmeta,
        &frows,
        Some(ncol7),
    );
    let mut r = frows.clone();
    r.push(row("vie2_pcf10", 0, 8, 0, 0, f64::INFINITY, 0.0));
    bad(&dir, "p7f_nan.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("fft_input", 3, 3, 0, 0, 0.0, 0.0));
    bad(&dir, "p7f_fftj.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("fft_forward", 2, 0, 1, 0, 0.0, 0.0));
    bad(&dir, "p7f_ffta.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.remove(0);
    bad(&dir, "p7f_fftinc.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("dftcorr", 1, 2, 0, 0, 0.0, 0.0));
    bad(&dir, "p7f_order.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("dftcorr", 1, 1, 4, 0, 0.0, 0.0));
    bad(&dir, "p7f_alpha.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("dftcorr", 1, 3, 0, 0, 0.0, 1.0));
    bad(&dir, "p7f_theta.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("adaptive", 3, 0, 1, 0, 0.0, 0.0));
    bad(&dir, "p7f_fid.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("matsubara_ft_result", 12, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7f_ftord.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(row("vie2_pcf20", 0, 0, 2, 0, 0.0, 0.0));
    bad(&dir, "p7f_orb.dat", "phase7_fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(frows[frows.len() - 1].clone());
    bad(&dir, "p7f_dup.dat", "phase7_fourier", &fmeta, &r);
    bad(
        &dir,
        "p7f_nosize.dat",
        "phase7_fourier",
        &without(&fmeta, "size_max"),
        &frows,
    );
    bad(
        &dir,
        "p7f_noset.dat",
        "phase7_fourier",
        &without(&fmeta, "case_set"),
        &frows,
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn phase7_ness_schema() {
    let dir = tempdir("p7n");
    let nmeta = [
        ("generator", "phase7_ness_ref"),
        ("case_set", "selftest"),
        ("nft_max", "4"),
        ("size_max", "2"),
    ];
    let v = |n: f64| C64::new(-n / 7.0, n / 9.0);
    let mut nrows = Vec::new();
    let mut n = 0i64;
    for c in ["ret_time", "les_time", "ret_freq", "les_freq"] {
        n += 1;
        nrows.push(row(c, 1, 3, 1, 0, v(n as f64).re, v(n as f64).im));
    }
    for c in ["bath_ret", "bath_les"] {
        n += 1;
        nrows.push(row(c, 3, 12, 0, 0, v(n as f64).re, v(n as f64).im));
    }
    for c in ["density", "convolution_density"] {
        n += 1;
        nrows.push(row(c, 0, 0, 1, 1, v(n as f64).re, v(n as f64).im));
    }
    for c in [
        "bubble1",
        "bubble2",
        "resample",
        "force_equilibrium",
        "cntr_conversion",
    ] {
        n += 1;
        nrows.push(row(c, 0, 1, 0, 1, v(n as f64).re, v(n as f64).im));
        n += 1;
        nrows.push(row(c, 0, 7, 0, 1, v(n as f64).re, v(n as f64).im));
    }
    let d = roundtrip7(&dir, "ok_p7n.dat", "phase7_ness", &nmeta, &nrows);
    assert_eq!(d.rows.len(), nrows.len());
    // Bad inputs
    let mut r = nrows.clone();
    r.push(row("gtr_time", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_comp.dat", "phase7_ness", &nmeta, &r);
    bad_edit(
        &dir,
        "p7n_ncol.dat",
        "phase7_ness",
        &nmeta,
        &nrows,
        Some(ncol7),
    );
    let mut r = nrows.clone();
    r.push(row("ret_time", 0, 0, 0, 0, 0.0, f64::NAN));
    bad(&dir, "p7n_nan.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("ret_freq", 0, 4, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_mode.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bubble1", 0, 8, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_mode2.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("les_time", 0, 0, 0, 2, 0.0, 0.0));
    bad(&dir, "p7n_orb.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bath_ret", 4, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_bath.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bath_les", 1, 0, 1, 0, 0.0, 0.0));
    bad(&dir, "p7n_batho.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("density", 0, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_dens.dat", "phase7_ness", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(nrows[0].clone());
    bad(&dir, "p7n_dup.dat", "phase7_ness", &nmeta, &r);
    bad(
        &dir,
        "p7n_nonft.dat",
        "phase7_ness",
        &without(&nmeta, "nft_max"),
        &nrows,
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn phase7_example_schema() {
    let dir = tempdir("p7e");
    let emeta2 = |c: &'static str| {
        vec![
            (
                "generator",
                Box::leak(format!("phase7_examples_{c}_ref").into_boxed_str()) as &'static str,
            ),
            ("case_set", "selftest"),
            ("case", c),
        ]
    };
    let _ = emeta2;
    let siam = vec![
        row("gf_ret_time", 0, 5, 0, 0, 0.1, -0.2),
        row("gf_les_time", 0, 5, 0, 0, 0.0, 0.3),
        row("gf_ret_freq", 0, 9, 0, 0, 0.0, -0.4),
        row("gf_les_freq", 0, 9, 0, 0, 0.0, 0.5),
        row("density", 0, 0, 0, 0, 0.9, 0.0),
        row("current", 0, 0, 0, 0, 0.3, 0.0),
        row("current", 0, 1, 0, 0, 0.3, 0.0),
        row("eint", 0, 0, 0, 0, -0.1, 0.0),
    ];
    let bethe = vec![
        row("gf_ret_time", 1, 0, 0, 0, 0.0, -1.0),
        row("gf_les_freq", 1, 2, 0, 0, 0.0, 0.1),
        row("density", 1, 0, 0, 0, 1.0, 0.0),
        row("eint", 1, 0, 0, 0, -0.2, 0.0),
        row("iteration_error", 1, 1, 0, 0, 2.0, 0.0),
        row("iteration_error", 1, 2, 0, 0, 1.0, 0.0),
    ];
    let prop = vec![
        row("gf_ret_time", 2, 3, 0, 0, 0.1, -0.9),
        row("gf_les_time", 2, 3, 0, 0, 0.0, 0.5),
        row("density", 2, 0, 0, 0, 1.0, 0.0),
        row("density", 2, 9, 0, 0, 1.0, 0.0),
        row("iteration_error", 2, 0, 0, 0, 0.5, 0.0),
        row("iteration_error", 2, 4, 1, 4, 1e-9, 0.0),
        row("iteration_error", 2, 2, 2, 9, 1e-14, 0.0),
    ];
    roundtrip7(
        &dir,
        "ok_p7e_siam.dat",
        "phase7_example",
        &emeta2("siam"),
        &siam,
    );
    roundtrip7(
        &dir,
        "ok_p7e_bethe.dat",
        "phase7_example",
        &emeta2("bethe"),
        &bethe,
    );
    roundtrip7(
        &dir,
        "ok_p7e_prop.dat",
        "phase7_example",
        &emeta2("bethe_prop"),
        &prop,
    );
    // Bad inputs
    let hub = [
        ("generator", "phase7_examples_hubbard_ref"),
        ("case_set", "selftest"),
        ("case", "hubbard"),
    ];
    bad(&dir, "p7e_case.dat", "phase7_example", &hub, &siam);
    let m = [
        ("generator", "phase7_examples_siam_ref"),
        ("case_set", "selftest"),
    ];
    bad(&dir, "p7e_nocase.dat", "phase7_example", &m, &siam);
    let m = [
        ("generator", "phase7_examples_bethe_ref"),
        ("case_set", "selftest"),
        ("case", "bethe"),
    ];
    bad(&dir, "p7e_caseid.dat", "phase7_example", &m, &siam);
    let mut r = siam.clone();
    r.push(row("gf_tv", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7e_comp.dat", "phase7_example", &emeta2("siam"), &r);
    bad_edit(
        &dir,
        "p7e_ncol.dat",
        "phase7_example",
        &emeta2("siam"),
        &siam,
        Some(ncol7),
    );
    let mut r = siam.clone();
    r.push(row("gf_ret_time", 0, 6, 0, 0, f64::NAN, 0.0));
    bad(&dir, "p7e_nan.dat", "phase7_example", &emeta2("siam"), &r);
    let mut r = bethe.clone();
    r.push(bethe[bethe.len() - 1].clone());
    bad(&dir, "p7e_dup.dat", "phase7_example", &emeta2("bethe"), &r);
    let mut r = bethe.clone();
    r.push(row("current", 1, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7e_cur.dat", "phase7_example", &emeta2("bethe"), &r);
    let mut r = prop.clone();
    r.push(row("eint", 2, 0, 0, 0, 0.0, 0.0));
    bad(
        &dir,
        "p7e_eint.dat",
        "phase7_example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = prop.clone();
    r.push(row("gf_ret_freq", 2, 0, 0, 0, 0.0, 0.0));
    bad(
        &dir,
        "p7e_freq.dat",
        "phase7_example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = siam.clone();
    r.push(row("iteration_error", 0, 1, 0, 0, 1.0, 0.0));
    bad(
        &dir,
        "p7e_siamit.dat",
        "phase7_example",
        &emeta2("siam"),
        &r,
    );
    let mut r = bethe.clone();
    r.push(row("iteration_error", 1, 0, 0, 0, 1.0, 0.0));
    bad(&dir, "p7e_it0.dat", "phase7_example", &emeta2("bethe"), &r);
    let mut r = prop.clone();
    r.push(row("iteration_error", 2, 0, 3, 0, 1.0, 0.0));
    bad(
        &dir,
        "p7e_stage.dat",
        "phase7_example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = bethe.clone();
    r.push(row("iteration_error", 1, 3, 0, 0, 1.0, 1.0));
    bad(
        &dir,
        "p7e_itimag.dat",
        "phase7_example",
        &emeta2("bethe"),
        &r,
    );
    let mut r = siam.clone();
    r.push(row("current", 0, 2, 0, 0, 0.0, 0.0));
    bad(&dir, "p7e_cur2.dat", "phase7_example", &emeta2("siam"), &r);
    let mut r = bethe.clone();
    r.push(row("density", 1, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p7e_dens.dat", "phase7_example", &emeta2("bethe"), &r);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p8_moving_schema() {
    let dir = tempdir("p8m");
    let mmeta = [
        ("generator", "p8_moving_ref"),
        ("case_set", "moving"),
        ("case_id", "0"),
        ("source", "selftest"),
        ("tc", "1"),
        ("size", "2"),
        ("sign", "-1"),
        ("tstp", "3"),
    ];
    let mut mrows = Vec::new();
    for c in ["ret", "less"] {
        for i in 0..2i64 {
            for j in 0..2i64 {
                for b in 0..2i64 {
                    for a in 0..2i64 {
                        mrows.push(row(
                            c,
                            i,
                            j,
                            a,
                            b,
                            (i + 5 * j / 10) as f64,
                            (a as f64 - 0.25 * b as f64),
                        ));
                    }
                }
            }
        }
    }
    let d = roundtrip7(&dir, "ok_p8m.dat", "moving", &mmeta, &mrows);
    assert_eq!(d.rows.len(), 32);
    // Bad inputs
    bad(&dir, "p8m_kind.dat", "movin", &mmeta, &mrows);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(row("gtr", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p8m_comp.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(row("ret", 2, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p8m_win.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(row("ret", 0, -1, 0, 0, 0.0, 0.0));
    bad(&dir, "p8m_neg.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(row("less", 0, 0, 2, 0, 0.0, 0.0));
    bad(&dir, "p8m_orb.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.push(mrows[0].clone());
    bad(&dir, "p8m_dup.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    bad(&dir, "p8m_inc.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(row("ret", 0, 0, 0, 0, f64::NAN, 0.0));
    bad(&dir, "p8m_nan.dat", "moving", &mmeta, &r);
    bad_edit(&dir, "p8m_ncol.dat", "moving", &mmeta, &mrows, Some(ncol7));
    for key in ["tc", "size", "sign", "tstp", "source", "case_id"] {
        let m = without(&mmeta, key);
        let name = format!("p8m_no{key}.dat");
        bad(&dir, &name, "moving", &m, &mrows);
    }
    let mut m = mmeta.to_vec();
    m[0] = ("generator", "p8_gw_ref");
    bad(&dir, "p8m_gen.dat", "moving", &m, &mrows);
    let mut m = mmeta.to_vec();
    m[6] = ("sign", "2");
    bad(&dir, "p8m_sign.dat", "moving", &m, &mrows);
    let mut m = mmeta.to_vec();
    m[4] = ("tc", "2");
    bad(&dir, "p8m_tc.dat", "moving", &m, &mrows);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p8_series_schema() {
    let dir = tempdir("p8s");
    for cs in KBREF_SERIES_CASE_SETS {
        let cls = kbref_series_components(cs).unwrap();
        let mut comps = cls.clone();
        comps.sort_unstable();
        let generator_ = series_generator(cs).unwrap();
        let comps_str = comps.join(",");
        let smeta_vec = vec![
            ("generator", generator_),
            ("case_set", cs),
            (
                "components",
                Box::leak(comps_str.clone().into_boxed_str()) as &str,
            ),
            ("nsteps", "3"),
            ("nk", "2"),
            ("size", "2"),
            ("source", "selftest"),
        ];
        let smeta: Vec<(&str, &str)> = smeta_vec;
        let mut srows = Vec::new();
        for (n, c) in comps.iter().enumerate() {
            let is_matrix = kadanoffbaym::kbref::series_class(cs, c).unwrap() == 2;
            if is_matrix {
                srows.push(row(c, 0, 1, 1, 0, -(n as f64), 1.5));
            } else {
                srows.push(row(c, 2, -1, -1, -1, n as f64, 0.5));
            }
        }
        let name = format!("ok_p8s_{cs}.dat");
        let d = roundtrip7(&dir, &name, "series", &smeta, &srows);
        let got: std::collections::BTreeSet<&str> =
            d.rows.iter().map(|r| r.component.as_str()).collect();
        let want: std::collections::BTreeSet<&str> = comps.iter().copied().collect();
        assert_eq!(got, want);
        let cm = comps
            .iter()
            .find(|c| kadanoffbaym::kbref::series_class(cs, c).unwrap() == 2)
            .unwrap();
        let c0 = comps
            .iter()
            .find(|c| kadanoffbaym::kbref::series_class(cs, c).unwrap() == 0)
            .unwrap();
        // Bad inputs
        let mut r = srows.clone();
        r.push(row("foo", 0, 0, 0, 0, 0.0, 0.0));
        bad(&dir, &format!("p8s_{cs}_comp.dat"), "series", &smeta, &r);
        // a foreign component from another case_set
        let mut cx = "";
        'outer: for other in KBREF_SERIES_CASE_SETS {
            if other == cs {
                continue;
            }
            for c in kbref_series_components(other).unwrap() {
                if !comps.contains(&c) {
                    cx = Box::leak(c.to_string().into_boxed_str());
                    break 'outer;
                }
            }
        }
        assert!(!cx.is_empty());
        let mut r = srows.clone();
        r.push(row(cx, 0, -1, -1, -1, 0.0, 0.0));
        bad(&dir, &format!("p8s_{cs}_foreign.dat"), "series", &smeta, &r);
        let cases: Vec<(&str, KbrefRow)> = vec![
            ("step", row(cm, 3, 0, 0, 0, 0.0, 0.0)),
            ("k", row(cm, 1, 2, 0, 0, 0.0, 0.0)),
            ("orb", row(cm, 1, 0, 2, 0, 0.0, 0.0)),
            ("scal", row(c0, 1, 0, -1, -1, 0.0, 0.0)),
            ("dup", srows[0].clone()),
            ("nan", row(c0, 0, -1, -1, -1, f64::INFINITY, 0.0)),
        ];
        for (tag, extra) in cases {
            let mut r = srows.clone();
            r.push(extra);
            bad(&dir, &format!("p8s_{cs}_{tag}.dat"), "series", &smeta, &r);
        }
        bad_edit(
            &dir,
            &format!("p8s_{cs}_ncol.dat"),
            "series",
            &smeta,
            &srows,
            Some(ncol7),
        );
        let mut r = srows.clone();
        r.remove(0);
        bad(&dir, &format!("p8s_{cs}_inc.dat"), "series", &smeta, &r);
        for key in ["components", "nsteps", "nk", "size", "source"] {
            bad(
                &dir,
                &format!("p8s_{cs}_no{key}.dat"),
                "series",
                &without(&smeta, key),
                &srows,
            );
        }
        let mut m = smeta.clone();
        let g = KBREF_SERIES_CASE_SETS
            .iter()
            .map(|k| series_generator(k).unwrap())
            .find(|x| *x != generator_)
            .unwrap();
        m[0] = ("generator", g);
        bad(&dir, &format!("p8s_{cs}_gen.dat"), "series", &m, &srows);
        let mut m = smeta.clone();
        m[1] = ("case_set", "bethe");
        bad(&dir, &format!("p8s_{cs}_cs.dat"), "series", &m, &srows);
        let mut m = smeta.clone();
        let with_foo = format!("{comps_str},foo");
        m[2] = ("components", Box::leak(with_foo.into_boxed_str()));
        bad(&dir, &format!("p8s_{cs}_meta.dat"), "series", &m, &srows);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p5_observable_schema() {
    let dir = tempdir("p5o");
    for m in P5_MODELS {
        let cls = kbref_observable_components(m).unwrap();
        let mut comps = cls.clone();
        comps.sort_unstable();
        let size: i64 = if m == "holstein_nambu" || m == "holstein_impurity" {
            2
        } else {
            1
        };
        let ometa = vec![
            (
                "generator",
                Box::leak(format!("{m}_text").into_boxed_str()) as &str,
            ),
            (
                "case_set",
                Box::leak(format!("{m}_quick").into_boxed_str()) as &str,
            ),
            ("model", m),
            ("approximation", p5_approximations(m).unwrap()[0]),
            ("profile", "quick"),
            ("nt", "3"),
            ("ntau", "10"),
            ("h", "0.05"),
            ("beta", "5"),
            ("solve_order", "3"),
            ("mats_max_iter", "100"),
            ("bootstrap_max_iter", "50"),
            ("corrector_steps", "3"),
            ("size", Box::leak(size.to_string().into_boxed_str()) as &str),
            ("params", "U=1"),
        ];
        let mut orows = Vec::new();
        for (n, c) in comps.iter().enumerate() {
            let is_matrix = kadanoffbaym::kbref::observable_class(m, c).unwrap() == 2;
            for t in -1i64..=3 {
                if is_matrix {
                    for a in 0..size {
                        for b in 0..size {
                            orows.push(row(c, t, -1, a, b, n as f64 - a as f64, 0.25 * b as f64));
                        }
                    }
                } else {
                    orows.push(row(c, t, -1, -1, -1, n as f64 + 0.1 * t as f64, 0.0));
                }
            }
        }
        let d = roundtrip7(
            &dir,
            &format!("ok_p5o_{m}.dat"),
            "observable",
            &ometa,
            &orows,
        );
        assert_eq!(d.rows.len(), orows.len());
        let c0 = comps
            .iter()
            .find(|c| kadanoffbaym::kbref::observable_class(m, c).unwrap() == 0)
            .unwrap();
        bad(
            &dir,
            &format!("p5o_{m}_kind.dat"),
            "observables",
            &ometa,
            &orows,
        );
        let mut r = orows.clone();
        r.push(row("foo", 0, -1, -1, -1, 0.0, 0.0));
        bad(&dir, &format!("p5o_{m}_comp.dat"), "observable", &ometa, &r);
        // a component of another model
        let mut cx = "";
        'outer: for other in P5_MODELS {
            if other == m {
                continue;
            }
            for c in kbref_observable_components(other).unwrap() {
                if !comps.contains(&c) {
                    cx = Box::leak(c.to_string().into_boxed_str());
                    break 'outer;
                }
            }
        }
        assert!(!cx.is_empty());
        let mut r = orows.clone();
        r.push(row(cx, 0, -1, -1, -1, 0.0, 0.0));
        bad(
            &dir,
            &format!("p5o_{m}_foreign.dat"),
            "observable",
            &ometa,
            &r,
        );
        bad_edit(
            &dir,
            &format!("p5o_{m}_ncol.dat"),
            "observable",
            &ometa,
            &orows,
            Some(ncol7),
        );
        let mut r = orows.clone();
        r.push(orows[0].clone());
        bad(&dir, &format!("p5o_{m}_dup.dat"), "observable", &ometa, &r);
        let mut r = orows.clone();
        r.push(row(c0, 3, -1, -1, -1, f64::NAN, 0.0));
        r.remove(0);
        bad(&dir, &format!("p5o_{m}_nan.dat"), "observable", &ometa, &r);
        let mut r = orows.clone();
        r.remove(0);
        r.push(row(
            orows[0].component.as_str(),
            orows[0].i,
            -1,
            orows[0].a,
            orows[0].b,
            f64::INFINITY,
            0.0,
        ));
        bad(&dir, &format!("p5o_{m}_inf.dat"), "observable", &ometa, &r);
        let cases: Vec<(&str, KbrefRow)> = vec![
            ("tstp", row(c0, 4, -1, -1, -1, 0.0, 0.0)),
            ("tstp2", row(c0, -2, -1, -1, -1, 0.0, 0.0)),
            ("sent", row(c0, -1, -1, 0, -1, 1.0, 0.0)),
            ("j", row(c0, -1, 0, -1, -1, 1.0, 0.0)),
            (
                "im",
                row(
                    orows[0].component.as_str(),
                    orows[0].i,
                    -1,
                    -1,
                    -1,
                    1.0,
                    1.0,
                ),
            ),
        ];
        for (tag, extra) in cases {
            let mut r = orows.clone();
            r.remove(0);
            r.push(extra);
            bad(
                &dir,
                &format!("p5o_{m}_{tag}.dat"),
                "observable",
                &ometa,
                &r,
            );
        }
        if comps.contains(&"rho") {
            let mut r = orows.clone();
            r.push(row("rho", 0, -1, size, 0, 0.0, 0.0));
            bad(&dir, &format!("p5o_{m}_orb.dat"), "observable", &ometa, &r);
        }
        for key in KBREF_P5_META {
            bad(
                &dir,
                &format!("p5o_{m}_no{key}.dat"),
                "observable",
                &without(&ometa, key),
                &orows,
            );
        }
        let mut mm = ometa.clone();
        mm[3] = ("approximation", "RPA");
        bad(
            &dir,
            &format!("p5o_{m}_approx.dat"),
            "observable",
            &mm,
            &orows,
        );
        // model/approximation mismatch
        let approx0 = p5_approximations(m).unwrap()[0];
        let mo = P5_MODELS
            .iter()
            .find(|k| !p5_approximations(k).unwrap().contains(&approx0))
            .unwrap();
        let mut mm = ometa.clone();
        mm[2] = ("model", mo);
        bad(
            &dir,
            &format!("p5o_{m}_model.dat"),
            "observable",
            &mm,
            &orows,
        );
        let mut mm = ometa.clone();
        mm[4] = ("profile", "fast");
        bad(
            &dir,
            &format!("p5o_{m}_prof.dat"),
            "observable",
            &mm,
            &orows,
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p5_selected_green_schema() {
    let dir = tempdir("p5g");
    let gmeta = [
        ("generator", "hubbard_chain_text"),
        ("case_set", "hubbard_2b_quick"),
        ("model", "hubbard"),
        ("approximation", "2B"),
        ("profile", "quick"),
        ("nt", "10"),
        ("ntau", "40"),
        ("h", "0.05"),
        ("beta", "20"),
        ("solve_order", "3"),
        ("mats_max_iter", "100"),
        ("bootstrap_max_iter", "50"),
        ("corrector_steps", "3"),
        ("size", "2"),
        ("params", "U=1"),
    ];
    let grows: Vec<KbrefRow> = KBREF_SELECTED_GREEN_KEYS
        .iter()
        .enumerate()
        .map(|(n, (c, i, j, a, b))| {
            row(
                c,
                *i,
                *j,
                *a,
                *b,
                0.1 * (n + 1) as f64,
                -0.2 * (n + 1) as f64,
            )
        })
        .collect();
    let d = roundtrip7(&dir, "ok_p5g.dat", "selected_green", &gmeta, &grows);
    assert_eq!(d.rows.len(), 4);
    let mut r = grows.clone();
    r.pop();
    r.push(row("tv", 3, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p5g_key.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    r.push(row("gtr", 3, 2, 0, 0, 0.0, 0.0));
    bad(&dir, "p5g_comp.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    bad(&dir, "p5g_inc.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.push(grows[0].clone());
    bad(&dir, "p5g_dup.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    let k4 = KBREF_SELECTED_GREEN_KEYS[3];
    r.push(row(k4.0, k4.1, k4.2, k4.3, k4.4, 0.0, f64::NAN));
    bad(&dir, "p5g_nan.dat", "selected_green", &gmeta, &r);
    bad_edit(
        &dir,
        "p5g_ncol.dat",
        "selected_green",
        &gmeta,
        &grows,
        Some(ncol7),
    );
    let mut m = gmeta.to_vec();
    m[2] = ("model", "fkm");
    m[3] = ("approximation", "none");
    bad(&dir, "p5g_model.dat", "selected_green", &m, &grows);
    for key in KBREF_P5_META {
        bad(
            &dir,
            &format!("p5g_no{key}.dat"),
            "selected_green",
            &without(&gmeta, key),
            &grows,
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p1_data_files_are_still_accepted() {
    let data = format!("{}/data/test", env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<String> = Vec::new();
    for entry in fs::read_dir(&data).unwrap() {
        let f = entry.unwrap().file_name().into_string().unwrap();
        if f.ends_with(".dat")
            && !f.starts_with("p2_")
            && !f.starts_with("phase7_")
            && !f.starts_with("p8_")
            && !f.starts_with("hubbard_")
            && !f.starts_with("fkm_")
            && !f.starts_with("holstein_")
        {
            files.push(f);
        }
    }
    assert_eq!(files.len(), 21);
    for f in &files {
        let d = read_kbref(&format!("{data}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert!(
            d.metadata["data_kind"] == "weights" || d.metadata["data_kind"] == "herm_matrix",
            "{}: unexpected kind {}",
            f,
            d.metadata["data_kind"]
        );
    }
}

#[test]
fn fmt_writer_uses_c_style_floats() {
    // write_kbref must produce C-compatible %.17e output (checked
    // against the fmt unit test; here we only assert the line shape)
    let dir = tempdir("fmt");
    let p = dir.join("fmt.dat");
    let rows = vec![row("greg", 1, 0, 0, -1, 0.5, 0.0)];
    write_kbref(
        p.to_str().unwrap(),
        "weights",
        &[("case".to_string(), "x".to_string())],
        &rows,
    )
    .unwrap();
    let text = fs::read_to_string(&p).unwrap();
    let last = text.lines().last().unwrap();
    assert_eq!(
        last,
        format!("greg 1 0 0 -1 {} {}", fmt_c_e17(0.5), fmt_c_e17(0.0))
    );
    let _ = fs::remove_dir_all(&dir);
}
