// Project : KadanoffBaym
// File    : tests/ref_schema/moving_series.rs
// Purpose : the moving-window and series schema contract tests
//

use std::collections::BTreeSet;

use std::fs::remove_dir_all;

use super::util::series_class;

use super::support::*;

#[test]
fn moving_schema() {
    let dir = tempdir_path("p8m");
    let mmeta = [
        ("generator", "moving_ref"),
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
                        mrows.push(reference_row(
                            c,
                            i,
                            j,
                            a,
                            b,
                            (i + 5 * j / 10) as f64,
                            a as f64 - 0.25 * b as f64,
                        ));
                    }
                }
            }
        }
    }
    let d = roundtrip_with_extra_column(&dir, "ok_p8m.dat", "moving", &mmeta, &mrows);
    assert_eq!(d.rows.len(), 32);
    // Bad inputs
    expect_rejected(&dir, "p8m_kind.dat", "movin", &mmeta, &mrows);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(reference_row("gtr", 0, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "p8m_comp.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(reference_row("ret", 2, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "p8m_win.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(reference_row("ret", 0, -1, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "p8m_neg.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(reference_row("less", 0, 0, 2, 0, 0.0, 0.0));
    expect_rejected(&dir, "p8m_orb.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.push(mrows[0].clone());
    expect_rejected(&dir, "p8m_dup.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    expect_rejected(&dir, "p8m_inc.dat", "moving", &mmeta, &r);
    let mut r = mrows.clone();
    r.remove(0);
    r.push(reference_row("ret", 0, 0, 0, 0, f64::NAN, 0.0));
    expect_rejected(&dir, "p8m_nan.dat", "moving", &mmeta, &r);
    expect_rejected_after_edit(
        &dir,
        "p8m_ncol.dat",
        "moving",
        &mmeta,
        &mrows,
        Some(with_seventh_column),
    );
    for key in ["tc", "size", "sign", "tstp", "source", "case_id"] {
        let m = without_key(&mmeta, key);
        let name = format!("p8m_no{key}.dat");
        expect_rejected(&dir, &name, "moving", &m, &mrows);
    }
    let mut m = mmeta.to_vec();
    m[0] = ("generator", "gw_ref");
    expect_rejected(&dir, "p8m_gen.dat", "moving", &m, &mrows);
    let mut m = mmeta.to_vec();
    m[6] = ("sign", "2");
    expect_rejected(&dir, "p8m_sign.dat", "moving", &m, &mrows);
    let mut m = mmeta.to_vec();
    m[4] = ("tc", "2");
    expect_rejected(&dir, "p8m_tc.dat", "moving", &m, &mrows);
    let _ = remove_dir_all(&dir);
}

#[test]
fn series_schema() {
    let dir = tempdir_path("p8s");
    for cs in REF_SERIES_CASE_SETS {
        let cls = series_components(cs).unwrap();
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
            let is_matrix = series_class(cs, c).unwrap() == 2;
            if is_matrix {
                srows.push(reference_row(c, 0, 1, 1, 0, -(n as f64), 1.5));
            } else {
                srows.push(reference_row(c, 2, -1, -1, -1, n as f64, 0.5));
            }
        }
        let name = format!("ok_p8s_{cs}.dat");
        let d = roundtrip_with_extra_column(&dir, &name, "series", &smeta, &srows);
        let got: BTreeSet<&str> = d.rows.iter().map(|r| r.component.as_str()).collect();
        let want: BTreeSet<&str> = comps.iter().copied().collect();
        assert_eq!(got, want);
        let cm = comps
            .iter()
            .find(|c| series_class(cs, c).unwrap() == 2)
            .unwrap();
        let c0 = comps
            .iter()
            .find(|c| series_class(cs, c).unwrap() == 0)
            .unwrap();
        // Bad inputs
        let mut r = srows.clone();
        r.push(reference_row("foo", 0, 0, 0, 0, 0.0, 0.0));
        expect_rejected(&dir, &format!("p8s_{cs}_comp.dat"), "series", &smeta, &r);
        // a foreign component from another case_set
        let mut cx = "";
        'outer: for other in REF_SERIES_CASE_SETS {
            if other == cs {
                continue;
            }
            for c in series_components(other).unwrap() {
                if !comps.contains(&c) {
                    cx = Box::leak(c.to_string().into_boxed_str());
                    break 'outer;
                }
            }
        }
        assert!(!cx.is_empty());
        let mut r = srows.clone();
        r.push(reference_row(cx, 0, -1, -1, -1, 0.0, 0.0));
        expect_rejected(&dir, &format!("p8s_{cs}_foreign.dat"), "series", &smeta, &r);
        let cases: Vec<(&str, RefRow)> = vec![
            ("step", reference_row(cm, 3, 0, 0, 0, 0.0, 0.0)),
            ("k", reference_row(cm, 1, 2, 0, 0, 0.0, 0.0)),
            ("orb", reference_row(cm, 1, 0, 2, 0, 0.0, 0.0)),
            ("scal", reference_row(c0, 1, 0, -1, -1, 0.0, 0.0)),
            ("dup", srows[0].clone()),
            ("nan", reference_row(c0, 0, -1, -1, -1, f64::INFINITY, 0.0)),
        ];
        for (tag, extra) in cases {
            let mut r = srows.clone();
            r.push(extra);
            expect_rejected(&dir, &format!("p8s_{cs}_{tag}.dat"), "series", &smeta, &r);
        }
        expect_rejected_after_edit(
            &dir,
            &format!("p8s_{cs}_ncol.dat"),
            "series",
            &smeta,
            &srows,
            Some(with_seventh_column),
        );
        let mut r = srows.clone();
        r.remove(0);
        expect_rejected(&dir, &format!("p8s_{cs}_inc.dat"), "series", &smeta, &r);
        for key in ["components", "nsteps", "nk", "size", "source"] {
            expect_rejected(
                &dir,
                &format!("p8s_{cs}_no{key}.dat"),
                "series",
                &without_key(&smeta, key),
                &srows,
            );
        }
        let mut m = smeta.clone();
        let g = REF_SERIES_CASE_SETS
            .iter()
            .map(|k| series_generator(k).unwrap())
            .find(|x| *x != generator_)
            .unwrap();
        m[0] = ("generator", g);
        expect_rejected(&dir, &format!("p8s_{cs}_gen.dat"), "series", &m, &srows);
        let mut m = smeta.clone();
        m[1] = ("case_set", "bethe");
        expect_rejected(&dir, &format!("p8s_{cs}_cs.dat"), "series", &m, &srows);
        let mut m = smeta.clone();
        let with_foo = format!("{comps_str},foo");
        m[2] = ("components", Box::leak(with_foo.into_boxed_str()));
        expect_rejected(&dir, &format!("p8s_{cs}_meta.dat"), "series", &m, &srows);
    }
    let _ = remove_dir_all(&dir);
}
