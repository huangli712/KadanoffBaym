// Project : KadanoffBaym
// File    : tests/ref_schema/observable.rs
// Purpose : the observable and selected-green schema contract
//           tests
//

use std::fs::remove_dir_all;

use super::util::observable_class;

use super::support::*;

#[test]
fn observable_schema() {
    let dir = tempdir_path("p5o");
    for m in OBSERVABLE_MODELS {
        let cls = observable_components(m).unwrap();
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
            ("approximation", approximations(m).unwrap()[0]),
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
            let is_matrix = observable_class(m, c).unwrap() == 2;
            for t in -1i64..=3 {
                if is_matrix {
                    for a in 0..size {
                        for b in 0..size {
                            orows.push(reference_row(
                                c,
                                t,
                                -1,
                                a,
                                b,
                                n as f64 - a as f64,
                                0.25 * b as f64,
                            ));
                        }
                    }
                } else {
                    orows.push(reference_row(
                        c,
                        t,
                        -1,
                        -1,
                        -1,
                        n as f64 + 0.1 * t as f64,
                        0.0,
                    ));
                }
            }
        }
        let d = roundtrip_with_extra_column(
            &dir,
            &format!("ok_p5o_{m}.dat"),
            "observable",
            &ometa,
            &orows,
        );
        assert_eq!(d.rows.len(), orows.len());
        let c0 = comps
            .iter()
            .find(|c| observable_class(m, c).unwrap() == 0)
            .unwrap();
        expect_rejected(
            &dir,
            &format!("p5o_{m}_kind.dat"),
            "observables",
            &ometa,
            &orows,
        );
        let mut r = orows.clone();
        r.push(reference_row("foo", 0, -1, -1, -1, 0.0, 0.0));
        expect_rejected(&dir, &format!("p5o_{m}_comp.dat"), "observable", &ometa, &r);
        // a component of another model
        let mut cx = "";
        'outer: for other in OBSERVABLE_MODELS {
            if other == m {
                continue;
            }
            for c in observable_components(other).unwrap() {
                if !comps.contains(&c) {
                    cx = Box::leak(c.to_string().into_boxed_str());
                    break 'outer;
                }
            }
        }
        assert!(!cx.is_empty());
        let mut r = orows.clone();
        r.push(reference_row(cx, 0, -1, -1, -1, 0.0, 0.0));
        expect_rejected(
            &dir,
            &format!("p5o_{m}_foreign.dat"),
            "observable",
            &ometa,
            &r,
        );
        expect_rejected_after_edit(
            &dir,
            &format!("p5o_{m}_ncol.dat"),
            "observable",
            &ometa,
            &orows,
            Some(with_seventh_column),
        );
        let mut r = orows.clone();
        r.push(orows[0].clone());
        expect_rejected(&dir, &format!("p5o_{m}_dup.dat"), "observable", &ometa, &r);
        let mut r = orows.clone();
        r.push(reference_row(c0, 3, -1, -1, -1, f64::NAN, 0.0));
        r.remove(0);
        expect_rejected(&dir, &format!("p5o_{m}_nan.dat"), "observable", &ometa, &r);
        let mut r = orows.clone();
        r.remove(0);
        r.push(reference_row(
            orows[0].component.as_str(),
            orows[0].i,
            -1,
            orows[0].a,
            orows[0].b,
            f64::INFINITY,
            0.0,
        ));
        expect_rejected(&dir, &format!("p5o_{m}_inf.dat"), "observable", &ometa, &r);
        let cases: Vec<(&str, RefRow)> = vec![
            ("tstp", reference_row(c0, 4, -1, -1, -1, 0.0, 0.0)),
            ("tstp2", reference_row(c0, -2, -1, -1, -1, 0.0, 0.0)),
            ("sent", reference_row(c0, -1, -1, 0, -1, 1.0, 0.0)),
            ("j", reference_row(c0, -1, 0, -1, -1, 1.0, 0.0)),
            (
                "im",
                reference_row(
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
            expect_rejected(
                &dir,
                &format!("p5o_{m}_{tag}.dat"),
                "observable",
                &ometa,
                &r,
            );
        }
        if comps.contains(&"rho") {
            let mut r = orows.clone();
            r.push(reference_row("rho", 0, -1, size, 0, 0.0, 0.0));
            expect_rejected(&dir, &format!("p5o_{m}_orb.dat"), "observable", &ometa, &r);
        }
        for key in REF_OBSERVABLE_META {
            expect_rejected(
                &dir,
                &format!("p5o_{m}_no{key}.dat"),
                "observable",
                &without_key(&ometa, key),
                &orows,
            );
        }
        let mut mm = ometa.clone();
        mm[3] = ("approximation", "RPA");
        expect_rejected(
            &dir,
            &format!("p5o_{m}_approx.dat"),
            "observable",
            &mm,
            &orows,
        );
        // model/approximation mismatch
        let approx0 = approximations(m).unwrap()[0];
        let mo = OBSERVABLE_MODELS
            .iter()
            .find(|k| !approximations(k).unwrap().contains(&approx0))
            .unwrap();
        let mut mm = ometa.clone();
        mm[2] = ("model", mo);
        expect_rejected(
            &dir,
            &format!("p5o_{m}_model.dat"),
            "observable",
            &mm,
            &orows,
        );
        let mut mm = ometa.clone();
        mm[4] = ("profile", "fast");
        expect_rejected(
            &dir,
            &format!("p5o_{m}_prof.dat"),
            "observable",
            &mm,
            &orows,
        );
    }
    let _ = remove_dir_all(&dir);
}

#[test]
fn selected_green_schema() {
    let dir = tempdir_path("p5g");
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
    let grows: Vec<RefRow> = REF_SELECTED_GREEN_KEYS
        .iter()
        .enumerate()
        .map(|(n, (c, i, j, a, b))| {
            reference_row(
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
    let d = roundtrip_with_extra_column(&dir, "ok_p5g.dat", "selected_green", &gmeta, &grows);
    assert_eq!(d.rows.len(), 4);
    let mut r = grows.clone();
    r.pop();
    r.push(reference_row("tv", 3, 1, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "p5g_key.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    r.push(reference_row("gtr", 3, 2, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "p5g_comp.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    expect_rejected(&dir, "p5g_inc.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.push(grows[0].clone());
    expect_rejected(&dir, "p5g_dup.dat", "selected_green", &gmeta, &r);
    let mut r = grows.clone();
    r.pop();
    let k4 = REF_SELECTED_GREEN_KEYS[3];
    r.push(reference_row(k4.0, k4.1, k4.2, k4.3, k4.4, 0.0, f64::NAN));
    expect_rejected(&dir, "p5g_nan.dat", "selected_green", &gmeta, &r);
    expect_rejected_after_edit(
        &dir,
        "p5g_ncol.dat",
        "selected_green",
        &gmeta,
        &grows,
        Some(with_seventh_column),
    );
    let mut m = gmeta.to_vec();
    m[2] = ("model", "fkm");
    m[3] = ("approximation", "none");
    expect_rejected(&dir, "p5g_model.dat", "selected_green", &m, &grows);
    for key in REF_OBSERVABLE_META {
        expect_rejected(
            &dir,
            &format!("p5g_no{key}.dat"),
            "selected_green",
            &without_key(&gmeta, key),
            &grows,
        );
    }
    let _ = remove_dir_all(&dir);
}
