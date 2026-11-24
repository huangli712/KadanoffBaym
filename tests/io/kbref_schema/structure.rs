// Project : KadanoffBaym
// File    : tests/io/kbref_schema/structure.rs
// Purpose : container, tool, iteration and pseudo schema contract tests
//

use super::*;

#[test]
fn herm_matrix_and_weights_schema() {
    let dir = tempdir_path("p1");
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
        Some(|s| s.replace("les 1 1 0 0 0.0", "les 1 1 0 0 0.0 0")),
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

#[test]
fn tool_scalar_function_element_labels() {
    let dir = tempdir_path("p2");
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
    assert_eq!(m[&1].alpha, Some(Complex::new(0.5, -0.25)));
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
fn iteration_and_single_case_schema() {
    let dir = tempdir_path("p4");
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
fn pseudo_schema() {
    // NOTE: the `kbref_to_pseudo` rebuild assertions of the blueprint
    // test need PseudoGreen and land with R8; the schema contract is
    // fully covered here.
    let dir = tempdir_path("p6");
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
    let v = |n: &mut i64| {
        *n += 1;
        Complex::new(*n as f64 / 7.0, -(*n as f64) / 11.0)
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
