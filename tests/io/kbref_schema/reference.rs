// Project : KadanoffBaym
// File    : tests/io/kbref_schema/reference.rs
// Purpose : NESSi reference-file schema contract tests
//

use super::*;

#[test]
fn reference_fourier_schema() {
    let dir = tempdir_path("p7f");
    let fmeta = [
        ("generator", "phase7_fourier_ref"),
        ("case_set", "selftest"),
        ("size_max", "2"),
    ];
    let v = |n: f64| Complex::new(n / 3.0, -n / 5.0);
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
fn reference_ness_schema() {
    let dir = tempdir_path("p7n");
    let nmeta = [
        ("generator", "phase7_steady_ref"),
        ("case_set", "selftest"),
        ("nft_max", "4"),
        ("size_max", "2"),
    ];
    let v = |n: f64| Complex::new(-n / 7.0, n / 9.0);
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
    let d = roundtrip7(&dir, "ok_p7n.dat", "phase7_steady", &nmeta, &nrows);
    assert_eq!(d.rows.len(), nrows.len());
    // Bad inputs
    let mut r = nrows.clone();
    r.push(row("gtr_time", 0, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_comp.dat", "phase7_steady", &nmeta, &r);
    bad_edit(
        &dir,
        "p7n_ncol.dat",
        "phase7_steady",
        &nmeta,
        &nrows,
        Some(ncol7),
    );
    let mut r = nrows.clone();
    r.push(row("ret_time", 0, 0, 0, 0, 0.0, f64::NAN));
    bad(&dir, "p7n_nan.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("ret_freq", 0, 4, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_mode.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bubble1", 0, 8, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_mode2.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("les_time", 0, 0, 0, 2, 0.0, 0.0));
    bad(&dir, "p7n_orb.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bath_ret", 4, 0, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_bath.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("bath_les", 1, 0, 1, 0, 0.0, 0.0));
    bad(&dir, "p7n_batho.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(row("density", 0, 1, 0, 0, 0.0, 0.0));
    bad(&dir, "p7n_dens.dat", "phase7_steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(nrows[0].clone());
    bad(&dir, "p7n_dup.dat", "phase7_steady", &nmeta, &r);
    bad(
        &dir,
        "p7n_nonft.dat",
        "phase7_steady",
        &without(&nmeta, "nft_max"),
        &nrows,
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn reference_example_schema() {
    let dir = tempdir_path("p7e");
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
