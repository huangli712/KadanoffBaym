// Project : KadanoffBaym
// File    : tests/ref_schema/reference.rs
// Purpose : the reference-file schema contract tests
//

use std::collections::BTreeSet;

use std::fs::remove_dir_all;

use super::support::*;

#[test]
fn reference_fourier_schema() {
    let dir = tempdir_path("p7f");
    let fmeta = [
        ("generator", "fourier_ref"),
        ("case_set", "selftest"),
        ("size_max", "2"),
    ];
    let v = |n: f64| Complex::new(n / 3.0, -n / 5.0);
    let mut frows = Vec::new();
    for c in ["fft_input", "fft_forward", "fft_backward"] {
        for j in 0..3i64 {
            frows.push(reference_row(c, 3, j, 0, 0, v(j as f64).re, v(j as f64).im));
        }
    }
    frows.push(reference_row("dftcorr", 0, 1, 0, 0, 0.7, 0.0));
    frows.push(reference_row("dftcorr", 0, 1, 1, 0, 0.9, 0.0));
    frows.push(reference_row("dftcorr", 0, 1, 3, 0, v(1.0).re, v(1.0).im));
    frows.push(reference_row("dftcorr", 0, 3, 5, 0, v(2.0).re, v(2.0).im));
    frows.push(reference_row("adaptive", 2, 4, 0, 0, -1.5, 0.0));
    frows.push(reference_row("adaptive", 2, 4, 2, 0, v(3.0).re, v(3.0).im));
    frows.push(reference_row(
        "matsubara_dft_result",
        1,
        5,
        1,
        0,
        v(4.0).re,
        v(4.0).im,
    ));
    frows.push(reference_row(
        "matsubara_ft_result",
        13,
        -25,
        0,
        1,
        v(5.0).re,
        v(5.0).im,
    ));
    frows.push(reference_row(
        "vie2_pcf20",
        0,
        7,
        1,
        1,
        v(6.0).re,
        v(6.0).im,
    ));
    frows.push(reference_row(
        "vie2_pcf10",
        0,
        7,
        1,
        1,
        v(7.0).re,
        v(7.0).im,
    ));
    let d = roundtrip_with_extra_column(&dir, "ok_fourier.dat", "fourier", &fmeta, &frows);
    let comps: BTreeSet<&str> = d.rows.iter().map(|r| r.component.as_str()).collect();
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
    expect_rejected(&dir, "fourier_kind.dat", "fft", &fmeta, &frows);
    let mut r = frows.clone();
    r.push(reference_row("fft", 3, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_comp.dat", "fourier", &fmeta, &r);
    expect_rejected_after_edit(
        &dir,
        "fourier_ncol.dat",
        "fourier",
        &fmeta,
        &frows,
        Some(with_seventh_column),
    );
    let mut r = frows.clone();
    r.push(reference_row("vie2_pcf10", 0, 8, 0, 0, f64::INFINITY, 0.0));
    expect_rejected(&dir, "fourier_nan.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("fft_input", 3, 3, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_fftj.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("fft_forward", 2, 0, 1, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_ffta.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.remove(0);
    expect_rejected(&dir, "fourier_fftinc.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("dftcorr", 1, 2, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_order.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("dftcorr", 1, 1, 4, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_alpha.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("dftcorr", 1, 3, 0, 0, 0.0, 1.0));
    expect_rejected(&dir, "fourier_theta.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("adaptive", 3, 0, 1, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_fid.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("matsubara_ft_result", 12, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_ftord.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(reference_row("vie2_pcf20", 0, 0, 2, 0, 0.0, 0.0));
    expect_rejected(&dir, "fourier_orb.dat", "fourier", &fmeta, &r);
    let mut r = frows.clone();
    r.push(frows[frows.len() - 1].clone());
    expect_rejected(&dir, "fourier_dup.dat", "fourier", &fmeta, &r);
    expect_rejected(
        &dir,
        "fourier_nosize.dat",
        "fourier",
        &without_key(&fmeta, "size_max"),
        &frows,
    );
    expect_rejected(
        &dir,
        "fourier_noset.dat",
        "fourier",
        &without_key(&fmeta, "case_set"),
        &frows,
    );
    let _ = remove_dir_all(&dir);
}

#[test]
fn reference_steady_schema() {
    let dir = tempdir_path("p7n");
    let nmeta = [
        ("generator", "steady_ref"),
        ("case_set", "selftest"),
        ("nft_max", "4"),
        ("size_max", "2"),
    ];
    let v = |n: f64| Complex::new(-n / 7.0, n / 9.0);
    let mut nrows = Vec::new();
    let mut n = 0i64;
    for c in ["ret_time", "les_time", "ret_freq", "les_freq"] {
        n += 1;
        nrows.push(reference_row(c, 1, 3, 1, 0, v(n as f64).re, v(n as f64).im));
    }
    for c in ["bath_ret", "bath_les"] {
        n += 1;
        nrows.push(reference_row(
            c,
            3,
            12,
            0,
            0,
            v(n as f64).re,
            v(n as f64).im,
        ));
    }
    for c in ["density", "convolution_density"] {
        n += 1;
        nrows.push(reference_row(c, 0, 0, 1, 1, v(n as f64).re, v(n as f64).im));
    }
    for c in [
        "bubble1",
        "bubble2",
        "resample",
        "force_equilibrium",
        "conversion",
    ] {
        n += 1;
        nrows.push(reference_row(c, 0, 1, 0, 1, v(n as f64).re, v(n as f64).im));
        n += 1;
        nrows.push(reference_row(c, 0, 7, 0, 1, v(n as f64).re, v(n as f64).im));
    }
    let d = roundtrip_with_extra_column(&dir, "ok_steady.dat", "steady", &nmeta, &nrows);
    assert_eq!(d.rows.len(), nrows.len());
    // Bad inputs
    let mut r = nrows.clone();
    r.push(reference_row("gtr_time", 0, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_comp.dat", "steady", &nmeta, &r);
    expect_rejected_after_edit(
        &dir,
        "steady_ncol.dat",
        "steady",
        &nmeta,
        &nrows,
        Some(with_seventh_column),
    );
    let mut r = nrows.clone();
    r.push(reference_row("ret_time", 0, 0, 0, 0, 0.0, f64::NAN));
    expect_rejected(&dir, "steady_nan.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("ret_freq", 0, 4, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_mode.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("bubble1", 0, 8, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_mode2.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("les_time", 0, 0, 0, 2, 0.0, 0.0));
    expect_rejected(&dir, "steady_orb.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("bath_ret", 4, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_bath.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("bath_les", 1, 0, 1, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_batho.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(reference_row("density", 0, 1, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "steady_dens.dat", "steady", &nmeta, &r);
    let mut r = nrows.clone();
    r.push(nrows[0].clone());
    expect_rejected(&dir, "steady_dup.dat", "steady", &nmeta, &r);
    expect_rejected(
        &dir,
        "steady_nonft.dat",
        "steady",
        &without_key(&nmeta, "nft_max"),
        &nrows,
    );
    let _ = remove_dir_all(&dir);
}

#[test]
fn reference_example_schema() {
    let dir = tempdir_path("p7e");
    let emeta2 = |c: &'static str| {
        vec![
            (
                "generator",
                Box::leak(format!("examples_{c}_ref").into_boxed_str()) as &'static str,
            ),
            ("case_set", "selftest"),
            ("case", c),
        ]
    };
    let _ = emeta2;
    let siam = vec![
        reference_row("gf_ret_time", 0, 5, 0, 0, 0.1, -0.2),
        reference_row("gf_les_time", 0, 5, 0, 0, 0.0, 0.3),
        reference_row("gf_ret_freq", 0, 9, 0, 0, 0.0, -0.4),
        reference_row("gf_les_freq", 0, 9, 0, 0, 0.0, 0.5),
        reference_row("density", 0, 0, 0, 0, 0.9, 0.0),
        reference_row("current", 0, 0, 0, 0, 0.3, 0.0),
        reference_row("current", 0, 1, 0, 0, 0.3, 0.0),
        reference_row("eint", 0, 0, 0, 0, -0.1, 0.0),
    ];
    let bethe = vec![
        reference_row("gf_ret_time", 1, 0, 0, 0, 0.0, -1.0),
        reference_row("gf_les_freq", 1, 2, 0, 0, 0.0, 0.1),
        reference_row("density", 1, 0, 0, 0, 1.0, 0.0),
        reference_row("eint", 1, 0, 0, 0, -0.2, 0.0),
        reference_row("iteration_error", 1, 1, 0, 0, 2.0, 0.0),
        reference_row("iteration_error", 1, 2, 0, 0, 1.0, 0.0),
    ];
    let prop = vec![
        reference_row("gf_ret_time", 2, 3, 0, 0, 0.1, -0.9),
        reference_row("gf_les_time", 2, 3, 0, 0, 0.0, 0.5),
        reference_row("density", 2, 0, 0, 0, 1.0, 0.0),
        reference_row("density", 2, 9, 0, 0, 1.0, 0.0),
        reference_row("iteration_error", 2, 0, 0, 0, 0.5, 0.0),
        reference_row("iteration_error", 2, 4, 1, 4, 1e-9, 0.0),
        reference_row("iteration_error", 2, 2, 2, 9, 1e-14, 0.0),
    ];
    roundtrip_with_extra_column(
        &dir,
        "ok_example_siam.dat",
        "example",
        &emeta2("siam"),
        &siam,
    );
    roundtrip_with_extra_column(
        &dir,
        "ok_example_bethe.dat",
        "example",
        &emeta2("bethe"),
        &bethe,
    );
    roundtrip_with_extra_column(
        &dir,
        "ok_example_prop.dat",
        "example",
        &emeta2("bethe_prop"),
        &prop,
    );
    // Bad inputs
    let hub = [
        ("generator", "examples_hubbard_ref"),
        ("case_set", "selftest"),
        ("case", "hubbard"),
    ];
    expect_rejected(&dir, "example_case.dat", "example", &hub, &siam);
    let m = [("generator", "examples_siam_ref"), ("case_set", "selftest")];
    expect_rejected(&dir, "example_nocase.dat", "example", &m, &siam);
    let m = [
        ("generator", "examples_bethe_ref"),
        ("case_set", "selftest"),
        ("case", "bethe"),
    ];
    expect_rejected(&dir, "example_caseid.dat", "example", &m, &siam);
    let mut r = siam.clone();
    r.push(reference_row("gf_tv", 0, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "example_comp.dat", "example", &emeta2("siam"), &r);
    expect_rejected_after_edit(
        &dir,
        "example_ncol.dat",
        "example",
        &emeta2("siam"),
        &siam,
        Some(with_seventh_column),
    );
    let mut r = siam.clone();
    r.push(reference_row("gf_ret_time", 0, 6, 0, 0, f64::NAN, 0.0));
    expect_rejected(&dir, "example_nan.dat", "example", &emeta2("siam"), &r);
    let mut r = bethe.clone();
    r.push(bethe[bethe.len() - 1].clone());
    expect_rejected(&dir, "example_dup.dat", "example", &emeta2("bethe"), &r);
    let mut r = bethe.clone();
    r.push(reference_row("current", 1, 0, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "example_cur.dat", "example", &emeta2("bethe"), &r);
    let mut r = prop.clone();
    r.push(reference_row("eint", 2, 0, 0, 0, 0.0, 0.0));
    expect_rejected(
        &dir,
        "example_eint.dat",
        "example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = prop.clone();
    r.push(reference_row("gf_ret_freq", 2, 0, 0, 0, 0.0, 0.0));
    expect_rejected(
        &dir,
        "example_freq.dat",
        "example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = siam.clone();
    r.push(reference_row("iteration_error", 0, 1, 0, 0, 1.0, 0.0));
    expect_rejected(&dir, "example_siamit.dat", "example", &emeta2("siam"), &r);
    let mut r = bethe.clone();
    r.push(reference_row("iteration_error", 1, 0, 0, 0, 1.0, 0.0));
    expect_rejected(&dir, "example_it0.dat", "example", &emeta2("bethe"), &r);
    let mut r = prop.clone();
    r.push(reference_row("iteration_error", 2, 0, 3, 0, 1.0, 0.0));
    expect_rejected(
        &dir,
        "example_stage.dat",
        "example",
        &emeta2("bethe_prop"),
        &r,
    );
    let mut r = bethe.clone();
    r.push(reference_row("iteration_error", 1, 3, 0, 0, 1.0, 1.0));
    expect_rejected(&dir, "example_itimag.dat", "example", &emeta2("bethe"), &r);
    let mut r = siam.clone();
    r.push(reference_row("current", 0, 2, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "example_cur2.dat", "example", &emeta2("siam"), &r);
    let mut r = bethe.clone();
    r.push(reference_row("density", 1, 1, 0, 0, 0.0, 0.0));
    expect_rejected(&dir, "example_dens.dat", "example", &emeta2("bethe"), &r);
    let _ = remove_dir_all(&dir);
}
