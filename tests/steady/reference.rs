// Project : KadanoffBaym
// File    : tests/steady/reference.rs
// Purpose : point-wise comparison of the steady-state algorithms with
//           the reference data
//
// Tolerances: FFT and the corrected DFT atol = rtol = 1e-12, all the
// other results atol = rtol = 1e-10. The quick profile consumes a
// fixed subset of the records of every component, the release profile
// all records.
//
// Deviation: force_equilibrium of reference 2.0.0 returns -1 times the
// physical lesser component; KB uses the physical sign, so the lesser
// records of `force_equilibrium` are compared with -1 times the
// reference.

use std::collections::HashMap;

use kadanoffbaym::dyson::DysonBathType;
use kadanoffbaym::fourier::{FftDirection, FftIntegralMethod};
use kadanoffbaym::math::{BetheDos, GaussianDos};
use kadanoffbaym::types::{FftDomain, SteadyGreen, SteadyGrid};

use kadanoffbaym::base::matsubara_seed;
use kadanoffbaym::convolution::{
    bubble1_steady, bubble1_steady_scalar, bubble2_steady, bubble2_steady_scalar,
};
use kadanoffbaym::dyson::{dyson_steady, sigma_bath};
use kadanoffbaym::equilibrium::{equilibrate_lesser, from_dos, from_dos_steady};
use kadanoffbaym::fourier::{
    adft_func, contour_to_steady, dft_coefficients_cubic, dft_coefficients_linear, dft_cplx,
    downsample, fft, matsubara_dft, matsubara_ft, steady_to_contour, upsample,
};
use kadanoffbaym::observables::{steady_convolution_density_matrix, steady_density_matrix};

use super::util::RefRow;

use super::util::read_ref;

use super::util::*;

use super::util::prelude::*;

/// Integrand of the corrected and adaptive Fourier tests with its limits.
type TestFunction = (&'static dyn Fn(f64) -> f64, f64, f64);

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Compare the values with the reference rows; returns the failures.
fn compare_rows(
    rows: &[RefRow],
    kbval: &dyn Fn(&RefRow) -> Complex,
    atol: f64,
    rtol: f64,
    sgn: &dyn Fn(&RefRow) -> f64,
) -> Vec<String> {
    let mut fails = Vec::new();
    for r in rows {
        let x = kbval(r);
        let y = c(sgn(r), 0.0) * r.value;
        let ok = (x - y).abs() <= atol + rtol * y.abs();
        if !ok {
            fails.push(format!(
                "{} ({}, {}, {}, {}): kb = {x}, reference = {y}",
                r.component, r.i, r.j, r.a, r.b
            ));
        }
    }
    fails
}

/// Records consumed by the profile: all records in release, otherwise
/// a fixed subset of every component.
fn select_rows(rows: Vec<RefRow>) -> Vec<RefRow> {
    if is_release() {
        return rows;
    }
    let mut sel: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.j % 17 == 0 || r.j == 1 || r.j == 2 || r.j == 3)
        .cloned()
        .collect();
    let mut comps: Vec<String> = sel.iter().map(|r| r.component.clone()).collect();
    for r in &rows {
        if !comps.contains(&r.component) {
            sel.push(r.clone());
            comps.push(r.component.clone());
        }
    }
    sel
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn reference_fourier() {
    let refdata = read_ref("data/test/fourier.dat").unwrap();
    let rows = &refdata.rows;

    // FFT
    let mut fftval: HashMap<(String, i64), Vec<Complex>> = HashMap::new();
    for nn in [2i64, 3, 4, 5, 6, 10, 12, 16, 25, 60, 97, 101] {
        let x: Vec<Complex> = (0..nn)
            .map(|n| {
                c(
                    (0.3 * n as f64 + 0.1 * nn as f64).cos(),
                    (0.7 * n as f64 * n as f64 / nn as f64).sin(),
                )
            })
            .collect();
        fftval.insert(("fft_input".into(), nn), x.clone());
        fftval.insert(("fft_forward".into(), nn), fft(&x, FftDirection::Forward));
        fftval.insert(("fft_backward".into(), nn), fft(&x, FftDirection::Backward));
    }
    let r_fft: Vec<&RefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("fft"))
        .collect();
    let fails = compare_rows(
        &r_fft.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
        &|r: &RefRow| fftval[&(r.component.clone(), r.i)][r.j as usize],
        1e-12,
        1e-12,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // correction factors
    let thetas = [
        0.0, 1e-8, -1e-8, 0.049999, -0.049999, 0.050001, -0.050001, 0.7,
    ];
    let dftcorr = |r: &RefRow| -> Complex {
        let th = thetas[r.i as usize];
        if r.a == 0 {
            return c(th, 0.0);
        }
        let (corfac, ec): (f64, Vec<Complex>) = if r.j == 1 {
            let (cf, ec) = dft_coefficients_linear(th);
            (cf, ec.to_vec())
        } else {
            let (cf, ec) = dft_coefficients_cubic(th);
            (cf, ec.to_vec())
        };
        if r.a == 1 {
            c(corfac, 0.0)
        } else {
            ec[(r.a - 2) as usize]
        }
    };
    let r_cor: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "dftcorr")
        .cloned()
        .collect();
    assert_eq!(r_cor.len(), 8 * (4 + 6));
    let fails = compare_rows(&r_cor, &dftcorr, 1e-12, 1e-12, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // corrected and adaptive Fourier integrals
    let funcs: [TestFunction; 3] = [
        (&|_| 1.0, 0.0, 1.0),
        (&|x: f64| (-x).exp(), 0.0, 2.0),
        (&|x: f64| (-x * x).exp(), -5.0, 5.0),
    ];
    let omegas = [-3.0f64, -0.7, 0.0, 0.5, 2.3, 10.0];
    let adaptive = |r: &RefRow| -> Complex {
        let (f, lo, hi) = funcs[r.i as usize];
        let w = omegas[r.j as usize];
        if r.a == 0 {
            return c(w, 0.0);
        }
        if r.a == 1 {
            let vals: Vec<Complex> = (0..=32)
                .map(|k| c(f(lo + (hi - lo) * k as f64 / 32.0), 0.0))
                .collect();
            return dft_cplx(w, lo, hi, &vals).0;
        }
        adft_func(f, w, lo, hi, 20, 40).0
    };
    let r_ad: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "adaptive")
        .cloned()
        .collect();
    assert_eq!(r_ad.len(), 3 * 6 * 3);
    let sub: Vec<RefRow> = r_ad.iter().filter(|r| r.a <= 1).cloned().collect();
    let fails = compare_rows(&sub, &adaptive, 1e-12, 1e-12, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let sub: Vec<RefRow> = r_ad.iter().filter(|r| r.a == 2).cloned().collect();
    let fails = compare_rows(&sub, &adaptive, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // Matsubara Fourier transforms
    let mats = |tau: f64| -> Vec<Complex> {
        vec![
            c(-0.6 * (-0.4 * tau).exp(), 0.0),
            0.2 * (1.3 * tau).cos() + c(0.0, 0.05 * tau),
            0.2 * (1.3 * tau).cos() + c(0.0, -0.05 * tau),
            c(-0.3 * (0.25 * tau).exp(), 0.0),
        ]
    };
    let mut mdft: Vec<Vec<Vec<Complex>>> = Vec::new();
    let mut mft: HashMap<(i64, i64), Vec<Complex>> = HashMap::new();
    for cc in 0..=1i64 {
        let sign = if cc == 0 { Sign::Fermi } else { Sign::Bose };
        let beta = 2.5f64;
        let cont = Contour::new(1, 20, 2, 2, 1.0, beta);
        let mut g = ContourGreen::new(&cont, sign, Complex::ZERO);
        for r in 0..=20i64 {
            let blk = mats(r as f64 * beta / 20.0);
            g.mat.block_mut(r as usize).copy_from_slice(&blk);
        }
        let raw = matsubara_dft(&g.mat, sign);
        mdft.push(raw.clone());
        for &order in &[0i64, 1, 3] {
            for &m in &[-25i64, -3, -1, 0, 1, 2, 7, 30] {
                let dest = matsubara_ft(m, &g.mat, &raw, sign, beta, order);
                mft.insert((10 * cc + order, m), dest);
            }
        }
    }
    let r_md: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "matsubara_dft_result")
        .cloned()
        .collect();
    assert_eq!(r_md.len(), 2 * 21 * 4);
    let sel = select_rows(r_md);
    let fails = compare_rows(
        &sel,
        &|r: &RefRow| mdft[r.i as usize][r.j as usize][(r.a * 2 + r.b) as usize],
        1e-12,
        1e-12,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let r_mf: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "matsubara_ft_result")
        .cloned()
        .collect();
    assert_eq!(r_mf.len(), 2 * 3 * 8 * 4);
    let fails = compare_rows(
        &r_mf,
        &|r: &RefRow| mft[&(r.i, r.j)][(r.a * 2 + r.b) as usize],
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // Fourier Matsubara VIE2, pcf = 20 and pcf = 10
    let vf = |tau: f64| -> Vec<Complex> {
        vec![
            c(0.4 * (-0.5 * tau).exp(), 0.0),
            c(0.4 * 0.1, 0.4 * 0.05 * tau),
            c(0.4 * 0.1, -0.4 * 0.05 * tau),
            c(0.4 * (-1.2 * tau).exp(), 0.0),
        ]
    };
    let vq = |tau: f64| -> Vec<Complex> {
        vec![
            c(-(-0.3 * tau).exp(), 0.0),
            c(0.05, 0.0),
            c(0.05, 0.0),
            c(-0.5 * (0.2 * tau).exp(), 0.0),
        ]
    };
    let mut vie: HashMap<(String, i64), ContourGreen> = HashMap::new();
    for cc in 0..=1i64 {
        for &pcf in &[20i64, 10] {
            let sign = if cc == 0 { Sign::Fermi } else { Sign::Bose };
            let beta = 3.0f64;
            let cont = Contour::new(1, 40, 2, 2, 1.0, beta);
            let mut f = ContourGreen::new(&cont, sign, Complex::ZERO);
            let mut q = ContourGreen::new(&cont, sign, Complex::ZERO);
            let mut x = ContourGreen::new(&cont, sign, Complex::ZERO);
            for r in 0..=40i64 {
                let fb = vf(r as f64 * beta / 40.0);
                f.mat.block_mut(r as usize).copy_from_slice(&fb);
                let qb = vq(r as f64 * beta / 40.0);
                q.mat.block_mut(r as usize).copy_from_slice(&qb);
            }
            matsubara_seed(&mut x.mat, &f.mat, &q.mat, beta, x.sign, pcf, 3);
            vie.insert((format!("vie2_pcf{pcf}"), cc), x);
        }
    }
    let vieval = |r: &RefRow| -> Complex {
        vie[&(r.component.clone(), r.i)].mat.block(r.j as usize)[(r.a * 2 + r.b) as usize]
    };
    let r_v: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("vie2_"))
        .cloned()
        .collect();
    assert_eq!(r_v.len(), 2 * 2 * 41 * 4);
    let sel = select_rows(r_v);
    let fails = compare_rows(&sel, &vieval, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    // counterexample: pcf = 10 does not reproduce the pcf = 20 result
    let mut mx = 0.0f64;
    for r in rows.iter().filter(|r| r.component == "vie2_pcf20") {
        let got = vie[&("vie2_pcf10".to_string(), r.i)]
            .mat
            .block(r.j as usize)[(r.a * 2 + r.b) as usize];
        mx = mx.max((got - r.value).abs());
    }
    assert!(mx > 1e-10);
}

#[test]
fn reference_steady() {
    let refdata = read_ref("data/test/steady.dat").unwrap();
    let rows = &refdata.rows;
    let nftmax: i64 = refdata.metadata["nft_max"].parse().unwrap();
    assert_eq!(nftmax, 512);

    let (nft, h, beta) = (256i64, 0.05f64, 5.0);
    let g = SteadyGrid::new(nft, h);
    let ng = |d: usize| SteadyGreen::new(g, d);

    // equilibrium Green's functions
    let mut g0 = ng(1);
    from_dos_steady(
        &mut g0,
        Sign::Fermi,
        &|x| GaussianDos::new(0.0, 1.0).at(x),
        beta,
        0.3,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let mut g1 = ng(1);
    from_dos_steady(
        &mut g1,
        Sign::Fermi,
        &|x| BetheDos::new(-2.0, 2.0).at(x),
        beta,
        0.1,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let mut g2 = ng(1);
    let dos2 = GaussianDos::new(0.0, 1.0);
    from_dos_steady(
        &mut g2,
        Sign::Fermi,
        &|x| dos2.at(x),
        beta,
        0.3,
        FftIntegralMethod::Adaptive,
        Some(dos2.lo),
        Some(dos2.hi),
        100,
        20,
    );
    let mut g3 = ng(1);
    from_dos_steady(
        &mut g3,
        Sign::Bose,
        &|x| BetheDos::new(1.0, 5.0).at(x),
        beta,
        0.0,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );

    // Dyson equations
    let mut s = g1.clone();
    s.smul(c(0.25, 0.0), FftDomain::Time);
    let mut h1 = DenseMat::zeros(1);
    h1.set(0, 0, c(0.2, 0.0));
    let mut g4 = ng(1);
    dyson_steady(
        &mut g4,
        0.0,
        &h1,
        &s,
        FftIntegralMethod::Trapez,
        DysonBathType::Gauss,
        0.1,
        beta,
        0.0,
    );
    let mut s5 = ng(2);
    for n in 0..nft {
        let r0 = g0.ret.time[n as usize];
        let r1 = g1.ret.time[n as usize];
        let l0 = g0.les.time[n as usize];
        let l1 = g1.les.time[n as usize];
        s5.ret.time[n as usize * 4..(n as usize + 1) * 4].copy_from_slice(&[
            0.3 * r0,
            0.1 * r0,
            0.1 * r0,
            0.2 * r1,
        ]);
        s5.les.time[n as usize * 4..(n as usize + 1) * 4].copy_from_slice(&[
            0.3 * l0,
            0.1 * l0,
            0.1 * l0,
            0.2 * l1,
        ]);
    }
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, c(0.3, 0.0));
    h2.set(0, 1, c(0.0, 0.2));
    h2.set(1, 0, c(0.0, -0.2));
    h2.set(1, 1, c(-0.4, 0.0));
    let mut g5 = ng(2);
    dyson_steady(
        &mut g5,
        0.0,
        &h2,
        &s5,
        FftIntegralMethod::Trapez,
        DysonBathType::Const,
        0.05,
        beta,
        0.0,
    );
    let mut h4 = DenseMat::zeros(1);
    h4.set(0, 0, c(0.4, 0.0));
    let mut g6 = ng(1);
    dyson_steady(
        &mut g6,
        0.0,
        &h4,
        &ng(1),
        FftIntegralMethod::Trapez,
        DysonBathType::Ohmic,
        0.1,
        beta,
        0.0,
    );
    let h0 = DenseMat::zeros(1);
    let mut g7 = ng(1);
    dyson_steady(
        &mut g7,
        0.0,
        &h0,
        &s,
        FftIntegralMethod::Trapez,
        DysonBathType::None,
        0.0,
        1.0,
        0.0,
    );
    let cases = [&g0, &g1, &g2, &g3, &g4, &g5, &g6, &g7];

    let gfval = |r: &RefRow| -> Complex {
        let gg = cases[r.i as usize];
        let d = gg.ret.size1;
        let (arr, dom) = if r.component.starts_with("ret") {
            (&gg.ret, r.component.ends_with("time"))
        } else {
            (&gg.les, r.component.ends_with("time"))
        };
        let q = r.j as usize * d * d + (r.a as usize * d + r.b as usize);
        if dom { arr.time[q] } else { arr.freq[q] }
    };
    let r_gf: Vec<RefRow> = rows
        .iter()
        .filter(|r| {
            ["ret_time", "les_time", "ret_freq", "les_freq"].contains(&r.component.as_str())
        })
        .cloned()
        .collect();
    let sel = select_rows(r_gf);
    let fails = compare_rows(&sel, &gfval, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // bath self-energies
    let baths = [
        DysonBathType::None,
        DysonBathType::Const,
        DysonBathType::Gauss,
        DysonBathType::Ohmic,
    ];
    let bathval = |r: &RefRow| -> Complex {
        let (ret, les) = sigma_bath(
            -3.0 + 0.25 * r.j as f64,
            h,
            baths[r.i as usize],
            0.2,
            beta,
            0.1,
        );
        if r.component == "bath_ret" { ret } else { les }
    };
    let r_b: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("bath_"))
        .cloned()
        .collect();
    assert_eq!(r_b.len(), 2 * 4 * 25);
    let fails = compare_rows(&r_b, &bathval, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // density matrices
    let mut dens: HashMap<i64, Vec<Complex>> = HashMap::new();
    for &i in &[0i64, 1, 4, 5] {
        dens.insert(i, steady_density_matrix(cases[i as usize], Sign::Fermi));
    }
    let mut cdens: HashMap<i64, Vec<Complex>> = HashMap::new();
    cdens.insert(0, steady_convolution_density_matrix(&g0, &g1, Sign::Fermi));
    cdens.insert(1, steady_convolution_density_matrix(&g5, &g5, Sign::Fermi));
    let r_d: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "density")
        .cloned()
        .collect();
    let fails = compare_rows(
        &r_d,
        &|r: &RefRow| {
            // case dimensions: only case 5 is 2 x 2
            let d = if r.i == 5 { 2usize } else { 1 };
            dens[&r.i][r.a as usize * d + r.b as usize]
        },
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let r_cd: Vec<RefRow> = rows
        .iter()
        .filter(|r| r.component == "convolution_density")
        .cloned()
        .collect();
    assert_eq!(r_cd.len(), 1 + 4);
    let fails = compare_rows(
        &r_cd,
        &|r: &RefRow| cdens[&r.i][(r.a * 2 + r.b) as usize],
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // operations, time domain: ret at j = n, les at j = nft_max + n
    let mut b1 = ng(1);
    bubble1_steady_scalar(&mut b1, &g0, &g1);
    let mut b2 = ng(1);
    bubble2_steady_scalar(&mut b2, &g0, &g1);
    let mut d1 = ng(2);
    let mut d2 = ng(2);
    for a in 0..2usize {
        for b in 0..2usize {
            bubble1_steady(&mut d1, a, b, &g5, a, b, &g0, 0, 0);
            bubble2_steady(&mut d2, a, b, &g5, a, b, &g0, 0, 0);
        }
    }
    let mut ops: HashMap<(String, i64), SteadyGreen> = HashMap::new();
    ops.insert(("bubble1".into(), 0), b1);
    ops.insert(("bubble2".into(), 0), b2);
    ops.insert(("bubble1".into(), 1), d1);
    ops.insert(("bubble2".into(), 1), d2);
    ops.insert(("resample".into(), 0), downsample(&g0, 4));
    ops.insert(("resample".into(), 1), upsample(&g1, 2));
    for (i, &(sign, ref dos, mu)) in [
        (Sign::Fermi, GaussianDos::new(0.0, 1.0), 0.3),
        (Sign::Bose, GaussianDos::new(3.0, 0.5), -1.0),
    ]
    .iter()
    .enumerate()
    {
        let mut f = ng(1);
        from_dos_steady(
            &mut f,
            sign,
            &|x| dos.at(x),
            beta,
            mu,
            FftIntegralMethod::Trapez,
            None,
            None,
            100,
            20,
        );
        f.les.time.fill(Complex::ZERO);
        f.les.freq.fill(Complex::ZERO);
        equilibrate_lesser(&mut f, sign, beta, mu);
        ops.insert(("force_equilibrium".into(), i as i64), f);
    }
    let cont = Contour::new(128, 50, 1, 1, 128.0 * h, beta);
    let mut gc = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    let dos = GaussianDos::new(0.0, 1.0);
    from_dos(
        &mut gc,
        &|x| dos.at(x),
        dos.lo,
        dos.hi,
        beta,
        h,
        0.3,
        100,
        20,
    );
    let mut n0 = ng(1);
    contour_to_steady(&mut n0, &gc, 128);
    let cont2 = Contour::new(137, 50, 1, 1, 137.0 * h, beta);
    let mut gc2 = ContourGreen::new(&cont2, Sign::Fermi, Complex::ZERO);
    steady_to_contour(&mut gc2, &g0);
    let mut n1 = ng(1);
    contour_to_steady(&mut n1, &gc2, 137);
    ops.insert(("conversion".into(), 0), n0);
    ops.insert(("conversion".into(), 1), n1);

    let timeval = |r: &RefRow| -> Complex {
        let gg = &ops[&(r.component.clone(), r.i)];
        let n = if r.j < nftmax { r.j } else { r.j - nftmax };
        let arr = if r.j < nftmax { &gg.ret } else { &gg.les };
        let d = gg.ret.size1;
        arr.time[n as usize * d * d + (r.a * d as i64 + r.b) as usize]
    };
    let r_op: Vec<RefRow> = rows
        .iter()
        .filter(|r| {
            [
                "bubble1",
                "bubble2",
                "resample",
                "force_equilibrium",
                "conversion",
            ]
            .contains(&r.component.as_str())
        })
        .cloned()
        .collect();
    let sel = select_rows(r_op);
    let sgn = |r: &RefRow| -> f64 {
        if r.component == "force_equilibrium" && r.j >= nftmax {
            -1.0
        } else {
            1.0
        }
    };
    let fails = compare_rows(&sel, &timeval, 1e-10, 1e-10, &sgn);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn reference_release_profile() {
    if is_release() {
        // the testsets above consumed all records in release
        let rs = vec![
            RefRow {
                component: "x".into(),
                i: 0,
                j: 5,
                a: 0,
                b: 0,
                value: Complex::ZERO,
            },
            RefRow {
                component: "x".into(),
                i: 0,
                j: 6,
                a: 0,
                b: 0,
                value: Complex::ZERO,
            },
        ];
        let sel = select_rows(rs.clone());
        assert_eq!(sel.len(), rs.len());
    }
    // (the quick-profile branch of the reference checks the profile
    // string, which the Rust common module guarantees)
}
