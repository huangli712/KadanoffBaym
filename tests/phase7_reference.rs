// Project : KadanoffBaym
// File    : tests/phase7_reference.rs
// Purpose : blueprint t_phase7_reference.jl: point-wise comparison of
//           the Phase 7 algorithms with the NESSi reference data
//           data/test/phase7_fourier.dat and phase7_ness.dat
//
// Tolerances: FFT and the corrected DFT atol = rtol = 1e-12, all the
// other results atol = rtol = 1e-10. The quick profile consumes a
// fixed subset of the records of every component, the release profile
// all records.
//
// Deviation: force_equilibrium of NESSi 2.0.0 returns -1 times the
// physical lesser component; KB uses the physical sign, so the lesser
// records of `force_equilibrium` are compared with -1 times the
// reference.

mod common;

use common::*;
use kadanoffbaym::c64::C64;
use kadanoffbaym::equilibrium::from_dos;
use kadanoffbaym::equilibrium_ness::{equilibrate_lesser, from_dos_ness};
use kadanoffbaym::fourier::adft::{adft_func, dft_cplx};
use kadanoffbaym::fourier::convert::{cntr2ness, downsample, ness2cntr, upsample};
use kadanoffbaym::fourier::dftcor::{dftcorr_cubic, dftcorr_linear};
use kadanoffbaym::fourier::fft::{fft, FftDirection};
use kadanoffbaym::fourier::matsubara::{matsubara_dft, matsubara_ft};
use kadanoffbaym::fourier::transforms::FftIntegralMethod;
use kadanoffbaym::kbref::{read_kbref, KbrefRow};
use kadanoffbaym::math::dos::{BetheDos, GaussianDos};
use kadanoffbaym::observables::{convolution_density_matrix_ness, density_matrix_ness};
use kadanoffbaym::types::ness::smul_ness_mat;
use kadanoffbaym::solvers::dyson_ness::{dyson_ness, sigma_bath, DysonBathType};
use kadanoffbaym::solvers::vie2_mat::vie2_mat_fourier;
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::ContourGreen;
use kadanoffbaym::types::ness::{FftDomain, NessGrid, NessHermMat};

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Compare the values with the reference rows; returns the failures.
fn p7_compare(
    rows: &[KbrefRow],
    kbval: &dyn Fn(&KbrefRow) -> C64,
    atol: f64,
    rtol: f64,
    sgn: &dyn Fn(&KbrefRow) -> f64,
) -> Vec<String> {
    let mut fails = Vec::new();
    for r in rows {
        let x = kbval(r);
        let y = c(sgn(r), 0.0) * r.value;
        let ok = (x - y).abs() <= atol + rtol * y.abs();
        if !ok {
            fails.push(format!(
                "{} ({}, {}, {}, {}): kb = {x}, nessi = {y}",
                r.component, r.i, r.j, r.a, r.b
            ));
        }
    }
    fails
}

/// Records consumed by the profile: all records in release, otherwise
/// a fixed subset of every component.
fn p7_select(rows: Vec<KbrefRow>) -> Vec<KbrefRow> {
    if is_release() {
        return rows;
    }
    let mut sel: Vec<KbrefRow> = rows
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

#[test]
fn phase7_reference_fourier() {
    let refdata = read_kbref("data/test/phase7_fourier.dat").unwrap();
    let rows = &refdata.rows;

    // FFT
    use std::collections::HashMap;
    let mut fftval: HashMap<(String, i64), Vec<C64>> = HashMap::new();
    for nn in [2i64, 3, 4, 5, 6, 10, 12, 16, 25, 60, 97, 101] {
        let x: Vec<C64> = (0..nn)
            .map(|n| {
                c(
                    (0.3 * n as f64 + 0.1 * nn as f64).cos(),
                    (0.7 * n as f64 * n as f64 / nn as f64).sin(),
                )
            })
            .collect();
        fftval.insert(("fft_input".into(), nn), x.clone());
        fftval.insert(
            ("fft_forward".into(), nn),
            fft(&x, FftDirection::Forward),
        );
        fftval.insert(
            ("fft_backward".into(), nn),
            fft(&x, FftDirection::Backward),
        );
    }
    let r_fft: Vec<&KbrefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("fft"))
        .collect();
    let fails = p7_compare(
        &r_fft.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
        &|r: &KbrefRow| fftval[&(r.component.clone(), r.i)][r.j as usize],
        1e-12,
        1e-12,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // correction factors
    let thetas = [0.0, 1e-8, -1e-8, 0.049999, -0.049999, 0.050001, -0.050001, 0.7];
    let dftcorr = |r: &KbrefRow| -> C64 {
        let th = thetas[r.i as usize];
        if r.a == 0 {
            return c(th, 0.0);
        }
        let (corfac, ec): (f64, Vec<C64>) = if r.j == 1 {
            let (cf, ec) = dftcorr_linear(th);
            (cf, ec.to_vec())
        } else {
            let (cf, ec) = dftcorr_cubic(th);
            (cf, ec.to_vec())
        };
        if r.a == 1 {
            c(corfac, 0.0)
        } else {
            ec[(r.a - 2) as usize]
        }
    };
    let r_cor: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "dftcorr")
        .cloned()
        .collect();
    assert_eq!(r_cor.len(), 8 * (4 + 6));
    let fails = p7_compare(&r_cor, &dftcorr, 1e-12, 1e-12, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // corrected and adaptive Fourier integrals
    let funcs: [(&dyn Fn(f64) -> f64, f64, f64); 3] = [
        (&|_| 1.0, 0.0, 1.0),
        (&|x: f64| (-x).exp(), 0.0, 2.0),
        (&|x: f64| (-x * x).exp(), -5.0, 5.0),
    ];
    let omegas = [-3.0f64, -0.7, 0.0, 0.5, 2.3, 10.0];
    let adaptive = |r: &KbrefRow| -> C64 {
        let (f, lo, hi) = funcs[r.i as usize];
        let w = omegas[r.j as usize];
        if r.a == 0 {
            return c(w, 0.0);
        }
        if r.a == 1 {
            let vals: Vec<C64> = (0..=32)
                .map(|k| c(f(lo + (hi - lo) * k as f64 / 32.0), 0.0))
                .collect();
            return dft_cplx(w, lo, hi, &vals).0;
        }
        adft_func(f, w, lo, hi, 20, 40).0
    };
    let r_ad: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "adaptive")
        .cloned()
        .collect();
    assert_eq!(r_ad.len(), 3 * 6 * 3);
    let sub: Vec<KbrefRow> = r_ad.iter().filter(|r| r.a <= 1).cloned().collect();
    let fails = p7_compare(&sub, &adaptive, 1e-12, 1e-12, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let sub: Vec<KbrefRow> = r_ad.iter().filter(|r| r.a == 2).cloned().collect();
    let fails = p7_compare(&sub, &adaptive, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // Matsubara Fourier transforms
    let mats = |tau: f64| -> Vec<C64> {
        vec![
            c(-0.6 * (-0.4 * tau).exp(), 0.0),
            0.2 * (1.3 * tau).cos() + c(0.0, 0.05 * tau),
            0.2 * (1.3 * tau).cos() + c(0.0, -0.05 * tau),
            c(-0.3 * (0.25 * tau).exp(), 0.0),
        ]
    };
    let mut mdft: Vec<Vec<Vec<C64>>> = Vec::new();
    let mut mft: HashMap<(i64, i64), Vec<C64>> = HashMap::new();
    for cc in 0..=1i64 {
        let sign = if cc == 0 { Sign::Fermi } else { Sign::Bose };
        let beta = 2.5f64;
        let cont = Contour::new(1, 20, 2, 2, 1.0, beta);
        let mut g = ContourGreen::new(&cont, sign, C64::ZERO);
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
    let r_md: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "matsubara_dft_result")
        .cloned()
        .collect();
    assert_eq!(r_md.len(), 2 * 21 * 4);
    let sel = p7_select(r_md);
    let fails = p7_compare(
        &sel,
        &|r: &KbrefRow| mdft[r.i as usize][r.j as usize][(r.a * 2 + r.b) as usize],
        1e-12,
        1e-12,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let r_mf: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "matsubara_ft_result")
        .cloned()
        .collect();
    assert_eq!(r_mf.len(), 2 * 3 * 8 * 4);
    let fails = p7_compare(
        &r_mf,
        &|r: &KbrefRow| mft[&(r.i, r.j)][(r.a * 2 + r.b) as usize],
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // Fourier Matsubara VIE2, pcf = 20 and pcf = 10
    let vf = |tau: f64| -> Vec<C64> {
        vec![
            c(0.4 * (-0.5 * tau).exp(), 0.0),
            c(0.4 * 0.1, 0.4 * 0.05 * tau),
            c(0.4 * 0.1, -0.4 * 0.05 * tau),
            c(0.4 * (-1.2 * tau).exp(), 0.0),
        ]
    };
    let vq = |tau: f64| -> Vec<C64> {
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
            let mut f = ContourGreen::new(&cont, sign, C64::ZERO);
            let mut q = ContourGreen::new(&cont, sign, C64::ZERO);
            let mut x = ContourGreen::new(&cont, sign, C64::ZERO);
            for r in 0..=40i64 {
                let fb = vf(r as f64 * beta / 40.0);
                f.mat.block_mut(r as usize).copy_from_slice(&fb);
                let qb = vq(r as f64 * beta / 40.0);
                q.mat.block_mut(r as usize).copy_from_slice(&qb);
            }
            vie2_mat_fourier(&mut x, &f, &q, beta, pcf, 3);
            vie.insert((format!("vie2_pcf{pcf}"), cc), x);
        }
    }
    let vieval = |r: &KbrefRow| -> C64 {
        vie[&(r.component.clone(), r.i)].mat.block(r.j as usize)[(r.a * 2 + r.b) as usize]
    };
    let r_v: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("vie2_"))
        .cloned()
        .collect();
    assert_eq!(r_v.len(), 2 * 2 * 41 * 4);
    let sel = p7_select(r_v);
    let fails = p7_compare(&sel, &vieval, 1e-10, 1e-10, &|_| 1.0);
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
fn phase7_reference_ness() {
    let refdata = read_kbref("data/test/phase7_ness.dat").unwrap();
    let rows = &refdata.rows;
    let nftmax: i64 = refdata.metadata["nft_max"].parse().unwrap();
    assert_eq!(nftmax, 512);

    let (nft, h, beta) = (256i64, 0.05f64, 5.0);
    let g = NessGrid::new(nft, h);
    let ng = |d: usize| NessHermMat::new(g, d);

    // equilibrium Green's functions
    let mut g0 = ng(1);
    from_dos_ness(&mut g0, Sign::Fermi, &|x| GaussianDos::new(0.0, 1.0).at(x), beta, 0.3, FftIntegralMethod::Trapez, None, None, 100, 20);
    let mut g1 = ng(1);
    from_dos_ness(&mut g1, Sign::Fermi, &|x| BetheDos::new(-2.0, 2.0).at(x), beta, 0.1, FftIntegralMethod::Trapez, None, None, 100, 20);
    let mut g2 = ng(1);
    let dos2 = GaussianDos::new(0.0, 1.0);
    from_dos_ness(&mut g2, Sign::Fermi, &|x| dos2.at(x), beta, 0.3, FftIntegralMethod::Adaptive, Some(dos2.lo), Some(dos2.hi), 100, 20);
    let mut g3 = ng(1);
    from_dos_ness(&mut g3, Sign::Bose, &|x| BetheDos::new(1.0, 5.0).at(x), beta, 0.0, FftIntegralMethod::Trapez, None, None, 100, 20);

    // Dyson equations
    let mut s = g1.clone();
    smul_ness_mat(&mut s, c(0.25, 0.0), FftDomain::Time);
    let mut h1 = kadanoffbaym::mat::DenseMat::zeros(1);
    h1.set(0, 0, c(0.2, 0.0));
    let mut g4 = ng(1);
    dyson_ness(&mut g4, 0.0, &h1, &s, FftIntegralMethod::Trapez, DysonBathType::Gauss, 0.1, beta, 0.0);
    let mut s5 = ng(2);
    for n in 0..nft {
        let r0 = g0.ret.time[n as usize];
        let r1 = g1.ret.time[n as usize];
        let l0 = g0.les.time[n as usize];
        let l1 = g1.les.time[n as usize];
        s5.ret.time[n as usize * 4..(n as usize + 1) * 4].copy_from_slice(&[0.3 * r0, 0.1 * r0, 0.1 * r0, 0.2 * r1]);
        s5.les.time[n as usize * 4..(n as usize + 1) * 4].copy_from_slice(&[0.3 * l0, 0.1 * l0, 0.1 * l0, 0.2 * l1]);
    }
    let mut h2 = kadanoffbaym::mat::DenseMat::zeros(2);
    h2.set(0, 0, c(0.3, 0.0));
    h2.set(0, 1, c(0.0, 0.2));
    h2.set(1, 0, c(0.0, -0.2));
    h2.set(1, 1, c(-0.4, 0.0));
    let mut g5 = ng(2);
    dyson_ness(&mut g5, 0.0, &h2, &s5, FftIntegralMethod::Trapez, DysonBathType::Const, 0.05, beta, 0.0);
    let mut h4 = kadanoffbaym::mat::DenseMat::zeros(1);
    h4.set(0, 0, c(0.4, 0.0));
    let mut g6 = ng(1);
    dyson_ness(&mut g6, 0.0, &h4, &ng(1), FftIntegralMethod::Trapez, DysonBathType::Ohmic, 0.1, beta, 0.0);
    let mut h0 = kadanoffbaym::mat::DenseMat::zeros(1);
    let mut g7 = ng(1);
    dyson_ness(&mut g7, 0.0, &h0, &s, FftIntegralMethod::Trapez, DysonBathType::None, 0.0, 1.0, 0.0);
    let cases = [&g0, &g1, &g2, &g3, &g4, &g5, &g6, &g7];

    let gfval = |r: &KbrefRow| -> C64 {
        let gg = cases[r.i as usize];
        let d = gg.ret.size1;
        let (arr, dom) = if r.component.starts_with("ret") {
            (&gg.ret, r.component.ends_with("time"))
        } else {
            (&gg.les, r.component.ends_with("time"))
        };
        let q = r.j as usize * d * d + (r.a as usize * d + r.b as usize);
        if dom {
            arr.time[q]
        } else {
            arr.freq[q]
        }
    };
    let r_gf: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| {
            ["ret_time", "les_time", "ret_freq", "les_freq"].contains(&r.component.as_str())
        })
        .cloned()
        .collect();
    let sel = p7_select(r_gf);
    let fails = p7_compare(&sel, &gfval, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // bath self-energies
    let baths = [
        DysonBathType::None,
        DysonBathType::Const,
        DysonBathType::Gauss,
        DysonBathType::Ohmic,
    ];
    let bathval = |r: &KbrefRow| -> C64 {
        let (ret, les) = sigma_bath(-3.0 + 0.25 * r.j as f64, h, baths[r.i as usize], 0.2, beta, 0.1);
        if r.component == "bath_ret" {
            ret
        } else {
            les
        }
    };
    let r_b: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component.starts_with("bath_"))
        .cloned()
        .collect();
    assert_eq!(r_b.len(), 2 * 4 * 25);
    let fails = p7_compare(&r_b, &bathval, 1e-10, 1e-10, &|_| 1.0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // density matrices
    use std::collections::HashMap;
    let mut dens: HashMap<i64, Vec<C64>> = HashMap::new();
    for &i in &[0i64, 1, 4, 5] {
        dens.insert(i, density_matrix_ness(cases[i as usize], Sign::Fermi));
    }
    let mut cdens: HashMap<i64, Vec<C64>> = HashMap::new();
    cdens.insert(0, convolution_density_matrix_ness(&g0, &g1, Sign::Fermi));
    cdens.insert(1, convolution_density_matrix_ness(&g5, &g5, Sign::Fermi));
    let r_d: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "density")
        .cloned()
        .collect();
    let fails = p7_compare(
        &r_d,
        &|r: &KbrefRow| {
            // case dimensions: only case 5 is 2 x 2
            let d = if r.i == 5 { 2usize } else { 1 };
            dens[&r.i][r.a as usize * d + r.b as usize]
        },
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    let r_cd: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| r.component == "convolution_density")
        .cloned()
        .collect();
    assert_eq!(r_cd.len(), 1 + 4);
    let fails = p7_compare(
        &r_cd,
        &|r: &KbrefRow| cdens[&r.i][(r.a * 2 + r.b) as usize],
        1e-10,
        1e-10,
        &|_| 1.0,
    );
    assert!(fails.is_empty(), "{}", fails.join("\n"));

    // operations, time domain: ret at j = n, les at j = nft_max + n
    use kadanoffbaym::convolution::bubble::{bubble1_ness, bubble2_ness, bubble1_ness_scalar, bubble2_ness_scalar};
    let mut b1 = ng(1);
    bubble1_ness_scalar(&mut b1, &g0, &g1);
    let mut b2 = ng(1);
    bubble2_ness_scalar(&mut b2, &g0, &g1);
    let mut d1 = ng(2);
    let mut d2 = ng(2);
    for a in 0..2usize {
        for b in 0..2usize {
            bubble1_ness(&mut d1, a, b, &g5, a, b, &g0, 0, 0);
            bubble2_ness(&mut d2, a, b, &g5, a, b, &g0, 0, 0);
        }
    }
    let mut ops: HashMap<(String, i64), NessHermMat> = HashMap::new();
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
        from_dos_ness(&mut f, sign, &|x| dos.at(x), beta, mu, FftIntegralMethod::Trapez, None, None, 100, 20);
        f.les.time.fill(C64::ZERO);
        f.les.freq.fill(C64::ZERO);
        equilibrate_lesser(&mut f, sign, beta, mu);
        ops.insert(("force_equilibrium".into(), i as i64), f);
    }
    let cont = Contour::new(128, 50, 1, 1, 128.0 * h, beta);
    let mut gc = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    let dos = GaussianDos::new(0.0, 1.0);
    from_dos(&mut gc, &|x| dos.at(x), dos.lo, dos.hi, beta, h, 0.3, 100, 20);
    let mut n0 = ng(1);
    cntr2ness(&mut n0, &gc, 128);
    let cont2 = Contour::new(137, 50, 1, 1, 137.0 * h, beta);
    let mut gc2 = ContourGreen::new(&cont2, Sign::Fermi, C64::ZERO);
    ness2cntr(&mut gc2, &g0);
    let mut n1 = ng(1);
    cntr2ness(&mut n1, &gc2, 137);
    ops.insert(("cntr_conversion".into(), 0), n0);
    ops.insert(("cntr_conversion".into(), 1), n1);

    let timeval = |r: &KbrefRow| -> C64 {
        let gg = &ops[&(r.component.clone(), r.i)];
        let n = if r.j < nftmax { r.j } else { r.j - nftmax };
        let arr = if r.j < nftmax { &gg.ret } else { &gg.les };
        let d = gg.ret.size1;
        arr.time[n as usize * d * d + (r.a * d as i64 + r.b) as usize]
    };
    let r_op: Vec<KbrefRow> = rows
        .iter()
        .filter(|r| {
            ["bubble1", "bubble2", "resample", "force_equilibrium", "cntr_conversion"]
                .contains(&r.component.as_str())
        })
        .cloned()
        .collect();
    let sel = p7_select(r_op);
    let sgn = |r: &KbrefRow| -> f64 {
        if r.component == "force_equilibrium" && r.j >= nftmax {
            -1.0
        } else {
            1.0
        }
    };
    let fails = p7_compare(&sel, &timeval, 1e-10, 1e-10, &sgn);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn phase7_reference_release_profile() {
    if is_release() {
        // the testsets above consumed all records in release
        let rs = vec![
            KbrefRow {
                component: "x".into(),
                i: 0,
                j: 5,
                a: 0,
                b: 0,
                value: C64::ZERO,
            },
            KbrefRow {
                component: "x".into(),
                i: 0,
                j: 6,
                a: 0,
                b: 0,
                value: C64::ZERO,
            },
        ];
        let sel = p7_select(rs.clone());
        assert_eq!(sel.len(), rs.len());
    }
    // (the quick-profile branch of the blueprint checks the profile
    // string, which the Rust common module guarantees)
}
