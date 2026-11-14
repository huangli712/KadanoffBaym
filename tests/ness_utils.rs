// Project : KadanoffBaym
// File    : tests/ness_utils.rs
// Purpose : blueprint t_ness_utils.jl: the NESS2 algorithms - the bath
//           self-energy, the steady-state Dyson equation, the DOS
//           types, the equilibrium constructors, the density matrices,
//           the bubbles, the resampling and the conversions with the
//           contour functions
//
// Known deviations: the alias check (G === Sigma) and the invalid
// integer sign (0) of the blueprint are unrepresentable in the Rust
// API (enum signs, borrow checker); the ADAPTIVE method requires the
// window `lo`/`hi` because a plain closure carries no bounds.

mod common;

use common::*;
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::convolution::bubble::{bubble1, bubble1_ness, bubble2, bubble2_ness, bubble2_ness_scalar, bubble1_ness_scalar};
use kadanoffbaym::convolution::density_matrix::convolution_density_matrix;
use kadanoffbaym::equilibrium::from_dos;
use kadanoffbaym::equilibrium_ness::{equilibrate_lesser, from_dos_ness};
use kadanoffbaym::fourier::convert::{cntr2ness, downsample, downsample_array, ness2cntr, resample, upsample};
use kadanoffbaym::fourier::transforms::{integral_transform_to_freq, FftIntegralMethod};
use kadanoffbaym::math::dos::{BetheDos, GaussianDos, OhmicSymDos};
use kadanoffbaym::observables::{convolution_density_matrix_ness, density_matrix_green, density_matrix_ness};
use kadanoffbaym::solvers::dyson_ness::{dyson_ness, sigma_bath, DysonBathType};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::ContourGreen;
use kadanoffbaym::types::ness::{
    distance_ness, distance_ness_array, time_at, FftDomain, NessArray, NessGrid, NessHermMat,
};
use kadanoffbaym::weights::Integrator;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn nu_contour(ntime: i64, ntau: i64, d: i64, h: f64, beta: f64) -> Contour {
    Contour::new(ntime - 1, ntau - 1, d, d, (ntime - 1) as f64 * h, beta)
}

#[test]
fn ness_utils_sigma_bath() {
    let (r, l) = sigma_bath(0.0, 0.1, DysonBathType::Const, 0.2, 10.0, 0.0);
    assert!((r - c(0.0, -0.2)).abs() < 1e-14 && (l - c(0.0, 0.2)).abs() < 1e-14);
    assert_eq!(sigma_bath(0.3, 0.1, DysonBathType::None, 0.2, 10.0, 0.0), (C64::ZERO, C64::ZERO));
    let (r1, l1) = sigma_bath(0.7, 0.1, DysonBathType::Gauss, 0.2, 10.0, 0.1);
    let (r2, l2) = sigma_bath(0.7, 0.01, DysonBathType::Gauss, 0.2, 10.0, 0.1);
    assert_eq!((r1, l1), (r2, l2));
    assert!((r1 - c(0.0, -0.2) * (-0.49f64).exp()).abs() < 1e-14);
    let want_l = c(0.0, 0.4) * (-0.49f64).exp() / (1.0 + (6.0f64).exp());
    assert!((l1 - want_l).abs() < 1e-14);
    let (r, l) = sigma_bath(0.0, 0.1, DysonBathType::Ohmic, 0.2, 10.0, 0.0);
    assert_eq!(r, C64::ZERO);
    assert!((l - c(0.0, -0.04)).abs() < 1e-14);
    let (_, l) = sigma_bath(1e-9, 0.1, DysonBathType::Ohmic, 0.2, 10.0, 0.0);
    assert!(l.re.is_finite() && l.im.is_finite() && (l - c(0.0, -0.04)).abs() < 1e-10);
    let (_, l) = sigma_bath(0.5, 0.1, DysonBathType::Ohmic, 0.2, 10.0, 0.0);
    let want = c(0.0, -0.4) * 0.5 * (-0.25f64).exp() / ((5.0f64).exp() - 1.0);
    assert!((l - want).abs() < 1e-14);
    // cut-offs
    assert_eq!(sigma_bath(20.0, 0.1, DysonBathType::Const, 0.2, 10.0, 0.0).1, C64::ZERO);
    let (_, l) = sigma_bath(-20.0, 0.1, DysonBathType::Const, 0.2, 10.0, 0.0);
    assert!((l - c(0.0, 0.4)).abs() < 1e-14);
    assert!(panics(|| sigma_bath(0.0, 0.1, DysonBathType::Const, -0.2, 10.0, 0.0)));
    assert!(panics(|| sigma_bath(0.0, 0.1, DysonBathType::Const, 0.2, 0.0, 0.0)));
    assert!(panics(|| sigma_bath(f64::NAN, 0.1, DysonBathType::Const, 0.2, 1.0, 0.0)));
}

#[test]
fn ness_utils_dos_types() {
    let d = BetheDos::new(-2.0, 2.0);
    assert_eq!(d.at(3.0), 0.0);
    assert!(d.at(0.0) > 0.0);
    assert!((d.at(0.0) - 1.0 / std::f64::consts::PI).abs() < 1e-14);
    assert!(panics(|| BetheDos::new(2.0, -2.0)));
    let g = GaussianDos::new(0.0, 2.0);
    assert_eq!((g.lo, g.hi), (-12.0, 12.0));
    assert_eq!(g.at(0.0), 1.0 / (2.0 * 2.506623299999997));
    assert!(g.at(20.0) > 0.0);
    assert!(panics(|| GaussianDos::new(0.0, 0.0)));
    let o = OhmicSymDos::new(1.0);
    assert_eq!(o.at(-0.5), -o.at(0.5));
    assert_eq!(o.at(0.0), 0.0);
    assert_eq!((o.lo, o.hi), (-20.0, 20.0));
    assert!(panics(|| OhmicSymDos::new(-1.0)));
}

#[test]
fn ness_utils_dyson_ness_single_level() {
    let (nft, h) = (8192i64, 0.01f64);
    let (eps, eta, beta, mu) = (0.3f64, 0.5, 5.0, 0.1);
    let mut g = NessHermMat::new(NessGrid::new(nft, h), 1);
    let s = NessHermMat::new(NessGrid::new(nft, h), 1);
    let mut h0 = kadanoffbaym::mat::DenseMat::zeros(1);
    h0.set(0, 0, c(eps, 0.0));
    dyson_ness(
        &mut g,
        mu,
        &h0,
        &s,
        FftIntegralMethod::Trapez,
        DysonBathType::Const,
        eta,
        beta,
        mu,
    );
    let mut e = 0.0f64;
    for n in 1..=399i64 {
        let t = n as f64 * h;
        let want = c(0.0, 1.0) * C64::from_polar(1.0, 0.0) * (-(eta * t)).exp()
            * C64::from_polar(1.0, -(eps - mu) * t);
        e = e.max((g.ret.time[n as usize] + want).abs());
    }
    assert!(e < 5e-3, "{e}");
    for n in (nft / 2) as usize..nft as usize {
        assert_eq!(g.ret.time[n], C64::ZERO);
    }
    // (the alias check G === Sigma is subsumed by the borrow checker)
    assert!(panics(|| {
        let mut g2 = NessHermMat::new(NessGrid::new(nft, h), 1);
        let mut h2 = kadanoffbaym::mat::DenseMat::zeros(2);
        dyson_ness(&mut g2, mu, &h2, &s, FftIntegralMethod::Trapez, DysonBathType::None, 0.0, 1.0, 0.0);
    }));
    assert!(panics(|| {
        let mut g2 = NessHermMat::new(NessGrid::new(nft, h), 1);
        dyson_ness(&mut g2, mu, &h0, &s, FftIntegralMethod::Cubic, DysonBathType::None, 0.0, 1.0, 0.0);
    }));
}

#[test]
fn ness_utils_dyson_ness_sentinels() {
    let (nft, h) = (64i64, 0.1f64);
    let grid = NessGrid::new(nft, h);
    let sigma = |scale: f64| -> NessHermMat {
        let mut s = NessHermMat::new(grid, 2);
        for n in 0..nft {
            let t = time_at(&grid, n);
            let w = c(scale * (-t * t).exp(), 0.0);
            let d = [w * c(1.0, 0.0), w * c(0.2, 0.0), w * c(0.2, 0.0), w * c(0.5, 0.0)];
            let l: Vec<C64> = d.iter().map(|z| c(0.0, 1.0) * *z).collect();
            s.set_les(n, FftDomain::Time, &l);
            if t >= 0.0 && n <= nft / 2 {
                let r: Vec<C64> = d.iter().map(|z| c(0.0, -1.0) * *z).collect();
                s.set_ret(n, FftDomain::Time, &r);
            }
        }
        s
    };
    let s1 = sigma(0.3);
    let s0 = s1.clone();
    let mut g = NessHermMat::new(grid, 2);
    for (q, z) in g.ret.freq.iter_mut().enumerate() {
        *z = c((q + 1) as f64, 0.5);
    }
    for (q, z) in g.les.freq.iter_mut().enumerate() {
        *z = c((g.ret.freq.len() - q) as f64, -0.25);
    }
    let mut h0 = kadanoffbaym::mat::DenseMat::zeros(2);
    h0.set(0, 0, c(0.2, 0.0));
    h0.set(0, 1, c(0.0, 0.1));
    h0.set(1, 0, c(0.0, -0.1));
    h0.set(1, 1, c(-0.3, 0.0));
    dyson_ness(&mut g, 0.1, &h0, &s1, FftIntegralMethod::Trapez, DysonBathType::Const, 0.1, 2.0, 0.0);
    // the input self-energy is not changed
    assert_eq!(s1.ret.time, s0.ret.time);
    assert_eq!(s1.les.time, s0.les.time);
    // only the central band is filled
    let nfreq = (nft / 2 - 1) / 3;
    for n in 0..nft {
        let in_band = ((-nfreq + 1)..=(nfreq - 1)).any(|w| w.rem_euclid(nft) == n);
        if !in_band {
            assert!(g.ret.freq[n as usize * 4..(n as usize + 1) * 4].iter().all(|z| *z == C64::ZERO));
            assert!(g.les.freq[n as usize * 4..(n as usize + 1) * 4].iter().all(|z| *z == C64::ZERO));
        }
    }
    // second call with other sentinels and another self-energy
    let s2 = sigma(0.7);
    g.ret.freq.fill(c(7.0, 0.0));
    g.les.freq.fill(c(0.0, -3.0));
    dyson_ness(&mut g, 0.1, &h0, &s2, FftIntegralMethod::Trapez, DysonBathType::Const, 0.1, 2.0, 0.0);
    let mut f = NessHermMat::new(grid, 2);
    dyson_ness(&mut f, 0.1, &h0, &s2, FftIntegralMethod::Trapez, DysonBathType::Const, 0.1, 2.0, 0.0);
    assert_eq!(g.ret.time, f.ret.time);
    assert_eq!(g.les.time, f.les.time);
    // lesser hermiticity and Nyquist
    for m in 1..(nft / 2) as usize {
        let a = g.les.block(FftDomain::Time, m as i64).to_vec();
        let b = g.les.block(FftDomain::Time, nft - m as i64).to_vec();
        for q in 0..4 {
            assert!((b[q] + a[(q % 2) * 2 + q / 2].conj()).abs() < 1e-12);
        }
    }
    assert!(g.les.block(FftDomain::Time, nft / 2).iter().all(|z| *z == C64::ZERO));
}

#[test]
fn ness_utils_from_dos_ness_trapez_adaptive() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let (dos, mu): (&dyn Fn(f64) -> f64, f64) = if sign == Sign::Fermi {
            (&|x| GaussianDos::new(0.0, 1.0).at(x), 0.3)
        } else {
            (&|x| OhmicSymDos::new(1.0).at(x), 0.0)
        };
        let (lo, hi) = if sign == Sign::Fermi { (-12.0, 12.0) } else { (-20.0, 20.0) };
        let mut a = NessHermMat::new(NessGrid::new(1024, 0.1), 2);
        let mut b = NessHermMat::new(NessGrid::new(1024, 0.1), 2);
        from_dos_ness(&mut a, sign, dos, 2.0, mu, FftIntegralMethod::Trapez, None, None, 100, 20);
        from_dos_ness(&mut b, sign, dos, 2.0, mu, FftIntegralMethod::Adaptive, Some(lo), Some(hi), 100, 20);
        for q in 0..(200 * 4) {
            assert!((a.ret.time[q] - b.ret.time[q]).abs() < 1e-5, "ret q {q}");
            assert!((a.les.time[q] - b.les.time[q]).abs() < 1e-3, "les q {q}");
        }
        // spectral normalization and diagonal structure
        if sign == Sign::Fermi {
            let want = c(0.0, -(2.0 * std::f64::consts::PI).sqrt() / 2.506623299999997);
            assert!((a.ret.time[0] - want).abs() < 1e-10);
        } else {
            assert!(a.ret.time[0].abs() < 1e-10);
        }
        assert_eq!(a.ret.time[4 * 5 + 0 * 2 + 1], C64::ZERO);
        for q in (512 * 4)..(1024 * 4) {
            assert_eq!(a.ret.time[q], C64::ZERO);
        }
        if sign == Sign::Fermi {
            // occupation n = integral A f, G^<(0) = i n
            let n: f64 = (0..24001)
                .map(|k| {
                    let x = -6.0 + 12.0 * k as f64 / 24000.0;
                    dos(x) / (1.0 + (2.0 * (x - mu)).exp())
                })
                .sum::<f64>()
                * 12.0
                / 24000.0;
            assert!((a.les.time[0] - c(0.0, n)).abs() < 1e-8);
        }
        // the adaptive method without a window is an error (a plain
        // closure carries no bounds)
        assert!(panics(|| {
            let mut bb = NessHermMat::new(NessGrid::new(1024, 0.1), 2);
            from_dos_ness(&mut bb, sign, dos, 2.0, mu, FftIntegralMethod::Adaptive, None, None, 100, 20)
        }));
        from_dos_ness(&mut b, sign, dos, 2.0, mu, FftIntegralMethod::Adaptive, Some(lo), Some(hi), 100, 20);
        assert!(b.les.time[0].re.is_finite() && b.les.time[0].im.is_finite());
        // (the invalid integer sign of the blueprint is unrepresentable)
        assert!(panics(|| {
            let mut bb = NessHermMat::new(NessGrid::new(1024, 0.1), 2);
            from_dos_ness(&mut bb, sign, dos, 2.0, mu, FftIntegralMethod::Cubic, None, None, 100, 20)
        }));
    }
}

#[test]
fn ness_utils_from_dos_ness_vs_contour() {
    let (nft, beta, mu, h) = (64i64, 10.0f64, 0.0, 0.1);
    let dos = GaussianDos::new(0.0, 2.0);
    let cont = nu_contour(nft / 2 + 1, 11, 1, h, beta);
    let mut gc = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    from_dos(&mut gc, &|x| dos.at(x), dos.lo, dos.hi, beta, h, mu, 100, 20);
    let mut gf = NessHermMat::new(NessGrid::new(nft, h), 1);
    cntr2ness(&mut gf, &gc, nft / 2);
    let mut gd = NessHermMat::new(NessGrid::new(nft, h), 1);
    from_dos_ness(
        &mut gd,
        Sign::Fermi,
        &|x| dos.at(x),
        beta,
        mu,
        FftIntegralMethod::Adaptive,
        Some(dos.lo),
        Some(dos.hi),
        100,
        20,
    );
    for q in 0..gd.ret.time.len() {
        assert!((gf.ret.time[q] - gd.ret.time[q]).abs() < 1e-12, "ret {q}");
        assert!((gf.les.time[q] - gd.les.time[q]).abs() < 1e-12, "les {q}");
    }
}

#[test]
fn ness_utils_equilibrate_lesser() {
    // Catch case: the retarded data is put into the frequency domain
    let (nft, beta, mu, h) = (64i64, 10.0f64, 0.0, 0.1);
    let dos = GaussianDos::new(0.0, 2.0);
    let mut gr = NessHermMat::new(NessGrid::new(nft, h), 1);
    from_dos_ness(
        &mut gr,
        Sign::Fermi,
        &|x| dos.at(x),
        beta,
        mu,
        FftIntegralMethod::Adaptive,
        Some(dos.lo),
        Some(dos.hi),
        100,
        20,
    );
    let mut ge = NessHermMat::new(NessGrid::new(nft, h), 1);
    ge.ret.freq.copy_from_slice(&gr.ret.time);
    equilibrate_lesser(&mut ge, Sign::Fermi, beta, mu);
    assert!((distance_ness(&ge, &gr, FftDomain::Freq) / nft as f64) < 1e-10);
    // physical case: agrees with from_dos_ness! (TRAPEZ)
    for &(sign, d, m) in &[
        (Sign::Fermi, GaussianDos::new(0.0, 1.0), 0.3),
        (Sign::Bose, GaussianDos::new(3.0, 0.5), -1.0),
    ] {
        let mut g = NessHermMat::new(NessGrid::new(1024, 0.1), 2);
        from_dos_ness(&mut g, sign, &|x| d.at(x), 2.0, m, FftIntegralMethod::Trapez, None, None, 100, 20);
        let mut f = g.clone();
        f.les.time.fill(C64::ZERO);
        let t = f.ret.time.clone();
        equilibrate_lesser(&mut f, sign, 2.0, m);
        assert_eq!(f.ret.time, t);
        assert!((distance_ness_array(&f.les, &g.les, FftDomain::Time) / 1024.0) < 1e-10);
        assert!(f.les.block(FftDomain::Time, 512).iter().all(|z| *z == C64::ZERO));
    }
    // bosonic zero mode w = mu = 0 with a linear zero crossing
    let lin = |x: f64| x * (-x * x).exp();
    let mut g = NessHermMat::new(NessGrid::new(1024, 0.1), 1);
    from_dos_ness(&mut g, Sign::Bose, &lin, 2.0, 0.0, FftIntegralMethod::Trapez, None, None, 100, 20);
    let mut f = g.clone();
    equilibrate_lesser(&mut f, Sign::Bose, 2.0, 0.0);
    assert!(f.les.time.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    let r = &f.ret.freq;
    let spec = |n: usize| -(r[n] - r[n].conj()) / (2.0 * std::f64::consts::PI * c(0.0, 1.0));
    let slope = 0.5 * ((spec(1) - spec(0)) + (spec(0) - spec(1023))) / f.grid.dw;
    let mut gfreq = f.clone();
    integral_transform_to_freq(&mut gfreq, FftIntegralMethod::Trapez);
    assert!((f.les.freq[0] - (-2.0 * std::f64::consts::PI * c(0.0, 1.0) / 2.0) * slope).abs()
        < 1e-12 * slope.abs());
    assert!((distance_ness_array(&f.les, &g.les, FftDomain::Time) / 1024.0) < 1e-4);
    // non-zero and non-linear zero modes
    let bads: Vec<&(dyn Fn(f64) -> f64)> = vec![
        &|x: f64| (-x * x).exp(),
        &|x: f64| x * x * (-x * x).exp(),
    ];
    for bad in bads {
        let mut g = NessHermMat::new(NessGrid::new(1024, 0.1), 1);
        from_dos_ness(&mut g, Sign::Fermi, bad, 2.0, 0.0, FftIntegralMethod::Trapez, None, None, 100, 20);
        assert!(panics(move || {
            let mut gg = g.clone();
            equilibrate_lesser(&mut gg, Sign::Bose, 2.0, 0.0);
        }));
    }
}

#[test]
fn ness_utils_density_matrices() {
    let (h, nt, ntau, beta, mu) = (0.01f64, 1024i64, 100i64, 10.0, 0.0);
    let mut gn = NessHermMat::new(NessGrid::new(nt, h), 1);
    let dos = GaussianDos::new(0.0, 2.0);
    from_dos_ness(
        &mut gn,
        Sign::Fermi,
        &|x| dos.at(x),
        beta,
        mu,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let cont = nu_contour(nt + 1, ntau + 1, 1, h, beta);
    let mut gc = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    from_dos(&mut gc, &|x| dos.at(x), dos.lo, dos.hi, beta, h, mu, 100, 20);
    let rhoc = density_matrix_green(&gc, 0);
    let rhon = density_matrix_ness(&gn, Sign::Fermi);
    assert!((rhoc.get(0, 0).re - rhon[0].re).abs() < 1e-6);
    // (the invalid integer sign is unrepresentable)
    // convolution density matrix
    let dosb = GaussianDos::new(0.0, 4.0);
    let an = gn.clone();
    let mut bn = NessHermMat::new(NessGrid::new(nt, h), 1);
    from_dos_ness(
        &mut bn,
        Sign::Fermi,
        &|x| dosb.at(x),
        beta,
        mu,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let mut bc = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    from_dos(&mut bc, &|x| dosb.at(x), dosb.lo, dosb.hi, beta, h, mu, 100, 20);
    let a0 = an.clone();
    let b0 = bn.clone();
    let rhon = convolution_density_matrix_ness(&an, &bn, Sign::Fermi);
    assert_eq!(an.ret.time, a0.ret.time);
    assert_eq!(bn.ret.time, b0.ret.time);
    let rhoc = convolution_density_matrix(
        0,
        &gc,
        &gc,
        None,
        &bc,
        &bc,
        &Integrator::new(5),
        beta,
        h,
    );
    assert!((rhoc.get(0, 0).re - rhon[0].re).abs() < 1e-3);
    assert!(panics(|| convolution_density_matrix_ness(&an, &NessHermMat::new(NessGrid::new(nt, h), 2), Sign::Fermi)));
    assert!(panics(|| convolution_density_matrix_ness(&an, &NessHermMat::new(NessGrid::new(2 * nt, h), 1), Sign::Fermi)));
}

#[test]
fn ness_utils_bubble_vs_contour() {
    let (nft, ntau, beta, h) = (32i64, 20i64, 10.0f64, 0.1);
    let nt = nft / 2 - 1;
    let ca = nu_contour(nt + 1, ntau + 1, 2, h, beta);
    let cb = nu_contour(nt + 1, ntau + 1, 1, h, beta);
    let mut a = ContourGreen::new(&ca, Sign::Fermi, C64::ZERO);
    let mut b = ContourGreen::new(&cb, Sign::Fermi, C64::ZERO);
    for t in 0..=nt {
        for t1 in 0..=t {
            let mr_base = |i: i64, j: i64| c(0.1 * t as f64 + 0.01 * t1 as f64 * t1 as f64 + 0.01 * i as f64, -0.02 * j as f64);
            let ml_base = |i: i64, j: i64| c(0.05 * t1 as f64 + 0.02 * i as f64 + 0.01 * t as f64 * t as f64, 0.03 * j as f64 + 0.005 * t as f64 * t1 as f64);
            let mut mr = vec![C64::ZERO; 4];
            let mut ml = vec![C64::ZERO; 4];
            for i in 0..2i64 {
                for j in 0..2i64 {
                    mr[(i * 2 + j) as usize] = mr_base(i, j);
                    ml[(i * 2 + j) as usize] = ml_base(i, j);
                }
            }
            let br0 = c(0.2 * t as f64 + 0.05 * t1 as f64, -0.1 * t1 as f64 + 0.03);
            let bl0 = c(0.1 * t1 as f64 + 0.03 * t as f64, 0.1 * t as f64 + 0.02);
            let (mut mr, mut ml) = (mr, ml);
            let (mut br, mut bl) = (vec![br0], vec![bl0]);
            if t == t1 {
                let ml0 = ml.clone();
                let mr0 = mr.clone();
                for q in 0..4usize {
                    let j2 = (q % 2) * 2 + q / 2;
                    ml[q] = ml0[q] - ml0[j2].conj();
                    mr[q] = mr0[q] - mr0[j2].conj();
                }
                br[0] = br[0] - br[0].conj();
                bl[0] = bl[0] - bl[0].conj();
            }
            a.ret.raw_block_at_mut(t, t1).copy_from_slice(&mr);
            a.less.raw_block_at_mut(t1, t).copy_from_slice(&ml);
            b.ret.raw_block_at_mut(t, t1).copy_from_slice(&br);
            b.less.raw_block_at_mut(t1, t).copy_from_slice(&bl);
        }
    }
    let mut an = NessHermMat::new(NessGrid::new(nft, h), 2);
    let mut bn = NessHermMat::new(NessGrid::new(nft, h), 1);
    cntr2ness(&mut an, &a, nt);
    cntr2ness(&mut bn, &b, nt);
    for bc in [0, 1] {
        for order in 1..=2 {
            let mut cc = ContourGreen::new(&ca, Sign::Bose, C64::ZERO);
            let mut cn2 = NessHermMat::new(NessGrid::new(nft, h), 2);
            for i in 0..2usize {
                for j in 0..2usize {
                    if order == 1 {
                        if bc == 0 {
                            bubble1(nt, &mut cc, i, j, &a, &a, i, j, &b, &b, 0, 0);
                            bubble1_ness(&mut cn2, i, j, &an, i, j, &bn, 0, 0);
                        } else {
                            bubble2(nt, &mut cc, i, j, &a, &a, i, j, &b, &b, 0, 0);
                            bubble2_ness(&mut cn2, i, j, &an, i, j, &bn, 0, 0);
                        }
                    } else if bc == 0 {
                        bubble1(nt, &mut cc, i, j, &b, &b, 0, 0, &a, &a, i, j);
                        bubble1_ness(&mut cn2, i, j, &bn, 0, 0, &an, i, j);
                    } else {
                        bubble2(nt, &mut cc, i, j, &b, &b, 0, 0, &a, &a, i, j);
                        bubble2_ness(&mut cn2, i, j, &bn, 0, 0, &an, i, j);
                    }
                }
            }
            let mut cn1 = NessHermMat::new(NessGrid::new(nft, h), 2);
            cntr2ness(&mut cn1, &cc, nt);
            assert!(distance_ness(&cn1, &cn2, FftDomain::Time) < 1e-10);
        }
    }
    // scalar version resets the output
    let mut cs = NessHermMat::new(NessGrid::new(nft, h), 1);
    cs.ret.time.fill(c(9.0, 0.0));
    cs.les.time.fill(c(9.0, 0.0));
    bubble1_ness_scalar(&mut cs, &bn, &bn);
    for q in ((nft / 2) * 1) as usize..nft as usize {
        assert_eq!(cs.ret.time[q], C64::ZERO);
    }
    assert_eq!(cs.les.time[(nft / 2) as usize], C64::ZERO);
    assert!(panics(|| bubble2_ness(&mut cs, 0, 1, &bn, 0, 0, &bn, 0, 0)));
    assert!(panics(|| bubble2_ness_scalar(&mut cs, &bn, &NessHermMat::new(NessGrid::new(2 * nft, h), 1))));
}

#[test]
fn ness_utils_contour_round_trip() {
    let (nft, sz, ntau) = (32i64, 2usize, 20i64);
    let nt = nft / 2 - 1;
    let h = 0.1f64;
    let mut gn = NessHermMat::new(NessGrid::new(nft, h), sz);
    for t in 0..=nt {
        let mut mr = vec![C64::ZERO; sz * sz];
        let mut ml = vec![C64::ZERO; sz * sz];
        for i in 0..sz as i64 {
            for j in 0..sz as i64 {
                mr[(i * sz as i64 + j) as usize] =
                    c(0.01 * t as f64 + 0.001 * i as f64, -0.002 * j as f64 + 0.1 * t as f64);
                ml[(i * sz as i64 + j) as usize] =
                    c(0.002 * t as f64 + 0.001 * i as f64, 0.001 * j as f64 + 0.1 * t as f64 * t as f64);
            }
        }
        if t == 0 {
            let ml0 = ml.clone();
            let mr0 = mr.clone();
            for q in 0..(sz * sz) {
                let j2 = (q % sz) * sz + q / sz;
                mr[q] = mr0[q] - mr0[j2].conj();
                ml[q] = ml0[q] - ml0[j2].conj();
            }
        }
        gn.set_ret(t, FftDomain::Time, &mr);
        gn.set_les(t, FftDomain::Time, &ml);
        if t > 0 {
            let mut neg = vec![C64::ZERO; sz * sz];
            for i in 0..sz {
                for j in 0..sz {
                    neg[i * sz + j] = -ml[j * sz + i].conj();
                }
            }
            gn.set_les(nft - t, FftDomain::Time, &neg);
        }
    }
    let cont = nu_contour(nt + 1, ntau + 1, sz as i64, h, 1.0);
    let mut g = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    // sentinels in the Matsubara and mixing components
    g.mat.block_mut(3).fill(c(7.0, 0.0));
    g.lmix.raw_block_at_mut(2, 4).fill(c(-5.0, 0.0));
    ness2cntr(&mut g, &gn);
    assert!(g.mat.block(3).iter().all(|z| *z == c(7.0, 0.0)));
    assert!(g.lmix.raw_block_at(2, 4).iter().all(|z| *z == c(-5.0, 0.0)));
    let mut gb = NessHermMat::new(NessGrid::new(nft, h), sz);
    cntr2ness(&mut gb, &g, nt);
    assert!(distance_ness(&gn, &gb, FftDomain::Time) < 1e-12);
    // translation invariance
    assert_eq!(
        g.ret.raw_block_at(nt, 2).to_vec(),
        g.ret.raw_block_at(nt - 2, 0).to_vec()
    );
    // beyond the half grid the values are zero
    let cont2 = nu_contour(nft, ntau + 1, sz as i64, h, 1.0);
    let mut g2 = ContourGreen::new(&cont2, Sign::Fermi, C64::ZERO);
    g2.ret.raw_block_at_mut(nft - 1, 0).fill(C64::ONE);
    ness2cntr(&mut g2, &gn);
    assert!(g2.ret.raw_block_at(nft - 1, 0).iter().all(|z| *z == C64::ZERO));
    assert!(g2.less.raw_block_at(0, nft - 1).iter().all(|z| *z == C64::ZERO));
    assert_eq!(
        g2.ret.raw_block_at(nft - 1, nft / 2).to_vec(),
        gn.ret.block(FftDomain::Time, nft / 2 - 1).to_vec()
    );
    // tstp smaller than the half grid: the tail is zero
    let mut gt = NessHermMat::new(NessGrid::new(nft, h), sz);
    cntr2ness(&mut gt, &g, 3);
    for q in (4 * sz * sz)..(nft as usize * sz * sz) {
        assert_eq!(gt.ret.time[q], C64::ZERO);
    }
    assert_eq!(
        gt.ret.block(FftDomain::Time, 3).to_vec(),
        g.ret.raw_block_at(3, 0).to_vec()
    );
    assert!(panics(|| cntr2ness(&mut gt, &g, -1)));
    assert!(panics(|| cntr2ness(&mut gt, &g, nt + 1)));
    assert!(panics(|| cntr2ness(&mut NessHermMat::new(NessGrid::new(nft, h), 1), &g, 2)));
    assert!(panics(|| ness2cntr(&mut g, &NessHermMat::new(NessGrid::new(nft, h), 1))));
}

#[test]
fn ness_utils_resampling() {
    let (h1, h2, n1, n2) = (0.01f64, 0.1f64, 1000i64, 100i64);
    let (beta, mu) = (10.0f64, 0.0);
    let dos = GaussianDos::new(0.0, 2.0);
    let mut g1 = NessHermMat::new(NessGrid::new(n1, h1), 1);
    let mut g2 = NessHermMat::new(NessGrid::new(n2, h2), 1);
    from_dos_ness(&mut g1, Sign::Fermi, &|x| dos.at(x), beta, mu, FftIntegralMethod::Trapez, None, None, 100, 20);
    from_dos_ness(&mut g2, Sign::Fermi, &|x| dos.at(x), beta, mu, FftIntegralMethod::Trapez, None, None, 100, 20);
    let g10 = g1.clone();
    let gdo = downsample(&g1, 10);
    assert!((gdo.grid.h - h2).abs() < 1e-14);
    assert!((gdo.grid.dw - g1.grid.dw).abs() < 1e-14);
    assert!((distance_ness(&gdo, &g2, FftDomain::Time) / n2 as f64) < 1e-6);
    let gup = upsample(&g2, 10);
    assert!((gup.grid.h - h1).abs() < 1e-14);
    assert_eq!(gup.grid.nft, n1);
    assert!((distance_ness(&gup, &g1, FftDomain::Time) / n1 as f64) < 1e-6);
    assert!((distance_ness(&resample(&g1, 10), &g2, FftDomain::Time) / n2 as f64) < 1e-6);
    assert!((distance_ness(&resample(&g2, -10), &g1, FftDomain::Time) / n1 as f64) < 1e-6);
    assert_eq!(g1.ret.time, g10.ret.time);
    assert!(panics(|| resample(&g1, 0)));
    assert!(panics(|| downsample(&g1, 3)));
    assert!(panics(|| upsample(&g1, 0)));
    // NessArray: every factor-th time point
    let mut a = NessArray::new(8, 1);
    for k in 0..8 {
        a.time[k] = c((k + 1) as f64, 0.0);
        a.freq[k] = c((k + 11) as f64, 0.0);
    }
    let b = downsample_array(&a, 2);
    assert_eq!(b.time, vec![c(1.0, 0.0), c(3.0, 0.0), c(5.0, 0.0), c(7.0, 0.0)]);
    assert_eq!(b.freq, vec![c(11.0, 0.0), c(12.0, 0.0), c(17.0, 0.0), c(18.0, 0.0)]);
}
