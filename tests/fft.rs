// Project : KadanoffBaym
// File    : tests/fft.rs
// Purpose : port of julia/test/t_fft.jl — pure Rust FFT against the
//           naive DFT oracle, plan cache semantics, failure paths,
//           strided/batched transforms, and the release production gate

mod common;

use common::prelude::*;
use kadanoffbaym::fourier::{
    allocate_fft_scratch, clear_fft_plan_cache, dft_direct, fft, fft_exec, fft_inplace,
    fft_strided, plan_fft, FftDirection,
};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_c64(state: &mut u64) -> C64 {
    let re = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    C64::new(re, im)
}

fn rand_unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn randn(state: &mut u64) -> f64 {
    // Box-Muller
    let u1 = rand_unit(state).max(1e-300);
    let u2 = rand_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Relative error with the absolute fallback for near-zero references
/// (blueprint `ft_err`).
fn ft_err(y: &[C64], r: &[C64]) -> f64 {
    let nr: f64 = r.iter().map(|z| z.abs() * z.abs()).sum::<f64>().sqrt();
    let d: f64 = y
        .iter()
        .zip(r)
        .map(|(a, b) | {
            let dz = *a - *b;
            dz.abs() * dz.abs()
        })
        .sum::<f64>()
        .sqrt();
    if nr > 1e-12 {
        d / nr
    } else {
        d
    }
}

const NS: [usize; 22] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 25, 27, 32, 49, 60, 64, 97, 101, 127, 256];

#[test]
fn accuracy_against_the_naive_dft() {
    let mut rng = 0x20261110u64;
    for n in NS {
        // four data sets: zeros, delta, centered random, lognormal phase
        let mut data: Vec<Vec<C64>> = Vec::new();
        data.push(vec![C64::ZERO; n]);
        let mut delta = vec![C64::ZERO; n];
        if n >= 1 {
            delta[0] = C64::ONE;
        }
        data.push(delta);
        data.push((0..n).map(|_| rand_c64(&mut rng)).collect());
        data.push(
            (0..n)
                .map(|_| {
                    let m = (10.0 * randn(&mut rng)).exp();
                    let ph = 2.0 * std::f64::consts::PI * rand_unit(&mut rng);
                    C64::from_polar(m, ph)
                })
                .collect(),
        );
        for x in &data {
            for d in [FftDirection::Forward, FftDirection::Backward] {
                let r = dft_direct(x, d);
                let x0 = x.clone();
                // out of place: the input is not changed
                let y = fft(x, d);
                assert_eq!(x, &x0, "input changed, n {n}");
                assert!(ft_err(&y, &r) <= 1e-12, "n {n} dir {d:?}: {}", ft_err(&y, &r));
                // in place with an explicit plan and scratch
                let p = plan_fft(n, d);
                let mut s = allocate_fft_scratch(&p);
                let mut z = x.clone();
                fft_exec(&mut z, &p, &mut s);
                assert!(ft_err(&z, &r) <= 1e-12, "in-place n {n} dir {d:?}");
            }
        }
        // real input
        let xr: Vec<C64> = (0..n).map(|_| C64::new(rand_unit(&mut rng), 0.0)).collect();
        let r = dft_direct(&xr, FftDirection::Forward);
        assert!(ft_err(&fft(&xr, FftDirection::Forward), &r) <= 1e-12, "real n {n}");
        // round trip
        let x: Vec<C64> = (0..n).map(|_| rand_c64(&mut rng)).collect();
        let mut y = fft(&x, FftDirection::Forward);
        y = fft(&y, FftDirection::Backward);
        for v in &mut y {
            *v = *v / C64::new(n as f64, 0.0);
        }
        assert!(ft_err(&y, &x) <= 1e-12, "round trip n {n}");
    }
}

#[test]
fn sign_and_normalization() {
    // single frequency: x_n = exp(+2 pi i n k0/N) is a delta for Forward
    for (n, k0) in [(12usize, 1usize), (12, 5), (97, 1), (97, 5)] {
        let x: Vec<C64> = (0..n)
            .map(|j| C64::from_polar(1.0, 2.0 * std::f64::consts::PI * ((j * k0) % n) as f64 / n as f64))
            .collect();
        let y = fft(&x, FftDirection::Forward);
        assert!((y[k0] - C64::new(n as f64, 0.0)).abs() < 1e-10, "n {n} k0 {k0}");
        let mut rest = 0.0f64;
        for (k, &v) in y.iter().enumerate() {
            if k != k0 {
                rest = rest.max(v.abs());
            }
        }
        assert!(rest < 1e-10, "n {n} k0 {k0}: rest {rest}");
        let z = fft(&x, FftDirection::Backward);
        let kn = (n - k0) % n;
        assert!((z[kn] - C64::new(n as f64, 0.0)).abs() < 1e-10, "backward n {n} k0 {k0}");
    }
    // unnormalized
    let ones7 = vec![C64::ONE; 7];
    assert!((fft(&ones7, FftDirection::Forward)[0] - C64::new(7.0, 0.0)).abs() < 1e-12);
    let ones8 = vec![C64::ONE; 8];
    assert!((fft(&ones8, FftDirection::Backward)[0] - C64::new(8.0, 0.0)).abs() < 1e-12);
}

#[test]
fn plans_cache_and_scratch() {
    clear_fft_plan_cache();
    let p = plan_fft(4093, FftDirection::Forward);
    assert_eq!(p.n, 4093);
    assert_eq!(p.bluestein_n, 8192);
    assert_eq!(p.scratch_len, 2 * p.bluestein_n);
    assert_eq!(allocate_fft_scratch(&p).len(), p.scratch_len);
    let p2 = plan_fft(4093, FftDirection::Forward);
    assert!(std::sync::Arc::ptr_eq(&p, &p2));
    let p3 = plan_fft(4093, FftDirection::Backward);
    assert!(!std::sync::Arc::ptr_eq(&p, &p3));
    clear_fft_plan_cache();
    let p4 = plan_fft(4093, FftDirection::Forward);
    assert!(!std::sync::Arc::ptr_eq(&p, &p4));

    let q = plan_fft(60, FftDirection::Forward);
    assert_eq!(q.bluestein_n, 0);
    assert_eq!(q.scratch_len, 60);
    assert_eq!(q.factors.iter().product::<usize>(), 60);
    assert_eq!(plan_fft(1, FftDirection::Forward).scratch_len, 1);
    // mixed products with a large prime use Bluestein
    assert_eq!(plan_fft(34, FftDirection::Forward).bluestein_n, 128);
    assert_eq!(plan_fft(26, FftDirection::Forward).bluestein_n, 0);
}

#[test]
#[should_panic]
fn fail_plan_fft_zero() {
    let _ = plan_fft(0, FftDirection::Forward);
}

#[test]
#[should_panic]
fn fail_fft_empty() {
    let mut v: Vec<C64> = Vec::new();
    fft_inplace(&mut v, FftDirection::Forward);
}

#[test]
#[should_panic]
fn fail_data_plan_mismatch() {
    let p = plan_fft(8, FftDirection::Forward);
    let mut z = vec![C64::ZERO; 9];
    let mut s = allocate_fft_scratch(&p);
    fft_exec(&mut z, &p, &mut s);
}

#[test]
#[should_panic]
fn fail_scratch_too_short() {
    let p = plan_fft(8, FftDirection::Forward);
    let mut z = vec![C64::ZERO; 8];
    let mut s = vec![C64::ZERO; 7];
    fft_exec(&mut z, &p, &mut s);
}

#[test]
#[should_panic]
fn fail_direction_mismatch() {
    // `fft_exec` encodes the direction in the plan (a mismatch is
    // unrepresentable); `fft_strided` takes both and checks them
    let p = plan_fft(8, FftDirection::Forward);
    let mut b = vec![C64::ZERO; 8];
    let a = vec![C64::ZERO; 8];
    let mut s = allocate_fft_scratch(&p);
    fft_strided(&mut b, &a, (1, 1, 8), 2, FftDirection::Backward, &p, &mut s);
}

#[test]
fn strided_and_batched_transforms() {
    let mut rng = 0xBEE2BEE2u64;
    for n in [30usize, 97] {
        let dims = (2usize, 2, n);
        let total = dims.0 * dims.1 * dims.2;
        let a: Vec<C64> = (0..total).map(|_| rand_c64(&mut rng)).collect();
        let mut b = vec![C64::ZERO; total];
        let p = plan_fft(n, FftDirection::Forward);
        let mut s = allocate_fft_scratch(&p);
        fft_strided(&mut b, &a, dims, 2, FftDirection::Forward, &p, &mut s);
        // every line along the last axis matches the naive DFT
        // (column-major layout: element (i, j, k) at i + j*2 + k*4)
        for i in 0..2 {
            for j in 0..2 {
                let base = i + j * 2;
                let line: Vec<C64> = (0..n).map(|k| a[base + k * 4]).collect();
                let want = dft_direct(&line, FftDirection::Forward);
                let got: Vec<C64> = (0..n).map(|k| b[base + k * 4]).collect();
                assert!(ft_err(&got, &want) <= 1e-12, "n {n} line ({i},{j})");
            }
        }
        // backward restores the input up to N
        let mut c = vec![C64::ZERO; total];
        let pb = plan_fft(n, FftDirection::Backward);
        let mut sb = allocate_fft_scratch(&pb);
        fft_strided(&mut c, &b, dims, 2, FftDirection::Backward, &pb, &mut sb);
        for q in 0..total {
            c[q] = c[q] / C64::new(n as f64, 0.0);
        }
        assert!(ft_err(&c, &a) < 1e-12, "round trip n {n}");
        // in place, dest === src
        let mut d = a.clone();
        kadanoffbaym::fourier::fft_strided_in_place(&mut d, dims, 2, FftDirection::Forward, &p, &mut s);
        assert_eq!(d, b, "in-place n {n}");
        // other dimensions: the first axis (stride n*2)
        let mut e = vec![C64::ZERO; total];
        let p1 = plan_fft(dims.0, FftDirection::Forward);
        let mut s1 = allocate_fft_scratch(&p1);
        fft_strided(&mut e, &a, dims, 0, FftDirection::Forward, &p1, &mut s1);
        // line (j, k) along the first axis (stride 1)
        let (j, k) = (1usize, 5usize.min(n - 1));
        let base = j * 2 + k * 4;
        let line: Vec<C64> = (0..2).map(|i| a[i + base]).collect();
        let want = dft_direct(&line, FftDirection::Forward);
        let got: Vec<C64> = (0..2).map(|i| e[i + base]).collect();
        assert!(ft_err(&got, &want) <= 1e-12, "dim-0 line n {n}");
    }
}

#[test]
fn release_production_gate() {
    if !common::is_release() {
        return;
    }
    let mut rng = 0x6A7EF00Du64;
    let mut times = std::collections::HashMap::new();
    for n in [4093usize, 4096, 16384] {
        // pulse
        let mut x = vec![C64::ZERO; n];
        x[0] = C64::ONE;
        let y = fft(&x, FftDirection::Forward);
        assert!(y.iter().all(|v| (*v - C64::ONE).abs() < 1e-12), "pulse {n}");
        // single frequency
        let k0 = 37usize;
        let x: Vec<C64> = (0..n)
            .map(|j| C64::from_polar(1.0, 2.0 * std::f64::consts::PI * ((j * k0) % n) as f64 / n as f64))
            .collect();
        let y = fft(&x, FftDirection::Forward);
        assert!((y[k0] - C64::new(n as f64, 0.0)).abs() / (n as f64) < 1e-12, "freq {n}");
        let mut rest = 0.0f64;
        for (k, &v) in y.iter().enumerate() {
            if k != k0 {
                rest = rest.max(v.abs());
            }
        }
        assert!(rest / (n as f64) < 1e-11, "freq rest {n}");
        // round trip
        let x: Vec<C64> = (0..n).map(|_| rand_c64(&mut rng)).collect();
        let p = plan_fft(n, FftDirection::Forward);
        let pb = plan_fft(n, FftDirection::Backward);
        let mut s = allocate_fft_scratch(&p);
        let mut z = x.clone();
        fft_exec(&mut z, &p, &mut s);
        fft_exec(&mut z, &pb, &mut s);
        for v in &mut z {
            *v = *v / C64::new(n as f64, 0.0);
        }
        assert!(ft_err(&z, &x) < 1e-12, "round trip {n}");
        // 2 x 2 batched along the last axis
        let dims = (2usize, 2, n);
        let total = 4 * n;
        let a: Vec<C64> = (0..total).map(|_| rand_c64(&mut rng)).collect();
        let mut b = vec![C64::ZERO; total];
        fft_strided(&mut b, &a, dims, 2, FftDirection::Forward, &p, &mut s);
        kadanoffbaym::fourier::fft_strided_in_place(&mut b, dims, 2, FftDirection::Backward, &pb, &mut s);
        for q in 0..total {
            b[q] = b[q] / C64::new(n as f64, 0.0);
        }
        assert!(ft_err(&b, &a) < 1e-12, "batched {n}");
        // minimum of the wall times over 5 runs
        let mut best = f64::INFINITY;
        let mut z = x.clone();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            fft_exec(&mut z, &p, &mut s);
            let dt = t.elapsed().as_secs_f64();
            best = best.min(dt);
        }
        times.insert(n, best);
    }
    // relative scaling, rejects O(N^2) paths
    assert!(times[&16384] / times[&4096] < 10.0, "scaling 16384/4096");
    assert!(times[&4093] / times[&4096] < 8.0, "scaling 4093/4096 (Bluestein)");
}
