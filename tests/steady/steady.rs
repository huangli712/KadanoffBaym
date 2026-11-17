// Project : KadanoffBaym
// File    : tests/ness.rs
// Purpose : blueprint t_ness: the NESS2 data layer - grid, arrays,
//           elements, algebra, the integral transforms and the plain
//           text I/O (ported NESSi Catch tests)
//
// Known deviations: the FFT scratch is internal to the transform
// calls (no public scratch field), the blueprint `strides` check
// is replaced by the block-size check, and the random data of the I/O
// testset use the deterministic splitmix64 generator.

use super::util::*;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::fourier::{
    FftIntegralMethod, integral_transform_to_freq, integral_transform_to_time,
};
use kadanoffbaym::types::{FftDomain, SteadyGreen, SteadyGrid};

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// The Catch integral-transform round trip (blueprint
/// `steady_catch_roundtrip`): four round-trip errors / Nft.
fn steady_catch_roundtrip(nft: i64, method: FftIntegralMethod) -> Vec<f64> {
    let h = 0.01f64;
    let nt = nft / 2 - 1;
    let f = |x: f64| {
        let e = (-h * h * x * x).exp();
        c((h * x).sin() * e, (h * x).cos() * e)
    };
    let mut errs = Vec::new();
    for comp in 0..2 {
        let set = |s: &mut SteadyGreen, m: i64, v: Complex| {
            if comp == 0 {
                s.set_ret(m, FftDomain::Time, &[v]);
            } else {
                s.set_les(m, FftDomain::Time, &[v]);
            }
        };
        // initialized in time
        let mut s = SteadyGreen::new(SteadyGrid::new(nft, h), 1);
        for t in 0..=nt {
            let mut v = f(t as f64);
            if t == 0 {
                v -= v.conj();
            }
            set(&mut s, t, v);
        }
        integral_transform_to_freq(&mut s, method);
        let mut s1 = s.clone();
        integral_transform_to_time(&mut s1, method);
        let d = if comp == 0 {
            s.ret.distance(&s1.ret, FftDomain::Time)
        } else {
            s.les.distance(&s1.les, FftDomain::Time)
        };
        errs.push(d / nft as f64);
        // initialized in frequency
        let mut w = SteadyGreen::new(SteadyGrid::new(nft, h), 1);
        for m in 0..nft {
            let (mm, v) = (m, f(m as f64));
            if comp == 0 {
                w.set_ret(mm, FftDomain::Freq, &[v]);
            } else {
                w.set_les(mm, FftDomain::Freq, &[v]);
            }
        }
        integral_transform_to_time(&mut w, method);
        let mut w1 = w.clone();
        integral_transform_to_freq(&mut w1, method);
        let d = if comp == 0 {
            w.ret.distance(&w1.ret, FftDomain::Freq)
        } else {
            w.les.distance(&w1.les, FftDomain::Freq)
        };
        errs.push(d / nft as f64);
    }
    errs
}

#[test]
fn steady_transforms_catch_round_trip_1024() {
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        let errs = steady_catch_roundtrip(1024, method);
        assert!(
            errs.iter().all(|e| e.is_finite() && *e < 1.6e-2),
            "{errs:?}"
        );
    }
}
#[test]
fn steady_release_catch_round_trip_16384() {
    if !is_release() {
        return;
    }
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        let errs = steady_catch_roundtrip(16384, method);
        assert!(errs.iter().all(|e| *e < 1e-3), "{errs:?}");
    }
}
