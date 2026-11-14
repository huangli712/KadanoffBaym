// Project : KadanoffBaym
// File    : tests/ness.rs
// Purpose : blueprint t_ness.jl: the NESS2 data layer - grid, arrays,
//           elements, algebra, the integral transforms and the plain
//           text I/O (ported NESSi Catch tests)
//
// Known deviations: the FFT scratch is internal to the transform
// calls in Rust (no public scratch field), the Julia `strides` check
// is replaced by the block-size check, and the random data of the I/O
// testset use the deterministic splitmix64 generator.

mod common;

use common::*;
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::fourier::transforms::{
    fft_to_freq_array, fft_to_time_array, integral_transform_to_freq, integral_transform_to_time,
    FftIntegralMethod,
};
use kadanoffbaym::io::{ness_array_from_str, ness_array_to_string, ness_from_str, ness_to_string};
use kadanoffbaym::core::ness::{
    distance_ness, distance_ness_array, freq_at, incr_ness, incr_ness_mat, left_multiply_ness,
    left_multiply_ness_mat, memset_ness, memset_ness_mat, right_multiply_ness, right_multiply_ness_mat,
    set_matrixelement, set_matrixelement_mat, smul_ness, smul_ness_mat, time_at, FftDomain,
    NessArray, NessGrid, NessHermMat,
};

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn splitmix(state: &mut u64) -> u64 {
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

#[test]
fn ness_grid_info() {
    let (nt, h1, h2, dw2) = (1000i64, 0.1f64, 0.01, 0.1);
    let g = NessGrid::new(nt, h1);
    assert!(g.h == h1 && g.nft == nt);
    assert!((g.dw - std::f64::consts::PI / ((nt / 2) as f64 * h1)).abs() < 1e-10);
    let expect = |g: &NessGrid, m: i64, s: f64| {
        (if m < g.nft / 2 { m } else { m - g.nft }) as f64 * s
    };
    for gg in [g, g.set_h(h2), g.set_dw(dw2)] {
        let mut st = 0.0;
        let mut sf = 0.0;
        for m in 0..nt {
            st += (time_at(&gg, m) - expect(&gg, m, gg.h)).abs();
            sf += (freq_at(&gg, m) - expect(&gg, m, gg.dw)).abs();
        }
        assert!((st / nt as f64) < 1e-10);
        assert!((sf / nt as f64) < 1e-10);
    }
    assert!(g.set_h(h2).h == h2);
    assert!(g.set_dw(dw2).dw == dw2);
    assert!((g.set_dw(dw2).h - std::f64::consts::PI / ((nt / 2) as f64 * dw2)).abs() < 1e-14);
    // storage modes
    let g = NessGrid::new(16, 0.25);
    assert!((g.dw - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert_eq!(time_at(&g, 0), 0.0);
    assert_eq!(time_at(&g, 7), 7.0 * 0.25);
    assert_eq!(time_at(&g, 8), -8.0 * 0.25);
    assert_eq!(time_at(&g, 15), -0.25);
    assert_eq!(freq_at(&g, 8), -8.0 * g.dw);
    assert!(panics(|| freq_at(&g, -1)));
    assert!(panics(|| time_at(&g, 16)));
    assert!(panics(|| NessGrid::new(15, 0.1)));
    assert!(panics(|| NessGrid::new(2, 0.1)));
    assert!(panics(|| NessGrid::new(16, 0.0)));
    assert!(panics(|| g.set_h(-1.0)));
    assert!(panics(|| g.set_dw(0.0)));
}

#[test]
fn ness_array_basic() {
    let a = NessArray::new(16, 2);
    assert!(a.nft == 16 && a.size1 == 2 && a.size2 == 2);
    assert_eq!(a.time.len(), 2 * 2 * 16);
    assert_eq!(a.freq.len(), 2 * 2 * 16);
    // round trip of the bare transforms, unnormalized
    let nft = 8i64;
    let mut a = NessArray::new(nft, 2);
    for t in 0..nft {
        for i in 0..2usize {
            for j in 0..2usize {
                let v = c(
                    t as f64 + i as f64 * 0.1,
                    j as f64 * 0.2,
                );
                a.block_mut(FftDomain::Time, t)[i * 2 + j] = v;
            }
        }
    }
    let orig = a.time.clone();
    fft_to_freq_array(&mut a);
    assert_eq!(a.time, orig);
    fft_to_time_array(&mut a);
    for q in 0..a.time.len() {
        assert!((a.time[q] / 8.0 - orig[q]).abs() < 1e-10);
    }
    // a pulse at t = h gives exp(+i w h) in frequency
    let mut b = NessArray::new(nft, 1);
    b.block_mut(FftDomain::Time, 1)[0] = C64::ONE;
    fft_to_freq_array(&mut b);
    for k in 0..nft as usize {
        let want = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * k as f64 / nft as f64);
        assert!((b.freq[k] - want).abs() < 1e-12);
    }
}

#[test]
fn ness_array_elements() {
    let mut a = NessArray::new(4, 2);
    let m = vec![C64::ONE; 4];
    a.block_mut(FftDomain::Time, 0).copy_from_slice(&m);
    assert_eq!(a.block(FftDomain::Time, 0), &m[..]);
    // copies on get and set, no aliasing
    let mut x = a.block(FftDomain::Time, 0).to_vec();
    x[0] = c(5.0, 0.0);
    assert_eq!(a.block(FftDomain::Time, 0), &m[..]);
    let mut m2 = m.clone();
    m2[3] = c(7.0, 0.0);
    assert_eq!(a.block(FftDomain::Time, 0)[3], C64::ONE);
    // scalar
    let mut s = NessArray::new(4, 1);
    s.block_mut(FftDomain::Time, 0)[0] = c(1.23, 4.56);
    assert_eq!(s.block(FftDomain::Time, 0)[0], c(1.23, 4.56));
    // wrong sizes and out-of-range modes
    assert!(panics(|| a.block_mut(FftDomain::Time, 0).copy_from_slice(&[C64::ONE])));
    assert!(panics(|| a.block_mut(FftDomain::Time, 0).copy_from_slice(&vec![C64::ZERO; 9])));
    assert!(panics(|| a.block(FftDomain::Time, 4)));
    // set_matrixelement
    let mut b = NessArray::new(4, 2);
    for t in 0..4 {
        let blk = [C64::ZERO, c(1.0, 1.0), C64::ZERO, C64::ZERO];
        b.block_mut(FftDomain::Time, t).copy_from_slice(&blk);
    }
    let mut a2 = NessArray::new(4, 2);
    set_matrixelement(&mut a2, 1, 0, &b, 0, 1, FftDomain::Time);
    for t in 0..4 {
        assert_eq!(a2.block(FftDomain::Time, t)[1 * 2 + 0], c(1.0, 1.0));
    }
    assert!(panics(|| set_matrixelement(&mut a2, 1, 0, &NessArray::new(8, 2), 0, 1, FftDomain::Time)));
}

#[test]
fn ness_array_ops() {
    let mut a = NessArray::new(4, 2);
    let mut b = NessArray::new(4, 2);
    for t in 0..4 {
        for i in 0..2usize {
            for j in 0..2usize {
                b.block_mut(FftDomain::Time, t)[i * 2 + j] =
                    c(t as f64 + i as f64 + j as f64 + (i * j) as f64, 0.0);
            }
        }
    }
    a.zeros_domain(FftDomain::Time);
    incr_ness(&mut a, &b, c(2.0, 0.0), FftDomain::Time);
    for q in 0..a.time.len() {
        assert!((a.time[q] - 2.0 * b.time[q]).abs() < 1e-12);
    }
    smul_ness(&mut a, c(0.5, 0.0), FftDomain::Time);
    for q in 0..a.time.len() {
        assert!((a.time[q] - b.time[q]).abs() < 1e-12);
    }
    // the frequency data is not touched
    assert!(a.freq.iter().all(|z| *z == C64::ZERO));
    // left and right multiplication by the swap matrix
    for t in 0..4 {
        for i in 0..2usize {
            for j in 0..2usize {
                a.block_mut(FftDomain::Time, t)[i * 2 + j] =
                    c(1.0 + i as f64 + j as f64, t as f64);
            }
        }
    }
    let mswap = [C64::ZERO, C64::ONE, C64::ONE, C64::ZERO];
    let mut cc = a.clone();
    left_multiply_ness(&mut cc, &mswap, FftDomain::Time, false);
    left_multiply_ness(&mut cc, &mswap, FftDomain::Time, false);
    for q in 0..cc.time.len() {
        assert!((cc.time[q] - a.time[q]).abs() < 1e-12);
    }
    right_multiply_ness(&mut cc, &mswap, FftDomain::Time, false);
    right_multiply_ness(&mut cc, &mswap, FftDomain::Time, false);
    for q in 0..cc.time.len() {
        assert!((cc.time[q] - a.time[q]).abs() < 1e-12);
    }
    let p = [C64::ONE, c(0.0, 2.0), C64::ZERO, c(3.0, 0.0)];
    left_multiply_ness(&mut cc, &p, FftDomain::Time, false);
    // C[3] = P * A[3]
    let src = a.block(FftDomain::Time, 2).to_vec();
    let got = cc.block(FftDomain::Time, 2);
    for i in 0..2 {
        for j in 0..2 {
            let want = p[i * 2] * src[j] + p[i * 2 + 1] * src[2 + j];
            assert!((got[i * 2 + j] - want).abs() < 1e-12);
        }
    }
    let mut cc = a.clone();
    right_multiply_ness(&mut cc, &p, FftDomain::Time, true);
    // C[2] = A[2] * P'
    let src = a.block(FftDomain::Time, 1).to_vec();
    let got = cc.block(FftDomain::Time, 1);
    for i in 0..2 {
        for j in 0..2 {
            let want = src[i * 2] * p[j * 2].conj() + src[i * 2 + 1] * p[j * 2 + 1].conj();
            assert!((got[i * 2 + j] - want).abs() < 1e-12);
        }
    }
    let mut cc = a.clone();
    left_multiply_ness(&mut cc, &p, FftDomain::Time, true);
    let src = a.block(FftDomain::Time, 1).to_vec();
    let got = cc.block(FftDomain::Time, 1);
    for i in 0..2 {
        for j in 0..2 {
            let want = p[i].conj() * src[j] + p[2 + i].conj() * src[2 + j];
            assert!((got[i * 2 + j] - want).abs() < 1e-12);
        }
    }
    assert!(panics(|| left_multiply_ness(&mut cc, &vec![C64::ZERO; 9], FftDomain::Time, false)));
    // memset / zeros / memcpy / distance
    memset_ness(&mut a, FftDomain::Freq, c(2.0, 0.0));
    assert!(a.freq.iter().all(|z| *z == c(2.0, 0.0)));
    a.zeros_domain(FftDomain::Time);
    a.zeros_domain(FftDomain::Freq);
    assert!(a.time.iter().chain(a.freq.iter()).all(|z| *z == C64::ZERO));
    let d = b.clone();
    assert_eq!(d.time, b.time);
    assert!(panics(|| incr_ness(&mut a, &NessArray::new(8, 2), C64::ONE, FftDomain::Time)));
    assert_eq!(distance_ness_array(&a, &a, FftDomain::Time), 0.0);
    let mut e = NessArray::new(4, 2);
    e.time[2 * 4 + 0 * 2 + 1] = c(3.0, 4.0);
    assert_eq!(distance_ness_array(&e, &NessArray::new(4, 2), FftDomain::Time), 5.0);
}

#[test]
fn ness_herm_mat_elements_and_ops() {
    let mut g = NessHermMat::new(NessGrid::new(16, 0.1), 2);
    assert_eq!(g.ret.time.len(), 2 * 2 * 16);
    assert!(g.grid.nft == 16 && g.ret.size1 == 2);
    g.set_les(15, FftDomain::Time, &[C64::ONE; 4]);
    assert_eq!(g.get_les(15, FftDomain::Time), vec![C64::ONE; 4]);
    let m = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    g.set_ret(0, FftDomain::Freq, &m);
    assert_eq!(g.get_ret(0, FftDomain::Freq), m.to_vec());
    // scalar
    let mut s = NessHermMat::new(NessGrid::new(4, 0.1), 1);
    s.set_ret(0, FftDomain::Time, &[c(1.23, 4.56)]);
    s.set_les(0, FftDomain::Time, &[c(1.23, 4.56)]);
    assert_eq!(s.get_ret(0, FftDomain::Time)[0], c(1.23, 4.56));
    assert_eq!(s.get_les(0, FftDomain::Time)[0], c(1.23, 4.56));
    // construction against NessArray
    let mut g = NessHermMat::new(NessGrid::new(4, 0.1), 2);
    let mut r = NessArray::new(4, 2);
    for t in 0..4i64 {
        let x: Vec<C64> = (0..2)
            .flat_map(|i| {
                (0..2).map(move |j| {
                    c(
                        t as f64 + i as f64,
                        -2.0 * t as f64 + j as f64,
                    )
                })
            })
            .collect();
        r.block_mut(FftDomain::Time, t).copy_from_slice(&x);
        g.set_ret(t, FftDomain::Time, &x);
        g.set_les(t, FftDomain::Time, &x);
    }
    assert!(distance_ness_array(&g.ret, &r, FftDomain::Time) < 1e-12);
    assert!(distance_ness_array(&g.les, &r, FftDomain::Time) < 1e-12);
    // ops
    let mut a = NessHermMat::new(NessGrid::new(4, 0.1), 2);
    incr_ness_mat(&mut a, &g, c(2.0, 0.0), FftDomain::Time);
    for q in 0..a.ret.time.len() {
        assert!((a.ret.time[q] - 2.0 * g.ret.time[q]).abs() < 1e-12);
        assert!((a.les.time[q] - 2.0 * g.les.time[q]).abs() < 1e-12);
    }
    smul_ness_mat(&mut a, c(0.5, 0.0), FftDomain::Time);
    assert!(distance_ness(&a, &g, FftDomain::Time) < 1e-12);
    let mswap = [C64::ZERO, C64::ONE, C64::ONE, C64::ZERO];
    let mut b = g.clone();
    left_multiply_ness_mat(&mut b, &mswap, FftDomain::Time, false);
    left_multiply_ness_mat(&mut b, &mswap, FftDomain::Time, false);
    right_multiply_ness_mat(&mut b, &mswap, FftDomain::Time, false);
    right_multiply_ness_mat(&mut b, &mswap, FftDomain::Time, false);
    assert!(distance_ness(&b, &g, FftDomain::Time) < 1e-12);
    memset_ness_mat(&mut b, FftDomain::Freq, C64::ONE);
    b.zeros_domain(FftDomain::Time);
    b.zeros_domain(FftDomain::Freq);
    assert!(b.ret.time.iter().chain(b.les.freq.iter()).all(|z| *z == C64::ZERO));
    let cc = g.clone();
    assert_eq!(cc.ret.time, g.ret.time);
    let mut c2 = g.clone();
    set_matrixelement_mat(&mut c2, 0, 0, &g, 1, 1, FftDomain::Time);
    for t in 0..4 {
        assert_eq!(
            c2.get_les(t, FftDomain::Time)[0],
            g.get_les(t, FftDomain::Time)[1 * 2 + 1]
        );
    }
    assert!(c2.is_compatible(&g));
    assert!(!c2.is_compatible(&NessHermMat::new(NessGrid::new(8, 0.1), 2)));
}

#[test]
fn ness_transforms_analytic() {
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        // G^R(t) = -i e^{-(eta + i eps) t}, G^R(w) = 1/(w - eps + i eta)
        let (eps, eta) = (0.3f64, 1.0f64);
        let mut errs = Vec::new();
        for &h in &[0.1f64, 0.05, 0.025] {
            let nft = (102.4 / h).round() as i64;
            let mut g = NessHermMat::new(NessGrid::new(nft, h), 1);
            for n in 0..(nft / 2) {
                let t = n as f64 * h;
                let decay = (-(eta * t)).exp();
                g.set_ret(
                    n,
                    FftDomain::Time,
                    &[c(0.0, -decay) * C64::from_polar(1.0, -eps * t)],
                );
            }
            integral_transform_to_freq(&mut g, method);
            // Nyquist mode
            assert_eq!(g.ret.freq[(nft / 2) as usize], C64::ZERO);
            let grid = g.grid;
            let mut e = 0.0f64;
            for &n in &[0i64, 1, 4, 10, nft - 1] {
                let want = C64::ONE / (c(freq_at(&grid, n) - eps, eta));
                e = e.max((g.ret.freq[n as usize] - want).abs());
            }
            errs.push(e);
        }
        if method == FftIntegralMethod::Trapez {
            assert!((errs[0] / errs[1]) > 3.5 && (errs[1] / errs[2]) > 3.5, "{errs:?}");
        } else {
            assert!((errs[0] / errs[1]) > 12.0 && (errs[1] / errs[2]) > 12.0, "{errs:?}");
            assert!(errs[2] < 1e-8, "{errs:?}");
        }
    }
}

#[test]
fn ness_transforms_physical_round_trip() {
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        let (h, nft) = (0.1f64, 512i64);
        let mut g = NessHermMat::new(NessGrid::new(nft, h), 2);
        let u = [
            c(1.0 / 2f64.sqrt(), 0.0),
            c(0.0, 1.0 / 2f64.sqrt()),
            c(0.0, 1.0 / 2f64.sqrt()),
            c(1.0 / 2f64.sqrt(), 0.0),
        ];
        for n in 0..nft {
            let t = time_at(&g.grid, n);
            let d0 = (-t * t / 2.0).exp();
            let d1 = 0.5 * (-t * t / 3.0).exp();
            // L = i U diag(d0, d1) U'
            // U diag(d0, d1) U^dagger
            let ud = [
                u[0] * d0 * u[0].conj() + u[1] * d1 * u[1].conj(),
                u[0] * d0 * u[2].conj() + u[1] * d1 * u[3].conj(),
                u[2] * d0 * u[0].conj() + u[3] * d1 * u[2].conj(),
                u[2] * d0 * u[2].conj() + u[3] * d1 * u[3].conj(),
            ];
            let mut l = [C64::ZERO; 4];
            for q in 0..4 {
                l[q] = c(0.0, 1.0) * ud[q];
            }
            g.set_les(n, FftDomain::Time, &l);
            if t >= 0.0 && n <= nft / 2 - 1 {
                let ph = C64::from_polar(1.0, -0.3 * t);
                let mut r = [C64::ZERO; 4];
                for q in 0..4 {
                    r[q] = c(0.0, -1.0) * ud[q] * ph;
                }
                g.set_ret(n, FftDomain::Time, &r);
            }
        }
        let g0 = g.clone();
        integral_transform_to_freq(&mut g, method);
        integral_transform_to_time(&mut g, method);
        let tol = if method == FftIntegralMethod::Trapez {
            1e-10
        } else {
            1e-6
        };
        assert!(
            (distance_ness_array(&g.ret, &g0.ret, FftDomain::Time) / nft as f64) < tol,
            "ret"
        );
        assert!(
            (distance_ness_array(&g.les, &g0.les, FftDomain::Time) / nft as f64) < tol,
            "les"
        );
        // lesser hermiticity G^<(-t) = -G^<(t)'
        for m in 1..(nft / 2) as usize {
            let a = g.les.block(FftDomain::Time, m as i64).to_vec();
            let b = g.les.block(FftDomain::Time, nft - m as i64).to_vec();
            for q in 0..4 {
                assert!((b[q] + a[(q % 2) * 2 + q / 2].conj()).abs() < 1e-10);
            }
        }
        assert!(g.les.block(FftDomain::Time, nft / 2).iter().all(|z| *z == C64::ZERO));
        // retarded at negative times is zero
        for n in (nft / 2)..nft {
            assert!(g.ret.block(FftDomain::Time, n).iter().all(|z| *z == C64::ZERO));
        }
        assert!(panics(|| integral_transform_to_freq(&mut g, FftIntegralMethod::Adaptive)));
        assert!(panics(|| integral_transform_to_time(&mut g, FftIntegralMethod::Adaptive)));
    }
}

/// The Catch integral-transform round trip (blueprint
/// `ness_catch_roundtrip`): four round-trip errors / Nft.
fn ness_catch_roundtrip(nft: i64, method: FftIntegralMethod) -> Vec<f64> {
    let h = 0.01f64;
    let nt = nft / 2 - 1;
    let f = |x: f64| {
        let e = (-h * h * x * x).exp();
        c((h * x).sin() * e, (h * x).cos() * e)
    };
    let mut errs = Vec::new();
    for comp in 0..2 {
        let set = |s: &mut NessHermMat, m: i64, v: C64| {
            if comp == 0 {
                s.set_ret(m, FftDomain::Time, &[v]);
            } else {
                s.set_les(m, FftDomain::Time, &[v]);
            }
        };
        // initialized in time
        let mut s = NessHermMat::new(NessGrid::new(nft, h), 1);
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
            distance_ness_array(&s.ret, &s1.ret, FftDomain::Time)
        } else {
            distance_ness_array(&s.les, &s1.les, FftDomain::Time)
        };
        errs.push(d / nft as f64);
        // initialized in frequency
        let mut w = NessHermMat::new(NessGrid::new(nft, h), 1);
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
            distance_ness_array(&w.ret, &w1.ret, FftDomain::Freq)
        } else {
            distance_ness_array(&w.les, &w1.les, FftDomain::Freq)
        };
        errs.push(d / nft as f64);
    }
    errs
}

#[test]
fn ness_transforms_catch_round_trip_1024() {
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        let errs = ness_catch_roundtrip(1024, method);
        assert!(errs.iter().all(|e| e.is_finite() && *e < 1.6e-2), "{errs:?}");
    }
}

#[test]
fn ness_plain_text_io() {
    let mut state = 42u64;
    let mut g = NessHermMat::new(NessGrid::new(8, 0.1), 2);
    for data in [&mut g.ret.time, &mut g.ret.freq, &mut g.les.time, &mut g.les.freq] {
        for z in data.iter_mut() {
            *z = rand_c64(&mut state);
        }
    }
    let text = ness_to_string(&g);
    let h = ness_from_str(&text).unwrap();
    assert_eq!(h.ret.time, g.ret.time);
    assert_eq!(h.ret.freq, g.ret.freq);
    assert_eq!(h.les.time, g.les.time);
    assert_eq!(h.les.freq, g.les.freq);
    assert!((h.grid.h - g.grid.h).abs() < 1e-15);
    assert!((h.grid.dw - g.grid.dw).abs() < 1e-12);

    let mut a = NessArray::new(4, 2);
    for z in a.time.iter_mut() {
        *z = rand_c64(&mut state);
    }
    for z in a.freq.iter_mut() {
        *z = rand_c64(&mut state);
    }
    let text = ness_array_to_string(&a);
    let b = ness_array_from_str(&text).unwrap();
    assert_eq!(b.time, a.time);
    assert_eq!(b.freq, a.freq);

    // corrupted files
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let bad = |ls: Vec<String>| -> String { ls.join("\n") };
    assert!(ness_array_from_str(&bad(lines[..lines.len() - 1].to_vec())).is_err());
    let mut dup = lines[..lines.len() - 1].to_vec();
    dup.push(lines[1].clone());
    assert!(ness_array_from_str(&bad(dup)).is_err());
    let mut hdr: Vec<String> = vec!["# FFTArray 4".into()];
    hdr.extend(lines[1..].iter().cloned());
    assert!(ness_array_from_str(&bad(hdr.clone())).is_err());
    let mut hdr2: Vec<String> = vec!["# HermMatrixNESS 4 2 0.1 1.0".into()];
    hdr2.extend(lines[1..].iter().cloned());
    assert!(ness_array_from_str(&bad(hdr2)).is_err());
    let mut tyme = lines.clone();
    tyme[1] = tyme[1].replacen("time", "tyme", 1);
    assert!(ness_array_from_str(&bad(tyme)).is_err());
    let mut oob = lines.clone();
    oob[1] = "time data 9 1 1 0.0 0.0".into();
    assert!(ness_array_from_str(&bad(oob)).is_err());
    let mut nan = lines.clone();
    nan[1] = "time data 1 1 1 NaN 0.0".into();
    assert!(ness_array_from_str(&bad(nan)).is_err());
    let mut cols = lines.clone();
    cols[1] = "time data 1 1 1 0.0".into();
    assert!(ness_array_from_str(&bad(cols)).is_err());
    // ness files
    let text = ness_to_string(&g);
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let mut adv = lines.clone();
    adv[1] = adv[1].replacen("ret", "adv", 1);
    assert!(ness_from_str(&bad(adv)).is_err());
    let mut nft: Vec<String> = vec!["# HermMatrixNESS 7 2 0.1 1.0".into()];
    nft.extend(lines[1..].iter().cloned());
    assert!(ness_from_str(&bad(nft)).is_err());
}

#[test]
fn ness_release_catch_round_trip_16384() {
    if !is_release() {
        return;
    }
    for &method in &[FftIntegralMethod::Trapez, FftIntegralMethod::Cubic] {
        let errs = ness_catch_roundtrip(16384, method);
        assert!(errs.iter().all(|e| *e < 1e-3), "{errs:?}");
    }
}
