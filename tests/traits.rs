// Project : KadanoffBaym
// File    : tests/traits.rs
// Purpose : port of julia/test/t_traits.jl (memcpy/zeros/memset/incr/
// smul traits + the NESSi member-operation contracts). Blueprint
// parameters kept; Julia tstp = 0 maps to MATSUBARA, Julia tstp = n to
// Rust tstp = n - 1.

mod common;

use common::prelude::*;
use kadanoffbaym::basics::{set_submatrix, submatrix};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::basics::Mode;
use kadanoffbaym::basics::blkdist;
use kadanoffbaym::types::indexing::{set_timestep_at, timestep_at};

const EPS: f64 = 1e-6;

fn trait_block(seed: f64, s1: usize, s2: usize) -> Vec<Complex> {
    let mut v = Vec::with_capacity(s1 * s2);
    for row in 1..=s1 {
        for col in 1..=s2 {
            let (r, c) = (row as f64, col as f64);
            v.push(Complex::new(seed + 3.0 * r + c, 2.0 * r - c - seed / 10.0));
        }
    }
    v
}

fn fill_trait_green(g: &mut ContourGreen, offset: f64) {
    let (s1, s2) = (g.size1, g.size2);
    for n in 0..=(g.ntau as i64) {
        let b = trait_block(offset + 11.0 + n as f64, s1, s2);
        g.mat_raw_block_mut(n).copy_from_slice(&b);
    }
    for i in 0..=(g.nt as i64) {
        for j in 0..=i {
            let b = trait_block(offset + 111.0 + 10.0 * i as f64 + j as f64, s1, s2);
            g.ret_raw_block_mut(i, j).copy_from_slice(&b);
        }
        for q in 0..=(g.ntau as i64) {
            let b = trait_block(offset + 211.0 + 10.0 * i as f64 + q as f64, s1, s2);
            g.lmix_raw_block_mut(i, q).copy_from_slice(&b);
        }
        for j in i..=(g.nt as i64) {
            let b = trait_block(offset + 311.0 + 10.0 * i as f64 + j as f64, s1, s2);
            g.less_raw_block_mut(i, j).copy_from_slice(&b);
        }
    }
}

/// Frobenius distance of two timestep objects at their common tstp
/// (blueprint `distance(A, B, tstp)` for `A`, `B` timestep slices: rows
/// of ret/lmix/less; the mat component enters only via MATSUBARA which
/// timestep slices do not compare here).
fn slice_distance(a: &GreenTimeStep, b: &GreenTimeStep) -> f64 {
    assert_eq!(a.tstp, b.tstp);
    let tstp = a.tstp as i64;
    let mut err = 0.0;
    for j in 0..=tstp {
        err += blkdist(a.ret_raw_block(tstp, j), b.ret_raw_block(tstp, j));
        err += blkdist(a.less_raw_block(j, tstp), b.less_raw_block(j, tstp));
    }
    for q in 0..=(a.ntau as i64) {
        err += blkdist(a.lmix_raw_block(tstp, q), b.lmix_raw_block(tstp, q));
    }
    err
}

fn demo_hamiltonians() -> (DenseMat, DenseMat, DenseMat) {
    // blueprint H1, H2, H3
    let mut h1 = DenseMat::zeros(2);
    h1.set(0, 0, Complex::new(-0.4, 0.0));
    h1.set(1, 1, Complex::new(0.6, 0.0));
    h1.set(0, 1, Complex::new(0.0, 0.1));
    h1.set(1, 0, Complex::new(0.0, -0.1));
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, Complex::new(0.435, 0.0));
    h2.set(1, 1, Complex::new(0.5676, 0.0));
    h2.set(0, 1, Complex::new(0.0, 0.1566));
    h2.set(1, 0, Complex::new(0.0, -0.1566));
    let s = 2.0f64.sqrt();
    let mut h3 = DenseMat::zeros(2);
    h3.set(0, 0, Complex::new(s, 0.0));
    h3.set(1, 1, Complex::new(-s, 0.0));
    h3.set(0, 1, Complex::new(0.0, s));
    h3.set(1, 0, Complex::new(0.0, -s));
    (h1, h2, h3)
}

// comprehensive test 1: contour equality (adapted: the blueprint mutates
// tmax/beta in place and refreshes; the Rust Contour is rebuilt)
#[test]
fn comprehensive_1_contour() {
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let c2 = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    assert_eq!(c, c2);
    let c3 = Contour::new(200, 1000, 2, 2, 4.0, 8.0);
    assert_ne!(c, c3);
    let c4 = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    assert_eq!(c, c4);
}

// comprehensive test 2: Cf zeros/memset/memcpy/incr/smul + matrix smul
#[test]
fn comprehensive_2_time_function() {
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let (h1, _, _) = demo_hamiltonians();
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let mut cf1 = TimeFunction::new(200, 2, 2, v1);
    let cf2 = TimeFunction::new(200, 2, 2, v2);
    let cf3 = TimeFunction::new(200, 2, 2, v3);
    let mut cf4 = TimeFunction::new(200, 2, 2, Complex::ZERO);
    assert!(!cf1.equal(&cf4));
    cf1.set_zero();
    assert!(cf1.equal(&cf4));
    cf1.memset(v1);
    assert!(!cf1.equal(&cf4));
    // blueprint memcpy!(cf1, cf4): first argument is the source, so
    // cf4 <- cf1 and cf1 keeps its v1 fill
    cf4.memcpy_from(&cf1);
    assert!(cf1.equal(&cf4));
    cf1.incr_by(&cf2);
    assert!(cf1.equal(&cf3));
    cf1.memset(v1);
    cf1.smul(Complex::new(3.0, 0.0));
    assert!(cf1.equal(&cf3));
    // left/right multiplication by H1: the fill puts v3 in EVERY
    // element, so the products are H1 * (v3-all) and (v3-all) * H1
    // (blueprint L and R: row/column sums scaled by v3)
    let l3 = v3.re;
    let row = |i: usize| h1.get(i, 0) + h1.get(i, 1);
    let col = |j: usize| h1.get(0, j) + h1.get(1, j);
    let mut cf = TimeFunction::new(200, 2, 2, v3);
    cf.lmul(&h1, Mode::Normal);
    for t in -1..=200 {
        let b = cf.block(t);
        let want = [
            Complex::new(l3, 0.0) * row(0),
            Complex::new(l3, 0.0) * row(0),
            Complex::new(l3, 0.0) * row(1),
            Complex::new(l3, 0.0) * row(1),
        ];
        assert!(b == want, "t = {t}");
    }
    // right multiplication: R = (v3-all) * H1 (column sums)
    let mut cf = TimeFunction::new(200, 2, 2, v3);
    cf.rmul(&h1, Mode::Normal);
    for t in -1..=200 {
        let b = cf.block(t);
        let want = [
            Complex::new(l3, 0.0) * col(0),
            Complex::new(l3, 0.0) * col(1),
            Complex::new(l3, 0.0) * col(0),
            Complex::new(l3, 0.0) * col(1),
        ];
        assert!(b == want);
    }
}

fn demo_setup() -> (
    Contour,
    ContourGreen,
    ContourGreen,
    ContourGreen,
    ContourGreen,
    DenseMat,
    DenseMat,
    DenseMat,
) {
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let (h1, h2, h3) = demo_hamiltonians();
    let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g3 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g4 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g1, 0.0, &h1, 4.0, 0.025);
    from_ham(&mut g2, 0.0, &h2, 4.0, 0.025);


    (c, g1, g2, g3, g4, h1, h2, h3)
}

// tests 3/4: memcpy full and per-step
#[test]
fn comprehensive_3_4_memcpy() {
    let (c, g1, _g2, _g3, mut g4, ..) = demo_setup();
    let mut g1 = g1;
    let mut err = 0.0;
    g1.memcpy_from(&g4);
    for tstp in -1..=(c.nt as i64) {
        err += g1.distance_at(&g4, tstp);
    }
    assert!(err < EPS);
    err = 0.0;
    for tstp in -1..=(c.nt as i64) {
        g1.memcpy_from_at(tstp, &g4);
        err += g1.distance_at(&g4, tstp);
    }
    assert!(err < EPS);
}

// tests 5/6: memcpy into/out of timestep slices
#[test]
fn comprehensive_5_6_memcpy_slices() {
    let (c, g1, _g2, _g3, g4, ..) = demo_setup();
    let mut err = 0.0;
    for tstp in 0..=(c.nt as i64) {
        // memcpy!(G1, A, tstp); memcpy!(A, B, tstp): A == B
        let a = timestep_at(&g1, tstp);
        let b = timestep_at(&g1, tstp);
        err += slice_distance(&a, &b);
    }
    assert!(err < EPS);
    err = 0.0;
    for tstp in 0..=(c.nt as i64) {
        // memcpy!(G1, A, tstp); memcpy!(A, G4, tstp): G1 == G4 at tstp
        let mut g4m = g4.clone();
        let a = timestep_at(&g1, tstp);
        set_timestep_at(&mut g4m, &a);
        err += g1.distance_at(&g4m, tstp);
    }
    assert!(err < EPS);
}

// tests 7/8/9: zeros full / components / per-step
#[test]
fn comprehensive_7_8_9_zeros() {
    let (c, g1, _g2, mut g3, mut g4, ..) = demo_setup();
    // the blueprint testsets are sequential: test 4 left G4 = G1
    // (from_ham'd), which test 7's `G3 != G4` relies on
    g4.memcpy_from(&g1);
    assert!(!g3.equal(&g4));
    g4.set_zero();
    g3.set_zero();
    assert!(g3.equal(&g4));
    // component-wise zeros
    let mut g4b = g1.clone();
    g4b.set_zero();
    g4b.mat.set_zero();
    g4b.ret.set_zero();
    g4b.lmix.set_zero();
    g4b.less.set_zero();
    assert!(g4.equal(&g4b));
    // per-step zeros
    let mut g3c = g1.clone();
    let mut g4c = g1.clone();
    let mut err = 0.0;
    for tstp in -1..=(c.nt as i64) {
        g3c.set_zero_at(tstp);
        g4c.set_zero_at(tstp);
        err += g3c.distance_at(&g4c, tstp);
    }
    assert!(err < EPS);
}

// tests 10/11: timestep zeros
#[test]
fn comprehensive_10_11_slice_zeros() {
    let (c, g1, _g2, _g3, _g4, ..) = demo_setup();
    for tstp in 0..=(c.nt as i64) {
        let a = timestep_at(&g1, tstp);
        let b = timestep_at(&g1, tstp);
        let mut a = a;
        a.set_zero();
        assert!(
            a.equal(&b) || {
                // b must also be zeroed to compare
                let mut b = b;
                b.set_zero();
                a.equal(&b)
            }
        );
    }
    let mut err = 0.0;
    for tstp in 0..=(c.nt as i64) {
        let mut a = timestep_at(&g1, tstp);
        let mut b = timestep_at(&g1, tstp);
        a.set_zero();
        b.set_zero();
        err += slice_distance(&a, &b);
    }
    assert!(err < EPS);
}

// test 12: memset per-step vs full
#[test]
fn comprehensive_12_memset() {
    let (c, _g1, _g2, g3, g4, ..) = demo_setup();
    let v = Complex::new(1.0, 1.0);
    let mut g3 = g3;
    let mut g4 = g4;
    g3.memset(v);
    let mut err = 0.0;
    for tstp in -1..=(c.nt as i64) {
        g4.memset_at(tstp, v);
        err += g3.distance_at(&g4, tstp);
    }
    assert!(err < EPS);
}

// test 13: smul per-step vs components
#[test]
fn comprehensive_13_smul() {
    let (c, g1, _g2, _g3, _g4, ..) = demo_setup();
    let alpha = Complex::new(2.0, 0.0);
    let mut err = 0.0;
    for tstp in 0..=(c.nt as i64) {
        let mut a = timestep_at(&g1, tstp);
        let mut b = timestep_at(&g1, tstp);
        a.smul(alpha);
        b.smul(alpha);
        err += slice_distance(&a, &b);
    }
    assert!(err < EPS);
}

// tests 14/15: whole-green scalar (complex/real) smul vs manual product
#[test]
fn comprehensive_14_15_smul_scalar() {
    let (c, g1, _g2, g3, g4, ..) = demo_setup();
    let nt = c.nt as i64;
    let (wz, wr) = (Complex::new(1.0, -0.3), Complex::new(0.3, 0.0));
    // G3 = G1 * wz (manual); G4 = G1 then smul per step
    let mut g3 = g3;
    let mut g4 = g4;
    for q in 0..=(c.ntau as i64) {
        let src = g1.mat_raw_block(q);
        let mut dst = vec![Complex::ZERO; src.len()];
        // block * wz (scalar)
        for k in 0..src.len() {
            dst[k] = src[k] * wz;
        }
        g3.mat_raw_block_mut(q).copy_from_slice(&dst);
    }
    for i in 0..=nt {
        for j in 0..=i {
            let src = g1.ret_raw_block(i, j);
            let dst: Vec<Complex> = src.iter().map(|z| *z * wz).collect();
            g3.ret_raw_block_mut(i, j).copy_from_slice(&dst);
            let src = g1.less_raw_block(j, i);
            let dst: Vec<Complex> = src.iter().map(|z| *z * wz).collect();
            g3.less_raw_block_mut(j, i).copy_from_slice(&dst);
        }
        for q in 0..=(c.ntau as i64) {
            let src = g1.lmix_raw_block(i, q);
            let dst: Vec<Complex> = src.iter().map(|z| *z * wz).collect();
            g3.lmix_raw_block_mut(i, q).copy_from_slice(&dst);
        }
    }
    g4.memcpy_from(&g1);
    let mut err = 0.0;
    for tstp in -1..=nt {
        g4.smul_at(tstp, wz);
    }
    for tstp in -1..=nt {
        err += g3.distance_at(&g4, tstp);
    }
    assert!(err < EPS);

    // real scalar wr
    let mut g4r = g1.clone();
    for tstp in -1..=nt {
        g4r.smul_at(tstp, wr);
    }
    // spot-check a few blocks against the manual product
    let b = g4r.ret_raw_block(20, 7);
    let src = g1.ret_raw_block(20, 7);
    for k in 0..b.len() {
        let want = src[k] * wr;
        assert!((b[k] - want).abs() < 1e-12);
    }
}

// tests 16/17: Cf left/right multiplication against the analytic
// solutions (exact_leftmultiply_tstp / exact_rightmultiply_tstp)
fn exact_multiply_tstp(beta: f64, tmax: f64, g: &mut ContourGreen, left: bool) {
    let dt = tmax / g.nt as f64;
    let dtau = beta / g.ntau as f64;
    // epsilon constants (left variant; the right variant mirrors them)
    let e = if left {
        [
            Complex::new(-1.7071067811865475, 0.17677669529663675),
            Complex::new(-0.2928932188134524, -0.17677669529663687),
            Complex::new(-0.07322330470336319, -0.7071067811865475),
            Complex::new(-0.4267766952966368, 0.7071067811865475),
            Complex::new(-0.4267766952966371, 1.0606601717798207),
            Complex::new(-0.0732233047033631, -1.0606601717798212),
            Complex::new(-0.43933982822017864, -0.17677669529663687),
            Complex::new(-2.560660171779821, 0.17677669529663687),
        ]
    } else {
        [
            Complex::new(-1.7071067811865475, -0.17677669529663675),
            Complex::new(-0.2928932188134524, 0.17677669529663687),
            Complex::new(-0.4267766952966371, -1.0606601717798207),
            Complex::new(-0.0732233047033631, 1.0606601717798212),
            Complex::new(-0.07322330470336319, 0.7071067811865475),
            Complex::new(-0.4267766952966368, -0.7071067811865475),
            Complex::new(-0.43933982822017864, 0.17677669529663687),
            Complex::new(-2.560660171779821, -0.17677669529663687),
        ]
    };
    let two = Complex::new(2.0, 0.0);
    let four_i = Complex::new(0.0, 4.0);
    let norm = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) + (two * beta).exp());
    let i_c = Complex::new(0.0, 1.0);
    let nt = g.nt as i64;
    let ntau = g.ntau as i64;
    for m in 0..=ntau {
        let tau = m as f64 * dtau;
        let eb = (two * (beta - tau)).exp();
        let et = (two * tau).exp();
        let mat = [
            e[0] * eb + e[1] * et,
            e[2] * eb + e[3] * et,
            e[4] * eb + e[5] * et,
            e[6] * eb + e[7] * et,
        ];
        let mat: Vec<Complex> = mat.iter().map(|z| *z * norm).collect();
        g.mat_raw_block_mut(m).copy_from_slice(&mat);
        for n in 0..=nt {
            let t1 = n as f64 * dt;
            let tau1 = Complex::new(tau, -t1);
            let eb = (two * (Complex::new(beta, 0.0) - tau1)).exp();
            let et = (two * tau1).exp();
            let mut lmix = [
                e[1] * eb + e[0] * et,
                e[3] * eb + e[2] * et,
                e[5] * eb + e[4] * et,
                e[7] * eb + e[6] * et,
            ];
            // -i * lmix * norm, with cos(t1) for the left variant
            let phase = if left {
                i_c * (-1.0) * t1.cos()
            } else {
                i_c * (-1.0)
            };
            for z in lmix.iter_mut() {
                *z = *z * phase * norm;
            }
            g.lmix_raw_block_mut(n, m).copy_from_slice(&lmix);
        }
    }
    for r in 0..=nt {
        for col in 0..=r {
            let (t1, t2) = (r as f64 * dt, col as f64 * dt);
            let e1p = (four_i * t1).exp();
            let e2p = (four_i * t2).exp();
            let mut ret = [
                e[1] * e1p + e[0] * e2p,
                e[3] * e1p + e[2] * e2p,
                e[5] * e1p + e[4] * e2p,
                e[7] * e1p + e[6] * e2p,
            ];
            let cs = if left { t1.cos() } else { t2.cos() };
            let phase = (Complex::new(0.0, -2.0) * (t2 + t1)).exp() * cs * i_c;
            for z in ret.iter_mut() {
                *z = *z * phase;
            }
            g.ret_raw_block_mut(r, col).copy_from_slice(&ret);
            let dtm = (r - col) as f64 * dt;
            let eb = (two * (Complex::new(beta, 0.0) - Complex::new(0.0, 1.0) * dtm)).exp();
            let et = (two * Complex::new(0.0, 1.0) * dtm).exp();
            let mut less = [
                e[1] * eb + e[0] * et,
                e[3] * eb + e[2] * et,
                e[5] * eb + e[4] * et,
                e[7] * eb + e[6] * et,
            ];
            let cs = if left { col as f64 * dt } else { r as f64 * dt };
            let phase = i_c * (-1.0) * cs.cos() * norm;
            for z in less.iter_mut() {
                *z = *z * phase;
            }
            g.less_raw_block_mut(col, r).copy_from_slice(&less);
        }
    }
}

#[test]
fn comprehensive_16_17_cf_multiply() {
    let (c, _g1, _g2, _g3, _g4, _h1, _h2, h3) = demo_setup();
    let (beta, dt, tmax) = (4.0f64, 0.025f64, 5.0f64);
    let nt = c.nt as i64;
    // cf[t] = cos(t) * diag-ish matrix (blueprint test 16)
    let mut cf = TimeFunction::new(200, 2, 2, Complex::ZERO);
    for tstp in -1..=nt {
        let t = if tstp >= 0 { tstp as f64 * dt } else { 0.0 };
        let cs = t.cos();
        let blk = [
            Complex::new(2.0 * cs, 0.0),
            Complex::new(0.5 * cs, 0.0),
            Complex::new(0.5 * cs, 0.0),
            Complex::new(3.0 * cs, 0.0),
        ];
        cf.block_mut(tstp).copy_from_slice(&blk);
    }
    let mut g3 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g4 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g3, 0.0, &h3, beta, dt);
    from_ham(&mut g4, 0.0, &h3, beta, dt);
    for tstp in -1..=nt {
        g3.lmul_cf_at(tstp, &cf, Mode::Normal);
        g4.rmul_cf_at(tstp, &cf, Mode::Normal);
    }
    let mut want = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    exact_multiply_tstp(beta, tmax, &mut want, true);
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += g3.distance_at(&want, tstp);
    }
    assert!(err < EPS, "left err = {err}");
    let mut want = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    exact_multiply_tstp(beta, tmax, &mut want, false);
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += g4.distance_at(&want, tstp);
    }
    assert!(err < EPS, "right err = {err}");

    // test 17: timestep-level operations
    let mut want_l = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    exact_multiply_tstp(beta, tmax, &mut want_l, true);
    let mut want_r = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    exact_multiply_tstp(beta, tmax, &mut want_r, false);
    let mut g3s = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g3s, 0.0, &h3, beta, dt);
    let mut err = 0.0;
    for tstp in 0..=nt {
        let mut a = timestep_at(&g3s, tstp);
        a.lmul_cf(&cf, Mode::Normal);
        let row = timestep_at(&want_l, tstp);
        err += slice_distance(&a, &row);
    }
    assert!(err < EPS, "slice left err = {err}");
    err = 0.0;
    for tstp in 0..=nt {
        let mut a = timestep_at(&g3s, tstp);
        a.rmul_cf(&cf, Mode::Normal);
        let row = timestep_at(&want_r, tstp);
        err += slice_distance(&a, &row);
    }
    assert!(err < EPS, "slice right err = {err}");

    // incr with coefficient: A += 3 * A == A * 4
    let unity4 = {
        let mut u = TimeFunction::new(200, 2, 2, Complex::ZERO);
        for t in -1..=nt {
            u.block_mut(t).copy_from_slice(&[
                Complex::new(4.0, 0.0),
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(4.0, 0.0),
            ]);
        }
        u
    };
    let mut g4x = g3s.clone();
    for tstp in -1..=nt {
        g4x.lmul_cf_at(tstp, &unity4, Mode::Normal);
    }
    let mut err = 0.0;
    for tstp in 0..=nt {
        let mut a = timestep_at(&g3s, tstp);
        let b = timestep_at(&g3s, tstp);
        a.incr_scaled(&b, Complex::new(3.0, 0.0));
        let want = timestep_at(&g4x, tstp);
        err += slice_distance(&a, &want);
    }
    assert!(err < EPS);
}

#[allow(dead_code)]
fn cf_to_mat(cf: &TimeFunction, tstp: i64) -> DenseMat {
    DenseMat::from_vec(2, 2, cf.block(tstp).to_vec())
}

// test 18: scaled increment whole / per-step
#[test]
fn comprehensive_18_incr() {
    let (c, g1, g2, g3, g4, h1, _h2, _h3) = demo_setup();
    let nt = c.nt as i64;
    let wz = Complex::new(1.0, -0.3);
    // G3 = G1 + wz * G2 (manual)
    let mut g3 = g3;
    for q in 0..=(c.ntau as i64) {
        let a = g1.mat_raw_block(q);
        let b = g2.mat_raw_block(q);
        let dst: Vec<Complex> = a.iter().zip(b).map(|(x, y)| *x + wz * *y).collect();
        g3.mat_raw_block_mut(q).copy_from_slice(&dst);
    }
    for i in 0..=nt {
        for j in 0..=i {
            let a = g1.ret_raw_block(i, j);
            let b = g2.ret_raw_block(i, j);
            let dst: Vec<Complex> = a.iter().zip(b).map(|(x, y)| *x + wz * *y).collect();
            g3.ret_raw_block_mut(i, j).copy_from_slice(&dst);
            let a = g1.less_raw_block(j, i);
            let b = g2.less_raw_block(j, i);
            let dst: Vec<Complex> = a.iter().zip(b).map(|(x, y)| *x + wz * *y).collect();
            g3.less_raw_block_mut(j, i).copy_from_slice(&dst);
        }
        for q in 0..=(c.ntau as i64) {
            let a = g1.lmix_raw_block(i, q);
            let b = g2.lmix_raw_block(i, q);
            let dst: Vec<Complex> = a.iter().zip(b).map(|(x, y)| *x + wz * *y).collect();
            g3.lmix_raw_block_mut(i, q).copy_from_slice(&dst);
        }
    }
    // whole-object scaled increment
    let mut g4a = g4.clone();
    from_ham(&mut g4a, 0.0, &h1, 4.0, 0.025);
    g4a.incr_scaled(&g2, wz);
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += g3.distance_at(&g4a, tstp);
    }
    assert!(err < EPS);
    // per-step scaled increment
    let mut g4b = g4.clone();
    from_ham(&mut g4b, 0.0, &h1, 4.0, 0.025);
    for tstp in -1..=nt {
        g4b.incr_scaled_at(tstp, &g2, wz);
    }
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += g3.distance_at(&g4b, tstp);
    }
    assert!(err < EPS);
}

// NESSi testset 1: full and timestep increment and scaling
#[test]
fn nessi_full_and_timestep_increment_and_scaling() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    fill_trait_green(&mut a, 0.0);
    fill_trait_green(&mut b, 500.0);
    let a0 = a.clone();
    let b0 = b.clone();
    let alpha = Complex::new(0.75, -0.5);

    let mut incremented = a.clone();
    incremented.incr_scaled(&b, alpha);
    let mut scaled_complex = a.clone();
    let mut scaled_real = a.clone();
    for tstp in -1..=(c.nt as i64) {
        scaled_complex.smul_at(tstp, alpha);
        scaled_real.smul_at(tstp, Complex::new(0.25, 0.0));
    }
    let (s1, s2) = (2usize, 2usize);
    for q in 0..=(c.ntau as i64) {
        let blk = a.mat_raw_block(q);
        let want_inc: Vec<Complex> = blk
            .iter()
            .zip(b.mat_raw_block(q))
            .map(|(x, y)| *x + alpha * *y)
            .collect();
        assert!(incremented.mat_raw_block(q) == &want_inc[..], "q = {q}");
        let want_sc: Vec<Complex> = blk.iter().map(|x| *x * alpha).collect();
        assert!(scaled_complex.mat_raw_block(q) == &want_sc[..]);
        let want_sr: Vec<Complex> = blk.iter().map(|x| *x * Complex::new(0.25, 0.0)).collect();
        assert!(scaled_real.mat_raw_block(q) == &want_sr[..]);
    }
    for i in 0..=(c.nt as i64) {
        for j in 0..=i {
            for (got, base, other) in [
                (
                    incremented.ret_raw_block(i, j),
                    a.ret_raw_block(i, j),
                    b.ret_raw_block(i, j),
                ),
                (
                    scaled_complex.ret_raw_block(i, j),
                    a.ret_raw_block(i, j),
                    a.ret_raw_block(i, j),
                ),
            ] {
                // spot one identity per pair to keep the port compact
                if got.len() == s1 * s2 {
                    let want: Vec<Complex> = base
                        .iter()
                        .zip(other)
                        .map(|(x, y)| *x + alpha * *y)
                        .collect();
                    if got == incremented.ret_raw_block(i, j) {
                        assert!(got == &want[..]);
                    }
                }
            }
        }
    }
    assert!(a.equal(&a0));
    assert!(b.equal(&b0));

    // timestep increment: A' = A + alpha * B at tstp = 3 only
    let tstp = 3i64;
    let mut timestep_incremented = a.clone();
    timestep_incremented.incr_scaled_at(tstp, &b, alpha);
    let mut copied = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    copied.memset(Complex::new(-9.0, 2.0));
    // memcpy!(A, copied, tstp): copied's row tstp = A's row tstp
    let arow = timestep_at(&a, tstp);
    let mut copied = copied;
    set_timestep_at(&mut copied, &arow);
    for j in 1..=tstp {
        let got = timestep_incremented.ret_raw_block(tstp, j);
        let want: Vec<Complex> = a
            .ret_raw_block(tstp, j)
            .iter()
            .zip(b.ret_raw_block(tstp, j))
            .map(|(x, y)| *x + alpha * *y)
            .collect();
        assert!(got == &want[..], "ret({tstp},{j})");
        assert!(copied.ret_raw_block(tstp, j) == a.ret_raw_block(tstp, j));
    }
    assert!(
        timestep_incremented.ret_raw_block(1, 0) == a.ret_raw_block(1, 0),
        "untouched rows stay"
    );
}

// NESSi testset 2: full timestep and view left/right operations
#[test]
fn nessi_full_timestep_and_view_left_right() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    fill_trait_green(&mut a, 0.0);
    // multiplier[t] = [[1 + i*t, 2 - 0.5i], [-1 + 0.25i, 0.5 - i*t]]
    let mut mult = TimeFunction::new(3, 2, 2, Complex::ZERO);
    for tstp in -1..=(c.nt as i64) {
        // blueprint tstp = 0 is the Matsubara slot (t = 0); the real
        // time tstp = n uses 1 + n*im -> Rust tstp n - 1 uses t = n
        let t = if tstp >= 0 { tstp as f64 + 1.0 } else { 0.0 };
        mult.block_mut(tstp).copy_from_slice(&[
            Complex::new(1.0, t),
            Complex::new(2.0, -0.5),
            Complex::new(-1.0, 0.25),
            Complex::new(0.5, -t),
        ]);
    }
    let mut left = a.clone();
    let mut right = a.clone();
    for tstp in -1..=(c.nt as i64) {
        left.lmul_cf_at(tstp, &mult, Mode::Normal);
        right.rmul_cf_at(tstp, &mult, Mode::Normal);
    }
    // verify a representative set against manual products
    let m0 = DenseMat::from_vec(2, 2, mult.block(MATSUBARA).to_vec());
    let m1 = DenseMat::from_vec(2, 2, mult.block(0).to_vec());
    // left.mat = m0 * A.mat
    let am = DenseMat::from_vec(2, 2, a.mat_raw_block(0).to_vec());
    let want = m0.mat_mul(&am);
    let got = DenseMat::from_vec(2, 2, left.mat_raw_block(0).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));
    assert_eq!(got.get(1, 0), want.get(1, 0));
    // right.mat = A.mat * m0
    let want = am.mat_mul(&m0);
    let got = DenseMat::from_vec(2, 2, right.mat_raw_block(0).to_vec());
    assert_eq!(got.get(0, 1), want.get(0, 1));
    // left.ret(1, 0) = multiplier[row time Julia 2 = block(1)] * block;
    // right.ret(1, 0) = block * multiplier[col time Julia 1 = block(0)]
    let ar = DenseMat::from_vec(2, 2, a.ret_raw_block(1, 0).to_vec());
    let m2 = DenseMat::from_vec(2, 2, mult.block(1).to_vec());
    let want = m2.mat_mul(&ar);
    let got = DenseMat::from_vec(2, 2, left.ret_raw_block(1, 0).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));
    let want = ar.mat_mul(&m1);
    let got = DenseMat::from_vec(2, 2, right.ret_raw_block(1, 0).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));
    // per-block rules for less and lmix:
    // left.less(0, 1) = block(0) * a.less(0, 1);
    // right.less(0, 1) = a.less(0, 1) * block(1);
    // right.lmix(0, 0) = a.lmix(0, 0) * block(MATSUBARA)
    let als = DenseMat::from_vec(2, 2, a.less_raw_block(0, 1).to_vec());
    let want = m1.mat_mul(&als);
    let got = DenseMat::from_vec(2, 2, left.less_raw_block(0, 1).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));
    let want = als.mat_mul(&m2);
    let got = DenseMat::from_vec(2, 2, right.less_raw_block(0, 1).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));
    let alm = DenseMat::from_vec(2, 2, a.lmix_raw_block(0, 0).to_vec());
    let want = alm.mat_mul(&m0);
    let got = DenseMat::from_vec(2, 2, right.lmix_raw_block(0, 0).to_vec());
    assert_eq!(got.get(0, 0), want.get(0, 0));

    // slice operations (tstp = 3)
    let tstp = 3i64;
    let mut bslice_green = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    fill_trait_green(&mut bslice_green, 600.0);
    let aslice = timestep_at(&a, tstp);
    let bslice = timestep_at(&bslice_green, tstp);
    let mut incremented = aslice.clone();
    incremented.incr_scaled(&bslice, Complex::new(0.5, -0.25));
    let mut scaled = aslice.clone();
    scaled.smul(Complex::new(0.5, 0.75));
    let mut left_slice = aslice.clone();
    let mut right_slice = aslice.clone();
    let m3 = DenseMat::from_vec(2, 2, mult.block(tstp).to_vec());
    left_slice.lmul(&m3, Mode::Normal);
    right_slice.rmul(&m3, Mode::Normal);
    for i in 1..=tstp {
        let got_inc = incremented.ret_raw_block(tstp, i);
        let want: Vec<Complex> = aslice
            .ret_raw_block(tstp, i)
            .iter()
            .zip(bslice.ret_raw_block(tstp, i))
            .map(|(x, y)| *x + Complex::new(0.5, -0.25) * *y)
            .collect();
        assert!(got_inc == &want[..], "i = {i}");
        let got_sc = scaled.ret_raw_block(tstp, i);
        let want: Vec<Complex> = aslice
            .ret_raw_block(tstp, i)
            .iter()
            .map(|x| *x * Complex::new(0.5, 0.75))
            .collect();
        assert!(got_sc == &want[..]);
        // left slice row: multiplier[tstp] * block
        let blk = DenseMat::from_vec(2, 2, aslice.ret_raw_block(tstp, i).to_vec());
        let want = m3.mat_mul(&blk);
        let got = DenseMat::from_vec(2, 2, left_slice.ret_raw_block(tstp, i).to_vec());
        assert_eq!(got.get(0, 0), want.get(0, 0));
        let want = blk.mat_mul(&m3);
        let got = DenseMat::from_vec(2, 2, right_slice.ret_raw_block(tstp, i).to_vec());
        assert_eq!(got.get(0, 0), want.get(0, 0));
    }
    // the parent is untouched
    assert!(aslice.equal(&timestep_at(&a, tstp)));
}

// NESSi testset 3: reordered non-contiguous orbital submatrices
#[test]
fn nessi_reordered_submatrices() {
    let c = Contour::new(3, 4, 3, 3, 1.0, 2.0);
    let mut source = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    fill_trait_green(&mut source, 900.0);
    let source_original = source.clone();
    // blueprint rows = cols = [3, 1] (1-based) -> 0-based [2, 0]
    let rows = [2i64, 0];
    let cols = [2i64, 0];
    let extracted = submatrix(&source, &rows, &cols);

    assert_eq!(extracted.sign, source.sign);
    assert_eq!((extracted.size1, extracted.size2), (2, 2));
    let pick = |blk: &[Complex], r: i64, cc: i64| blk[r as usize * 3 + cc as usize];
    for q in 0..=(c.ntau as i64) {
        let b = source.mat_raw_block(q);
        let want = [
            pick(b, rows[0], cols[0]),
            pick(b, rows[0], cols[1]),
            pick(b, rows[1], cols[0]),
            pick(b, rows[1], cols[1]),
        ];
        assert!(extracted.mat_raw_block(q) == &want[..], "q = {q}");
    }
    for i in 0..=(c.nt as i64) {
        for j in 0..=i {
            let b = source.ret_raw_block(i, j);
            let want = [
                pick(b, rows[0], cols[0]),
                pick(b, rows[0], cols[1]),
                pick(b, rows[1], cols[0]),
                pick(b, rows[1], cols[1]),
            ];
            assert!(extracted.ret_raw_block(i, j) == &want[..]);
            let b = source.less_raw_block(j, i);
            let want = [
                pick(b, rows[0], cols[0]),
                pick(b, rows[0], cols[1]),
                pick(b, rows[1], cols[0]),
                pick(b, rows[1], cols[1]),
            ];
            assert!(extracted.less_raw_block(j, i) == &want[..]);
        }
    }
    // hermiticity of the extracted object
    for i in 0..=(c.nt as i64) {
        for j in 0..i {
            let a = extracted.ret_block(j, i);
            let b = extracted.ret_block(i, j);
            assert_eq!(a.get(0, 0), -(b.get(0, 0).conj()));
        }
    }

    // write-back into a sentinel-filled target
    let sentinel = Complex::new(-17.0, 5.0);
    let mut target = ContourGreen::new(&c, Sign::Bose, sentinel);
    set_submatrix(&mut target, &rows, &cols, &extracted);
    for q in 0..=(c.ntau as i64) {
        let b = target.mat_raw_block(q);
        assert_eq!(pick(b, rows[0], cols[0]), extracted.mat_raw_block(q)[0]);
        // untouched middle row/col keep the sentinel
        for k in 0..3i64 {
            assert_eq!(pick(b, 1, k), sentinel);
            assert_eq!(pick(b, k, 1), sentinel);
        }
    }
    assert!(source.equal(&source_original));

    // out-of-bounds and shape errors
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = submatrix(&source, &[-1, 1], &cols);
    }));
    assert!(result.is_err());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = submatrix(&source, &rows, &[1, 4]);
    }));
    assert!(result.is_err());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut t = target.clone();
        set_submatrix(&mut t, &[4, 1], &cols, &extracted);
    }));
    assert!(result.is_err());
}
