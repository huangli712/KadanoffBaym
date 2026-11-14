// Project : KadanoffBaym
// File    : tests/vie.rs
// Purpose : blueprint t_vie.jl: the VIE2 solvers vie2_mat, vie2_start,
//           vie2_timestep, vie2, and vie2_timestep_sin
//
// Model: the 2 x 2 block of the free 4 x 4 Hamiltonian [Ha V; V' Hb]
// solves G = G0 + G0 * Sigma * G, i.e. the VIE2 [1 + F] * G = Q with
// F = -G0 * Sigma, Fcc = -Sigma * G0, and Q = G0.
//
// The release-only allocation testset of the blueprint is
// Julia-specific and skipped (known deviation).

mod common;

use common::{check_order_gate, is_release};
use kadanoffbaym::c64::C64;
use kadanoffbaym::convolution::mat::conv_mat;
use kadanoffbaym::convolution::timestep::convolution_ab;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::lu::mat_inv;
use kadanoffbaym::mat::DenseMat;
use kadanoffbaym::ops::Mode;
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, mat_inner, MatsubaraMethod};
use kadanoffbaym::solvers::vie2::vie2;
use kadanoffbaym::solvers::vie2_mat::vie2_mat;
use kadanoffbaym::solvers::vie2_sin::{vie2_timestep_sin, Vie2SinWorkspace};
use kadanoffbaym::solvers::vie2_start::vie2_start_impl;
use kadanoffbaym::solvers::vie2_timestep::{
    vie2_timestep, vie2_timestep_less, vie2_timestep_lmix, vie2_timestep_ret,
};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, MatComp, TimeFunction};
use kadanoffbaym::types::indexing::timestep_at;
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::weights::Integrator;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn ha_mat() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(-1.0, 0.0));
    m.set(0, 1, c(0.0, 0.3));
    m.set(1, 0, c(0.0, -0.3));
    m.set(1, 1, c(0.5, 0.0));
    m
}

fn hb_mat() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(1.0, 0.0));
    m.set(0, 1, c(0.2, 0.1));
    m.set(1, 0, c(0.2, -0.1));
    m.set(1, 1, c(1.5, 0.0));
    m
}

fn v_mat() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(0.4, 0.0));
    m.set(0, 1, c(0.0, 0.1));
    m.set(1, 0, c(-0.2, 0.0));
    m.set(1, 1, c(0.3, 0.0));
    m
}

fn tf_const(nt: i64, m: &DenseMat) -> TimeFunction {
    let d = m.nrows();
    let mut h = TimeFunction::new(nt as usize, d, d, C64::ZERO);
    for t in -1..=nt {
        for i in 0..d {
            for j in 0..d {
                h.set(t, i, j, m.get(i, j));
            }
        }
    }
    h
}

struct VtSetup {
    g: ContourGreen,
    f: ContourGreen,
    fcc: ContourGreen,
    q: ContourGreen,
    contour: Contour,
    beta: f64,
    h: f64,
    nt: i64,
    sign: Sign,
}

/// Exact solution G, kernel F, partner Fcc, and source Q; the kernels
/// are evaluated by the convolution of order `ks` (blueprint
/// `vt_setup`). `nt` is the number of real time intervals.
fn vt_setup(nt: i64, ks: i64, sign: Sign) -> VtSetup {
    let (ntau, beta, tmax) = (200i64, 1.0, 2.0);
    let mu = if sign == Sign::Fermi { 0.1 } else { -2.5 };
    let h = tmax / nt as f64;
    let c4 = Contour::new(nt, ntau, 4, 4, tmax, beta);
    let c2 = Contour::new(nt, ntau, 2, 2, tmax, beta);
    let (ha, hb, v) = (ha_mat(), hb_mat(), v_mat());
    let mut hfull = DenseMat::zeros(4);
    for i in 0..4 {
        for j in 0..4 {
            let x = if i < 2 && j < 2 {
                ha.get(i, j)
            } else if i < 2 {
                v.get(i, j - 2)
            } else if j < 2 {
                v.get(j, i - 2).conj()
            } else {
                hb.get(i - 2, j - 2)
            };
            hfull.set(i, j, x);
        }
    }
    let mut gf = ContourGreen::new(&c4, sign, C64::ZERO);
    from_ham(&mut gf, mu, &hfull, beta, h);
    let mut gx = ContourGreen::new(&c2, sign, C64::ZERO);
    // upper-left 2 x 2 block of each 4 x 4 block (strided extraction)
    for m in 0..=(ntau as usize) {
        for a in 0..2usize {
            for b in 0..2usize {
                gx.mat.block_mut(m)[a * 2 + b] = gf.mat.block(m)[a * 4 + b];
            }
        }
    }
    for i in 0..=nt {
        for m in 0..=(ntau as usize) {
            for a in 0..2usize {
                for b in 0..2usize {
                    gx.lmix.raw_block_at_mut(i, m as i64)[a * 2 + b] =
                        gf.lmix.raw_block_at(i, m as i64)[a * 4 + b];
                }
            }
        }
        for j in 0..=i {
            for a in 0..2usize {
                for b in 0..2usize {
                    gx.ret.raw_block_at_mut(i, j)[a * 2 + b] =
                        gf.ret.raw_block_at(i, j)[a * 4 + b];
                    gx.less.raw_block_at_mut(j, i)[a * 2 + b] =
                        gf.less.raw_block_at(j, i)[a * 4 + b];
                }
            }
        }
    }
    let mut gb = ContourGreen::new(&c2, sign, C64::ZERO);
    from_ham(&mut gb, mu, &hb, beta, h);
    let vadj = v.adjoint();
    let mut sigma = ContourGreen::new(&c2, sign, C64::ZERO);
    let put = |blk: &[C64], out: &mut [C64]| {
        let src = DenseMat::from_vec(2, 2, blk.to_vec());
        let s2 = v.mat_mul(&src).mat_mul(&vadj);
        out.copy_from_slice(s2.raw());
    };
    for m in 0..=(ntau as usize) {
        put(gb.mat.block(m), sigma.mat.block_mut(m));
    }
    for i in 0..=nt {
        for m in 0..=(ntau as usize) {
            put(
                gb.lmix.raw_block_at(i, m as i64),
                sigma.lmix.raw_block_at_mut(i, m as i64),
            );
        }
        for j in 0..=i {
            put(gb.ret.raw_block_at(i, j), sigma.ret.raw_block_at_mut(i, j));
            put(
                gb.less.raw_block_at(j, i),
                sigma.less.raw_block_at_mut(j, i),
            );
        }
    }
    let mut qg = ContourGreen::new(&c2, sign, C64::ZERO);
    from_ham(&mut qg, mu, &ha, beta, h);
    let ik = Integrator::new(ks);
    let mut f = ContourGreen::new(&c2, sign, C64::ZERO);
    let mut fcc = ContourGreen::new(&c2, sign, C64::ZERO);
    convolution_ab(&mut f, &qg, &sigma, &ik, beta, h);
    convolution_ab(&mut fcc, &sigma, &qg, &ik, beta, h);
    f.smul(c(-1.0, 0.0));
    fcc.smul(c(-1.0, 0.0));
    VtSetup {
        g: gx,
        f,
        fcc,
        q: qg,
        contour: c2,
        beta,
        h,
        nt,
        sign,
    }
}

/// Errors of the ret, tv, and les components per element; the
/// Matsubara component is taken from the exact solution (blueprint
/// `vt_errors`).
fn vt_errors(ks: i64, nt: i64, sign: Sign) -> (f64, f64, f64) {
    let s = vt_setup(nt, ks, sign);
    let mut g = ContourGreen::new(&s.contour, s.sign, C64::ZERO);
    g.mat.raw_mut().copy_from_slice(s.g.mat.raw());
    let ik = Integrator::new(ks);
    vie2_start_impl(&mut g, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h);
    for tstp in (ks + 1)..=s.nt {
        vie2_timestep(tstp, &mut g, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
    }
    let ntau = g.ntau as i64;
    let mut er = 0.0f64;
    let mut et = 0.0f64;
    let mut el = 0.0f64;
    for n in 0..=s.nt {
        er = er.max(g.distance_ret(&s.g, n) / (n + 1) as f64);
        et = et.max(g.distance_lmix(&s.g, n) / (ntau + 1) as f64);
        el = el.max(g.distance_less(&s.g, n) / (n + 1) as f64);
    }
    (er, et, el)
}

/// Reference implementation of vie2_timestep_sin, a literal port of
/// NESSi vie2_timestep_sin (blueprint `vt_sin_reference!`): full copies
/// of F, Fcc, and Q are transformed at the time steps MATSUBARA, .., n2
/// in every call.
#[allow(clippy::too_many_arguments)]
fn vt_sin_reference(
    tstp: i64,
    g: &mut ContourGreen,
    gsin: &mut TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    fsin: &TimeFunction,
    q: &ContourGreen,
    qsin: &TimeFunction,
    ik: &Integrator,
    beta: f64,
    h: f64,
) {
    let k = ik.k;
    let n2 = if tstp == MATSUBARA {
        MATSUBARA
    } else if tstp <= k {
        k + 1
    } else {
        tstp
    };
    let id = DenseMat::identity(2);
    let steps: Vec<i64> = {
        let mut v = Vec::new();
        if n2 >= MATSUBARA {
            v.push(MATSUBARA);
        }
        for i in 0..=n2.max(-1) {
            v.push(i);
        }
        v
    };
    let mut finv_blocks: Vec<(i64, DenseMat)> = Vec::new();
    for &i in &steps {
        let mut d = id.clone();
        for q in 0..4 {
            d.raw_mut()[q] += fsin.block(i)[q];
        }
        let dinv = mat_inv(&d);
        let qs = DenseMat::from_vec(2, 2, qsin.block(i).to_vec());
        let gs = dinv.mat_mul(&qs);
        for a in 0..2usize {
            for b in 0..2usize {
                gsin.set(i, a, b, gs.get(a, b));
            }
        }
        finv_blocks.push((i, dinv));
    }
    let mut tf = f.clone();
    let mut tfcc = fcc.clone();
    let mut tq = q.clone();
    for (i, dinv) in &finv_blocks {
        tf.lmul_at(*i, dinv, Mode::Normal);
        tfcc.rmul_at(*i, dinv, Mode::Normal);
        tq.lmul_at(*i, dinv, Mode::Normal);
    }
    let mut tf1 = tf.clone();
    for &(i, _) in &finv_blocks {
        let gsm = DenseMat::from_vec(2, 2, gsin.block(i).to_vec());
        tf1.rmul_at(i, &gsm, Mode::Normal);
        let step = timestep_at(&tf1, i.max(0));
        tq.incr_by_step(&step, i, c(-1.0, 0.0));
    }
    vie2_timestep(tstp, g, &tf, &tfcc, &tq, ik, beta, h, MatsubaraMethod::Fixpoint);
}

/// GW-like calling pattern (blueprint `vt_sin_pattern`); `nt = 12`.
fn vt_sin_pattern<F>(k: i64, sign: Sign, mut call: F) -> (ContourGreen, TimeFunction)
where
    F: FnMut(
        i64,
        &mut ContourGreen,
        &mut TimeFunction,
        &ContourGreen,
        &ContourGreen,
        &TimeFunction,
        &ContourGreen,
        &TimeFunction,
        &Integrator,
        f64,
        f64,
    ),
{
    let s = vt_setup(12, k, sign);
    let nt = s.nt;
    let (mut f, mut fcc, mut q) = (s.f.clone(), s.fcc.clone(), s.q.clone());
    let mut fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fs_blk = [
            c(0.3 + 0.01 * ijl, 0.0),
            c(0.0, 0.1),
            c(0.0, -0.1),
            c(-0.2, 0.0),
        ];
        let qs_blk = [
            c(0.5, 0.0),
            c(0.2 + 0.01 * ijl, 0.0),
            c(-0.3, 0.0),
            c(-0.4, 0.0),
        ];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, fs_blk[a * 2 + b]);
                qs.set(t, a, b, qs_blk[a * 2 + b]);
            }
        }
    }
    let mut g = ContourGreen::new(&s.contour, sign, C64::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let ik = Integrator::new(k);
    let id = DenseMat::identity(2);

    // repeated calls at tstp = 0 (the Matsubara axis)
    for it in 1..=2i64 {
        for x in [&mut f, &mut fcc, &mut q] {
            x.smul_at(MATSUBARA, c(1.0 + 0.01 * it as f64, 0.0));
        }
        call(MATSUBARA, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h);
    }
    // repeated start-up with modified inputs at tstp = 1, .., k + 1
    for it in 1..=2i64 {
        for t in 0..=k {
            for x in [&mut f, &mut fcc, &mut q] {
                x.smul_at(t, c(1.0 + 0.01 * it as f64, 0.0));
            }
        }
        for a in 0..2usize {
            for b in 0..2usize {
                let v = fs.block(k)[a * 2 + b]
                    + c(0.001 * it as f64, 0.0) * id.get(a, b);
                fs.set(k, a, b, v);
            }
        }
        for tstp in 0..=k {
            call(tstp, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h);
        }
    }
    // repeated calls at each later time step with modified inputs
    for tstp in (k + 1)..=nt {
        for it in 1..=2i64 {
            for x in [&mut f, &mut fcc, &mut q] {
                x.smul_at(tstp, c(1.0 + 0.01 * it as f64, 0.0));
            }
            for a in 0..2usize {
                for b in 0..2usize {
                    let v = fs.block(tstp)[a * 2 + b]
                        + c(0.001 * it as f64, 0.0) * id.get(a, b);
                    fs.set(tstp, a, b, v);
                }
            }
            call(tstp, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h);
        }
    }
    (g, gs)
}

fn vt_same(a: &ContourGreen, b: &ContourGreen) -> bool {
    a.mat.raw() == b.mat.raw()
        && a.ret.raw() == b.ret.raw()
        && a.lmix.raw() == b.lmix.raw()
        && a.less.raw() == b.less.raw()
}

// ---- testsets -------------------------------------------------------------

#[test]
fn vie2_matsubara_fixpoint_and_cg() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let s = vt_setup(10, 5, sign);
        let i5 = Integrator::new(5);
        let ntau = s.g.ntau as i64;
        let mut g1 = ContourGreen::new(&s.contour, sign, C64::ZERO);
        vie2_mat(&mut g1, &s.f, &s.fcc, &s.q, s.beta, &i5, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
        let mut g2 = ContourGreen::new(&s.contour, sign, C64::ZERO);
        vie2_mat(&mut g2, &s.f, &s.fcc, &s.q, s.beta, &i5, MatsubaraMethod::Cg, 6, 40, 1e-12);
        assert!(g1.distance_at(&s.g, MATSUBARA) / (ntau as f64) < 1e-12);
        assert!(g2.distance_at(&s.g, MATSUBARA) / (ntau as f64) < 1e-12);

        // residual of the discrete equation G + F * G - Q
        for g in [&g1, &g2] {
            let mut r = MatComp::new(ntau as usize, 2, 2, C64::ZERO);
            conv_mat(&mut r, &s.f.mat, &g.mat, &i5, s.beta, sign);
            let mut m = 0.0f64;
            for q in 0..r.raw().len() {
                m = m.max((r.raw()[q] + g.mat.raw()[q] - s.q.mat.raw()[q]).abs());
            }
            assert!(m < 1e-10, "residual {m}");
        }
        // only the Matsubara component is written
        assert!(g1.ret.raw().iter().all(|z| *z == C64::ZERO));
        assert!(g1.less.raw().iter().all(|z| *z == C64::ZERO));
    }
    // the inner product of CG is the sum of all the elements of X'Y,
    // not the trace
    let mut x = MatComp::new(10, 2, 2, C64::ZERO);
    let xb = [c(1.0, 0.0), c(0.0, 2.0), c(0.5, 0.0), c(-1.0, 0.0)];
    for m in 0..=10 {
        x.block_mut(m).copy_from_slice(&xb);
    }
    let i2 = Integrator::new(2);
    let wsum: f64 = i2.giw.omega[0] + i2.giw.omega[1] + i2.giw.omega[2] + 5.0
        + i2.giw.omega[2] + i2.giw.omega[1] + i2.giw.omega[0];
    // P = X' X, per block
    let prod = [
        xb[0].conj() * xb[0] + xb[2].conj() * xb[2],
        xb[0].conj() * xb[1] + xb[2].conj() * xb[3],
        xb[1].conj() * xb[0] + xb[3].conj() * xb[2],
        xb[1].conj() * xb[1] + xb[3].conj() * xb[3],
    ];
    let sum_reals: f64 = prod.iter().map(|z| z.re).sum();
    let tr = prod[0].re + prod[3].re;
    let got = mat_inner(&x, &x, &i2);
    assert!((got - wsum * sum_reals).abs() < 1e-12 * wsum.abs());
    assert!((got - wsum * tr).abs() > 1e-6);
}

#[test]
fn vie2_matsubara_dispatch_and_errors() {
    let s = vt_setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_mat(&mut g, &s.f, &s.fcc, &s.q, s.beta, &i2, MatsubaraMethod::Fourier, 6, 40, 1e-12);
    for z in g.mat.raw() {
        assert!(z.re.is_finite() && z.im.is_finite());
    }
}

#[test]
#[should_panic]
fn vie2_mat_rejects_negative_beta() {
    let s = vt_setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_mat(&mut g, &s.f, &s.fcc, &s.q, -1.0, &i2, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
}

#[test]
#[should_panic]
fn vie2_mat_rejects_order_six() {
    let s = vt_setup(6, 2, Sign::Fermi);
    let i6 = Integrator::new(6);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_mat(&mut g, &s.f, &s.fcc, &s.q, s.beta, &i6, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
}

#[test]
#[should_panic]
fn vie2_mat_rejects_statistics_mismatch() {
    let s = vt_setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Bose, C64::ZERO);
    vie2_mat(&mut g, &s.f, &s.fcc, &s.q, s.beta, &i2, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
}

#[test]
#[should_panic]
fn vie2_mat_rejects_grid_mismatch() {
    let s = vt_setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let c7 = Contour::new(7, 11, 2, 2, 1.0, 1.0);
    let mut g = ContourGreen::new(&c7, Sign::Fermi, C64::ZERO);
    vie2_mat(&mut g, &s.f, &s.fcc, &s.q, s.beta, &i2, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
}

#[test]
fn vie2_realtime_routing_and_drivers() {
    for &k in &[1i64, 3, 5] {
        for &sign in &[Sign::Fermi, Sign::Bose] {
            let s = vt_setup(12, k, sign);
            let ik = Integrator::new(k);
            // full driver
            let mut g1 = ContourGreen::new(&s.contour, sign, C64::ZERO);
            vie2(&mut g1, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
            // step by step, including the reduced orders at the start
            let mut g2 = ContourGreen::new(&s.contour, sign, C64::ZERO);
            for tstp in MATSUBARA..=s.nt {
                vie2_timestep(tstp, &mut g2, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
                if tstp == 0 {
                    assert_eq!(g2.ret.raw_block_at(0, 0), s.q.ret.raw_block_at(0, 0));
                } else if tstp >= 1 && tstp <= k {
                    let mut g3 = g2.clone();
                    vie2_start_impl(&mut g3, &s.f, &s.fcc, &s.q, &Integrator::new(tstp), s.beta, s.h);
                    for n in MATSUBARA..=tstp {
                        assert_eq!(g2.distance_at(&g3, n), 0.0, "row {n}");
                    }
                }
            }
            for n in MATSUBARA..=s.nt {
                assert_eq!(g1.distance_at(&g2, n), 0.0, "driver row {n}");
            }
            // accuracy, the sum of the block norms at each time step
            let mut dmax = 0.0f64;
            for n in MATSUBARA..=s.nt {
                dmax = dmax.max(g1.distance_at(&s.g, n));
            }
            let tol = if k == 1 { 1e-1 } else { 2e-2 };
            assert!(dmax < tol, "k = {k}: {dmax}");
            // both F and Fcc are used: exchanging them changes the result
            let mut g4 = ContourGreen::new(&s.contour, sign, C64::ZERO);
            vie2(&mut g4, &s.fcc, &s.f, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
            let mut dsw = 0.0f64;
            for n in 0..=s.nt {
                dsw = dsw.max(g1.distance_at(&g4, n));
            }
            assert!(dsw > 1e-6, "swap must change the result: {dsw}");
        }
    }
}

#[test]
fn vie2_realtime_sentinels() {
    let k = 3i64;
    let s = vt_setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    for tstp in MATSUBARA..=(s.nt - 1) {
        vie2_timestep(tstp, &mut g, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
    }
    let tstp = s.nt;
    let mut gref = g.clone();
    vie2_timestep(tstp, &mut gref, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
    let ntau = g.ntau as i64;

    // the current left-mixing row is cleared before the convolution
    for val in [1e6f64, f64::NAN] {
        let mut x = g.clone();
        vie2_timestep_ret(tstp, &mut x, &s.fcc, &s.f, &s.q, s.h, &ik);
        for m in 0..=ntau {
            x.lmix.raw_block_at_mut(tstp, m).fill(c(val, 0.0));
        }
        vie2_timestep_lmix(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for m in 0..=ntau {
            assert_eq!(x.lmix.raw_block_at(tstp, m), gref.lmix.raw_block_at(tstp, m));
        }
    }

    // the current lesser column does not enter the source term
    for val in [1e6f64, f64::NAN] {
        let mut x = g.clone();
        vie2_timestep_ret(tstp, &mut x, &s.fcc, &s.f, &s.q, s.h, &ik);
        vie2_timestep_lmix(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for j in 0..=tstp {
            x.less.raw_block_at_mut(j, tstp).fill(c(val, 0.0));
        }
        vie2_timestep_less(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for j in 0..=tstp {
            assert_eq!(x.less.raw_block_at(j, tstp), gref.less.raw_block_at(j, tstp));
        }
    }
}

#[test]
#[should_panic]
fn vie2_rejects_timestep_out_of_range() {
    let s = vt_setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_timestep(s.nt + 1, &mut g, &s.f, &s.fcc, &s.q, &i2, s.beta, s.h, MatsubaraMethod::Fixpoint);
}

#[test]
#[should_panic]
fn vie2_rejects_negative_timestep() {
    let s = vt_setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_timestep(-2, &mut g, &s.f, &s.fcc, &s.q, &i2, s.beta, s.h, MatsubaraMethod::Fixpoint);
}

#[test]
#[should_panic]
fn vie2_rejects_zero_step() {
    let s = vt_setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_timestep(1, &mut g, &s.f, &s.fcc, &s.q, &i2, s.beta, 0.0, MatsubaraMethod::Fixpoint);
}

#[test]
#[should_panic]
fn vie2_start_rejects_negative_beta() {
    let s = vt_setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_start_impl(&mut g, &s.f, &s.fcc, &s.q, &i2, s.beta, -0.1);
}

#[test]
#[should_panic]
fn vie2_start_rejects_order_six() {
    let s = vt_setup(8, 2, Sign::Fermi);
    let i6 = Integrator::new(6);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_start_impl(&mut g, &s.f, &s.fcc, &s.q, &i6, s.beta, 0.25);
}

#[test]
#[should_panic]
fn vie2_start_rejects_short_grid() {
    // 3 points = 2 intervals, solve order 3 needs more
    let s = vt_setup(2, 2, Sign::Fermi);
    let i3 = Integrator::new(3);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    vie2_start_impl(&mut g, &s.f, &s.fcc, &s.q, &i3, s.beta, s.h);
}

#[test]
fn vie2_singular_kernel() {
    let k = 3i64;
    let s = vt_setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let nt = s.nt;
    let id = DenseMat::identity(2);

    // zero singular parts: the ordinary VIE2
    let mut fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut g1 = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut g2 = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for tstp in MATSUBARA..=nt {
        vie2_timestep(tstp, &mut g1, &s.f, &s.fcc, &s.q, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint);
        let mut ws = Vie2SinWorkspace::new(&g2);
        vie2_timestep_sin(tstp, &mut g2, &mut gs, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    }
    for n in MATSUBARA..=nt {
        assert_eq!(g1.distance_at(&g2, n), 0.0, "row {n}");
    }

    // constructed problem: D = 1 + Fsin, F' = D F, Fcc' = Fcc D,
    // Q' = D Q + F' Gsin
    let mut dm = DenseMat::zeros(2);
    dm.set(0, 0, c(1.3, 0.0));
    dm.set(0, 1, c(0.0, 0.1));
    dm.set(1, 0, c(0.0, -0.1));
    dm.set(1, 1, c(0.8, 0.0));
    let mut dc = tf_const(nt, &dm);
    let mut gsx = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fblk = [
            c(0.5, 0.0),
            c(0.2, 0.0),
            c(0.2, 0.0),
            c(-0.4, 0.0),
        ];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, dm.get(a, b) - id.get(a, b));
                qs.set(t, a, b, fblk[a * 2 + b] * c(1.0 + 0.1 * ijl, 0.0));
            }
        }
        // Gsx = D^-1 Qs
        let di = mat_inv(&dm);
        let qsm = DenseMat::from_vec(2, 2, qs.block(t).to_vec());
        let gx = di.mat_mul(&qsm);
        for a in 0..2usize {
            for b in 0..2usize {
                gsx.set(t, a, b, gx.get(a, b));
            }
        }
    }
    let mut fd = s.f.clone();
    let mut fccd = s.fcc.clone();
    let mut qd = s.q.clone();
    for t in MATSUBARA..=nt {
        let dm2 = DenseMat::from_vec(2, 2, dc.block(t).to_vec());
        fd.lmul_at(t, &dm2, Mode::Normal);
        fccd.rmul_at(t, &dm2, Mode::Normal);
        qd.lmul_at(t, &dm2, Mode::Normal);
    }
    let mut fg = fd.clone();
    for t in MATSUBARA..=nt {
        let gsm = DenseMat::from_vec(2, 2, gsx.block(t).to_vec());
        fg.rmul_at(t, &gsm, Mode::Normal);
        let step = timestep_at(&fg, t.max(0));
        qd.incr_by_step(&step, t, C64::ONE);
    }
    let mut g3 = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gs3 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for tstp in MATSUBARA..=nt {
        let mut ws = Vie2SinWorkspace::new(&g3);
        vie2_timestep_sin(tstp, &mut g3, &mut gs3, &fd, &fccd, &fs, &qd, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    }
    let mut dmax = 0.0f64;
    for n in MATSUBARA..=nt {
        dmax = dmax.max(g1.distance_at(&g3, n));
    }
    assert!(dmax < 1e-12, "regular part: {dmax}");
    let mut gerr = 0.0f64;
    for t in MATSUBARA..=nt {
        for q in 0..4 {
            gerr = gerr.max((gs3.block(t)[q] - gsx.block(t)[q]).abs());
        }
    }
    assert!(gerr < 1e-13, "singular part: {gerr}");
}

#[test]
#[should_panic(expected = "singular")]
fn vie2_sin_rejects_singular_kernel() {
    let k = 3i64;
    let s = vt_setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let nt = s.nt;
    let mut fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let id = DenseMat::identity(2);
    for t in MATSUBARA..=nt {
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, -id.get(a, b));
            }
        }
    }
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut ws = Vie2SinWorkspace::new(&g);
    vie2_timestep_sin(0, &mut g, &mut gs, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
}

#[test]
#[should_panic(expected = "hermitian")]
fn vie2_sin_rejects_non_hermitian_kernel() {
    let k = 3i64;
    let s = vt_setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let nt = s.nt;
    let mut fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for t in MATSUBARA..=nt {
        fs.set(t, 0, 0, c(0.1, 0.0));
        fs.set(t, 0, 1, c(0.2, 0.0));
        fs.set(t, 1, 0, c(0.0, 0.0));
        fs.set(t, 1, 1, c(0.1, 0.0));
    }
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut ws = Vie2SinWorkspace::new(&g);
    vie2_timestep_sin(0, &mut g, &mut gs, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
}

#[test]
fn vie2_singular_kernel_workspace() {
    let cases: Vec<(Sign, i64)> = if is_release() {
        vec![
            (Sign::Fermi, 1),
            (Sign::Fermi, 2),
            (Sign::Fermi, 3),
            (Sign::Fermi, 4),
            (Sign::Fermi, 5),
            (Sign::Bose, 1),
            (Sign::Bose, 2),
            (Sign::Bose, 3),
            (Sign::Bose, 4),
            (Sign::Bose, 5),
        ]
    } else {
        vec![(Sign::Fermi, 1), (Sign::Bose, 3), (Sign::Fermi, 5)]
    };
    for &(sign, k) in &cases {
        let (gr, gsr) = vt_sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            vt_sin_reference(tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h);
        });
        // without a persistent workspace
        let (g1, gs1) = vt_sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            let mut ws = Vie2SinWorkspace::new(g);
            vie2_timestep_sin(tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h, MatsubaraMethod::Fixpoint, &mut ws);
        });
        assert!(vt_same(&g1, &gr), "fresh ws, k = {k}");
        for t in MATSUBARA..=(gr.nt as i64) {
            assert_eq!(gs1.block(t), gsr.block(t), "gs k = {k} at {t}");
        }
        // one workspace for all the calls
        let mut wsopt: Option<Vie2SinWorkspace> = None;
        let (g2, gs2) = vt_sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            if wsopt.is_none() {
                wsopt = Some(Vie2SinWorkspace::new(g));
            }
            vie2_timestep_sin(tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h, MatsubaraMethod::Fixpoint, wsopt.as_mut().unwrap());
        });
        assert!(vt_same(&g2, &gr), "shared ws, k = {k}");
        for t in MATSUBARA..=(gr.nt as i64) {
            assert_eq!(gs2.block(t), gsr.block(t));
        }
    }
}

#[test]
fn vie2_sin_workspace_reset_and_invalidation() {
    let k = 2i64;
    let s = vt_setup(10, k, Sign::Fermi);
    let nt = s.nt;
    let ik = Integrator::new(k);
    let mut fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fblk = [
            c(0.2 * (1.0 + 0.02 * ijl), 0.0),
            c(0.0, 0.05),
            c(0.0, -0.05),
            c(-0.1 * (1.0 + 0.02 * ijl), 0.0),
        ];
        let qblk = [c(0.4, 0.0), c(0.1, 0.0), c(0.1, 0.0), c(-0.3, 0.0)];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, fblk[a * 2 + b]);
                qs.set(t, a, b, qblk[a * 2 + b]);
            }
        }
    }
    let mut f2 = s.f.clone();
    for t in MATSUBARA..=nt {
        f2.smul_at(t, c(0.9, 0.0));
    }
    let mut gr = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gsr = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut gw = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gsw = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut ws = Vie2SinWorkspace::new(&gw);
    for tstp in MATSUBARA..=nt {
        vt_sin_reference(tstp, &mut gr, &mut gsr, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h);
        vie2_timestep_sin(tstp, &mut gw, &mut gsw, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    }
    assert!(vt_same(&gw, &gr));
    assert_eq!(ws.ndone, nt);

    // another kernel object at the last time step
    vt_sin_reference(nt, &mut gr, &mut gsr, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h);
    vie2_timestep_sin(nt, &mut gw, &mut gsw, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    assert!(vt_same(&gw, &gr));

    // back to an earlier time step with modified inputs, then forward
    let id = DenseMat::identity(2);
    let t0 = k + 3;
    for t in t0..=nt {
        for a in 0..2usize {
            for b in 0..2usize {
                let v = fs.block(t)[a * 2 + b] + c(0.01, 0.0) * id.get(a, b);
                fs.set(t, a, b, v);
            }
        }
        f2.smul_at(t, c(1.1, 0.0));
    }
    for tstp in t0..=nt {
        vt_sin_reference(tstp, &mut gr, &mut gsr, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h);
        vie2_timestep_sin(tstp, &mut gw, &mut gsw, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    }
    assert!(vt_same(&gw, &gr));
    for t in MATSUBARA..=nt {
        assert_eq!(gsw.block(t), gsr.block(t));
    }

    // another Gsin object at a later time step: the workspace resets
    let mut gsn = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut gsr2 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    vt_sin_reference(nt, &mut gr, &mut gsr2, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h);
    vie2_timestep_sin(nt, &mut gw, &mut gsn, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut ws);
    assert!(vt_same(&gw, &gr));
    for t in MATSUBARA..=nt {
        assert_eq!(gsn.block(t), gsr2.block(t));
    }
}

#[test]
#[should_panic(expected = "workspace")]
fn vie2_sin_rejects_incompatible_workspace() {
    let k = 2i64;
    let s = vt_setup(10, k, Sign::Fermi);
    let nt = s.nt;
    let ik = Integrator::new(k);
    let fs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let qs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let s2 = vt_setup(6, k, Sign::Fermi);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, C64::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut bad = Vie2SinWorkspace::new(&ContourGreen::new(&s2.contour, Sign::Fermi, C64::ZERO));
    vie2_timestep_sin(0, &mut g, &mut gs, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h, MatsubaraMethod::Fixpoint, &mut bad);
}

#[test]
fn vie2_convergence_order() {
    for k in 1..=5i64 {
        let (ns, caps): (&[i64], [f64; 5]) = if is_release() {
            (
                &[20, 40, 80, 160],
                [5e-7, 1e-8, 2e-10, 1e-11, 1e-12],
            )
        } else {
            (
                &[8, 16, 32, 64],
                [5e-6, 2e-7, 1e-8, 1e-9, 1e-10],
            )
        };
        let mut errs3 = [Vec::new(), Vec::new(), Vec::new()];
        for &n in ns {
            let e = vt_errors(k, n, Sign::Fermi);
            errs3[0].push(e.0);
            errs3[1].push(e.1);
            errs3[2].push(e.2);
        }
        for errs in &errs3 {
            let gate = check_order_gate(ns, errs, (k + 2) as f64, caps[(k - 1) as usize], 1e-13, 0.35);
            assert!(gate.ok, "k = {k}: {:?} (errs {errs:?})", gate.msg);
        }
        // reduced-order mutant at k = 2
        if k == 2 {
            let mut m3 = [Vec::new(), Vec::new(), Vec::new()];
            for &n in ns {
                let e = vt_errors(1, n, Sign::Fermi);
                m3[0].push(e.0);
                m3[1].push(e.1);
                m3[2].push(e.2);
            }
            for errs in &m3 {
                let gate = check_order_gate(ns, errs, 4.0, caps[1], 1e-13, 0.35);
                assert!(!gate.ok, "mutant must be rejected");
            }
        }
    }
}
