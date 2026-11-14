// Project : KadanoffBaym
// File    : tests/moving_solvers.rs
// Purpose : blueprint t_moving_solvers.jl: pointwise comparison of the
//           moving (truncated) solver functions with the NESSi
//           reference data `p8_moving_case*.dat` (window algebra) and
//           `p8_moving_solver.dat` (truncated solvers on closed-form
//           inputs)
//
// The NESSi time step `step` of a series record is the 0-based time
// step of the Rust moving types. The primary condition is
// |actual - expected| <= 1e-10 + 1e-10 |expected| for every record.

mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::{bubble1_moving_herm, bubble2_moving_herm};
use kadanoffbaym::equilibrium::from_ham;
use common::kbref_raw::{read_kbref, KbrefData};
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::moving::{
    extrapolate_timestep_moving_order, incr_step, lmul_step, rmul_step, smul_step,
};
use kadanoffbaym::moving::{copy_timestep, forward, init_from_function, init_from_g_herm};
use kadanoffbaym::observables::kinetic_energy_trunc;
use kadanoffbaym::dyson::dyson;
use kadanoffbaym::dyson::MatsubaraMethod;
use kadanoffbaym::dyson::dyson_timestep_moving;
use kadanoffbaym::vie2::moving::vie2_timestep_moving;
use kadanoffbaym::types::Contour;
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::TimeFunction;
use kadanoffbaym::types::{MovingTimeFunction, MovingGreen};
use kadanoffbaym::weights::Integrator;
use std::collections::BTreeMap;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn movref_ok(a: Complex, b: Complex) -> bool {
    (a - b).abs() <= 1e-10 + 1e-10 * b.abs()
}

fn movref_ok_re(a: f64, b: Complex) -> bool {
    (a - b.re).abs() <= 1e-10 + 1e-10 * b.re.abs()
}

struct MovCase {
    gm: MovingGreen,
    sm: MovingGreen,
    wm: MovingGreen,
    hm: MovingTimeFunction,
    n: i64,
    tc: i64,
    mu: f64,
    h: f64,
}

fn h0_of(size: usize) -> DenseMat {
    if size == 1 {
        let mut m = DenseMat::zeros(1);
        m.set(0, 0, c(-0.3, 0.0));
        m
    } else {
        let mut m = DenseMat::zeros(2);
        m.set(0, 0, c(-0.3, 0.0));
        m.set(0, 1, c(0.0, 0.2));
        m.set(1, 0, c(0.0, -0.2));
        m.set(1, 1, c(0.4, 0.0));
        m
    }
}

/// Inputs of the reference generator (blueprint `movref_solver_case`;
/// see .nessi/kbref/p8_moving_solver_ref.cpp).
fn movref_solver_case(size: usize, k: i64, md: &BTreeMap<String, String>) -> MovCase {
    let nt: i64 = md["nt"].parse().unwrap();
    let ntau: i64 = md["ntau"].parse().unwrap();
    let tc: i64 = md["tc"].parse().unwrap();
    let (beta, h, mu) = (1.0f64, 0.02, -0.1);
    let cont = Contour::new(nt, ntau, size as i64, size as i64, nt as f64 * h, beta);
    let h0 = h0_of(size);
    let mut sigma = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    from_ham(&mut sigma, mu, &h0, beta, h);
    for n in 0..=nt {
        for m in 0..=n {
            let f = c(0.3 * (-(((n - m) as f64) * h).powi(2)).exp(), 0.0);
            let blk = sigma.ret.raw_block_at_mut(n, m);
            for z in blk.iter_mut() {
                *z *= f;
            }
            let blk = sigma.less.raw_block_at_mut(m, n);
            for z in blk.iter_mut() {
                *z *= f;
            }
        }
    }
    for z in sigma.lmix.raw_mut().iter_mut() {
        *z = Complex::ZERO;
    }
    for z in sigma.mat.raw_mut().iter_mut() {
        *z = Complex::ZERO;
    }
    let mut hf = TimeFunction::new(nt as usize, size, size, Complex::ZERO);
    for t in -1..=nt {
        for a in 0..size {
            for b in 0..size {
                hf.set(t, a, b, h0.get(a, b));
            }
        }
    }
    let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    dyson(
        &mut g,
        mu,
        &hf,
        &sigma,
        beta,
        h,
        k,
        MatsubaraMethod::Fixpoint,
        true,
    );
    let mk = |x: &ContourGreen| -> MovingGreen {
        let mut m = MovingGreen::new(tc, x.ret.size1, x.sign);
        init_from_g_herm(&mut m, x, nt);
        m
    };
    let w_full = ContourGreen::new(&cont, Sign::Bose, Complex::ZERO);
    let mut hm = MovingTimeFunction::new(tc, size);
    init_from_function(&mut hm, &hf, nt);
    MovCase {
        gm: mk(&g),
        sm: mk(&sigma),
        wm: mk(&w_full),
        hm,
        n: nt,
        tc,
        mu,
        h,
    }
}

fn refpath(name: &str) -> String {
    format!("data/test/{name}")
}

#[test]
fn moving_solvers_data_and_algebra() {
    let d0 = read_kbref(&refpath("p8_moving_case0.dat")).unwrap();
    let mut md = d0.metadata.clone();
    md.insert("nt".to_string(), "30".to_string());
    md.insert("ntau".to_string(), "40".to_string());
    let x = movref_solver_case(2, 3, &md);
    let (n, h) = (x.n, x.h);
    let mut g = x.gm.clone();
    let mut s = x.sm.clone();

    // window of the full G after the four NESSi operations
    let fm = |m: i64| -> [Complex; 4] {
        [
            c(1.0 + 0.1 * m as f64 * h, 0.0),
            c(0.0, 0.2),
            c(-0.1, 0.0),
            c(0.8 - 0.05 * m as f64 * h, 0.0),
        ]
    };
    let mut states: Vec<MovingGreen> = vec![g.clone()];
    // forward! + extrapolation of order 3
    forward(&mut g);
    extrapolate_timestep_moving_order(&mut g, 3);
    states.push(g.clone());
    // left and right multiplication by the moving single-time function
    let mut cf = TimeFunction::new((n + 1) as usize, 2, 2, Complex::ZERO);
    for m in 0..=n + 1 {
        let blk = fm(m);
        for a in 0..2usize {
            for b in 0..2usize {
                cf.set(m, a, b, blk[a * 2 + b]);
            }
        }
    }
    let mut fmv = MovingTimeFunction::new(x.tc, 2);
    init_from_function(&mut fmv, &cf, n + 1);
    let mut row = copy_timestep(&g, 0);
    lmul_step(&fmv, &mut row, false);
    rmul_step(&fmv, &mut row, false);
    g.ret.mrow_mut(0).copy_from_slice(row.ret.raw());
    g.less.mrow_mut(0).copy_from_slice(row.less.raw());
    states.push(g.clone());
    // smul by 0.7 and incr by the shifted self-energy row
    forward(&mut s);
    let r1r = s.ret.mrow(1).to_vec();
    let r1l = s.less.mrow(1).to_vec();
    s.ret.mrow_mut(0).copy_from_slice(&r1r);
    s.less.mrow_mut(0).copy_from_slice(&r1l);
    let mut row = copy_timestep(&g, 0);
    smul_step(&mut row, c(0.7, 0.0));
    let s0 = copy_timestep(&s, 0);
    incr_step(&mut row, &s0, c(0.3, 0.0));
    g.ret.mrow_mut(0).copy_from_slice(row.ret.raw());
    g.less.mrow_mut(0).copy_from_slice(row.less.raw());
    states.push(g.clone());

    let mut ok = true;
    let mut used = 0usize;
    for case in 0..=3i64 {
        let d = read_kbref(&refpath(&format!("p8_moving_case{case}.dat"))).unwrap();
        let xx = &states[case as usize];
        let want_tstp: i64 = d.metadata["tstp"].parse().unwrap();
        assert_eq!(xx.tstp, want_tstp, "case {case}");
        for r in &d.rows {
            let comp: &[u8] = r.component.as_bytes();
            let blk: &[Complex] = if comp == b"ret" {
                xx.ret.mblock(r.i, r.j)
            } else {
                xx.less.mblock(r.i, r.j)
            };
            let v = blk[r.a as usize * 2 + r.b as usize];
            ok &= movref_ok(v, r.value);
            used += 1;
        }
    }
    assert!(ok, "reference mismatch");
    assert_eq!(used as i64, 4 * 2 * (x.tc + 1) * (x.tc + 1) * 4);
}

#[test]
fn moving_solvers_reference() {
    let d: KbrefData = read_kbref(&refpath("p8_moving_solver.dat")).unwrap();
    let cases = [(1usize, 1i64), (1, 3), (1, 5), (2, 1), (2, 3), (2, 5)];
    let mut used = 0usize;
    let mut ok = true;
    for (ci, &(size, k)) in cases.iter().enumerate() {
        let x = movref_solver_case(size, k, &d.metadata);
        let n = x.n;
        let rows: Vec<_> = d.rows.iter().filter(|r| r.i == ci as i64).collect();
        assert!(!rows.is_empty(), "case {ci} has no rows");
        // bubbles of the orbitals a (G) and b (Sigma), before the Dyson step
        let mut b1: Vec<MovingGreen> = Vec::new();
        let mut b2: Vec<MovingGreen> = Vec::new();
        for a in 0..size {
            for b in 0..size {
                let mut w1 = x.wm.clone();
                bubble1_moving_herm(n, &mut w1, 0, 0, &x.gm, a, a, &x.sm, b, b);
                let mut w2 = x.wm.clone();
                bubble2_moving_herm(n, &mut w2, 0, 0, &x.gm, a, a, &x.sm, b, b);
                b1.push(w1);
                b2.push(w2);
            }
        }
        let kin = kinetic_energy_trunc(n, &x.gm, &x.sm, &Integrator::new(k), x.h);
        // truncated VIE2 G + Sigma * G = G (Q = the window of G), before
        // the Dyson step changes the leading row of x.gm
        let mut gv = x.gm.clone();
        vie2_timestep_moving(n, &mut gv, &x.sm, &x.sm, &x.gm, x.h, k);
        let mut gm = x.gm.clone();
        dyson_timestep_moving(n, &mut gm, x.mu, &x.hm, &x.sm, x.h, k);
        for r in &rows {
            let (a, b) = (r.a as usize, r.b as usize);
            let comp = r.component.as_str();
            let v: Complex = if comp == "kinetic" {
                c(kin, 0.0)
            } else if comp.starts_with("dyson") {
                let blk = if comp == "dyson_ret" {
                    gm.ret.mblock(0, r.j)
                } else {
                    gm.less.mblock(0, r.j)
                };
                blk[a * size + b]
            } else if comp.starts_with("vie2") {
                let blk = if comp == "vie2_ret" {
                    gv.ret.mblock(0, r.j)
                } else {
                    gv.less.mblock(0, r.j)
                };
                blk[a * size + b]
            } else if comp.starts_with("bubble1") {
                let w = &b1[a * size + b];
                let blk = if comp.ends_with("ret") {
                    w.ret.mblock(0, r.j)
                } else {
                    w.less.mblock(0, r.j)
                };
                blk[0]
            } else {
                let w = &b2[a * size + b];
                let blk = if comp.ends_with("ret") {
                    w.ret.mblock(0, r.j)
                } else {
                    w.less.mblock(0, r.j)
                };
                blk[0]
            };
            if comp == "kinetic" {
                ok &= movref_ok_re(kin, r.value);
            } else {
                ok &= movref_ok(v, r.value);
            }
            used += 1;
        }
    }
    assert!(ok, "reference mismatch");
    assert_eq!(used, d.rows.len());
}
