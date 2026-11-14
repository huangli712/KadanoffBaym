// Project : KadanoffBaym
// File    : tests/dyson_moving.rs
// Purpose : blueprint t_dyson_moving.jl: the truncated (moving) Dyson
//           equation and the truncated kinetic energy integral
//
// The moving Dyson equation of NESSi uses a different discretization
// than the full one, so the full and the moving solutions agree up to
// the time propagation error only; the agreement is checked by its
// convergence with h and for a fine grid. The pointwise NESSi
// reference is tests/moving_solvers.rs. The alias check
// (Sigma === G) of the blueprint is subsumed by the Rust borrow
// checker (known deviation).

mod common;

use common::*;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::moving::algebra::distance_moving;
use kadanoffbaym::moving::window::{copy_timestep, forward, init_from_function, init_from_g_herm};
use kadanoffbaym::observables::kinetic_energy_trunc;
use kadanoffbaym::solvers::dyson::dyson;
use kadanoffbaym::solvers::dyson_mat::MatsubaraMethod;
use kadanoffbaym::solvers::dyson_moving::dyson_timestep_moving;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::moving::{CfMoving, MovingGreen, MovingGreenTimeStep};
use kadanoffbaym::weights::Integrator;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn dm_h0() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(-1.0, 0.0));
    m.set(0, 1, c(0.0, 0.2));
    m.set(1, 0, c(0.0, -0.2));
    m.set(1, 1, c(1.0, 0.0));
    m
}

/// Self-energy of the reference generator: a free 2x2 Green's function
/// with a short range in t - t', time translation symmetry broken by
/// cosines (blueprint `dm_setup`). `nt_points`/`ntau_points` are the
/// blueprint point counts.
#[allow(clippy::too_many_arguments)]
fn dm_setup(
    nt_points: i64,
    ntau_points: i64,
    h: f64,
    beta: f64,
    mu: f64,
    k: i64,
    trange: f64,
    tv: bool,
) -> (ContourGreen, ContourGreen, TimeFunction) {
    let h0 = dm_h0();
    let cont = Contour::new(
        nt_points - 1,
        ntau_points - 1,
        2,
        2,
        (nt_points - 1) as f64 * h,
        beta,
    );
    let mut sigma = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    from_ham(&mut sigma, mu, &h0, beta, h);
    for n in 0..=(nt_points - 1) {
        for m in 0..=n {
            let t = n as f64 * h;
            let s = m as f64 * h;
            let f = c(
                (-(((t - s) / trange) * ((t - s) / trange))).exp()
                    * (0.2 * t).cos()
                    * (0.2 * s).cos(),
                0.0,
            );
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
    for z in sigma.mat.raw_mut().iter_mut() {
        *z = Complex::ZERO;
    }
    if tv {
        for z in sigma.lmix.raw_mut().iter_mut() {
            *z *= c(0.5, 0.0);
        }
    } else {
        for z in sigma.lmix.raw_mut().iter_mut() {
            *z = Complex::ZERO;
        }
    }
    let mut hf = TimeFunction::new((nt_points - 1) as usize, 2, 2, Complex::ZERO);
    for t in -1..=(nt_points - 1) {
        let w = if t == -1 {
            1.0
        } else {
            (0.1 * t as f64 * h).cos()
        };
        for a in 0..2usize {
            for b in 0..2usize {
                hf.set(t, a, b, h0.get(a, b) * c(w, 0.0));
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
    (g, sigma, hf)
}

/// Moving copies of the full functions at the 0-based time step `n0`
/// (blueprint `dm_moving`).
fn dm_moving(
    g: &ContourGreen,
    sigma: &ContourGreen,
    hf: &TimeFunction,
    tc: i64,
    n0: i64,
) -> (MovingGreen, MovingGreen, CfMoving) {
    let d = g.ret.size1;
    let mut gm = MovingGreen::new(tc, d, Sign::Fermi);
    let mut sm = MovingGreen::new(tc, d, Sign::Fermi);
    let mut hm = CfMoving::new(tc, d);
    init_from_g_herm(&mut gm, g, n0);
    init_from_g_herm(&mut sm, sigma, n0);
    init_from_function(&mut hm, hf, n0);
    (gm, sm, hm)
}

/// Leading row of the moving `gm` against the full `g` at the 0-based
/// time step `n` (blueprint `dm_rowerr`).
fn dm_rowerr(gm: &MovingGreen, g: &ContourGreen, n: i64) -> (f64, f64) {
    let d = gm.ret.size1;
    let (mut er, mut el) = (0.0f64, 0.0f64);
    for j in 0..=gm.tc {
        if n - j < 0 {
            continue;
        }
        let gr = gm.ret_block(n, n - j);
        let fr = g.ret_block(n, n - j);
        for q in 0..(d * d) {
            er = er.max((gr[q] - fr.raw()[q]).abs());
        }
        let gl = gm.less_block(n - j, n);
        let fl = g.less_block(n - j, n);
        for q in 0..(d * d) {
            el = el.max((gl[q] - fl.raw()[q]).abs());
        }
    }
    (er, el)
}

/// Owning moving time step of the full `g` at the 0-based time step
/// `n` (blueprint `dm_fullrow`).
fn dm_fullrow(g: &ContourGreen, tc: i64, n: i64) -> MovingGreenTimeStep {
    let d = g.ret.size1;
    let mut x = MovingGreenTimeStep::new(tc, d, n, g.sign);
    for j in 0..=tc.min(n) {
        let r = g.ret.raw_block_at(n, n - j);
        x.ret.mblock_mut(j).copy_from_slice(r);
        let s = g.less.raw_block_at(n - j, n);
        let blk = x.less.mblock_mut(j);
        for a in 0..d {
            for b in 0..d {
                blk[a * d + b] = -s[b * d + a].conj();
            }
        }
    }
    x
}

/// One step of the moving propagation, `sigma` and `hf` are taken from
/// the full ones (blueprint `dm_step!`).
#[allow(clippy::too_many_arguments)]
fn dm_step(
    gm: &mut MovingGreen,
    sm: &mut MovingGreen,
    hm: &mut CfMoving,
    sigma: &ContourGreen,
    hf: &TimeFunction,
    mu: f64,
    h: f64,
    k: i64,
) -> i64 {
    forward(sm);
    hm.forward();
    forward(gm);
    let n = gm.tstp;
    let d = gm.ret.size1;
    for j in 0..=sm.tc.min(n) {
        let r = sigma.ret.raw_block_at(n, n - j);
        sm.ret.mblock_mut(0, j).copy_from_slice(r);
        let s = sigma.less.raw_block_at(n - j, n);
        let blk = sm.less.mblock_mut(0, j);
        for a in 0..d {
            for b in 0..d {
                blk[a * d + b] = -s[b * d + a].conj();
            }
        }
    }
    hm.fslot_mut(0).copy_from_slice(hf.block(n));
    dyson_timestep_moving(n, gm, mu, hm, sm, h, k);
    n
}

#[test]
fn dyson_moving_free_propagation() {
    let (beta, mu) = (1.0, -0.1);
    let (d, h, k, tc) = (2usize, 0.01, 5i64, 14i64);
    let mut h0 = DenseMat::zeros(2);
    h0.set(0, 0, c(-0.7, 0.0));
    h0.set(0, 1, c(0.3, 0.0));
    h0.set(1, 0, c(0.3, 0.0));
    h0.set(1, 1, c(0.4, 0.0));
    let nt_points = 40i64;
    let cont = Contour::new(nt_points - 1, 10, 2, 2, (nt_points - 1) as f64 * h, beta);
    let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g, mu, &h0, beta, h);
    let sigma = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    let mut hf = TimeFunction::new((nt_points - 1) as usize, d, d, Complex::ZERO);
    for t in -1..=(nt_points - 1) {
        for a in 0..d {
            for b in 0..d {
                hf.set(t, a, b, h0.get(a, b));
            }
        }
    }
    let (mut gm, mut sm, mut hm) = dm_moving(&g, &sigma, &hf, tc, tc);
    for _ in (tc + 1)..=(nt_points - 1) {
        dm_step(&mut gm, &mut sm, &mut hm, &sigma, &hf, mu, h, k);
    }
    let (er, el) = dm_rowerr(&gm, &g, nt_points - 1);
    assert!(er < 1e-10, "ret {er}");
    assert!(el < 1e-10, "less {el}");
    // G^{<}(t, t) is anti-hermitian
    let blk = gm.less.mblock(0, 0);
    let mut ah = 0.0f64;
    for a in 0..d {
        for b in 0..d {
            ah = ah.max((blk[a * d + b] + blk[b * d + a].conj()).abs());
        }
    }
    assert!(ah < 1e-12, "{ah}");
}

#[test]
fn dyson_moving_zero_mixing_window_covers_history() {
    let (beta, mu) = (1.0, -0.1);
    let tmax = 0.8;
    let mut errs = Vec::new();
    let mut ahs = Vec::new();
    for &h in &[0.02f64, 0.01] {
        let nt_points = (tmax / h).round() as i64 + 1;
        let (g, sigma, hf) = dm_setup(nt_points, 21, h, beta, mu, 5, 1.0, false);
        let n = nt_points - 1;
        let (mut gm, sm, hm) = dm_moving(&g, &sigma, &hf, nt_points - 2, n);
        dyson_timestep_moving(n, &mut gm, mu, &hm, &sm, h, 5);
        let (er, el) = dm_rowerr(&gm, &g, n);
        errs.push(er.max(el));
        let d = 2usize;
        let blk = gm.less.mblock(0, 0);
        let mut ah = 0.0f64;
        for a in 0..d {
            for b in 0..d {
                ah = ah.max((blk[a * d + b] + blk[b * d + a].conj()).abs());
            }
        }
        ahs.push(ah);
    }
    assert!(errs[1] < 1e-10, "{errs:?}");
    assert!(errs[0] / errs[1] > 32.0, "{errs:?}");
    assert!(ahs[1] < 1e-10, "{ahs:?}");
    assert!(ahs[0] / ahs[1] > 32.0, "{ahs:?}");
}

#[test]
fn dyson_moving_truncated_propagation() {
    let (beta, mu) = (1.0, -0.1);
    // blueprint release config ntau = 100 gives an odd interval count
    // 99 which the Matsubara seed rejects; the blueprint's own release
    // branch never activates (verified by a Julia probe). Use 101
    // points = 100 intervals.
    let (nt_points, ntau_points, h, tc) = if is_release() {
        (200i64, 101i64, 0.02, 100i64)
    } else {
        (60i64, 21i64, 0.02, 20i64)
    };
    let k = 5i64;
    let (g, sigma, hf) = dm_setup(nt_points, ntau_points, h, beta, mu, k, tc as f64 * h / 5.0, false);
    let (mut gm, mut sm, mut hm) = dm_moving(&g, &sigma, &hf, tc, tc);
    let mut err = 0.0f64;
    for _ in (tc + 1)..=(nt_points - 1) {
        let n = dm_step(&mut gm, &mut sm, &mut hm, &sigma, &hf, mu, h, k);
        err += distance_moving(&copy_timestep(&gm, 0), &dm_fullrow(&g, tc, n));
    }
    assert!((err / nt_points as f64) < 1e-3, "{err}");
}

#[test]
fn dyson_moving_nonzero_mixing_no_full_equivalence() {
    let (beta, mu) = (1.0, -0.1);
    let (h, nt_points) = (0.01, 41i64);
    let (_g0, _s0, _h0) = dm_setup(nt_points, 21, h, beta, mu, 5, 1.0, false);
    let (g1, sigma1, hf1) = dm_setup(nt_points, 21, h, beta, mu, 5, 1.0, true);
    let lmax = sigma1
        .lmix
        .raw()
        .iter()
        .map(|z| z.abs())
        .fold(0.0f64, f64::max);
    assert!(lmax > 0.01, "{lmax}");
    let n = nt_points - 1;
    let (mut gm, sm, hm) = dm_moving(&g1, &sigma1, &hf1, nt_points - 2, n);
    dyson_timestep_moving(n, &mut gm, mu, &hm, &sm, h, 5);
    let (er, el) = dm_rowerr(&gm, &g1, n);
    // the initial correlations are missing in the moving solver
    assert!(er.max(el) > 1e-6, "er {er} el {el}");
}

#[test]
fn dyson_moving_orders_converge() {
    let (beta, mu) = (1.0, -0.1);
    let tmax = 0.4;
    for k in 1..=5i64 {
        let mut e = Vec::new();
        for &h in &[0.02f64, 0.01] {
            let nt_points = (tmax / h).round() as i64 + 1;
            let (g, sigma, hf) = dm_setup(nt_points, 11, h, beta, mu, k, 1.0, false);
            let n = nt_points - 1;
            let (mut gm, sm, hm) = dm_moving(&g, &sigma, &hf, nt_points - 2, n);
            dyson_timestep_moving(n, &mut gm, mu, &hm, &sm, h, k);
            let (er, el) = dm_rowerr(&gm, &g, n);
            e.push(er.max(el));
        }
        assert!(
            e[0] / e[1] > 2f64.powi(k as i32) / 2f64.sqrt(),
            "k = {k}: {e:?}"
        );
    }
}

#[test]
fn dyson_moving_kinetic_energy() {
    let (beta, mu) = (1.0, -0.1);
    let (h, nt_points, k, tc) = (0.02, 30i64, 3i64, 12i64);
    let (g, sigma, hf) = dm_setup(nt_points, 11, h, beta, mu, k, 1.0, false);
    let n = nt_points - 1;
    let (gm, sm, _hm) = dm_moving(&g, &sigma, &hf, tc, n);
    let integ = Integrator::new(k);
    // direct sum over the full blocks
    let d = 2usize;
    let mut s = DenseMat::zeros(d);
    for l in 0..=tc {
        let w = if l <= k {
            integ.giw.omega[l as usize]
        } else if l >= tc - k {
            integ.giw.omega[(tc - l) as usize]
        } else {
            1.0
        };
        let gr = g.ret_block(n, n - l);
        let gl_src = g.less_block(n - l, n);
        let dr = sigma.ret_block(n, n - l);
        let dl_src = sigma.less_block(n - l, n);
        // DL = -src', GL = -src'
        let neg_adj = |m: &DenseMat| {
            let mut o = DenseMat::zeros(d);
            for a in 0..d {
                for b in 0..d {
                    o.set(a, b, -m.get(b, a).conj());
                }
            }
            o
        };
        let dl = neg_adj(&dl_src);
        let gl = neg_adj(&gl_src);
        let p1 = gr.adjoint().mat_mul(&dl);
        let p2 = gl.adjoint().mat_mul(&dr);
        for q in 0..(d * d) {
            let v = c(h * w, 0.0) * (p1.raw()[q] - p2.raw()[q]);
            *s.raw_mut().get_mut(q).unwrap() += v;
        }
    }
    let mut tr = Complex::ZERO;
    for a in 0..d {
        tr += s.get(a, a);
    }
    let got = kinetic_energy_trunc(n, &gm, &sm, &integ, h);
    assert!((got - tr.im).abs() < 1e-12, "{got} vs {}", tr.im);
    // tc >= 2k + 2 is required
    let (gs, ss, _hs) = dm_moving(&g, &sigma, &hf, 11, n);
    let i5 = Integrator::new(5);
    assert!(panics(move || kinetic_energy_trunc(n, &gs, &ss, &i5, h)));
}

#[test]
fn dyson_moving_failure_paths() {
    let (beta, mu) = (1.0, -0.1);
    let (h, nt_points) = (0.02, 30i64);
    let (g, sigma, hf) = dm_setup(nt_points, 11, h, beta, mu, 3, 1.0, false);
    let (mut gm, mut sm, hm) = dm_moving(&g, &sigma, &hf, 9, 19);
    // tc >= 2k + 2
    assert!(panics(|| dyson_timestep_moving(19, &mut gm, mu, &hm, &sm, h, 4)));
    assert!(panics(|| dyson_timestep_moving(19, &mut gm, mu, &hm, &sm, h, 6)));
    // wrong leading time step
    assert!(panics(|| dyson_timestep_moving(20, &mut gm, mu, &hm, &sm, h, 3)));
    // uninitialized G
    let mut gun = MovingGreen::new(9, 2, Sign::Fermi);
    assert!(panics(|| dyson_timestep_moving(19, &mut gun, mu, &hm, &sm, h, 3)));
    // uninitialized H
    let hun = CfMoving::new(9, 2);
    assert!(panics(|| dyson_timestep_moving(19, &mut gm, mu, &hun, &sm, h, 3)));
    // (the alias check Sigma === G is subsumed by the borrow checker)
    // the history rows are not changed
    let before = gm.ret.mblock(1, 3).to_vec();
    dyson_timestep_moving(19, &mut gm, mu, &hm, &sm, h, 3);
    assert_eq!(gm.ret.mblock(1, 3), before);
}
