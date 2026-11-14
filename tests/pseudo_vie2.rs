mod common;

use common::*;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::pseudo::pseudo_convolution;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::solvers::dyson_pseudo::pseudo_dyson;
use kadanoffbaym::solvers::vie2_pseudo::{
    pseudo_vie2, pseudo_vie2_mat, pseudo_vie2_start, pseudo_vie2_timestep,
};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::TimeFunction;
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::types::pseudo::{
    pseudo_distance_at, pseudo_incr, pseudo_smul_step, pseudo_timestep, PseudoGreen,
};
use kadanoffbaym::weights::Integrator;

const PV_L0: f64 = 0.5;
const BETA: f64 = 2.0;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn pv_h0(d: usize) -> DenseMat {
    if d == 2 {
        DenseMat::from_vec(2, 2, vec![c(0.3, 0.0), c(0.2, -0.1), c(0.2, 0.1), c(-0.4, 0.0)])
    } else {
        DenseMat::from_vec(1, 1, vec![c(0.3, 0.0)])
    }
}

/// Build G0 (free), Sigma, F = -G0 * Sigma and Fcc = -Sigma * G0
/// (blueprint `pv_setup`; nt/ntau are the blueprint POINT counts).
struct PvSetup {
    cc: Contour,
    hf: TimeFunction,
    g0: PseudoGreen,
    sigma: PseudoGreen,
    f: PseudoGreen,
    fcc: PseudoGreen,
}

fn pv_setup(sign: Sign, k: i64, nt_pts: i64, ntau_pts: i64, h: f64, d: usize) -> PvSetup {
    let (nt, ntau) = (nt_pts - 1, ntau_pts - 1);
    let cc = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, BETA);
    let ik = Integrator::new(k);
    let h0 = pv_h0(d);
    let mut hf = TimeFunction::new(nt as usize, d, d, Complex::ZERO);
    hf.block_mut(MATSUBARA).copy_from_slice(h0.raw());
    for i in 0..=nt {
        let a = 0.3 * (2.0 * i as f64 * h).sin();
        for q in 0..(d * d) {
            let (r0, c0) = (q / d, q % d);
            let base = h0.get(r0, c0);
            hf.block_mut(i)[q] = if r0 == c0 { base + c(a, 0.0) } else { base };
        }
    }
    // G0: the free pseudo-particle Dyson solution
    let mut g0 = PseudoGreen::new(&cc, sign, Complex::ZERO);
    let zero = PseudoGreen::new(&cc, sign, Complex::ZERO);
    pseudo_dyson(&mut g0, PV_L0, &hf, &zero, &ik, BETA, h);
    // Sigma: another Dyson solution, scaled by 0.3
    let mut sigma = PseudoGreen::new(&cc, sign, Complex::ZERO);
    pseudo_dyson(&mut sigma, 0.2, &hf, &zero, &ik, BETA, h);
    for t in MATSUBARA..=nt {
        pseudo_smul_step(&mut sigma, t, c(0.3, 0.0));
    }
    // F = -(G0 * Sigma), Fcc = -(Sigma * G0)
    let mut f = PseudoGreen::new(&cc, sign, Complex::ZERO);
    let mut fcc = PseudoGreen::new(&cc, sign, Complex::ZERO);
    pseudo_convolution(&mut f, &g0, &g0, None, &sigma, &sigma, &ik, BETA, h);
    pseudo_convolution(&mut fcc, &sigma, &sigma, None, &g0, &g0, &ik, BETA, h);
    for t in MATSUBARA..=nt {
        pseudo_smul_step(&mut f, t, c(-1.0, 0.0));
        pseudo_smul_step(&mut fcc, t, c(-1.0, 0.0));
    }
    PvSetup { cc, hf, g0, sigma, f, fcc }
}

fn norm_blk(blk: &[Complex]) -> f64 {
    blk.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt()
}

#[test]
fn pseudo_vie2_zero_kernel() {
    for sign in [Sign::Fermi, Sign::Bose] {
        for k in [1i64, 3, 5] {
            let s = pv_setup(sign, k, 12, 31, 0.01, 2);
            let z = PseudoGreen::new(&s.cc, sign, Complex::ZERO);
            let mut x = PseudoGreen::new(&s.cc, sign, Complex::ZERO);
            pseudo_vie2(&mut x, &z, &z, &s.g0, &Integrator::new(k), BETA, 0.01);
            let mut dmax = 0.0f64;
            for t in MATSUBARA..=11 {
                dmax = dmax.max(pseudo_distance_at(&x, &s.g0, t));
            }
            assert!(dmax < 1e-13, "sign={:?} k={k} dmax {dmax}", sign);
        }
    }
}

#[test]
fn pseudo_vie2_discrete_residuals() {
    for k in [1i64, 3, 5] {
        for d in [1usize, 2] {
            let s = pv_setup(Sign::Fermi, k, 14, 41, 0.01, d);
            let ik = Integrator::new(k);
            let (nt, ntau) = (13i64, 40i64);
            let mut g = PseudoGreen::new(&s.cc, Sign::Fermi, Complex::ZERO);
            pseudo_vie2(&mut g, &s.f, &s.fcc, &s.g0, &ik, BETA, 0.01);
            // R = G + F * G - Q
            let mut r = PseudoGreen::new(&s.cc, Sign::Fermi, Complex::ZERO);
            pseudo_convolution(&mut r, &s.f, &s.fcc, None, &g, &g, &ik, BETA, 0.01);
            pseudo_incr(&mut r, &g, Complex::ONE);
            pseudo_incr(&mut r, &s.g0, -Complex::ONE);
            // Rc = G + G * Fcc - Q
            let mut rc = PseudoGreen::new(&s.cc, Sign::Fermi, Complex::ZERO);
            pseudo_convolution(&mut rc, &g, &g, None, &s.fcc, &s.f, &ik, BETA, 0.01);
            pseudo_incr(&mut rc, &g, Complex::ONE);
            pseudo_incr(&mut rc, &s.g0, -Complex::ONE);
            //
            for m in 0..=ntau {
                assert!(norm_blk(r.mat.block(m as usize)) < 1e-10, "k={k} mat m={m}");
            }
            for n in 0..=nt {
                for j in 0..=n {
                    assert!(norm_blk(r.ret.raw_block_at(n, j)) < 1e-10, "k={k} ret {n},{j}");
                }
            }
            // left-mixing: start-up (my n <= k) solves the conjugate form,
            // time stepping (my n >= k+1) solves the F form
            for n in 0..=k {
                for m in 0..=ntau {
                    assert!(norm_blk(rc.lmix.raw_block_at(n, m)) < 1e-10, "k={k} lmix-start {n}");
                }
            }
            for n in (k + 1)..=nt {
                for m in 0..=ntau {
                    assert!(norm_blk(r.lmix.raw_block_at(n, m)) < 1e-10, "k={k} lmix-step {n}");
                }
            }
            // lesser: start-up solves the F form, time stepping conjugate
            for n in 0..=k {
                for j in 0..=n {
                    assert!(norm_blk(r.less.raw_block_at(j, n)) < 1e-10, "k={k} less-start {n}");
                }
            }
            for n in (k + 1)..=nt {
                for j in 0..=n {
                    assert!(norm_blk(rc.less.raw_block_at(j, n)) < 1e-10, "k={k} less-step {n}");
                }
            }
            if k == 1 {
                // the two forms differ by the discretization error
                let mut e1 = 0.0f64;
                for m in 0..=ntau {
                    e1 = e1.max(norm_blk(r.lmix.raw_block_at(k, m)));
                }
                let mut e2 = 0.0f64;
                for m in 0..=ntau {
                    e2 = e2.max(norm_blk(rc.lmix.raw_block_at(nt, m)));
                }
                assert!(e1 > 1e-10);
                assert!(e2 > 1e-10);
            }
        }
    }
}

#[test]
fn pseudo_vie2_consistency_with_dyson() {
    for k in [2i64, 3] {
        let mut err = Vec::new();
        for (nt_pts, ntau_pts, h) in [(14i64, 41i64, 0.01f64), (27, 81, 0.005), (53, 161, 0.0025)] {
            let s = pv_setup(Sign::Fermi, k, nt_pts, ntau_pts, h, 2);
            let ik = Integrator::new(k);
            let nt = nt_pts - 1;
            let ntau = ntau_pts - 1;
            let mut y = PseudoGreen::new(&s.cc, Sign::Fermi, Complex::ZERO);
            pseudo_vie2(&mut y, &s.f, &s.fcc, &s.g0, &ik, BETA, h);
            let mut g = PseudoGreen::new(&s.cc, Sign::Fermi, Complex::ZERO);
            pseudo_dyson(&mut g, PV_L0, &s.hf, &s.sigma, &ik, BETA, h);
            // the same physical time (the last row)
            let mut e = 0.0f64;
            for m in 0..=ntau {
                e = e
                    .max(norm_blk(&y.mat.block(m as usize).iter().zip(g.mat.block(m as usize)).map(|(a, b)| *a - *b).collect::<Vec<_>>()))
                    .max(norm_blk(&y.lmix.raw_block_at(nt, m).iter().zip(g.lmix.raw_block_at(nt, m)).map(|(a, b)| *a - *b).collect::<Vec<_>>()));
            }
            for j in 0..=nt {
                e = e
                    .max(norm_blk(&y.ret.raw_block_at(nt, j).iter().zip(g.ret.raw_block_at(nt, j)).map(|(a, b)| *a - *b).collect::<Vec<_>>()))
                    .max(norm_blk(&y.less.raw_block_at(j, nt).iter().zip(g.less.raw_block_at(j, nt)).map(|(a, b)| *a - *b).collect::<Vec<_>>()));
            }
            err.push(e);
        }
        assert!(err[0] < 2e-3, "k={k} err0 {0}", err[0]);
        assert!(err[1] < err[0] / 2f64.powi(k as i32), "k={k}");
        assert!(err[2] < err[1] / 2f64.powi(k as i32), "k={k}");
    }
}

#[test]
fn pseudo_vie2_q_unchanged_stepwise_overall() {
    let (k, h) = (4i64, 0.01f64);
    let s = pv_setup(Sign::Bose, k, 14, 31, h, 2);
    let ik = Integrator::new(k);
    let nt = 13i64;
    let mut q = s.g0.clone();
    let mut g = PseudoGreen::new(&s.cc, Sign::Bose, Complex::ZERO);
    pseudo_vie2(&mut g, &s.f, &s.fcc, &q, &ik, BETA, h);
    assert!(q.mat.raw() == s.g0.mat.raw() && q.ret.raw() == s.g0.ret.raw());
    let mut x = PseudoGreen::new(&s.cc, Sign::Bose, Complex::ZERO);
    pseudo_vie2_mat(&mut x, &s.f, &s.fcc, &q, BETA, &ik);
    pseudo_vie2_start(&mut x, &s.f, &s.fcc, &q, &ik, BETA, h);
    // the start-up results at my step k are kept
    let xs = pseudo_timestep(&x, k);
    for tstp in (k + 1)..=nt {
        pseudo_vie2_timestep(tstp, &mut x, &s.f, &s.fcc, &q, &ik, BETA, h);
    }
    assert!(q.mat.raw() == s.g0.mat.raw() && q.ret.raw() == s.g0.ret.raw());
    // the start-up results at my step k are kept (the stored blocks of
    // the extracted time step match the driver's)
    for j in 0..=k {
        assert_eq!(x.ret.raw_block_at(k, j), xs.ret.raw_block_at(k, j));
    }
    for m in 0..=30i64 {
        assert_eq!(x.lmix.raw_block_at(k, m), xs.lmix.raw_block_at(0, m));
    }
    for t in MATSUBARA..=nt {
        assert_eq!(pseudo_distance_at(&g, &x, t), 0.0);
    }
    // repeated time stepping gives the same result
    pseudo_vie2_timestep(nt, &mut x, &s.f, &s.fcc, &q, &ik, BETA, h);
    assert_eq!(pseudo_distance_at(&g, &x, nt), 0.0);
    // the start-up time step is rejected
    assert!(panics(|| {
        let mut xx = x.clone();
        pseudo_vie2_timestep(k, &mut xx, &s.f, &s.fcc, &q, &ik, BETA, h);
    }));
}

#[test]
fn pseudo_vie2_argument_checks() {
    let cc = Contour::new(9, 30, 2, 2, 0.1, BETA);
    let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let f_bose = PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO);
    let zero = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    assert!(panics(|| {
        pseudo_vie2(&mut g, &f_bose, &zero, &zero, &Integrator::new(3), BETA, 0.01);
    }));
    assert!(panics(|| {
        pseudo_vie2(&mut g, &zero, &zero, &zero, &Integrator::new(6), BETA, 0.01);
    }));
    // a different imaginary-time grid is rejected
    let cc2 = Contour::new(9, 20, 2, 2, 0.1, BETA);
    let f2 = PseudoGreen::new(&cc2, Sign::Fermi, Complex::ZERO);
    assert!(panics(|| {
        pseudo_vie2(&mut g, &f2, &zero, &zero, &Integrator::new(3), BETA, 0.01);
    }));
}
