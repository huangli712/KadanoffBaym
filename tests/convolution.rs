// Project : KadanoffBaym
// File    : tests/convolution.rs
// Purpose : port of julia/test/t_convolution.jl

mod common;

use common::prelude::*;
use kadanoffbaym::convolution::{convolution, convolution_ab};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::basics::timestep_at;
use kadanoffbaym::weights::Integrator;

const WA: f64 = 1.123;
const WB: f64 = 0.345;

fn ham(w: f64) -> DenseMat {
    let mut h = DenseMat::zeros(1);
    h.set(0, 0, Complex::new(w, 0.0));
    h
}

fn ham2() -> (DenseMat, DenseMat) {
    let mut h1 = DenseMat::zeros(2);
    h1.set(0, 0, Complex::new(1.123, 0.0));
    h1.set(0, 1, Complex::new(0.1, 0.0));
    h1.set(1, 0, Complex::new(0.1, 0.0));
    h1.set(1, 1, Complex::new(0.567, 0.0));
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, Complex::new(0.345, 0.0));
    h2.set(0, 1, Complex::new(0.2, 0.0));
    h2.set(1, 0, Complex::new(0.2, 0.0));
    h2.set(1, 1, Complex::new(0.876, 0.0));
    (h1, h2)
}

/// Exact solution (resolvent identity): G1 * G2 = (G1 - G2) / (wa - wb).
fn exact_solution(g1: &ContourGreen, g2: &ContourGreen) -> ContourGreen {
    let fac = 1.0 / (WA - WB);
    let mut g3 = g1.clone();
    let w = Complex::new(-1.0, 0.0);
    // Matsubara + all real rows
    let f2 = timestep_at(g2, 0);
    g3.incr_by_step(&f2, MATSUBARA, w);
    g3.smul_at(MATSUBARA, Complex::new(fac, 0.0));
    for tstp in 0..=(g3.nt as i64) {
        let f2 = timestep_at(g2, tstp);
        g3.incr_by_step(&f2, tstp, w);
        g3.smul_at(tstp, Complex::new(fac, 0.0));
    }
    g3
}

fn total_distance(a: &ContourGreen, b: &ContourGreen) -> f64 {
    let mut err = 0.0;
    err += a.distance_at(b, MATSUBARA);
    for tstp in 0..=(a.nt as i64) {
        err += a.distance_at(b, tstp);
    }
    err
}

fn run_scalar_case(
    nt: i64,
    ntau: i64,
    tmax: f64,
    beta: f64,
    dt: f64,
    sign: Sign,
    order: i64,
    eps: f64,
) -> f64 {
    let c = Contour::new(nt, ntau, 1, 1, tmax, beta);
    let mut g1 = ContourGreen::new(&c, sign, Complex::ZERO);
    let mut g2 = ContourGreen::new(&c, sign, Complex::ZERO);
    from_ham(&mut g1, 0.0, &ham(WA), beta, dt);
    from_ham(&mut g2, 0.0, &ham(WB), beta, dt);
    let g3 = exact_solution(&g1, &g2);
    let mut g4 = ContourGreen::new(&c, sign, Complex::ZERO);
    let integ = Integrator::new(order);
    convolution_ab(&mut g4, &g1, &g2, &integ, beta, dt);
    let err = total_distance(&g4, &g3);
    assert!(err < eps, "err = {err}");
    err
}

#[test]
fn scalar_bose_exact() {
    run_scalar_case(14, 500, 0.08, 5.0, 0.01, Sign::Bose, 5, 1e-6);
}

#[test]
fn scalar_fermi_exact() {
    run_scalar_case(14, 500, 0.08, 5.0, 0.01, Sign::Fermi, 5, 1e-6);
}

#[test]
fn scalar_bose_convergence() {
    // quick profile: n = 1..4; release adds n = 5..7
    let nmax = if common::is_release() { 7 } else { 4 };
    for n in 1..=nmax {
        let ntime = (1 << (n - 1)) * 10 + 1;
        let (tmax, beta) = (2.0, 0.1);
        let nt = ntime - 1;
        let c = Contour::new(nt, 500, 1, 1, tmax, beta);
        let dt = c.dt;
        let mut g1 = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        from_ham(&mut g1, 0.0, &ham(WA), beta, dt);
        from_ham(&mut g2, 0.0, &ham(WB), beta, dt);
        let g3 = exact_solution(&g1, &g2);
        let mut g4 = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        let integ = Integrator::new(5);
        convolution_ab(&mut g4, &g1, &g2, &integ, beta, dt);
        let err = total_distance(&g4, &g3);
        if n >= 4 {
            assert!(err < 1e-6, "n = {n}: err = {err}");
        }
    }
}

#[test]
fn scalar_bose_convergence_release_only_note() {
    if !common::is_release() {
        // the release-only tail (n = 5..7) is exercised by
        // scalar_bose_convergence under the release profile
    }
}

/// 2x2 element-wise decomposition against the direct convolution.
#[test]
fn matrix_elements_bose() {
    matrix_elements_case(Sign::Bose);
}

#[test]
fn matrix_elements_fermi() {
    matrix_elements_case(Sign::Fermi);
}

fn matrix_elements_case(sign: Sign) {
    matrix_elements_grid(sign, 10, 400, 0.2, 0.1, 0.02);
}

fn matrix_elements_grid(sign: Sign, nt: i64, ntau: i64, tmax: f64, beta: f64, dt: f64) {
    let c = Contour::new(nt, ntau, 2, 2, tmax, beta);
    let cs = Contour::new(nt, ntau, 1, 1, tmax, beta);
    let (h1, h2) = ham2();
    let mut g1 = ContourGreen::new(&c, sign, Complex::ZERO);
    let mut g2 = ContourGreen::new(&c, sign, Complex::ZERO);
    from_ham(&mut g1, 0.0, &h1, beta, dt);
    from_ham(&mut g2, 0.0, &h2, beta, dt);
    // scalar projections (a, b) -> 1x1 greens
    let proj = |g: &ContourGreen, a: i64, b: i64| -> ContourGreen {
        let mut out = ContourGreen::new(&cs, sign, Complex::ZERO);
        for m in 0..=ntau as i64 {
            let v = g.get_mat(m, a as usize, b as usize);
            out.set_mat(m, 0, 0, v);
        }
        for i in 0..=nt {
            for j in 0..=i {
                let v = g.get_ret(i, j, a as usize, b as usize);
                out.set_ret(i, j, 0, 0, v);
            }
            for j in i..=nt {
                let v = g.get_less(i, j, a as usize, b as usize);
                out.set_less(i, j, 0, 0, v);
            }
            for m in 0..=ntau as i64 {
                let v = g.get_lmix(i, m, a as usize, b as usize);
                out.set_lmix(i, m, 0, 0, v);
            }
        }
        out
    };
    let integ = Integrator::new(5);
    // reference element results: C[a,b] = sum_c conv(G1[a,c], G2[c,b])
    let mut g3 = ContourGreen::new(&c, sign, Complex::ZERO);
    for (a, b) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)] {
        let mut acc = ContourGreen::new(&cs, sign, Complex::ZERO);
        for cc in 0..2i64 {
            let p1 = proj(&g1, a, cc);
            let p2 = proj(&g2, cc, b);
            let mut tmp = ContourGreen::new(&cs, sign, Complex::ZERO);
            convolution_ab(&mut tmp, &p1, &p2, &integ, beta, dt);
            // acc += tmp per step
            let f = timestep_at(&tmp, 0);
            acc.incr_by_step(&f, MATSUBARA, Complex::ONE);
            for t in 0..=nt {
                let f = timestep_at(&tmp, t);
                acc.incr_by_step(&f, t, Complex::ONE);
            }
        }
        // write back into g3
        for m in 0..=ntau as i64 {
            g3.set_mat(m, a as usize, b as usize, acc.get_mat(m, 0, 0));
        }
        for i in 0..=nt {
            for j in 0..=i {
                g3.set_ret(i, j, a as usize, b as usize, acc.get_ret(i, j, 0, 0));
            }
            for j in i..=nt {
                g3.set_less(i, j, a as usize, b as usize, acc.get_less(i, j, 0, 0));
            }
            for m in 0..=ntau as i64 {
                g3.set_lmix(i, m, a as usize, b as usize, acc.get_lmix(i, m, 0, 0));
            }
        }
    }
    // direct 2x2 convolution
    let mut g4 = ContourGreen::new(&c, sign, Complex::ZERO);
    convolution_ab(&mut g4, &g1, &g2, &integ, beta, dt);
    let err = total_distance(&g4, &g3);
    assert!(err < 1e-6, "err = {err}");
}

#[test]
fn matrix_elements_bose_convergence() {
    let imax = if common::is_release() { 4 } else { 2 };
    for i in 1..=imax {
        let ntime = (1 << (i - 1)) * 10 + 1;
        let nt = ntime - 1;
        let dt = 2.0 / nt as f64;
        matrix_elements_grid(Sign::Bose, nt, 400, 2.0, 0.1, dt);
    }
}

/// Type-2: f = I equals Type-1, f = alpha*I scales it.
#[test]
fn type2_identity_and_scalar_function() {
    let (nt, ntau) = (11i64, 40i64);
    let (tmax, beta) = (0.11, 2.0);
    let dt = tmax / nt as f64;
    let mu = 0.1;
    let alpha = Complex::new(0.7, -0.4);
    let tol = 1e-12;
    let mut h1 = DenseMat::zeros(2);
    h1.set(0, 0, Complex::new(0.5, 0.0));
    h1.set(0, 1, Complex::new(0.2, 0.1));
    h1.set(1, 0, Complex::new(0.2, -0.1));
    h1.set(1, 1, Complex::new(-0.3, 0.0));
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, Complex::new(-0.4, 0.0));
    h2.set(0, 1, Complex::new(0.1, -0.3));
    h2.set(1, 0, Complex::new(0.1, 0.3));
    h2.set(1, 1, Complex::new(0.6, 0.0));
    let ident = |scale: Complex| -> TimeFunction {
        let mut f = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        for t in -1..=nt {
            f.set(t, 0, 0, scale);
            f.set(t, 1, 1, scale);
        }
        f
    };
    let compare = |a: &ContourGreen, b: &ContourGreen, x: Complex| -> bool {
        let mut ok = true;
        for m in 0..=ntau as i64 {
            for q in 0..4usize {
                let va = a.get_mat(m, q / 2, q % 2);
                let vb = b.get_mat(m, q / 2, q % 2);
                ok &= (va - x * vb).abs() <= tol * (1.0 + vb.abs());
            }
        }
        for i in 0..=nt {
            for j in 0..=i {
                for q in 0..4usize {
                    let va = a.get_ret(i, j, q / 2, q % 2);
                    let vb = b.get_ret(i, j, q / 2, q % 2);
                    ok &= (va - x * vb).abs() <= tol * (1.0 + vb.abs());
                }
            }
            for j in i..=nt {
                for q in 0..4usize {
                    let va = a.get_less(i, j, q / 2, q % 2);
                    let vb = b.get_less(i, j, q / 2, q % 2);
                    ok &= (va - x * vb).abs() <= tol * (1.0 + vb.abs());
                }
            }
            for m in 0..=ntau as i64 {
                for q in 0..4usize {
                    let va = a.get_lmix(i, m, q / 2, q % 2);
                    let vb = b.get_lmix(i, m, q / 2, q % 2);
                    ok &= (va - x * vb).abs() <= tol * (1.0 + vb.abs());
                }
            }
        }
        ok
    };
    for (sign, order) in [(Sign::Fermi, 1i64), (Sign::Bose, 5i64)] {
        let c = Contour::new(nt, ntau, 2, 2, tmax, beta);
        let mut a = ContourGreen::new(&c, sign, Complex::ZERO);
        let mut b = ContourGreen::new(&c, sign, Complex::ZERO);
        from_ham(&mut a, mu, &h1, beta, dt);
        from_ham(&mut b, mu, &h2, beta, dt);
        // reference Type-1
        let mut c1 = ContourGreen::new(&c, sign, Complex::ZERO);
        let integ = Integrator::new(order);
        convolution_ab(&mut c1, &a, &b, &integ, beta, dt);
        // f = I
        let f = ident(Complex::ONE);
        let mut c2 = ContourGreen::new(&c, sign, Complex::ZERO);
        convolution(&mut c2, &a, &a, Some(&f), &b, &b, &integ, beta, dt);
        assert!(compare(&c2, &c1, Complex::ONE), "f = I, sign {sign:?}");
        // f = alpha I
        let g = ident(alpha);
        let mut c3 = ContourGreen::new(&c, sign, Complex::ZERO);
        convolution(&mut c3, &a, &a, Some(&g), &b, &b, &integ, beta, dt);
        assert!(compare(&c3, &c1, alpha), "f = alpha I, sign {sign:?}");
    }
}

