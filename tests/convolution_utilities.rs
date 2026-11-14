// Project : KadanoffBaym
// File    : tests/convolution_utilities.rs
// Purpose : port of julia/test/t_convolution_utilities.jl (adapted:
//           deterministic RNG; start-up scratch buffer deferred to the
//           solver phase; allocation gate is Julia-specific)

mod common;

use common::prelude::*;
use kadanoffbaym::convolution::density_matrix::convolution_les_timediag_herm;
use kadanoffbaym::convolution::incr::{incr_convolution, incr_convolution_step};
use kadanoffbaym::convolution::{
    convolution, incr_convolution_less_adv, incr_convolution_les, incr_convolution_lmix,
    incr_convolution_lmix_rmix, incr_convolution_mat, incr_convolution_ret,
    incr_convolution_ret_less,
};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::weights::Integrator;
use kadanoffbaym::types::indexing::timestep_at;

const NTIME: i64 = 9;
const NTAU: i64 = 13;
const BETA: f64 = 2.0;
const HSTEP: f64 = 0.05;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_c64(state: &mut u64) -> Complex {
    let re = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(re, im)
}

/// Draw (deg+1)^2 random coefficients.
fn cu_coefs(state: &mut u64, deg: usize) -> Vec<Vec<Complex>> {
    let mut c = vec![vec![Complex::ZERO; deg + 1]; deg + 1];
    for row in c.iter_mut() {
        for v in row.iter_mut() {
            *v = rand_c64(state);
        }
    }
    c
}

/// Smooth two-time function with random matrix coefficients.
fn cu_poly(c: Vec<Vec<Complex>>) -> impl Fn(f64, f64) -> Complex {
    let deg = c.len() - 1;
    move |t: f64, s: f64| {
        let mut acc = Complex::ZERO;
        for a in 0..=deg {
            for b in 0..=deg {
                if a + b <= deg {
                    acc += c[a][b] * Complex::new(t.powi(a as i32) * s.powi(b as i32), 0.0);
                }
            }
        }
        acc
    }
}

fn poly_block(f: &impl Fn(f64, f64) -> Complex, t: f64, s: f64) -> [Complex; 4] {
    [
        f(t, s),
        f(t + 0.31, s - 0.17),
        f(t - 0.11, s + 0.23),
        f(t * 0.7, s + 0.05),
    ]
}

fn elem(blk: &[Complex], d: usize, a: usize, b: usize) -> Complex {
    blk[a * d + b]
}

/// A general (non-hermitian) pair (A, Acc) whose reconstructed elements
/// are smooth continuations (blueprint `cu_make`).
fn cu_make(c: &Contour, sign: Sign, state: &mut u64) -> (ContourGreen, ContourGreen) {
    let d = c.size1 as usize;
    let cr = cu_coefs(state, 2);
    let cl = cu_coefs(state, 2);
    let cm = cu_coefs(state, 2);
    let cc2 = cu_coefs(state, 2);
    let cx = cu_coefs(state, 2);
    let fr = cu_poly(cr);
    let fl = cu_poly(cl);
    let fm = cu_poly(cm);
    let fc = cu_poly(cc2);
    let fx = cu_poly(cx);
    let hv = BETA / c.ntau as f64;
    let mut a = ContourGreen::new(c, sign, Complex::ZERO);
    let mut acc = ContourGreen::new(c, sign, Complex::ZERO);
    for i in 0..=(c.nt as i64) {
        for j in 0..=i {
            let v = poly_block(&fr, i as f64 * HSTEP, j as f64 * HSTEP);
            let w = poly_block(&fr, j as f64 * HSTEP, i as f64 * HSTEP);
            for q in 0..(d * d) {
                a.set_ret(i, j, q / d, q % d, elem(&v, d, q / d, q % d));
                // dagger partner: Acc[a][b] = -conj(Fr(t_j, t_i)[b][a])
                acc.set_ret(i, j, q / d, q % d, -elem(&w, d, q % d, q / d).conj());
            }
        }
    }
    for j in 0..=(c.nt as i64) {
        for i in 0..=j {
            let v = poly_block(&fl, i as f64 * HSTEP, j as f64 * HSTEP);
            let w = poly_block(&fl, j as f64 * HSTEP, i as f64 * HSTEP);
            for q in 0..(d * d) {
                a.set_less(i, j, q / d, q % d, elem(&v, d, q / d, q % d));
                // dagger partner: Acc[a][b] = -conj(Fl(t_j, t_i)[b][a])
                acc.set_less(i, j, q / d, q % d, -elem(&w, d, q % d, q / d).conj());
            }
        }
    }
    for i in 0..=(c.nt as i64) {
        for m in 0..=(c.ntau as i64) {
            let v = poly_block(&fm, i as f64 * HSTEP, m as f64 * hv);
            let w = poly_block(&fc, i as f64 * HSTEP, m as f64 * hv);
            for q in 0..(d * d) {
                a.set_lmix(i, m, q / d, q % d, elem(&v, d, q / d, q % d));
                acc.set_lmix(i, m, q / d, q % d, elem(&w, d, q / d, q % d));
            }
        }
    }
    for m in 0..=(c.ntau as i64) {
        let v = poly_block(&fx, m as f64 * hv, 0.0);
        for q in 0..(d * d) {
            let val = elem(&v, d, q / d, q % d);
            a.set_mat(m, q / d, q % d, val);
            acc.set_mat(m, q / d, q % d, val);
        }
    }
    (a, acc)
}

fn cu_random(c: &Contour, sign: Sign, state: &mut u64) -> ContourGreen {
    let d = c.size1 as usize;
    let mut g = ContourGreen::new(c, sign, Complex::ZERO);
    for m in 0..=(c.ntau as i64) {
        for a in 0..d {
            for b in 0..d {
                g.set_mat(m, a, b, rand_c64(state));
            }
        }
    }
    for i in 0..=(c.nt as i64) {
        for j in 0..=i {
            for a in 0..d {
                for b in 0..d {
                    g.set_ret(i, j, a, b, rand_c64(state));
                }
            }
        }
        for m in 0..=(c.ntau as i64) {
            for a in 0..d {
                for b in 0..d {
                    g.set_lmix(i, m, a, b, rand_c64(state));
                }
            }
        }
        for j in i..=(c.nt as i64) {
            for a in 0..d {
                for b in 0..d {
                    g.set_less(i, j, a, b, rand_c64(state));
                }
            }
        }
    }
    g
}

fn cu_func(c: &Contour, state: &mut u64) -> TimeFunction {
    let d = c.size1 as usize;
    let mut m0 = [[Complex::ZERO; 2]; 2];
    let mut m1 = [[Complex::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m0[i][j] = rand_c64(state);
            m1[i][j] = rand_c64(state);
        }
    }
    let mut f = TimeFunction::new(c.nt as usize, d, d, Complex::ZERO);
    let matsubara_slot = kadanoffbaym::types::MATSUBARA;
    for a in 0..d {
        for b in 0..d {
            f.set(
                matsubara_slot,
                a,
                b,
                m0[a.min(1)][b.min(1)] + m1[a.min(1)][b.min(1)].scale(0.2),
            );
        }
    }
    for i in 0..=(c.nt as i64) {
        let s = (1.3 * i as f64 * HSTEP).sin();
        for a in 0..d {
            for b in 0..d {
                f.set(
                    i,
                    a,
                    b,
                    m0[a.min(1)][b.min(1)] + m1[a.min(1)][b.min(1)].scale(s),
                );
            }
        }
    }
    f
}

fn cu_tol(g: &ContourGreen, tstp: i64) -> f64 {
    1e-12 * 1.0f64.max(g.distance_at(&zero_green(g), tstp))
}

fn zero_green(g: &ContourGreen) -> ContourGreen {
    ContourGreen::new(
        &Contour::new(g.nt as i64, g.ntau as i64, g.size1 as i64, g.size2 as i64, 1.0, 1.0),
        g.sign,
        Complex::ZERO,
    )
}

/// distance of a timestep slice from the green's row at `tstp`
/// (blueprint `distance(x::𝒻, G, tstp)`).
fn slice_distance(exp: &ContourGreen, tstp: i64, x: &kadanoffbaym::types::green::GreenTimeStep) -> f64 {
    let mut err = 0.0f64;
    if tstp == kadanoffbaym::types::MATSUBARA {
        for m in 0..=(exp.ntau as i64) {
            for q in 0..(exp.size1 * exp.size2) {
                err = err.max((x.mat.block(m as usize)[q] - exp.mat.block(m as usize)[q]).abs());
            }
        }
    } else {
        for j in 0..=tstp {
            for q in 0..(exp.size1 * exp.size2) {
                err = err.max((x.ret.raw_block_at(tstp, j)[q] - exp.ret.raw_block_at(tstp, j)[q]).abs());
                err = err.max((x.less.raw_block_at(j, tstp)[q] - exp.less.raw_block_at(j, tstp)[q]).abs());
            }
        }
        for m in 0..=(exp.ntau as i64) {
            for q in 0..(exp.size1 * exp.size2) {
                err = err.max((x.lmix.raw_block_at(0, m)[q] - exp.lmix.raw_block_at(tstp, m)[q]).abs());
            }
        }
    }
    err
}

const STEPS: i64 = 1 + NTIME; // MATSUBARA + real steps 0..=NTIME-1

fn all_steps() -> impl Iterator<Item = i64> {
    let n = NTIME - 1;
    std::iter::once(kadanoffbaym::types::MATSUBARA).chain(0..=n)
}

#[test]
fn incr_convolution_equals_c0_plus_alpha_conv() {
    let korders: &[i64] = if common::is_release() {
        &[1, 2, 3, 4, 5]
    } else {
        &[1, 3, 5]
    };
    let mut rng = 0x5A17C0FFEEu64;
    for sign in [Sign::Fermi, Sign::Bose] {
        for dims in [(1usize, 1usize), (2, 2)] {
            for k in korders {
                let c = Contour::new(
                    NTIME - 1,
                    NTAU,
                    dims.0 as i64,
                    dims.1 as i64,
                    (NTIME - 1) as f64 * HSTEP,
                    BETA,
                );
                let integ = Integrator::new(*k);
                let (a, acc) = cu_make(&c, sign, &mut rng);
                let (b, bcc) = cu_make(&c, sign, &mut rng);
                let f = cu_func(&c, &mut rng);
                let c0 = cu_random(&c, sign, &mut rng);
                for (use_f, alpha) in [
                    (false, Complex::ONE),
                    (true, Complex::new(0.3, -0.7)),
                    (true, Complex::ZERO),
                ] {
                    let fx = if use_f { Some(&f) } else { None };
                    let mut cref = ContourGreen::new(&c, sign, Complex::ZERO);
                    convolution(&mut cref, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP);
                    let mut exp = c0.clone();
                    for t in all_steps() {
                        let row = timestep_at(
                            &cref,
                            if t == kadanoffbaym::types::MATSUBARA {
                                0
                            } else {
                                t
                            },
                        );
                        exp.incr_by_step(&row, t, alpha);
                    }
                    let mut x = c0.clone();
                    for t in all_steps() {
                        incr_convolution(
                            t, &mut x, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP, alpha,
                        );
                    }
                    for t in all_steps() {
                        let d = x.distance_at(&exp, t);
                        assert!(d <= cu_tol(&exp, t), "t {t}: {d} vs tol");
                    }
                    if alpha == Complex::ZERO {
                        assert!(x.equal(&c0), "alpha = 0 must not change the target");
                    }
                    // slice target: only the given step is touched
                    for t in all_steps() {
                        let mut sl = timestep_at(
                            &c0,
                            if t == kadanoffbaym::types::MATSUBARA {
                                0
                            } else {
                                t
                            },
                        );
                        incr_convolution_step(
                            t, &mut sl, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP, alpha,
                        );
                        let d = slice_distance(&exp, t, &sl);
                        assert!(d <= cu_tol(&exp, t), "slice t {t}: {d}");
                        let mut y = c0.clone();
                        incr_convolution(
                            t, &mut y, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP, alpha,
                        );
                        for t2 in all_steps() {
                            if t2 == t {
                                continue;
                            }
                            assert_eq!(y.distance_at(&c0, t2), 0.0, "t2 {t2}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn component_entries() {
    let korders: &[i64] = if common::is_release() {
        &[1, 2, 3, 4, 5]
    } else {
        &[1, 3, 5]
    };
    let mut rng = 0xC0FFEE5A17u64;
    let alpha = Complex::new(0.6, 0.2);
    for sign in [Sign::Fermi, Sign::Bose] {
        for k in korders {
            let c = Contour::new(NTIME - 1, NTAU, 2, 2, (NTIME - 1) as f64 * HSTEP, BETA);
            let integ = Integrator::new(*k);
            let (a, acc) = cu_make(&c, sign, &mut rng);
            let (b, bcc) = cu_make(&c, sign, &mut rng);
            let f = cu_func(&c, &mut rng);
            let mut cref = ContourGreen::new(&c, sign, Complex::ZERO);
            convolution(&mut cref, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP);
            let c0 = cu_random(&c, sign, &mut rng);
            let mut exp = c0.clone();
            for t in all_steps() {
                let row = timestep_at(
                    &cref,
                    if t == kadanoffbaym::types::MATSUBARA {
                        0
                    } else {
                        t
                    },
                );
                exp.incr_by_step(&row, t, alpha);
            }
            // mat entry
            let mut x = c0.clone();
            incr_convolution_mat(&mut x, &a, Some(&f), &b, &integ, BETA, alpha);
            assert!(x.distance_mat(&exp) <= cu_tol(&exp, kadanoffbaym::types::MATSUBARA));
            for tstp in 0..=(c.nt as i64) {
                let mut x = c0.clone();
                incr_convolution_ret(
                    tstp,
                    &mut x,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    HSTEP,
                    alpha,
                );
                assert!(x.distance_ret(&exp, tstp) <= cu_tol(&exp, tstp));
                assert_eq!(x.distance_lmix(&c0, tstp), 0.0);
                assert_eq!(x.distance_less(&c0, tstp), 0.0);

                let mut x = c0.clone();
                incr_convolution_lmix(
                    tstp,
                    &mut x,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                assert!(x.distance_lmix(&exp, tstp) <= cu_tol(&exp, tstp));
                assert_eq!(x.distance_ret(&c0, tstp), 0.0);
                assert_eq!(x.distance_less(&c0, tstp), 0.0);

                let mut x = c0.clone();
                incr_convolution_les(
                    tstp,
                    &mut x,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                assert!(x.distance_less(&exp, tstp) <= cu_tol(&exp, tstp));
                assert_eq!(x.distance_ret(&c0, tstp), 0.0);
                assert_eq!(x.distance_lmix(&c0, tstp), 0.0);

                // the three partials sum to the full lesser increment
                let mut y = c0.clone();
                incr_convolution_ret_less(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                incr_convolution_less_adv(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                incr_convolution_lmix_rmix(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                assert!(y.equal(&x), "partial sum differs at tstp {tstp}");
            }
        }
    }
}

#[test]
fn incr_les_nessi_diagonal_rule() {
    let korders: &[i64] = if common::is_release() {
        &[1, 2, 3, 4, 5]
    } else {
        &[1, 3, 5]
    };
    let mut rng = 0x1234ABCDu64;
    let alpha = Complex::new(0.6, 0.2);
    for sign in [Sign::Fermi, Sign::Bose] {
        for k in korders {
            let c = Contour::new(NTIME - 1, NTAU, 2, 2, (NTIME - 1) as f64 * HSTEP, BETA);
            let integ = Integrator::new(*k);
            let a = cu_random(&c, sign, &mut rng);
            let acc = cu_random(&c, sign, &mut rng);
            let b = cu_random(&c, sign, &mut rng);
            let bcc = cu_random(&c, sign, &mut rng);
            let f = cu_func(&c, &mut rng);
            for tstp in 1..=(c.nt as i64) {
                let mut x = cu_random(&c, sign, &mut rng);
                let mut y = x.clone();
                incr_convolution_les(
                    tstp,
                    &mut x,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                incr_convolution_ret_less(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                incr_convolution_less_adv(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                incr_convolution_lmix_rmix(
                    tstp,
                    &mut y,
                    &a,
                    &acc,
                    Some(&f),
                    &b,
                    &bcc,
                    &integ,
                    BETA,
                    HSTEP,
                    alpha,
                );
                if tstp <= 2 * k + 2 {
                    let mut dmax = 0.0f64;
                    for m in 0..=tstp {
                        // Delta = alpha h Omega[tstp,m] (-Acc[m,m]^dag - A[m,m]) f[m] Bcc.ret[tstp,m]^dag
                        let w = integ.giw.at(tstp as usize, m as usize);
                        let aadj = acc.less.raw_block_at(m, m);
                        let adir = a.less.raw_block_at(m, m);
                        let fm = f.block(m);
                        let badj = bcc.ret.raw_block_at(tstp, m);
                        // 2x2 matrix product Delta = P * F * Q
                        let mut p = [Complex::ZERO; 4];
                        for i in 0..2 {
                            for j in 0..2 {
                                let mut s = Complex::ZERO;
                                for l in 0..2 {
                                    // (P)[i][j] = sum_l (-Acc^dag - A)[i][l] * f[l][j]
                                    let pv = -(aadj[l * 2 + i].conj()) - adir[i * 2 + l];
                                    s += pv * fm[l * 2 + j];
                                }
                                p[i * 2 + j] = s;
                            }
                        }
                        let mut delta = [Complex::ZERO; 4];
                        for i in 0..2 {
                            for j in 0..2 {
                                let mut s = Complex::ZERO;
                                for l in 0..2 {
                                    s += p[i * 2 + l] * badj[j * 2 + l].conj();
                                }
                                delta[i * 2 + j] = alpha * Complex::new(HSTEP * w, 0.0) * s;
                            }
                        }
                        let blk = y.less.raw_block_at_mut(m, tstp);
                        for q in 0..4 {
                            blk[q] += delta[q];
                        }
                        let mut nrm = 0.0f64;
                        for q in 0..4 {
                            nrm = nrm.max(delta[q].abs());
                        }
                        dmax = dmax.max(nrm);
                    }
                    if tstp >= 2 {
                        assert!(dmax > 1e-6, "correction vanished");
                    }
                    let dd = x.distance_less(&y, tstp);
                    if dd > 1e-13 {
                        eprintln!("FAIL k {k} tstp {tstp} dist {dd:.3e}");
                        for m in 0..=tstp {
                            for q in 0..4 {
                                let xv = x.get_less(m, tstp, q / 2, q % 2);
                                let yv = y.get_less(m, tstp, q / 2, q % 2);
                                if (xv - yv).abs() > 1e-14 {
                                    eprintln!("  m {m} q {q}: x {xv} y {yv}");
                                }
                            }
                        }
                    }
                    assert!(dd <= 1e-13);
                } else {
                    assert_eq!(x.distance_less(&y, tstp), 0.0);
                }
            }
        }
    }
}

#[test]
fn partial_lesser_target_not_input() {
    let mut rng = 0xBADC0DEu64;
    for sign in [Sign::Fermi, Sign::Bose] {
        for k in [1i64, 3, 5] {
            let c = Contour::new(NTIME - 1, NTAU, 2, 2, (NTIME - 1) as f64 * HSTEP, BETA);
            let integ = Integrator::new(k);
            let (a, acc) = cu_make(&c, sign, &mut rng);
            let (b, bcc) = cu_make(&c, sign, &mut rng);
            let f = cu_func(&c, &mut rng);
            for tstp in 1..=(c.nt as i64) {
                for which in 0..3 {
                    let mut x1 = ContourGreen::new(&c, sign, Complex::ZERO);
                    let mut x2 = ContourGreen::new(&c, sign, Complex::new(1e6, 0.0));
                    match which {
                        0 => {
                            incr_convolution_ret_less(
                                tstp,
                                &mut x1,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                            incr_convolution_ret_less(
                                tstp,
                                &mut x2,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                        }
                        1 => {
                            incr_convolution_less_adv(
                                tstp,
                                &mut x1,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                            incr_convolution_less_adv(
                                tstp,
                                &mut x2,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                        }
                        _ => {
                            incr_convolution_lmix_rmix(
                                tstp,
                                &mut x1,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                            incr_convolution_lmix_rmix(
                                tstp,
                                &mut x2,
                                &a,
                                &acc,
                                Some(&f),
                                &b,
                                &bcc,
                                &integ,
                                BETA,
                                HSTEP,
                                Complex::ONE,
                            );
                        }
                    }
                    for j in 1..=tstp {
                        for q in 0..4usize {
                            let d = x2.get_less(j, tstp, q / 2, q % 2)
                                - x1.get_less(j, tstp, q / 2, q % 2);
                            assert!((d - Complex::new(1e6, 0.0)).abs() < 1e-6, "which {which} j {j}");
                        }
                    }
                }
                // the Dyson/VIE2 source does not read G^<(:, tstp)
                let mut srcs = Vec::new();
                for v in [0.0f64, 1e6] {
                    let mut g = b.clone();
                    let mut gcc = bcc.clone();
                    for j in 1..=tstp {
                        for q in 0..4usize {
                            g.set_less(j, tstp, q / 2, q % 2, Complex::new(v, 0.0));
                            gcc.set_less(j, tstp, q / 2, q % 2, Complex::new(v, 0.0));
                        }
                    }
                    let mut x = ContourGreen::new(&c, sign, Complex::ZERO);
                    incr_convolution_less_adv(
                        tstp,
                        &mut x,
                        &a,
                        &acc,
                        Some(&f),
                        &g,
                        &gcc,
                        &integ,
                        BETA,
                        HSTEP,
                        Complex::ONE,
                    );
                    incr_convolution_lmix_rmix(
                        tstp,
                        &mut x,
                        &a,
                        &acc,
                        Some(&f),
                        &g,
                        &gcc,
                        &integ,
                        BETA,
                        HSTEP,
                        Complex::ONE,
                    );
                    srcs.push(x);
                }
                assert_eq!(srcs[0].distance_less(&srcs[1], tstp), 0.0);
            }
        }
    }
}

#[test]
fn les_timediag_and_density_matrix() {
    let mut rng = 0xFEEDFACEu64;
    for sign in [Sign::Fermi, Sign::Bose] {
        for dims in [(1usize, 1usize), (2, 2)] {
            for k in 1..=5i64 {
                let c = Contour::new(
                    NTIME - 1,
                    NTAU,
                    dims.0 as i64,
                    dims.1 as i64,
                    (NTIME - 1) as f64 * HSTEP,
                    BETA,
                );
                let integ = Integrator::new(k);
                let (a, acc) = cu_make(&c, sign, &mut rng);
                let (b, bcc) = cu_make(&c, sign, &mut rng);
                let f = cu_func(&c, &mut rng);
                for use_f in [false, true] {
                    let fx = if use_f { Some(&f) } else { None };
                    let mut cref = ContourGreen::new(&c, sign, Complex::ZERO);
                    convolution(&mut cref, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP);
                    for t in all_steps() {
                        let x = kadanoffbaym::convolution::density_matrix::convolution_les_timediag(
                            t, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP,
                        );
                        let want = if t == kadanoffbaym::types::MATSUBARA {
                            Complex::new(0.0, -1.0) * cref.mat.block(c.ntau as usize)[0]
                        } else {
                            cref.less.raw_block_at(t, t)[0]
                        };
                        assert!(
                            (x.get(0, 0) - want).abs() <= 1e-11 * 1.0f64.max(want.abs()),
                            "t {t} k {k}: {} vs {want}",
                            x.get(0, 0)
                        );
                        let rho =
                            kadanoffbaym::convolution::density_matrix::convolution_density_matrix(
                                t, &a, &acc, fx, &b, &bcc, &integ, BETA, HSTEP,
                            );
                        assert_eq!(rho.get(0, 0), Complex::new(0.0, -1.0) * x.get(0, 0));
                    }
                }
            }
        }
    }
}

#[test]
fn nessi_time_diagonal_convolution() {
    let (nt, ntau) = (7i64, 500i64);
    let (beta, h) = (5.0, 0.01);
    let (wa, wb) = (1.123f64, 0.345);
    let kt = 5i64;
    let c = Contour::new(nt, ntau, 1, 1, nt as f64 * h, beta);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut ha = DenseMat::zeros(1);
    ha.set(0, 0, Complex::new(wa, 0.0));
    let mut hb = DenseMat::zeros(1);
    hb.set(0, 0, Complex::new(wb, 0.0));
    from_ham(&mut a, 0.0, &ha, beta, h);
    from_ham(&mut b, 0.0, &hb, beta, h);
    let mut ab = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    convolution(&mut ab, &a, &a, None, &b, &b, &Integrator::new(kt), beta, h);
    let nmax = ab.nt as i64;
    let steps: Vec<i64> = std::iter::once(kadanoffbaym::types::MATSUBARA).chain(0..=nmax).collect();
    for tstp in steps {
        let rho = kadanoffbaym::convolution::density_matrix::convolution_density_matrix(
            tstp,
            &a,
            &a,
            None,
            &b,
            &b,
            &Integrator::new(kt),
            beta,
            h,
        );
        if tstp == kadanoffbaym::types::MATSUBARA {
            let want = -ab.mat.block(ntau as usize)[0];
            assert!((rho.get(0, 0) - want).abs() < 1e-6);
        } else {
            let want = Complex::new(0.0, 1.0) * rho.get(0, 0);
            assert!(
                (ab.get_less(tstp, tstp, 0, 0) - want).abs() < 1e-6,
                "t {tstp}"
            );
        }
    }
}

#[test]
fn preconditions_bounds() {
    let c = Contour::new(NTIME - 1, NTAU, 2, 2, (NTIME - 1) as f64 * HSTEP, BETA);
    let mut rng = 0x1111u64;
    let (a, acc) = cu_make(&c, Sign::Fermi, &mut rng);
    let integ = Integrator::new(2);
    let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut run = |tstp: i64| {
        incr_convolution(
            tstp,
            &mut x,
            &a,
            &acc,
            None,
            &a,
            &acc,
            &integ,
            BETA,
            HSTEP,
            Complex::ONE,
        );
    };
    // out of range: Julia ntime + 1
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(NTIME + 1))).is_err());
    // ret at MATSUBARA (Julia tstp = 0)
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        incr_convolution_ret(
            kadanoffbaym::types::MATSUBARA,
            &mut x,
            &a,
            &acc,
            None,
            &a,
            &acc,
            &integ,
            HSTEP,
            Complex::ONE,
        )
    }))
    .is_err());
}

#[test]
#[cfg_attr(not(feature = "never"), allow(unused))]
fn release_equal_time_order_gate() {
    if !common::is_release() {
        return;
    }
    let (wa, wb) = (1.123f64, 0.345);
    let beta = 1.0f64;
    let tmax = 0.25;
    let na = |w: f64| 1.0 / ((beta * w).exp() + 1.0);
    let exact = Complex::new(0.0, (na(wa) - na(wb)) / (wa - wb));
    let grids: [[usize; 4]; 5] = [
        [16, 32, 64, 128],
        [8, 16, 32, 64],
        [4, 8, 16, 32],
        [6, 8, 12, 16],
        [6, 8, 10, 12],
    ];
    let eqt_error = |kk: i64, n: usize| -> f64 {
        let h = tmax / n as f64;
        let c = Contour::new(n as i64, n as i64, 1, 1, tmax, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut ha = DenseMat::zeros(1);
        ha.set(0, 0, Complex::new(wa, 0.0));
        let mut hb = DenseMat::zeros(1);
        hb.set(0, 0, Complex::new(wb, 0.0));
        from_ham(&mut a, 0.0, &ha, beta, h);
        from_ham(&mut b, 0.0, &hb, beta, h);
        let x =
            convolution_les_timediag_herm(n as i64, &a, None, &b, &Integrator::new(kk), beta, h);
        (x.get(0, 0) - exact).abs()
    };
    for (k, ns) in grids.iter().enumerate() {
        let kk = k as i64 + 1;
        let errs: Vec<f64> = ns.iter().map(|&n| eqt_error(kk, n)).collect();
        let ns2: Vec<i64> = ns.iter().map(|&n| n as i64).collect();
        let gate = common::check_order_gate(&ns2, &errs, (kk + 1) as f64, 1e-8, 1e-13, 0.8);
        assert!(gate.ok, "k {kk}: observed {} errs {errs:?}", gate.observed);
        if kk >= 3 {
            let errm: Vec<f64> = ns.iter().map(|&n| eqt_error(kk - 2, n)).collect();
            assert!(!common::check_order_gate(&ns2, &errm, (kk + 1) as f64, 1e-8, 1e-13, 0.8).ok);
        }
    }
}

#[test]
fn probe_diagcc_rust() {
    // mirror of the Julia probe: X - Y must equal Delta exactly
    let mut rng = 0xDEADBEEFu64;
    let c = Contour::new(8, 12, 2, 2, 8.0 * 0.05, 2.0);
    let k = 1i64;
    let integ = Integrator::new(k);
    let a = cu_random(&c, Sign::Fermi, &mut rng);
    let acc = cu_random(&c, Sign::Fermi, &mut rng);
    let b = cu_random(&c, Sign::Fermi, &mut rng);
    let bcc = cu_random(&c, Sign::Fermi, &mut rng);
    let f = cu_func(&c, &mut rng);
    let alpha = Complex::new(0.6, 0.2);
    let tstp = 1i64;
    let mut x = cu_random(&c, Sign::Fermi, &mut rng);
    let mut y = x.clone();
    incr_convolution_les(tstp, &mut x, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_ret_less(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_less_adv(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_lmix_rmix(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    for m in 0..=tstp {
        let w = integ.giw.at(tstp as usize, m as usize);
        let aadj = acc.less.raw_block_at(m, m);
        let adir = a.less.raw_block_at(m, m);
        let fm = f.block(m);
        let badj = bcc.ret.raw_block_at(tstp, m);
        let mut p = [Complex::ZERO; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = Complex::ZERO;
                for l in 0..2 {
                    let pv = -(aadj[l * 2 + i].conj()) - adir[i * 2 + l];
                    s += pv * fm[l * 2 + j];
                }
                p[i * 2 + j] = s;
            }
        }
        let mut delta = [Complex::ZERO; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = Complex::ZERO;
                for l in 0..2 {
                    s += p[i * 2 + l] * badj[j * 2 + l].conj();
                }
                delta[i * 2 + j] = alpha * Complex::new(HSTEP * w, 0.0) * s;
            }
        }
        for q in 0..4 {
            let xv = x.get_less(m, tstp, q / 2, q % 2);
            let yv = y.get_less(m, tstp, q / 2, q % 2);
            eprintln!("m {m} q {q}: x-y {} delta {} resid {}", xv - yv, delta[q], (xv - yv) - delta[q]);
        }
    }
}

#[test]
fn probe_diagcc_only_less_adv() {
    use kadanoffbaym::convolution::{incr_convolution_less_adv, incr_convolution_les};
    let mut rng = 0xDEADBEEFu64;
    let c = Contour::new(8, 12, 2, 2, 8.0 * 0.05, 2.0);
    let k = 1i64;
    let integ = Integrator::new(k);
    let a = cu_random(&c, Sign::Fermi, &mut rng);
    let acc = cu_random(&c, Sign::Fermi, &mut rng);
    let b = cu_random(&c, Sign::Fermi, &mut rng);
    let bcc = cu_random(&c, Sign::Fermi, &mut rng);
    let f = cu_func(&c, &mut rng);
    let alpha = Complex::new(0.6, 0.2);
    let tstp = 1i64;
    // ONLY the diagcc branch of less_adv: recompute via the kernel both ways
    let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut y = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    // combined = ret_less + less_adv(diagcc) + lmix_rmix
    incr_convolution_les(tstp, &mut x, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    // partials
    incr_convolution_ret_less(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_less_adv(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_lmix_rmix(tstp, &mut y, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    // isolate: subtract ret_less and lmix_rmix from x by running them on a third copy
    let mut z = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    incr_convolution_ret_less(tstp, &mut z, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    incr_convolution_lmix_rmix(tstp, &mut z, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, HSTEP, alpha);
    // x_adv = x - z (component arithmetic), y_adv = y - z
    for m in 0..=tstp {
        for q in 0..4usize {
            let xv = x.get_less(m, tstp, q / 2, q % 2) - z.get_less(m, tstp, q / 2, q % 2);
            let yv = y.get_less(m, tstp, q / 2, q % 2) - z.get_less(m, tstp, q / 2, q % 2);
            eprintln!("m {m} q {q}: xadv {} yadv {} xadv-yadv {}", xv, yv, xv - yv);
        }
    }
}
