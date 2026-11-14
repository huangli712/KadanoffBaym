// Project : KadanoffBaym
// File    : tests/moving_algebra.rs
// Purpose : blueprint t_moving_algebra.jl: the traits of the moving
//           time steps (copy, incr, smul, multiplication by a moving
//           single-time function), the moving extrapolation and the
//           moving distance
//
// The random data use the deterministic splitmix64 generator (the
// Julia MersenneTwister is not reproduced; the tests are
// self-consistent). The stale-view checks are subsumed by the Rust
// borrow checker (known deviation).

mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::moving::algebra::{
    distance_moving, distance_moving_at, extrapolate_timestep_moving_order, incr_step, lmul_step,
    rmul_step, smul_step,
};
use kadanoffbaym::moving::window::{copy_timestep, forward, init_from_function, init_from_g_herm};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::moving::{ CfMoving, MovingGreen };

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
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

fn rand_c64(state: &mut u64) -> Complex {
    let re = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(re, im)
}

/// Moving Green's function filled with fixed random data at the
/// leading time step `tstp` (blueprint `ma_random`; `tstp` is the KB
/// step).
fn ma_random(tc: i64, d: usize, tstp: i64, seed: u64) -> MovingGreen {
    let mut state = seed;
    let nt = tstp + 2;
    let cont = Contour::new(nt - 1, 2, d as i64, d as i64, (nt - 1) as f64 * 0.1, 1.0);
    let mut f = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    for n in 1..=nt {
        for m in 1..=n {
            for q in 0..(d * d) {
                f.ret.raw_block_at_mut(n - 1, m - 1)[q] = rand_c64(&mut state);
            }
            let l: Vec<Complex> = (0..(d * d)).map(|_| rand_c64(&mut state)).collect();
            let ll = if m == n {
                let mut a = l.clone();
                for i in 0..d {
                    for j in 0..d {
                        a[i * d + j] = (l[i * d + j] - l[j * d + i].conj()) / 2.0;
                    }
                }
                a
            } else {
                l
            };
            f.less.raw_block_at_mut(m - 1, n - 1).copy_from_slice(&ll);
        }
    }
    let mut g = MovingGreen::new(tc, d, Sign::Fermi);
    init_from_g_herm(&mut g, &f, tstp - 1);
    g
}

/// Moving single-time function with fixed random data (blueprint
/// `ma_func`).
fn ma_func(tc: i64, d: usize, tstp: i64, seed: u64) -> CfMoving {
    let mut state = seed;
    let nt = tstp + 2;
    let mut cf = TimeFunction::new((nt - 1) as usize, d, d, Complex::ZERO);
    for i in -1..=(nt - 1) {
        for a in 0..d {
            for b in 0..d {
                cf.set(i, a, b, rand_c64(&mut state));
            }
        }
    }
    let mut f = CfMoving::new(tc, d);
    init_from_function(&mut f, &cf, tstp - 1);
    f
}

fn rowcopy_ret(g: &MovingGreen, delti: i64) -> Vec<Vec<Complex>> {
    (0..=g.tc).map(|j| g.ret.mblock(delti, j).to_vec()).collect()
}

fn rowcopy_less(g: &MovingGreen, delti: i64) -> Vec<Vec<Complex>> {
    (0..=g.tc).map(|j| g.less.mblock(delti, j).to_vec()).collect()
}

#[test]
fn moving_algebra_matrix_elements() {
    for &d in &[1usize, 3] {
        let (tc, tstp) = (5i64, 9i64);
        let mut g = ma_random(tc, d, tstp, 1);
        let x = copy_timestep(&g, 2);
        for j in 0..=tc {
            assert_eq!(g.ret.mblock(2, j), x.ret.mblock(j));
            assert_eq!(g.less.mblock(2, j), x.less.mblock(j));
        }
        // single matrix elements written through the row access
        let mut blk = g.ret.mblock(2, 3).to_vec();
        blk[0 * d + (d - 1)] = c(4.0, -2.0);
        g.ret.mblock_mut(2, 3).copy_from_slice(&blk);
        assert_eq!(g.ret.mblock(2, 3)[0 * d + (d - 1)], c(4.0, -2.0));
        // absolute check: KB (tstp-2, tstp-5) = my (6, 3)
        assert_eq!(g.ret_block(tstp - 3, tstp - 6)[0 * d + (d - 1)], c(4.0, -2.0));
    }
}

#[test]
fn moving_algebra_incr_and_smul() {
    for &d in &[1usize, 2] {
        let (tc, tstp) = (4i64, 8i64);
        let mut g1 = ma_random(tc, d, tstp, 2);
        let mut g2 = ma_random(tc, d, tstp, 3);
        let alpha = c(0.7, -1.3);
        let r1 = rowcopy_ret(&g1, 1);
        let l1 = rowcopy_less(&g1, 1);
        let r2 = rowcopy_ret(&g2, 0);
        let l2 = rowcopy_less(&g2, 0);
        // incr row 1 of g1 by row 0 of g2
        let src = copy_timestep(&g2, 0);
        let mut dst = copy_timestep(&g1, 1);
        incr_step(&mut dst, &src, alpha);
        g1.ret.mrow_mut(1).copy_from_slice(dst.ret.raw());
        g1.less.mrow_mut(1).copy_from_slice(dst.less.raw());
        for j in 0..=tc {
            for q in 0..(d * d) {
                assert!((g1.ret.mblock(1, j)[q] - (r1[j as usize][q] + alpha * r2[j as usize][q])).abs() < 1e-14);
                assert!((g1.less.mblock(1, j)[q] - (l1[j as usize][q] + alpha * l2[j as usize][q])).abs() < 1e-14);
            }
        }
        // other rows are not touched
        let fresh = ma_random(tc, d, tstp, 2);
        assert_eq!(rowcopy_ret(&g1, 0), rowcopy_ret(&fresh, 0));
        // owning time step: x = 2 * r2
        let mut x = copy_timestep(&g2, 0);
        let src2 = copy_timestep(&g2, 0);
        incr_step(&mut x, &src2, Complex::ONE);
        for j in 0..=tc {
            for q in 0..(d * d) {
                assert_eq!(x.ret.mblock(j)[q], 2.0 * r2[j as usize][q]);
            }
        }
        // smul row 0 of g2 by alpha
        let mut y = copy_timestep(&g2, 0);
        smul_step(&mut y, alpha);
        g2.ret.mrow_mut(0).copy_from_slice(y.ret.raw());
        g2.less.mrow_mut(0).copy_from_slice(y.less.raw());
        for j in 0..=tc {
            for q in 0..(d * d) {
                assert!((g2.ret.mblock(0, j)[q] - alpha * r2[j as usize][q]).abs() < 1e-14);
                assert!((g2.less.mblock(0, j)[q] - alpha * l2[j as usize][q]).abs() < 1e-14);
            }
        }
        // alias dst === src: x = 3 * x (adapted through a clone)
        let mut g3 = ma_random(tc, d, tstp, 4);
        let r3 = rowcopy_ret(&g3, 0);
        let x0 = copy_timestep(&g3, 0);
        let mut x3 = copy_timestep(&g3, 0);
        incr_step(&mut x3, &x0, c(2.0, 0.0));
        g3.ret.mrow_mut(0).copy_from_slice(x3.ret.raw());
        for j in 0..=tc {
            for q in 0..(d * d) {
                assert_eq!(g3.ret.mblock(0, j)[q], 3.0 * r3[j as usize][q]);
            }
        }
        // memcpy between rows and zeros
        let mut g4 = ma_random(tc, d, tstp, 5);
        let row0r = g4.ret.mrow(0).to_vec();
        let row0l = g4.less.mrow(0).to_vec();
        g4.ret.mrow_mut(3).copy_from_slice(&row0r);
        g4.less.mrow_mut(3).copy_from_slice(&row0l);
        assert_eq!(rowcopy_ret(&g4, 0), rowcopy_ret(&g4, 3));
        assert_eq!(rowcopy_less(&g4, 0), rowcopy_less(&g4, 3));
        for z in g4.ret.mrow_mut(1) {
            *z = Complex::ZERO;
        }
        for z in g4.less.mrow_mut(1) {
            *z = Complex::ZERO;
        }
        assert!(rowcopy_ret(&g4, 1).iter().flatten().all(|z| *z == Complex::ZERO));
        assert!(rowcopy_less(&g4, 1).iter().flatten().all(|z| *z == Complex::ZERO));
        // whole object copy
        let g5 = g4.clone();
        assert_eq!(g5.tstp, g4.tstp);
        for n in (g4.tstp - g4.tc)..=g4.tstp {
            for m in (n - g4.tc).max(0)..=n {
                assert_eq!(g5.ret_block(n, m), g4.ret_block(n, m));
            }
        }
        // incompatible rows
        let other = ma_random(tc + 1, d, tstp, 6);
        let mut xd = copy_timestep(&g1, 0);
        let xs = copy_timestep(&other, 0);
        assert!(panics(move || incr_step(&mut xd, &xs, Complex::ONE)));
    }
}

#[test]
fn moving_algebra_function_multiplication() {
    for &adj in &[false, true] {
        for &d in &[1usize, 2] {
            let (tc, tstp) = (4i64, 8i64);
            let g = ma_random(tc, d, tstp, 11);
            let f = ma_func(tc, d, tstp, 7);
            let r = rowcopy_ret(&g, 0);
            let l = rowcopy_less(&g, 0);
            let op = |blk: &[Complex]| -> DenseMat {
                let m = DenseMat::from_vec(d, d, blk.to_vec());
                if adj {
                    m.adjoint()
                } else {
                    m
                }
            };
            // left: f(t) G(t, t - j)
            let mut x = copy_timestep(&g, 0);
            lmul_step(&f, &mut x, adj);
            let f0 = op(f.fslot(0));
            for j in 0..=tc {
                let want = f0.mat_mul(&DenseMat::from_vec(d, d, r[j as usize].clone()));
                let got = DenseMat::from_vec(d, d, x.ret.mblock(j).to_vec());
                for q in 0..(d * d) {
                    assert!((got.raw()[q] - want.raw()[q]).abs() < 1e-13, "left ret j={j}");
                }
                let wantl = f0.mat_mul(&DenseMat::from_vec(d, d, l[j as usize].clone()));
                let gotl = DenseMat::from_vec(d, d, x.less.mblock(j).to_vec());
                for q in 0..(d * d) {
                    assert!((gotl.raw()[q] - wantl.raw()[q]).abs() < 1e-13, "left less j={j}");
                }
            }
            // right: G(t, t - j) f(t - j)
            let mut x = copy_timestep(&g, 0);
            rmul_step(&f, &mut x, adj);
            for j in 0..=tc {
                let fj = op(f.fslot(j));
                let want = DenseMat::from_vec(d, d, r[j as usize].clone()).mat_mul(&fj);
                let got = DenseMat::from_vec(d, d, x.ret.mblock(j).to_vec());
                for q in 0..(d * d) {
                    assert!((got.raw()[q] - want.raw()[q]).abs() < 1e-13, "right ret j={j}");
                }
                let wantl = DenseMat::from_vec(d, d, l[j as usize].clone()).mat_mul(&fj);
                let gotl = DenseMat::from_vec(d, d, x.less.mblock(j).to_vec());
                for q in 0..(d * d) {
                    assert!((gotl.raw()[q] - wantl.raw()[q]).abs() < 1e-13, "right less j={j}");
                }
            }
            // the time step must be the leading step of f
            let mut x1 = copy_timestep(&g, 1);
            assert!(panics(|| lmul_step(&f, &mut x1, false)));
            let mut x0 = copy_timestep(&g, 0);
            let fun = CfMoving::new(tc, d);
            assert!(panics(|| lmul_step(&fun, &mut x0, false)));
        }
    }
}

#[test]
fn moving_algebra_ring_wrap_around() {
    // (the stale-view part of the blueprint is subsumed by the Rust
    // borrow checker)
    let (tc, d, tstp) = (3i64, 2usize, 6i64);
    let g = ma_random(tc, d, tstp, 21);
    let mut f = g.clone();
    let mut vals: std::collections::HashMap<i64, Vec<Complex>> = std::collections::HashMap::new();
    for _ in 0..(2 * (tc + 1)) {
        forward(&mut f);
        let n = f.tstp;
        let blk: Vec<Complex> = (0..(d * d))
            .map(|q| {
                if q % (d + 1) == 0 {
                    c(n as f64, -(n as f64))
                } else {
                    c(n as f64, -(n as f64))
                }
            })
            .collect();
        vals.insert(n, blk.clone());
        for j in 0..=tc {
            f.ret.mblock_mut(0, j).copy_from_slice(&blk);
        }
    }
    let n0 = f.tstp;
    for n in (n0 - tc)..=n0 {
        let got = f.ret_block(n, n - 1);
        for q in 0..(d * d) {
            assert_eq!(got[q], vals[&n][q]);
        }
    }
}

#[test]
fn moving_algebra_extrapolation() {
    let (d, h) = (2usize, 0.1f64);
    for k in 0..=5i64 {
        let tc = 2 * k + 3;
        let n0 = tc + 2; // KB
        let nt = n0 + 1; // KB points
        let cont = Contour::new(nt - 1, 2, d as i64, d as i64, (nt - 1) as f64 * h, 1.0);
        let mut f = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
        let p = |t: f64, s: f64| -> Vec<Complex> {
            let mut out = vec![Complex::ZERO; d * d];
            for a in 1..=d {
                for b in 1..=d {
                    out[(a - 1) * d + (b - 1)] = if k == 0 {
                        c(a as f64, b as f64)
                    } else {
                        c(
                            (1.0 + 0.3 * a as f64) * t.powi(k as i32)
                                - 0.5 * b as f64 * s * t.powi((k - 1) as i32)
                                + a as f64,
                            0.2 * s.powi(k as i32) + b as f64 * t.powi(k as i32),
                        )
                    };
                }
            }
            out
        };
        for n in 1..=nt {
            for m in 1..=n {
                let t = (n - 1) as f64 * h;
                let s = (m - 1) as f64 * h;
                f.ret.raw_block_at_mut(n - 1, m - 1).copy_from_slice(&p(t, s));
                let lv = p(s, t);
                let ll = if m == n {
                    let mut a = lv.clone();
                    for i in 0..d {
                        for j in 0..d {
                            a[i * d + j] = (lv[i * d + j] - lv[j * d + i].conj()) / 2.0;
                        }
                    }
                    a
                } else {
                    lv
                };
                f.less.raw_block_at_mut(m - 1, n - 1).copy_from_slice(&ll);
            }
        }
        let mut g = MovingGreen::new(tc, d, Sign::Fermi);
        init_from_g_herm(&mut g, &f, n0 - 1);
        forward(&mut g);
        extrapolate_timestep_moving_order(&mut g, k);
        let n = n0; // my row of the KB step n0 + 1
        let mut emax = 0.0f64;
        for j in 0..=tc {
            let got = g.ret_block(n, n - j);
            let want = f.ret_block(n, n - j);
            for q in 0..(d * d) {
                emax = emax.max((got[q] - want.raw()[q]).abs());
            }
        }
        assert!(emax < 1e-12, "k = {k}: {emax}");
        // the history is unchanged
        let got = g.ret_block(n0 - 1, n0 - 3);
        let want = f.ret_block(n0 - 1, n0 - 3);
        for q in 0..(d * d) {
            assert_eq!(got[q], want.raw()[q]);
        }
    }
    // insufficient window
    let mut g = ma_random(4, 2, 7, 9);
    assert!(panics(move || extrapolate_timestep_moving_order(&mut g, 2)));
}

#[test]
fn moving_algebra_distance() {
    let (tc, d, tstp) = (2i64, 2usize, 5i64);
    let g1 = ma_random(tc, d, tstp, 31);
    let mut g2 = g1.clone();
    assert_eq!(distance_moving_at(&g1, &g2, tstp - 1), 0.0);
    // different block norms in two blocks
    let b1 = [c(1.0, 0.0), c(2.0, 0.0), c(0.0, 0.0), c(-2.0, 0.0)];
    let b2 = [c(0.0, 0.5), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
    {
        let blk = g2.ret.mblock_mut(0, 0);
        for q in 0..4 {
            blk[q] += b1[q];
        }
        let blk = g2.less.mblock_mut(0, 2);
        for q in 0..4 {
            blk[q] += b2[q];
        }
    }
    let n1 = (1.0f64 + 4.0 + 4.0).sqrt();
    let n2 = 0.5f64;
    let expect = n1 + n2;
    assert!((distance_moving_at(&g1, &g2, tstp - 1) - expect).abs() < 1e-14);
    assert!((distance_moving_at(&g2, &g1, tstp - 1) - expect).abs() < 1e-14);
    let x1 = copy_timestep(&g1, 0);
    let x2 = copy_timestep(&g2, 0);
    assert!((distance_moving(&x1, &x2) - expect).abs() < 1e-14);
    assert!((expect - (n1 * n1 + n2 * n2).sqrt()).abs() > 0.1);
    assert!((expect - (n1 * n1 + n2 * n2)).abs() > 0.1);
    // older rows are unchanged
    assert_eq!(distance_moving_at(&g1, &g2, tstp - 2), 0.0);
    // the same definition as the full contour distance
    let nt = 6i64;
    let cont = Contour::new(nt - 1, 2, d as i64, d as i64, 0.5, 1.0);
    let mut a = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    let mut b = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    let mut state = 5u64;
    for n in 1..=nt {
        for m in 1..=n {
            for q in 0..(d * d) {
                a.ret.raw_block_at_mut(n - 1, m - 1)[q] = rand_c64(&mut state);
                b.ret.raw_block_at_mut(n - 1, m - 1)[q] = rand_c64(&mut state);
            }
            for q in 0..(d * d) {
                a.less.raw_block_at_mut(m - 1, n - 1)[q] = rand_c64(&mut state);
                b.less.raw_block_at_mut(m - 1, n - 1)[q] = rand_c64(&mut state);
            }
        }
    }
    let mut ga = MovingGreen::new(nt - 1, d, Sign::Fermi);
    let mut gb = MovingGreen::new(nt - 1, d, Sign::Fermi);
    init_from_g_herm(&mut ga, &a, nt - 1);
    init_from_g_herm(&mut gb, &b, nt - 1);
    let want = a.distance_ret(&b, nt - 1) + a.distance_less(&b, nt - 1);
    assert!((distance_moving_at(&ga, &gb, nt - 1) - want).abs() < 1e-12);
}
