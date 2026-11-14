//! Port of julia/test/t_linalg.jl (4 testsets).
//!
//! NESSi Catch tests ported here: libcntr/test/linalg.cpp, sections
//! complex_sq_solve and complex_sq_solve_many.

mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::{solve_left, solve_right, solve_block_right, BlockSolver};
use kadanoffbaym::linalg::DenseMat;

/// Deterministic, non-hermitian and non-commuting complex block.
fn la_block(d: usize, s: i64) -> DenseMat {
    let mut m = DenseMat::zeros(d);
    for i in 1..=d {
        for j in 1..=d {
            let re = (1.3 * s as f64 + 0.7 * i as f64 + 1.1 * j as f64).sin();
            let im = (0.9 * s as f64 + 0.4 * i as f64 - 1.7 * j as f64).cos();
            m.set(i - 1, j - 1, Complex::new(re, im));
        }
    }
    m
}

/// Diagonally dominant block system with k x k blocks of size d x d.
fn la_system(k: usize, d: usize) -> Vec<Vec<DenseMat>> {
    let mut m: Vec<Vec<DenseMat>> = (0..k)
        .map(|p| (0..k).map(|q| la_block(d, 10 * (p + 1) as i64 + (q + 1) as i64)).collect())
        .collect();
    for p in 0..k {
        for i in 0..d {
            let v = m[p][p].get(i, i) + Complex::new(2.0 * k as f64 * d as f64, 0.0);
            m[p][p].set(i, i, v);
        }
    }
    m
}

/// sum_q M[p][q] X[q] - Q[p], normalized.
fn la_res_right(m: &[Vec<DenseMat>], x: &[DenseMat], q: &[DenseMat]) -> f64 {
    let k = q.len();
    let r: f64 = (0..k)
        .map(|p| {
            let mut acc = DenseMat::zeros_shape(q[p].nrows(), q[p].ncols());
            for qq in 0..k {
                acc.axpy(1.0, &m[p][qq].mat_mul(&x[qq]));
            }
            let mut d = acc;
            d.axpy(-1.0, &q[p]);
            d.fro_norm()
        })
        .sum();
    r / q.iter().map(|b| b.fro_norm()).sum::<f64>()
}

/// sum_q X[q] M[p][q] - Q[p], normalized.
fn la_res_left(m: &[Vec<DenseMat>], x: &[DenseMat], q: &[DenseMat]) -> f64 {
    let k = q.len();
    let r: f64 = (0..k)
        .map(|p| {
            let mut acc = DenseMat::zeros_shape(q[p].nrows(), q[p].ncols());
            for qq in 0..k {
                acc.axpy(1.0, &x[qq].mat_mul(&m[p][qq]));
            }
            let mut d = acc;
            d.axpy(-1.0, &q[p]);
            d.fro_norm()
        })
        .sum();
    r / q.iter().map(|b| b.fro_norm()).sum::<f64>()
}

#[test]
fn nessi_catch_linalg_cpp() {
    let mut a = DenseMat::zeros(3);
    let av = [
        (1.0, 0.2), (2.2, 0.0), (3.4, 0.0),
        (4.6, 0.0), (5.7, 1.0), (6.8, 0.0),
        (7.5, 0.0), (8.5, 0.0), (9.6, 0.0),
    ];
    for (idx, (re, im)) in av.iter().enumerate() {
        a.set(idx / 3, idx % 3, Complex::new(*re, *im));
    }
    // complex_sq_solve
    let b = DenseMat::from_vec(
        3,
        1,
        vec![Complex::new(1.1, 0.1), Complex::new(2.4, 0.0), Complex::new(3.5, 0.3)],
    );
    let x_exact = [
        Complex::new(0.17752793262472366, 0.01775980726928994),
        Complex::new(-0.1949605345360082, 0.015227852835187461),
        Complex::new(0.3985109425906921, 0.0038921558730451444),
    ];
    let m = vec![vec![a.clone()]];
    let x = solve_right(&m, &[b]);
    let mut dn = 0.0;
    for i in 0..3 {
        dn += (x[0].get(i, 0) - x_exact[i]).abs_sqr();
    }
    assert!(dn.sqrt() < 1e-12, "dn = {dn}");
    // complex_sq_solve_many: two right-hand sides
    let b2 = DenseMat::from_vec(
        3,
        2,
        vec![
            Complex::new(1.1, 0.1), Complex::new(0.3, 1.1),
            Complex::new(2.4, 0.0), Complex::new(2.4, 1.2),
            Complex::new(3.5, 0.3), Complex::new(1.2, 0.7),
        ],
    );
    let xe = [
        Complex::new(0.17752793262472366, 0.01775980726928994),
        Complex::new(-0.13432028547481906, 0.23896436702037188),
        Complex::new(-0.1949605345360082, 0.015227852835187461),
        Complex::new(0.369779042978215, -1.5729342395547015),
        Complex::new(0.3985109425906921, 0.0038921558730451444),
        Complex::new(-0.0974708046097587, 1.2789279462043928),
    ];
    let x = solve_right(&m, &[b2]);
    let mut dn = 0.0;
    for i in 0..3 {
        for j in 0..2 {
            dn += (x[0].get(i, j) - xe[i * 2 + j]).abs_sqr();
        }
    }
    assert!(dn.sqrt() < 1e-12, "dn = {dn}");
}

#[test]
fn block_systems_d_1_to_3() {
    for d in 1..=3usize {
        for k in 1..=5usize {
            let m = la_system(k, d);
            let q: Vec<DenseMat> = (0..k).map(|p| la_block(d, 100 + 7 * (p + 1) as i64)).collect();
            let xr = solve_right(&m, &q);
            assert!(la_res_right(&m, &xr, &q) < 1e-12);
            let xl = solve_left(&m, &q);
            assert!(la_res_left(&m, &xl, &q) < 1e-12);
            // The two actions differ for non-commuting blocks
            if d > 1 {
                assert!(la_res_right(&m, &xl, &q) > 1e-3);
                assert!(la_res_left(&m, &xr, &q) > 1e-3);
            }
        }
    }
}

#[test]
fn scalar_blocks_and_write_through_targets() {
    // d = 1 follows the same code path as the matrix case
    let m = vec![vec![DenseMat::from_vec(1, 1, vec![Complex::new(2.0, 1.0)])]];
    let q = vec![DenseMat::from_vec(1, 1, vec![Complex::new(1.0, -3.0)])];
    let want = Complex::new(1.0, -3.0) / Complex::new(2.0, 1.0);
    let xr = solve_right(&m, &q);
    assert!((xr[0].get(0, 0) - want).abs() < 1e-15);
    let xl = solve_left(&m, &q);
    assert!((xl[0].get(0, 0) - want).abs() < 1e-15);
    // The solution is written into user-provided blocks (the GF
    // write-through views arrive with R2; owned blocks exercise the
    // same code path here).
    let (d, k) = (2usize, 3usize);
    let m = la_system(k, d);
    let q: Vec<DenseMat> = (0..k).map(|p| la_block(d, 50 + (p + 1) as i64)).collect();
    let mut x: Vec<DenseMat> = (0..k).map(|_| DenseMat::zeros(d)).collect();
    let mut ws = BlockSolver::new(k, d);
    solve_block_right(&mut x, &m, &q, &mut ws);
    assert!(la_res_right(&m, &x, &q) < 1e-12);
    // Several right-hand side columns (d x m blocks, m = 2d)
    let q3: Vec<DenseMat> = (0..k)
        .map(|p| {
            let mut wide = DenseMat::zeros_shape(d, 2 * d);
            let b1 = la_block(d, 70 + (p + 1) as i64);
            let b2 = la_block(d, 80 + (p + 1) as i64);
            for i in 0..d {
                for j in 0..d {
                    wide.set(i, j, b1.get(i, j));
                    wide.set(i, d + j, b2.get(i, j));
                }
            }
            wide
        })
        .collect();
    let x3 = solve_right(&m, &q3);
    assert_eq!((x3[0].nrows(), x3[0].ncols()), (d, 2 * d));
    assert!(la_res_right(&m, &x3, &q3) < 1e-12);
}

#[test]
fn error_paths() {
    let (d, k) = (2usize, 2usize);
    let m = la_system(k, d);
    let q: Vec<DenseMat> = (0..k).map(|p| la_block(d, (p + 1) as i64)).collect();
    // Shape mismatch: ragged M
    let bad_m = vec![vec![m[0][0].clone(), m[0][1].clone()]];
    let r = std::panic::catch_unwind(|| solve_right(&bad_m, &q));
    assert!(r.is_err());
    // Shape mismatch: too few Q blocks
    let r = std::panic::catch_unwind(|| solve_right(&m, &[q[0].clone()]));
    assert!(r.is_err());
    // Shape mismatch: wrong Q block size
    let bad_q = vec![q[0].clone(), DenseMat::zeros_shape(3, 2)];
    let r = std::panic::catch_unwind(|| solve_right(&m, &bad_q));
    assert!(r.is_err());
    // Shape mismatch: non-square M block
    let mut mbad = m.clone();
    mbad[0][1] = DenseMat::zeros_shape(3, 3);
    let r = std::panic::catch_unwind(|| solve_right(&mbad, &q));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(|| solve_left(&mbad, &q));
    assert!(r.is_err());
    // Singular block system
    let ms: Vec<Vec<DenseMat>> = (0..k)
        .map(|_| (0..k).map(|_| DenseMat::from_vec(d, d, vec![Complex::ONE; d * d])).collect())
        .collect();
    let r = std::panic::catch_unwind(|| solve_right(&ms, &q));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(|| solve_left(&ms, &q));
    assert!(r.is_err());
    // Non-finite input
    let mut mn = m.clone();
    mn[1][1] = DenseMat::from_vec(d, d, vec![Complex::new(f64::NAN, 0.0); d * d]);
    let r = std::panic::catch_unwind(|| solve_right(&mn, &q));
    assert!(r.is_err());
    let mut qn = q.clone();
    qn[0] = DenseMat::from_vec(d, d, vec![Complex::new(f64::INFINITY, 0.0); d * d]);
    let r = std::panic::catch_unwind(|| solve_left(&m, &qn));
    assert!(r.is_err());
}
