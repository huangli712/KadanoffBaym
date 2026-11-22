// Project : KadanoffBaym
// File    : src/linalg/linsolve.rs
// Purpose : block linear systems
//
// The start-up and time-stepping procedures of the Dyson equation lead
// to small linear systems whose unknowns are d x d matrices:
//
//   right action: sum_q M[p][q] * X[q] = Q[p]
//   left  action: sum_q X[q] * M[p][q] = Q[p]
//
// The right-action system is assembled into an (nd) x (nd) matrix with
// multiple right-hand sides; the left-action system is transformed into
// a right one by taking the hermitian conjugate of every block (the
// block structure itself is not transposed).

//!
//! *Remarks* : *Block Linear Systems*
//!
//! The start-up and time-stepping procedures of the Dyson equation lead to
//! small linear systems whose unknowns are 𝑑 × 𝑑 matrices. The right action
//! system reads
//!
//! ```math
//! \begin{equation}
//! \sum^{n}_{q=1} M_{pq} X_q = Q_p, \quad p = 1, \cdots, n,
//! \end{equation}
//! ```
//!
//! and the left action system reads
//!
//! ```math
//! \begin{equation}
//! \sum^{n}_{q=1} X_q M_{pq} = Q_p, \quad p = 1, \cdots, n.
//! \end{equation}
//! ```
//!
//! The right action system is solved by assembling the 𝑛 × 𝑛 blocks into a
//! (𝑛𝑑) × (𝑛𝑑) matrix, and all the columns of ``Q_p`` are treated as multiple
//! right-hand sides. The left action system is transformed into a right one
//! by taking the hermitian conjugate of every block,
//! ``\sum_q M^{\dagger}_{pq} X^{\dagger}_q = Q^{\dagger}_p``. Note that the
//! block structure itself is not transposed.

use crate::complex::Complex;
use crate::linalg::DenseMat;

use crate::linalg::{lu_factor, lu_solve};

/// Workspace for the block linear systems.
/// `n` is the number of blocks, `d` the block dimension, and `m` the
/// number of columns of the right-hand sides (right action) or rows
/// (left action).
pub(crate) struct BlockSolver {
    pub n: usize,
    pub d: usize,
    pub m: usize,

    /// Assembled matrix (n*d) x (n*d).
    pub mbig: DenseMat,

    /// LU factorization of `mbig`.
    pub mlu: DenseMat,

    /// Right-hand sides (n*d) x m.
    pub qbig: DenseMat,

    /// Solution (n*d) x m.
    pub xbig: DenseMat,

    /// Residual (n*d) x m.
    pub rbig: DenseMat,

    /// Pivot table of `mlu`.
    piv: Vec<usize>,
}

impl BlockSolver {
    /// Create the workspace (defaults to `m = d`).
    pub fn new(n: usize, d: usize) -> BlockSolver {
        BlockSolver::new_rect(n, d, d)
    }

    /// Create the workspace with an explicit number of right-hand-side
    /// columns `m`.
    pub(crate) fn new_rect(n: usize, d: usize, m: usize) -> BlockSolver {
        assert!(n >= 1, "BlockSolver: n >= 1, got {n}");
        assert!(d >= 1, "BlockSolver: d >= 1, got {d}");
        assert!(m >= 1, "BlockSolver: m >= 1, got {m}");
        let big = n * d;
        BlockSolver {
            n,
            d,
            m,
            mbig: DenseMat::zeros(big),
            mlu: DenseMat::zeros(big),
            qbig: DenseMat::zeros_shape(big, m),
            xbig: DenseMat::zeros_shape(big, m),
            rbig: DenseMat::zeros_shape(big, m),
            piv: Vec::new(),
        }
    }

    /// Copy the (p, q) block (0-based) into the assembled matrix.
    fn set_mbig(&mut self, p: usize, q: usize, block: &DenseMat) {
        let d = self.d;
        for i in 0..d {
            for j in 0..d {
                self.mbig.set(p * d + i, q * d + j, block.get(i, j));
            }
        }
    }

    /// Adjoint-copy the (p, q) block into the assembled matrix
    /// (conjugate transpose of the d x d block).
    fn set_mbig_adjoint(&mut self, p: usize, q: usize, block: &DenseMat) {
        let d = self.d;
        for i in 0..d {
            for j in 0..d {
                self.mbig.set(p * d + i, q * d + j, block.get(j, i).conj());
            }
        }
    }

    /// Copy the p-th block into the right-hand sides.
    fn set_qbig(&mut self, p: usize, block: &DenseMat) {
        let d = self.d;
        for i in 0..block.nrows() {
            for j in 0..block.ncols() {
                self.qbig.set(p * d + i, j, block.get(i, j));
            }
        }
    }

    /// Adjoint-copy the p-th block into the right-hand sides.
    fn set_qbig_adjoint(&mut self, p: usize, block: &DenseMat) {
        let d = self.d;
        // block is m x d -> adjoint d x m
        for i in 0..d {
            for j in 0..self.m {
                self.qbig.set(p * d + i, j, block.get(j, i).conj());
            }
        }
    }

    /// Solve `Mbig * Xbig = Qbig` in place:
    /// finite input, regular system, and normalized residual are checked.
    fn solve_core(&mut self) {
        for z in &self.mbig.data {
            assert!(z.re.is_finite() && z.im.is_finite(), "matrix is not finite");
        }
        for z in &self.qbig.data {
            assert!(
                z.re.is_finite() && z.im.is_finite(),
                "right-hand side is not finite"
            );
        }
        // LU factorization with partial pivoting (panics when singular)
        self.mlu.set_from(&self.mbig);
        self.piv = lu_factor(&mut self.mlu);
        // Solve column-wise
        let big = self.n * self.d;
        for j in 0..self.m {
            let col: Vec<Complex> = (0..big).map(|i| self.qbig.get(i, j)).collect();
            let sol = lu_solve(&self.mlu, &self.piv, &col);
            for (i, v) in sol.iter().enumerate() {
                self.xbig.set(i, j, *v);
            }
        }
        for z in &self.xbig.data {
            assert!(
                z.re.is_finite() && z.im.is_finite(),
                "solution is not finite"
            );
        }
        // Normalized residual check
        self.rbig.set_from(&self.qbig);
        for i in 0..big {
            for j in 0..self.m {
                let mut acc = Complex::ZERO;
                for kk in 0..big {
                    acc += self.mbig.get(i, kk) * self.xbig.get(kk, j);
                }
                self.rbig.set(i, j, acc - self.qbig.get(i, j));
            }
        }
        let res = self.rbig.fro_norm();
        let scale = self.mbig.fro_norm() * self.xbig.fro_norm() + self.qbig.fro_norm();
        assert!(
            res <= 1e-10 * scale,
            "The block linear system is ill-conditioned (residual = {res})"
        );
    }
}

/// Check block shapes: `M` is `n x n` blocks
/// of `d x d`, `Q` is `n` blocks of the expected shape.
fn check_blocks(m: &[Vec<DenseMat>], q: &[DenseMat], ws: &BlockSolver, qrows: usize, qcols: usize) {
    let (n, d) = (ws.n, ws.d);
    assert_eq!(m.len(), n, "M must have {n} x {n} blocks");
    assert_eq!(q.len(), n, "Q must have {n} blocks");
    for p in 0..n {
        assert_eq!(
            (q[p].nrows(), q[p].ncols()),
            (qrows, qcols),
            "wrong size of Q[{p}]"
        );
        assert_eq!(m[p].len(), n, "M must have {n} x {n} blocks");
        for (qq, mpq) in m[p].iter().enumerate() {
            assert_eq!(
                (mpq.nrows(), mpq.ncols()),
                (d, d),
                "M[{p}][{qq}] must be {d} x {d}"
            );
        }
    }
}

/// Solve `sum_q M[p][q] * X[q] = Q[p]`;
/// `X[p]` are d x m blocks, overwritten with the solution.
pub(crate) fn solve_block_right(
    x: &mut [DenseMat],
    m: &[Vec<DenseMat>],
    q: &[DenseMat],
    ws: &mut BlockSolver,
) {
    check_blocks(m, q, ws, ws.d, ws.m);
    assert_eq!(x.len(), ws.n, "X must have {} blocks", ws.n);
    for p in 0..ws.n {
        ws.set_qbig(p, &q[p]);
        for (qq, mpq) in m[p].iter().enumerate() {
            ws.set_mbig(p, qq, mpq);
        }
    }
    ws.solve_core();
    let d = ws.d;
    for (p, xp) in x.iter_mut().enumerate() {
        for i in 0..d {
            for j in 0..ws.m {
                xp.set(i, j, ws.xbig.get(p * d + i, j));
            }
        }
    }
}

/// Solve `sum_q X[q] * M[p][q] = Q[p]`;
/// `X[p]` are m x d blocks, overwritten with the solution. Implemented
/// via the hermitian conjugate system `sum_q M[p][q]^dagger X[q]^dagger
/// = Q[p]^dagger`.
pub(crate) fn solve_block_left(
    x: &mut [DenseMat],
    m: &[Vec<DenseMat>],
    q: &[DenseMat],
    ws: &mut BlockSolver,
) {
    check_blocks(m, q, ws, ws.m, ws.d);
    assert_eq!(x.len(), ws.n, "X must have {} blocks", ws.n);
    for p in 0..ws.n {
        ws.set_qbig_adjoint(p, &q[p]);
        for (qq, mpq) in m[p].iter().enumerate() {
            ws.set_mbig_adjoint(p, qq, mpq);
        }
    }
    ws.solve_core();
    let d = ws.d;
    for (p, xp) in x.iter_mut().enumerate() {
        for i in 0..ws.m {
            for j in 0..d {
                xp.set(i, j, ws.xbig.get(p * d + j, i).conj());
            }
        }
    }
}

/// Convenience single-block `M X = Q`.
pub(crate) fn solve_block_right_single(
    x: &mut DenseMat,
    m: &DenseMat,
    q: &DenseMat,
    ws: &mut BlockSolver,
) {
    assert_eq!(ws.n, 1, "single-block solve needs ws.n == 1");
    ws.mbig.set_from(m);
    ws.qbig.set_from(q);
    ws.solve_core();
    x.set_from(&ws.xbig);
}

/// Convenience single-block `X M = Q`.
pub(crate) fn solve_block_left_single(
    x: &mut DenseMat,
    m: &DenseMat,
    q: &DenseMat,
    ws: &mut BlockSolver,
) {
    assert_eq!(ws.n, 1, "single-block solve needs ws.n == 1");
    m.adjoint_into(&mut ws.mbig);
    q.adjoint_into(&mut ws.qbig);
    ws.solve_core();
    ws.xbig.adjoint_into(x);
}

/// Solve `sum_q M[p][q] X[q] = Q[p]` with a fresh workspace and return
/// the `d x m` solution blocks.
#[allow(dead_code)]
fn solve_right(m: &[Vec<DenseMat>], q: &[DenseMat]) -> Vec<DenseMat> {
    let n = q.len();
    assert!(n >= 1, "Q is empty");
    let (d, mc) = (q[0].nrows(), q[0].ncols());
    let mut ws = BlockSolver::new_rect(n, d, mc);
    let mut x: Vec<DenseMat> = (0..n).map(|_| DenseMat::zeros_shape(d, mc)).collect();
    solve_block_right(&mut x, m, q, &mut ws);
    x
}

/// Solve `sum_q X[q] M[p][q] = Q[p]` with a fresh workspace and return
/// the `m x d` solution blocks.
#[allow(dead_code)]
fn solve_left(m: &[Vec<DenseMat>], q: &[DenseMat]) -> Vec<DenseMat> {
    let n = q.len();
    assert!(n >= 1, "Q is empty");
    let (mc, d) = (q[0].nrows(), q[0].ncols());
    let mut ws = BlockSolver::new_rect(n, d, mc);
    let mut x: Vec<DenseMat> = (0..n).map(|_| DenseMat::zeros_shape(mc, d)).collect();
    solve_block_left(&mut x, m, q, &mut ws);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::catch_unwind;

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
            .map(|p| {
                (0..k)
                    .map(|q| la_block(d, 10 * (p + 1) as i64 + (q + 1) as i64))
                    .collect()
            })
            .collect();
        for (p, row) in m.iter_mut().enumerate() {
            let blk = &mut row[p];
            for i in 0..d {
                let v = blk.get(i, i) + Complex::new(2.0 * k as f64 * d as f64, 0.0);
                blk.set(i, i, v);
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
    fn catch_linalg_cpp() {
        let mut a = DenseMat::zeros(3);
        let av = [
            (1.0, 0.2),
            (2.2, 0.0),
            (3.4, 0.0),
            (4.6, 0.0),
            (5.7, 1.0),
            (6.8, 0.0),
            (7.5, 0.0),
            (8.5, 0.0),
            (9.6, 0.0),
        ];
        for (idx, (re, im)) in av.iter().enumerate() {
            a.set(idx / 3, idx % 3, Complex::new(*re, *im));
        }
        // complex_sq_solve
        let b = DenseMat::from_vec(
            3,
            1,
            vec![
                Complex::new(1.1, 0.1),
                Complex::new(2.4, 0.0),
                Complex::new(3.5, 0.3),
            ],
        );
        let x_exact = [
            Complex::new(0.17752793262472366, 0.01775980726928994),
            Complex::new(-0.1949605345360082, 0.015227852835187461),
            Complex::new(0.3985109425906921, 0.0038921558730451444),
        ];
        let m = vec![vec![a.clone()]];
        let x = solve_right(&m, &[b]);
        let mut dn = 0.0;
        for (i, xe) in x_exact.iter().enumerate() {
            dn += (x[0].get(i, 0) - *xe).abs_sqr();
        }
        assert!(dn.sqrt() < 1e-12, "dn = {dn}");
        // complex_sq_solve_many: two right-hand sides
        let b2 = DenseMat::from_vec(
            3,
            2,
            vec![
                Complex::new(1.1, 0.1),
                Complex::new(0.3, 1.1),
                Complex::new(2.4, 0.0),
                Complex::new(2.4, 1.2),
                Complex::new(3.5, 0.3),
                Complex::new(1.2, 0.7),
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
                let q: Vec<DenseMat> = (0..k)
                    .map(|p| la_block(d, 100 + 7 * (p + 1) as i64))
                    .collect();
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
        let r = catch_unwind(|| solve_right(&bad_m, &q));
        assert!(r.is_err());
        // Shape mismatch: too few Q blocks
        let r = catch_unwind(|| solve_right(&m, &[q[0].clone()]));
        assert!(r.is_err());
        // Shape mismatch: wrong Q block size
        let bad_q = vec![q[0].clone(), DenseMat::zeros_shape(3, 2)];
        let r = catch_unwind(|| solve_right(&m, &bad_q));
        assert!(r.is_err());
        // Shape mismatch: non-square M block
        let mut mbad = m.clone();
        mbad[0][1] = DenseMat::zeros_shape(3, 3);
        let r = catch_unwind(|| solve_right(&mbad, &q));
        assert!(r.is_err());
        let r = catch_unwind(|| solve_left(&mbad, &q));
        assert!(r.is_err());
        // Singular block system
        let ms: Vec<Vec<DenseMat>> = (0..k)
            .map(|_| {
                (0..k)
                    .map(|_| DenseMat::from_vec(d, d, vec![Complex::ONE; d * d]))
                    .collect()
            })
            .collect();
        let r = catch_unwind(|| solve_right(&ms, &q));
        assert!(r.is_err());
        let r = catch_unwind(|| solve_left(&ms, &q));
        assert!(r.is_err());
        // Non-finite input
        let mut mn = m.clone();
        mn[1][1] = DenseMat::from_vec(d, d, vec![Complex::new(f64::NAN, 0.0); d * d]);
        let r = catch_unwind(|| solve_right(&mn, &q));
        assert!(r.is_err());
        let mut qn = q.clone();
        qn[0] = DenseMat::from_vec(d, d, vec![Complex::new(f64::INFINITY, 0.0); d * d]);
        let r = catch_unwind(|| solve_left(&m, &qn));
        assert!(r.is_err());
    }
}
