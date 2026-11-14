// Project : KadanoffBaym
// File    : src/linalg/linalg_block.rs
// Purpose : block linear systems (blueprint: julia/src/solvers/linalg.jl)
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
// block structure itself is not transposed). See NESSi
// element_linsolve_right / element_linsolve_left.
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
//!
//! *References* :
//!
//! Please see [`NESSi`] `cntr_elements.hpp`, `element_linsolve_right` and
//! `element_linsolve_left`.

use crate::complex::Complex;
use crate::linalg::{lu_factor, lu_solve};
use crate::linalg::DenseMat;

/// Workspace for the block linear systems (blueprint `_BlockSolver`).
/// `n` is the number of blocks, `d` the block dimension, and `m` the
/// number of columns of the right-hand sides (right action) or rows
/// (left action).
pub struct BlockSolver {
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
    /// Create the workspace (defaults to `m = d`, like the blueprint).
    pub fn new(n: usize, d: usize) -> BlockSolver {
        BlockSolver::new_rect(n, d, d)
    }

    /// Create the workspace with an explicit number of right-hand-side
    /// columns `m`.
    pub fn new_rect(n: usize, d: usize, m: usize) -> BlockSolver {
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

    /// Solve `Mbig * Xbig = Qbig` in place (blueprint `_solve_block!`):
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

/// Check block shapes (blueprint `_check_blocks`): `M` is `n x n` blocks
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
        for qq in 0..n {
            assert_eq!(
                (m[p][qq].nrows(), m[p][qq].ncols()),
                (d, d),
                "M[{p}][{qq}] must be {d} x {d}"
            );
        }
    }
}

/// Solve `sum_q M[p][q] * X[q] = Q[p]` (blueprint `_solve_block_right!`);
/// `X[p]` are d x m blocks, overwritten with the solution.
pub fn solve_block_right(
    x: &mut [DenseMat],
    m: &[Vec<DenseMat>],
    q: &[DenseMat],
    ws: &mut BlockSolver,
) {
    check_blocks(m, q, ws, ws.d, ws.m);
    assert_eq!(x.len(), ws.n, "X must have {} blocks", ws.n);
    for p in 0..ws.n {
        ws.set_qbig(p, &q[p]);
        for qq in 0..ws.n {
            ws.set_mbig(p, qq, &m[p][qq]);
        }
    }
    ws.solve_core();
    let d = ws.d;
    for p in 0..ws.n {
        for i in 0..d {
            for j in 0..ws.m {
                x[p].set(i, j, ws.xbig.get(p * d + i, j));
            }
        }
    }
}

/// Solve `sum_q X[q] * M[p][q] = Q[p]` (blueprint `_solve_block_left!`);
/// `X[p]` are m x d blocks, overwritten with the solution. Implemented
/// via the hermitian conjugate system `sum_q M[p][q]^dagger X[q]^dagger
/// = Q[p]^dagger`.
pub fn solve_block_left(
    x: &mut [DenseMat],
    m: &[Vec<DenseMat>],
    q: &[DenseMat],
    ws: &mut BlockSolver,
) {
    check_blocks(m, q, ws, ws.m, ws.d);
    assert_eq!(x.len(), ws.n, "X must have {} blocks", ws.n);
    for p in 0..ws.n {
        ws.set_qbig_adjoint(p, &q[p]);
        for qq in 0..ws.n {
            ws.set_mbig_adjoint(p, qq, &m[p][qq]);
        }
    }
    ws.solve_core();
    let d = ws.d;
    for p in 0..ws.n {
        for i in 0..ws.m {
            for j in 0..d {
                x[p].set(i, j, ws.xbig.get(p * d + j, i).conj());
            }
        }
    }
}

/// Convenience single-block `M X = Q` (blueprint `_solve_block_right!`
/// for matrices, `ws` from `BlockSolver::new_rect(1, d, m)`).
pub fn solve_block_right_single(
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

/// Convenience single-block `X M = Q` (blueprint `_solve_block_left!`
/// for matrices).
pub fn solve_block_left_single(x: &mut DenseMat, m: &DenseMat, q: &DenseMat, ws: &mut BlockSolver) {
    assert_eq!(ws.n, 1, "single-block solve needs ws.n == 1");
    m.adjoint_into(&mut ws.mbig);
    q.adjoint_into(&mut ws.qbig);
    ws.solve_core();
    ws.xbig.adjoint_into(x);
}

/// Solve `sum_q M[p][q] X[q] = Q[p]` with a fresh workspace and return
/// the `d x m` solution blocks (blueprint `_solve_block_right`).
pub fn solve_right(m: &[Vec<DenseMat>], q: &[DenseMat]) -> Vec<DenseMat> {
    let n = q.len();
    assert!(n >= 1, "Q is empty");
    let (d, mc) = (q[0].nrows(), q[0].ncols());
    let mut ws = BlockSolver::new_rect(n, d, mc);
    let mut x: Vec<DenseMat> = (0..n).map(|_| DenseMat::zeros_shape(d, mc)).collect();
    solve_block_right(&mut x, m, q, &mut ws);
    x
}

/// Solve `sum_q X[q] M[p][q] = Q[p]` with a fresh workspace and return
/// the `m x d` solution blocks (blueprint `_solve_block_left`).
pub fn solve_left(m: &[Vec<DenseMat>], q: &[DenseMat]) -> Vec<DenseMat> {
    let n = q.len();
    assert!(n >= 1, "Q is empty");
    let (mc, d) = (q[0].nrows(), q[0].ncols());
    let mut ws = BlockSolver::new_rect(n, d, mc);
    let mut x: Vec<DenseMat> = (0..n).map(|_| DenseMat::zeros_shape(mc, d)).collect();
    solve_block_left(&mut x, m, q, &mut ws);
    x
}
