//
// Project : KadanoffBaym
// File    : src/base/scratch.rs
// Purpose : shared real-time time-step scratch of the Dyson and
//           VIE2 solver workspaces
//

use crate::complex::Complex;
use crate::convolution::ConvWorkspace;
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::LmixComp;

/// The shared scratch anatomy of a contour time-step block solver:
/// the `k x k` start-up block system (`wsk`/`mblocks`/`xblocks`), the
/// single-block marching solve (`ws1`/`mm`/`qqj`/`gnj`), the identity,
/// the left-mixing rectangular system (`lrow`/`lsolver`/`lmblocks`/
/// `lqblocks`/`lxblocks`), the lesser source buffer `gles`, and the
/// convolution scratch `cws`.
///
/// Pure data: the workspace carries no invariants. Re-zeroing and
/// cache discipline stay with the equation code that borrows it.
pub(crate) struct TimestepScratch {
    pub(crate) wsk: BlockSolver,
    pub(crate) ws1: BlockSolver,
    pub(crate) mblocks: Vec<Vec<DenseMat>>,
    pub(crate) xblocks: Vec<DenseMat>,
    pub(crate) mm: DenseMat,
    pub(crate) qqj: DenseMat,
    pub(crate) gnj: DenseMat,
    pub(crate) id: DenseMat,
    pub(crate) lrow: LmixComp,
    pub(crate) lsolver: BlockSolver,
    pub(crate) lmblocks: Vec<Vec<DenseMat>>,
    pub(crate) lqblocks: Vec<DenseMat>,
    pub(crate) lxblocks: Vec<DenseMat>,
    pub(crate) gles: Vec<Vec<Complex>>,
    pub(crate) cws: ConvWorkspace,
}

impl TimestepScratch {
    pub(crate) fn new(k: i64, d: usize, ntau: usize) -> TimestepScratch {
        let ku = k as usize;
        let mcols = d * (ntau + 1);
        TimestepScratch {
            wsk: BlockSolver::new(ku, d),
            ws1: BlockSolver::new(1, d),
            mblocks: vec![vec![DenseMat::zeros(d); ku]; ku],
            xblocks: vec![DenseMat::zeros(d); ku],
            mm: DenseMat::zeros(d),
            qqj: DenseMat::zeros(d),
            gnj: DenseMat::zeros(d),
            id: DenseMat::identity(d),
            lrow: LmixComp::new(0, ntau, d, d, Complex::ZERO),
            lsolver: BlockSolver::new_rect(1, d, mcols),
            lmblocks: vec![vec![DenseMat::zeros(d)]],
            lqblocks: vec![DenseMat::zeros_shape(d, mcols)],
            lxblocks: vec![DenseMat::zeros_shape(d, mcols)],
            gles: Vec::new(),
            cws: ConvWorkspace::new(d, d, d, d),
        }
    }
}

/// The start-up block system of a moving-window solver: the `n x n`
/// matrix blocks, the right-hand-side `n`-vector, the solution
/// `n`-vector, and the block solver. `n` is `k` for the retarded
/// (Dyson / VIE2 ret) system and `k + 1` for the VIE2 lesser system.
pub(crate) struct MovingBlockScratch {
    pub(crate) mblocks: Vec<Vec<DenseMat>>,
    pub(crate) rhsblocks: Vec<DenseMat>,
    pub(crate) xblocks: Vec<DenseMat>,
    pub(crate) ws: BlockSolver,
}

impl MovingBlockScratch {
    pub(crate) fn new(n: usize, d: usize) -> MovingBlockScratch {
        MovingBlockScratch {
            mblocks: vec![vec![DenseMat::zeros(d); n]; n],
            rhsblocks: vec![DenseMat::zeros(d); n],
            xblocks: vec![DenseMat::zeros(d); n],
            ws: BlockSolver::new(n, d),
        }
    }
}
