// Project : KadanoffBaym
// File    : src/dyson/workspace.rs
// Purpose : reusable scratch of the Dyson family time stepping
//
// One instance per time-step loop; all inner loops are allocation-free.

use crate::base::{MovingBlockScratch, TimestepScratch};
use crate::complex::Complex;
use crate::linalg::{BlockSolver, DenseMat};

/// Reusable scratch of the Dyson time stepping: one instance per
/// driver call, all inner loops are allocation-free.
pub struct DysonWorkspace {
    pub(super) ts: TimestepScratch,
    pub(super) qblocks: Vec<DenseMat>,
    pub(super) prod: DenseMat,
    pub(super) xj: DenseMat,
    pub(super) gnn: Vec<Complex>,
    pub(super) gnn_dm: DenseMat,
    pub(super) adj: Vec<Complex>,
    pub(super) diffw: Vec<Complex>,
    pub(super) qq: Vec<Complex>,
}

impl DysonWorkspace {
    pub fn new(k: i64, d: usize, ntau: usize) -> DysonWorkspace {
        let ku = k as usize;
        let mut gnn = vec![Complex::ZERO; d * d];
        for i in 0..d {
            gnn[i * (d + 1)] = Complex::new(0.0, -1.0);
        }
        let mut ws = DysonWorkspace {
            ts: TimestepScratch::new(k, d, ntau),
            qblocks: vec![DenseMat::zeros(d); ku],
            prod: DenseMat::zeros(d),
            xj: DenseMat::zeros(d),
            gnn,
            gnn_dm: DenseMat::zeros_shape(1, 1),
            adj: vec![Complex::ZERO; d * d],
            diffw: Vec::new(),
            qq: Vec::new(),
        };
        ws.gnn_dm = DenseMat::from_vec(d, d, ws.gnn.clone());
        ws
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.ts.wsk.n as i64
    }

    /// Drop the cached `f_0 * Sigma^M` blocks; call this when the
    /// Matsubara blocks of `sigma` change between two `_ws` calls
    /// (e.g. between fixed-point iterations).
    pub fn reset(&mut self) {
        self.ts.cws.reset();
    }
}

/// Reusable scratch of the pseudo-particle Dyson time stepping: one
/// instance per time-step loop. It owns the standard-family scratch
/// (retarded and lesser solves) and the left-mixing scratch; all
/// blocks are re-zeroed on every call, so reuse needs no reset.
///
/// The embedded [`DysonWorkspace`] is deliberate: the pseudo retarded
/// and lesser paths delegate to the standard time-step kernels, so
/// they need the full standard start-up and marching scratch.
pub struct PseudoDysonWorkspace {
    pub(super) dw: DysonWorkspace,
    pub(super) id: DenseMat,
    pub(super) xx: Vec<DenseMat>,
    pub(super) lm_mblocks: Vec<Vec<DenseMat>>,
    pub(super) lm_ws: BlockSolver,
    pub(super) ws1: BlockSolver,
    pub(super) qq: DenseMat,
    pub(super) mm: DenseMat,
}

impl PseudoDysonWorkspace {
    pub fn new(k: i64, d: usize, ntau: usize) -> PseudoDysonWorkspace {
        let ku = k as usize;
        PseudoDysonWorkspace {
            dw: DysonWorkspace::new(k, d, ntau),
            id: DenseMat::identity(d),
            xx: vec![DenseMat::zeros(d); ntau + 1],
            lm_mblocks: vec![vec![DenseMat::zeros(d); ku]; ku],
            lm_ws: BlockSolver::new(ku, d),
            ws1: BlockSolver::new(1, d),
            qq: DenseMat::zeros(d),
            mm: DenseMat::zeros(d),
        }
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.dw.order()
    }

    /// Drop the cached `f_0 * Sigma^M` blocks of the embedded
    /// standard-family scratch; call this when the Matsubara blocks of
    /// `sigma` change between two `_ws` calls (e.g. between DMFT
    /// iterations).
    pub fn reset(&mut self) {
        self.dw.reset();
    }
}

/// Reusable scratch of the truncated (moving) Dyson time stepping: one
/// instance per time-step loop. The start-up block system, the
/// single-block solves, and the accumulation temporaries are owned
/// here; only the small per-expression temporaries (borrowing slices)
/// remain on the call path.
pub struct MovingDysonWorkspace {
    pub(super) id: DenseMat,
    pub(super) blocks: MovingBlockScratch,
    pub(super) ws1: BlockSolver,
    pub(super) mm: DenseMat,
    pub(super) mm2: DenseMat,
    pub(super) g00: Vec<Complex>,
    pub(super) diffw: Vec<Complex>,
}

impl MovingDysonWorkspace {
    pub fn new(k: i64, d: usize) -> MovingDysonWorkspace {
        let ku = k as usize;
        MovingDysonWorkspace {
            id: DenseMat::identity(d),
            blocks: MovingBlockScratch::new(ku, d),
            ws1: BlockSolver::new(1, d),
            mm: DenseMat::zeros(d),
            mm2: DenseMat::zeros(d),
            g00: Vec::new(),
            diffw: Vec::new(),
        }
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.blocks.ws.n as i64
    }
}
