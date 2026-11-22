// Project : KadanoffBaym
// File    : src/vie2/workspace.rs
// Purpose : reusable scratch of the VIE2 time stepping
//
// One instance per time-step loop; all inner loops are allocation-free.

use crate::base::TimestepScratch;
use crate::complex::Complex;
use crate::convolution::ConvWorkspace;
use crate::linalg::{BlockSolver, DenseMat};

/// Reusable scratch of the VIE2 time stepping: one instance per
/// time-step loop, all inner loops are allocation-free.
pub struct Vie2Workspace {
    pub(super) ts: TimestepScratch,
    // retarded-path scratch specific to the VIE2 right action
    pub(super) rblocks: Vec<DenseMat>,
    pub(super) adjbuf: Vec<Complex>,
    pub(super) qqbuf: Vec<Complex>,
}

impl Vie2Workspace {
    pub fn new(k: i64, d: usize, ntau: usize) -> Vie2Workspace {
        let ku = k as usize;
        Vie2Workspace {
            ts: TimestepScratch::new(k, d, ntau),
            rblocks: vec![DenseMat::zeros(d); ku],
            adjbuf: vec![Complex::ZERO; d * d],
            qqbuf: Vec::new(),
        }
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.ts.wsk.n as i64
    }

    /// Drop the cached `f_0 * F^M` blocks; call this when the
    /// Matsubara blocks of the kernel `f` change between two `_ws`
    /// calls.
    pub fn reset(&mut self) {
        self.ts.cws.reset();
    }
}

/// Reusable scratch of the pseudo-particle VIE2 time stepping: one
/// instance per time-step loop. All blocks are re-zeroed on every
/// call; the convolution scratch carries no cross-step cache (the
/// incremental kernels use only its buffers).
///
/// Unlike [`crate::dyson::PseudoDysonWorkspace`], this workspace does
/// not embed [`Vie2Workspace`]: the pseudo VIE2 path uses the
/// incremental convolution kernels and never builds the standard
/// VIE2 start-up block systems, so only the convolution scratch, one
/// block solver, and one block temporary are needed.
pub struct PseudoVie2Workspace {
    pub(super) k: i64,
    pub(super) id: DenseMat,
    pub(super) ws1: BlockSolver,
    pub(super) mm: DenseMat,
    pub(super) cws: ConvWorkspace,
}

impl PseudoVie2Workspace {
    pub fn new(k: i64, d: usize) -> PseudoVie2Workspace {
        PseudoVie2Workspace {
            k,
            id: DenseMat::identity(d),
            ws1: BlockSolver::new(1, d),
            mm: DenseMat::zeros(d),
            cws: ConvWorkspace::new(d, d, d, d),
        }
    }

    /// Solver order `k` the workspace was built for.
    pub fn order(&self) -> i64 {
        self.k
    }

    /// Drop the cached convolution buffers; the incremental kernels
    /// carry no cross-step Matsubara cache, so reuse needs no reset in
    /// the current kernels, but the call keeps the workspace contract
    /// uniform.
    pub fn reset(&mut self) {
        self.cws.reset();
    }
}
