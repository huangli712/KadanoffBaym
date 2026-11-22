// Project : KadanoffBaym
// File    : src/vie2/workspace.rs
// Purpose : reusable scratch of the VIE2 time stepping
//
// One instance per time-step loop; all inner loops are allocation-free.

use crate::base::TimestepScratch;
use crate::complex::Complex;
use crate::linalg::DenseMat;

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
