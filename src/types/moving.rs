// Project : KadanoffBaym
// File    : src/types/moving.rs
// Purpose : the moving (truncated) window types
//
// A moving function stores the retarded and lesser components on a
// window of `tc + 1` time steps. The leading time step is `tstp`. The
// block `(delti, deltj)`, `0 <= delti, deltj <= tc`, contains
// G^{R/<}(t0 - delt_i, t0 - delt_i - delt_j). Every component owns one
// contiguous array of the size (tc + 1) * (tc + 1) blocks; the logical
// row `delti` is stored in the physical row
// `(head + delti) % (tc + 1)`, so that moving the window forward only
// rotates `head`. The Matsubara and mixing components are not stored.
//
// Index convention: the leading time step `tstp` is the 0-based real
// time step; the 1-based step n0 maps to n0 - 1.

//!
//! *Remarks* : *Moving (Truncated) Window*
//!
//! A moving function stores the retarded and lesser components on a window
//! of `tc + 1` time steps. The leading time step is `t₀`. The block
//! `(delti, deltj)`, `0 ≤ delti, deltj ≤ tc`, contains
//!
//! ```math
//! G^{R}(t_0 - \delta_i, t_0 - \delta_i - \delta_j),\quad
//! G^{<}(t_0 - \delta_i, t_0 - \delta_i - \delta_j).
//! ```
//!
//! Every component owns one contiguous array of the size
//! `(ndim1, ndim2, tc + 1, tc + 1)`, the last two indices are `deltj + 1`
//! and the physical row. The logical row `delti` is stored in the physical
//! row `(head + delti) mod (tc + 1)`, so that moving the window forward only
//! rotates `head`. The Matsubara and mixing components are not stored.

use crate::complex::Complex;
use crate::types::Sign;

/// Ring-buffer component (retarded or lesser) of a moving Green's function.
///
/// The payload is `(tc + 1)` physical rows of `(tc + 1)` blocks each,
/// row-major: the block `(delti, deltj)` lives at
/// `(physical_row(delti) * (tc + 1) + deltj) * block`.
#[derive(Clone, Debug, PartialEq)]
pub struct RingComp {
    pub tc: i64,
    pub size1: usize,
    pub size2: usize,
    pub head: i64,
    pub(crate) data: Vec<Complex>,
}

impl RingComp {
    pub fn new(tc: i64, size1: usize, size2: usize, v: Complex) -> Self {
        let n =
            check_payload(tc, size1, size2, (tc + 1) as usize).expect("moving payload overflow");
        RingComp {
            tc,
            size1,
            size2,
            head: 0,
            data: vec![v; n],
        }
    }

    pub fn block(&self) -> usize {
        self.size1 * self.size2
    }

    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// Physical row of the logical row `delti`.
    #[inline]
    fn mrow_index(&self, delti: i64) -> usize {
        debug_assert!(0 <= delti && delti <= self.tc);
        ((self.head + delti).rem_euclid(self.tc + 1)) as usize
    }

    /// The `(tc + 1)` blocks of the logical row `delti`, contiguous.
    pub fn mrow(&self, delti: i64) -> &[Complex] {
        let p = self.mrow_index(delti);
        let b = self.block();
        &self.data[p * (self.tc + 1) as usize * b..(p + 1) * (self.tc + 1) as usize * b]
    }

    /// Mutable row view.
    pub fn mrow_mut(&mut self, delti: i64) -> &mut [Complex] {
        let p = self.mrow_index(delti);
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        &mut self.data[p * tcl * b..(p + 1) * tcl * b]
    }

    /// The block `(delti, deltj)`.
    pub fn mblock(&self, delti: i64, deltj: i64) -> &[Complex] {
        assert!(
            0 <= delti && delti <= self.tc && 0 <= deltj && deltj <= self.tc,
            "mblock: ({delti}, {deltj}) outside 0..={}",
            self.tc
        );
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        let off = (self.mrow_index(delti) * tcl + deltj as usize) * b;
        &self.data[off..off + b]
    }

    /// Mutable block view.
    pub fn mblock_mut(&mut self, delti: i64, deltj: i64) -> &mut [Complex] {
        assert!(
            0 <= delti && delti <= self.tc && 0 <= deltj && deltj <= self.tc,
            "mblock: ({delti}, {deltj}) outside 0..={}",
            self.tc
        );
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        let off = (self.mrow_index(delti) * tcl + deltj as usize) * b;
        &mut self.data[off..off + b]
    }

    /// Rotate the ring forward by one physical row (the leading row
    /// moves onto the slot of the oldest row) and zero the new leading row.
    pub(crate) fn rotate_forward(&mut self) {
        self.head = (self.head + self.tc).rem_euclid(self.tc + 1);
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        let p = self.mrow_index(0);
        for z in &mut self.data[p * tcl * b..(p + 1) * tcl * b] {
            *z = Complex::ZERO;
        }
    }
}

/// Row component of a moving time step: the `(tc + 1)` blocks `G(t, t - deltj)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RowComp {
    pub tc: i64,
    pub size1: usize,
    pub size2: usize,
    pub(crate) data: Vec<Complex>,
}

impl RowComp {
    pub fn new(tc: i64, size1: usize, size2: usize, v: Complex) -> Self {
        let n = check_payload(tc, size1, size2, 1).expect("moving row payload overflow");
        RowComp {
            tc,
            size1,
            size2,
            data: vec![v; n],
        }
    }

    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// The block `deltj`.
    pub fn mblock(&self, deltj: i64) -> &[Complex] {
        assert!(
            0 <= deltj && deltj <= self.tc,
            "mblock: deltj {deltj} outside 0..={}",
            self.tc
        );
        let b = self.size1 * self.size2;
        let off = deltj as usize * b;
        &self.data[off..off + b]
    }

    pub fn mblock_mut(&mut self, deltj: i64) -> &mut [Complex] {
        assert!(
            0 <= deltj && deltj <= self.tc,
            "mblock: deltj {deltj} outside 0..={}",
            self.tc
        );
        let b = self.size1 * self.size2;
        let off = deltj as usize * b;
        &mut self.data[off..off + b]
    }
}

/// Moving (truncated) contour-ordered Green's function. `tstp` is the
/// 0-based real time step of the leading row
/// (`delti = 0`); it is meaningful only if `initialized` is true.
/// `generation` is incremented by every operation that changes the
/// mapping between the logical time and the storage, so that stale
/// views can be detected.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingGreen {
    pub tc: i64,
    pub tstp: i64,
    pub initialized: bool,
    pub sign: Sign,
    pub generation: u64,
    pub ret: RingComp,
    pub less: RingComp,
}

impl MovingGreen {
    /// Constructor: an uninitialized moving Green's function.
    pub fn new(tc: i64, ndim: usize, sign: Sign) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        MovingGreen {
            tc,
            tstp: 0,
            initialized: false,
            sign,
            generation: 0,
            ret: RingComp::new(tc, ndim, ndim, Complex::ZERO),
            less: RingComp::new(tc, ndim, ndim, Complex::ZERO),
        }
    }

    /// Hermitian-reconstructed retarded block `G^{R}(t_n, t_m)` inside
    /// the window; `n >= m` reads the stored block, `n < m` returns
    /// `-G^{R}(t_m, t_n)^dagger`.
    pub fn ret_block(&self, n: i64, m: i64) -> Vec<Complex> {
        assert!(self.initialized, "the moving object is not initialized");
        assert!(m >= 0, "({n}, {m}) outside the moving window");
        if n >= m {
            self.ret.mblock(self.tstp - n, n - m).to_vec()
        } else {
            let src = self.ret.mblock(self.tstp - m, m - n);
            adjoint_neg(src)
        }
    }

    /// Hermitian-reconstructed lesser block `G^{<}(t_n, t_m)`.
    pub fn less_block(&self, n: i64, m: i64) -> Vec<Complex> {
        assert!(self.initialized, "the moving object is not initialized");
        assert!(m >= 0, "({n}, {m}) outside the moving window");
        if n >= m {
            self.less.mblock(self.tstp - n, n - m).to_vec()
        } else {
            let src = self.less.mblock(self.tstp - m, m - n);
            adjoint_neg(src)
        }
    }
}

/// Owning time step of a moving Green's function:
/// `G^{R/<}(t, t - deltj)` for `deltj = 0, .., tc`.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingGreenTimeStep {
    pub tc: i64,
    pub tstp: i64,
    pub sign: Sign,
    pub ret: RowComp,
    pub less: RowComp,
}

impl MovingGreenTimeStep {
    /// Constructor.
    pub fn new(tc: i64, ndim: usize, tstp: i64, sign: Sign) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        MovingGreenTimeStep {
            tc,
            tstp,
            sign,
            ret: RowComp::new(tc, ndim, ndim, Complex::ZERO),
            less: RowComp::new(tc, ndim, ndim, Complex::ZERO),
        }
    }
}

/// Moving single-time function: the slot `delti`
/// contains `f(t0 - delt_i)` and is stored in the physical slot
/// `(head + delti) % (tc + 1)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingTimeFunction {
    pub tc: i64,
    pub tstp: i64,
    pub initialized: bool,
    pub head: i64,
    pub size1: usize,
    pub size2: usize,
    pub(crate) data: Vec<Complex>,
}

impl MovingTimeFunction {
    /// Constructor: uninitialized.
    pub fn new(tc: i64, ndim: usize) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        let n = check_payload(tc, ndim, ndim, 1).expect("moving payload overflow");
        MovingTimeFunction {
            tc,
            tstp: 0,
            initialized: false,
            head: 0,
            size1: ndim,
            size2: ndim,
            data: vec![Complex::ZERO; n],
        }
    }

    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    fn slot_index(&self, delti: i64) -> usize {
        debug_assert!(0 <= delti && delti <= self.tc);
        ((self.head + delti).rem_euclid(self.tc + 1)) as usize
    }

    /// The slot `delti`.
    pub fn fslot(&self, delti: i64) -> &[Complex] {
        assert!(
            0 <= delti && delti <= self.tc,
            "fslot: {delti} out of range"
        );
        let b = self.size1 * self.size2;
        let off = self.slot_index(delti) * b;
        &self.data[off..off + b]
    }

    /// Mutable slot view.
    pub fn fslot_mut(&mut self, delti: i64) -> &mut [Complex] {
        assert!(
            0 <= delti && delti <= self.tc,
            "fslot: {delti} out of range"
        );
        let b = self.size1 * self.size2;
        let off = self.slot_index(delti) * b;
        &mut self.data[off..off + b]
    }

    /// Rotate the ring forward and zero the new leading slot.
    pub fn forward(&mut self) {
        assert!(self.initialized, "the moving object is not initialized");
        self.head = (self.head + self.tc).rem_euclid(self.tc + 1);
        let b = self.size1 * self.size2;
        let off = self.slot_index(0) * b;
        for z in &mut self.data[off..off + b] {
            *z = Complex::ZERO;
        }
        self.tstp += 1;
    }
}

/// `-src^dagger` of a raw block (the hermitian reconstruction).
fn adjoint_neg(src: &[Complex]) -> Vec<Complex> {
    let d2 = src.len();
    let d = (d2 as f64).sqrt() as usize;
    let mut out = vec![Complex::ZERO; d2];
    for a in 0..d {
        for b in 0..d {
            out[a * d + b] = -src[b * d + a].conj();
        }
    }
    out
}

/// Checked number of elements of a moving payload: returns None on overflow.
fn check_payload(tc: i64, ndim1: usize, ndim2: usize, nrow: usize) -> Option<usize> {
    let n = ndim1.checked_mul(ndim2)?;
    let n = n.checked_mul((tc + 1) as usize)?;
    n.checked_mul(nrow)
}

// ------------------------------------------------------------------
// sanity checks of the moving solvers
// ------------------------------------------------------------------

/// Check the moving objects of the truncated solvers: initialized, the
/// same cut-off, the leading time step `tstp`, the matrix size, and
/// `tc >= 2k + 2`.
pub(crate) fn check_moving_solver(tstp: i64, k: i64, objs: &[&MovingGreen]) {
    assert!((1..=5).contains(&k), "solve_order must be in 1:5, got {k}");
    for x in objs {
        assert!(x.initialized, "the moving object is not initialized");
        assert!(x.tc == objs[0].tc, "the cut-offs differ");
        assert!(
            x.tstp == tstp,
            "tstp = {tstp} is not the leading time step {}",
            x.tstp
        );
        assert!(
            x.ret.size1 == objs[0].ret.size1 && x.ret.size2 == objs[0].ret.size2,
            "the matrix sizes differ"
        );
    }
    assert!(
        objs[0].tc >= 2 * k + 2,
        "the truncated solver of order {k} requires tc >= {}, got {}",
        2 * k + 2,
        objs[0].tc
    );
}

/// Check the moving Hamiltonian of the truncated Dyson solver:
/// initialized, `h.tc >= g.tc`, the leading time step, and the matrix size.
pub(crate) fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    check_initialized_cfmoving(h);
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}

/// Throw if the moving single-time function is not initialized.
fn check_initialized_cfmoving(f: &MovingTimeFunction) {
    assert!(f.initialized, "the moving object is not initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::catch_unwind;

    #[test]
    fn moving_type_tree_and_construction() {
        let g = MovingGreen::new(4, 2, Sign::Fermi);
        let _x = MovingGreenTimeStep::new(4, 2, 7, Sign::Fermi);
        let f = MovingTimeFunction::new(4, 2);
        assert_eq!(g.ret.raw().len(), 2 * 2 * 5 * 5);
        assert_eq!(g.less.raw().len(), 2 * 2 * 5 * 5);
        assert_eq!(f.raw().len(), 2 * 2 * 5);
        assert!(!g.initialized && g.tstp == 0 && !f.initialized);
        assert_eq!(g.tc, 4);
        assert_eq!(g.ret.size1, 2);
        assert_eq!(g.sign.as_i64(), Sign::Fermi.as_i64());
        assert_eq!(
            MovingGreen::new(4, 1, Sign::Bose).sign.as_i64(),
            Sign::Bose.as_i64()
        );
        assert!(catch_unwind(|| MovingGreen::new(-1, 2, Sign::Fermi)).is_err());
        assert!(catch_unwind(|| MovingGreen::new(3, 0, Sign::Fermi)).is_err());
        // payload overflow
        assert!(catch_unwind(|| MovingGreen::new(i64::MAX / 2, 2, Sign::Fermi)).is_err());
    }
}
