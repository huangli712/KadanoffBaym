// Project : KadanoffBaym
// File    : src/types/moving.rs
// Purpose : the moving (truncated) window types (blueprint
//           julia/src/types/moving.jl)
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
// Index convention (D6): the leading time step `tstp` is the 0-based
// real time step; the blueprint KB step n0 corresponds to n0 - 1.

use crate::Complex;
use crate::types::Sign;

/// Ring-buffer component (retarded or lesser) of a moving Green's
/// function (blueprint `Gʳᵉᵗᵐᵒᵛ` / `Gˡᵉˢˢᵐᵒᵛ`).
///
/// The payload is `(tc + 1)` physical rows of `(tc + 1)` blocks each,
/// row-major: the block `(delti, deltj)` lives at
/// `(physical_row(delti) * (tc + 1) + deltj) * block`.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingComp {
    pub tc: i64,
    pub size1: usize,
    pub size2: usize,
    pub head: i64,
    pub(crate) data: Vec<Complex>,
}

impl MovingComp {
    pub fn new(tc: i64, size1: usize, size2: usize, v: Complex) -> Self {
        let n = check_payload(tc, size1, size2, (tc + 1) as usize)
            .expect("moving payload overflow");
        MovingComp {
            tc,
            size1,
            size2,
            head: 0,
            data: vec![v; n],
        }
    }

    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    pub fn block(&self) -> usize {
        self.size1 * self.size2
    }

    /// Physical row of the logical row `delti` (blueprint `_mrow`).
    #[inline]
    pub fn mrow_index(&self, delti: i64) -> usize {
        debug_assert!(0 <= delti && delti <= self.tc);
        ((self.head + delti).rem_euclid(self.tc + 1)) as usize
    }

    /// The `(tc + 1)` blocks of the logical row `delti`, contiguous
    /// (blueprint `_mrow_view`).
    pub fn mrow(&self, delti: i64) -> &[Complex] {
        let p = self.mrow_index(delti);
        let b = self.block();
        &self.data[p * (self.tc + 1) as usize * b..(p + 1) * (self.tc + 1) as usize * b]
    }

    /// Mutable row view (blueprint `_mrow_view`, write-through).
    pub fn mrow_mut(&mut self, delti: i64) -> &mut [Complex] {
        let p = self.mrow_index(delti);
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        &mut self.data[p * tcl * b..(p + 1) * tcl * b]
    }

    /// The block `(delti, deltj)` (blueprint `_mblock`).
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

    /// Mutable block view (blueprint `_mblock`, write-through).
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
    /// moves onto the slot of the oldest row) and zero the new leading
    /// row (blueprint part of `forward!`).
    pub fn rotate_forward(&mut self) {
        self.head = (self.head + self.tc).rem_euclid(self.tc + 1);
        let b = self.block();
        let tcl = (self.tc + 1) as usize;
        let p = self.mrow_index(0);
        for z in &mut self.data[p * tcl * b..(p + 1) * tcl * b] {
            *z = Complex::ZERO;
        }
    }
}

/// Row component of a moving time step (blueprint `gʳᵉᵗᵐᵒᵛ` /
/// `gˡᵉˢˢᵐᵒᵛ`): the `(tc + 1)` blocks `G(t, t - deltj)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingRowComp {
    pub tc: i64,
    pub size1: usize,
    pub size2: usize,
    pub(crate) data: Vec<Complex>,
}

impl MovingRowComp {
    pub fn new(tc: i64, size1: usize, size2: usize, v: Complex) -> Self {
        let n = check_payload(tc, size1, size2, 1).expect("moving row payload overflow");
        MovingRowComp {
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

    /// The block `deltj` (blueprint `_mblock` on the vector types).
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

/// Moving (truncated) contour-ordered Green's function (blueprint
/// `ℱᵐᵒᵛ`). `tstp` is the 0-based real time step of the leading row
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
    pub ret: MovingComp,
    pub less: MovingComp,
}

impl MovingGreen {
    /// Constructor (blueprint `ℱᵐᵒᵛ(tc, ndim, sign)`): an
    /// uninitialized moving Green's function.
    pub fn new(tc: i64, ndim: usize, sign: Sign) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        MovingGreen {
            tc,
            tstp: 0,
            initialized: false,
            sign,
            generation: 0,
            ret: MovingComp::new(tc, ndim, ndim, Complex::ZERO),
            less: MovingComp::new(tc, ndim, ndim, Complex::ZERO),
        }
    }

    /// Hermitian-reconstructed retarded block `G^{R}(t_n, t_m)` inside
    /// the window; `n >= m` reads the stored block, `n < m` returns
    /// `-G^{R}(t_m, t_n)^dagger` (blueprint layout Remarks).
    pub fn ret_block(&self, n: i64, m: i64) -> Vec<Complex> {
        assert!(self.initialized, "the moving object is not initialized");
        assert!(m >= 0, "mrel: ({n}, {m}) outside the moving window");
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
        assert!(m >= 0, "mrel: ({n}, {m}) outside the moving window");
        if n >= m {
            self.less.mblock(self.tstp - n, n - m).to_vec()
        } else {
            let src = self.less.mblock(self.tstp - m, m - n);
            adjoint_neg(src)
        }
    }

    /// Translate absolute time steps `(n, m)`, `n >= m`, to
    /// `(delti, deltj)` (blueprint `_mrel`); panics outside the window.
    pub fn mrel(&self, n: i64, m: i64) -> (i64, i64) {
        assert!(self.initialized, "the moving object is not initialized");
        let delti = self.tstp - n;
        let deltj = n - m;
        assert!(
            m >= 0 && 0 <= delti && delti <= self.tc && 0 <= deltj && deltj <= self.tc,
            "mrel: ({n}, {m}) outside the window [{}..={}]",
            self.tstp - self.tc,
            self.tstp
        );
        (delti, deltj)
    }
}

/// Owning time step of a moving Green's function (blueprint `𝒻ᵐᵒᵛ`):
/// `G^{R/<}(t, t - deltj)` for `deltj = 0, .., tc`.
#[derive(Clone, Debug, PartialEq)]
pub struct MovingGreenTimeStep {
    pub tc: i64,
    pub tstp: i64,
    pub sign: Sign,
    pub ret: MovingRowComp,
    pub less: MovingRowComp,
}

impl MovingGreenTimeStep {
    /// Constructor (blueprint `𝒻ᵐᵒᵛ(tc, ndim, tstp, sign)`).
    pub fn new(tc: i64, ndim: usize, tstp: i64, sign: Sign) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        MovingGreenTimeStep {
            tc,
            tstp,
            sign,
            ret: MovingRowComp::new(tc, ndim, ndim, Complex::ZERO),
            less: MovingRowComp::new(tc, ndim, ndim, Complex::ZERO),
        }
    }
}

/// Moving single-time function (blueprint `Cfᵐᵒᵛ`): the slot `delti`
/// contains `f(t0 - delt_i)` and is stored in the physical slot
/// `(head + delti) % (tc + 1)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CfMoving {
    pub tc: i64,
    pub tstp: i64,
    pub initialized: bool,
    pub head: i64,
    pub size1: usize,
    pub size2: usize,
    pub(crate) data: Vec<Complex>,
}

impl CfMoving {
    /// Constructor (blueprint `Cfᵐᵒᵛ(tc, ndim)`): uninitialized.
    pub fn new(tc: i64, ndim: usize) -> Self {
        assert!(tc >= 0, "tc must be >= 0, got {tc}");
        assert!(ndim >= 1, "ndim must be >= 1, got {ndim}");
        let n = check_payload(tc, ndim, ndim, 1).expect("moving payload overflow");
        CfMoving {
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

    /// The slot `delti` (blueprint `_fslot`).
    pub fn fslot(&self, delti: i64) -> &[Complex] {
        assert!(0 <= delti && delti <= self.tc, "fslot: {delti} out of range");
        let b = self.size1 * self.size2;
        let off = self.slot_index(delti) * b;
        &self.data[off..off + b]
    }

    /// Mutable slot view.
    pub fn fslot_mut(&mut self, delti: i64) -> &mut [Complex] {
        assert!(0 <= delti && delti <= self.tc, "fslot: {delti} out of range");
        let b = self.size1 * self.size2;
        let off = self.slot_index(delti) * b;
        &mut self.data[off..off + b]
    }

    /// Rotate the ring forward and zero the new leading slot
    /// (blueprint `forward!(f::Cfᵐᵒᵛ)`).
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
pub(crate) fn adjoint_neg(src: &[Complex]) -> Vec<Complex> {
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

/// Checked number of elements of a moving payload (blueprint
/// `_check_payload`): returns None on overflow.
pub(crate) fn check_payload(tc: i64, ndim1: usize, ndim2: usize, nrow: usize) -> Option<usize> {
    let n = ndim1.checked_mul(ndim2)?;
    let n = n.checked_mul((tc + 1) as usize)?;
    n.checked_mul(nrow)
}
