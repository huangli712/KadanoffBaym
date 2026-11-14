//
// Project : KadanoffBaym
// File    : src/types/ret.rs
// Purpose : retarded component container (blueprint: types/ret.jl)
//!
//! *Remarks : Retarded Green's Function*
//!
//! The retarded component of contour-ordered Green's function reads
//!
//! ```math
//! \begin{equation}
//! G^{R}(t,t') =
//!     -i \theta(t-t') \langle [c(t), c^{\dagger}(t')]_{\mp} \rangle,
//! \end{equation}
//! ```text
//!
//! Here, ``t``, ``t'`` belong to ``\mathcal{C}_1 ∪ \mathcal{C}_2``,
//! ``\theta(t)`` is a step function, ``[,]_{-(+)}`` denotes an
//! (anti-)commutator. We choose the -(+) sign if the operators ``c``
//! and ``c^{\dagger}`` are bosonic (fermionic).
//!
//! The retarded component is related to the advanced component by
//! hermitian conjugate:
//!
//! ```math
//! \begin{equation}
//! G^{R}(t,t') = G^{A}(t',t)^{*},
//! \end{equation}
//! ```text
//!
//! and
//!
//! ```math
//! \begin{equation}
//! G^{R}(t,t')^{*} = G^{A}(t',t).
//! \end{equation}
//! ```text
//!
//! The retarded component can be calculated with the lesser and greater
//! components:
//!
//! ```math
//! \begin{equation}
//! G^{R}(t,t') = \theta(t-t')[G^{>}(t,t') - G^{<}(t,t')].
//! \end{equation}
//! ```text
//!
//! Note that ``G^{R}(t,t') = 0`` if ``t' > t``, which expresses the causality
//! of the retarded component. However, for the implementation of numerical
//! algorithms, it can be more convenient to drop the Heaviside function in
//! the above equation. Therefore, we define a modified retarded component by
//!
//! ```math
//! \begin{equation}
//! \tilde{G}^{R}(t,t') = G^{>}(t,t') - G^{<}(t,t').
//! \end{equation}
//! ```text
//!
//! Its hermitian conjugate is as follows:
//!
//! ```math
//! \begin{equation}
//! \tilde{G}^{R}(t,t') = -\tilde{G}^{R}(t',t)^{*}.
//! \end{equation}
//! ```text
//!
//! *Reference* :
//!
//! Please see [`NESSi`] Session `3` [especially Eq. (20)] for more details.
//!
//! *Remarks : Advanced Green's Function*
//!
//! The advanced component of contour-ordered Green's function reads
//!
//! ```math
//! \begin{equation}
//! G^{A}(t,t') =
//!     i \theta(t'-t) \langle [c(t), c^{\dagger}(t')]_{\mp} \rangle,
//! \end{equation}
//! ```text
//!
//! Here, ``t``, ``t'`` belong to ``\mathcal{C}_1 ∪ \mathcal{C}_2``,
//! ``\theta(t)`` is a step function, ``[,]_{-(+)}`` denotes an
//! (anti-)commutator. We choose the -(+) sign if the operators ``c``
//! and ``c^{\dagger}`` are bosonic (fermionic).

use crate::types::layout::tri_off;
use crate::complex::Complex;

/// Retarded component: lower triangle, row `tstp` holds columns
/// `j = 0..=tstp`, block index `tri_off(tstp) + j`.
#[derive(Clone, Debug, PartialEq)]
pub struct RetComp {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl RetComp {
    /// Full triangle for `nt` (rows 0..=nt).
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> RetComp {
        let block = size1 * size2;
        RetComp {
            nt,
            size1,
            size2,
            block,
            data: vec![v; tri_off(nt + 1) * block],
        }
    }

    /// Number of stored blocks.
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.data.len() / self.block
    }

    /// Borrow the whole raw storage (blocks in row-major order).
    #[inline]
    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    /// Mutably borrow the whole raw storage.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// Borrow block `k` as a row-major `(size1 x size2)` slice.
    #[inline]
    pub fn block(&self, k: usize) -> &[Complex] {
        let b = self.block;
        &self.data[k * b..(k + 1) * b]
    }

    /// Mutably borrow block `k`.
    #[inline]
    pub fn block_mut(&mut self, k: usize) -> &mut [Complex] {
        let b = self.block;
        &mut self.data[k * b..(k + 1) * b]
    }

    /// Element access of block `k` at `(a, b_orb)` (0-based).
    #[inline]
    pub fn get(&self, k: usize, a: usize, b_orb: usize) -> Complex {
        self.data[k * self.block + a * self.size2 + b_orb]
    }

    /// Element write of block `k` at `(a, b_orb)`.
    #[inline]
    pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
        let i = k * self.block + a * self.size2 + b_orb;
        self.data[i] = v;
    }

    /// Validate `(tstp, j)` with `0 <= j <= tstp <= nt` and return the
    /// block index.
    #[inline]
    pub fn idx(&self, tstp: i64, j: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "ret: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=tstp).contains(&j),
            "ret: column {j} out of range 0..={tstp}"
        );
        tri_off(tstp as usize) + j as usize
    }

    /// Borrow the whole row `tstp` (blocks `j = 0..=tstp`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }

    /// Borrow the stored block `ret(i, j)` (`j <= i`).
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j) as usize)
    }

    /// Mutably borrow the stored block `ret(i, j)` (`j <= i`).
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j) as usize;
        self.block_mut(k)
    }
}
