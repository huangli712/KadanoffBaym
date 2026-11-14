//
// Project : KadanoffBaym
// File    : src/types/mat.rs
// Purpose : Matsubara component container (blueprint: types/mat.jl)
//!
//! *Remarks : Matsubara Green's Function*
//!
//! The Matsubara component of contour-ordered Green's function reads
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau, \tau') =
//!     -\langle \mathcal{T}_{\tau} c(\tau) c^{\dagger}(\tau') \rangle,
//! \end{equation}
//! ```text
//!
//! where ``\tau`` and ``\tau'`` lie in ``\mathcal{C}_3``.
//!
//! Note that the Matsubara component ``G^{M}`` plays a somewhat special
//! role, since it is always translationally invariant (``\mathcal{H}``
//! does not depend on imaginary time):
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau,\tau') \equiv G^{M}(\tau - \tau').
//! \end{equation}
//! ```text
//!
//! Furthermore, it is real (Hermitian),
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau)^{*} = G^{M}(\tau),
//! \end{equation}
//! ```text
//!
//! and it is also periodic (antiperiodic) for bosons (fermions),
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau) = \pm G^{M}(\tau + \beta).
//! \end{equation}
//! ```text
//!
//! One can thus use its Fourier decomposition in terms of Matsubara
//! frequencies
//!
//! ```math
//! \begin{equation}
//! G^{M}(\tau,\tau') =
//!     T \sum_n e^{-i\omega_n(\tau-\tau')} G^{M}(i\omega_n),
//! \end{equation}
//! ```text
//!
//! and
//!
//! ```math
//! \begin{equation}
//! G^{M}(i\omega_n) = \int^{\beta}_0 d\tau e^{i\omega_n}G^{M}(\tau).
//! \end{equation}
//! ```text

use crate::complex::Complex;

/// Matsubara component: `ntau + 1` blocks, `G^M(tau_n)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MatComp {
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl MatComp {
    pub fn new(ntau: usize, size1: usize, size2: usize, v: Complex) -> MatComp {
        let block = size1 * size2;
        MatComp {
            ntau,
            size1,
            size2,
            block,
            data: vec![v; (ntau + 1) * block],
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

    /// Validate a tau index and return it (0-based).
    #[inline]
    pub fn check_tau(&self, n: i64) -> usize {
        assert!(
            (0..=self.ntau as i64).contains(&n),
            "mat: tau index {n} out of range 0..={}",
            self.ntau
        );
        n as usize
    }
}
