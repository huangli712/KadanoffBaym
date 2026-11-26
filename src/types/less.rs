//
// Project : KadanoffBaym
// File    : src/types/less.rs
// Purpose : lesser component container

//!
//! *Remarks* : *Lesser Green's Function*
//!
//! The lesser component of contour-ordered Green's function reads
//!
//! ```math
//! \begin{equation}
//! G^{<}(t,t') = \mp i \langle c^{\dagger}(t') c(t) \rangle,
//! \end{equation}
//! ```
//!
//! where ``t,\ t' \in \mathcal{C}_1 \cup \mathcal{C}_2``. We choose the
//! upper (lower) sign if the operators ``c`` and ``c^{\dagger}`` are
//! bosonic (fermionic). Its hermitian conjugate yields
//!
//! ```math
//! \begin{equation}
//! G^{<}(t,t')^{*} = -G^{<}(t',t),
//! \end{equation}
//! ```
//!
//! or equivalently
//!
//! ```math
//! \begin{equation}
//! G^{<}(t,t') = -G^{<}(t',t)^{*},
//! \end{equation}
//! ```
//!
//! and
//!
//! ```math
//! \begin{equation}
//! G^{>}(t,t') = -G^{>}(t',t)^{*},
//! \end{equation}
//! ```
//!
//! The lesser component is related to the retarded, advanced, and Keldysh
//! Green's functions via
//!
//! ```math
//! \begin{equation}
//! G^{<} = \frac{1}{2}(G^{K} - G^{R} + G^{A}).
//! \end{equation}
//! ```
//!
//! *Remarks* : *Greater Green's Function*
//!
//! The greater component of contour-ordered Green's function reads
//!
//! ```math
//! \begin{equation}
//! G^{>}(t,t') = - i \langle c(t) c^{\dagger}(t') \rangle,
//! \end{equation}
//! ```
//!
//! where ``t,\ t' \in \mathcal{C}_1 \cup \mathcal{C}_2``. Its hermitian
//! conjugate yields
//!
//! ```math
//! \begin{equation}
//! G^{>}(t,t')^{*} = -G^{>}(t',t).
//! \end{equation}
//! ```
//!
//! The greater component is related to the retarded, advanced, and Keldysh
//! Green's functions via
//!
//! ```math
//! \begin{equation}
//! G^{>} = \frac{1}{2}(G^{K} + G^{R} - G^{A}).
//! \end{equation}
//! ```

use crate::complex::Complex;

use crate::types::triangular_row_offset;

/// Lesser component: mirror triangle, row `tstp` (the larger time)
/// holds columns `i = 0..=tstp`, block index `triangular_row_offset(tstp) + i`.
#[derive(Clone, Debug, PartialEq)]
pub struct LessComp {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl LessComp {
    /// Mirror triangle for `nt` (rows 0..=nt).
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> LessComp {
        let block = size1 * size2;
        LessComp {
            nt,
            size1,
            size2,
            block,
            data: vec![v; triangular_row_offset(nt + 1) * block],
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
        assert!(
            a < self.size1 && b_orb < self.size2,
            "less: element ({a}, {b_orb}) is out of bounds for the {} x {} block",
            self.size1,
            self.size2
        );
        self.data[k * self.block + a * self.size2 + b_orb]
    }

    /// Element write of block `k` at `(a, b_orb)`.
    #[inline]
    pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
        assert!(
            a < self.size1 && b_orb < self.size2,
            "less: element ({a}, {b_orb}) is out of bounds for the {} x {} block",
            self.size1,
            self.size2
        );
        let i = k * self.block + a * self.size2 + b_orb;
        self.data[i] = v;
    }

    /// Validate `(i, tstp)` with `0 <= i <= tstp <= nt` and return the
    /// block index.
    #[inline]
    pub fn idx(&self, i: i64, tstp: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "less: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=tstp).contains(&i),
            "less: column {i} out of range 0..={tstp}"
        );
        triangular_row_offset(tstp as usize) + i as usize
    }

    /// Borrow the whole row `tstp` (blocks `i = 0..=tstp`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = triangular_row_offset(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub(crate) fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = triangular_row_offset(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }

    /// Borrow the stored block `less(i, j)` (`i <= j`).
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j))
    }

    /// Mutably borrow the stored block `less(i, j)` (`i <= j`).
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j);
        self.block_mut(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gless_struct_constructors() {
        // ntime = 201 points -> triangle 201*202/2
        let s1 = LessComp::new(200, 2, 2, Complex::ZERO);
        let s2 = LessComp::new(200, 2, 2, Complex::ZERO);
        assert!(s1.equal(&s2));
        assert_eq!(s1.nblocks(), 201 * 202 / 2);
    }
}
