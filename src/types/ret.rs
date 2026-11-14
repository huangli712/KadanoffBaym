// Project : KadanoffBaym
// File    : src/types/ret.rs
// Purpose : retarded component container (blueprint: types/ret.jl)

use crate::basics::layout::Layout;
use crate::Complex;

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
            data: vec![v; Layout::tri_off(nt + 1) * block],
        }
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
        Layout::tri_off(tstp as usize) + j as usize
    }

    /// Borrow the whole row `tstp` (blocks `j = 0..=tstp`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }
}

// Two-index raw block accessors (stored domain only). MatComp uses its
// single-index `block(m)`.
impl RetComp {
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
