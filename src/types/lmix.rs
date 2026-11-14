//
// Project : KadanoffBaym
// File    : src/types/lmix.rs
// Purpose : mixed component container (blueprint: types/lmix.jl)

use crate::Complex;

/// Mixed component: flat `(nt + 1) x (ntau + 1)` blocks,
/// `G^mix(tstp, tau_n)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LmixComp {
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl LmixComp {
    pub fn new(nt: usize, ntau: usize, size1: usize, size2: usize, v: Complex) -> LmixComp {
        let block = size1 * size2;
        LmixComp {
            nt,
            ntau,
            size1,
            size2,
            block,
            data: vec![v; (nt + 1) * (ntau + 1) * block],
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

    /// Validate `(tstp, n)` and return the block index.
    #[inline]
    pub fn idx(&self, tstp: i64, n: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "lmix: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=self.ntau as i64).contains(&n),
            "lmix: tau index {n} out of range 0..={}",
            self.ntau
        );
        tstp as usize * (self.ntau + 1) + n as usize
    }

    /// Borrow the whole row `tstp` (blocks `n = 0..=ntau`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = self.idx(tstp, 0);
        let n = (self.ntau + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = self.idx(tstp, 0);
        let n = (self.ntau + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }

    /// Borrow the stored block `lmix(i, j)`.
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j) as usize)
    }

    /// Mutably borrow the stored block `lmix(i, j)`.
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j) as usize;
        self.block_mut(k)
    }
}
