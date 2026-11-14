//
// Project : KadanoffBaym
// File    : src/types/function.rs
// Purpose : single-time function (blueprint: types/function.jl `Cf`)

use crate::basics::layout::MATSUBARA;
use crate::Complex;

/// Single-time function (blueprint `Cf`): one matrix block per real-time
/// step `0..=nt` plus one Matsubara slot.
///
/// Storage: `data[0..=nt]` are the time rows, `data[nt + 1]` is the
/// Matsubara value (Julia `cf[0]` <-> `get(MATSUBARA)`).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeFunction {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl TimeFunction {
    /// Constructor (blueprint `Cf(ntime, ndim1, ndim2, v)` with the -1
    /// translation): `nt >= 0`, all elements set to `v`.
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> TimeFunction {
        let block = size1 * size2;
        TimeFunction {
            nt,
            size1,
            size2,
            block,
            data: vec![v; (nt + 2) * block],
        }
    }

    /// Zero-initialized constructor.
    pub fn zeros(nt: usize, size1: usize, size2: usize) -> TimeFunction {
        TimeFunction::new(nt, size1, size2, Complex::ZERO)
    }

    /// Borrow the whole raw storage (time rows then Matsubara slot).
    #[inline]
    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    /// Mutably borrow the whole raw storage.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// Borrow the block of `tstp` as a row-major slice.
    #[inline]
    pub fn block(&self, tstp: i64) -> &[Complex] {
        let k = self.idx(tstp);
        &self.data[k * self.block..(k + 1) * self.block]
    }

    /// Mutably borrow the block of `tstp`.
    #[inline]
    pub fn block_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = self.idx(tstp);
        &mut self.data[k * self.block..(k + 1) * self.block]
    }

    /// Element access at `(tstp, a, b)` (0-based; `MATSUBARA` for the
    /// Matsubara slot).
    #[inline]
    pub fn get(&self, tstp: i64, a: usize, b: usize) -> Complex {
        self.data[self.idx(tstp) * self.block + a * self.size2 + b]
    }

    /// Element write at `(tstp, a, b)`.
    #[inline]
    pub fn set(&mut self, tstp: i64, a: usize, b: usize, v: Complex) {
        let i = self.idx(tstp) * self.block + a * self.size2 + b;
        self.data[i] = v;
    }

    /// Block index for timestep `tstp` (`0..=nt` or `MATSUBARA`).
    #[inline]
    pub fn idx(&self, tstp: i64) -> usize {
        if tstp == MATSUBARA {
            self.nt + 1
        } else {
            assert!(
                (0..=self.nt as i64).contains(&tstp),
                "TimeFunction: tstp {tstp} out of range 0..={}",
                self.nt
            );
            tstp as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_function_matsubara_slot() {
        let mut cf = TimeFunction::new(3, 1, 1, Complex::ZERO);
        assert_eq!(cf.block(0).len(), 1);
        cf.set(MATSUBARA, 0, 0, Complex::new(9.0, 0.0));
        cf.set(2, 0, 0, Complex::new(2.0, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), Complex::new(9.0, 0.0));
        assert_eq!(cf.get(2, 0, 0), Complex::new(2.0, 0.0));
        assert_eq!(cf.block(MATSUBARA)[0], Complex::new(9.0, 0.0));
    }
}
