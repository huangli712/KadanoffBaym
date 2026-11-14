// Project : KadanoffBaym
// File    : src/types/layout.rs
// Purpose : contiguous triangular storage offsets (plan rust-03 §2.1)
//
// All components are stored as one contiguous `Vec<C64>` per component,
// blocks in row-major `(size1 x size2)` order:
//
//   mat   : (ntau + 1) blocks, index n = 0..ntau;
//   ret   : triangle, row tstp = n holds columns j = 0..n,
//           block index tri_off(n) + j;
//   less  : triangle, row tstp' = j holds columns i = 0..j,
//           block index tri_off(j) + i (mirror of ret);
//   lmix  : flat (nt + 1) * (ntau + 1), index tstp * (ntau + 1) + n.
//
// tri_off(n) = n * (n + 1) / 2 (prefix sum of 1, 2, ..., n).

/// Matsubara branch marker: `tstp == MATSUBARA` addresses the
/// Matsubara component (`-1`, decision D6).
pub const MATSUBARA: i64 = -1;

/// Grid geometry and index helpers of the contiguous storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
}

impl Layout {
    pub fn new(nt: usize, ntau: usize, size1: usize, size2: usize) -> Layout {
        assert!(nt >= 1 && ntau >= 1 && size1 >= 1 && size2 >= 1);
        Layout {
            nt,
            ntau,
            size1,
            size2,
        }
    }

    /// Elements per orbital block.
    #[inline]
    pub fn block(&self) -> usize {
        self.size1 * self.size2
    }

    /// Offset (in blocks) of triangular row `n`.
    #[inline]
    pub fn tri_off(n: usize) -> usize {
        n * (n + 1) / 2
    }

    /// Number of blocks of a ret/less triangle: (nt+1)(nt+2)/2.
    #[inline]
    pub fn tri_len(&self) -> usize {
        (self.nt + 1) * (self.nt + 2) / 2
    }

    /// Block index inside the ret triangle for `(tstp, j)` with
    /// `0 <= j <= tstp <= nt`.
    #[inline]
    pub fn ret_idx(&self, tstp: usize, j: usize) -> usize {
        debug_assert!(j <= tstp && tstp <= self.nt);
        Self::tri_off(tstp) + j
    }

    /// Block index inside the less triangle for `(i, tstp)` with
    /// `0 <= i <= tstp <= nt` (the row index is the larger time).
    #[inline]
    pub fn less_idx(&self, i: usize, tstp: usize) -> usize {
        debug_assert!(i <= tstp && tstp <= self.nt);
        Self::tri_off(tstp) + i
    }

    /// Number of blocks of the lmix component: (nt+1) * (ntau+1).
    #[inline]
    pub fn lmix_len(&self) -> usize {
        (self.nt + 1) * (self.ntau + 1)
    }

    /// Block index inside lmix for `(tstp, n)` with
    /// `0 <= tstp <= nt`, `0 <= n <= ntau`.
    #[inline]
    pub fn lmix_idx(&self, tstp: usize, n: usize) -> usize {
        debug_assert!(tstp <= self.nt && n <= self.ntau);
        tstp * (self.ntau + 1) + n
    }

    /// Number of blocks of the mat component (`ntau + 1`).
    #[inline]
    pub fn mat_len(&self) -> usize {
        self.ntau + 1
    }

    /// Validate a tstp for the real-time components (`0..=nt`).
    #[inline]
    pub fn check_tstp(&self, tstp: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "tstp {tstp} out of range 0..={}",
            self.nt
        );
        tstp as usize
    }

    /// Validate a tau index (`0..=ntau`).
    #[inline]
    pub fn check_tau(&self, n: i64) -> usize {
        assert!(
            (0..=self.ntau as i64).contains(&n),
            "tau index {n} out of range 0..={}",
            self.ntau
        );
        n as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_prefix_sums() {
        let mut acc = 0;
        for n in 0..100usize {
            assert_eq!(Layout::tri_off(n), acc);
            acc += n + 1;
        }
    }

    #[test]
    fn small_grid_shapes() {
        let l = Layout::new(1, 1, 1, 1);
        assert_eq!(l.tri_len(), 3);
        assert_eq!(l.lmix_len(), 4);
        assert_eq!(l.mat_len(), 2);
        assert_eq!(l.block(), 1);
        assert_eq!(l.ret_idx(1, 1), 2);
        assert_eq!(l.ret_idx(1, 0), 1);
        assert_eq!(l.less_idx(0, 1), 1);
        assert_eq!(l.lmix_idx(1, 1), 3);
    }

    #[test]
    fn offsets_at_1001_grid_do_not_overflow() {
        // 1001x401 grid (nt = 1000, ntau = 400), 2x2 blocks
        let l = Layout::new(1000, 400, 2, 2);
        let last_ret = l.ret_idx(1000, 1000);
        let last_less = l.less_idx(1000, 1000);
        let last_lmix = l.lmix_idx(1000, 400);
        assert_eq!(last_ret, l.tri_len() - 1);
        assert_eq!(last_less, l.tri_len() - 1);
        assert_eq!(last_lmix, l.lmix_len() - 1);
        // total elements of a full ContourGreen in f64 units
        let nblocks = 2 * l.tri_len() + l.lmix_len() + l.mat_len();
        let nbytes = nblocks * l.block() * 2 * 8;
        assert!(nbytes < usize::MAX / 2);
    }

    #[test]
    #[should_panic]
    fn check_tstp_rejects_out_of_range() {
        let l = Layout::new(10, 10, 1, 1);
        l.check_tstp(11);
    }
}
