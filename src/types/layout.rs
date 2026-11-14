// Project : KadanoffBaym
// File    : src/types/layout.rs
// Purpose : contiguous triangular storage offsets (plan rust-03 §2.1)
//
// All components are stored as one contiguous `Vec<Complex>` per component,
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

/// Offset (in blocks) of triangular row `n`: the prefix sum
/// `1 + 2 + ... + n`, i.e. `n * (n + 1) / 2`.
#[inline]
pub fn tri_off(n: usize) -> usize {
    n * (n + 1) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_prefix_sums() {
        let mut acc = 0;
        for n in 0..100usize {
            assert_eq!(tri_off(n), acc);
            acc += n + 1;
        }
    }

    #[test]
    fn offsets_at_1001_grid_do_not_overflow() {
        // 1001x401 grid (nt = 1000, ntau = 400), 2x2 orbital blocks: the
        // last block indices and the total storage stay far below
        // usize::MAX
        let (nt, ntau, block) = (1000usize, 400usize, 4usize);
        let tri_len = (nt + 1) * (nt + 2) / 2;
        let lmix_len = (nt + 1) * (ntau + 1);
        let last_ret = tri_off(nt) + nt;
        let last_lmix = nt * (ntau + 1) + ntau;
        assert_eq!(last_ret, tri_len - 1);
        assert_eq!(last_lmix, lmix_len - 1);
        // mat + ret + lmix + less of a full ContourGreen, in bytes
        let nblocks = 2 * tri_len + lmix_len + ntau + 1;
        let nbytes = nblocks * block * 2 * 8;
        assert!(nbytes < usize::MAX / 2);
    }
}
