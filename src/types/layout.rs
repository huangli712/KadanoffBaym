// Project : KadanoffBaym
// File    : src/types/layout.rs
// Purpose : contiguous triangular storage offsets
//
// All components are stored as one contiguous `Vec<Complex>` per component,
// blocks in row-major `(size1 x size2)` order:
//
//   mat   : (ntau + 1) blocks, index n = 0..ntau;
//   ret   : triangle, row tstp = n holds columns j = 0..n,
//           block index triangular_row_offset(n) + j;
//   less  : triangle, row tstp' = j holds columns i = 0..j,
//           block index triangular_row_offset(j) + i (mirror of ret);
//   lmix  : flat (nt + 1) * (ntau + 1), index tstp * (ntau + 1) + n.
//
// triangular_row_offset(n) = n * (n + 1) / 2 (prefix sum of 1, 2, ..., n).

//!
//! *Remarks* : *Contiguous Component Storage*
//!
//! The four independent components of the full contour-ordered Green's
//! function are stored in contiguous arrays, with the orbital dimensions in
//! front, so that each block is a contiguous `ndim1 × ndim2` matrix.
//!
//! * `Gᵐᵃᵗ`  : `data[:,:,m]`,   0 ≤ m ≤ ntau.
//! * `Gʳᵉᵗ`  : `data[:,:,k]`,   k = (i+1)i/2 + j for i ≥ j (lower triangle).
//! * `Gˡᵐⁱˣ` : `data[:,:,i,m]`, 0 ≤ i ≤ ntime, 0 ≤ m ≤ ntau.
//! * `Gˡᵉˢˢ` : `data[:,:,k]`,   k = (j+1)j/2 + i for i ≤ j (upper triangle).
//!
//! The other triangles of `Gʳᵉᵗ` and `Gˡᵉˢˢ` are reconstructed from the
//! stored partner by the hermitian conjugate relations.
//!

use crate::complex::Complex;

/// Matsubara branch marker: `tstp == MATSUBARA` addresses the
/// Matsubara component (`-1`).
pub const MATSUBARA: i64 = -1;

/// Offset (in blocks) of triangular row `n`: the prefix sum
/// `1 + 2 + ... + n`, i.e. `n * (n + 1) / 2`.
#[inline]
pub(crate) fn triangular_row_offset(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Adjoint (conjugate transpose) of a row-major block `src` (`s1 x s2`)
/// into `out` (`s2 x s1`): `out[a][b] = src[b][a]^*`.
#[inline]
pub(crate) fn adjoint_block_into(src: &[Complex], out: &mut [Complex], s1: usize, s2: usize) {
    for a in 0..s1 {
        for b in 0..s2 {
            out[a * s2 + b] = src[b * s1 + a].conj();
        }
    }
}

/// Negative adjoint of a square block `src` (`d x d`) into `out`:
/// `out[a][b] = -src[b][a]^*`. This is the hermitian reconstruction of
/// the non-stored triangle, `G(t_n, t_m) = -G(t_m, t_n)^dagger`.
#[inline]
pub(crate) fn neg_adjoint_block_into(src: &[Complex], out: &mut [Complex], d: usize) {
    for a in 0..d {
        for b in 0..d {
            out[a * d + b] = -src[b * d + a].conj();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_prefix_sums() {
        let mut acc = 0;
        for n in 0..100usize {
            assert_eq!(triangular_row_offset(n), acc);
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
        let last_ret = triangular_row_offset(nt) + nt;
        let last_lmix = nt * (ntau + 1) + ntau;
        assert_eq!(last_ret, tri_len - 1);
        assert_eq!(last_lmix, lmix_len - 1);
        // mat + ret + lmix + less of a full ContourGreen, in bytes
        let nblocks = 2 * tri_len + lmix_len + ntau + 1;
        let nbytes = nblocks * block * 2 * 8;
        assert!(nbytes < usize::MAX / 2);
    }
}
