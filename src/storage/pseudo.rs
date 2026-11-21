// Project : KadanoffBaym
// File    : src/storage/pseudo.rs
// Purpose : block accessors of the pseudo-particle Green's functions

use crate::linalg::DenseMat;
use crate::types::ContourGreen;

/// The greater block `G^{>}(t_n, t_j)` of the pseudo-particle Green's
/// function (the stored `ret` component); for `j > n` the value is
/// reconstructed as `-(G^{>}(t_j, t_n))^dagger`.
pub(crate) fn gtr_blk(g: &ContourGreen, n: i64, j: i64) -> DenseMat {
    if n >= j {
        DenseMat::from_slice(g.size1, g.size2, g.ret.raw_block_at(n, j))
    } else {
        let src = g.ret.raw_block_at(j, n);
        let d1 = g.size1;
        let d2 = g.size2;
        let mut out = DenseMat::zeros_shape(d1, d2);
        for a in 0..d1 {
            for b in 0..d2 {
                out.set(a, b, -src[b * d2 + a].conj());
            }
        }
        out
    }
}
