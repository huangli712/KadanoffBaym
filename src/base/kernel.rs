//
// Project : KadanoffBaym
// File    : src/base/kernel.rs
// Purpose : retarded kernel blocks shared by the Dyson and VIE2 start-up
//

use crate::complex::Complex;
use crate::types::ContourGreen;

use crate::types::adjoint_block_into;

/// Return `(K^R(n, l), sign)` for 0-based indices: the block itself with
/// the sign `1` for `n >= l`, or its adjoint partner `Kcc^R(l, n)` with
/// the sign `-1` otherwise. The Dyson start-up passes the self-energy as
/// both `k` and `kcc`; the VIE2 start-up passes `F` and `Fcc`.
pub(crate) fn kernel_ret(
    k: &ContourGreen,
    kcc: &ContourGreen,
    n: i64,
    l: i64,
) -> (Vec<Complex>, f64) {
    if n >= l {
        (k.ret.raw_block_at(n, l).to_vec(), 1.0)
    } else {
        let mut out = vec![Complex::ZERO; kcc.ret.size1 * kcc.ret.size2];
        adjoint_block_into(
            kcc.ret.raw_block_at(l, n),
            &mut out,
            kcc.ret.size2,
            kcc.ret.size1,
        );
        (out, -1.0)
    }
}
