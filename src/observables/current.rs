//
// Project : KadanoffBaym
// File    : src/observables/current.rs
// Purpose : current observables
//
// Blueprint: julia/src/observables/ (current).

use crate::complex::Complex;
use crate::linalg::DenseMat;

/// Discrete k sum of the current `J = sum_k w_k Re Tr[v_k rho_k]`
/// (blueprint `current`).
pub fn current(v: &[DenseMat], rho: &[DenseMat], w: &[f64]) -> f64 {
    assert_eq!(v.len(), rho.len());
    assert_eq!(v.len(), w.len());
    let mut j = 0.0;
    for k in 0..v.len() {
        let p = v[k].mat_mul(&rho[k]);
        let mut tr = Complex::ZERO;
        for i in 0..v[k].nrows() {
            tr += p.get(i, i);
        }
        j += w[k] * tr.re;
    }
    j
}
