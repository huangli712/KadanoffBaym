// Project : KadanoffBaym
// File    : src/weights/calc_backward.rs
// Purpose : backward differentiation (BDF) weight calculator
//           (blueprint: julia/src/weights/calc.jl)

/// Backward differentiation weights (blueprint Eq. 90):
/// `Wb[j] = -Wi[a=1][j]`, length `k + 1`.
pub(crate) fn calc_backward_differentiation(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    (0..n).map(|j| -wi[1 * n + j]).collect()
}
