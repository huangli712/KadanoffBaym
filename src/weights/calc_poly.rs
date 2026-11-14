// Project : KadanoffBaym
// File    : src/weights/calc_poly.rs
// Purpose : polynomial interpolation / differentiation / integration
//           weight calculators (blueprint: julia/src/weights/calc.jl)

use crate::complex::Complex;
use crate::linalg::mat::DenseMat;

/// Polynomial interpolation weights `Wi[i][j] = (M^{-1})_{ij}` with
/// `M_{ja} = j^a`, row-major `(k+1) * (k+1)` (blueprint Eq. 80).
pub(crate) fn calc_poly_interpolation(k: i64) -> Vec<f64> {
    let n = (k + 1) as usize;
    let mut m = DenseMat::zeros(n);
    for i in 0..n {
        for j in 0..n {
            // M[i][j] = i^j (Julia: i^j with 0^0 = 1)
            let v = (i as f64).powi(j as i32);
            m.set(i, j, Complex::new(v, 0.0));
        }
    }
    let inv = crate::linalg::lu::mat_inv(&m);
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = inv.get(i, j).re;
        }
    }
    out
}

/// Polynomial differentiation weights (blueprint Eq. 85), row-major
/// `(k+1) * (k+1)`, from the interpolation weights `wi`.
pub(crate) fn calc_poly_differentiation(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    let mut wd = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = 0.0;
            for a in 1..n {
                acc += wi[a * n + j] * a as f64 * (i as f64).powi(a as i32 - 1);
            }
            wd[i * n + j] = acc;
        }
    }
    wd
}

/// Polynomial integration weights `Wt[m][n][l]` (blueprint Eq. 89),
/// row-major `(k+1)^3` indexed as `(m * (k+1) + n) * (k+1) + l`.
pub(crate) fn calc_poly_integration(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    let mut wt = vec![0.0f64; n * n * n];
    for l in 0..n {
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for a in 0..n {
                    let af = a as f64;
                    acc += wi[a * n + l]
                        * ((j as f64).powi(a as i32 + 1) - (i as f64).powi(a as i32 + 1))
                        / (af + 1.0);
                }
                wt[(i * n + j) * n + l] = acc;
            }
        }
    }
    wt
}
