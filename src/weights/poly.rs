// Project : KadanoffBaym
// File    : src/weights/poly.rs
// Purpose : polynomial interpolation / differentiation / integration
//           weight calculators

use crate::weights::rational::Rational;

/// Exact inverse of the Vandermonde matrix `M[i][j] = i^j` of the nodes
/// `0, ..., n-1`, row-major `n * n`:
/// `(M^{-1})_{p l} = (-1)^(n-1-p) e_{n-1-p} / prod_{m != l} (l - m)`, where
/// `e_j` is the elementary symmetric polynomial of degree `j` of the nodes
/// other than `l`.
pub(super) fn vandermonde_inverse(n: usize) -> Vec<Rational> {
    let mut inv = vec![Rational::from_int(0); n * n];
    for l in 0..n {
        let others: Vec<i128> = (0..n as i128).filter(|m| *m != l as i128).collect();
        let den: i128 = others.iter().map(|m| l as i128 - m).product();
        // elementary symmetric polynomials e_0, ..., e_{n-1} of the other nodes
        let mut e = vec![0i128; n];
        e[0] = 1;
        let mut count = 0usize;
        for x in &others {
            count += 1;
            for j in (1..=count).rev() {
                e[j] += x * e[j - 1];
            }
        }
        for p in 0..n {
            let sign: i128 = if (n - 1 - p).is_multiple_of(2) { 1 } else { -1 };
            inv[p * n + l] = Rational::new(sign * e[n - 1 - p], den);
        }
    }
    inv
}

/// Polynomial interpolation weights `Wi[i][j] = (M^{-1})_{ij}` with
/// `M_{ja} = j^a`, row-major `(k+1) * (k+1)`. The table is exact up to
/// one rounding per entry.
pub(super) fn calc_poly_interpolation(k: i64) -> Vec<f64> {
    vandermonde_inverse((k + 1) as usize)
        .iter()
        .map(|r| r.to_f64())
        .collect()
}

/// Polynomial differentiation weights, row-major
/// `(k+1) * (k+1)`, summed exactly from the interpolation weights.
pub(super) fn calc_poly_differentiation(k: i64) -> Vec<f64> {
    let n = (k + 1) as usize;
    let wi = vandermonde_inverse(n);
    let mut wd = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = Rational::from_int(0);
            for a in 1..n {
                let w = &wi[a * n + j];
                let f = a as i128 * (i as i128).pow(a as u32 - 1);
                acc = acc.add(Rational::new(w.num * f, w.den));
            }
            wd[i * n + j] = acc.to_f64();
        }
    }
    wd
}

/// Polynomial integration weights `Wt[m][n][l]` ,
/// row-major `(k+1)^3` indexed as `(m * (k+1) + n) * (k+1) + l`, summed
/// exactly from the interpolation weights.
pub(super) fn calc_poly_integration(k: i64) -> Vec<f64> {
    let n = (k + 1) as usize;
    let wi = vandermonde_inverse(n);
    let mut wt = vec![0.0f64; n * n * n];
    for l in 0..n {
        for j in 0..n {
            for i in 0..n {
                let mut acc = Rational::from_int(0);
                for a in 0..n {
                    let diff = (j as i128).pow(a as u32 + 1) - (i as i128).pow(a as u32 + 1);
                    let w = &wi[a * n + l];
                    acc = acc.add(Rational::new(w.num * diff, w.den * (a as i128 + 1)));
                }
                wt[(i * n + j) * n + l] = acc.to_f64();
            }
        }
    }
    wt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vandermonde_inverse_is_the_exact_inverse() {
        for n in 1..=12usize {
            let inv = vandermonde_inverse(n);
            for i in 0..n {
                for l in 0..n {
                    // (M * M^{-1})_{i l} = sum_j i^j (M^{-1})_{j l} = delta_{i l}
                    let mut acc = Rational::from_int(0);
                    for j in 0..n {
                        let v = &inv[j * n + l];
                        acc = acc.add(Rational::new((i as i128).pow(j as u32) * v.num, v.den));
                    }
                    assert_eq!(
                        acc,
                        Rational::from_int(if i == l { 1 } else { 0 }),
                        "n = {n}, (i, l) = ({i}, {l})"
                    );
                }
            }
        }
    }

    #[test]
    fn interpolation_weights_are_correctly_rounded() {
        // k = 5: the LU-based table used to miss these entries by up to
        // 1.8e-13; the exact values are small fractions
        let wi = calc_poly_interpolation(5);
        // row-major `(p, l)`, generated with exact rational arithmetic
        let exact: Vec<(i128, i128)> = vec![
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (-137, 60),
            (5, 1),
            (-5, 1),
            (10, 3),
            (-5, 4),
            (1, 5),
            (15, 8),
            (-77, 12),
            (107, 12),
            (-13, 2),
            (61, 24),
            (-5, 12),
            (-17, 24),
            (71, 24),
            (-59, 12),
            (49, 12),
            (-41, 24),
            (7, 24),
            (1, 8),
            (-7, 12),
            (13, 12),
            (-1, 1),
            (11, 24),
            (-1, 12),
            (-1, 120),
            (1, 24),
            (-1, 12),
            (1, 12),
            (-1, 24),
            (1, 120),
        ];
        for (idx, (num, den)) in exact.iter().enumerate() {
            assert_eq!(
                wi[idx],
                Rational::new(*num, *den).to_f64(),
                "k = 5, entry {idx}"
            );
        }
    }
}
