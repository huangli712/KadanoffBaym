// Project : KadanoffBaym
// File    : src/linalg/lu.rs
// Purpose : LU factorization with partial pivoting (solve/inverse)

use crate::linalg::DenseMat;

/// In-place LU factorization with partial (row) pivoting. `a` is
/// overwritten with the compact L\U storage; the returned vector is the
/// pivot table (`piv[k]` = row swapped with row `k` at step `k`).
///
/// # Panics
/// Panics on a exactly singular matrix (programming error, blueprint
/// `assert` semantics).
pub fn lu_factor(a: &mut DenseMat) -> Vec<usize> {
    let n = a.size();
    let mut piv = Vec::with_capacity(n);
    for k in 0..n {
        // Pivot: largest |a[i][k]| in column k, rows i >= k.
        let mut p = k;
        let mut pmax = a.get(k, k).abs();
        for i in (k + 1)..n {
            let v = a.get(i, k).abs();
            if v > pmax {
                pmax = v;
                p = i;
            }
        }
        piv.push(p);
        assert!(pmax > 0.0, "lu_factor: singular matrix at step {k}");
        if p != k {
            for j in 0..n {
                let tmp = a.get(k, j);
                a.set(k, j, a.get(p, j));
                a.set(p, j, tmp);
            }
        }
        for i in (k + 1)..n {
            let l = a.get(i, k) / a.get(k, k);
            a.set(i, k, l);
            for j in (k + 1)..n {
                let v = a.get(i, j) - l * a.get(k, j);
                a.set(i, j, v);
            }
        }
    }
    piv
}

/// Apply the pivot table to a right-hand-side vector (in place):
/// performs the same row interchanges the factorization did.
fn apply_piv(b: &mut [crate::complex::Complex], piv: &[usize]) {
    for (k, &p) in piv.iter().enumerate() {
        if p != k {
            b.swap(k, p);
        }
    }
}

/// Solve `A x = b` given the factorization `lu` (L\U storage) and pivot
/// table of [`lu_factor`].
pub fn lu_solve(lu: &DenseMat, piv: &[usize], b: &[crate::complex::Complex]) -> Vec<crate::complex::Complex> {
    use crate::complex::Complex;
    let n = lu.size();
    assert_eq!(b.len(), n, "lu_solve: rhs length mismatch");
    let mut x: Vec<Complex> = b.to_vec();
    apply_piv(&mut x, piv);
    // Forward substitution (L has unit diagonal).
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu.get(i, j) * x[j];
        }
        x[i] = s;
    }
    // Backward substitution.
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= lu.get(i, j) * x[j];
        }
        x[i] = s / lu.get(i, i);
    }
    x
}

/// Solve `A X = B` with a matrix right-hand side (column-wise; `b` may
/// be rectangular `n x m`).
pub fn lu_solve_mat(lu: &DenseMat, piv: &[usize], b: &DenseMat) -> DenseMat {
    let n = lu.size();
    assert_eq!(b.nrows(), n, "lu_solve_mat: rhs size mismatch");
    let m = b.ncols();
    let mut out = DenseMat::zeros_shape(n, m);
    for j in 0..m {
        let col: Vec<crate::complex::Complex> = (0..n).map(|i| b.get(i, j)).collect();
        let sol = lu_solve(lu, piv, &col);
        for (i, v) in sol.iter().enumerate() {
            out.set(i, j, *v);
        }
    }
    out
}

/// Matrix inverse via LU.
pub fn mat_inv(a: &DenseMat) -> DenseMat {
    let mut lu = a.clone();
    let piv = lu_factor(&mut lu);
    lu_solve_mat(&lu, &piv, &DenseMat::identity(a.size()))
}

#[cfg(test)]
mod tests {
    use crate::complex::Complex;
    use crate::linalg::DenseMat;

    fn randomish_hilbert(n: usize) -> DenseMat {
        // Deterministic moderately-conditioned complex matrix:
        // H[i][j] = 1/(i+j+1) + i/(i+j+2)
        let mut m = DenseMat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let re = 1.0 / ((i + j + 1) as f64);
                let im = 1.0 / ((i + j + 2) as f64);
                m.set(i, j, Complex::new(re, im));
            }
        }
        m
    }

    /// Well-conditioned variant `I + 0.1 * H` for tight residual tests.
    fn well_conditioned(n: usize) -> DenseMat {
        let mut m = randomish_hilbert(n);
        m.axpy(0.1, &DenseMat::zeros(n));
        for i in 0..n {
            *m.get_mut(i, i) += Complex::ONE;
        }
        m
    }

    #[test]
    fn solve_residual_small_dimensions() {
        for n in 1..=8 {
            let a = well_conditioned(n);
            let b: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64 + 1.0, 0.5)).collect();
            let mut lu = a.clone();
            let piv = super::lu_factor(&mut lu);
            let x = super::lu_solve(&lu, &piv, &b);
            // Residual ||Ax - b|| / ||b||
            let mut ax = vec![Complex::ZERO; n];
            for i in 0..n {
                for j in 0..n {
                    ax[i] += a.get(i, j) * x[j];
                }
            }
            let bn: f64 = b.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
            let rn: f64 = ax
                .iter()
                .zip(&b)
                .map(|(u, v)| (*u - *v).abs_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(rn / bn < 1e-13, "n={n}: residual {rn}/{bn}");
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let a = well_conditioned(5);
        let inv = super::mat_inv(&a);
        let id = a.mat_mul(&inv);
        for i in 0..5 {
            for j in 0..5 {
                let want = if i == j { Complex::ONE } else { Complex::ZERO };
                assert!(
                    (id.get(i, j) - want).abs() < 1e-12,
                    "A*A^-1[{i}][{j}] = {:?}",
                    id.get(i, j)
                );
            }
        }
    }

    #[test]
    fn solve_mat_matches_solve() {
        let a = randomish_hilbert(4);
        let mut lu = a.clone();
        let piv = super::lu_factor(&mut lu);
        let mut b = DenseMat::zeros(4);
        for j in 0..4 {
            b.set(j, 0, Complex::new(j as f64, 1.0));
        }
        let x = super::lu_solve_mat(&lu, &piv, &b);
        let col: Vec<Complex> = (0..4).map(|i| b.get(i, 0)).collect();
        let x1 = super::lu_solve(&lu, &piv, &col);
        for i in 0..4 {
            assert!((x.get(i, 0) - x1[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn hilbert3_relative_error() {
        // Classic ill-conditioned Hilbert 3x3: inverse entries are exact
        // integers; check relative error stays below 1e-10.
        let mut h = DenseMat::zeros(3);
        for i in 0..3 {
            for j in 0..3 {
                h.set(i, j, Complex::new(1.0 / ((i + j + 1) as f64), 0.0));
            }
        }
        let inv = super::mat_inv(&h);
        // H^-1 = [9 -36 30; -36 192 -180; 30 -180 180] (exact integers)
        assert!((inv.get(0, 0).re - 9.0).abs() / 9.0 < 1e-10);
        assert!((inv.get(0, 1).re + 36.0).abs() / 36.0 < 1e-10);
        assert!((inv.get(1, 1).re - 192.0).abs() / 192.0 < 1e-10);
        assert!((inv.get(2, 2).re - 180.0).abs() / 180.0 < 1e-10);
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn singular_matrix_panics() {
        let mut s = DenseMat::zeros(2);
        s.set(0, 0, Complex::new(1.0, 0.0));
        // row 1 left zero -> singular
        super::lu_factor(&mut s);
    }
}
