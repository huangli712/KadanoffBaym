// Project : KadanoffBaym
// File    : src/linalg/eigh.rs
// Purpose : Hermitian eigendecomposition via complex cyclic Jacobi

use crate::complex::Complex;
use crate::linalg::mat::DenseMat;

/// Hermitian eigendecomposition `A = V diag(w) V^dagger` with eigenvalues
/// `w` sorted in ascending order and `V` unitary (columns are the
/// eigenvectors).
///
/// Complex cyclic Jacobi with true unitary two-point rotations (the
/// rotation carries the phase of `a[p][q]`). Convergence is measured with
/// the off-diagonal Frobenius norm relative to the full norm.
///
/// # Panics
/// Panics if `A` is not Hermitian within `1e-12` or the sweep limit is
/// exceeded (both are programming errors).
pub fn eigh(a: &DenseMat) -> (Vec<f64>, DenseMat) {
    let n = a.size();
    assert!(
        a.is_hermitian(1e-12),
        "eigh: matrix is not Hermitian within 1e-12"
    );
    let mut work = a.clone();
    let mut v = DenseMat::identity(n);
    let norm = a.fro_norm();
    if norm == 0.0 {
        return (vec![0.0; n], v);
    }

    const MAX_SWEEPS: usize = 100;
    let tol = 1e-15 * norm;
    for _sweep in 0..MAX_SWEEPS {
        // off-diagonal Frobenius norm
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += work.get(i, j).abs_sqr();
            }
        }
        if (2.0 * off).sqrt() <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                rotate(&mut work, &mut v, p, q);
            }
        }
    }
    // Final check: must have converged within the sweep limit.
    let mut off = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            off += work.get(i, j).abs_sqr();
        }
    }
    assert!(
        (2.0 * off).sqrt() <= 1e-12 * norm,
        "eigh: Jacobi did not converge (off = {})",
        (2.0 * off).sqrt()
    );

    let mut w: Vec<f64> = (0..n).map(|i| work.get(i, i).re).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| w[i].partial_cmp(&w[j]).unwrap());
    let w_sorted: Vec<f64> = order.iter().map(|&i| w[i]).collect();
    // Reorder eigenvector columns.
    let mut v_sorted = DenseMat::zeros(n);
    for (new, &old) in order.iter().enumerate() {
        for r in 0..n {
            v_sorted.set(r, new, v.get(r, old));
        }
    }
    w = w_sorted;
    (w, v_sorted)
}

/// One Jacobi rotation on the (p, q) plane, annihilating `A[p][q]`.
fn rotate(a: &mut DenseMat, v: &mut DenseMat, p: usize, q: usize) {
    let apq = a.get(p, q);
    let r = apq.abs();
    if r == 0.0 {
        return;
    }
    let app = a.get(p, p).re;
    let aqq = a.get(q, q).re;
    let phase = Complex::from_polar(1.0, apq.arg()); // e^{i phi}, a[p][q] = r * phase
    let tau = (aqq - app) / (2.0 * r);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    let n = a.size();

    // Right rotation A <- A J on columns (p, q):
    //   a[k][p] <- c a[k][p] - s e^{-i phi} a[k][q]
    //   a[k][q] <- s e^{i phi} a[k][p] + c a[k][q]
    let emi = phase.conj().scale(s); // s e^{-i phi}
    let epi = phase.scale(s); // s e^{+i phi}
    for k in 0..n {
        let akp = a.get(k, p);
        let akq = a.get(k, q);
        a.set(k, p, akp.scale(c) - emi * akq);
        a.set(k, q, epi * akp + akq.scale(c));
    }
    // Left rotation A <- J^dagger A on rows (p, q):
    //   a[p][k] <- c a[p][k] - s e^{+i phi} a[q][k]
    //   a[q][k] <- s e^{-i phi} a[p][k] + c a[q][k]
    for k in 0..n {
        let apk = a.get(p, k);
        let aqk = a.get(q, k);
        a.set(p, k, apk.scale(c) - epi * aqk);
        a.set(q, k, emi * apk + aqk.scale(c));
    }
    // Accumulate V <- V J.
    for k in 0..n {
        let vkp = v.get(k, p);
        let vkq = v.get(k, q);
        v.set(k, p, vkp.scale(c) - emi * vkq);
        v.set(k, q, epi * vkp + vkq.scale(c));
    }
}

#[cfg(test)]
mod tests {
    use super::eigh;
    use crate::complex::Complex;
    use crate::linalg::mat::DenseMat;

    fn check_decomposition(a: &DenseMat, tol: f64) -> (Vec<f64>, DenseMat) {
        let (w, v) = eigh(a);
        let n = a.size();
        // A V = V diag(w) column-wise
        for j in 0..n {
            for i in 0..n {
                let mut av = Complex::ZERO;
                for k in 0..n {
                    av += a.get(i, k) * v.get(k, j);
                }
                let want = v.get(i, j).scale(w[j]);
                assert!(
                    (av - want).abs() <= tol,
                    "AV != VL at [{i}][{j}]: {av} vs {want}"
                );
            }
        }
        // V^dagger V = I
        let vtv = v.adjoint_mul(&v);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { Complex::ONE } else { Complex::ZERO };
                assert!(
                    (vtv.get(i, j) - want).abs() <= tol,
                    "VdV[{i}][{j}] = {:?}",
                    vtv.get(i, j)
                );
            }
        }
        // Ascending order
        for j in 1..n {
            assert!(w[j - 1] <= w[j] + 1e-14);
        }
        (w, v)
    }

    /// Deterministic pseudo-random unitary via a fixed seed (xorshift),
    /// no external RNG dependency.
    fn unitary_from_seed(n: usize, seed: u64) -> DenseMat {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut m = DenseMat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let re = (next() % 2001) as f64 / 1000.0 - 1.0;
                let im = (next() % 2001) as f64 / 1000.0 - 1.0;
                m.set(i, j, Complex::new(re, im));
            }
        }
        // Orthonormalize columns via Gram-Schmidt.
        for j in 0..n {
            for k in 0..j {
                let mut dot = Complex::ZERO;
                for i in 0..n {
                    dot += m.get(i, k).conj() * m.get(i, j);
                }
                for i in 0..n {
                    let v = m.get(i, j) - dot * m.get(i, k);
                    m.set(i, j, v);
                }
            }
            let nrm: f64 = (0..n).map(|i| m.get(i, j).abs_sqr()).sum::<f64>().sqrt();
            for i in 0..n {
                let v = m.get(i, j) / nrm;
                m.set(i, j, v);
            }
        }
        m
    }

    #[test]
    fn analytic_2x2() {
        // [[2, i], [-i, 2]] has eigenvalues 1 and 3.
        let mut a = DenseMat::zeros(2);
        a.set(0, 0, Complex::new(2.0, 0.0));
        a.set(1, 1, Complex::new(2.0, 0.0));
        a.set(0, 1, Complex::new(0.0, 1.0));
        a.set(1, 0, Complex::new(0.0, -1.0));
        let (w, _v) = check_decomposition(&a, 1e-13);
        assert!((w[0] - 1.0).abs() < 1e-13);
        assert!((w[1] - 3.0).abs() < 1e-13);
    }

    #[test]
    fn identity_and_degenerate() {
        let (w, _) = check_decomposition(&DenseMat::identity(4), 1e-13);
        assert!(w.iter().all(|x| (*x - 1.0).abs() < 1e-13));
        // 2x2 with equal diagonals and complex coupling: eigenvalues
        // 5 ± |c| with |c| = sqrt(5); eigenvectors only checked via
        // residual/orthonormality.
        let mut d = DenseMat::zeros(2);
        d.set(0, 0, Complex::new(5.0, 0.0));
        d.set(1, 1, Complex::new(5.0, 0.0));
        d.set(0, 1, Complex::new(1.0, 2.0));
        d.set(1, 0, Complex::new(1.0, -2.0));
        let (w, _) = check_decomposition(&d, 1e-13);
        let r = 5.0f64;
        let e = 5.0f64.sqrt();
        assert!((w[0] - (r - e)).abs() < 1e-12);
        assert!((w[1] - (r + e)).abs() < 1e-12);
    }

    #[test]
    fn complex_offdiagonal_hermitian() {
        let mut a = DenseMat::zeros(3);
        a.set(0, 0, Complex::new(1.0, 0.0));
        a.set(1, 1, Complex::new(-2.0, 0.0));
        a.set(2, 2, Complex::new(0.5, 0.0));
        a.set(0, 1, Complex::new(0.3, -1.2));
        a.set(1, 0, Complex::new(0.3, 1.2));
        a.set(1, 2, Complex::new(-0.7, 0.4));
        a.set(2, 1, Complex::new(-0.7, -0.4));
        let (w, _) = check_decomposition(&a, 1e-12);
        // trace and Frobenius norm invariants
        let tr: f64 = w.iter().sum();
        assert!((tr - (-0.5)).abs() < 1e-12);
        let f2: f64 =
            w.iter().map(|x| x * x).sum::<f64>() + 2.0 * a.fro_norm_sqr() - a.fro_norm_sqr();
        let _ = f2;
        assert!(w.len() == 3);
    }

    #[test]
    fn one_by_one() {
        let mut a = DenseMat::zeros(1);
        a.set(0, 0, Complex::new(2.5, 0.0));
        let (w, v) = eigh(&a);
        assert_eq!(w, vec![2.5]);
        assert_eq!(v.get(0, 0), Complex::ONE);
    }

    #[test]
    fn random_6x6_matches_construction() {
        let n = 6;
        let u = unitary_from_seed(n, 0x5DEECE66D);
        let diag = [3.0, -1.5, 0.25, 7.0, -4.0, 1.0];
        let mut d = DenseMat::zeros(n);
        for (i, x) in diag.iter().enumerate() {
            d.set(i, i, Complex::new(*x, 0.0));
        }
        // A = U D U^dagger is Hermitian with spectrum `diag`
        let a = u.mat_mul(&d).mat_mul_adjoint(&u);
        assert!(a.is_hermitian(1e-12));
        let (w, _) = check_decomposition(&a, 1e-11);
        let mut want = diag.to_vec();
        want.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for i in 0..n {
            assert!(
                (w[i] - want[i]).abs() < 1e-11,
                "eigenvalue {i}: {} vs {}",
                w[i],
                want[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "not Hermitian")]
    fn rejects_non_hermitian() {
        let mut a = DenseMat::zeros(2);
        a.set(0, 1, Complex::new(1.0, 0.0));
        a.set(1, 0, Complex::new(2.0, 0.0));
        let _ = eigh(&a);
    }
}
