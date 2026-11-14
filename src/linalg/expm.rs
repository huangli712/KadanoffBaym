// Project : KadanoffBaym
// File    : src/linalg/expm.rs
// Purpose : matrix exponential via scaling-and-squaring with Pade-13

use crate::linalg::{lu_factor, lu_solve_mat};
use crate::linalg::DenseMat;

/// Pade-13 coefficients `b_j` of the [13/13] Pade approximant of `exp`
/// (Higham 2005 table, same values used by scipy/Julia expm backends).
const B: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// `theta_13` bound: scaling to 1-norm <= 5.37 keeps the backward error
/// of the [13/13] Pade approximant below unit roundoff.
const THETA_13: f64 = 5.371920351148152;

/// Matrix exponential `exp(A)` of a general complex square matrix
/// (scaling-and-squaring with the [13/13] Pade approximant).
pub fn expm(a: &DenseMat) -> DenseMat {
    let n = a.size();
    if n == 1 {
        let z = a.get(0, 0);
        let e = z.exp();
        let mut out = DenseMat::zeros(1);
        out.set(0, 0, e);
        return out;
    }

    // 1-norm (max absolute column sum).
    let mut norm1 = 0.0f64;
    for j in 0..n {
        let s: f64 = (0..n).map(|i| a.get(i, j).abs()).sum();
        norm1 = norm1.max(s);
    }

    let mut s = 0usize;
    if norm1 > THETA_13 {
        s = (norm1 / THETA_13).log2().ceil().max(0.0) as usize;
    }
    let scale = f64::exp2(-(s as f64));
    let as_ = {
        let mut m = a.clone();
        m.scale(scale);
        m
    };

    // Powers of the scaled matrix.
    let a2 = as_.mat_mul(&as_);
    let a4 = a2.mat_mul(&a2);
    let a6 = a2.mat_mul(&a4);

    // U (odd powers) and V (even powers).
    let u = {
        // A6 * (b13 A6 + b11 A4 + b9 A2)
        let mut w = DenseMat::zeros(n);
        for j in 0..n {
            for i in 0..n {
                w.set(
                    i,
                    j,
                    a6.get(i, j).scale(B[13])
                        + a4.get(i, j).scale(B[11])
                        + a2.get(i, j).scale(B[9]),
                );
            }
        }
        let mut u = a6.mat_mul(&w);
        // + b7 A6 + b5 A4 + b3 A2, then U = A * (...)
        for idx in 0..n * n {
            u.data[idx] +=
                a6.data[idx].scale(B[7]) + a4.data[idx].scale(B[5]) + a2.data[idx].scale(B[3]);
            u.data[idx] += if is_diag(idx, n) {
                crate::complex::Complex::new(B[1], 0.0)
            } else {
                crate::complex::Complex::ZERO
            };
        }
        as_.mat_mul(&u)
    };
    let v = {
        let mut w = DenseMat::zeros(n);
        for j in 0..n {
            for i in 0..n {
                w.set(
                    i,
                    j,
                    a6.get(i, j).scale(B[12])
                        + a4.get(i, j).scale(B[10])
                        + a2.get(i, j).scale(B[8]),
                );
            }
        }
        let mut v = a6.mat_mul(&w);
        for idx in 0..n * n {
            v.data[idx] +=
                a6.data[idx].scale(B[6]) + a4.data[idx].scale(B[4]) + a2.data[idx].scale(B[2]);
            v.data[idx] += if is_diag(idx, n) { crate::complex::Complex::new(B[0], 0.0) } else { crate::complex::Complex::ZERO };
        }
        v
    };

    // exp(A_s) = (V - U)^{-1} (V + U)
    let mut vmu = DenseMat::zeros(n);
    let mut vpu = DenseMat::zeros(n);
    for idx in 0..n * n {
        vmu.data[idx] = v.data[idx] - u.data[idx];
        vpu.data[idx] = v.data[idx] + u.data[idx];
    }
    let mut piv_mat = vmu;
    let piv = lu_factor(&mut piv_mat);
    let mut r = lu_solve_mat(&piv_mat, &piv, &vpu);

    // Undo scaling by repeated squaring.
    for _ in 0..s {
        r = r.mat_mul(&r);
    }
    r
}

#[inline]
fn is_diag(idx: usize, n: usize) -> bool {
    idx % (n + 1) == 0
}

#[cfg(test)]
mod tests {
    use crate::linalg::expm;
    use crate::complex::Complex;
    use crate::linalg::eigh;
    use crate::linalg::DenseMat;

    fn close(a: &DenseMat, b: &DenseMat, tol: f64, what: &str) {
        for i in 0..a.size() {
            for j in 0..a.size() {
                let d = (a.get(i, j) - b.get(i, j)).abs();
                assert!(
                    d <= tol,
                    "{what} [{i}][{j}]: {} vs {}, |d|={d}",
                    a.get(i, j),
                    b.get(i, j)
                );
            }
        }
    }

    #[test]
    fn skew_hermitian_matches_eig_analytic() {
        // exp(-i h H) == V exp(-i h w) V^dagger for Hermitian H.
        let h = DenseMat::from_rows(&[
            vec![Complex::new(1.0, 0.0), Complex::new(0.5, -0.3)],
            vec![Complex::new(0.5, 0.3), Complex::new(-0.7, 0.0)],
        ]);
        let step = 0.35;
        let mut m = h.clone();
        m.scale(0.0);
        // m = -i*step*H
        for i in 0..2 {
            for j in 0..2 {
                m.set(i, j, h.get(i, j) * Complex::new(0.0, -step));
            }
        }
        let got = expm(&m);
        let (w, v) = eigh(&h);
        let mut d = DenseMat::zeros(2);
        for i in 0..2 {
            d.set(i, i, Complex::from_polar(1.0, -step * w[i]));
        }
        let want = v.mat_mul(&d).mat_mul_adjoint(&v);
        close(&got, &want, 1e-12, "skew-hermitian expm");
    }

    #[test]
    fn diagonal_matrix() {
        let mut d = DenseMat::zeros(3);
        d.set(0, 0, Complex::new(0.5, 0.0));
        d.set(1, 1, Complex::new(-1.0, 0.0));
        d.set(2, 2, Complex::new(2.0, 1.0));
        let got = expm(&d);
        let mut want = DenseMat::zeros(3);
        want.set(0, 0, d.get(0, 0).exp());
        want.set(1, 1, d.get(1, 1).exp());
        want.set(2, 2, d.get(2, 2).exp());
        close(&got, &want, 1e-14, "diagonal expm");
    }

    #[test]
    fn nilpotent_exact_polynomial() {
        // N = [[0, 1], [0, 0]], N^2 = 0, exp(N) = I + N (any scaling keeps
        // the Pade numerator/denominator exact up to roundoff).
        let mut nmat = DenseMat::zeros(2);
        nmat.set(0, 1, Complex::ONE);
        let got = expm(&nmat);
        let mut want = DenseMat::identity(2);
        want.set(0, 1, Complex::ONE);
        close(&got, &want, 1e-15, "nilpotent expm");
    }

    #[test]
    fn identity_and_zero() {
        // exp(I) = e * I
        let got = expm(&DenseMat::identity(3));
        let mut want = DenseMat::identity(3);
        want.scale(std::f64::consts::E);
        close(&got, &want, 1e-12, "exp(I)");
        let z = DenseMat::zeros(2);
        let got = expm(&z);
        close(&got, &DenseMat::identity(2), 1e-15, "exp(zero)");
    }

    #[test]
    fn exp_times_exp_minus_is_identity() {
        // Deterministic 4x4, exp(A) exp(-A) = I (tolerances looser than
        // the analytic tests because the product compounds errors).
        let mut a = DenseMat::zeros(4);
        for i in 0..4 {
            for j in 0..4 {
                let re = ((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.5;
                let im = ((i * 5 + j * 13) % 7) as f64 / 7.0 - 0.5;
                a.set(i, j, Complex::new(re, im));
            }
        }
        // Hermitize to keep the norm moderate
        let a = {
            let mut ah = a.adjoint();
            ah.axpy(1.0, &a);
            ah.scale(0.5);
            ah
        };
        let mut b = a.clone();
        b.scale(-1.0);
        let prod = expm(&a).mat_mul(&expm(&b));
        close(&prod, &DenseMat::identity(4), 1e-11, "exp(A) exp(-A)");
    }

    #[test]
    fn large_norm_scales_correctly() {
        // norm > theta_13 forces the squaring path: exp(20 * I) = e^20 I.
        let mut a = DenseMat::zeros(2);
        a.set(0, 0, Complex::new(20.0, 0.0));
        a.set(1, 1, Complex::new(20.0, 0.0));
        let got = expm(&a);
        let want_val = (20.0f64).exp();
        assert!((got.get(0, 0).re - want_val).abs() / want_val < 1e-12);
        assert!((got.get(1, 1).re - want_val).abs() / want_val < 1e-12);
        assert!(got.get(0, 1).abs() < 1e-12);
    }
}
