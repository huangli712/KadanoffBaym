// Project : KadanoffBaym
// File    : src/equilibrium/mat.rs
// Purpose : initial steps from the Matsubara component
//
//
//   G^mix(t_n, tau_m) = i * xi * G^M(beta - tau_m)
//   G^<(t_n, t_n)     = G^mix(0, 0)
//   G^R(t_n, t_n)     = i * G^M(0) - G^<(0, 0)
//
// with `xi = -1 (+1)` for fermions (bosons). The blueprint's 1-based
// `n = 1` is `n = 0` here.

use crate::complex::Complex;
use crate::types::ContourGreen;

/// Set up the lmix row at (0-based) time step `n` and the diagonal
/// elements of ret/less at `(n, n)` from the Matsubara component
/// (blueprint `_from_mat_step!`).
fn from_mat_step(n: i64, g: &mut ContourGreen) {
    let ntau = g.ntau;
    let xi = g.sign.as_i64() as f64;
    let f_i_xi = Complex::new(0.0, xi);
    // G^mix(n, m) = i * xi * G^M(beta - tau_m); the blueprint's
    // mat[ntau - m + 1] becomes the mat index ntau - m here
    for m in 0..=(ntau as i64) {
        let src = g.mat.block(ntau - m as usize).to_vec();
        let dst = g.lmix_raw_block_mut(n, m);
        for (d, s) in dst.iter_mut().zip(&src) {
            *d = *s * f_i_xi;
        }
    }
    // G^<(n, n) = G^mix(0, 0)
    let lmix00 = g.lmix_raw_block(0, 0).to_vec();
    let dst = g.less_raw_block_mut(n, n);
    dst.copy_from_slice(&lmix00);
    // G^R(n, n) = i * G^M(0) - G^<(0, 0)
    let mat0 = g.mat_raw_block(0).to_vec();
    let less00 = g.less_raw_block(0, 0).to_vec();
    let dst = g.ret_raw_block_mut(n, n);
    let f_i = Complex::new(0.0, 1.0);
    for ((d, mm), ll) in dst.iter_mut().zip(&mat0).zip(&less00) {
        *d = *mm * f_i - *ll;
    }
}

/// Set up the first real time step (`n = 0`, blueprint `from_mat!(G)`).
///
/// *Remarks* : *Boundary Values From Matsubara Component*
///
/// At ``t = 0``, the real-time components are determined by the Matsubara
/// component:
///
/// ```math
/// \begin{equation}
/// G^{\rceil}(0,\tau) = i \xi G^{M}(\beta - \tau),
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// G^{<}(0,0) = G^{\rceil}(0,0),
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// G^{R}(0,0) = i G^{M}(0) - G^{<}(0,0).
/// \end{equation}
/// ```
///
/// Here ``\xi = -1`` (``+1``) for fermions (bosons).
pub fn from_mat(g: &mut ContourGreen) {
    from_mat_step(0, g);
}

/// Set up the steps `0..=k` (blueprint `from_mat!(G, k)`, whose range
/// `1 : k + 1` maps to `0 : k` here).
pub fn from_mat_k(g: &mut ContourGreen, k: i64) {
    assert!(k >= 0);
    assert!(g.nt as i64 > k);
    for n in 0..=k {
        from_mat_step(n, g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    use crate::complex::Complex;
    use crate::linalg::DenseMat;
    use crate::types::{Contour, ContourGreen, Sign};

    use crate::types::MATSUBARA;

    use crate::basics::blkdist;
    use crate::equilibrium::from_ham;

    #[test]
    fn fermion_relations() {
        let c = Contour::new(3, 4, 2, 2, 1.0, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // mat(0) = 1, mat(beta) = 2
        g.set_mat(0, 0, 0, Complex::new(1.0, 0.0));
        g.set_mat(4, 0, 0, Complex::new(2.0, 0.0));
        from_mat_k(&mut g, 1);
        // lmix(n, 0) = i*(-1)*mat(beta) = -2i
        for n in 0..=1 {
            assert_eq!(g.get_lmix(n, 0, 0, 0), Complex::new(0.0, -2.0));
        }
        // lmix(n, beta) = i*(-1)*mat(0) = -i
        assert_eq!(g.get_lmix(0, 4, 0, 0), Complex::new(0.0, -1.0));
        // less(n, n) = lmix(0, 0) = -2i
        assert_eq!(g.get_less(0, 0, 0, 0), Complex::new(0.0, -2.0));
        assert_eq!(g.get_less(1, 1, 0, 0), Complex::new(0.0, -2.0));
        // ret(n, n) = i*mat(0) - less(0,0) = i - (-2i) = 3i
        assert_eq!(g.get_ret(0, 0, 0, 0), Complex::new(0.0, 3.0));
    }

    #[test]
    fn boson_sign_flips() {
        let c = Contour::new(2, 2, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        g.set_mat(2, 0, 0, Complex::new(1.0, 0.0));
        from_mat(&mut g);
        // lmix(0, 0) = i*(+1)*mat(beta)
        assert_eq!(g.get_lmix(0, 0, 0, 0), Complex::new(0.0, 1.0));
    }

    #[test]
    fn from_mat_and_from_mat() {
        let (ntime, ntau, beta, h) = (9i64, 41i64, 5.0, 0.05);
        let mut hmat = DenseMat::zeros(2);
        hmat.set(0, 0, Complex::new(-1.0, 0.0));
        hmat.set(1, 1, Complex::new(0.8, 0.0));
        hmat.set(0, 1, Complex::new(0.0, -0.7));
        hmat.set(1, 0, Complex::new(0.0, 0.7));
        for sign in [Sign::Fermi, Sign::Bose] {
            let c = Contour::new(ntime - 1, ntau - 1, 2, 2, (ntime - 1) as f64 * h, beta);
            let mut gref = ContourGreen::new(&c, sign, Complex::ZERO);
            let mut h_use = hmat.clone();
            if sign == Sign::Bose {
                h_use.axpy(2.0, &DenseMat::identity(2));
            }
            from_ham(&mut gref, 0.0, &h_use, beta, h);

            let mut g = ContourGreen::new(&c, sign, Complex::ZERO);
            g.memcpy_from_at(MATSUBARA, &gref);
            from_mat(&mut g);
            assert!(g.distance_at(&gref, 0) <= 1e-13);
            assert!(g.mat.equal(&gref.mat));

            let k = 4i64;
            let mut g = ContourGreen::new(&c, sign, Complex::ZERO);
            g.memcpy_from_at(MATSUBARA, &gref);
            let sentinel = [
                Complex::new(9.0, 0.0),
                Complex::new(8.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(6.0, 0.0),
            ];
            for n in 1..=k + 1 {
                g.ret_raw_block_mut(n, 0).copy_from_slice(&sentinel);
                g.less_raw_block_mut(0, n).copy_from_slice(&sentinel);
            }
            from_mat_k(&mut g, k);
            for n in 0..=k {
                for m in 0..=(ntau - 1) {
                    let d = blkdist(g.lmix_raw_block(n, m), gref.lmix_raw_block(0, m));
                    assert!(d <= 1e-13, "lmix n={n} m={m}");
                }
                let d = blkdist(g.less_raw_block(n, n), gref.less_raw_block(0, 0));
                assert!(d <= 1e-13);
                let d = blkdist(g.ret_raw_block(n, n), gref.ret_raw_block(0, 0));
                assert!(d <= 1e-13);
            }
            for n in 1..=(k + 1) {
                assert_eq!(g.ret_raw_block(n, 0), sentinel);
                assert_eq!(g.less_raw_block(0, n), sentinel);
            }
            let zero = ContourGreen::new(&c, sign, Complex::ZERO);
            assert_eq!(g.distance_at(&zero, k + 2), 0.0);
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut g2 = g.clone();
                from_mat_k(&mut g2, ntime);
            }));
            assert!(r.is_err());
        }
    }
}
