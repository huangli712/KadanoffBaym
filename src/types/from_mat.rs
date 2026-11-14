// Project : KadanoffBaym
// File    : src/types/from_mat.rs
// Purpose : initial steps from the Matsubara component
//
// Blueprint: julia/src/equilibrium/from_mat.jl (pulled forward from R5
// into R2 because t_utilities' from_mat tests are part of R2).
//
//   G^mix(t_n, tau_m) = i * xi * G^M(beta - tau_m)
//   G^<(t_n, t_n)     = G^mix(0, 0)
//   G^R(t_n, t_n)     = i * G^M(0) - G^<(0, 0)
//
// with `xi = -1 (+1)` for fermions (bosons). Julia 1-based `n = 1` is
// Rust `n = 0`.

use crate::c64::C64;
use crate::types::green::ContourGreen;

/// Set up the lmix row at (0-based) time step `n` and the diagonal
/// elements of ret/less at `(n, n)` from the Matsubara component
/// (blueprint `_from_mat_step!`).
pub fn from_mat_step(n: i64, g: &mut ContourGreen) {
    let ntau = g.ntau;
    let xi = g.sign.as_i64() as f64;
    let f_i_xi = C64::new(0.0, xi);
    // G^mix(n, m) = i * xi * G^M(beta - tau_m); Julia mat[ntau - m + 1]
    // -> Rust mat index ntau - m
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
    let f_i = C64::new(0.0, 1.0);
    for ((d, mm), ll) in dst.iter_mut().zip(&mat0).zip(&less00) {
        *d = *mm * f_i - *ll;
    }
}

/// Set up the first real time step (`n = 0`, blueprint `from_mat!(G)`).
pub fn from_mat(g: &mut ContourGreen) {
    from_mat_step(0, g);
}

/// Set up the steps `0..=k` (blueprint `from_mat!(G, k)`, whose Julia
/// range `1 : k + 1` maps to Rust `0 : k`).
pub fn from_mat_k(g: &mut ContourGreen, k: i64) {
    assert!(k >= 0);
    assert!(g.nt as i64 >= k + 1);
    for n in 0..=k {
        from_mat_step(n, g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn fermion_relations() {
        let c = Contour::new(3, 4, 2, 2, 1.0, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        // mat(0) = 1, mat(beta) = 2
        g.set_mat(0, 0, 0, C64::new(1.0, 0.0));
        g.set_mat(4, 0, 0, C64::new(2.0, 0.0));
        from_mat_k(&mut g, 1);
        // lmix(n, 0) = i*(-1)*mat(beta) = -2i
        for n in 0..=1 {
            assert_eq!(g.get_lmix(n, 0, 0, 0), C64::new(0.0, -2.0));
        }
        // lmix(n, beta) = i*(-1)*mat(0) = -i
        assert_eq!(g.get_lmix(0, 4, 0, 0), C64::new(0.0, -1.0));
        // less(n, n) = lmix(0, 0) = -2i
        assert_eq!(g.get_less(0, 0, 0, 0), C64::new(0.0, -2.0));
        assert_eq!(g.get_less(1, 1, 0, 0), C64::new(0.0, -2.0));
        // ret(n, n) = i*mat(0) - less(0,0) = i - (-2i) = 3i
        assert_eq!(g.get_ret(0, 0, 0, 0), C64::new(0.0, 3.0));
    }

    #[test]
    fn boson_sign_flips() {
        let c = Contour::new(2, 2, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Bose, C64::ZERO);
        g.set_mat(2, 0, 0, C64::new(1.0, 0.0));
        from_mat(&mut g);
        // lmix(0, 0) = i*(+1)*mat(beta)
        assert_eq!(g.get_lmix(0, 0, 0, 0), C64::new(0.0, 1.0));
    }
}
