// Project : KadanoffBaym
// File    : src/math/deriv.rs
// Purpose : left/right derivatives of a contour Green's function
//
// All 1-based times (i, j, n, m, t) map to 0-based indices; the
// relative comparisons keep their meaning.

//!
//! *Remarks* : *Differentiation of Contour-Ordered Green's Functions*
//!
//! The left and right derivatives of a two-time contour function are
//! defined as
//!
//! ```math
//! \begin{equation}
//! [\partial_1 A](t,t') = +i \partial_t A(t,t'),
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! [\partial_2 A](t,t') = -i \partial_{t'} A(t,t').
//! \end{equation}
//! ```
//!
//! For the Matsubara component, which is translationally invariant, the
//! two derivatives are identical,
//!
//! ```math
//! \begin{equation}
//! [\partial_1 A]^{M}(\tau) = [\partial_2 A]^{M}(\tau) = -\partial_\tau A^{M}(\tau).
//! \end{equation}
//! ```
//!
//! At the real time axis, the derivative at the time step ``n`` is
//! evaluated by the ``k``-th order polynomial differentiation weights
//! ``D^{(k)}_{nl}`` if ``n \le k`` (the values at ``0, \cdots, k`` are
//! needed), and by the backward differentiation formula of order ``k + 1``
//! otherwise. For the mixing component, the derivative with respect to the
//! imaginary time argument is evaluated in the same way, and it carries no
//! factor ``i``.
//!
//! The elements outside the stored triangles of the `ret` and `less`
//! components are reconstructed from `Acc` via the hermitian conjugate
//! relations, ``A^{R}(t,t') = -[A_{cc}^{R}(t',t)]^{\dagger}`` and
//! ``A^{<}(t,t') = -[A_{cc}^{<}(t',t)]^{\dagger}``. If `A` is hermitian,
//! simply provide `Acc = A`.
//!

use std::ptr;

use crate::complex::Complex;
use crate::types::ContourGreen;
use crate::weights::Integrator;

use crate::types::adjoint_block_into;

/// Imaginary-time derivative of the Matsubara component.
///
/// Calculate the derivative ``-\\partial_\\tau A^{M}(\\tau)`` and store it in
/// the Matsubara component of `dA`. Forward (backward) differentiation is
/// used near ``\\tau = 0`` (``\\tau = \\beta``), and backward differentiation
/// is used for the remaining points.
///
/// ### Arguments
/// * dA   -> Result, a contour-ordered Green's function.
/// * A    -> A contour-ordered Green's function.
/// * I    -> An Integrator struct.
/// * beta -> Inverse temperature.
///
/// ### Returns
/// N/A
pub fn deriv1_matsubara(da: &mut ContourGreen, a: &ContourGreen, integ: &Integrator, beta: f64) {
    let k = integ.k;
    let k1 = k + 1;
    let n = a.ntau as i64; // last tau index (0-based)
    assert!(beta > 0.0);
    assert!(da.ntau == a.ntau && da.equaldims(a));
    assert!(a.ntau as i64 > 2 * k, "deriv1_matsubara: ntau >= 2k+1");
    let hv = beta / n as f64;
    let b = a.size1 * a.size2;
    let mut c = vec![Complex::ZERO; b];

    // both boundaries
    for t in 0..=k {
        // c = -sum_l BDW[l] * mat[t + l]
        c.fill(Complex::ZERO);
        for l in 0..=k1 {
            let blk = a.mat.block((t + l) as usize);
            for (cc, s) in c.iter_mut().zip(blk) {
                *cc -= *s * integ.bdw.at(l as usize);
            }
        }
        let d = da.mat.block_mut(t as usize);
        for (d, cc) in d.iter_mut().zip(&c) {
            *d = *cc / (-hv);
        }
        // c = sum_l BDW[l] * mat[n - t - l] -> dA.mat[n - t]
        c.fill(Complex::ZERO);
        for l in 0..=k1 {
            let blk = a.mat.block((n - t - l) as usize);
            for (cc, s) in c.iter_mut().zip(blk) {
                *cc += *s * integ.bdw.at(l as usize);
            }
        }
        let d = da.mat.block_mut((n - t) as usize);
        for (d, cc) in d.iter_mut().zip(&c) {
            *d = *cc / (-hv);
        }
    }
    // interior: BDF
    for t in k1..=(n - k1) {
        c.fill(Complex::ZERO);
        for l in 0..=k1 {
            let blk = a.mat.block((t - l) as usize);
            for (cc, s) in c.iter_mut().zip(blk) {
                *cc += *s * integ.bdw.at(l as usize);
            }
        }
        let d = da.mat.block_mut(t as usize);
        for (d, cc) in d.iter_mut().zip(&c) {
            *d = *cc / (-hv);
        }
    }
}

/// Left derivative `i d/dt_1 A(t_i, t_j)` of one element.
///
/// Calculate the left derivative ``i \\partial_t A(t_i,t_j)``. If `j ≤ i`,
/// the `ret` component of `dA` at `(i,j)` is set. If `j ≥ i`, the `less`
/// component of `dA` at `(i,j)` is set. If `i ≤ k + 1`, `A` must be known at
/// the time steps `1, ..., k + 1`.
///
/// ### Arguments
/// * i   -> Index of the first real time argument.
/// * j   -> Index of the second real time argument.
/// * dA  -> Result, a contour-ordered Green's function.
/// * A   -> A contour-ordered Green's function.
/// * Acc -> Hermitian conjugate partner of `A`.
/// * I   -> An Integrator struct.
/// * h   -> Time step at real time axis.
///
/// ### Returns
/// N/A
pub fn deriv1_element(
    i: i64,
    j: i64,
    da: &mut ContourGreen,
    a: &ContourGreen,
    acc: Option<&ContourGreen>,
    integ: &Integrator,
    h: f64,
) {
    let k = integ.k;
    let k1 = k + 1;
    let nt = a.nt as i64;
    assert!(h > 0.0);
    assert!((0..=nt).contains(&i) && (0..=nt).contains(&j));
    assert!(nt >= k1);
    assert!(!ptr::eq(da, a), "output must differ from input A");
    if let Some(x) = acc {
        assert!(!ptr::eq(da, x), "output must differ from input Acc");
    }
    let b = a.size1 * a.size2;
    let mut c = vec![Complex::ZERO; b];
    let mut adj = vec![Complex::ZERO; b];
    let herm = acc.unwrap_or(a);
    if j <= i {
        c.fill(Complex::ZERO);
        if i < k1 {
            // startup: polynomial differentiation weights
            for l in 0..=k {
                let w = integ.pdw.at(i as usize, l as usize);
                if l < j {
                    // 1-based: l + 1 < j  <=>  l < j - 1        <=> l < j here
                    adjoint_block_into(herm.ret_raw_block(j, l), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.ret_raw_block(l, j);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        } else {
            for l in 0..=k1 {
                let w = integ.bdw.at(l as usize);
                if i - l < j {
                    adjoint_block_into(herm.ret_raw_block(j, i - l), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.ret_raw_block(i - l, j);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        }
        let dst = da.ret_raw_block_mut(i, j);
        let f = Complex::new(0.0, 1.0 / h);
        for (d, cc) in dst.iter_mut().zip(&c) {
            *d = *cc * f;
        }
    }
    if j >= i {
        c.fill(Complex::ZERO);
        if i < k1 {
            for l in 0..=k {
                let w = integ.pdw.at(i as usize, l as usize);
                if l > j {
                    // 1-based: l + 1 > j <=> l >= j here
                    adjoint_block_into(herm.less_raw_block(j, l), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.less_raw_block(l, j);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        } else {
            for l in 0..=k1 {
                let w = integ.bdw.at(l as usize);
                let blk = a.less_raw_block(i - l, j);
                for (cc, s) in c.iter_mut().zip(blk) {
                    *cc += *s * w;
                }
            }
        }
        let dst = da.less_raw_block_mut(i, j);
        let f = Complex::new(0.0, 1.0 / h);
        for (d, cc) in dst.iter_mut().zip(&c) {
            *d = *cc * f;
        }
    }
}

/// Right derivative `i d/dt_2 A(t_i, t_j)` of one element.
///
/// Calculate the right derivative ``-i \\partial_{t'} A(t_i,t_j)``. If
/// `j ≤ i`, the `ret` component of `dA` at `(i,j)` is set. If `j ≥ i`, the
/// `less` component of `dA` at `(i,j)` is set. If `j ≤ k + 1`, `A` must be
/// known at the time steps `1, ..., k + 1`.
///
/// ### Arguments
/// * i   -> Index of the first real time argument.
/// * j   -> Index of the second real time argument.
/// * dA  -> Result, a contour-ordered Green's function.
/// * A   -> A contour-ordered Green's function.
/// * Acc -> Hermitian conjugate partner of `A`.
/// * I   -> An Integrator struct.
/// * h   -> Time step at real time axis.
///
/// ### Returns
/// N/A
pub fn deriv2_element(
    i: i64,
    j: i64,
    da: &mut ContourGreen,
    a: &ContourGreen,
    acc: Option<&ContourGreen>,
    integ: &Integrator,
    h: f64,
) {
    let k = integ.k;
    let k1 = k + 1;
    let nt = a.nt as i64;
    assert!(h > 0.0);
    assert!((0..=nt).contains(&i) && (0..=nt).contains(&j));
    let b = a.size1 * a.size2;
    let mut c = vec![Complex::ZERO; b];
    let mut adj = vec![Complex::ZERO; b];
    let herm = acc.unwrap_or(a);
    if j <= i {
        c.fill(Complex::ZERO);
        if j < k1 {
            for l in 0..=k {
                let w = integ.pdw.at(j as usize, l as usize);
                if i < l {
                    // 1-based: i < l + 1 <=> i < l here
                    adjoint_block_into(herm.ret_raw_block(l, i), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.ret_raw_block(i, l);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        } else {
            for l in 0..=k1 {
                let w = integ.bdw.at(l as usize);
                let blk = a.ret_raw_block(i, j - l);
                for (cc, s) in c.iter_mut().zip(blk) {
                    *cc += *s * w;
                }
            }
        }
        let dst = da.ret_raw_block_mut(i, j);
        let f = Complex::new(0.0, -1.0 / h);
        for (d, cc) in dst.iter_mut().zip(&c) {
            *d = *cc * f;
        }
    }
    if j >= i {
        c.fill(Complex::ZERO);
        if j < k1 {
            for l in 0..=k {
                let w = integ.pdw.at(j as usize, l as usize);
                if i > l {
                    // 1-based: i > l + 1 <=> i > l here
                    adjoint_block_into(herm.less_raw_block(l, i), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.less_raw_block(i, l);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        } else {
            for l in 0..=k1 {
                let w = integ.bdw.at(l as usize);
                if i > j - l {
                    adjoint_block_into(herm.less_raw_block(j - l, i), &mut adj, a.size1, a.size2);
                    for (cc, aa) in c.iter_mut().zip(&adj) {
                        *cc -= *aa * w;
                    }
                } else {
                    let blk = a.less_raw_block(i, j - l);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
        }
        let dst = da.less_raw_block_mut(i, j);
        let f = Complex::new(0.0, -1.0 / h);
        for (d, cc) in dst.iter_mut().zip(&c) {
            *d = *cc * f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic;

    use crate::linalg::DenseMat;
    use crate::types::{Contour, Sign};

    use crate::types::MATSUBARA;

    use crate::equilibrium::from_ham;

    /// Deterministic pseudo-random complex stream (splitmix64).
    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }

        fn c64(&mut self) -> Complex {
            Complex::new(self.next_f64(), self.next_f64())
        }
    }

    /// Two-variable matrix polynomial of degree `deg` with `a + b <= deg`
    /// terms.
    struct Poly2 {
        deg: usize,
        nel: usize,
        c: Vec<Vec<Vec<Complex>>>, // c[a][b] = block
    }

    impl Poly2 {
        fn new(deg: usize, nel: usize, seed: u64) -> Self {
            let mut rng = Rng(seed);
            let mut c = vec![vec![vec![Complex::ZERO; nel]; deg + 1]; deg + 1];
            for (a, ca) in c.iter_mut().enumerate() {
                for (b, cab) in ca.iter_mut().enumerate() {
                    if a + b <= deg {
                        for z in cab.iter_mut() {
                            *z = rng.c64();
                        }
                    }
                }
            }
            Self { deg, nel, c }
        }

        fn f(&self, t: f64, s: f64) -> Vec<Complex> {
            let mut out = vec![Complex::ZERO; self.nel];
            for a in 0..=self.deg {
                for b in 0..=self.deg {
                    if a + b <= self.deg {
                        let w = Complex::new(t.powi(a as i32) * s.powi(b as i32), 0.0);
                        for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                            *o += *cz * w;
                        }
                    }
                }
            }
            out
        }

        /// d/dt.
        fn d1(&self, t: f64, s: f64) -> Vec<Complex> {
            let mut out = vec![Complex::ZERO; self.nel];
            for a in 1..=self.deg {
                for b in 0..=self.deg {
                    if a + b <= self.deg {
                        let w =
                            Complex::new((a as f64) * t.powi(a as i32 - 1) * s.powi(b as i32), 0.0);
                        for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                            *o += *cz * w;
                        }
                    }
                }
            }
            out
        }

        /// d/ds.
        fn d2(&self, t: f64, s: f64) -> Vec<Complex> {
            let mut out = vec![Complex::ZERO; self.nel];
            for a in 0..=self.deg {
                for b in 1..=self.deg {
                    if a + b <= self.deg {
                        let w =
                            Complex::new((b as f64) * t.powi(a as i32) * s.powi(b as i32 - 1), 0.0);
                        for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                            *o += *cz * w;
                        }
                    }
                }
            }
            out
        }
    }

    /// Left derivative of the lmix row (test helper).
    fn deriv1_lmix(n: i64, da: &mut ContourGreen, a: &ContourGreen, integ: &Integrator, h: f64) {
        let k = integ.k;
        let k1 = k + 1;
        assert!(h > 0.0);
        assert!((0..=a.nt as i64).contains(&n));
        let b = a.size1 * a.size2;
        let mut c = vec![Complex::ZERO; b];
        for m in 0..=(a.ntau as i64) {
            c.fill(Complex::ZERO);
            if n < k1 {
                for l in 0..=k {
                    let w = integ.pdw.at(n as usize, l as usize);
                    let blk = a.lmix_raw_block(l, m);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            } else {
                for l in 0..=k1 {
                    let w = integ.bdw.at(l as usize);
                    let blk = a.lmix_raw_block(n - l, m);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
            let dst = da.lmix_raw_block_mut(n, m);
            let f = Complex::new(0.0, 1.0 / h);
            for (d, cc) in dst.iter_mut().zip(&c) {
                *d = *cc * f;
            }
        }
    }

    /// Left derivative at one time step (test helper); `MATSUBARA`
    /// addresses the Matsubara component.
    fn deriv1_timestep(
        tstp: i64,
        da: &mut ContourGreen,
        a: &ContourGreen,
        acc: Option<&ContourGreen>,
        integ: &Integrator,
        beta: f64,
        h: f64,
    ) {
        let nt = a.nt as i64;
        if tstp == MATSUBARA {
            deriv1_matsubara(da, a, integ, beta);
            return;
        }
        assert!((0..=nt).contains(&tstp));
        assert!(!ptr::eq(da, a), "output must differ from input A");
        if let Some(x) = acc {
            assert!(!ptr::eq(da, x), "output must differ from input Acc");
        }
        for j in 0..=tstp {
            deriv1_element(tstp, j, da, a, acc, integ, h);
        }
        for j in tstp..=nt {
            deriv1_element(tstp, j, da, a, acc, integ, h);
        }
        deriv1_lmix(tstp, da, a, integ, h);
    }

    /// Right derivative of the lmix row w.r.t. tau (test helper).
    fn deriv2_lmix(n: i64, da: &mut ContourGreen, a: &ContourGreen, integ: &Integrator, beta: f64) {
        let k = integ.k;
        let k1 = k + 1;
        assert!(beta > 0.0);
        assert!((0..=a.nt as i64).contains(&n));
        let hv = beta / a.ntau as f64;
        let mut c = vec![Complex::ZERO; a.size1 * a.size2];
        for m in 0..=(a.ntau as i64) {
            c.fill(Complex::ZERO);
            if m < k1 {
                for l in 0..=k {
                    let w = integ.pdw.at(m as usize, l as usize);
                    let blk = a.lmix_raw_block(n, l);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            } else {
                for l in 0..=k1 {
                    let w = integ.bdw.at(l as usize);
                    let blk = a.lmix_raw_block(n, m - l);
                    for (cc, s) in c.iter_mut().zip(blk) {
                        *cc += *s * w;
                    }
                }
            }
            let dst = da.lmix_raw_block_mut(n, m);
            for (d, cc) in dst.iter_mut().zip(&c) {
                *d = *cc / hv;
            }
        }
    }

    /// Right derivative at one time step (test helper).
    fn deriv2_timestep(
        tstp: i64,
        da: &mut ContourGreen,
        a: &ContourGreen,
        acc: Option<&ContourGreen>,
        integ: &Integrator,
        beta: f64,
        h: f64,
    ) {
        let nt = a.nt as i64;
        if tstp == MATSUBARA {
            deriv1_matsubara(da, a, integ, beta);
            return;
        }
        assert!((0..=nt).contains(&tstp));
        assert!(!ptr::eq(da, a), "output must differ from input A");
        if let Some(x) = acc {
            assert!(!ptr::eq(da, x), "output must differ from input Acc");
        }
        for j in 0..=tstp {
            deriv2_element(tstp, j, da, a, acc, integ, h);
        }
        for j in tstp..=nt {
            deriv2_element(tstp, j, da, a, acc, integ, h);
        }
        deriv2_lmix(tstp, da, a, integ, beta);
    }

    #[allow(clippy::too_many_arguments)]
    fn df_fill(
        a: &mut ContourGreen,
        acc: &mut ContourGreen,
        h: f64,
        fr: &Poly2,
        fl: &Poly2,
        fm: &Poly2,
        fx: &Poly2,
        hv: f64,
    ) {
        let (nt, ntau) = (a.nt as i64, a.ntau as i64);
        for i in 0..=nt {
            for j in 0..=i {
                a.ret_raw_block_mut(i, j)
                    .copy_from_slice(&fr.f(i as f64 * h, j as f64 * h));
                acc.ret_raw_block_mut(i, j)
                    .copy_from_slice(&neg_adj(&fr.f(j as f64 * h, i as f64 * h)));
            }
        }
        for j in 0..=nt {
            for i in 0..=j {
                a.less_raw_block_mut(i, j)
                    .copy_from_slice(&fl.f(i as f64 * h, j as f64 * h));
                acc.less_raw_block_mut(i, j)
                    .copy_from_slice(&neg_adj(&fl.f(j as f64 * h, i as f64 * h)));
            }
        }
        for i in 0..=nt {
            for m in 0..=ntau {
                let v = fm.f(i as f64 * h, m as f64 * hv);
                a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
            }
        }
        for m in 0..=ntau {
            let v = fx.f(m as f64 * hv, 0.0);
            a.mat_raw_block_mut(m).copy_from_slice(&v);
            acc.mat_raw_block_mut(m).copy_from_slice(&v);
        }
    }

    /// ret(i, j) = i e^{-i eps (t_i - t_j)}: anti-hermitian (matches the
    /// tilde-G^R continuation G(t,t') = -G(t',t)^dagger) and solves
    /// i d/dt_1 G = eps G. Scalar size 1.
    fn oscillating(nt: usize, ntau: usize, eps: f64, dt: f64, beta: f64) -> ContourGreen {
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * dt, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt {
            for j in 0..=i {
                let v =
                    Complex::from_polar(1.0, -eps * (i - j) as f64 * dt) * Complex::new(0.0, 1.0);
                g.set_ret(i as i64, j as i64, 0, 0, v);
            }
            for jj in i..=nt {
                let v =
                    Complex::from_polar(1.0, -eps * (jj - i) as f64 * dt) * Complex::new(0.0, -1.0);
                g.set_less(i as i64, jj as i64, 0, 0, v);
            }
        }
        g
    }

    /// -(block)† for square blocks.
    fn neg_adj(blk: &[Complex]) -> Vec<Complex> {
        let n = (blk.len() as f64).sqrt() as usize;
        let mut out = vec![Complex::ZERO; blk.len()];
        for a in 0..n {
            for b in 0..n {
                out[b * n + a] = -blk[a * n + b].conj();
            }
        }
        out
    }

    fn dm_at(m: &DenseMat, q: usize) -> Complex {
        m.get(q / m.ncols(), q % m.ncols())
    }

    fn max_blk_err(err: f64, got: &[Complex], want: &[Complex]) -> f64 {
        got.iter()
            .zip(want)
            .fold(err, |e, (g, w)| e.max((*g - *w).abs()))
    }

    #[test]
    fn deriv1_matsubara_of_exponential() {
        // A.mat(n) = e^{eps tau_n}: dA = c / (-hv) patterns of the BDF;
        // check the interior consistency: d/dtau e^{eps tau} = eps e^{eps tau}
        let (ntau, eps, beta) = (40usize, 0.3, 8.0);
        let dtau = beta / ntau as f64;
        let c = Contour::new(4, ntau as i64, 1, 1, 1.0, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for n in 0..=(ntau as i64) {
            a.set_mat(n, 0, 0, Complex::new((eps * n as f64 * dtau).exp(), 0.0));
        }
        let integ = Integrator::new(2);
        deriv1_matsubara(&mut da, &a, &integ, beta);
        // the formula divides by -hv (contour orientation): the result
        // is -(d/dtau) of the stored values
        for n in 5..=(ntau as i64 - 5) {
            let want = a.get_mat(n, 0, 0).scale(-eps);
            let got = da.get_mat(n, 0, 0);
            assert!((got - want).abs() < 2e-4, "n={n}: {got} vs {want}");
        }
    }

    #[test]
    fn deriv1_element_ret_of_oscillation() {
        // ret(i, j) = e^{-i eps (t_i - t_j)}: i d/dt_1 A = eps A
        let (nt, eps, dt) = (30usize, 0.5, 0.1);
        let a = oscillating(nt, 8, eps, dt, 4.0);
        let mut da = a.clone();
        let integ = Integrator::new(3);
        // interior points of a row
        let i = 20i64;
        for j in 0..=i {
            deriv1_element(i, j, &mut da, &a, None, &integ, dt);
        }
        // skip the first few columns (startup corrections touch Acc
        // regions we did not fill consistently here)
        for j in 5..=i {
            let want = a.get_ret(i, j, 0, 0).scale(eps);
            let got = da.get_ret(i, j, 0, 0);
            assert!((got - want).abs() < 5e-5, "j={j}: {got} vs {want}");
        }
    }

    #[test]
    fn deriv1_lmix_of_oscillation() {
        // A.lmix(n, m) = e^{-i eps t_n}: i d/dt A = eps * A
        let (nt, ntau, eps, dt, beta) = (30usize, 8, 0.7, 0.1, 4.0);
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * dt, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for n in 0..=nt as i64 {
            for m in 0..=(ntau as i64) {
                a.set_lmix(n, m, 0, 0, Complex::from_polar(1.0, -eps * n as f64 * dt));
            }
        }
        let integ = Integrator::new(3);
        for n in (3 + 1)..=(nt as i64 - 1) {
            deriv1_lmix(n, &mut da, &a, &integ, dt);
            for m in 0..=(ntau as i64) {
                let want = Complex::from_polar(1.0, -eps * n as f64 * dt).scale(eps);
                let got = da.get_lmix(n, m, 0, 0);
                assert!((got - want).abs() < 5e-5, "n={n} m={m}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn deriv2_lmix_of_exponential_tau() {
        // A.lmix(n, m) = e^{eps tau_m}: d/d tau A = eps A
        let (nt, ntau, eps, beta) = (10usize, 40usize, 0.3, 8.0);
        let dtau = beta / ntau as f64;
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, 1.0, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for m in 0..=(ntau as i64) {
            let v = Complex::new((eps * m as f64 * dtau).exp(), 0.0);
            a.set_lmix(5, m, 0, 0, v);
        }
        let integ = Integrator::new(3);
        deriv2_lmix(5, &mut da, &a, &integ, beta);
        for m in 4..=(ntau as i64 - 4) {
            let want = a.get_lmix(5, m, 0, 0).scale(eps);
            let got = da.get_lmix(5, m, 0, 0);
            assert!((got - want).abs() < 5e-5, "m={m}: {got} vs {want}");
        }
    }

    #[test]
    fn polynomial_exactness() {
        let ic = Complex::new(0.0, 1.0);
        for k in 1..=5i64 {
            for nel in [1usize, 4] {
                let dims = if nel == 1 { (1, 1) } else { (2, 2) };
                // grid: ntime = 2k+5 points, ntau = 2k+4 points
                let (nt, ntau) = (2 * k + 4, 2 * k + 3);
                let (h, beta) = (0.1f64, 0.7f64);
                let hv = beta / ntau as f64;
                let c = Contour::new(nt, ntau, dims.0, dims.1, nt as f64 * h, beta);
                let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                let fr = Poly2::new(k as usize, nel, 0x1234 + k as u64);
                let fl = Poly2::new(k as usize, nel, 0x2345 + k as u64);
                let fm = Poly2::new(k as usize, nel, 0x3456 + k as u64);
                let fx = Poly2::new(k as usize + 1, nel, 0x4567 + k as u64);
                df_fill(&mut a, &mut acc, h, &fr, &fl, &fm, &fx, hv);

                // left derivative, all time steps (MATSUBARA + real)
                let integ = Integrator::new(k);
                let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                for tstp in -1..=nt {
                    deriv1_timestep(tstp, &mut da, &a, Some(&acc), &integ, beta, h);
                }
                let mut err = 0.0f64;
                for i in 0..=nt {
                    for j in 0..=i {
                        let want = fr
                            .d1(i as f64 * h, j as f64 * h)
                            .iter()
                            .map(|z| ic * *z)
                            .collect::<Vec<_>>();
                        err = max_blk_err(err, da.ret_raw_block(i, j), &want);
                    }
                    for j in i..=nt {
                        let want = fl
                            .d1(i as f64 * h, j as f64 * h)
                            .iter()
                            .map(|z| ic * *z)
                            .collect::<Vec<_>>();
                        err = max_blk_err(err, da.less_raw_block(i, j), &want);
                    }
                    for m in 0..=ntau {
                        let want = fm
                            .d1(i as f64 * h, m as f64 * hv)
                            .iter()
                            .map(|z| ic * *z)
                            .collect::<Vec<_>>();
                        err = max_blk_err(err, da.lmix_raw_block(i, m), &want);
                    }
                }
                for m in 0..=ntau {
                    // identity: dA.mat + Dx1 == 0  =>  dA.mat = -Dx1
                    let want: Vec<Complex> =
                        fx.d1(m as f64 * hv, 0.0).iter().map(|z| -*z).collect();
                    err = max_blk_err(err, da.mat_raw_block(m), &want);
                }
                assert!(err <= 1e-10, "k={k} nel={nel} left err={err}");

                // right derivative
                let mut db = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                for tstp in -1..=nt {
                    deriv2_timestep(tstp, &mut db, &a, Some(&acc), &integ, beta, h);
                }
                let mut err = 0.0f64;
                for i in 0..=nt {
                    for j in 0..=i {
                        let want = fr
                            .d2(i as f64 * h, j as f64 * h)
                            .iter()
                            .map(|z| -(ic * *z))
                            .collect::<Vec<_>>();
                        err = max_blk_err(err, db.ret_raw_block(i, j), &want);
                    }
                    for j in i..=nt {
                        let want = fl
                            .d2(i as f64 * h, j as f64 * h)
                            .iter()
                            .map(|z| -(ic * *z))
                            .collect::<Vec<_>>();
                        err = max_blk_err(err, db.less_raw_block(i, j), &want);
                    }
                    for m in 0..=ntau {
                        let want = fm.d2(i as f64 * h, m as f64 * hv);
                        err = max_blk_err(err, db.lmix_raw_block(i, m), &want);
                    }
                }
                assert!(err <= 1e-10, "k={k} nel={nel} right err={err}");
                assert!(db.mat.equal(&da.mat));
            }
        }
    }

    #[test]
    fn hermitian_gf_left_and_right_derivatives() {
        // grid: ntime = 40 points, ntau = 30 points
        let (nt, ntau) = (39i64, 29i64);
        let (beta, h, mu) = (2.0f64, 0.01f64, 0.1f64);
        let mut hm = DenseMat::zeros(2);
        hm.set(0, 0, Complex::new(-0.4, 0.0));
        hm.set(1, 1, Complex::new(0.6, 0.0));
        hm.set(0, 1, Complex::new(0.0, 0.3));
        hm.set(1, 0, Complex::new(0.0, -0.3));
        let ident = DenseMat::identity(2);
        for sign in [Sign::Fermi, Sign::Bose] {
            let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
            let mut g = ContourGreen::new(&c, sign, Complex::ZERO);
            // H + 2I for bosons
            let mut h_use = hm.clone();
            if sign == Sign::Bose {
                h_use.axpy(2.0, &ident);
            }
            from_ham(&mut g, mu, &h_use, beta, h);
            // Hs = H - mu I (+ 2I for bosons)
            let mut hs = hm.clone();
            if sign == Sign::Bose {
                hs.axpy(2.0, &ident);
            }
            let mut hs2 = hs.clone();
            hs2.axpy(-mu, &ident);
            let k = 5;
            let integ = Integrator::new(k);
            let mut da = ContourGreen::new(&c, sign, Complex::ZERO);
            let mut db = ContourGreen::new(&c, sign, Complex::ZERO);
            for tstp in 0..=nt {
                deriv1_timestep(tstp, &mut da, &g, None, &integ, beta, h);
                deriv2_timestep(tstp, &mut db, &g, None, &integ, beta, h);
            }
            let mut err = 0.0f64;
            for i in 0..=nt {
                for j in 0..=i {
                    let gr = DenseMat::from_vec(2, 2, g.ret_raw_block(i, j).to_vec());
                    let want_l = hs2.mat_mul(&gr);
                    let want_r = gr.mat_mul(&hs2);
                    let dr = DenseMat::from_vec(2, 2, da.ret_raw_block(i, j).to_vec());
                    let drr = DenseMat::from_vec(2, 2, db.ret_raw_block(i, j).to_vec());
                    for q in 0..4 {
                        err = err.max((dm_at(&dr, q) - dm_at(&want_l, q)).abs());
                        err = err.max((dm_at(&drr, q) - dm_at(&want_r, q)).abs());
                    }
                }
                for j in i..=nt {
                    let gl = DenseMat::from_vec(2, 2, g.less_raw_block(i, j).to_vec());
                    let want_ll = hs2.mat_mul(&gl);
                    let want_lr = gl.mat_mul(&hs2);
                    let dl = DenseMat::from_vec(2, 2, da.less_raw_block(i, j).to_vec());
                    let dlr = DenseMat::from_vec(2, 2, db.less_raw_block(i, j).to_vec());
                    for q in 0..4 {
                        err = err.max((dm_at(&dl, q) - dm_at(&want_ll, q)).abs());
                        err = err.max((dm_at(&dlr, q) - dm_at(&want_lr, q)).abs());
                    }
                }
                for m in 0..=ntau {
                    let gm = DenseMat::from_vec(2, 2, g.lmix_raw_block(i, m).to_vec());
                    let want = hs2.mat_mul(&gm);
                    let dm = DenseMat::from_vec(2, 2, da.lmix_raw_block(i, m).to_vec());
                    for q in 0..4 {
                        err = err.max((dm_at(&dm, q) - dm_at(&want, q)).abs());
                    }
                }
            }
            assert!(err <= 1e-8, "sign={:?} err={err}", sign);
        }
    }

    #[test]
    fn timestep_entry_equals_element_calls() {
        let (nt, ntau) = (9i64, 8i64);
        let (h, beta) = (0.1f64, 1.0f64);
        let _hv = beta / ntau as f64;
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // the test reuses a degree-4 polynomial with tau = 0.1 * m
        let fm = Poly2::new(4, 4, 0x9999);
        let fr = Poly2::new(4, 4, 0xAAAA);
        let fl = Poly2::new(4, 4, 0xBBBB);
        // fm-only fill with the tau rule (t, 0.1*(m+1)) etc.
        let fill = |a: &mut ContourGreen, acc: &mut ContourGreen| {
            for i in 0..=nt {
                for j in 0..=i {
                    a.ret_raw_block_mut(i, j)
                        .copy_from_slice(&fr.f(i as f64 * h, j as f64 * h));
                    acc.ret_raw_block_mut(i, j)
                        .copy_from_slice(&neg_adj(&fr.f(j as f64 * h, i as f64 * h)));
                }
                for j in i..=nt {
                    a.less_raw_block_mut(i, j)
                        .copy_from_slice(&fl.f(i as f64 * h, j as f64 * h));
                    acc.less_raw_block_mut(i, j)
                        .copy_from_slice(&neg_adj(&fl.f(j as f64 * h, i as f64 * h)));
                }
            }
            for i in 0..=nt {
                for m in 0..=ntau {
                    let v = fm.f(i as f64 * h, 0.1 * (m + 1) as f64);
                    a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                    acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                }
            }
            for m in 0..=ntau {
                let v = fm.f(0.1 * (m + 1) as f64, 0.0);
                a.mat_raw_block_mut(m).copy_from_slice(&v);
                acc.mat_raw_block_mut(m).copy_from_slice(&v);
            }
        };
        fill(&mut a, &mut acc);

        let integ = Integrator::new(3);
        let sentinel = Complex::new(3.0, -1.0);
        for tstp in [1i64, 3, 6] {
            // left
            let mut da = ContourGreen::new(&c, Sign::Fermi, sentinel);
            let mut de = ContourGreen::new(&c, Sign::Fermi, sentinel);
            deriv1_timestep(tstp, &mut da, &a, Some(&acc), &integ, beta, h);
            for j in 0..=nt {
                deriv1_element(tstp, j, &mut de, &a, Some(&acc), &integ, h);
            }
            deriv1_lmix(tstp, &mut de, &a, &integ, h);
            assert!(da.equal(&de), "tstp {tstp} left");
            // only the target time step is touched
            for i in 0..=nt {
                if i == tstp {
                    continue;
                }
                for j in 0..=i {
                    assert!(da.ret_raw_block(i, j).iter().all(|z| *z == sentinel));
                }
                for j in i..=nt {
                    assert!(da.less_raw_block(i, j).iter().all(|z| *z == sentinel));
                }
                for m in 0..=ntau {
                    assert!(da.lmix_raw_block(i, m).iter().all(|z| *z == sentinel));
                }
            }
            assert!(da.mat.raw().iter().all(|z| *z == sentinel));
            // right
            let mut db = ContourGreen::new(&c, Sign::Fermi, sentinel);
            let mut de = ContourGreen::new(&c, Sign::Fermi, sentinel);
            deriv2_timestep(tstp, &mut db, &a, Some(&acc), &integ, beta, h);
            for j in 0..=nt {
                deriv2_element(tstp, j, &mut de, &a, Some(&acc), &integ, h);
            }
            deriv2_lmix(tstp, &mut de, &a, &integ, beta);
            assert!(db.equal(&de), "tstp {tstp} right");
        }
    }

    #[test]
    fn matsubara_grid_lower_bound() {
        for k in 1..=5i64 {
            let integ = Integrator::new(k);
            // minimal valid grid: ntau = 2k+1 intervals (2k+2 points)
            let c = Contour::new(k + 1, 2 * k + 1, 1, 1, 1.0, 1.0);
            let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            for m in 0..=c.ntau {
                a.set_mat(m, 0, 0, Complex::new((m + 1) as f64 * (m + 1) as f64, 0.0));
            }
            let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            deriv1_matsubara(&mut da, &a, &integ, 1.0);
            assert!(
                da.mat
                    .raw()
                    .iter()
                    .all(|z| z.re.is_finite() && z.im.is_finite())
            );
        }
        // one point less must be rejected
        {
            let result = panic::catch_unwind(|| {
                let k = 3i64;
                let integ = Integrator::new(k);
                let c = Contour::new(k + 1, 2 * k, 1, 1, 1.0, 1.0);
                let a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                deriv1_matsubara(&mut da, &a, &integ, 1.0);
            });
            assert!(result.is_err(), "ntau = 2k must be rejected");
        }
        // other preconditions
        let c = Contour::new(5, 5, 1, 1, 1.0, 1.0);
        let a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let acc_g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let _out = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ2 = Integrator::new(2);
        // invalid tstp
        let r = panic::catch_unwind(|| {
            let mut o = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            deriv1_element(-1, 0, &mut o, &a, Some(&acc_g), &integ2, 0.1);
        });
        assert!(r.is_err());
        let r = panic::catch_unwind(|| {
            let mut o = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            deriv1_lmix(6, &mut o, &a, &integ2, 0.1);
        });
        assert!(r.is_err());
        // NOTE (deviation): the original test suite also asserts that using the output
        // object as input A or Acc is rejected. In safe Rust the aliased call
        // `deriv1_element(.., &mut g, &g, ..)` cannot even be constructed
        // (E0502), so the library-side `ptr::eq` guards stand untested here;
        // the compiler enforces the same guarantee statically.
    }
}
