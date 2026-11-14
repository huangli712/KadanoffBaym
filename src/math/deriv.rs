// Project : KadanoffBaym
// File    : src/math/deriv.rs
// Purpose : left/right derivatives of a contour Green's function
//
// Blueprint: julia/src/math/{deriv1,deriv2}.jl, translated to 0-based
// indices. All Julia 1-based times (i, j, n, m, t) map to 0-based
// Rust indices; the relative comparisons keep their meaning.
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
//! *References* :
//!
//! Please see [`NESSi`] `cntr_differentiation_impl.hpp`.

use crate::complex::Complex;
use crate::types::ContourGreen;
use crate::types::MATSUBARA;
use crate::weights::Integrator;

#[inline]
fn adjoint_block_into(src: &[Complex], out: &mut [Complex], s1: usize, s2: usize) {
    for a in 0..s1 {
        for b in 0..s2 {
            out[a * s2 + b] = src[b * s1 + a].conj();
        }
    }
}

/// Imaginary-time derivative of the Matsubara component
/// (blueprint `deriv1_matsubara!`); used by both `deriv1_timestep` and
/// `deriv2_timestep` at `MATSUBARA`.
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
    assert!(a.ntau as i64 >= 2 * k + 1, "deriv1_matsubara: ntau >= 2k+1");
    let hv = beta / n as f64;
    let b = a.size1 * a.size2;
    let mut c = vec![Complex::ZERO; b];

    // both boundaries
    for t in 0..=k {
        // c = -sum_l BDW[l] * mat[t + l]  (Julia mat[t+l+1])
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

/// Left derivative `i d/dt_1 A(t_i, t_j)` of one element
/// (blueprint `deriv1_element!`, 0-based `i`, `j`).
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
    assert!(!std::ptr::eq(da, a), "output must differ from input A");
    if let Some(x) = acc {
        assert!(!std::ptr::eq(da, x), "output must differ from input Acc");
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
                    // Julia: l + 1 < j  <=>  l < j - 1 (1-based) <=> l < j_rust
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
                    // Julia: l + 1 > j <=> l >= j_rust
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

/// Left derivative of the lmix row (blueprint `deriv1_lmix!`).
///
/// Calculate the derivative ``i \\partial_t A^{\\rceil}(t_n,\\tau)`` and store
/// it in the `lmix` component of `dA` at time step `n`.
///
/// ### Arguments
/// * n  -> Index of time step.
/// * dA -> Result, a contour-ordered Green's function.
/// * A  -> A contour-ordered Green's function.
/// * I  -> An Integrator struct.
/// * h  -> Time step at real time axis.
///
/// ### Returns
/// N/A
pub fn deriv1_lmix(n: i64, da: &mut ContourGreen, a: &ContourGreen, integ: &Integrator, h: f64) {
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

/// Left derivative at one time step (blueprint `deriv1_timestep!`);
/// `MATSUBARA` addresses the Matsubara component. `acc = None` means a
/// hermitian `A` (`Acc = A`).
///
/// Calculate the left derivative of `A` at time step `tstp`. If `tstp = 0`,
/// the Matsubara component is calculated. Otherwise, the `ret` component at
/// `(tstp, j ≤ tstp)`, the `less` component at `(tstp, j ≥ tstp)`, and the
/// `lmix` component at `tstp` are calculated.
///
/// ### Arguments
/// * tstp -> Index of time step.
/// * dA   -> Result, a contour-ordered Green's function.
/// * A    -> A contour-ordered Green's function.
/// * Acc  -> Hermitian conjugate partner of `A`.
/// * I    -> An Integrator struct.
/// * beta -> Inverse temperature.
/// * h    -> Time step at real time axis.
///
/// ### Returns
/// N/A
pub fn deriv1_timestep(
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
    assert!(!std::ptr::eq(da, a), "output must differ from input A");
    if let Some(x) = acc {
        assert!(!std::ptr::eq(da, x), "output must differ from input Acc");
    }
    for j in 0..=tstp {
        deriv1_element(tstp, j, da, a, acc, integ, h);
    }
    for j in tstp..=nt {
        deriv1_element(tstp, j, da, a, acc, integ, h);
    }
    deriv1_lmix(tstp, da, a, integ, h);
}

/// Right derivative `i d/dt_2 A(t_i, t_j)` of one element
/// (blueprint `deriv2_element!`, 0-based `i`, `j`).
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
                    // Julia: i < l + 1 <=> i_rust < l
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
                    // Julia: i > l + 1 <=> i_rust > l
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

/// Right derivative of the lmix row w.r.t. tau (blueprint
/// `deriv2_lmix!`).
///
/// Calculate the derivative ``\\partial_\\tau A^{\\rceil}(t_n,\\tau)`` and store
/// it in the `lmix` component of `dA` at time step `n`. Note that there is
/// no factor ``i``.
///
/// ### Arguments
/// * n    -> Index of time step.
/// * dA   -> Result, a contour-ordered Green's function.
/// * A    -> A contour-ordered Green's function.
/// * I    -> An Integrator struct.
/// * beta -> Inverse temperature.
///
/// ### Returns
/// N/A
pub fn deriv2_lmix(n: i64, da: &mut ContourGreen, a: &ContourGreen, integ: &Integrator, beta: f64) {
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

/// Right derivative at one time step (blueprint `deriv2_timestep!`);
/// `MATSUBARA` delegates to `deriv1_matsubara` (identical).
///
/// Calculate the right derivative of `A` at time step `tstp`. If `tstp = 0`,
/// the Matsubara component is calculated (it is identical to the left
/// derivative). Otherwise, the `ret` component at `(tstp, j ≤ tstp)`, the
/// `less` component at `(tstp, j ≥ tstp)`, and the `lmix` component at
/// `tstp` are calculated.
///
/// ### Arguments
/// * tstp -> Index of time step.
/// * dA   -> Result, a contour-ordered Green's function.
/// * A    -> A contour-ordered Green's function.
/// * Acc  -> Hermitian conjugate partner of `A`.
/// * I    -> An Integrator struct.
/// * beta -> Inverse temperature.
/// * h    -> Time step at real time axis.
///
/// ### Returns
/// N/A
pub fn deriv2_timestep(
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
    assert!(!std::ptr::eq(da, a), "output must differ from input A");
    if let Some(x) = acc {
        assert!(!std::ptr::eq(da, x), "output must differ from input Acc");
    }
    for j in 0..=tstp {
        deriv2_element(tstp, j, da, a, acc, integ, h);
    }
    for j in tstp..=nt {
        deriv2_element(tstp, j, da, a, acc, integ, h);
    }
    deriv2_lmix(tstp, da, a, integ, beta);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Contour;
    use crate::types::Sign;

    /// ret(i, j) = i e^{-i eps (t_i - t_j)}: anti-hermitian (matches the
    /// tilde-G^R continuation G(t,t') = -G(t',t)^dagger) and solves
    /// i d/dt_1 G = eps G. Scalar size 1.
    fn oscillating(nt: usize, ntau: usize, eps: f64, dt: f64, beta: f64) -> ContourGreen {
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * dt, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt {
            for j in 0..=i {
                let v = Complex::from_polar(1.0, -eps * (i - j) as f64 * dt) * Complex::new(0.0, 1.0);
                g.set_ret(i as i64, j as i64, 0, 0, v);
            }
            for jj in i..=nt {
                let v = Complex::from_polar(1.0, -eps * (jj - i) as f64 * dt) * Complex::new(0.0, -1.0);
                g.set_less(i as i64, jj as i64, 0, 0, v);
            }
        }
        g
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
        // the blueprint divides by -hv (contour orientation): the result
        // is -(d/dtau) of the stored values
        for n in 5..=(ntau as i64 - 5) {
            let want = a.get_mat(n, 0, 0).scale(-eps);
            let got = da.get_mat(n, 0, 0);
            assert!((got - want).abs() < 2e-4, "n={n}: {got} vs {want}");
        }
    }
}
