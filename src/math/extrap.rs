// Project : KadanoffBaym
// File    : src/math/extrap.rs
// Purpose : polynomial extrapolation of a time step
//
// Blueprint: julia/src/math/extrap.jl. Weights
// `p_l = sum_m (1 - 2*(l % 2)) * PIW[l, m]`; the extrapolated row
// `n + 1` of ret/lmix/less combines the previous `k + 1` rows with the
// hermiticity relations for the innermost columns/rows.
//
// Julia -> Rust index translation: the Rust argument `n` is the 0-based
// index of the last FILLED row; everything is extrapolated to `n + 1`.
// Julia's 1-based absolute times `j` map to `j - 1`.

use crate::complex::c64::C64;
use crate::core::green::ContourGreen;
use crate::weights::Integrator;

/// Extrapolation weights `p_l` (`l = 0..k`), blueprint
/// `_extrapolation_weights`.
pub fn extrapolation_weights(integ: &Integrator) -> Vec<f64> {
    extrapolation_weights_order(integ.k)
}

/// Extrapolation weights for an explicit order `k` (0 = copy).
pub fn extrapolation_weights_order(k: i64) -> Vec<f64> {
    assert!((0..=5).contains(&k), "extrapolation order 0 <= k <= 5, got {k}");
    let piw = crate::weights::PolynomialInterpolationWeights::new(k);
    let n = (k + 1) as usize;
    let mut p = vec![0.0f64; n];
    for l in 0..n {
        let sign = 1.0 - 2.0 * (l % 2) as f64;
        for m in 0..n {
            p[m] += sign * piw.at(l, m);
        }
    }
    p
}

/// Extrapolate a `TimeFunction` one step: `f[n+1] = sum_l p[l] * f[n-l]`
/// (blueprint `extrapolate_timestep!(n, f::Cf, k)`).
pub fn extrapolate_timestep_cf(n: i64, f: &mut crate::core::green::TimeFunction, k: i64) {
    let p = extrapolation_weights_order(k);
    assert!(n >= k, "extrapolate_timestep_cf: n >= k");
    assert!(n <= f.nt as i64 - 1, "extrapolate_timestep_cf: n <= nt-1");
    let nel = f.size1 * f.size2;
    let mut acc = vec![C64::ZERO; nel];
    for l in 0..=(k as usize) {
        let src = f.block(n - l as i64);
        for (a, s) in acc.iter_mut().zip(src) {
            *a += *s * p[l];
        }
    }
    f.block_mut(n + 1).copy_from_slice(&acc);
}

/// Zeroth order (copy) extrapolation of the ret/lmix/less rows to
/// `n + 1`, blueprint `_extrapolate_copy!` (fixed relative time).
pub fn extrapolate_copy(n: i64, g: &mut ContourGreen) {
    assert!(n >= 0 && n < g.nt as i64, "extrapolate_copy: 0 <= n < nt");
    let n1 = n + 1;
    // lmix row
    let b = g.size1 * g.size2;
    let rowlen = (g.ntau + 1) * b;
    let off = n as usize * rowlen;
    g.lmix
        .raw_mut()
        .copy_within(off..off + rowlen, off + rowlen);
    // ret row: ret(n1, n1-j) = ret(n, n-j) for j = 0..n (Julia
    // `j = 0 : n-1` with n_julia = n + 1)
    for j in 0..=n {
        let sb = g.ret.block(g.ret.idx(n, n - j)).to_vec();
        let dst = g.ret.idx(n1, n1 - j);
        g.ret.block_mut(dst).copy_from_slice(&sb);
    }
    // plus the first column: ret(n1, 0) = ret(n, 0)
    {
        let sb = g.ret.block(g.ret.idx(n, 0)).to_vec();
        let dst = g.ret.idx(n1, 0);
        g.ret.block_mut(dst).copy_from_slice(&sb);
    }
    // less rows: less(j, n1) = less(j-1, n) for j = 1..=n1 (Julia
    // `j = 2 : n1`), plus the first row less(0, n1) = less(0, n)
    for j in 1..=n1 {
        let sb = g.less.block(g.less.idx(j - 1, n)).to_vec();
        let dst = g.less.idx(j, n1);
        g.less.block_mut(dst).copy_from_slice(&sb);
    }
    {
        let sb = g.less.block(g.less.idx(0, n)).to_vec();
        let dst = g.less.idx(0, n1);
        g.less.block_mut(dst).copy_from_slice(&sb);
    }
}

/// Adjoint of a row-major block (`size1 x size2`) into `out`
/// (`adjoint: out[a][b] = src[b][a]^*`).
#[inline]
fn adjoint_block_into(src: &[C64], out: &mut [C64], s1: usize, s2: usize) {
    for a in 0..s1 {
        for b in 0..s2 {
            out[a * s2 + b] = src[b * s1 + a].conj();
        }
    }
}

/// Polynomial extrapolation of the ret/lmix/less rows of `g` to the
/// time step `n + 1` (0-based `n`), blueprint `extrapolate_timestep!`.
pub fn extrapolate_timestep(n: i64, g: &mut ContourGreen, integ: &Integrator) {
    let k = integ.k as i64;
    let n1 = n + 1;
    assert!(n >= k, "extrapolate_timestep: n {n} >= k {k}");
    assert!(n1 <= g.nt as i64, "extrapolate_timestep: n + 1 <= nt");
    let p = extrapolation_weights(integ);
    let (s1, s2) = (g.size1, g.size2);
    let b = s1 * s2;
    let mut gtmp = vec![C64::ZERO; b];
    let mut adj = vec![C64::ZERO; b];

    // lmix row: Julia `for m = 1:ntau` -> Rust m = 0..=ntau
    for m in 0..=(g.ntau as i64) {
        gtmp.fill(C64::ZERO);
        for l in 0..=k {
            let src = g.lmix.idx(n - l, m);
            for (t, s) in gtmp.iter_mut().zip(g.lmix.block(src)) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.lmix.idx(n1, m);
        g.lmix.block_mut(dst).copy_from_slice(&gtmp);
    }

    // ret row, fixed relative time: Julia `j = 0 : n-1-k` (relative,
    // inclusive) with Julia n = Rust n + 1 -> Rust j = 0..=(n-k)
    for j in 0..=(n - k) {
        gtmp.fill(C64::ZERO);
        for l in 0..=k {
            let src = g.ret.idx(n - l, n - l - j);
            for (t, s) in gtmp.iter_mut().zip(g.ret.block(src)) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.ret.idx(n1, n1 - j);
        g.ret.block_mut(dst).copy_from_slice(&gtmp);
    }

    // ret row, the first k + 1 columns: Julia `j = 1 : k+1` (absolute)
    // -> Rust column c = j - 1 = 0..=k
    for c in 0..=k {
        gtmp.fill(C64::ZERO);
        for l in 0..=k {
            // Julia condition `n - l <= j` <=> Rust `n - l <= c + 1 - 1 + 1`
            // i.e. `n_julia - l <= j` with n_julia = n + 1, j = c + 1
            if n + 1 - l <= c + 1 {
                // src = ret(c, n - l), subtract its adjoint
                let src = g.ret.idx(c, n - l);
                adjoint_block_into(g.ret.block(src), &mut adj, s1, s2);
                for (t, a) in gtmp.iter_mut().zip(&adj) {
                    *t -= *a * p[l as usize];
                }
            } else {
                let src = g.ret.idx(n - l, c);
                for (t, s) in gtmp.iter_mut().zip(g.ret.block(src)) {
                    *t += *s * p[l as usize];
                }
            }
        }
        let dst = g.ret.idx(n1, c);
        g.ret.block_mut(dst).copy_from_slice(&gtmp);
    }

    // less, the first k + 1 rows: Julia `j = 1 : k+1` -> Rust r = 0..=k
    for r in 0..=k {
        gtmp.fill(C64::ZERO);
        for l in 0..=k {
            // Julia `n - l < j` with n_julia = n + 1, j = r + 1
            if n + 1 - l < r + 1 {
                // src = less(n - l, r), subtract its adjoint
                let src = g.less.idx(n - l, r);
                adjoint_block_into(g.less.block(src), &mut adj, s1, s2);
                for (t, a) in gtmp.iter_mut().zip(&adj) {
                    *t -= *a * p[l as usize];
                }
            } else {
                let src = g.less.idx(r, n - l);
                for (t, s) in gtmp.iter_mut().zip(g.less.block(src)) {
                    *t += *s * p[l as usize];
                }
            }
        }
        let dst = g.less.idx(r, n1);
        g.less.block_mut(dst).copy_from_slice(&gtmp);
    }

    // less, fixed relative time: Julia `j = k+2 : n1` (absolute rows)
    // -> Rust row r = j - 1 = k+1..=n+1; src less(r-1-l, n-l)
    for r in (k + 1)..=n1 {
        gtmp.fill(C64::ZERO);
        for l in 0..=k {
            let src = g.less.idx(r - 1 - l, n - l);
            for (t, s) in gtmp.iter_mut().zip(g.less.block(src)) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.less.idx(r, n1);
        g.less.block_mut(dst).copy_from_slice(&gtmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::contour::{Contour, Sign};

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 3, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, C64::ZERO)
    }

    #[test]
    fn weights_extrapolate_linear_functions() {
        // For k = 1 the weights reproduce f(n+1): p[0] pairs with f(n),
        // p[1] with f(n-1) (blueprint `p[l] * block(n - l)`)
        let w1 = extrapolation_weights(&Integrator::new(1));
        let t = 5.0f64;
        let got = w1[0] * t + w1[1] * (t - 1.0);
        assert!((got - (t + 1.0)).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn zeroth_order_copy_extrapolation() {
        let mut g = mk(4);
        for j in 0..=2 {
            g.set_ret(2, j, 0, 0, C64::new(10.0 + j as f64, 0.0));
            g.set_less(j, 2, 0, 0, C64::new(20.0 + j as f64, 0.0));
        }
        for m in 0..=3 {
            g.set_lmix(2, m, 0, 0, C64::new(30.0 + m as f64, 0.0));
        }
        extrapolate_copy(2, &mut g);
        // ret row 3 shifts by one: ret(3, j+1) = ret(2, j), and the
        // first column copies ret(2, 0)
        for j in 0..=2 {
            assert_eq!(g.get_ret(3, j + 1, 0, 0), C64::new(10.0 + j as f64, 0.0));
        }
        assert_eq!(g.get_ret(3, 0, 0, 0), C64::new(10.0, 0.0));
        // less rows shift: less(j, 3) = less(j-1, 2), first row copied
        assert_eq!(g.get_less(0, 3, 0, 0), C64::new(20.0, 0.0));
        assert_eq!(g.get_less(1, 3, 0, 0), C64::new(20.0, 0.0));
        assert_eq!(g.get_less(2, 3, 0, 0), C64::new(21.0, 0.0));
        assert_eq!(g.get_less(3, 3, 0, 0), C64::new(22.0, 0.0));
        for m in 0..=3 {
            assert_eq!(g.get_lmix(3, m, 0, 0), C64::new(30.0 + m as f64, 0.0));
        }
    }

    #[test]
    fn linear_data_extrapolates_exactly() {
        // ret(i, j) = i - j: linear in both arguments, k >= 1 is exact
        let mut g = mk(6);
        for i in 0..=4 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, C64::new((i - j) as f64, 0.0));
            }
            for jj in i..=4 {
                g.set_less(i, jj, 0, 0, C64::new(100.0 + (jj - i) as f64, 0.0));
            }
            for m in 0..=3 {
                g.set_lmix(i, m, 0, 0, C64::new(1000.0 + 10.0 * i as f64 + m as f64, 0.0));
            }
        }
        let integ = Integrator::new(2);
        extrapolate_timestep(4, &mut g, &integ);
        for j in 0..=5 {
            assert_eq!(
                g.get_ret(5, j, 0, 0),
                C64::new((5 - j) as f64, 0.0),
                "ret(5,{j})"
            );
        }
        for i in 0..=5 {
            assert_eq!(
                g.get_less(i, 5, 0, 0),
                C64::new(105.0 - i as f64, 0.0),
                "less({i},5)"
            );
        }
        for m in 0..=3 {
            assert_eq!(g.get_lmix(5, m, 0, 0), C64::new(1050.0 + m as f64, 0.0));
        }
    }
}

/// Extrapolation of the row `n + 1` with the explicit extrapolation
/// order `order`; order 0 copies along the time-translation diagonals
/// (blueprint `extrapolate_timestep!(n, G, order)`).
pub fn extrapolate_timestep_order(n: i64, g: &mut ContourGreen, order: i64) {
    assert!((0..=5).contains(&order), "order must be in 0..=5, got {order}");
    if order == 0 {
        crate::math::extrap::extrapolate_copy(n, g);
    } else {
        extrapolate_timestep(n, g, &crate::weights::Integrator::new(order));
    }
}
