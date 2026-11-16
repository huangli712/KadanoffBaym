// Project : KadanoffBaym
// File    : src/math/extrap.rs
// Purpose : polynomial extrapolation of a time step
//
// `p_l = sum_m (1 - 2*(l % 2)) * PIW[l, m]`; the extrapolated row
// `n + 1` of ret/lmix/less combines the previous `k + 1` rows with the
// hermiticity relations for the innermost columns/rows.
//
// Index translation: the argument `n` is the 0-based index of the last
// FILLED row; everything is extrapolated to `n + 1`. The blueprint's
// 1-based absolute times `j` map to `j - 1`.

//!
//! *Remarks* : *Polynomial Extrapolation*
//!
//! Suppose that the contour function is known at the time steps ``t_{n-l}``
//! (``l = 0, \cdots, k``). The ``k``-th order polynomial approximant through
//! these points is evaluated at ``t_{n+1}``, which yields
//!
//! ```math
//! \begin{equation}
//! y_{n+1} \approx \sum^{k}_{l=0} p^{(k)}_l y_{n-l},
//! \end{equation}
//! ```
//!
//! with
//!
//! ```math
//! \begin{equation}
//! p^{(k)}_l = \sum^{k}_{a=0} (-1)^a P^{(k)}_{al}.
//! \end{equation}
//! ```
//!
//! Here ``P^{(k)}_{al}`` denotes the polynomial interpolation weights.
//!
//! For the retarded component, the extrapolation is done along the lines
//! of fixed relative time ``t - t'``. For the lesser component, it is done
//! along the stored lines of fixed right time. The hermitian conjugate
//! relations are used if the needed elements are not in the stored triangles.

use crate::complex::Complex;
use crate::types::{ContourGreen, TimeFunction};
use crate::weights::{Integrator, PolynomialInterpolationWeights};

use crate::types::adjoint_block_into;

/// Extrapolation weights `p_l` (`l = 0..k`), blueprint
/// `_extrapolation_weights`.
pub fn extrapolation_weights(integ: &Integrator) -> Vec<f64> {
    extrapolation_weights_order(integ.k)
}

/// Extrapolation weights for an explicit order `k` (0 = copy).
pub(crate) fn extrapolation_weights_order(k: i64) -> Vec<f64> {
    assert!(
        (0..=5).contains(&k),
        "extrapolation order 0 <= k <= 5, got {k}"
    );
    let piw = PolynomialInterpolationWeights::new(k);
    let n = (k + 1) as usize;
    let mut p = vec![0.0f64; n];
    for l in 0..n {
        let sign = 1.0 - 2.0 * (l % 2) as f64;
        for (m, pm) in p.iter_mut().enumerate() {
            *pm += sign * piw.at(l, m);
        }
    }
    p
}

/// Zeroth order (copy) extrapolation of the ret/lmix/less rows to
/// `n + 1`, blueprint `_extrapolate_copy!` (fixed relative time).
pub(crate) fn extrapolate_copy(n: i64, g: &mut ContourGreen) {
    assert!(n >= 0 && n < g.nt as i64, "extrapolate_copy: 0 <= n < nt");
    let n1 = n + 1;
    // lmix row
    let b = g.size1 * g.size2;
    let rowlen = (g.ntau + 1) * b;
    let off = n as usize * rowlen;
    g.lmix
        .raw_mut()
        .copy_within(off..off + rowlen, off + rowlen);
    // ret row: ret(n1, n1-j) = ret(n, n-j) for j = 0..n (blueprint
    // `j = 0 : n-1` with a 1-based n + 1)
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
    // less rows: less(j, n1) = less(j-1, n) for j = 1..=n1 (blueprint
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

/// Polynomial extrapolation of the ret/lmix/less rows of `g` to the
/// time step `n + 1` (0-based `n`), blueprint `extrapolate_timestep!`.
///
/// Extrapolate the leading row (`delti = 0`) of the moving Green's function
/// from the rows `1, ⋯, k + 1` with the polynomial extrapolation weights of
/// order `k` (`order = 0` copies the row 1). It is usually called after
/// `forward!`. The history rows are not changed. For
/// `deltj < tc - k` the extrapolation runs along the lines of fixed relative
/// time,
///
/// ```math
/// G(t, t - \\delta_j) = \\sum_{l=0}^{k} p_l\\ G(t - l - 1, t - l - 1 - \\delta_j),
/// ```
///
/// for `deltj ≥ tc - k` along the lines of fixed second time,
/// ``G(t, t - \\delta_j) = \\sum_l p_l\\ G(t - l - 1, t - \\delta_j)``.
/// `extrapolate_timestep_dispatch` for `herm_matrix_moving`. NESSi reads
/// the relative index `deltj - l - 1` without a check; here `tc ≥ 2k + 1` is
/// required so that every accessed block lies in the window.
pub fn extrapolate_timestep(n: i64, g: &mut ContourGreen, integ: &Integrator) {
    let k = integ.k;
    let n1 = n + 1;
    assert!(n >= k, "extrapolate_timestep: n {n} >= k {k}");
    assert!(n1 <= g.nt as i64, "extrapolate_timestep: n + 1 <= nt");
    let p = extrapolation_weights(integ);
    let (s1, s2) = (g.size1, g.size2);
    let b = s1 * s2;
    let mut gtmp = vec![Complex::ZERO; b];
    let mut adj = vec![Complex::ZERO; b];

    // lmix row: the blueprint's `for m = 1:ntau` becomes m = 0..=ntau
    for m in 0..=(g.ntau as i64) {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            let src = g.lmix.idx(n - l, m);
            for (t, s) in gtmp.iter_mut().zip(g.lmix.block(src)) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.lmix.idx(n1, m);
        g.lmix.block_mut(dst).copy_from_slice(&gtmp);
    }

    // ret row, fixed relative time: the blueprint's `j = 0 : n-1-k`
    // (relative, inclusive) with a 1-based n + 1 -> j = 0..=(n-k)
    for j in 0..=(n - k) {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            let src = g.ret.idx(n - l, n - l - j);
            for (t, s) in gtmp.iter_mut().zip(g.ret.block(src)) {
                *t += *s * p[l as usize];
            }
        }
        let dst = g.ret.idx(n1, n1 - j);
        g.ret.block_mut(dst).copy_from_slice(&gtmp);
    }

    // ret row, the first k + 1 columns: the blueprint's `j = 1 : k+1`
    // (absolute) -> column c = j - 1 = 0..=k
    for c in 0..=k {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            // Blueprint condition `n - l <= j` <=> `n - l <= c + 1 - 1 + 1`
            // i.e. `n_1based - l <= j` with n_1based = n + 1, j = c + 1
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

    // less, the first k + 1 rows: the blueprint's `j = 1 : k+1` -> r = 0..=k
    for r in 0..=k {
        gtmp.fill(Complex::ZERO);
        for l in 0..=k {
            // Blueprint `n - l < j` with n_1based = n + 1, j = r + 1
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

    // less, fixed relative time: the blueprint's `j = k+2 : n1`
    // (absolute rows) -> r = j - 1 = k+1..=n+1; src less(r-1-l, n-l)
    for r in (k + 1)..=n1 {
        gtmp.fill(Complex::ZERO);
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

/// Extrapolate a `TimeFunction` one step: `f[n+1] = sum_l p[l] * f[n-l]`
/// (blueprint `extrapolate_timestep!(n, f::Cf, k)`).
pub fn extrapolate_timestep_cf(n: i64, f: &mut TimeFunction, k: i64) {
    let p = extrapolation_weights_order(k);
    assert!(n >= k, "extrapolate_timestep_cf: n >= k");
    assert!(n < f.nt as i64, "extrapolate_timestep_cf: n <= nt-1");
    let nel = f.size1 * f.size2;
    let mut acc = vec![Complex::ZERO; nel];
    for (l, pl) in p.iter().enumerate().take(k as usize + 1) {
        let src = f.block(n - l as i64);
        for (a, s) in acc.iter_mut().zip(src) {
            *a += *s * *pl;
        }
    }
    f.block_mut(n + 1).copy_from_slice(&acc);
}

/// Extrapolation of the row `n + 1` with the explicit extrapolation
/// order `order`; order 0 copies along the time-translation diagonals
/// (blueprint `extrapolate_timestep!(n, G, order)`).
pub fn extrapolate_timestep_order(n: i64, g: &mut ContourGreen, order: i64) {
    assert!(
        (0..=5).contains(&order),
        "order must be in 0..=5, got {order}"
    );
    if order == 0 {
        extrapolate_copy(n, g);
    } else {
        extrapolate_timestep(n, g, &Integrator::new(order));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::env::var;
    use std::panic::catch_unwind;

    use crate::types::{Contour, Sign};

    use crate::types::MATSUBARA;

    use crate::math::interpolation;

    struct OrderGate {
        ok: bool,
        observed: f64,
        msg: String,
    }

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

        fn block(&mut self, nel: usize) -> Vec<Complex> {
            (0..nel)
                .map(|_| Complex::new(self.next_f64(), self.next_f64()))
                .collect()
        }
    }

    /// Smooth matrix test function (blueprint `ut_F`).
    fn ut_f(t: f64) -> [Complex; 4] {
        let e = (0.3 * t).exp();
        [
            Complex::new((-t).cos() * e, 0.0),
            Complex::new(0.0, -0.2 * (0.7 * t).sin() * e),
            Complex::new(0.0, 0.2 * (0.7 * t).sin() * e),
            Complex::new((0.8 * t).cos() * e, 0.0),
        ]
    }

    fn poly_eval(ms: &[Vec<Complex>], t: f64) -> Vec<Complex> {
        let nel = ms[0].len();
        let mut out = vec![Complex::ZERO; nel];
        for (r, m) in ms.iter().enumerate() {
            let w = Complex::new(t.powi(r as i32), 0.0);
            for (o, c) in out.iter_mut().zip(m) {
                *o += *c * w;
            }
        }
        out
    }

    /// Shared convergence-order gate (criteria identical to the
    /// blueprint):
    ///
    /// 1. at least 4 grids, strictly increasing grid sizes `ns`;
    /// 2. all errors are positive and finite;
    /// 3. at least 3 errors above the plateau threshold (round-off floor);
    /// 4. the least-squares slope of `-log(err)` versus `log(n)` over the
    ///    non-plateau points is `>= expected - order_tol`;
    /// 5. the error on the finest grid is `<= cap`.
    fn check_order_gate(
        ns: &[i64],
        errs: &[f64],
        expected: f64,
        cap: f64,
        plateau: f64,
        order_tol: f64,
    ) -> OrderGate {
        let fail = |observed: f64, msg: &str| OrderGate {
            ok: false,
            observed,
            msg: msg.to_string(),
        };
        if ns.len() != errs.len() {
            return fail(f64::NAN, "length(ns) != length(errs)");
        }
        if ns.len() < 4 {
            return fail(
                f64::NAN,
                &format!("at least 4 grids are required, got {}", ns.len()),
            );
        }
        if ns.windows(2).any(|w| w[1] <= w[0]) {
            return fail(f64::NAN, "grid sizes must be strictly increasing");
        }
        if errs.iter().any(|e| !e.is_finite() || *e <= 0.0) {
            return fail(f64::NAN, "errors must be positive and finite");
        }
        let sel: Vec<usize> = errs
            .iter()
            .enumerate()
            .filter(|(_, e)| **e > plateau)
            .map(|(i, _)| i)
            .collect();
        if sel.len() < 3 {
            return fail(
                f64::NAN,
                &format!(
                    "at least 3 non-plateau errors are required, got {}",
                    sel.len()
                ),
            );
        }
        let x: Vec<f64> = sel.iter().map(|&i| ns[i] as f64).map(f64::ln).collect();
        let y: Vec<f64> = sel.iter().map(|&i| errs[i].ln()).collect();
        let xm = x.iter().sum::<f64>() / x.len() as f64;
        let ym = y.iter().sum::<f64>() / y.len() as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            num += (x[i] - xm) * (y[i] - ym);
            den += (x[i] - xm) * (x[i] - xm);
        }
        let observed = -num / den;
        if observed < expected - order_tol {
            return fail(
                observed,
                &format!("observed order {observed} < expected {expected} - {order_tol}"),
            );
        }
        let last = errs[errs.len() - 1];
        if last > cap {
            return fail(
                observed,
                &format!("error on finest grid {last} > cap {cap}"),
            );
        }
        OrderGate {
            ok: true,
            observed,
            msg: String::new(),
        }
    }

    /// Assert form of [`check_order_gate`] for use in tests.
    fn assert_order_gate(ns: &[i64], errs: &[f64], expected: f64, cap: f64) {
        let g = check_order_gate(ns, errs, expected, cap, 1e-13, 0.3);
        assert!(
            g.ok,
            "order gate failed: {} (observed {})",
            g.msg, g.observed
        );
    }

    /// Whether the release-only tests should run.
    fn is_release() -> bool {
        match var("KB_TEST_PROFILE").ok().as_deref() {
            None | Some("quick") => false,
            Some("release") => true,
            Some(other) => panic!("KB_TEST_PROFILE must be `quick` or `release`, got `{other}`"),
        }
    }

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 3, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    /// Assert that `f` panics (the `std::panic` paths are wrapped once
    /// here instead of at every call site).
    fn assert_panics(f: impl FnOnce()) {
        assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
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
            g.set_ret(2, j, 0, 0, Complex::new(10.0 + j as f64, 0.0));
            g.set_less(j, 2, 0, 0, Complex::new(20.0 + j as f64, 0.0));
        }
        for m in 0..=3 {
            g.set_lmix(2, m, 0, 0, Complex::new(30.0 + m as f64, 0.0));
        }
        extrapolate_copy(2, &mut g);
        // ret row 3 shifts by one: ret(3, j+1) = ret(2, j), and the
        // first column copies ret(2, 0)
        for j in 0..=2 {
            assert_eq!(
                g.get_ret(3, j + 1, 0, 0),
                Complex::new(10.0 + j as f64, 0.0)
            );
        }
        assert_eq!(g.get_ret(3, 0, 0, 0), Complex::new(10.0, 0.0));
        // less rows shift: less(j, 3) = less(j-1, 2), first row copied
        assert_eq!(g.get_less(0, 3, 0, 0), Complex::new(20.0, 0.0));
        assert_eq!(g.get_less(1, 3, 0, 0), Complex::new(20.0, 0.0));
        assert_eq!(g.get_less(2, 3, 0, 0), Complex::new(21.0, 0.0));
        assert_eq!(g.get_less(3, 3, 0, 0), Complex::new(22.0, 0.0));
        for m in 0..=3 {
            assert_eq!(g.get_lmix(3, m, 0, 0), Complex::new(30.0 + m as f64, 0.0));
        }
    }

    #[test]
    fn linear_data_extrapolates_exactly() {
        // ret(i, j) = i - j: linear in both arguments, k >= 1 is exact
        let mut g = mk(6);
        for i in 0..=4 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::new((i - j) as f64, 0.0));
            }
            for jj in i..=4 {
                g.set_less(i, jj, 0, 0, Complex::new(100.0 + (jj - i) as f64, 0.0));
            }
            for m in 0..=3 {
                g.set_lmix(
                    i,
                    m,
                    0,
                    0,
                    Complex::new(1000.0 + 10.0 * i as f64 + m as f64, 0.0),
                );
            }
        }
        let integ = Integrator::new(2);
        extrapolate_timestep(4, &mut g, &integ);
        for j in 0..=5 {
            assert_eq!(
                g.get_ret(5, j, 0, 0),
                Complex::new((5 - j) as f64, 0.0),
                "ret(5,{j})"
            );
        }
        for i in 0..=5 {
            assert_eq!(
                g.get_less(i, 5, 0, 0),
                Complex::new(105.0 - i as f64, 0.0),
                "less({i},5)"
            );
        }
        for m in 0..=3 {
            assert_eq!(g.get_lmix(5, m, 0, 0), Complex::new(1050.0 + m as f64, 0.0));
        }
    }

    #[test]
    fn extrapolate_cf_polynomial_exactness() {
        let h = 0.1;
        for k in 0..=5i64 {
            for nel in [1usize, 4] {
                let dims = if nel == 1 { (1, 1) } else { (2, 2) };
                let mut rng = Rng(0x5EED + k as u64);
                let ms: Vec<Vec<Complex>> = (0..=k).map(|_| rng.block(nel)).collect();
                let ntime = (k + 4) as usize; // Rust nt
                let mut f = TimeFunction::new(ntime, dims.0, dims.1, Complex::ZERO);
                for i in 0..=ntime as i64 {
                    f.block_mut(i)
                        .copy_from_slice(&poly_eval(&ms, i as f64 * h));
                }
                f.block_mut(MATSUBARA)
                    .copy_from_slice(&vec![Complex::new(7.0, 0.0); nel]);
                let mut g = f.clone();
                let n = ntime as i64 - 1; // last filled row
                g.block_mut(n + 1).fill(Complex::ZERO);
                extrapolate_timestep_cf(n, &mut g, k);
                for (a, b) in g.block(n + 1).iter().zip(f.block(n + 1)) {
                    assert!((*a - *b).abs() < 1e-13, "k={k} nel={nel}");
                }
                assert_eq!(g.block(MATSUBARA), f.block(MATSUBARA));
                if k >= 1 {
                    g.block_mut(n + 1).fill(Complex::ZERO);
                    // Integrator overload (order from the Integrator)
                    let p = crate::math::extrapolation_weights(&Integrator::new(k));
                    let mut acc = vec![Complex::ZERO; nel];
                    for (l, pl) in p.iter().enumerate().take(k as usize + 1) {
                        for (a, s) in acc.iter_mut().zip(g.block(n - l as i64)) {
                            *a += *s * *pl;
                        }
                    }
                    g.block_mut(n + 1).copy_from_slice(&acc);
                    for (a, b) in g.block(n + 1).iter().zip(f.block(n + 1)) {
                        assert!((*a - *b).abs() < 1e-13);
                    }
                }
            }
        }
    }

    #[test]
    fn extrapolate_green_polynomial_exactness() {
        let (ntime, ntau, h) = (12i64, 7i64, 0.1);
        let c = Contour::new(ntime - 1, ntau - 1, 2, 2, (ntime - 1) as f64 * h, 2.0);
        let (nt, ntau_r) = (ntime - 1, ntau - 1);
        for k in 0..=5i64 {
            let mut rng = Rng(0xBEEF + k as u64);
            let kk = k as usize;
            let mut q = vec![vec![[Complex::ZERO; 4]; kk + 1]; kk + 1];
            let mut w = vec![vec![[Complex::ZERO; 4]; kk + 1]; kk + 1];
            for a in 0..=kk {
                for b in 0..=kk {
                    if a + b <= kk {
                        q[a][b] = {
                            let mut v = [Complex::ZERO; 4];
                            for z in v.iter_mut() {
                                *z = Complex::new(rng.next_f64(), rng.next_f64());
                            }
                            v
                        };
                        w[a][b] = {
                            let mut v = [Complex::ZERO; 4];
                            for z in v.iter_mut() {
                                *z = Complex::new(rng.next_f64(), rng.next_f64());
                            }
                            v
                        };
                    }
                }
            }
            let qf = |t: f64, s: f64| -> [Complex; 4] {
                let mut out = [Complex::ZERO; 4];
                for (a, qa) in q.iter().enumerate().take(kk + 1) {
                    for (b, qab) in qa.iter().enumerate().take(kk + 1) {
                        if a + b <= kk {
                            let wgt = t.powi(a as i32) * s.powi(b as i32);
                            for (o, cz) in out.iter_mut().zip(qab) {
                                *o += *cz * wgt;
                            }
                        }
                    }
                }
                out
            };
            let wf = |t: f64, s: f64| -> [Complex; 4] {
                let mut out = [Complex::ZERO; 4];
                for (a, wa) in w.iter().enumerate().take(kk + 1) {
                    for (b, wab) in wa.iter().enumerate().take(kk + 1) {
                        if a + b <= kk {
                            let wgt = t.powi(a as i32) * s.powi(b as i32);
                            for (o, cz) in out.iter_mut().zip(wab) {
                                *o += *cz * wgt;
                            }
                        }
                    }
                }
                out
            };
            let neg_t = |b: &[Complex]| -> [Complex; 4] {
                // -(block)^dagger for 2x2: conjugate + transpose
                [-b[0].conj(), -b[2].conj(), -b[1].conj(), -b[3].conj()]
            };
            let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            for i in 0..=nt {
                for j in 0..=i {
                    let r = qf(i as f64 * h, j as f64 * h);
                    let rs = qf(j as f64 * h, i as f64 * h);
                    let mut v = [Complex::ZERO; 4];
                    for (x, (y, z)) in v.iter_mut().zip(r.iter().zip(neg_t(&rs))) {
                        *x = *y + z;
                    }
                    g.ret_raw_block_mut(i, j).copy_from_slice(&v);
                }
                for j in i..=nt {
                    let l = wf(i as f64 * h, j as f64 * h);
                    let ls = wf(j as f64 * h, i as f64 * h);
                    let mut v = [Complex::ZERO; 4];
                    for (x, (y, z)) in v.iter_mut().zip(l.iter().zip(neg_t(&ls))) {
                        *x = *y + z;
                    }
                    g.less_raw_block_mut(i, j).copy_from_slice(&v);
                }
                for m in 0..=ntau_r {
                    g.lmix_raw_block_mut(i, m)
                        .copy_from_slice(&qf(i as f64 * h, 0.1 * (m + 1) as f64));
                }
            }
            for m in 0..=ntau_r {
                g.mat_raw_block_mut(m).copy_from_slice(&rng.block(4));
            }
            let gref = g.clone();
            let n = nt - 1; // last filled 0-based row (blueprint n = ntime-1)
            g.set_zero_at(n + 1);
            extrapolate_timestep(n, &mut g, &Integrator::new(k.max(1)));
            // order 0 has no Integrator; the copy case is covered by the
            // k >= 1 orders here plus the Cf testset
            let zero = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            let scale = gref.distance_at(&zero, n + 1);
            assert!(
                g.distance_at(&gref, n + 1) <= 1e-12 * scale.max(1.0),
                "k={k}"
            );
            assert!(g.mat.equal(&gref.mat));
            for t in 0..=n {
                assert_eq!(g.distance_at(&gref, t), 0.0);
            }
        }
    }

    #[test]
    fn extrapolate_preconditions() {
        let c = Contour::new(7, 4, 1, 1, 0.7, 1.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        assert_panics(|| extrapolate_timestep(2, &mut g, &Integrator::new(3)));
        assert_panics(|| extrapolate_timestep(7, &mut g, &Integrator::new(1)));
        let mut f = TimeFunction::new(7, 1, 1, Complex::ZERO);
        assert_panics(|| extrapolate_timestep_cf(1, &mut f, 2));
        assert_panics(|| extrapolate_timestep_cf(7, &mut f, 0));
        assert_panics(|| extrapolate_timestep_cf(4, &mut f, 6));
    }

    /// Release-only order gate.
    #[test]
    fn release_extrapolation_interpolation_order_gate() {
        if !is_release() {
            return;
        }
        let grids = &[(1i64, 12..16), (2, 6..10), (3, 5..9), (4, 4..8), (5, 3..7)];
        let extrap_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
            let h = 1.0 / n_grid as f64;
            let tplus = 1.3;
            let mut f = TimeFunction::new((k + 1) as usize, 2, 2, Complex::ZERO);
            for i in 0..=(k + 1) {
                let t = tplus - (k + 2 - (i + 1)) as f64 * h;
                f.block_mut(i).copy_from_slice(&ut_f(t));
            }
            extrapolate_timestep_cf(k, &mut f, ord);
            let mut e = 0.0;
            for (x, y) in f.block(k + 1).iter().zip(ut_f(tplus)) {
                e += (*x - y).abs();
            }
            e
        };
        let interp_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
            let h = 1.0 / n_grid as f64;
            let tplus = 1.3;
            let mut f = TimeFunction::new(k as usize, 2, 2, Complex::ZERO);
            for i in 0..=k {
                let t = tplus - (k + 1 - (i + 1)) as f64 * h;
                f.block_mut(i).copy_from_slice(&ut_f(t));
            }
            let x = interpolation(k, k as f64 - 0.5, &f, &Integrator::new(ord));
            let want = ut_f(tplus - 0.5 * h);
            let mut e = 0.0;
            for (q, wq) in want.iter().enumerate() {
                e += (x.get(q / 2, q % 2) - *wq).abs();
            }
            e
        };
        for (k, range) in grids {
            let ns: Vec<i64> = range.clone().map(|m| 1 << m).collect();
            let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, *k)).collect();
            assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
            let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, *k)).collect();
            assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
            let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, k - 1)).collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
            if *k >= 2 {
                let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, k - 1)).collect();
                assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
            }
        }
    }
}
