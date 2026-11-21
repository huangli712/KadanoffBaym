// Project : KadanoffBaym
// File    : src/math/extrap.rs
// Purpose : polynomial extrapolation of a time step
//
// `p_l = sum_m (1 - 2*(l % 2)) * PIW[l, m]`; the extrapolated row
// `n + 1` of ret/lmix/less combines the previous `k + 1` rows with the
// hermiticity relations for the innermost columns/rows.
//
// Index translation: the argument `n` is the 0-based index of the last
// FILLED row; everything is extrapolated to `n + 1`. The original
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
use crate::types::{ContourGreen, MovingGreen, TimeFunction};
use crate::weights::{Integrator, PolynomialInterpolationWeights};

use crate::types::adjoint_block_into;

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

/// Extrapolation weights `p_l` (`l = 0..k`)
pub fn extrapolation_weights(integ: &Integrator) -> Vec<f64> {
    extrapolation_weights_order(integ.k)
}

/// Zeroth order (copy) extrapolation of the ret/lmix/less rows to
/// `n + 1` (fixed relative time).
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
    // ret row: ret(n1, n1-j) = ret(n, n-j) for j = 0..n
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
    // less rows: less(j, n1) = less(j-1, n) for j = 1..=n1 , plus the first row less(0, n1) = less(0, n)
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
/// time step `n + 1` (0-based `n`).
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
/// The relative index `deltj - l - 1` is read without a check; here
/// `tc ≥ 2k + 1` is required so that every accessed block lies in the
/// window.
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

    // lmix row: `for m = 1:ntau` becomes m = 0..=ntau
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

    // ret row, fixed relative time: `j = 0 : n-1-k`
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

    // ret row, the first k + 1 columns: `j = 1 : k+1`
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

    // less, the first k + 1 rows: `j = 1 : k+1` -> r = 0..=k
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

    // less, fixed relative time: `j = k+2 : n1`
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

/// Extrapolate a `TimeFunction` one step: `f[n+1] = sum_l p[l] * f[n-l]`.
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
/// order `order`; order 0 copies along the time-translation diagonals.
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

/// Reusable scratch of [`extrapolate_timestep_moving`]: the history
/// snapshot, the block accumulator, and the row-copy buffer, sized once
/// for `(k, tc, d)` and reused across the time-step loop.
///
/// The sizes are fixed at construction: the `_ws` functions panic when
/// the workspace was built for a different order `k`, window width `tc`,
/// or block size `d` than the call passes in. Build one workspace per
/// solver order and window geometry.
pub struct MovingStageWorkspace {
    k: i64,
    tc: i64,
    d: usize,
    hist: Vec<Complex>,
    gtmp: Vec<Complex>,
    buf: Vec<Complex>,
}

impl MovingStageWorkspace {
    /// Allocate the scratch for extrapolation order `k`, window width
    /// `tc` (the leading index of `MovingGreen`), and `d × d` blocks.
    pub fn new(k: i64, tc: i64, d: usize) -> MovingStageWorkspace {
        let tcl = (tc + 1) as usize;
        let b = d * d;
        MovingStageWorkspace {
            k,
            tc,
            d,
            hist: vec![Complex::ZERO; (k + 1) as usize * tcl * b],
            gtmp: vec![Complex::ZERO; b],
            buf: vec![Complex::ZERO; tcl * b],
        }
    }

    /// Extrapolation order the scratch was sized for.
    pub fn order(&self) -> i64 {
        self.k
    }

    /// Window width (`MovingGreen::tc`) the scratch was sized for.
    pub fn tc(&self) -> i64 {
        self.tc
    }

    /// Block dimension the scratch was sized for.
    pub fn d(&self) -> usize {
        self.d
    }
}

/// Extrapolate the leading row of the window from the rows 1, .., k+1.
/// Requires `tc >= 2k + 1`.
///
/// Convenience wrapper: allocates a fresh workspace on every call. In
/// time-step loops build a [`MovingStageWorkspace`] once and call
/// [`extrapolate_timestep_moving_ws`] instead.
pub fn extrapolate_timestep_moving(g: &mut MovingGreen, integ: &Integrator) {
    let mut sws = MovingStageWorkspace::new(integ.k, g.tc, g.ret.size1);
    extrapolate_timestep_moving_ws(g, integ, &mut sws);
}

/// Workspace-reusing variant of [`extrapolate_timestep_moving`].
///
/// `sws` must have been built for the same `k` as `integ`, the same `tc`
/// as `g`, and the same block size as `g`; a mismatch panics.
pub fn extrapolate_timestep_moving_ws(
    g: &mut MovingGreen,
    integ: &Integrator,
    sws: &mut MovingStageWorkspace,
) {
    assert!(g.initialized, "the moving object is not initialized");
    let k = integ.k;
    let tc = g.tc;
    let d = g.ret.size1;
    assert!(
        sws.order() == k && sws.tc() == tc && sws.d() == d,
        "the stage workspace was built for (k, tc, d) = ({}, {}, {}), \
         the call uses ({k}, {tc}, {d})",
        sws.order(),
        sws.tc(),
        sws.d()
    );
    assert!(
        tc > 2 * k,
        "extrapolation of order {k} requires tc >= {}, got {tc}",
        2 * k + 1
    );
    let p = extrapolation_weights_order(k);
    let b = d * d;
    let tcl = (tc + 1) as usize;
    // the history snapshot and the block accumulator are reused by
    // both components; the vectors live in the workspace
    let hist = &mut sws.hist;
    let gtmp = &mut sws.gtmp;
    for comp in 0..2 {
        // snapshot the rows 1, .., k+1 (the leading row is written
        // while the history rows are read)
        for l in 0..=(k as usize) {
            let src = if comp == 0 {
                g.ret.mrow(l as i64 + 1)
            } else {
                g.less.mrow(l as i64 + 1)
            };
            hist[l * tcl * b..(l + 1) * tcl * b].copy_from_slice(src);
        }
        let lead = if comp == 0 {
            g.ret.mrow_mut(0)
        } else {
            g.less.mrow_mut(0)
        };
        for j in 0..=tc {
            // jj = j < tc - k ? j : j - l - 1 (per l)
            gtmp.iter_mut().for_each(|z| *z = Complex::ZERO);
            for l in 0..=(k as usize) {
                let j_l = if j < tc - k {
                    j as usize
                } else {
                    (j - l as i64 - 1) as usize
                };
                let w = Complex::new(p[l], 0.0);
                let blk = &hist[l * tcl * b + j_l * b..l * tcl * b + (j_l + 1) * b];
                for q in 0..b {
                    gtmp[q] += w * blk[q];
                }
            }
            lead[j as usize * b..(j as usize + 1) * b].copy_from_slice(gtmp);
        }
    }
}

/// `extrapolate_timestep_moving` with the order `order`; `order == 0`
/// copies the row 1 into the leading row.
///
/// Convenience wrapper: allocates a fresh workspace on every call. In
/// time-step loops use [`extrapolate_timestep_moving_order_ws`].
pub fn extrapolate_timestep_moving_order(g: &mut MovingGreen, order: i64) {
    let mut sws = MovingStageWorkspace::new(order.max(1), g.tc, g.ret.size1);
    extrapolate_timestep_moving_order_ws(g, order, &mut sws);
}

/// Workspace-reusing variant of [`extrapolate_timestep_moving_order`].
///
/// `sws` must have been built for the same `tc` and block size as `g`,
/// and — for `order >= 1` — the same order; a mismatch panics. The order
/// `0` row copy only uses the row buffer, so the workspace order does
/// not matter in that case.
pub fn extrapolate_timestep_moving_order_ws(
    g: &mut MovingGreen,
    order: i64,
    sws: &mut MovingStageWorkspace,
) {
    assert!(
        (0..=5).contains(&order),
        "order must be in 0:5, got {order}"
    );
    if order == 0 {
        assert!(g.initialized, "the moving object is not initialized");
        assert!(g.tc >= 1, "extrapolation requires tc >= 1");
        assert!(
            sws.tc() == g.tc && sws.d() == g.ret.size1,
            "the stage workspace was built for (tc, d) = ({}, {}), \
             the call uses ({}, {})",
            sws.tc(),
            sws.d(),
            g.tc,
            g.ret.size1
        );
        let b = g.ret.size1 * g.ret.size1;
        let n = (g.tc + 1) as usize * b;
        sws.buf[..n].copy_from_slice(g.ret.mrow(1));
        g.ret.mrow_mut(0).copy_from_slice(&sws.buf[..n]);
        sws.buf[..n].copy_from_slice(g.less.mrow(1));
        g.less.mrow_mut(0).copy_from_slice(&sws.buf[..n]);
        return;
    }
    let integ = Integrator::new(order);
    extrapolate_timestep_moving_ws(g, &integ, sws);
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    use crate::types::{Contour, Sign};

    use crate::types::MATSUBARA;

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

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 3, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    /// Assert that `f` panics (the `std::panic` paths are wrapped once
    /// here instead of at every call site).
    fn assert_panics(f: impl FnOnce()) {
        assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
    }
    use crate::types::MovingGreen;
    use crate::weights::Integrator;

    #[test]
    fn weights_extrapolate_linear_functions() {
        // For k = 1 the weights reproduce f(n+1): p[0] pairs with f(n),
        // p[1] with f(n-1)
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
                    let p = extrapolation_weights(&Integrator::new(k));
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
            let n = nt - 1; // last filled 0-based row
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
    /// Deterministic fill of a `2`-component window with pseudo-random
    /// ret/less blocks, so that two copies stay bit-identical.
    fn seeded(tc: i64, d: usize) -> MovingGreen {
        let mut g = MovingGreen::new(tc, d, Sign::Fermi);
        g.initialized = true;
        g.tstp = tc;
        let mut n: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            n = n
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((n >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        for i in 0..=tc {
            for j in 0..=tc {
                let ret = g.ret.mblock_mut(i, j);
                let less = g.less.mblock_mut(i, j);
                for q in 0..d * d {
                    let (re, im) = (next(), next());
                    ret[q] = Complex::new(re, im);
                    less[q] = Complex::new(-im, re);
                }
            }
        }
        g
    }

    #[test]
    fn workspace_reuse_matches_fresh_workspace() {
        let (k, tc, d) = (3i64, 9i64, 2usize);
        let integ = Integrator::new(k);
        let base = seeded(tc, d);

        // order-3 extrapolation: fresh workspace vs reused workspace
        let mut fresh = base.clone();
        extrapolate_timestep_moving(&mut fresh, &integ);
        let mut reused = base.clone();
        let mut sws = MovingStageWorkspace::new(k, tc, d);
        extrapolate_timestep_moving_ws(&mut reused, &integ, &mut sws);
        assert_eq!(fresh.ret.raw(), reused.ret.raw());
        assert_eq!(fresh.less.raw(), reused.less.raw());

        // the same workspace serves another call
        let mut again = base.clone();
        extrapolate_timestep_moving_ws(&mut again, &integ, &mut sws);
        assert_eq!(fresh.ret.raw(), again.ret.raw());
        assert_eq!(fresh.less.raw(), again.less.raw());

        // order-0 row copy: convenience vs reused workspace
        let mut zero_fresh = base.clone();
        extrapolate_timestep_moving_order(&mut zero_fresh, 0);
        let mut zero_ws = base.clone();
        let mut sws0 = MovingStageWorkspace::new(1, tc, d);
        extrapolate_timestep_moving_order_ws(&mut zero_ws, 0, &mut sws0);
        assert_eq!(zero_fresh.ret.raw(), zero_ws.ret.raw());
        assert_eq!(zero_fresh.less.raw(), zero_ws.less.raw());

        // order-aware wrapper: convenience vs reused workspace
        let mut ord_fresh = base.clone();
        extrapolate_timestep_moving_order(&mut ord_fresh, k);
        let mut ord_ws = base.clone();
        extrapolate_timestep_moving_order_ws(&mut ord_ws, k, &mut sws);
        assert_eq!(ord_fresh.ret.raw(), ord_ws.ret.raw());
        assert_eq!(ord_fresh.less.raw(), ord_ws.less.raw());
    }

    #[test]
    fn workspace_mismatch_panics() {
        let (tc, d) = (9i64, 2usize);
        let integ = Integrator::new(3);

        let mut g = seeded(tc, d);
        let mut sws = MovingStageWorkspace::new(2, tc, d);
        assert!(
            catch_unwind(AssertUnwindSafe(|| extrapolate_timestep_moving_ws(
                &mut g, &integ, &mut sws
            )))
            .is_err()
        );

        let mut g = seeded(tc, d);
        let mut sws = MovingStageWorkspace::new(3, tc, 1);
        assert!(
            catch_unwind(AssertUnwindSafe(|| extrapolate_timestep_moving_ws(
                &mut g, &integ, &mut sws
            )))
            .is_err()
        );

        let mut g = seeded(tc, d);
        let mut sws = MovingStageWorkspace::new(3, tc - 1, d);
        assert!(
            catch_unwind(AssertUnwindSafe(|| extrapolate_timestep_moving_ws(
                &mut g, &integ, &mut sws
            )))
            .is_err()
        );

        // order 0 ignores the workspace order but not tc/d
        let mut g = seeded(tc, d);
        let mut sws = MovingStageWorkspace::new(1, tc, d);
        extrapolate_timestep_moving_order_ws(&mut g, 0, &mut sws);
        let mut g = seeded(tc, d);
        let mut sws = MovingStageWorkspace::new(1, tc, 1);
        assert!(
            catch_unwind(AssertUnwindSafe(|| extrapolate_timestep_moving_order_ws(
                &mut g, 0, &mut sws
            )))
            .is_err()
        );
    }
}
