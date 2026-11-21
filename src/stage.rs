// Project : KadanoffBaym
// File    : src/stage.rs
// Purpose : the moving-window propagation stages: window
//           initialization, extrapolation (predictor), and advance

//!
//! *Remarks* : *Propagation Stages*
//!
//! The truncated (moving) solvers are driven one time step at a time:
//! initialize the window once ([`init_from_green`],
//! [`init_from_green_herm`], [`init_from_function`],
//! [`init_from_steady`]), then loop over
//! [`extrapolate_timestep_moving`] (predictor) -> solve
//! (`dyson_timestep_moving_ws`, `vie2_timestep_moving_ws`, ...) ->
//! [`forward`] (advance the window). This module owns the stages
//! around the solves; the solves live in `dyson`/`vie2`, the window
//! container algebra in `basics::moving`.

use crate::complex::Complex;
use crate::math::extrapolation_weights_order;
use crate::types::{
    ContourGreen, FftDomain, MovingGreen, MovingTimeFunction, SteadyGreen, TimeFunction,
};
use crate::weights::Integrator;

/// Initialize the window from the full contour Green's function `g` at
/// the 0-based time step `n0`: the row `delti` is the time step
/// `n0 - delti` of `g`,
/// `Gm^{R}(delti, deltj) = G^{R}(t_n, t_{n-deltj})` and
/// `Gm^{<}(delti, deltj) = -Gcc^{<}(t_{n-deltj}, t_n)^dagger` with
/// `n = n0 - delti`, for `deltj <= min(tc, n)`; the remaining blocks are
/// zero. `gcc` is the hermitian conjugate partner of `g` (`gcc = g` for
/// hermitian `g`). It requires `n0 >= tc` and `n0 <= g.nt` (and the same
/// for `gcc`). Afterwards `gm.tstp == n0`.
pub fn init_from_green(gm: &mut MovingGreen, g: &ContourGreen, gcc: &ContourGreen, n0: i64) {
    let tc = gm.tc;
    let d = gm.ret.size1;
    assert!(
        g.size1 == d && g.size2 == d && gcc.size1 == d && gcc.size2 == d,
        "the matrix sizes do not match"
    );
    assert!(
        n0 >= tc && n0 <= g.nt as i64 && n0 <= gcc.nt as i64,
        "init_from_green requires n0 >= tc and n0 <= nt"
    );
    assert!(g.sign == gm.sign, "the statistics of g and gm differ");
    gm.ret.raw_mut().fill(Complex::ZERO);
    gm.less.raw_mut().fill(Complex::ZERO);
    for delti in 0..=tc {
        let n = n0 - delti;
        for deltj in 0..=tc.min(n) {
            let m = n - deltj;
            gm.ret
                .mblock_mut(delti, deltj)
                .copy_from_slice(g.ret.raw_block_at(n, m));
            let s = gcc.less.raw_block_at(m, n);
            let b = gm.less.mblock_mut(delti, deltj);
            for a in 0..d {
                for bb in 0..d {
                    b[a * d + bb] = -s[bb * d + a].conj();
                }
            }
        }
    }
    gm.tstp = n0;
    gm.initialized = true;
    gm.generation += 1;
}

/// `init_from_green` with the hermitian partner `gcc = g`.
pub fn init_from_green_herm(gm: &mut MovingGreen, g: &ContourGreen, n0: i64) {
    init_from_green(gm, g, g, n0);
}

/// Initialize the slots from the single-time function `cf`: the slot
/// `delti` is `cf[n0 - delti]`. It requires `n0 >= tc` (the Matsubara
/// value never enters the window). Afterwards `f.tstp == n0`.
pub fn init_from_function(f: &mut MovingTimeFunction, cf: &TimeFunction, n0: i64) {
    assert!(
        cf.size1 == f.size1 && cf.size2 == f.size2,
        "the matrix sizes do not match"
    );
    assert!(
        n0 - f.tc >= 0 && n0 <= cf.nt as i64,
        "init_from_function requires n0 >= tc and n0 <= nt"
    );
    for delti in 0..=f.tc {
        f.fslot_mut(delti).copy_from_slice(cf.block(n0 - delti));
    }
    f.tstp = n0;
    f.initialized = true;
}

/// Initialize every row of the window with the time-translation
/// invariant steady state `gsteady`:
/// `G^{R/<}_m(delt_i, delt_j) = G^{R/<}_{steady}(delt_j h)` for
/// `delt_j <= min(tc, Nft/2 - 1)`, the remaining blocks are zero.
/// The relative steady step 0 is the 0-based time step 0, the first
/// `forward` leads to the step 1.
pub fn init_from_steady(gm: &mut MovingGreen, gsteady: &SteadyGreen) {
    let d = gm.ret.size1;
    assert!(
        gsteady.ret.size1 == d && gsteady.ret.size2 == d,
        "init_from_steady: matrix size mismatch"
    );
    let tc = gm.tc;
    let dtmax = tc.min(gsteady.grid.nft / 2 - 1);
    gm.ret.raw_mut().fill(Complex::ZERO);
    gm.less.raw_mut().fill(Complex::ZERO);
    for delti in 0..=tc {
        for dt in 0..=dtmax {
            let src = gsteady.ret.block(FftDomain::Time, dt).to_vec();
            gm.ret.mblock_mut(delti, dt).copy_from_slice(&src);
            let src = gsteady.les.block(FftDomain::Time, dt).to_vec();
            gm.less.mblock_mut(delti, dt).copy_from_slice(&src);
        }
    }
    gm.tstp = 0;
    gm.initialized = true;
    gm.generation += 1;
}

/// Move the window forward by one time step. The physical rows are
/// rotated, the new leading row (`delti = 0`) is set to zero, `tstp` is
/// incremented and every existing view becomes stale. No data are copied.
pub fn forward(g: &mut MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
    g.ret.rotate_forward();
    g.less.rotate_forward();
    g.tstp += 1;
    g.generation += 1;
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

    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::types::Sign;

    #[test]
    fn forward_requires_initialization() {
        let mut g = MovingGreen::new(3, 2, Sign::Fermi);
        assert!(catch_unwind(AssertUnwindSafe(|| forward(&mut g))).is_err());
    }

    #[test]
    fn extrapolate_order_zero_copies_row_one() {
        let mut g = MovingGreen::new(5, 1, Sign::Fermi);
        g.initialized = true;
        g.tstp = 5;
        for j in 0..=5 {
            g.ret.mblock_mut(1, j)[0] = Complex::new(j as f64, -1.0);
            g.less.mblock_mut(1, j)[0] = Complex::new(0.5, j as f64);
        }
        extrapolate_timestep_moving_order(&mut g, 0);
        for j in 0..=5 {
            assert_eq!(g.ret.mblock(1, j)[0], Complex::new(j as f64, -1.0));
            assert_eq!(g.ret.mblock(0, j)[0], Complex::new(j as f64, -1.0));
            assert_eq!(g.less.mblock(0, j)[0], Complex::new(0.5, j as f64));
        }
    }

    #[test]
    fn extrapolate_polynomial_rows() {
        // G(delti, deltj) = P(deltj) with a cubic P: the order-3
        // extrapolation must reproduce P(0) in every leading block
        let (k, tc) = (3i64, 9i64);
        let mut g = MovingGreen::new(tc, 1, Sign::Fermi);
        g.initialized = true;
        g.tstp = tc;
        let p = |x: f64| 1.5 - 0.7 * x + 0.25 * x * x - 0.05 * x * x * x;
        for i in 0..=tc {
            for j in 0..=tc {
                g.ret.mblock_mut(i, j)[0] = Complex::new(p(j as f64), 0.0);
                g.less.mblock_mut(i, j)[0] = Complex::new(0.0, p(j as f64));
            }
        }
        extrapolate_timestep_moving_order(&mut g, k);
        for j in 0..=tc {
            let got = g.ret.mblock(0, j)[0];
            assert!(
                (got - Complex::new(p(j as f64), 0.0)).abs() < 1e-12,
                "ret({j}): {got}"
            );
            let got = g.less.mblock(0, j)[0];
            assert!(
                (got - Complex::new(0.0, p(j as f64))).abs() < 1e-12,
                "less({j}): {got}"
            );
        }
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
