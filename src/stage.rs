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

/// Extrapolate the leading row of the window from the rows 1, .., k+1.
/// Requires `tc >= 2k + 1`.
pub fn extrapolate_timestep_moving(g: &mut MovingGreen, integ: &Integrator) {
    assert!(g.initialized, "the moving object is not initialized");
    let k = integ.k;
    let tc = g.tc;
    assert!(
        tc > 2 * k,
        "extrapolation of order {k} requires tc >= {}, got {tc}",
        2 * k + 1
    );
    let p = extrapolation_weights_order(k);
    let d = g.ret.size1;
    let b = d * d;
    let tcl = (tc + 1) as usize;
    for comp in 0..2 {
        // snapshot the rows 1, .., k+1 (the leading row is written
        // while the history rows are read)
        let mut hist: Vec<Complex> = vec![Complex::ZERO; (k + 1) as usize * tcl * b];
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
            let mut gtmp = vec![Complex::ZERO; b];
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
            lead[j as usize * b..(j as usize + 1) * b].copy_from_slice(&gtmp);
        }
    }
}

/// `extrapolate_timestep_moving` with the order `order`; `order == 0`
/// copies the row 1 into the leading row.
pub fn extrapolate_timestep_moving_order(g: &mut MovingGreen, order: i64) {
    assert!(
        (0..=5).contains(&order),
        "order must be in 0:5, got {order}"
    );
    if order == 0 {
        assert!(g.initialized, "the moving object is not initialized");
        assert!(g.tc >= 1, "extrapolation requires tc >= 1");
        let r1 = g.ret.mrow(1).to_vec();
        let l1 = g.less.mrow(1).to_vec();
        g.ret.mrow_mut(0).copy_from_slice(&r1);
        g.less.mrow_mut(0).copy_from_slice(&l1);
        return;
    }
    let integ = Integrator::new(order);
    extrapolate_timestep_moving(g, &integ);
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
}
