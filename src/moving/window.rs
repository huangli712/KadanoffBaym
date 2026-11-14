// Project : KadanoffBaym
// File    : src/moving/window.rs
// Purpose : window operations of the moving types (blueprint
//           julia/src/moving/window.jl)
//
// Only the boundary to the full contour (`init_from_g`) translates
// between the blueprint KB 1-based and the Rust 0-based time steps:
// the blueprint KB step n0 corresponds to n0 - 1 here, and the
// requirement `n0 - tc >= 1` (KB) becomes `n0 >= tc`.

use crate::c64::C64;
use crate::types::contour::Sign;
use crate::types::green::{ContourGreen, TimeFunction};
use crate::types::moving::{CfMoving, MovingGreen, MovingGreenTimeStep};
use crate::types::ness::NessHermMat;

/// Initialize the window from the full contour Green's function `g` at
/// the 0-based time step `n0`: the row `delti` is the time step
/// `n0 - delti` of `g`,
/// `Gm^{R}(delti, deltj) = G^{R}(t_n, t_{n-deltj})` and
/// `Gm^{<}(delti, deltj) = -Gcc^{<}(t_{n-deltj}, t_n)^dagger` with
/// `n = n0 - delti` and `deltj <= n` (the remaining blocks are zero).
/// `gcc` is the hermitian conjugate partner of `g` (`gcc = g` for
/// hermitian `g`). It requires `n0 >= tc` and `n0 <= g.nt`. Afterwards
/// `gm.tstp == n0` (blueprint `init_from_G!`).
pub fn init_from_g(gm: &mut MovingGreen, g: &ContourGreen, gcc: &ContourGreen, n0: i64) {
    let tc = gm.tc;
    let d = gm.ret.size1;
    assert!(
        g.size1 as usize == d
            && g.size2 as usize == d
            && gcc.size1 as usize == d
            && gcc.size2 as usize == d,
        "the matrix sizes do not match"
    );
    assert!(
        n0 >= tc && n0 <= g.nt as i64 && n0 <= gcc.nt as i64,
        "init_from_g requires n0 >= tc and n0 <= nt"
    );
    assert!(
        g.sign == gm.sign,
        "the statistics of g and gm differ"
    );
    gm.ret.raw_mut().fill(C64::ZERO);
    gm.less.raw_mut().fill(C64::ZERO);
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

/// `init_from_g` with the hermitian partner `gcc = g` (blueprint
/// `init_from_G!(Gm, G, n0)`).
pub fn init_from_g_herm(gm: &mut MovingGreen, g: &ContourGreen, n0: i64) {
    init_from_g(gm, g, g, n0);
}

/// Initialize the slots from the single-time function `cf`: the slot
/// `delti` is `cf[n0 - delti]`. It requires `n0 >= tc` (the Matsubara
/// value never enters the window). Afterwards `f.tstp == n0`
/// (blueprint `init_from_function!`).
pub fn init_from_function(f: &mut CfMoving, cf: &TimeFunction, n0: i64) {
    assert!(
        cf.size1 == f.size1 && cf.size2 == f.size2,
        "the matrix sizes do not match"
    );
    assert!(
        n0 - f.tc >= 0 && n0 <= cf.nt as i64,
        "init_from_function requires n0 >= tc and n0 <= nt"
    );
    for delti in 0..=f.tc {
        f.fslot_mut(delti)
            .copy_from_slice(cf.block(n0 - delti));
    }
    f.tstp = n0;
    f.initialized = true;
}

/// Move the window forward by one time step. The physical rows are
/// rotated, the new leading row (`delti = 0`) is set to zero, `tstp` is
/// incremented and every existing view becomes stale. No data are
/// copied (blueprint `forward!`).
pub fn forward(g: &mut MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
    g.ret.rotate_forward();
    g.less.rotate_forward();
    g.tstp += 1;
    g.generation += 1;
}

/// Return an owning copy (`MovingGreenTimeStep`) of the logical row
/// `delti` (blueprint `copy_timestep`).
pub fn copy_timestep(g: &MovingGreen, delti: i64) -> MovingGreenTimeStep {
    assert!(g.initialized, "the moving object is not initialized");
    assert!(0 <= delti && delti <= g.tc, "delti {delti} outside 0..={}", g.tc);
    let mut x = MovingGreenTimeStep::new(g.tc, g.ret.size1, g.tstp - delti, g.sign);
    x.ret.raw_mut().copy_from_slice(g.ret.mrow(delti));
    x.less.raw_mut().copy_from_slice(g.less.mrow(delti));
    x
}

/// The sign of a moving object (convenience for generic checks).
pub fn getsign_moving(g: &MovingGreen) -> Sign {
    g.sign
}

/// Initialize every row of the window with the time-translation
/// invariant steady state `gness` (blueprint `init_from_ness!`):
/// `G^{R/<}_m(delt_i, delt_j) = G^{R/<}_{ness}(delt_j h)` for
/// `delt_j <= min(tc, Nft/2 - 1)`, the remaining blocks are zero.
/// The relative NESS step 0 is the 0-based time step 0, the first
/// `forward` leads to the step 1.
pub fn init_from_ness(gm: &mut MovingGreen, gness: &NessHermMat) {
    use crate::types::ness::FftDomain;
    let d = gm.ret.size1;
    assert!(
        gness.ret.size1 == d && gness.ret.size2 == d,
        "ness2cntr: matrix size mismatch"
    );
    let tc = gm.tc;
    let dtmax = tc.min(gness.grid.nft / 2 - 1);
    gm.ret.raw_mut().fill(C64::ZERO);
    gm.less.raw_mut().fill(C64::ZERO);
    for delti in 0..=tc {
        for dt in 0..=dtmax {
            let src = gness.ret.block(FftDomain::Time, dt).to_vec();
            gm.ret.mblock_mut(delti, dt).copy_from_slice(&src);
            let src = gness.les.block(FftDomain::Time, dt).to_vec();
            gm.less.mblock_mut(delti, dt).copy_from_slice(&src);
        }
    }
    gm.tstp = 0;
    gm.initialized = true;
    gm.generation += 1;
}
