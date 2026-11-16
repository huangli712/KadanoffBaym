//
// Project : KadanoffBaym
// File    : src/basics/timestep.rs
// Purpose : single-timestep extraction family (green/pseudo)
//

use crate::complex::Complex;
use crate::types::{Contour, ContourGreen, GreenTimeStep, PseudoGreen, PseudoGreenTimeStep};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

#[allow(dead_code)]
/// Matsubara-branch dispatch: `MATSUBARA` addresses the Matsubara
/// component, `0..=nt` the real-time rows. This helper converts a tstp
/// to an optional tau index.
#[inline]
fn tau_of(tstp: i64) -> Option<usize> {
    if tstp == MATSUBARA {
        Some(0) // marker; callers use the mat component explicitly
    } else {
        None
    }
}

/// Extract an owned single-timestep object from a `ContourGreen`
/// (blueprint `G[tstp]`): the mat component is shared by reference
/// semantics (copied), ret/less hold the rows `0..=tstp`, lmix the row
/// `tstp`.
pub fn timestep_at(g: &ContourGreen, tstp: i64) -> GreenTimeStep {
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "timestep_at: tstp {tstp} out of range 0..={}",
        g.nt
    );
    let mut f = GreenTimeStep::new(
        // only the shape fields are consumed; nt is clamped to >= 1
        // because Contour rejects a degenerate single-point grid
        &Contour::new(
            tstp.max(1),
            g.ntau as i64,
            g.size1 as i64,
            g.size2 as i64,
            1.0,
            1.0,
        ),
        tstp,
        g.sign,
        Complex::ZERO,
    );
    f.mat.memcpy_from(&g.mat);
    let b = g.size1 * g.size2;
    // the ret/less triangles of the timestep are exactly the parent's
    // triangle PREFIX through tstp (contiguous in both layouts)
    let n_blocks = (tstp as usize + 1) * (tstp as usize + 2) / 2;
    f.ret.raw_mut()[..n_blocks * b].copy_from_slice(&g.ret.raw()[..n_blocks * b]);
    f.less.raw_mut()[..n_blocks * b].copy_from_slice(&g.less.raw()[..n_blocks * b]);
    for m in 0..=(g.ntau as i64) {
        // the timestep lmix component stores a SINGLE row: block index m
        f.lmix.raw_mut()[m as usize * b..(m as usize + 1) * b]
            .copy_from_slice(g.lmix_raw_block(tstp, m));
    }
    f
}

/// Write a single-timestep object back into a `ContourGreen`
/// (blueprint `G[tstp] = cfv`): mat, the ret/less triangle prefixes,
/// and the lmix row of `f` land at `f.tstp` of `g`.
pub fn set_timestep_at(g: &mut ContourGreen, f: &GreenTimeStep) {
    let tstp = f.tstp as i64;
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "set_timestep_at: tstp {tstp} out of range 0..={}",
        g.nt
    );
    assert!(
        f.ntau == g.ntau && f.size1 == g.size1 && f.size2 == g.size2,
        "set_timestep_at: incompatible shapes"
    );
    g.mat.memcpy_from(&f.mat);
    let b = g.size1 * g.size2;
    let n_blocks = (tstp as usize + 1) * (tstp as usize + 2) / 2;
    g.ret.raw_mut()[..n_blocks * b].copy_from_slice(&f.ret.raw()[..n_blocks * b]);
    g.less.raw_mut()[..n_blocks * b].copy_from_slice(&f.less.raw()[..n_blocks * b]);
    let rowlen = (g.ntau + 1) * b;
    let off = tstp as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].copy_from_slice(f.lmix.raw());
}

/// Extract the timestep `tstp` of a `PseudoGreen`. For the Matsubara
/// axis the returned object carries only the Matsubara component (its
/// `tstp` field is 0, matching the blueprint convention that the
/// Matsubara step is step 0).
pub fn pseudo_timestep(g: &PseudoGreen, tstp: i64) -> PseudoGreenTimeStep {
    if tstp == MATSUBARA {
        let mut x = GreenTimeStep {
            sign: g.sign,
            tstp: 0,
            ntau: g.ntau,
            size1: g.size1,
            size2: g.size2,
            mat: MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO),
            ret: RetComp::new(0, g.size1, g.size2, Complex::ZERO),
            lmix: LmixComp::new(0, g.ntau, g.size1, g.size2, Complex::ZERO),
            less: LessComp::new(0, g.size1, g.size2, Complex::ZERO),
        };
        x.mat.raw_mut().copy_from_slice(g.mat.raw());
        x
    } else {
        timestep_at(&g.to_contour(), tstp)
    }
}

/// Write the timestep `tstp` back into a `PseudoGreen`; for the
/// Matsubara axis only the Matsubara component is written.
pub fn pseudo_set_timestep(g: &mut PseudoGreen, tstp: i64, x: &PseudoGreenTimeStep) {
    if tstp == MATSUBARA {
        g.mat.raw_mut().copy_from_slice(x.mat.raw());
    } else {
        assert_eq!(
            x.tstp as i64, tstp,
            "the time step object carries tstp {}, not {tstp}",
            x.tstp
        );
        let mut gc = g.to_contour();
        set_timestep_at(&mut gc, x);
        g.copy_from_contour(&gc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msubara_marker() {
        assert_eq!(MATSUBARA, -1);
        assert!(tau_of(MATSUBARA).is_some());
        assert!(tau_of(3).is_none());
    }
}
