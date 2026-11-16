//
// Project : KadanoffBaym
// File    : src/basics/timestep.rs
// Purpose : single-timestep extraction family (green/pseudo)
//

use crate::complex::Complex;
use crate::types::{Contour, ContourGreen, GreenTimeStep};

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::DenseMat;
    use crate::types::Sign;

    use crate::equilibrium::from_ham;

    #[test]
    fn msubara_marker() {
        assert_eq!(MATSUBARA, -1);
        assert!(tau_of(MATSUBARA).is_some());
        assert!(tau_of(3).is_none());
    }

    /// Port of NESSi herm_matrix_setget_timestep.cpp:10-40.
    #[test]
    fn setget_timestep() {
        let (nt, ntau) = (50i64, 50i64);
        let (h, beta) = (0.01f64, 10.0f64);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, Complex::new(-0.4, 0.0));
        h0.set(1, 1, Complex::new(0.6, 0.0));
        h0.set(0, 1, Complex::new(0.0, 0.1));
        h0.set(1, 0, Complex::new(0.0, -0.1));
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g1, 0.0, &h0, beta, h);
        let mut err = 0.0;
        for tstp in -1..=nt {
            let a = timestep_at(&g1, if tstp == MATSUBARA { 0 } else { tstp });
            set_timestep_at(&mut g2, &a);
            err += g1.distance_at(&g2, tstp);
        }
        assert!(err < 1e-6, "err {err}");
    }
}
