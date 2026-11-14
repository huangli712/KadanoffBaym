// Project : KadanoffBaym
// File    : src/types/pseudo.rs
// Purpose : pseudo-particle Green's functions (blueprint
//           julia/src/types/pseudo.jl)
//
// The pseudo-particle Green's function has the same storage layout as
// the standard one (`mat`, `ret`, `lmix`, `less`). However, the real
// time triangle `ret` stores the greater component G^{>}(t, t') (with
// t >= t'), instead of G^{R} = G^{>} - G^{<}. There is no advanced
// component. `PseudoGreen` is separated from `ContourGreen`, so that a
// pseudo-particle object can not be passed to the standard Matsubara
// or left-mixing algorithms by mistake.
//
// The Julia blueprint dispatches the *standard* real-time kernels
// (`_dyson_start_ret!`, `_vie2_start_ret!`, ...) on the abstract
// function type, so they accept both. Rust has no such dispatch: the
// pseudo solvers transfer the object into a scratch `ContourGreen`
// once per public driver call (the pseudo-ness is purely semantic,
// the storage is identical), run all the kernels, and copy back.

use crate::complex::Complex;
use crate::types::contour::Contour;
use crate::types::green::{ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp};
use crate::types::sign::Sign;

/// Pseudo-particle contour-ordered Green's function `F^{ps}`. The `ret`
/// component stores the **greater** Green's function.
#[derive(Clone)]
pub struct PseudoGreen {
    pub sign: Sign,
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    pub mat: MatComp,
    pub ret: RetComp,
    pub lmix: LmixComp,
    pub less: LessComp,
}

impl PseudoGreen {
    /// Standard constructor (blueprint `F^{ps}(C, v, sign)`).
    pub fn new(c: &Contour, sign: Sign, v: Complex) -> PseudoGreen {
        assert!(c.size1 == c.size2, "the Green's function must be square");
        let (nt, ntau) = (c.nt as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        PseudoGreen {
            sign,
            nt,
            ntau,
            size1: s1,
            size2: s2,
            mat: MatComp::new(ntau, s1, s2, v),
            ret: RetComp::new(nt, s1, s2, v),
            lmix: LmixComp::new(nt, ntau, s1, s2, v),
            less: LessComp::new(nt, s1, s2, v),
        }
    }

    /// Transfer into a scratch `ContourGreen` (one clone per driver
    /// call, see the module remarks).
    pub fn to_contour(&self) -> ContourGreen {
        ContourGreen {
            sign: self.sign,
            nt: self.nt,
            ntau: self.ntau,
            size1: self.size1,
            size2: self.size2,
            mat: self.mat.clone(),
            ret: self.ret.clone(),
            lmix: self.lmix.clone(),
            less: self.less.clone(),
        }
    }

    /// Build from a scratch `ContourGreen` (the inverse transfer).
    pub fn from_contour(g: &ContourGreen) -> PseudoGreen {
        PseudoGreen {
            sign: g.sign,
            nt: g.nt,
            ntau: g.ntau,
            size1: g.size1,
            size2: g.size2,
            mat: g.mat.clone(),
            ret: g.ret.clone(),
            lmix: g.lmix.clone(),
            less: g.less.clone(),
        }
    }

    /// Copy the components back from a scratch `ContourGreen`.
    pub fn copy_from_contour(&mut self, g: &ContourGreen) {
        self.mat = g.mat.clone();
        self.ret = g.ret.clone();
        self.lmix = g.lmix.clone();
        self.less = g.less.clone();
    }

    /// Zero the components at time step `tstp` (blueprint `zeros!`;
    /// `MATSUBARA` zeroes the Matsubara component only).
    pub fn zeros_step(&mut self, tstp: i64) {
        if tstp == crate::types::layout::MATSUBARA {
            self.mat.raw_mut().fill(Complex::ZERO);
        } else {
            let n = tstp;
            for j in 0..=n {
                self.ret.raw_block_at_mut(n, j).fill(Complex::ZERO);
                self.less.raw_block_at_mut(j, n).fill(Complex::ZERO);
            }
            for m in 0..=self.ntau as i64 {
                self.lmix.raw_block_at_mut(n, m).fill(Complex::ZERO);
            }
        }
    }
}

/// Pseudo-particle Green's function at a given time step `f^{ps}`
/// (blueprint reuses the layout of the standard timestep object; the
/// `ret` row stores the greater component).
pub(crate) type PseudoGreenTimeStep = GreenTimeStep;

/// Extract the timestep `tstp` of a `PseudoGreen`. For the Matsubara
/// axis the returned object carries only the Matsubara component (its
/// `tstp` field is 0, matching the blueprint convention that the
/// Matsubara step is step 0).
pub fn pseudo_timestep(g: &PseudoGreen, tstp: i64) -> PseudoGreenTimeStep {
    if tstp == crate::types::layout::MATSUBARA {
        let mut x = GreenTimeStep {
            sign: g.sign,
            tstp: 0,
            ntau: g.ntau,
            size1: g.size1,
            size2: g.size2,
            mat: crate::types::green::MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO),
            ret: crate::types::green::RetComp::new(0, g.size1, g.size2, Complex::ZERO),
            lmix: crate::types::green::LmixComp::new(0, g.ntau, g.size1, g.size2, Complex::ZERO),
            less: crate::types::green::LessComp::new(0, g.size1, g.size2, Complex::ZERO),
        };
        x.mat.raw_mut().copy_from_slice(g.mat.raw());
        x
    } else {
        crate::basics::indexing::timestep_at(&g.to_contour(), tstp)
    }
}

/// Write the timestep `tstp` back into a `PseudoGreen`; for the
/// Matsubara axis only the Matsubara component is written.
pub fn pseudo_set_timestep(g: &mut PseudoGreen, tstp: i64, x: &PseudoGreenTimeStep) {
    if tstp == crate::types::layout::MATSUBARA {
        g.mat.raw_mut().copy_from_slice(x.mat.raw());
    } else {
        assert_eq!(
            x.tstp as i64, tstp,
            "the time step object carries tstp {}, not {tstp}",
            x.tstp
        );
        let mut gc = g.to_contour();
        crate::basics::indexing::set_timestep_at(&mut gc, x);
        g.copy_from_contour(&gc);
    }
}


// ------------------------------------------------------------------
// pseudo-particle ops layer (blueprint traits/properties dispatches)
// ------------------------------------------------------------------

/// Compatibility check (blueprint `iscompatible`).
pub fn pseudo_is_compatible(g: &PseudoGreen, h: &PseudoGreen) -> bool {
    g.sign == h.sign
        && g.nt == h.nt
        && g.ntau == h.ntau
        && g.size1 == h.size1
        && g.size2 == h.size2
}

/// Per-time-step distance (blueprint `distance(G, H, tstp)`).
pub fn pseudo_distance_at(g1: &PseudoGreen, g2: &PseudoGreen, tstp: i64) -> f64 {
    g1.to_contour().distance_at(&g2.to_contour(), tstp)
}

/// Copy the components at time step `tstp` (blueprint `memcpy!(A, Y, t)`).
pub fn pseudo_memcpy_step(dst: &mut PseudoGreen, src: &PseudoGreen, tstp: i64) {
    if tstp == crate::types::layout::MATSUBARA {
        dst.mat.raw_mut().copy_from_slice(src.mat.raw());
    } else {
        assert!((0..=dst.nt as i64).contains(&tstp));
        for j in 0..=tstp {
            let a = src.ret.raw_block_at(tstp, j);
            dst.ret.raw_block_at_mut(tstp, j).copy_from_slice(a);
        }
        for i in 0..=tstp {
            let a = src.less.raw_block_at(i, tstp);
            dst.less.raw_block_at_mut(i, tstp).copy_from_slice(a);
        }
        for m in 0..=dst.ntau as i64 {
            let a = src.lmix.raw_block_at(tstp, m);
            dst.lmix.raw_block_at_mut(tstp, m).copy_from_slice(a);
        }
    }
}

/// Fill the components at time step `tstp` with `v` (blueprint
/// `memset!(Z, t, v)`).
pub fn pseudo_memset_step(g: &mut PseudoGreen, tstp: i64, v: Complex) {
    if tstp == crate::types::layout::MATSUBARA {
        g.mat.raw_mut().fill(v);
    } else {
        assert!((0..=g.nt as i64).contains(&tstp));
        for j in 0..=tstp {
            g.ret.raw_block_at_mut(tstp, j).fill(v);
        }
        for i in 0..=tstp {
            g.less.raw_block_at_mut(i, tstp).fill(v);
        }
        for m in 0..=g.ntau as i64 {
            g.lmix.raw_block_at_mut(tstp, m).fill(v);
        }
    }
}

/// Zero all the components (blueprint `zeros!(Z)`).
pub fn pseudo_zeros_all(g: &mut PseudoGreen) {
    g.mat.raw_mut().fill(Complex::ZERO);
    g.ret.raw_mut().fill(Complex::ZERO);
    g.lmix.raw_mut().fill(Complex::ZERO);
    g.less.raw_mut().fill(Complex::ZERO);
}

/// Elementwise `dst += alpha * src` over all the components
/// (blueprint `incr!(S, B, alpha)`).
pub fn pseudo_incr(dst: &mut PseudoGreen, src: &PseudoGreen, alpha: Complex) {
    for (d, s) in [
        (dst.mat.raw_mut(), src.mat.raw()),
        (dst.ret.raw_mut(), src.ret.raw()),
        (dst.lmix.raw_mut(), src.lmix.raw()),
        (dst.less.raw_mut(), src.less.raw()),
    ] {
        for q in 0..d.len() {
            d[q] += alpha * s[q];
        }
    }
}

/// Scale the components at time step `tstp` by `alpha` (blueprint
/// `smul!(S, t, a)`).
pub fn pseudo_smul_step(g: &mut PseudoGreen, tstp: i64, alpha: Complex) {
    if tstp == crate::types::layout::MATSUBARA {
        for z in g.mat.raw_mut().iter_mut() {
            *z = *z * alpha;
        }
    } else {
        assert!((0..=g.nt as i64).contains(&tstp));
        for j in 0..=tstp {
            for z in g.ret.raw_block_at_mut(tstp, j).iter_mut() {
                *z = *z * alpha;
            }
            for z in g.less.raw_block_at_mut(j, tstp).iter_mut() {
                *z = *z * alpha;
            }
        }
        for m in 0..=g.ntau as i64 {
            for z in g.lmix.raw_block_at_mut(tstp, m).iter_mut() {
                *z = *z * alpha;
            }
        }
    }
}

/// Elementwise scaling of all the components (blueprint `smul!(S, a)`).
pub fn pseudo_smul(g: &mut PseudoGreen, alpha: Complex) {
    for d in [
        g.mat.raw_mut(),
        g.ret.raw_mut(),
        g.lmix.raw_mut(),
        g.less.raw_mut(),
    ] {
        for z in d.iter_mut() {
            *z = *z * alpha;
        }
    }
}

/// Left multiplication by a contour function at time step `tstp`
/// (blueprint `smul!(f, L, t)`): `f(t)` from the left, the lesser
/// blocks are weighted by their first time argument, the Matsubara
/// component by `f_0`.
pub fn pseudo_lmul_cf(g: &mut PseudoGreen, f: &crate::types::green::TimeFunction, tstp: i64) {
    let f0 = f.block(crate::types::layout::MATSUBARA);
    for m in 0..=g.ntau as i64 {
        let fm = DenseMat::from_vec(g.size1, g.size2, f0.to_vec());
        let blk = DenseMat::from_vec(g.size1, g.size2, g.mat.block(m as usize).to_vec());
        let p = fm.mat_mul(&blk);
        g.mat.block_mut(m as usize).copy_from_slice(p.raw());
    }
    if tstp == crate::types::layout::MATSUBARA {
        return;
    }
    let ft = DenseMat::from_vec(g.size1, g.size2, f.block(tstp).to_vec());
    for j in 0..=tstp {
        let blk = DenseMat::from_vec(g.size1, g.size2, g.ret.raw_block_at(tstp, j).to_vec());
        let p = ft.mat_mul(&blk);
        g.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
        let fj = DenseMat::from_vec(g.size1, g.size2, f.block(j).to_vec());
        let bl = DenseMat::from_vec(g.size1, g.size2, g.less.raw_block_at(j, tstp).to_vec());
        let pl = fj.mat_mul(&bl);
        g.less.raw_block_at_mut(j, tstp).copy_from_slice(pl.raw());
    }
    for m in 0..=g.ntau as i64 {
        let blk = DenseMat::from_vec(g.size1, g.size2, g.lmix.raw_block_at(tstp, m).to_vec());
        let p = ft.mat_mul(&blk);
        g.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
    }
}

/// Right multiplication by a contour function at time step `tstp`
/// (blueprint `smul!(R, f, t)`): `f` from the right, the ret blocks
/// are weighted by their column time, the left-mixing component by
/// `f_0`.
pub fn pseudo_rmul_cf(g: &mut PseudoGreen, f: &crate::types::green::TimeFunction, tstp: i64) {
    let f0 = DenseMat::from_vec(g.size1, g.size2, f.block(crate::types::layout::MATSUBARA).to_vec());
    for m in 0..=g.ntau as i64 {
        let blk = DenseMat::from_vec(g.size1, g.size2, g.mat.block(m as usize).to_vec());
        let p = blk.mat_mul(&f0);
        g.mat.block_mut(m as usize).copy_from_slice(p.raw());
    }
    if tstp == crate::types::layout::MATSUBARA {
        return;
    }
    let ft = DenseMat::from_vec(g.size1, g.size2, f.block(tstp).to_vec());
    for j in 0..=tstp {
        let fj = DenseMat::from_vec(g.size1, g.size2, f.block(j).to_vec());
        let blk = DenseMat::from_vec(g.size1, g.size2, g.ret.raw_block_at(tstp, j).to_vec());
        let p = blk.mat_mul(&fj);
        g.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
        let bl = DenseMat::from_vec(g.size1, g.size2, g.less.raw_block_at(j, tstp).to_vec());
        let pl = bl.mat_mul(&ft);
        g.less.raw_block_at_mut(j, tstp).copy_from_slice(pl.raw());
    }
    for m in 0..=g.ntau as i64 {
        let blk = DenseMat::from_vec(g.size1, g.size2, g.lmix.raw_block_at(tstp, m).to_vec());
        let p = blk.mat_mul(&f0);
        g.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
    }
}

/// Density matrix of the pseudo-particle Green's function (blueprint
/// `density_matrix(G::F^{ps}, t)`): `rho = -G^M(beta)` on the
/// Matsubara axis, `rho = i xi G^{<}(t, t)` at real time.
pub fn density_matrix_pseudo(g: &PseudoGreen, tstp: i64) -> Vec<Complex> {
    let xi = g.sign.as_i64() as f64;
    if tstp == crate::types::layout::MATSUBARA {
        g.mat
            .block(g.ntau)
            .iter()
            .map(|z| -*z)
            .collect()
    } else {
        assert!(
            (0..=g.nt as i64).contains(&tstp),
            "density_matrix_pseudo: tstp {tstp} out of range"
        );
        let blk = g.less.raw_block_at(tstp, tstp);
        (0..blk.len())
            .map(|q| Complex::new(0.0, xi) * blk[q])
            .collect()
    }
}

/// Density matrix at a time step object (blueprint
/// `density_matrix(g::f^{ps}, t)`).
pub fn density_matrix_pseudo_step(x: &PseudoGreenTimeStep, tstp: i64) -> Vec<Complex> {
    assert!(
        (x.tstp as i64) == tstp || (tstp == crate::types::layout::MATSUBARA && x.tstp == 0),
        "the time step object carries tstp {}, not {tstp}",
        x.tstp
    );
    let xi = x.sign.as_i64() as f64;
    if tstp == crate::types::layout::MATSUBARA {
        x.mat.block(x.ntau).iter().map(|z| -*z).collect()
    } else {
        let blk = x.less.raw_block_at(tstp, tstp);
        (0..blk.len())
            .map(|q| Complex::new(0.0, xi) * blk[q])
            .collect()
    }
}

/// Density (the trace of the density matrix, blueprint `density`).
pub fn density_pseudo(g: &PseudoGreen, tstp: i64) -> Complex {
    let rho = density_matrix_pseudo(g, tstp);
    let d = g.size1;
    (0..d).map(|i| rho[i * d + i]).sum()
}

use crate::linalg::mat::DenseMat;
