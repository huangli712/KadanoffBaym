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
//!
//! *Remarks* : *Pseudo-Particle Convention*
//!
//! The standard (`ℱ`) and pseudo-particle (`ℱᵖˢ`) Green's functions
//! share the same storage layout. They differ in the meaning of the stored
//! real time triangle (``G^{R}`` versus ``G^{>}``) and in the Matsubara and
//! left-mixing algorithms. The convention is a static property of the type,
//! and it is used to select the algorithms by dispatch.
//!
//! *References* :
//!
//! Please see [`NESSi`] `cntr_herm_pseudo_decl.hpp`.
//!
//! *Remarks : Pseudo-Particle Green's Functions*
//!
//! The pseudo-particle Green's function has the same storage layout as the
//! standard one, i.e., `mat`, `ret`, `lmix`, and `less`. However, the real
//! time triangle `ret` stores the greater component ``G^{>}(t,t')`` (with
//! ``t \ge t'``), instead of ``G^{R} = G^{>} - G^{<}``. See [`NESSi`]
//! `cntr_herm_pseudo_decl.hpp`.
//!
//! Thus `ℱᵖˢ` and `𝒻ᵖˢ` are separated from `ℱ` and `𝒻`, so that
//! a pseudo-particle object can not be passed to the standard Matsubara or
//! left-mixing algorithms by mistake.

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::Contour;
use crate::types::ContourGreen;
use crate::types::GreenTimeStep;
use crate::types::LessComp;
use crate::types::LmixComp;
use crate::types::MatComp;
use crate::types::RetComp;
use crate::types::Sign;

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
        if tstp == crate::types::MATSUBARA {
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

    /// Compatibility check (blueprint `iscompatible`).
    pub fn is_compatible(&self, h: &PseudoGreen) -> bool {
        self.sign == h.sign
            && self.nt == h.nt
            && self.ntau == h.ntau
            && self.size1 == h.size1
            && self.size2 == h.size2
    }

    /// Per-time-step distance (blueprint `distance(G, H, tstp)`).
    pub fn distance_at(&self, g2: &PseudoGreen, tstp: i64) -> f64 {
        self.to_contour().distance_at(&g2.to_contour(), tstp)
    }

    /// Copy the components at time step `tstp` (blueprint
    /// `memcpy!(A, Y, t)`).
    pub fn memcpy_step(&mut self, src: &PseudoGreen, tstp: i64) {
        if tstp == crate::types::MATSUBARA {
            self.mat.raw_mut().copy_from_slice(src.mat.raw());
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                let a = src.ret.raw_block_at(tstp, j);
                self.ret.raw_block_at_mut(tstp, j).copy_from_slice(a);
            }
            for i in 0..=tstp {
                let a = src.less.raw_block_at(i, tstp);
                self.less.raw_block_at_mut(i, tstp).copy_from_slice(a);
            }
            for m in 0..=self.ntau as i64 {
                let a = src.lmix.raw_block_at(tstp, m);
                self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(a);
            }
        }
    }

    /// Fill the components at time step `tstp` with `v` (blueprint
    /// `memset!(Z, t, v)`).
    pub fn memset_step(&mut self, tstp: i64, v: Complex) {
        if tstp == crate::types::MATSUBARA {
            self.mat.raw_mut().fill(v);
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                self.ret.raw_block_at_mut(tstp, j).fill(v);
            }
            for i in 0..=tstp {
                self.less.raw_block_at_mut(i, tstp).fill(v);
            }
            for m in 0..=self.ntau as i64 {
                self.lmix.raw_block_at_mut(tstp, m).fill(v);
            }
        }
    }

    /// Zero all the components (blueprint `zeros!(Z)`).
    pub fn zeros_all(&mut self) {
        self.mat.raw_mut().fill(Complex::ZERO);
        self.ret.raw_mut().fill(Complex::ZERO);
        self.lmix.raw_mut().fill(Complex::ZERO);
        self.less.raw_mut().fill(Complex::ZERO);
    }

    /// Elementwise `self += alpha * src` over all the components
    /// (blueprint `incr!(S, B, alpha)`).
    pub fn incr(&mut self, src: &PseudoGreen, alpha: Complex) {
        for (d, s) in [
            (self.mat.raw_mut(), src.mat.raw()),
            (self.ret.raw_mut(), src.ret.raw()),
            (self.lmix.raw_mut(), src.lmix.raw()),
            (self.less.raw_mut(), src.less.raw()),
        ] {
            for q in 0..d.len() {
                d[q] += alpha * s[q];
            }
        }
    }

    /// Scale the components at time step `tstp` by `alpha` (blueprint
    /// `smul!(S, t, a)`).
    pub fn smul_step(&mut self, tstp: i64, alpha: Complex) {
        if tstp == crate::types::MATSUBARA {
            for z in self.mat.raw_mut().iter_mut() {
                *z = *z * alpha;
            }
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                for z in self.ret.raw_block_at_mut(tstp, j).iter_mut() {
                    *z = *z * alpha;
                }
                for z in self.less.raw_block_at_mut(j, tstp).iter_mut() {
                    *z = *z * alpha;
                }
            }
            for m in 0..=self.ntau as i64 {
                for z in self.lmix.raw_block_at_mut(tstp, m).iter_mut() {
                    *z = *z * alpha;
                }
            }
        }
    }

    /// Elementwise scaling of all the components (blueprint
    /// `smul!(S, a)`).
    pub fn smul(&mut self, alpha: Complex) {
        for d in [
            self.mat.raw_mut(),
            self.ret.raw_mut(),
            self.lmix.raw_mut(),
            self.less.raw_mut(),
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
    pub fn lmul_cf(&mut self, f: &crate::types::TimeFunction, tstp: i64) {
        let f0 = f.block(crate::types::MATSUBARA);
        for m in 0..=self.ntau as i64 {
            let fm = DenseMat::from_vec(self.size1, self.size2, f0.to_vec());
            let blk = DenseMat::from_vec(self.size1, self.size2, self.mat.block(m as usize).to_vec());
            let p = fm.mat_mul(&blk);
            self.mat.block_mut(m as usize).copy_from_slice(p.raw());
        }
        if tstp == crate::types::MATSUBARA {
            return;
        }
        let ft = DenseMat::from_vec(self.size1, self.size2, f.block(tstp).to_vec());
        for j in 0..=tstp {
            let blk = DenseMat::from_vec(self.size1, self.size2, self.ret.raw_block_at(tstp, j).to_vec());
            let p = ft.mat_mul(&blk);
            self.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
            let fj = DenseMat::from_vec(self.size1, self.size2, f.block(j).to_vec());
            let bl = DenseMat::from_vec(self.size1, self.size2, self.less.raw_block_at(j, tstp).to_vec());
            let pl = fj.mat_mul(&bl);
            self.less.raw_block_at_mut(j, tstp).copy_from_slice(pl.raw());
        }
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_vec(self.size1, self.size2, self.lmix.raw_block_at(tstp, m).to_vec());
            let p = ft.mat_mul(&blk);
            self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
        }
    }

    /// Right multiplication by a contour function at time step `tstp`
    /// (blueprint `smul!(R, f, t)`): `f` from the right, the ret blocks
    /// are weighted by their column time, the left-mixing component by
    /// `f_0`.
    pub fn rmul_cf(&mut self, f: &crate::types::TimeFunction, tstp: i64) {
        let f0 = DenseMat::from_vec(self.size1, self.size2, f.block(crate::types::MATSUBARA).to_vec());
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_vec(self.size1, self.size2, self.mat.block(m as usize).to_vec());
            let p = blk.mat_mul(&f0);
            self.mat.block_mut(m as usize).copy_from_slice(p.raw());
        }
        if tstp == crate::types::MATSUBARA {
            return;
        }
        let ft = DenseMat::from_vec(self.size1, self.size2, f.block(tstp).to_vec());
        for j in 0..=tstp {
            let fj = DenseMat::from_vec(self.size1, self.size2, f.block(j).to_vec());
            let blk = DenseMat::from_vec(self.size1, self.size2, self.ret.raw_block_at(tstp, j).to_vec());
            let p = blk.mat_mul(&fj);
            self.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
            let bl = DenseMat::from_vec(self.size1, self.size2, self.less.raw_block_at(j, tstp).to_vec());
            let pl = bl.mat_mul(&ft);
            self.less.raw_block_at_mut(j, tstp).copy_from_slice(pl.raw());
        }
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_vec(self.size1, self.size2, self.lmix.raw_block_at(tstp, m).to_vec());
            let p = blk.mat_mul(&f0);
            self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
        }
    }
}

/// Pseudo-particle Green's function at a given time step `f^{ps}`
/// (blueprint reuses the layout of the standard timestep object; the
/// `ret` row stores the greater component).
pub(crate) type PseudoGreenTimeStep = GreenTimeStep;
