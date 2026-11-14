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

}

/// Pseudo-particle Green's function at a given time step `f^{ps}`
/// (blueprint reuses the layout of the standard timestep object; the
/// `ret` row stores the greater component).
pub(crate) type PseudoGreenTimeStep = GreenTimeStep;
