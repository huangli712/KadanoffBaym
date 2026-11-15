// Project : KadanoffBaym
// File    : src/types/green.rs
// Purpose : ContourGreen (GreenTimeStep and TimeFunction live in
//           step.rs and function.rs; the component containers in
//           mat.rs, ret.rs, less.rs, lmix.rs)
//
// Blueprint: julia/src/types/{green,function,mat,ret,less,lmix}.jl.
// Julia components store their own arrays (G* owned by ℱ, g* owned by
// 𝒻); the same layout is used here. Indices are 0-based (D6):
// tstp = 0..nt, tau = 0..ntau, Matsubara via tstp = MATSUBARA (-1).
//!
//! *Remarks* : *Type System*
//!
//! We need a few abstract types to construct the type systems.These abstract
//! types include:
//!
//! * *CnAbstractType*
//! * *CnAbstractContour*
//! * *CnAbstractMatrix*
//! * *CnAbstractVector*
//! * *CnAbstractFunction*
//! * *CnAbstractStationaryFunction*
//! * *CnAbstractMovingMatrix*
//! * *CnAbstractMovingVector*
//! * *CnAbstractMovingFunction*
//!
//! They should not be used in the user's codes directly. The hierarchical
//! types appeared in the KadanoffBaym library are collected as follows:
//!
//! * *CnAbstractType*
//!   * *CnAbstractContour*
//!     * *Cn*
//!   * *CnAbstractMatrix*
//!     * *Gᵐᵃᵗ*
//!     * *Gʳᵉᵗ*
//!     * *Gˡᵐⁱˣ*
//!     * *Gˡᵉˢˢ*
//!     * *Gᵐᵃᵗᵐ*
//!     * *Gᵃᵈᵛ*
//!     * *Gʳᵐⁱˣ*
//!     * *Gᵍᵗʳ*
//!     * *CnAbstractMovingMatrix*
//!       * *Gʳᵉᵗᵐᵒᵛ*
//!       * *Gˡᵉˢˢᵐᵒᵛ*
//!   * *CnAbstractVector*
//!     * *gᵐᵃᵗ*
//!     * *gʳᵉᵗ*
//!     * *gˡᵐⁱˣ*
//!     * *gˡᵉˢˢ*
//!     * *gᵐᵃᵗᵐ*
//!     * *gᵃᵈᵛ*
//!     * *gʳᵐⁱˣ*
//!     * *gᵍᵗʳ*
//!     * *CnAbstractMovingVector*
//!       * *gʳᵉᵗᵐᵒᵛ*
//!       * *gˡᵉˢˢᵐᵒᵛ*
//!   * *CnAbstractFunction*
//!     * *Cf*
//!     * *ℱ*
//!     * *𝒻*
//!     * *ℱᵖˢ*
//!     * *𝒻ᵖˢ*
//!     * *CnAbstractMovingFunction*
//!       * *ℱᵐᵒᵛ*
//!       * *𝒻ᵐᵒᵛ*
//!       * *𝒻ᵛⁱᵉʷᵐᵒᵛ*
//!       * *Cfᵐᵒᵛ*
//!   * *CnAbstractStationaryFunction*
//!     * *SteadyGreen*
//!
//! The moving (truncated time window) types store only the real-time
//! components in a window of `tc + 1` time steps, see `moving.jl`.
//!
//! `CnAbstractFunction` contains the functions defined on the contour,
//! while `CnAbstractStationaryFunction` contains the stationary functions
//! of the relative time or the frequency (NESS2). The numerical storage
//! `SteadyFunction` and the grid `SteadyGrid` are not in this type system.
//!
//! The `subtypetree()` function can be used to sketch this type system.
//!
//! #=
//! *Remarks : Contour-Ordered Green's Functions*
//!
//! In the many-body theories, single-particle Green's functions are the
//! fundamental objects. They describe single-particle excitations as well
//! as statistical distributions of particles, and play a central role in
//! the formulation of nonequilibrium dynamical mean-field theory. We define
//! the nonequilibrium Green's function as the contour-ordered expectation
//! value,
//!
//! ```math
//! \begin{equation}
//! G(t,t') \equiv
//!     -i \langle \mathcal{T}_{\mathcal{C}} c(t) c^{\dagger}(t') \rangle,
//! \end{equation}
//! ```
//!
//! where ``c^{\dagger}(c)`` is a creation (annihilation) operators of
//! particles, and ``t,\ t' \in \mathcal{C}``. For simplicity, spin and
//! orbital indices associated with the operators are not shown. Because
//! of the three branches, on which the time arguments ``t`` and ``t'``
//! can lie, the Green's function has ``3 \times 3 = 9`` components:
//!
//! ```math
//! \begin{equation}
//! G(t,t') \equiv G_{ij}(t,t'),
//! \end{equation}
//! ```
//!
//! where ``t \in \mathcal{C}_i``, ``t' \in \mathcal{C}_j``, and
//! ``i,\ j = 1,\ 2,\ 3``. Conventionally, we can express them in a
//! ``3 \times 3`` matrix form
//!
//! ```math
//! \begin{equation}
//! \hat{G} =
//! \begin{pmatrix}
//! G_{11} & G_{12} & G_{13} \\
//! G_{21} & G_{22} & G_{23} \\
//! G_{31} & G_{32} & G_{33} \\
//! \end{pmatrix}.
//! \end{equation}
//! ```
//!
//! In general, one can shift the operator with the largest real-time
//! argument from ``\mathcal{C}_1`` to ``\mathcal{C}_2`` (and vice versa),
//! because the time evolution along ``\mathcal{C}_1`` and ``\mathcal{C}_2``
//! to the right of that operator cancels. This kind of redundancy implies
//! the following relations among the components of the above matrix:
//!
//! ```math
//! \begin{equation}
//! G_{11}(t,t') = G_{12}(t,t'),\ \text{for}\ t \le t',
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! G_{11}(t,t') = G_{21}(t,t'),\ \text{for}\ t > t',
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! G_{22}(t,t') = G_{21}(t,t'),\ \text{for}\ t < t',
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! G_{22}(t,t') = G_{12}(t,t'),\ \text{for}\ t \ge t',
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! G_{13}(t,\tau') = G_{23}(t,\tau'),
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! G_{31}(\tau,t') = G_{32}(\tau,t').
//! \end{equation}
//! ```
//!
//! The above equations can be summarized as
//!
//! ```math
//! \begin{equation}
//! G_{11} + G_{22} = G_{12} + G_{21}.
//! \end{equation}
//! ```
//!
//! These equations thus allow one to eliminate three components out of
//! nine in the nonequilibrium Green's function. To this end, we introduce
//! six linearly independent physical Green's functions, namely the retarded
//! (``G^{R}``), advanced (``G^{A}``), Keldysh (``G^{K}``), left-mixing
//! (``G^{\rceil}``), right-mixing (``G^{\lceil}``), and Matsubara Green's
//! functions (``G^{M}``). Their definitions and relevant properties will
//! be given in the following remarks if needed.
//!
//! *Remarks* :
//!
//! In the KadanoffBaym library, the four independent components of contour
//! Green's function are as follows:
//!
//! * *Gᵐᵃᵗ*
//! * *Gʳᵉᵗ*
//! * *Gˡᵐⁱˣ*
//! * *Gˡᵉˢˢ*
//!
//! They have been defined above. Next, we would like to define some auxiliary
//! components of contour-ordered Green's function. They are:
//!
//! * *Gᵐᵃᵗᵐ*
//! * *Gᵃᵈᵛ*
//! * *Gʳᵐⁱˣ*
//! * *Gᵍᵗʳ*
//!
//! Note that these components are related with each other.
//!
//! Return the convention of the contour-ordered Green's function, namely
//! `Val(:standard)` or `Val(:pseudo)`. It is used to select the Matsubara
//! and left-mixing algorithms only.
//!
//! See also: [`ℱ`](@ref), [`ℱᵖˢ`](@ref).

use crate::types::{Contour, LessComp, LmixComp, MatComp, RetComp, Sign};
use crate::complex::Complex;

/// Full contour-ordered Green's function (blueprint `ℱ`): four owned
/// components on an L-shaped contour.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourGreen {
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

impl ContourGreen {
    /// Standard constructor (blueprint `ℱ(C, v, sign)`): every element
    /// initialized to `v`.
    pub fn new(c: &Contour, sign: Sign, v: Complex) -> ContourGreen {
        let (nt, ntau) = (c.nt as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        ContourGreen {
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

    /// Zero-initialized constructor with default sign (FERMI).
    pub fn zeros(c: &Contour) -> ContourGreen {
        ContourGreen::new(c, Sign::Fermi, Complex::ZERO)
    }

    /// Total number of stored f64 words (memory gate check).
    pub fn nwords(&self) -> usize {
        (self.mat.raw().len()
            + self.ret.raw().len()
            + self.lmix.raw().len()
            + self.less.raw().len())
            * 2
    }
}

// GreenTimeStep moved to step.rs, TimeFunction to function.rs.

#[cfg(test)]
mod tests {
    use super::*;

    fn small_contour() -> Contour {
        Contour::new(3, 4, 2, 2, 1.0, 2.0)
    }

    #[test]
    fn component_sizes() {
        let c = small_contour();
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        assert_eq!(g.mat.nblocks(), 5);
        assert_eq!(g.ret.nblocks(), 10); // 1+2+3+4
        assert_eq!(g.less.nblocks(), 10);
        assert_eq!(g.lmix.nblocks(), 4 * 5);
        // Each block is 2x2 = 4 elements.
        assert_eq!(g.ret.raw().len(), 40);
        // mat 20 + ret 40 + lmix 80 + less 40 = 180 elements = 360 words
        assert_eq!(g.nwords(), 360);
    }

    #[test]
    fn triangle_index_matches_offsets() {
        let r = RetComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(r.idx(0, 0), 0);
        assert_eq!(r.idx(1, 0), 1);
        assert_eq!(r.idx(1, 1), 2);
        assert_eq!(r.idx(4, 4), 14);
        let l = LessComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(l.idx(1, 2), 4);
        assert_eq!(l.idx(0, 4), 10);
    }

    #[test]
    fn row_slices_span_the_row() {
        let mut r = RetComp::new(3, 1, 2, Complex::ZERO);
        assert_eq!(r.row(0).len(), 2);
        assert_eq!(r.row(3).len(), 8);
        r.row_mut(2)[0] = Complex::ONE;
        assert_eq!(r.get(r.idx(2, 0), 0, 0), Complex::ONE);
    }

    #[test]
    fn lmix_row_slices_span_the_row() {
        let mut l = LmixComp::new(3, 4, 1, 2, Complex::ZERO);
        assert_eq!(l.row(0).len(), 10);
        assert_eq!(l.row(3).len(), 10);
        l.row_mut(2)[0] = Complex::ONE;
        assert_eq!(l.get(l.idx(2, 0), 0, 0), Complex::ONE);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn ret_rejects_upper_triangle() {
        let r = RetComp::new(2, 1, 1, Complex::ZERO);
        let _ = r.idx(0, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn lmix_rejects_bad_tau() {
        let l = LmixComp::new(2, 3, 1, 1, Complex::ZERO);
        let _ = l.idx(0, 4);
    }
}
