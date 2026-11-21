// Project : KadanoffBaym
// File    : src/types/green.rs
// Purpose : ContourGreen (GreenTimeStep and TimeFunction live in
//           step.rs and function.rs; the component containers in
//           mat.rs, ret.rs, less.rs, lmix.rs)
//
// Components store their own arrays (G* owned by the full
// ContourGreen, g* by GreenTimeStep). Indices are 0-based:
// tstp = 0..nt, tau = 0..ntau, Matsubara via tstp = MATSUBARA (-1).

//!
//! *Remarks* : *Type Layout*
//!
//! There is no abstract type tree in the Rust port; the containers are
//! concrete structs:
//!
//! * component containers (`mat.rs`, `ret.rs`, `lmix.rs`, `less.rs`):
//!   `MatComp` (Gᵐᵃᵗ), `RetComp` (Gʳᵉᵗ), `LmixComp` (Gˡᵐⁱˣ), and
//!   `LessComp` (Gˡᵉˢˢ), each storing its own flat `Complex` blocks
//! * whole objects: `ContourGreen` (this file, four owned components),
//!   `GreenTimeStep` (`step.rs`, one row at `tstp`), `TimeFunction`
//!   (`function.rs`, `Cf`), and `PseudoGreen` (`pseudo.rs`, the
//!   pseudo-particle function `ℱᵖˢ` with the same storage as
//!   `ContourGreen`)
//! * stationary (NESS2) containers: `SteadyGreen`, `SteadyFunction`,
//!   and the grid `SteadyGrid` (`steady.rs`)
//! * moving (truncated time window) containers: `MovingGreen`,
//!   `MovingGreenTimeStep`, and `MovingTimeFunction` (`moving.rs`),
//!   which store only the real-time components in a window of
//!   `tc + 1` time steps (see `crate::variants`)
//!
//! The auxiliary components Gᵐᵃᵗᵐ, Gᵃᵈᵛ, Gʳᵐⁱˣ, and Gᵍᵗʳ are not
//! types here: they are derived block views provided as accessor
//! methods on `ContourGreen` and `GreenTimeStep` (`basics/indexing.rs`).
//!
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
//! *Remarks : Implemented Components*
//!
//! In the KadanoffBaym library, the four independent components of a
//! contour Green's function are stored in the owned containers
//!
//! * `MatComp` (Gᵐᵃᵗ)
//! * `RetComp` (Gʳᵉᵗ)
//! * `LmixComp` (Gˡᵐⁱˣ)
//! * `LessComp` (Gˡᵉˢˢ)
//!
//! of a `ContourGreen`. The auxiliary components
//!
//! * Gᵐᵃᵗᵐ (Matsubara-lesser hybrid)
//! * Gᵃᵈᵛ (advanced)
//! * Gʳᵐⁱˣ (right-mixing)
//! * Gᵍᵗʳ (greater)
//!
//! are derived on demand as block views
//! (`matminus_block`/`adv_block`/`rmix_block`/`gtr_block` in
//! `basics/indexing.rs`); they own no storage. Note that these
//! components are related with each other.

use crate::complex::Complex;
use crate::types::{Contour, LessComp, LmixComp, MatComp, RetComp, Sign};

/// Full contour-ordered Green's function : four owned components on an
/// L-shaped contour.
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
    /// Standard constructor : every element initialized to `v`.
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
    fn lmix_rejects_bad_tau() {
        let l = LmixComp::new(2, 3, 1, 1, Complex::ZERO);
        let _ = l.idx(0, 4);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn ret_rejects_upper_triangle() {
        let r = RetComp::new(2, 1, 1, Complex::ZERO);
        let _ = r.idx(0, 1);
    }

    #[test]
    fn f_struct_constructors() {
        // three objects on the same contour are equal
        let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
        let f1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let f2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let f3 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        assert!(f1.equal(&f2));
        assert!(f1.equal(&f3));
        // different contour -> unequal
        let c_default = Contour::new(10, 10, 2, 2, 1.0, 1.0);
        let f4 = ContourGreen::new(&c_default, Sign::Fermi, Complex::ZERO);
        assert!(!f1.equal(&f4));
    }

    /// Total word count of the payload arithmetic.
    #[test]
    fn release_memory_and_allocation_gate() {
        // ntime = 1001, ntau = 401 points -> 1000/400, 2x2
        let c = Contour::new(1000, 400, 2, 2, 10.0, 4.0);
        let f = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // payload = 16 * 4 * (2*ntri + 1001*401 + 401) bytes
        assert_eq!(f.nwords() * 8, 89_907_456);
    }
}
