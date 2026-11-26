// Project : KadanoffBaym
// File    : src/algebra/properties.rs
// Purpose : compatibility, equality, and distance measures

//!
//! *Remarks* : *Distance Between Contour-Ordered Green's Functions*
//!
//! The distance between two contour-ordered Green's functions `A` and `B`
//! at given time step `tstp` is defined as the sum of the Frobenius norms
//! of the differences of the matrix blocks:
//!
//! ```math
//! \begin{equation}
//! d^{R}(t) = \sum_{j} \| A^{R}(t,t_j) - B^{R}(t,t_j) \|,
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! d^{\rceil}(t) = \sum_{m} \| A^{\rceil}(t,\tau_m) - B^{\rceil}(t,\tau_m) \|,
//! \end{equation}
//! ```
//!
//! ```math
//! \begin{equation}
//! d^{<}(t) = \sum_{i} \| A^{<}(t_i,t) - B^{<}(t_i,t) \|.
//! \end{equation}
//! ```
//!
//! Here ``t_j \le t`` and ``t_i \le t``. The total distance is
//! ``d^{R} + d^{\rceil} + d^{<}`` at the real time axis, and
//! ``\sum_m \| A^{M}(\tau_m) - B^{M}(\tau_m) \|`` at the Matsubara axis.
//! Note that it is not normalized.
//!
//! Two metrics are implemented: the formulas above (`frobenius_distance`, applied per
//! row and per Matsubara block), and the legacy component-pair distance
//! `sum_k |sum_elements (a_k - b_k)|`.
//!

use crate::complex::Complex;
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

impl ContourGreen {
    /// Same orbital dimensions.
    pub(crate) fn equaldims(&self, other: &ContourGreen) -> bool {
        self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Same grid and orbital shape (nt, ntau, sizes).
    pub(crate) fn is_compatible(&self, other: &ContourGreen) -> bool {
        self.nt == other.nt && self.ntau == other.ntau && self.equaldims(other)
    }

    /// Exact element-wise equality of all components.
    pub fn equal(&self, other: &ContourGreen) -> bool {
        self.is_compatible(other)
            && self.mat.equal(&other.mat)
            && self.ret.equal(&other.ret)
            && self.lmix.equal(&other.lmix)
            && self.less.equal(&other.less)
    }

    /// Frobenius distance of the retarded row at `tstp`.
    pub fn distance_ret(&self, other: &ContourGreen, tstp: i64) -> f64 {
        self.check_pair(other, tstp);
        if tstp == MATSUBARA {
            return 0.0;
        }
        let mut err = 0.0;
        for j in 0..=tstp as usize {
            err += frobenius_distance(
                &self.ret.row(tstp)[j * self.size1 * self.size2..(j + 1) * self.size1 * self.size2],
                &other.ret.row(tstp)
                    [j * self.size1 * self.size2..(j + 1) * self.size1 * self.size2],
            );
        }
        err
    }

    /// Frobenius distance of the left-mixing row at `tstp`; zero for
    /// `MATSUBARA`.
    pub fn distance_lmix(&self, other: &ContourGreen, tstp: i64) -> f64 {
        self.check_pair(other, tstp);
        if tstp == MATSUBARA {
            return 0.0;
        }
        let b = self.size1 * self.size2;
        let row = self.lmix.row(tstp);
        let row_other = other.lmix.row(tstp);
        let mut err = 0.0;
        for k in 0..=self.ntau {
            err += frobenius_distance(&row[k * b..(k + 1) * b], &row_other[k * b..(k + 1) * b]);
        }
        err
    }

    /// Frobenius distance of the lesser row at `tstp`; zero for `MATSUBARA`.
    pub fn distance_less(&self, other: &ContourGreen, tstp: i64) -> f64 {
        self.check_pair(other, tstp);
        if tstp == MATSUBARA {
            return 0.0;
        }
        let b = self.size1 * self.size2;
        let mut err = 0.0;
        for i in 0..=tstp as usize {
            err += frobenius_distance(
                &self.less.row(tstp)[i * b..(i + 1) * b],
                &other.less.row(tstp)[i * b..(i + 1) * b],
            );
        }
        err
    }

    /// Frobenius distance of the Matsubara component.
    pub fn distance_mat(&self, other: &ContourGreen) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible greens");
        let b = self.size1 * self.size2;
        let mut err = 0.0;
        for n in 0..=self.ntau {
            err += frobenius_distance(
                &self.mat.raw()[n * b..(n + 1) * b],
                &other.mat.raw()[n * b..(n + 1) * b],
            );
        }
        err
    }

    /// Total distance at `tstp`: Matsubara for `MATSUBARA`, otherwise
    /// ret + lmix + less rows.
    pub fn distance_at(&self, other: &ContourGreen, tstp: i64) -> f64 {
        if tstp == MATSUBARA {
            self.check_pair(other, tstp);
            self.distance_mat(other)
        } else {
            self.distance_ret(other, tstp)
                + self.distance_lmix(other, tstp)
                + self.distance_less(other, tstp)
        }
    }

    /// Range and compatibility check shared by the distance family.
    fn check_pair(&self, other: &ContourGreen, tstp: i64) {
        assert!(
            (MATSUBARA..=self.nt as i64).contains(&tstp),
            "tstp {tstp} out of range"
        );
        assert!(self.is_compatible(other), "distance: incompatible greens");
    }
}

impl GreenTimeStep {
    /// Same orbital dimensions.
    pub(crate) fn equaldims(&self, other: &GreenTimeStep) -> bool {
        self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Same grid and orbital shape (tstp, ntau, sizes).
    pub(crate) fn is_compatible(&self, other: &GreenTimeStep) -> bool {
        self.tstp == other.tstp && self.ntau == other.ntau && self.equaldims(other)
    }

    /// Exact element-wise equality of all components.
    pub fn equal(&self, other: &GreenTimeStep) -> bool {
        self.is_compatible(other)
            && self.mat.equal(&other.mat)
            && self.ret.equal(&other.ret)
            && self.lmix.equal(&other.lmix)
            && self.less.equal(&other.less)
    }
}

impl TimeFunction {
    /// Same grid shape.
    pub(crate) fn is_compatible(&self, other: &TimeFunction) -> bool {
        self.nt == other.nt && self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &TimeFunction) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance (per-block abs of the element-sum difference).
    pub fn distance(&self, other: &TimeFunction) -> f64 {
        block_sum_distance(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl MatComp {
    /// Whether the two components share the same grid shape.
    pub(crate) fn is_compatible(&self, other: &MatComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &MatComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`.
    pub fn distance(&self, other: &MatComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        block_sum_distance(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl RetComp {
    /// Whether the two components share the same grid shape.
    pub(crate) fn is_compatible(&self, other: &RetComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &RetComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`.
    pub fn distance(&self, other: &RetComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        block_sum_distance(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl LmixComp {
    /// Whether the two components share the same grid shape.
    pub(crate) fn is_compatible(&self, other: &LmixComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &LmixComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`.
    pub fn distance(&self, other: &LmixComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        block_sum_distance(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl LessComp {
    /// Whether the two components share the same grid shape.
    pub(crate) fn is_compatible(&self, other: &LessComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &LessComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`.
    pub fn distance(&self, other: &LessComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        block_sum_distance(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

/// Frobenius norm of the block difference.
#[inline]
pub fn frobenius_distance(a: &[Complex], b: &[Complex]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += (*x - *y).abs_sqr();
    }
    s.sqrt()
}

/// Legacy component metric: `sum_k |sum_elements (a_k - b_k)|`,
/// where `block` is the number of elements per orbital block.
fn block_sum_distance(a: &[Complex], b: &[Complex], block: usize) -> f64 {
    let mut err = 0.0;
    for (a_blk, b_blk) in a.chunks(block).zip(b.chunks(block)) {
        let mut s = Complex::ZERO;
        for (x, y) in a_blk.iter().zip(b_blk) {
            s += *x - *y;
        }
        err += s.abs();
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    fn mk(nt: usize) -> ContourGreen {
        let c = Contour::new(nt as i64, 2, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    #[test]
    fn compat_and_equal() {
        let a = mk(2);
        let mut b = mk(2);
        assert!(a.is_compatible(&b));
        assert!(a.equal(&b));
        b.set_mat(0, 0, 0, Complex::ONE);
        assert!(!a.equal(&b));
        let c = mk(3);
        // different nt: components differ, dims agree
        assert!(a.equaldims(&c));
        assert!(!a.is_compatible(&c));
    }

    #[test]
    fn frobenius_row_distances() {
        let a = mk(2);
        let mut b = mk(2);
        // mat differs by (1,0): distance_mat = 1
        b.set_mat(0, 0, 0, Complex::ONE);
        assert!((a.distance_mat(&b) - 1.0).abs() < 1e-15);
        // ret row 2: one element 3+4i -> Frobenius 5
        b.set_ret(2, 1, 0, 0, Complex::new(3.0, 4.0));
        assert!((a.distance_ret(&b, 2) - 5.0).abs() < 1e-14);
        assert!((a.distance_at(&b, 2) - 5.0).abs() < 1e-14);
        // MATSUBARA scope
        assert!((a.distance_at(&b, MATSUBARA) - 1.0).abs() < 1e-15);
        // lmix row
        b.set_lmix(1, 2, 0, 0, Complex::new(0.0, 2.0));
        assert!((a.distance_lmix(&b, 1) - 2.0).abs() < 1e-15);
        // less row
        b.set_less(0, 2, 0, 0, Complex::new(1.0, 1.0));
        assert!((a.distance_less(&b, 2) - 2.0f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn legacy_component_distance() {
        let a = mk(1);
        let mut b = mk(1);
        b.set_mat(0, 0, 0, Complex::new(1.0, 1.0));
        b.set_mat(0, 0, 1, Complex::new(2.0, -3.0));
        // |sum of differences| over the whole component: |(-1-i) + (-2+3i)| = |-3+2i| = sqrt(13)
        let d = a.mat.distance(&b.mat);
        assert!((d - 13.0f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn time_function_distance() {
        let a = TimeFunction::new(1, 1, 1, Complex::ZERO);
        let mut b = TimeFunction::new(1, 1, 1, Complex::ZERO);
        assert!(a.equal(&b));
        b.set(MATSUBARA, 0, 0, Complex::new(0.0, 1.0));
        assert!(!a.equal(&b));
        assert!((a.distance(&b) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn distance_block_norms() {
        let c = Contour::new(5, 4, 2, 2, 0.5, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let tstp = 4i64;
        // single block difference 3 + 4i -> norm 5
        let mut blk = [Complex::ZERO; 4];
        blk[0] = Complex::new(3.0, 4.0);
        a.ret_raw_block_mut(tstp, 1).copy_from_slice(&blk);
        assert_eq!(a.distance_ret(&b, tstp), 5.0);
        assert_eq!(a.distance_lmix(&b, tstp), 0.0);
        assert_eq!(a.distance_less(&b, tstp), 0.0);
        // elements with vanishing sum but non-zero norm
        let mut tv = [Complex::ZERO; 4];
        tv[0] = Complex::new(1.0, 0.0);
        tv[1] = Complex::new(-1.0, 0.0);
        a.lmix_raw_block_mut(tstp, 2).copy_from_slice(&tv);
        let mut les = [Complex::ZERO; 4];
        les[1] = Complex::new(2.0, 0.0);
        a.less_raw_block_mut(1, tstp).copy_from_slice(&les);
        assert!((a.distance_lmix(&b, tstp) - 2.0f64.sqrt()).abs() < 1e-14);
        assert_eq!(a.distance_less(&b, tstp), 2.0);
        let total = a.distance_at(&b, tstp);
        assert!((total - (5.0 + 2.0f64.sqrt() + 2.0)).abs() < 1e-14);
        assert!(
            (total
                - (a.distance_ret(&b, tstp)
                    + a.distance_lmix(&b, tstp)
                    + a.distance_less(&b, tstp)))
            .abs()
                < 1e-14
        );
        // Matsubara branch
        let mut mb = [Complex::ZERO; 4];
        mb[0] = Complex::new(1.0, 0.0);
        mb[1] = Complex::new(-1.0, 0.0);
        a.mat_raw_block_mut(1).copy_from_slice(&mb);
        assert!((a.distance_at(&b, MATSUBARA) - 2.0f64.sqrt()).abs() < 1e-14);
        assert_eq!(a.distance_ret(&b, MATSUBARA), 0.0);
    }

    #[test]
    fn component_properties() {
        // G-mat: ntau = 51 points -> 50 intervals
        let m1 = MatComp::new(50, 2, 3, Complex::ZERO);
        let m2 = MatComp::new(50, 2, 3, Complex::ZERO);
        assert_eq!(m1.nblocks(), 51);
        assert_eq!((m1.size1, m1.size2), (2, 3));
        assert_ne!(m1.size1, m1.size2);
        assert!(m1.is_compatible(&m2));
        assert!(m1.distance(&m2) < 1e-7);
        // G-ret: ntime = 101 points -> nt = 100
        let r1 = RetComp::new(100, 2, 3, Complex::ZERO);
        let r2 = RetComp::new(100, 2, 3, Complex::ZERO);
        assert_eq!(r1.nblocks(), (101 * 102) / 2);
        assert_ne!(r1.size1, r1.size2);
        assert!(r1.is_compatible(&r2));
        // G-lmix
        let l1 = LmixComp::new(100, 50, 2, 3, Complex::ZERO);
        let l2 = LmixComp::new(100, 50, 2, 3, Complex::ZERO);
        assert_eq!(l1.nblocks(), 101 * 51);
        assert!(l1.is_compatible(&l2));
        // G-less
        let s1 = LessComp::new(100, 2, 3, Complex::ZERO);
        let s2 = LessComp::new(100, 2, 3, Complex::ZERO);
        assert_eq!(s1.nblocks(), (101 * 102) / 2);
        assert!(s1.is_compatible(&s2));
    }
}
