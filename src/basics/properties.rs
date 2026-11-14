// Project : KadanoffBaym
// File    : src/basics/properties.rs
// Purpose : compatibility, equality, and distance measures
//
// Blueprint: julia/src/properties/{compat,equal,distance}.jl.
//
// Two metrics exist in the blueprint:
//   * component-pair distance (legacy): sum over blocks of
//     |sum of the element-wise difference| (abs of the block sum);
//   * Green-level distance (per time step): sum of the Frobenius norms
//     of the block differences (`_blkdist`), for the ret/lmix/less rows
//     and the Matsubara component.

use crate::Complex;
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};
use crate::types::MATSUBARA;

/// Frobenius norm of the block difference (blueprint `_blkdist`).
#[inline]
pub fn blkdist(a: &[Complex], b: &[Complex]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += (*x - *y).abs_sqr();
    }
    s.sqrt()
}

/// Legacy component metric: `sum_k |sum_elements (a_k - b_k)|`
/// where `block` is the number of elements per orbital block
/// (blueprint component-pair `distance`).
pub(crate) fn dist_sum(a: &[Complex], b: &[Complex], block: usize) -> f64 {
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

impl MatComp {
    /// Whether the two components share the same grid shape
    /// (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &MatComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality (blueprint `equal`).
    pub fn equal(&self, other: &MatComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`
    /// (blueprint component-pair `distance`).
    pub fn distance(&self, other: &MatComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        dist_sum(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl RetComp {
    /// Whether the two components share the same grid shape
    /// (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &RetComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality (blueprint `equal`).
    pub fn equal(&self, other: &RetComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`
    /// (blueprint component-pair `distance`).
    pub fn distance(&self, other: &RetComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        dist_sum(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl LmixComp {
    /// Whether the two components share the same grid shape
    /// (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &LmixComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality (blueprint `equal`).
    pub fn equal(&self, other: &LmixComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`
    /// (blueprint component-pair `distance`).
    pub fn distance(&self, other: &LmixComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        dist_sum(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl LessComp {
    /// Whether the two components share the same grid shape
    /// (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &LessComp) -> bool {
        self.raw().len() == other.raw().len()
            && self.size1 == other.size1
            && self.size2 == other.size2
    }

    /// Exact element-wise equality (blueprint `equal`).
    pub fn equal(&self, other: &LessComp) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance `sum_k |sum_elements (a_k - b_k)|`
    /// (blueprint component-pair `distance`).
    pub fn distance(&self, other: &LessComp) -> f64 {
        assert!(self.is_compatible(other), "distance: incompatible shapes");
        dist_sum(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl TimeFunction {
    /// Same grid shape.
    pub fn is_compatible(&self, other: &TimeFunction) -> bool {
        self.nt == other.nt && self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Exact element-wise equality.
    pub fn equal(&self, other: &TimeFunction) -> bool {
        self.is_compatible(other) && self.raw() == other.raw()
    }

    /// Legacy distance (per-block abs of the element-sum difference).
    pub fn distance(&self, other: &TimeFunction) -> f64 {
        dist_sum(self.raw(), other.raw(), self.size1 * self.size2)
    }
}

impl ContourGreen {
    /// Same orbital dimensions (blueprint `getdims` comparison).
    pub fn equaldims(&self, other: &ContourGreen) -> bool {
        self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Exact element-wise equality of all components.
    pub fn equal(&self, other: &ContourGreen) -> bool {
        self.is_compatible(other)
            && self.mat.equal(&other.mat)
            && self.ret.equal(&other.ret)
            && self.lmix.equal(&other.lmix)
            && self.less.equal(&other.less)
    }

    /// Same grid and orbital shape (nt, ntau, sizes).
    pub fn is_compatible(&self, other: &ContourGreen) -> bool {
        self.nt == other.nt && self.ntau == other.ntau && self.equaldims(other)
    }

    fn check_pair(&self, other: &ContourGreen, tstp: i64) {
        assert!(
            (MATSUBARA..=self.nt as i64).contains(&tstp),
            "tstp {tstp} out of range"
        );
        assert!(self.is_compatible(other), "distance: incompatible greens");
    }

    /// Frobenius distance of the retarded row at `tstp`
    /// (blueprint `distance(A, B, tstp, Val{:ret})`).
    pub fn distance_ret(&self, other: &ContourGreen, tstp: i64) -> f64 {
        self.check_pair(other, tstp);
        if tstp == MATSUBARA {
            return 0.0;
        }
        let mut err = 0.0;
        for j in 0..=tstp as usize {
            err += blkdist(
                &self.ret.row(tstp)[j * self.size1 * self.size2..(j + 1) * self.size1 * self.size2],
                &other.ret.row(tstp)
                    [j * self.size1 * self.size2..(j + 1) * self.size1 * self.size2],
            );
        }
        err
    }

    /// Frobenius distance of the lmix row at `tstp`
    /// (blueprint `Val{:tv}`; zero for `MATSUBARA`).
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
            err += blkdist(&row[k * b..(k + 1) * b], &row_other[k * b..(k + 1) * b]);
        }
        err
    }

    /// Frobenius distance of the lesser row at `tstp`
    /// (blueprint `Val{:les}`; zero for `MATSUBARA`).
    pub fn distance_less(&self, other: &ContourGreen, tstp: i64) -> f64 {
        self.check_pair(other, tstp);
        if tstp == MATSUBARA {
            return 0.0;
        }
        let b = self.size1 * self.size2;
        let mut err = 0.0;
        for i in 0..=tstp as usize {
            err += blkdist(
                &self.less.row(tstp)[i * b..(i + 1) * b],
                &other.less.row(tstp)[i * b..(i + 1) * b],
            );
        }
        err
    }

    /// Frobenius distance of the Matsubara component.
    pub fn distance_mat(&self, other: &ContourGreen) -> f64 {
        assert!(self.is_compatible(other));
        let b = self.size1 * self.size2;
        let mut err = 0.0;
        for n in 0..=self.ntau {
            err += blkdist(
                &self.mat.raw()[n * b..(n + 1) * b],
                &other.mat.raw()[n * b..(n + 1) * b],
            );
        }
        err
    }

    /// Total distance at `tstp`: Matsubara for `MATSUBARA`, otherwise
    /// ret + lmix + less rows (blueprint `distance(A, B, tstp)`).
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
}

impl GreenTimeStep {
    /// Same orbital dimensions (blueprint `getdims` comparison).
    pub fn equaldims(&self, other: &GreenTimeStep) -> bool {
        self.size1 == other.size1 && self.size2 == other.size2
    }

    /// Exact element-wise equality of all components.
    pub fn equal(&self, other: &GreenTimeStep) -> bool {
        self.is_compatible(other)
            && self.mat.equal(&other.mat)
            && self.ret.equal(&other.ret)
            && self.lmix.equal(&other.lmix)
            && self.less.equal(&other.less)
    }

    /// Same grid and orbital shape (tstp, ntau, sizes).
    pub fn is_compatible(&self, other: &GreenTimeStep) -> bool {
        self.tstp == other.tstp && self.ntau == other.ntau && self.equaldims(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::sign::Sign;

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
}
