//
// Project : KadanoffBaym
// File    : src/types/function.rs
// Purpose : single-time function

//!
//! *Remarks : Contour-Based Functions*
//!
//! It is a general matrix-valued function defined at the `Kadanoff-Baym`
//! contour:
//!
//! ```math
//! \begin{equation}
//! f_{\mathcal{C}} = f(t),
//! \end{equation}
//! ```
//!
//! where ``t \in \mathcal{C}_1 \cup \mathcal{C}_2 \cup \mathcal{C}_3``.
//!

use crate::complex::Complex;

use crate::types::MATSUBARA;

/// Single-time function : one matrix block per real-time
/// step `0..=nt` plus one Matsubara slot.
///
/// Storage: `data[0..=nt]` are the time rows, `data[nt + 1]` is the
/// Matsubara value (`cf[0]` <-> `get(MATSUBARA)`).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeFunction {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl TimeFunction {
    /// Constructor : `nt >= 0`, all elements set to `v`.
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> TimeFunction {
        let block = size1 * size2;
        TimeFunction {
            nt,
            size1,
            size2,
            block,
            data: vec![v; (nt + 2) * block],
        }
    }

    /// Zero-initialized constructor.
    pub fn zeros(nt: usize, size1: usize, size2: usize) -> TimeFunction {
        TimeFunction::new(nt, size1, size2, Complex::ZERO)
    }

    /// Borrow the whole raw storage (time rows then Matsubara slot).
    #[inline]
    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    /// Mutably borrow the whole raw storage.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// Borrow the block of `tstp` as a row-major slice.
    #[inline]
    pub fn block(&self, tstp: i64) -> &[Complex] {
        let k = self.idx(tstp);
        &self.data[k * self.block..(k + 1) * self.block]
    }

    /// Mutably borrow the block of `tstp`.
    #[inline]
    pub fn block_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = self.idx(tstp);
        &mut self.data[k * self.block..(k + 1) * self.block]
    }

    /// Element access at `(tstp, a, b)` (0-based; `MATSUBARA` for the
    /// Matsubara slot).
    #[inline]
    pub fn get(&self, tstp: i64, a: usize, b: usize) -> Complex {
        self.data[self.idx(tstp) * self.block + a * self.size2 + b]
    }

    /// Element write at `(tstp, a, b)`.
    #[inline]
    pub fn set(&mut self, tstp: i64, a: usize, b: usize, v: Complex) {
        let i = self.idx(tstp) * self.block + a * self.size2 + b;
        self.data[i] = v;
    }

    /// Block index for timestep `tstp` (`0..=nt` or `MATSUBARA`).
    #[inline]
    pub fn idx(&self, tstp: i64) -> usize {
        if tstp == MATSUBARA {
            self.nt + 1
        } else {
            assert!(
                (0..=self.nt as i64).contains(&tstp),
                "TimeFunction: tstp {tstp} out of range 0..={}",
                self.nt
            );
            tstp as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::{Adjoint, DenseMat};

    use crate::algebra::smul;

    fn to_vec(m: &DenseMat) -> Vec<Complex> {
        (0..m.nrows() * m.ncols())
            .map(|q| m.get(q / m.ncols(), q % m.ncols()))
            .collect()
    }

    #[test]
    fn time_function_matsubara_slot() {
        let mut cf = TimeFunction::new(3, 1, 1, Complex::ZERO);
        assert_eq!(cf.block(0).len(), 1);
        cf.set(MATSUBARA, 0, 0, Complex::new(9.0, 0.0));
        cf.set(2, 0, 0, Complex::new(2.0, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), Complex::new(9.0, 0.0));
        assert_eq!(cf.get(2, 0, 0), Complex::new(2.0, 0.0));
        assert_eq!(cf.block(MATSUBARA)[0], Complex::new(9.0, 0.0));
    }

    /// Element-wise arithmetic, left/right multiplication, and the
    /// matrix-element round trips.
    #[test]
    fn function_new_methods() {
        let nt = 1000i64;
        let rm = |v: &[f64], _n1: usize, _n2: usize| -> Vec<Complex> {
            v.iter().map(|x| Complex::new(*x, 0.0)).collect()
        };
        // fnorm: sqrt of summed block Frobenius differences over all slots
        let fnorm = |f: &TimeFunction, x: &[Complex]| -> f64 {
            let mut s = 0.0;
            for t in -1..=nt {
                for (a, b) in f.block(t).iter().zip(x) {
                    let d = *a - *b;
                    s += d.abs() * d.abs();
                }
            }
            s.sqrt()
        };
        let cf_of = |mat: &[Complex]| -> TimeFunction {
            let mut f = TimeFunction::new(nt as usize, 3, 5, Complex::ZERO);
            for t in -1..=nt {
                f.block_mut(t).copy_from_slice(mat);
            }
            f
        };

        // incr and smul
        let mat0 = rm(
            &[
                1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23.0, 12.0, 0.1, 0.0, 13.0, 231.0, 20.1, 2.1, 0.44,
            ],
            3,
            5,
        );
        let mat1 = rm(
            &[
                -1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3.0, 2.0, 6.1, 1.7, -3.0, -1.0, 2.1, -2.1, 8.44,
            ],
            3,
            5,
        );
        let alpha = Complex::new(0.5, 0.0);
        let mut f0 = cf_of(&mat0);
        let f1 = cf_of(&mat1);
        let mat3: Vec<Complex> = mat0
            .iter()
            .zip(&mat1)
            .map(|(a, b)| *a + alpha * *b)
            .collect();
        f0.incr_by(&smul(&f1, alpha));
        assert!(fnorm(&f0, &mat3) < 1e-5);
        let mat3: Vec<Complex> = mat3.iter().map(|z| alpha * *z).collect();
        f0.smul(alpha);
        assert!(fnorm(&f0, &mat3) < 1e-5);

        // multiply (5x5 constant blocks)
        let row0 = [0.1, 0.3, 0.7, 0.5, 1.1];
        let row1 = [2.0, 7.0, 4.0, 0.3, 2.0];
        let m0v: Vec<f64> = row0.iter().cloned().cycle().take(25).collect();
        let m1v: Vec<f64> = row1.iter().cloned().cycle().take(25).collect();
        let mat0 = rm(&m0v, 5, 5);
        let mat1 = rm(&m1v, 5, 5);
        let cf5 = |mat: &[Complex]| {
            let mut f = TimeFunction::new(nt as usize, 5, 5, Complex::ZERO);
            for t in -1..=nt {
                f.block_mut(t).copy_from_slice(mat);
            }
            f
        };
        let m0 = DenseMat::from_vec(5, 5, mat0.clone());
        let m1 = DenseMat::from_vec(5, 5, mat1.clone());
        // left: f0 <- f1 * f0
        let mut f0 = cf5(&mat0);
        let _f1 = cf5(&mat1);
        f0.lmul(&m1, Adjoint::No);
        assert!(fnorm(&f0, &to_vec(&m1.mat_mul(&m0))) < 1e-5);
        // right: f0 <- f0 * f1
        let mut f0 = cf5(&mat0);
        f0.rmul(&m1, Adjoint::No);
        assert!(fnorm(&f0, &to_vec(&m0.mat_mul(&m1))) < 1e-5);

        // matrixelements: element set/get round trips
        let mat0 = rm(
            &[
                1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23.0, 12.0, 0.1, 0.0, 13.0, 231.0, 20.1, 2.1, 0.44,
            ],
            3,
            5,
        );
        let mat1 = rm(
            &[
                -1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3.0, 2.0, 6.1, 1.7, -3.0, -1.0, 2.1, -2.1, 8.44,
            ],
            3,
            5,
        );
        let mut f0 = cf_of(&mat0);
        let f1 = cf_of(&mat1);
        let mut cur = mat0.clone();
        let mut ok = true;
        for i1 in 0..3usize {
            for i2 in 0..5usize {
                for j1 in 0..3usize {
                    for j2 in 0..5usize {
                        cur[i1 * 5 + i2] = mat1[j1 * 5 + j2];
                        for t in -1..=nt {
                            let v = f1.get(t, j1, j2);
                            f0.set(t, i1, i2, v);
                        }
                        ok &= fnorm(&f0, &cur) < 1e-5;
                    }
                }
            }
        }
        assert!(ok);
        // single-element extraction as a 1x1 function
        for i1 in 0..3usize {
            for i2 in 0..5usize {
                let v = f0.get(0, i1, i2);
                let mut f3 = TimeFunction::new(nt as usize, 1, 1, v);
                f3.set(MATSUBARA, 0, 0, v);
                assert!(fnorm(&f3, &[v]) < 1e-5);
            }
        }
    }

    #[test]
    fn cf_properties() {
        // ntime = 101 points -> nt = 100, so (nt + 2) blocks x 4 elements
        let cf1 = TimeFunction::new(100, 2, 2, Complex::ZERO);
        let cf2 = TimeFunction::new(100, 2, 2, Complex::ZERO);
        assert_eq!(cf1.raw().len(), 408);
        assert_eq!((cf1.size1, cf1.size2), (2, 2));
        assert_eq!(cf1.size1, cf1.size2);
        assert!(cf1.is_compatible(&cf2));
        assert!(cf1.distance(&cf2) < 1e-7);
    }

    #[test]
    fn cf_struct_constructors() {
        // ntime = 201 points -> nt = 200
        let cf1 = TimeFunction::new(200, 2, 2, Complex::ZERO);
        let cf2 = TimeFunction::new(200, 2, 2, Complex::ZERO);
        let cf3 = TimeFunction::new(200, 2, 2, Complex::ZERO);
        assert!(cf1.equal(&cf2));
        assert!(cf1.equal(&cf3));
        // different grid -> unequal
        let cf8 = TimeFunction::new(9, 2, 2, Complex::ZERO);
        assert!(!cf1.equal(&cf8));
    }
}
