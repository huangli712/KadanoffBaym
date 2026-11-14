// Project : KadanoffBaym
// File    : src/mat.rs
// Purpose : small dense complex matrices (row-major, no deps)

use crate::complex::c64::C64;

/// Small dense complex matrix, row-major `Vec<C64>`. Rectangular shapes
/// are supported; orbital blocks are square (`size()` asserts this).
#[derive(Clone, PartialEq, Debug)]
pub struct DenseMat {
    rows: usize,
    cols: usize,
    pub(crate) data: Vec<C64>,
}

impl DenseMat {
    /// Raw row-major element slice.
    #[inline]
    pub fn raw(&self) -> &[C64] {
        &self.data
    }

    /// Mutable raw row-major element slice.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [C64] {
        &mut self.data
    }

    /// Zero **square** matrix.
    pub fn zeros(size: usize) -> DenseMat {
        DenseMat::zeros_shape(size, size)
    }

    /// Zero matrix of the given shape (row-major).
    pub fn zeros_shape(rows: usize, cols: usize) -> DenseMat {
        assert!(rows >= 1 && cols >= 1, "DenseMat shape must be >= 1, got {rows}x{cols}");
        DenseMat {
            rows,
            cols,
            data: vec![C64::ZERO; rows * cols],
        }
    }

    /// Identity matrix of the given size.
    pub fn identity(size: usize) -> DenseMat {
        let mut m = DenseMat::zeros(size);
        for i in 0..size {
            m.data[i * (size + 1)] = C64::ONE;
        }
        m
    }

    /// Construct from a row-major vector (`rows * cols` elements).
    pub fn from_vec(rows: usize, cols: usize, data: Vec<C64>) -> DenseMat {
        assert_eq!(data.len(), rows * cols, "DenseMat::from_vec length mismatch");
        DenseMat { rows, cols, data }
    }

    /// Construct from rows (all rows share the same length).
    pub fn from_rows(rows: &[Vec<C64>]) -> DenseMat {
        let nr = rows.len();
        assert!(nr >= 1, "DenseMat needs at least one row");
        let nc = rows[0].len();
        assert!(nc >= 1, "DenseMat needs at least one column");
        let mut data = Vec::with_capacity(nr * nc);
        for r in rows {
            assert_eq!(r.len(), nc, "DenseMat::from_rows ragged input");
            data.extend_from_slice(r);
        }
        DenseMat { rows: nr, cols: nc, data }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Dimension of a square matrix (asserts squareness).
    #[inline]
    pub fn size(&self) -> usize {
        assert_eq!(self.rows, self.cols, "DenseMat::size on non-square matrix");
        self.rows
    }

    /// Number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Element access `(i, j)` (0-based, row-major).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> C64 {
        self.data[i * self.cols + j]
    }

    /// Mutable element access `(i, j)`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut C64 {
        &mut self.data[i * self.cols + j]
    }

    /// Set element `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: C64) {
        self.data[i * self.cols + j] = v;
    }

    /// Borrow row `i` as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[C64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Overwrite all elements from `src` (shapes must match).
    pub fn set_from(&mut self, src: &DenseMat) {
        assert_eq!((self.rows, self.cols), (src.rows, src.cols), "DenseMat::set_from shape mismatch");
        self.data.copy_from_slice(&src.data);
    }

    /// Frobenius norm.
    pub fn fro_norm(&self) -> f64 {
        self.fro_norm_sqr().sqrt()
    }

    /// Squared Frobenius norm.
    pub fn fro_norm_sqr(&self) -> f64 {
        self.data.iter().map(|z| z.abs_sqr()).sum()
    }

    /// Matrix product `self * other`.
    pub fn mat_mul(&self, other: &DenseMat) -> DenseMat {
        assert_eq!(self.cols, other.rows, "DenseMat::mat_mul shape mismatch");
        let (n, k, m) = (self.rows, self.cols, other.cols);
        let mut out = DenseMat::zeros_shape(n, m);
        for i in 0..n {
            for p in 0..k {
                let a = self.get(i, p);
                if a == C64::ZERO {
                    continue;
                }
                let orow = other.row(p);
                let orow = &orow[..m];
                for (j, b) in orow.iter().enumerate() {
                    let v = a * *b;
                    *out.get_mut(i, j) += v;
                }
            }
        }
        out
    }

    /// Matrix product `self * other.adjoint()`.
    pub fn mat_mul_adjoint(&self, other: &DenseMat) -> DenseMat {
        self.mat_mul(&other.adjoint())
    }

    /// Matrix product `self.adjoint() * other`.
    pub fn adjoint_mul(&self, other: &DenseMat) -> DenseMat {
        self.adjoint().mat_mul(other)
    }

    /// Conjugate transpose into a new matrix (`cols x rows`).
    pub fn adjoint(&self) -> DenseMat {
        let mut out = DenseMat::zeros_shape(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j).conj());
            }
        }
        out
    }

    /// In-place conjugate transpose into an existing matrix of the
    /// transposed shape (blueprint `adjoint!`).
    pub fn adjoint_into(&self, out: &mut DenseMat) {
        assert_eq!((out.rows, out.cols), (self.cols, self.rows), "adjoint_into shape mismatch");
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j).conj());
            }
        }
    }

    /// Trace of a square matrix.
    pub fn trace(&self) -> C64 {
        assert_eq!(self.rows, self.cols, "trace on non-square matrix");
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    /// In-place scaling by a real factor.
    pub fn scale(&mut self, s: f64) {
        for z in &mut self.data {
            *z = z.scale(s);
        }
    }

    /// In-place `self += alpha * other`.
    pub fn axpy(&mut self, alpha: f64, other: &DenseMat) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "DenseMat::axpy shape mismatch");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b.scale(alpha);
        }
    }

    /// Whether the square matrix is Hermitian within `tol`.
    pub fn is_hermitian(&self, tol: f64) -> bool {
        assert_eq!(self.rows, self.cols, "is_hermitian on non-square matrix");
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i).conj()).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m2() -> DenseMat {
        DenseMat::from_rows(&[
            vec![C64::new(1.0, 1.0), C64::new(2.0, 0.0)],
            vec![C64::new(0.0, -1.0), C64::new(3.0, 2.0)],
        ])
    }

    #[test]
    fn constructors_and_access() {
        let z = DenseMat::zeros(3);
        assert_eq!(z.len(), 9);
        assert_eq!(z.get(2, 2), C64::ZERO);
        let id = DenseMat::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(id.get(i, j), if i == j { C64::ONE } else { C64::ZERO });
            }
        }
        assert_eq!(id.trace(), C64::new(3.0, 0.0));
        assert_eq!(m2().size(), 2);
        assert_eq!(m2().row(0)[1], C64::new(2.0, 0.0));
        // rectangular shapes
        let r = DenseMat::zeros_shape(2, 3);
        assert_eq!((r.nrows(), r.ncols()), (2, 3));
        assert_eq!(r.len(), 6);
        assert_eq!(r.row(1).len(), 3);
    }

    #[test]
    fn rectangular_product() {
        let a = DenseMat::from_rows(&[vec![C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(0.0, 1.0)]]);
        let b = DenseMat::from_rows(&[
            vec![C64::new(1.0, 1.0), C64::new(0.0, 0.0)],
            vec![C64::new(2.0, -1.0), C64::new(1.0, 0.0)],
            vec![C64::new(0.0, 1.0), C64::new(3.0, 1.0)],
        ]);
        let c = a.mat_mul(&b); // 1x2
        assert_eq!((c.nrows(), c.ncols()), (1, 2));
        assert!((c.get(0, 0) - C64::new(4.0, -1.0)).abs() < 1e-15);
        assert!((c.get(0, 1) - C64::new(1.0, 3.0)).abs() < 1e-15);
        // adjoint of 1x3 -> 3x1
        let at = a.adjoint();
        assert_eq!((at.nrows(), at.ncols()), (3, 1));
        assert_eq!(at.get(2, 0), C64::new(0.0, -1.0));
        // adjoint_into roundtrip
        let mut out = DenseMat::zeros_shape(2, 3);
        b.adjoint_into(&mut out);
        assert_eq!(out.get(0, 1), b.get(1, 0).conj());
    }

    #[test]
    fn products_match_hand_computation() {
        let a = m2();
        let b = DenseMat::from_rows(&[
            vec![C64::new(0.0, 1.0), C64::new(1.0, 0.0)],
            vec![C64::new(2.0, 2.0), C64::new(-1.0, 0.0)],
        ]);
        // a * b (hand computed)
        let ab = a.mat_mul(&b);
        assert!((ab.get(0, 0) - C64::new(3.0, 5.0)).abs() < 1e-15);
        assert!((ab.get(0, 1) - C64::new(-1.0, 1.0)).abs() < 1e-15);
        assert!((ab.get(1, 0) - C64::new(3.0, 10.0)).abs() < 1e-15);
        assert!((ab.get(1, 1) - C64::new(-3.0, -3.0)).abs() < 1e-15);
        // (a*b).adjoint == b.adjoint * a.adjoint
        let lhs = ab.adjoint();
        let rhs = b.adjoint().mat_mul(&a.adjoint());
        assert!((lhs.get(1, 0) - rhs.get(1, 0)).abs() < 1e-15);
        assert!((lhs.fro_norm() - rhs.fro_norm()).abs() < 1e-15);
        // mat_mul_adjoint / adjoint_mul consistency
        let ba = a.mat_mul_adjoint(&b);
        let ba2 = a.mat_mul(&b.adjoint());
        assert!((ba.get(0, 0) - ba2.get(0, 0)).abs() < 1e-15);
        let aa = a.adjoint_mul(&b);
        let aa2 = a.adjoint().mat_mul(&b);
        assert!((aa.get(1, 1) - aa2.get(1, 1)).abs() < 1e-15);
    }

    #[test]
    fn scale_axpy_norm() {
        let mut a = m2();
        let b = a.clone();
        a.scale(2.0);
        assert!((a.get(0, 0) - C64::new(2.0, 2.0)).abs() < 1e-15);
        a.axpy(-0.5, &b);
        assert!((a.get(0, 0) - C64::new(1.5, 1.5)).abs() < 1e-15);
        // fro norm of identity 2x2 = sqrt(2)
        assert!((DenseMat::identity(2).fro_norm() - 2.0f64.sqrt()).abs() < 1e-15);
        let h = DenseMat::from_rows(&[
            vec![C64::new(1.0, 0.0), C64::new(2.0, 1.0)],
            vec![C64::new(2.0, -1.0), C64::new(3.0, 0.0)],
        ]);
        assert!(h.is_hermitian(1e-14));
    }

    #[test]
    fn set_from_and_identity_mul() {
        let mut a = DenseMat::zeros(2);
        a.set_from(&m2());
        assert_eq!(a, m2());
        let id = DenseMat::identity(2);
        assert!((id.mat_mul(&a).get(1, 0) - a.get(1, 0)).abs() < 1e-16);
        assert!((a.mat_mul(&id).fro_norm() - a.fro_norm()).abs() < 1e-15);
    }
}

/// Return `w * m` for a complex scalar `w` (leaves `m` unchanged).
pub fn scale_c(m: &DenseMat, w: C64) -> DenseMat {
    DenseMat::from_vec(
        m.nrows(),
        m.ncols(),
        m.raw().iter().map(|&z| z * w).collect(),
    )
}

/// `dst += w * src` for a complex scalar `w`.
pub fn axpy_c(dst: &mut DenseMat, w: C64, src: &DenseMat) {
    for (dz, sz) in dst.raw_mut().iter_mut().zip(src.raw()) {
        *dz += w * *sz;
    }
}
