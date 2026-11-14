//
// Project : KadanoffBaym
// File    : src/basics/submatrix.rs
// Purpose : orbital row/column selection into a new green, and back
//
// Blueprint: julia/src/blocks/submatrix.jl.

use crate::complex::Complex;
use crate::types::{Contour, ContourGreen};

use super::elemcpy::elemcpy_green_map;

/// Build a new `ContourGreen` containing the selected orbital rows and
/// columns of `G` (blueprint `submatrix`): `out[k, l] = G[rows[k], cols[l]]`.
pub fn submatrix(g: &ContourGreen, rows: &[i64], cols: &[i64]) -> ContourGreen {
    // The green container keeps no contour metadata (tmax/beta); only the
    // grid sizes are read back by `ContourGreen::new`, so placeholders do.
    let c = Contour::new(
        g.nt as i64,
        g.ntau as i64,
        rows.len() as i64,
        cols.len() as i64,
        1.0,
        1.0,
    );
    let mut out = ContourGreen::new(&c, g.sign, Complex::ZERO);
    let mut rows1 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols1 = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &cl in cols {
            rows1.push(r);
            cols1.push(cl);
        }
    }
    let mut rows2 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols2 = Vec::with_capacity(rows.len() * cols.len());
    for r in 0..rows.len() as i64 {
        for cl in 0..cols.len() as i64 {
            rows2.push(r);
            cols2.push(cl);
        }
    }
    elemcpy_green_map(&rows1, &cols1, g, &rows2, &cols2, &mut out);
    out
}

/// Write the selected orbital rows and columns of `X` into `G`
/// (blueprint `set_submatrix!`; `dst[rows[k], cols[l]] = src[k, l]`).
pub fn set_submatrix(g: &mut ContourGreen, rows: &[i64], cols: &[i64], x: &ContourGreen) {
    assert_eq!(x.size1, rows.len(), "row count mismatch");
    assert_eq!(x.size2, cols.len(), "col count mismatch");
    // dst index maps enumerate the cross product rows x cols
    let mut rows2 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols2 = Vec::with_capacity(rows.len() * cols.len());
    let mut rows1 = Vec::with_capacity(rows.len() * cols.len());
    let mut cols1 = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &c in cols {
            rows2.push(r);
            cols2.push(c);
        }
    }
    for r in 0..rows.len() as i64 {
        for c in 0..cols.len() as i64 {
            rows1.push(r);
            cols1.push(c);
        }
    }
    elemcpy_green_map(&rows1, &cols1, x, &rows2, &cols2, g);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Sign;

    fn two_greens() -> (ContourGreen, ContourGreen) {
        let c = Contour::new(2, 2, 2, 2, 1.0, 1.0);
        let mut src = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // deterministic fill: value = component tag + 10*i + j etc.
        for n in 0..=2 {
            src.set_mat(n, 0, 0, Complex::new(n as f64 + 100.0, 0.0));
            src.set_mat(n, 1, 1, Complex::new(n as f64 + 101.0, 0.0));
        }
        for tstp in 0..=2 {
            for j in 0..=tstp {
                let v = Complex::new(10.0 * tstp as f64 + j as f64, 0.5);
                src.set_ret(tstp, j, 0, 0, v);
                src.set_less(j, tstp, 0, 0, v);
            }
            for n in 0..=2 {
                src.set_lmix(
                    tstp,
                    n,
                    0,
                    0,
                    Complex::new(1000.0 + 10.0 * tstp as f64 + n as f64, 0.0),
                );
            }
        }
        let dst = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        (src, dst)
    }

    #[test]
    fn submatrix_selection() {
        let (src, _) = two_greens();
        // additionally fill (1,0) so the selection is visible
        let mut src = src;
        for n in 0..=2 {
            src.set_mat(n, 1, 0, Complex::new(-(n as f64), 0.0));
        }
        let sub = submatrix(&src, &[1, 0], &[0]);
        assert_eq!((sub.size1, sub.size2), (2, 1));
        // row order follows `rows`: row 0 of sub = orbital 1, row 1 = orbital 0
        assert_eq!(sub.get_mat(2, 0, 0), Complex::new(-2.0, 0.0));
        assert_eq!(sub.get_mat(2, 1, 0), Complex::new(102.0, 0.0));
        // set_submatrix back into a bigger object
        let mut back =
            ContourGreen::new(&Contour::new(2, 2, 2, 2, 1.0, 1.0), Sign::Fermi, Complex::ZERO);
        set_submatrix(&mut back, &[1, 0], &[0], &sub);
        assert_eq!(back.get_mat(2, 1, 0), Complex::new(-2.0, 0.0));
        assert_eq!(back.get_mat(2, 0, 0), Complex::new(102.0, 0.0));
    }
}
