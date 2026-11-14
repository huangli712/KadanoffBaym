//
// Project : KadanoffBaym
// File    : src/basics/blocks/mod.rs
// Purpose : orbital sub-block copies: zone selection, element copies,
//           and submatrix extraction
//
// Blueprint: julia/src/blocks/{copyzone,elemcpy,submatrix}.jl. The zone
// coordinates are ORBITAL indices (0-based here, 1-based in Julia).
// set_matrixelement (blocks/element.jl) is NESS-only and lands in R7.

pub mod copyzone;
pub mod elemcpy;
pub mod submatrix;

pub use copyzone::CopyZone;
pub use elemcpy::{elemcpy_green_at, elemcpy_green_map, elemcpy_green_map_at};
pub use submatrix::{set_submatrix, submatrix};

#[cfg(test)]
pub(crate) mod testutil {
    use crate::Complex;
    use crate::types::{Contour, ContourGreen, Sign};

    /// Deterministic pair of 2x2 greens on a 2x2x2 grid used by the
    /// block-copy tests.
    pub(crate) fn two_greens() -> (ContourGreen, ContourGreen) {
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
}
