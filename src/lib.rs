// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root and shared numeric type aliases
//
// Ricebook is the Julia blueprint (julia/src, frozen after R0); numerical
// results must match NESSi 2.0.0 point by point.

#![forbid(unsafe_code)]

/// Real floating-point type used throughout the crate (`f64` only, D2).
pub type F64 = f64;

/// Integer type used for contour indices and sizes (`i64` only, D2).
pub type I64 = i64;

pub mod basics;
pub mod complex;
pub mod convolution;
pub mod equilibrium;
pub mod fourier;
pub mod io;
pub mod kbref;
pub mod linalg;
pub mod math;
pub mod moving;
pub mod observables;
pub mod solvers;
pub mod types;
pub mod weights;

pub use kbref::{read_kbref, write_kbref, KbrefData, KbrefError, KbrefRow};

pub use linalg::{
    eigh, expm, lu_factor, lu_solve, lu_solve_mat, mat_inv, solve_block_left,
    solve_block_left_single, solve_block_right, solve_block_right_single, BlockSolver, DenseMat,
};

pub use complex::Complex;
pub use weights::Integrator;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_match_blueprint_widths() {
        assert_eq!(std::mem::size_of::<F64>(), 8);
        assert_eq!(std::mem::size_of::<I64>(), 8);
    }
}
