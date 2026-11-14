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

pub mod algebra;
pub mod blocks;
pub mod c64;
pub mod convolution;
pub mod observables;
pub mod eigh;
pub mod equilibrium;
pub mod equilibrium_ness;
pub mod fourier;
pub mod ops;
pub mod properties;
pub mod solvers;
pub mod expm;
pub mod io;
pub mod kbref;
pub mod linalg_block;
pub mod lu;
pub mod mat;
pub mod math;
pub mod types;
pub mod weights;
pub mod moving;

pub use kbref::{read_kbref, write_kbref, KbrefData, KbrefError, KbrefRow};

pub use linalg_block::{solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single, BlockSolver};

pub use c64::C64;
pub use eigh::eigh;
pub use expm::expm;
pub use weights::Integrator;
pub use lu::{lu_factor, lu_solve, lu_solve_mat, mat_inv};
pub use mat::DenseMat;

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
