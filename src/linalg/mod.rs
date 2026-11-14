//
// Project : KadanoffBaym
// File    : src/linalg/mod.rs
// Purpose : hand-written dense linear algebra (no crates): matrices, LU,
//           matrix exponential, Hermitian eigendecomposition, block systems
//

pub mod eigh;
pub mod expm;
pub mod linalg_block;
pub mod lu;
pub mod mat;

pub use eigh::eigh;
pub use expm::expm;
pub use linalg_block::{
    solve_block_left, solve_block_left_single, solve_block_right, solve_block_right_single,
    solve_left, solve_right, BlockSolver,
};
pub use lu::{lu_factor, lu_solve, lu_solve_mat, mat_inv};
pub use mat::{scale_c, DenseMat};
