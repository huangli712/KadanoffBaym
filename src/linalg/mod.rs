//
// Project : KadanoffBaym
// File    : src/linalg/mod.rs
// Purpose : hand-written dense linear algebra (no crates): matrices, LU,
//           matrix exponential, Hermitian eigendecomposition, block systems
//

mod eigh;
mod expm;
mod linalg_block;
mod lu;
mod mat;

pub use eigh::eigh;
pub use expm::expm;
pub use linalg_block::{solve_block_right, solve_left, solve_right, BlockSolver};
pub(crate) use linalg_block::{solve_block_left, solve_block_left_single, solve_block_right_single};
pub use lu::mat_inv;
pub(crate) use lu::{lu_factor, lu_solve, lu_solve_mat};
pub use mat::{scale_c, DenseMat};

pub(crate) use mat::axpy_c;
