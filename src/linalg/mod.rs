//
// Project : KadanoffBaym
// File    : src/linalg/mod.rs
// Purpose : hand-written dense linear algebra (no crates): matrices, LU,
//           matrix exponential, Hermitian eigendecomposition, block systems
//

mod eigh;
mod expm;
mod linsolve;
mod lu;
mod mat;

pub use linsolve::BlockSolver;
pub use mat::DenseMat;

pub use eigh::eigh;
pub use expm::expm;
pub use linsolve::{solve_block_right, solve_left, solve_right};
pub use lu::mat_inv;
pub use mat::scale_c;

pub(crate) use linsolve::{solve_block_left, solve_block_left_single, solve_block_right_single};
pub(crate) use lu::{lu_factor, lu_solve, lu_solve_mat};
pub(crate) use mat::axpy_c;
