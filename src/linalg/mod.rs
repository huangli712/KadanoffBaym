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
mod mul;

pub use mat::DenseMat;

pub use eigh::eigh;
pub use expm::expm;
pub use lu::mat_inv;
pub use mat::scale_c;

pub(crate) use linsolve::BlockSolver;

pub(crate) use linsolve::{solve_block_left, solve_block_left_single};
pub(crate) use linsolve::{solve_block_right, solve_block_right_single};
pub(crate) use lu::{lu_factor, lu_factor_into, lu_solve, lu_solve_mat, lu_solve_mat_into};
pub(crate) use mat::{axpy_c, axpy_c_flat};
pub(crate) use mul::{mul_acc, mul_acc_c, mul_acc_c2, mul_acc_c4};
pub(crate) use mul::{mul_acc_c2_u, mul_acc_c4_u, mul_acc_c8_u};
