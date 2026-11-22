//
// Project : KadanoffBaym
// File    : src/storage/mod.rs
// Purpose : container storage access: indexing, block copies,
//           submatrices, single-timestep extraction, and moving-window
//           preconditions
//

mod copyzone;
mod elemcpy;
mod indexing;
mod moving;
mod pseudo;
mod submatrix;
mod timestep;

pub(crate) use moving::check_moving_solver;
pub(crate) use pseudo::gtr_blk;
pub use submatrix::{set_submatrix, submatrix};
pub use timestep::{set_timestep_at, timestep_at};
