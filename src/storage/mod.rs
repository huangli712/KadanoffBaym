//
// Project : KadanoffBaym
// File    : src/storage/mod.rs
// Purpose : container storage access: indexing, block copies,
//           submatrices, and single-timestep extraction
//

mod copyzone;
mod elemcpy;
mod indexing;
mod pseudo;
mod submatrix;
mod timestep;

pub use submatrix::{set_submatrix, submatrix};
pub use timestep::{set_timestep_at, timestep_at};

pub(crate) use pseudo::gtr_blk;
