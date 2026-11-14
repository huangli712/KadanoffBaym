//
// Project : KadanoffBaym
// File    : src/basics/mod.rs
// Purpose : container-level basic operations: scaling/multiplication,
//           arithmetic, block copies, and distance measures
//

mod algebra;
mod copyzone;
mod elemcpy;
mod indexing;
mod ops;
mod properties;
mod submatrix;

pub use copyzone::CopyZone;

pub use algebra::{add, smul, smul_f64, sub, Algebra};
pub use elemcpy::{elemcpy_green_at, elemcpy_green_map, elemcpy_green_map_at};
pub use indexing::{pseudo_set_timestep, pseudo_timestep, set_timestep_at, timestep_at};
pub use ops::{force_matsubara_hermitian, MatsubaraHermitian, Mode};
pub use properties::blkdist;
pub use submatrix::{set_submatrix, submatrix};
