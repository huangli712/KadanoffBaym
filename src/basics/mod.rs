//
// Project : KadanoffBaym
// File    : src/basics/mod.rs
// Purpose : container-level operations in two layers: the standard
//           containers (algebra, element access, in-place ops,
//           properties, block copies) and the other container
//           families (moving, pseudo-particle, steady state)
//

mod algebra;
mod copyzone;
mod elemcpy;
mod indexing;
mod moving;
mod mul;
mod ops;
mod properties;
mod pseudo;
mod steady;
mod submatrix;
mod timestep;

pub use algebra::Algebra;
pub use copyzone::CopyZone;
pub use mul::Mode;

pub use algebra::{add, smul, smul_f64, sub};
pub use elemcpy::{elemcpy_green_at, elemcpy_green_map, elemcpy_green_map_at};
pub use moving::{copy_timestep, distance_moving, distance_moving_at};
pub use moving::{extrapolate_timestep_moving, extrapolate_timestep_moving_order};
pub use moving::{forward, incr_step, init_from_function, init_from_green};
pub use moving::{init_from_green_herm, init_from_steady, lmul_step, rmul_step, smul_step};
pub use ops::set_t0_from_mat;
pub use properties::blkdist;
pub use pseudo::{pseudo_timestep, set_pseudo_timestep, set_t0_from_mat_pseudo};
pub use submatrix::{set_submatrix, submatrix};
pub use timestep::{set_timestep_at, timestep_at};

pub(crate) use moving::{check_hamiltonian_moving, check_moving_solver};
