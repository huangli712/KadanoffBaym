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
mod ops;
mod properties;
mod pseudo;
mod steady;
mod submatrix;

// standard containers: block copies driven by zones and index maps
pub use copyzone::CopyZone;
pub use elemcpy::{elemcpy_green_at, elemcpy_green_map, elemcpy_green_map_at};
pub use submatrix::{set_submatrix, submatrix};

// standard containers: arithmetic, element access, in-place ops, properties
pub use algebra::{Algebra, add, smul, smul_f64, sub};
pub use indexing::{pseudo_set_timestep, pseudo_timestep, set_timestep_at, timestep_at};
pub use ops::force_matsubara_hermitian;
pub use ops::{MatsubaraHermitian, Mode};
pub use properties::blkdist;

// other container families
pub(crate) use moving::{check_hamiltonian_moving, check_moving_solver};
pub use moving::{
    copy_timestep, distance_moving, distance_moving_at, extrapolate_timestep_moving,
    extrapolate_timestep_moving_order, forward, incr_step, init_from_function, init_from_g,
    init_from_g_herm, init_from_ness, lmul_step, rmul_step, smul_step,
};
