//
// Project : KadanoffBaym
// File    : src/moving/mod.rs
// Purpose : the moving (truncated) window layer (blueprint
//           julia/src/moving/{window,layout}.jl, serial parts)
//

mod algebra;
mod layout;
mod window;

pub use algebra::{
    distance_moving, distance_moving_at, extrapolate_timestep_moving,
    extrapolate_timestep_moving_order, incr_step, lmul_step, rmul_step, smul_step,
};
pub use window::{
    copy_timestep, forward, init_from_function, init_from_g, init_from_g_herm, init_from_ness,
};

pub(crate) use layout::{check_hamiltonian_moving, check_moving_solver};
