//
// Project : KadanoffBaym
// File    : src/variants/mod.rs
// Purpose : container family operations: moving-window, pseudo-particle,
//           and steady-state containers
//

pub mod moving;
mod pseudo;
mod steady;

pub use moving::{
    MovingStageWorkspace, extrapolate_timestep_moving, extrapolate_timestep_moving_order,
    extrapolate_timestep_moving_order_ws, extrapolate_timestep_moving_ws, forward,
    init_from_function, init_from_green, init_from_green_herm, init_from_steady,
};
pub use moving::{
    copy_timestep, distance_moving, distance_moving_at, incr_step, lmul_step, rmul_step, smul_step,
};
pub use pseudo::{pseudo_timestep, set_pseudo_timestep, set_t0_from_mat_pseudo};

pub(crate) use pseudo::gtr_blk;
