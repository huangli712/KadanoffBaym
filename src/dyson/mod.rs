//
// Project : KadanoffBaym
// File    : src/dyson/mod.rs
// Purpose : the Dyson equation family (mat / start / timestep stages,
//           moving-window, NESS, pseudo-particle); the shared Matsubara
//           iteration helpers (MatsubaraMethod, the fixpoint/CG
//           drivers) live in `crate::base` and are
//           reused by the VIE2 family; `free_mat` stays in `mat`
//

mod check;
mod full;
mod mat;
mod moving;
mod pseudo;
mod start;
mod steady;
mod timestep;
mod workspace;

pub use steady::DysonBathType;
pub use workspace::{DysonWorkspace, MovingDysonWorkspace, PseudoDysonWorkspace};

pub use full::{dyson, dyson_timestep, dyson_timestep_ws};
pub use mat::{dyson_mat, dyson_mat_cg, dyson_mat_fixpoint, dyson_mat_kernel, free_mat};
pub use moving::dyson_moving_timestep;
pub use pseudo::{pseudo_dyson, pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep};
pub use start::{dyson_start, dyson_start_lmix, dyson_start_ret};
pub use steady::{dyson_steady, sigma_bath};
pub use timestep::dyson_less_core;

pub(crate) use start::dyson_start_less;
