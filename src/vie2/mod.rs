//
// Project : KadanoffBaym
// File    : src/vie2/mod.rs
// Purpose : the VIE2 family ([1+F]G = Q, the GW W equation): mat /
//           start / timestep stages, singular kernels, moving-window,
//           pseudo-particle
//

mod full;
mod mat;
mod moving;
mod pseudo;
mod singular;
mod start;
mod timestep;

pub use singular::Vie2SingularWorkspace;
pub use timestep::Vie2Workspace;

pub use full::vie2;
pub use mat::{vie2_mat, vie2_mat_cg, vie2_mat_fixpoint, vie2_mat_fourier};
pub use moving::{MovingVie2Workspace, vie2_timestep_moving, vie2_timestep_moving_ws};
pub use pseudo::{
    PseudoVie2Workspace, pseudo_vie2, pseudo_vie2_mat, pseudo_vie2_start, pseudo_vie2_timestep,
    pseudo_vie2_timestep_ws,
};
pub use singular::vie2_timestep_singular;
pub use start::{vie2_start, vie2_start_impl};
pub use timestep::{vie2_timestep, vie2_timestep_ws};
pub use timestep::{vie2_timestep_less, vie2_timestep_lmix, vie2_timestep_ret};

pub(crate) use full::{check_vie2, check_vie2_order};
pub(crate) use start::{vie2_bootstrap, vie2_start_less, vie2_start_ret};
pub(crate) use timestep::{vie2_kernel_ret, vie2_less_core};
