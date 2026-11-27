//
// Project : KadanoffBaym
// File    : src/vie2/mod.rs
// Purpose : the VIE2 family ([1+F]G = Q, the GW W equation): mat /
//           start / timestep stages, singular kernels, moving-window,
//           pseudo-particle
//

mod check;
mod full;
mod mat;
mod moving;
mod pseudo;
mod singular;
mod start;
mod timestep;
mod workspace;

pub use singular::Vie2SingularWorkspace;
pub use workspace::{MovingVie2Workspace, PseudoVie2Workspace, Vie2Workspace};

pub use full::{vie2, vie2_timestep, vie2_timestep_ws};
pub use mat::vie2_mat;
pub use moving::vie2_moving_timestep;
pub use pseudo::{pseudo_vie2, pseudo_vie2_mat, pseudo_vie2_start, pseudo_vie2_timestep};
pub use singular::vie2_timestep_singular;
pub use start::vie2_start;
pub use timestep::{vie2_timestep_less, vie2_timestep_lmix, vie2_timestep_ret};
