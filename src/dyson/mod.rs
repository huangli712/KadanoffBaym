//
// Project : KadanoffBaym
// File    : src/dyson/mod.rs
// Purpose : the Dyson equation family (mat / start / timestep stages,
//           moving-window, NESS, pseudo-particle); the shared Matsubara
//           iteration helpers (MatsubaraMethod, free_mat, mat_inner)
//           live in `mat` and are reused by the VIE2 family
//

mod full;
mod mat;
mod moving;
mod pseudo;
mod start;
mod steady;
mod timestep;

pub use mat::MatsubaraMethod;
pub use steady::DysonBathType;

pub use full::dyson;
pub use mat::{dyson_mat, dyson_mat_cg, dyson_mat_fixpoint, dyson_mat_fourier};
pub use mat::{dyson_mat_kernel, free_mat, mat_inner};
pub use moving::dyson_timestep_moving;
pub use pseudo::{pseudo_dyson, pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep};
pub use start::{dyson_start, dyson_start_lmix, dyson_start_ret};
pub use steady::{dyson_steady, sigma_bath};
pub use timestep::{DysonWorkspace, dyson_less_core, dyson_timestep, dyson_timestep_ws};
pub use timestep::{dyson_timestep_less, dyson_timestep_lmix, dyson_timestep_ret};

pub(crate) use pseudo::gtr_blk;
pub(crate) use start::dyson_start_less;
pub(crate) use steady::fermi_cut;
