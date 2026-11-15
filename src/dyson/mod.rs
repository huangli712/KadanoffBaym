//
// Project : KadanoffBaym
// File    : src/dyson/mod.rs
// Purpose : the Dyson equation family (mat / start / timestep stages,
//           moving-window, NESS, pseudo-particle); the shared Matsubara
//           iteration helpers (MatsubaraMethod, free_mat, mat_inner)
//           live in `mat` and are reused by the VIE2 family
//

pub(crate) mod full;
mod mat;
mod moving;
mod ness;
mod pseudo;
mod start;
mod timestep;

pub use full::dyson;
pub use mat::{
    MatsubaraMethod, dyson_mat, dyson_mat_cg, dyson_mat_fixpoint, dyson_mat_fourier,
    dyson_mat_kernel, free_mat, mat_inner,
};
pub use moving::dyson_timestep_moving;
pub use ness::{DysonBathType, dyson_ness, sigma_bath};
pub use pseudo::{pseudo_dyson, pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep};
pub use start::{dyson_start, dyson_start_lmix, dyson_start_ret};
pub use timestep::{dyson_less_core, dyson_timestep};

pub(crate) use ness::fermi_cut;
pub(crate) use pseudo::gtr_blk;
pub(crate) use start::dyson_start_less;
pub(crate) use timestep::{dyson_timestep_less, dyson_timestep_ret};
