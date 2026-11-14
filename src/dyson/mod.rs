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
mod ness;
mod pseudo;
mod start;
mod timestep;

pub use full::dyson;
pub use mat::{dyson_mat, dyson_mat_fourier, free_mat, mat_inner, MatsubaraMethod};
pub use moving::dyson_timestep_moving;
pub use ness::{dyson_ness, sigma_bath, DysonBathType};
pub use pseudo::pseudo_dyson;
pub use start::{dyson_start, dyson_start_lmix, dyson_start_ret};
pub use timestep::{dyson_less_core, dyson_timestep, dyson_timestep_less, dyson_timestep_ret};

pub(crate) use full::dyson_default;
pub(crate) use ness::fermi_cut;
pub(crate) use pseudo::gtr_blk;
