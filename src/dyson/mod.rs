//
// Project : KadanoffBaym
// File    : src/dyson/mod.rs
// Purpose : the Dyson equation family (mat / start / timestep stages,
//           moving-window, NESS, pseudo-particle); the shared Matsubara
//           iteration helpers (MatsubaraMethod, free_mat, mat_inner)
//           live in `mat` and are reused by the VIE2 family
//

pub mod full;
pub mod mat;
pub mod moving;
pub mod ness;
pub mod pseudo;
pub mod start;
pub mod timestep;

pub use full::dyson;
pub use mat::{dyson_mat, MatsubaraMethod};
