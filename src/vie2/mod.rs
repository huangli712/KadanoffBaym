//
// Project : KadanoffBaym
// File    : src/vie2/mod.rs
// Purpose : the VIE2 family ([1+F]G = Q, the GW W equation): mat /
//           start / timestep stages, singular kernels, moving-window,
//           pseudo-particle
//

pub mod full;
pub mod mat;
pub mod moving;
pub mod pseudo;
pub mod sin;
pub mod start;
pub mod timestep;

pub use full::vie2;
pub use mat::vie2_mat;
