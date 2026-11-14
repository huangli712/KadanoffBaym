//
// Project : KadanoffBaym
// File    : src/basics/mod.rs
// Purpose : container-level basic operations: scaling/multiplication,
//           arithmetic, block copies, and distance measures
//

pub mod algebra;
pub mod blocks;
pub mod ops;
pub mod properties;

pub use algebra::{add, smul, smul_f64, sub, Algebra};
pub use blocks::{
    elemcpy_green_at, elemcpy_green_map, elemcpy_green_map_at, set_submatrix, submatrix, CopyZone,
};
pub use ops::{force_matsubara_hermitian, MatsubaraHermitian, Mode};
pub use properties::blkdist;
