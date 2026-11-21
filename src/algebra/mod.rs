//
// Project : KadanoffBaym
// File    : src/algebra/mod.rs
// Purpose : container algebra: element-wise addition, scaling, and
//           multiplication
//

mod core;
mod mul;
mod ops;
mod properties;

pub use core::Algebra;
pub use core::{add, smul, smul_f64, sub};
pub use ops::set_t0_from_mat;
pub use properties::blkdist;
