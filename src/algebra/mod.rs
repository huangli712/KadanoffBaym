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

pub use core::{add, smul_c, sub};
pub use properties::frobenius_distance;

pub(crate) use mul::{lmul_cf_at, rmul_cf_at};
pub(crate) use ops::{memcpy_at, memset_at, smul_at};
