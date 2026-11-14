//
// Project : KadanoffBaym
// File    : src/core/mod.rs
// Purpose : the Green's-function data model: contour, storage layout,
//           the four container flavors (standard, moving, NESS,
//           pseudo-particle) with their views, and the container-level
//           operations (scaling/multiplication, arithmetic, block
//           copies, distance measures)
//

pub mod algebra;
pub mod blocks;
pub mod contour;
pub mod from_mat;
pub mod green;
pub mod indexing;
pub mod layout;
pub mod moving;
pub mod ness;
pub mod ops;
pub mod properties;
pub mod pseudo;

pub use contour::{Contour, Sign};
pub use green::{ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction};
pub use layout::{Layout, MATSUBARA};

// flat re-exports of the operation modules: `crate::core::smul_at`, ...
pub use algebra::*;
pub use blocks::*;
pub use ops::*;
pub use properties::*;
