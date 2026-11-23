//
// Project : KadanoffBaym
// File    : src/base/mod.rs
// Purpose : base components shared by the Dyson and VIE2 solver
//           families
//

mod check;
mod matsubara;
mod scratch;
mod seed;

pub(crate) use check::check_moving_solver;
pub(crate) use matsubara::matsubara_cg;
pub(crate) use matsubara::matsubara_fixpoint;
pub use matsubara::{MatsubaraMethod, mat_inner};
pub(crate) use scratch::{MovingBlockScratch, TimestepScratch};
pub(crate) use seed::MatsubaraSeed;
pub use seed::matsubara_seed;
