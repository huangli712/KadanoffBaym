// Project : KadanoffBaym
// File    : src/math/mod.rs
// Purpose : math utilities layer

mod bose;
mod deriv;
mod dos;
mod extrap;
mod fermi;
mod interp;

pub use dos::{BetheDos, GaussianDos, OhmicSymDos};
pub use extrap::MovingStageWorkspace;

pub use bose::bose;
pub use deriv::{deriv1_element, deriv1_matsubara, deriv2_element};
pub use extrap::{extrapolate_timestep, extrapolate_timestep_moving};
pub use extrap::{extrapolate_timestep_cf, extrapolate_timestep_of_order};
pub use interp::interpolation;

pub(crate) use bose::bose_tau;
pub(crate) use fermi::{fermi, fermi_tau};
