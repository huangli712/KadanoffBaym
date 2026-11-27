// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root: the crate documentation, the module tree and the
//           version constant
//
// Numerical results must match NESSi 2.0.0 point by point.

//!
//! KadanoffBaym
//!
//! `KadanoffBaym` is a Rust crate for simulating nonequilibrium
//! strongly correlated electron systems. It provides application
//! programming interfaces to manipulate the nonequilibrium Green's
//! functions defined on the L-shaped Kadanoff-Baym contour, including:
//!
//! * contour-ordered containers and their element-wise operations
//!   (`types`, `storage`, `algebra`, `variants`)
//! * convolutions between two contour-ordered Green's functions, and
//!   the response of an observable to a perturbation (`convolution`)
//! * shared Matsubara-axis solver components (`base`)
//! * Dyson and Volterra integro-differential equation solvers (`dyson`)
//! * second-order Volterra integral equation solvers (`vie2`)
//! * equilibrium and steady-state solvers (`equilibrium::from_dos_steady`,
//!   `dyson::dyson_steady`)
//! * Fourier tools and contour/steady conversion (`fourier`)
//! * quadrature weights and integrators (`weights`, `math`)
//! * complex dense linear algebra (`linalg`)
//! * plain-text IO (`io`) and observables (`observables`)
//!
//! The crate is a Rust reimplementation of the `NESSi` (the
//! Non-Equilibrium Systems Simulation package) code by Martin Eckstein
//! *et al*; the numerical results match NESSi 2.0.0 point by point.
//!

#![forbid(unsafe_code)]

pub mod algebra;
pub mod base;
pub mod complex;
pub mod convolution;
pub mod dyson;
pub mod equilibrium;
pub mod fourier;
pub mod io;
pub mod linalg;
pub mod math;
pub mod observables;
pub mod storage;
pub mod types;
pub mod variants;
pub mod vie2;
pub mod weights;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
