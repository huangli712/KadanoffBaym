// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root and shared numeric type aliases
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
//!   (`types`, `basics`)
//! * convolutions between two contour-ordered Green's functions
//!   (`convolution`)
//! * Dyson and Volterra integro-differential equation solvers (`dyson`)
//! * second-order Volterra integral equation solvers (`vie2`)
//! * equilibrium and steady-state solvers (`equilibrium`, NESS2 in
//!   `basics::steady`)
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

pub mod basics;
pub mod complex;
pub mod convolution;
pub mod dyson;
pub mod equilibrium;
pub mod fourier;
pub mod io;
pub mod linalg;
pub mod math;
pub mod observables;
pub mod types;
pub mod vie2;
pub mod weights;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
