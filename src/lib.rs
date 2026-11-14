// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root and shared numeric type aliases
//
// Ricebook is the Julia blueprint (julia/src, frozen after R0); numerical
// results must match NESSi 2.0.0 point by point.
//!
//! KadanoffBaym
//!
//! The `KadanoffBaym` package is a state-of-the-art computational framework
//! for simulating the non-equilibrium strongly correlated electron systems.
//! It provides some useful application programming interfaces to manipulate
//! the non-equilibrium Green's functions defined on the 𝐿-shape Kadanoff-Baym
//! contour, including:
//!
//! * Basic integration and differentiation rules
//! * Basic operations for contour-ordered Green's functions
//! * Basic diagrammatic algorithms based on many-body perturbation theory
//! * Solve Volterra integral equations
//! * Solve Volterra integro-differential equations
//! * Convolution between two contour-ordered Green's functions
//!
//! This package is inspired by the `NESSi` (The Non-Equilibrium Systems
//! Simulation package) code, which was developed and maintained by Martin
//! Eckstein *et al*. Actually, it can be regarded as a replacement of the
//! `NESSi` package for those peoples who don't like or aren't familiar
//! with `C++`.
//!

#![forbid(unsafe_code)]

/// Real floating-point type used throughout the crate (`f64` only, D2).
pub type F64 = f64;

/// Integer type used for contour indices and sizes (`i64` only, D2).
pub type I64 = i64;

pub mod basics;
pub mod complex;
pub mod convolution;
pub mod dyson;
pub mod equilibrium;
pub mod fourier;
pub mod io;
pub mod linalg;
pub mod math;
pub mod moving;
pub mod observables;
pub mod types;
pub mod vie2;
pub mod weights;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
