// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root and shared numeric type aliases
//
// Ricebook is the Julia blueprint (julia/src, frozen after R0); numerical
// results must match NESSi 2.0.0 point by point.

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
pub mod kbref;
pub mod linalg;
pub mod math;
pub mod moving;
pub mod observables;
pub mod types;
pub mod vie2;
pub mod weights;

pub use kbref::{read_kbref, write_kbref, KbrefData, KbrefError, KbrefRow};

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
