//
// Project : KadanoffBaym
// File    : src/io/mod.rs
// Purpose : plain-text serialization of the core containers (standard,
//           moving, NESS, pseudo-particle), split by container flavor
//

use std::fmt;

/// Parse/IO error with the 1-based line number.
#[derive(Clone, Debug, PartialEq)]
pub struct IoError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "io: line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for IoError {}

/// C-style `%16.12f` (width 16, precision 12, space padded).
pub fn fmt_f12(x: f64) -> String {
    format!("{:16.12}", x)
}

pub mod green;
pub mod moving;
pub mod ness;
pub mod pseudo;

pub use green::*;
pub use moving::*;
pub use ness::*;
pub use pseudo::*;

