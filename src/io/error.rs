// Project : KadanoffBaym
// File    : src/io/error.rs
// Purpose : parse/IO error type with the 1-based line number

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
