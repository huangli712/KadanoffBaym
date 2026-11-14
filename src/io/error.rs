// Project : KadanoffBaym
// File    : src/io/error.rs
// Purpose : parse/IO error type with the 1-based line number

use std::error::Error;
use std::fmt::{Display, Formatter, Result};

/// Parse/IO error with the 1-based line number.
#[derive(Clone, Debug, PartialEq)]
pub struct IoError {
    pub line: usize,
    pub msg: String,
}

impl Display for IoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "io: line {}: {}", self.line, self.msg)
    }
}

impl Error for IoError {}
