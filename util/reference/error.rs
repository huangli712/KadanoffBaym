// Project : KadanoffBaym
// File    : util/reference/error.rs
// Purpose : parse/validation error of the ref-v1 reader with the
//           file path and the 1-based line number
//

use std::error::Error;
use std::fmt::Result as FmtResult;
use std::fmt::{Display, Formatter};

/// Parse/validation error (`ref: <path>:<line>: <msg>`).
#[derive(Clone, Debug, PartialEq)]
pub struct RefError {
    pub path: String,
    pub line: usize,
    pub msg: String,
}

impl Display for RefError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ref: {}:{}: {}", self.path, self.line, self.msg)
    }
}

impl Error for RefError {}
