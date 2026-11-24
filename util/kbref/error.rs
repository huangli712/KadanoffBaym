// Project : KadanoffBaym
// File    : util/kbref/error.rs
// Purpose : parse/validation error of the kbref-v1 reader with the
//           file path and the 1-based line number
//

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Parse/validation error (`kbref: <path>:<line>: <msg>`).
#[derive(Clone, Debug, PartialEq)]
pub struct KbrefError {
    pub path: String,
    pub line: usize,
    pub msg: String,
}

impl Display for KbrefError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "kbref: {}:{}: {}", self.path, self.line, self.msg)
    }
}

impl Error for KbrefError {}
