//
// Project : KadanoffBaym
// File    : src/io/lines.rs
// Purpose : line-wise parser of the plain-text section format: a
//           cursor over the input lines carrying the current line
//           number for error messages
//

use crate::complex::Complex;
use crate::io::IoError;

pub(crate) struct Lines<'a> {
    lines: std::str::Lines<'a>,
    n: usize,
}

impl<'a> Lines<'a> {
    pub(crate) fn new(text: &'a str) -> Lines<'a> {
        Lines {
            lines: text.lines(),
            n: 0,
        }
    }

    pub(crate) fn next(&mut self) -> Result<&'a str, IoError> {
        self.n += 1;
        self.lines.next().ok_or_else(|| IoError {
            line: self.n,
            msg: "unexpected end of file".into(),
        })
    }

    /// Skip any single line without checking (the blueprint's bare
    /// `readline(io)`, e.g. the `data  : ` marker).
    pub(crate) fn skip_any(&mut self) -> Result<(), IoError> {
        self.next()?;
        Ok(())
    }

    /// Skip a comment line (`#...`).
    pub(crate) fn skip_comment(&mut self) -> Result<(), IoError> {
        let l = self.next()?;
        if !l.starts_with('#') {
            return Err(IoError {
                line: self.n,
                msg: format!("expected a comment line, got `{l}`"),
            });
        }
        Ok(())
    }

    /// Skip a blank line.
    pub(crate) fn skip_blank(&mut self) -> Result<(), IoError> {
        let l = self.next()?;
        if !l.trim().is_empty() {
            return Err(IoError {
                line: self.n,
                msg: format!("expected a blank line, got `{l}`"),
            });
        }
        Ok(())
    }

    /// Skip an `index : ...` line.
    pub(crate) fn skip_index(&mut self) -> Result<(), IoError> {
        let l = self.next()?;
        if !l.starts_with("index") {
            return Err(IoError {
                line: self.n,
                msg: format!("expected an index line, got `{l}`"),
            });
        }
        Ok(())
    }

    /// Read a `key : value` line, return the value token.
    pub(crate) fn kv(&mut self) -> Result<&'a str, IoError> {
        let l = self.next()?;
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() < 3 || f[1] != ":" {
            return Err(IoError {
                line: self.n,
                msg: format!("expected `key : value`, got `{l}`"),
            });
        }
        Ok(f[2])
    }

    pub(crate) fn kv_i64(&mut self) -> Result<i64, IoError> {
        self.kv()?.parse().map_err(|_| IoError {
            line: self.n,
            msg: "bad integer value".into(),
        })
    }

    pub(crate) fn kv_f64(&mut self) -> Result<f64, IoError> {
        self.kv()?.parse().map_err(|_| IoError {
            line: self.n,
            msg: "bad float value".into(),
        })
    }

    /// Read one element line `n m re [im]` and store it at `(n-1, m-1)`
    /// of the block (column-major incoming order).
    pub(crate) fn element(
        &mut self,
        block: &mut [Complex],
        s1: usize,
        s2: usize,
    ) -> Result<(), IoError> {
        let l = self.next()?;
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() < 3 {
            return Err(IoError {
                line: self.n,
                msg: format!("bad element line `{l}`"),
            });
        }
        let n: usize = f[0].parse().map_err(|_| IoError {
            line: self.n,
            msg: "bad row index".into(),
        })?;
        let m: usize = f[1].parse().map_err(|_| IoError {
            line: self.n,
            msg: "bad column index".into(),
        })?;
        let re: f64 = f[2].parse().map_err(|_| IoError {
            line: self.n,
            msg: "bad real part".into(),
        })?;
        let im = if f.len() >= 4 {
            f[3].parse().map_err(|_| IoError {
                line: self.n,
                msg: "bad imaginary part".into(),
            })?
        } else {
            0.0
        };
        if n < 1 || n > s1 || m < 1 || m > s2 {
            return Err(IoError {
                line: self.n,
                msg: format!("element index ({n},{m}) out of bounds"),
            });
        }
        block[(n - 1) * s2 + (m - 1)] = Complex::new(re, im);
        Ok(())
    }

    pub(crate) fn block(&mut self, s1: usize, s2: usize) -> Result<Vec<Complex>, IoError> {
        self.skip_index()?;
        let mut blk = vec![Complex::ZERO; s1 * s2];
        for _m in 0..s2 {
            for _n in 0..s1 {
                self.element(&mut blk, s1, s2)?;
            }
        }
        Ok(blk)
    }
}
