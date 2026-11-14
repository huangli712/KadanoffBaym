// Project : KadanoffBaym
// File    : src/io.rs
// Purpose : plain-text serialization (blueprint: julia/src/io/{show,write,read}.jl)
//
// Format (identical to the Julia blueprint):
//
//   - Comment lines start with `#`.
//   - Parameter lines read `key : value`.
//   - Every block is introduced by an `index :%4i [%4i] ` line (with a
//     trailing space) and followed by column-major element lines
//     `%4i %4i %16.12f %16.12f` (row `n`, column `m`, real, imag).
//   - The ret/less components serialize the FULL time square; the
//     non-stored triangle is reconstructed via the hermiticity
//     relations (zeros when the blocks are non-square).
//   - `# Standard Contour-Ordered Green's Function:` + `sign` header,
//     then the mat/ret/lmix/less sections separated by blank lines.
//
// Index translation (decision D6): the file stores Julia-style grid
// sizes (points); the Rust objects use intervals. `ntime`/`ntau` in the
// file map to `nt + 1` / `ntau + 1` in memory.

use std::fmt;
use std::fs;

use crate::complex::c64::C64;
use crate::types::contour::{Contour, Sign};
use crate::types::pseudo::PseudoGreen;
use crate::types::green::{
    ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction,
};

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

/// C-style `%4i`.
fn fmt_i4(x: i64) -> String {
    format!("{:4}", x)
}

// ------------------------------------------------------------------
// writers (to strings; file wrappers at the bottom)
// ------------------------------------------------------------------

fn write_element_lines(out: &mut String, block: &[C64], s1: usize, s2: usize) {
    for m in 0..s2 {
        for n in 0..s1 {
            let v = block[n * s2 + m];
            out.push_str(&format!(
                "{} {} {} {}\n",
                fmt_i4(n as i64 + 1),
                fmt_i4(m as i64 + 1),
                fmt_f12(v.re),
                fmt_f12(v.im)
            ));
        }
    }
}

fn write_mat_section(out: &mut String, mat: &MatComp) {
    out.push_str("# Contour-Ordered Green's Function: Matsubara Component (G)\n");
    out.push_str("type  : mat\n");
    out.push_str(&format!("ntau  : {}\n", mat.ntau + 1));
    out.push_str(&format!("ndim1 : {}\n", mat.size1));
    out.push_str(&format!("ndim2 : {}\n", mat.size2));
    out.push_str("data  : \n");
    for i in 0..mat.nblocks() {
        out.push_str(&format!("index :{} \n", fmt_i4(i as i64 + 1)));
        write_element_lines(out, mat.block(i), mat.size1, mat.size2);
    }
}

fn write_ret_section(out: &mut String, ret: &RetComp) {
    out.push_str("# Contour-Ordered Green's Function: Retarded Component (G)\n");
    out.push_str("type  : ret\n");
    out.push_str(&format!("ntime : {}\n", ret.nt + 1));
    out.push_str(&format!("ndim1 : {}\n", ret.size1));
    out.push_str(&format!("ndim2 : {}\n", ret.size2));
    out.push_str("data  : \n");
    let square = ret.size1 == ret.size2;
    for i in 0..=ret.nt {
        for j in 0..=ret.nt {
            out.push_str(&format!(
                "index :{} {} \n",
                fmt_i4(j as i64 + 1),
                fmt_i4(i as i64 + 1)
            ));
            if j >= i || square {
                // value ret[j, i] (1-based) via the hermiticity fallback
                let blk = ret_block_value(ret, j as i64, i as i64);
                write_element_lines(out, &blk, ret.size1, ret.size2);
            } else {
                let zeros = vec![C64::ZERO; ret.size1 * ret.size2];
                write_element_lines(out, &zeros, ret.size1, ret.size2);
            }
        }
    }
}

/// Stored-or-reconstructed ret block (0-based `(tstp, j)`).
fn ret_block_value(ret: &RetComp, tstp: i64, j: i64) -> Vec<C64> {
    if tstp >= j {
        ret.block(ret.idx(tstp, j)).to_vec()
    } else {
        let src = ret.block(ret.idx(j, tstp));
        let (s1, s2) = (ret.size1, ret.size2);
        let mut out = vec![C64::ZERO; s1 * s2];
        for a in 0..s1 {
            for b in 0..s2 {
                out[a * s2 + b] = -src[b * s1 + a].conj();
            }
        }
        out
    }
}

fn write_lmix_section(out: &mut String, lmix: &LmixComp) {
    out.push_str("# Contour-Ordered Green's Function: Left-Mixing Component (G)\n");
    out.push_str("type  : lmix\n");
    out.push_str(&format!("ntime : {}\n", lmix.nt + 1));
    out.push_str(&format!("ntau  : {}\n", lmix.ntau + 1));
    out.push_str(&format!("ndim1 : {}\n", lmix.size1));
    out.push_str(&format!("ndim2 : {}\n", lmix.size2));
    out.push_str("data  : \n");
    for i in 0..=lmix.ntau {
        for j in 0..=lmix.nt {
            out.push_str(&format!(
                "index :{} {} \n",
                fmt_i4(j as i64 + 1),
                fmt_i4(i as i64 + 1)
            ));
            let k = lmix.idx(j as i64, i as i64);
            write_element_lines(out, lmix.block(k), lmix.size1, lmix.size2);
        }
    }
}

fn write_less_section(out: &mut String, less: &LessComp) {
    out.push_str("# Contour-Ordered Green's Function: Lesser Component (G)\n");
    out.push_str("type  : less\n");
    out.push_str(&format!("ntime : {}\n", less.nt + 1));
    out.push_str(&format!("ndim1 : {}\n", less.size1));
    out.push_str(&format!("ndim2 : {}\n", less.size2));
    out.push_str("data  : \n");
    let square = less.size1 == less.size2;
    for i in 0..=less.nt {
        for j in 0..=less.nt {
            out.push_str(&format!(
                "index :{} {} \n",
                fmt_i4(j as i64 + 1),
                fmt_i4(i as i64 + 1)
            ));
            if j <= i || square {
                let blk = less_block_value(less, j as i64, i as i64);
                write_element_lines(out, &blk, less.size1, less.size2);
            } else {
                let zeros = vec![C64::ZERO; less.size1 * less.size2];
                write_element_lines(out, &zeros, less.size1, less.size2);
            }
        }
    }
}

/// Stored-or-reconstructed less block (0-based `(i, tstp)`).
fn less_block_value(less: &LessComp, i: i64, tstp: i64) -> Vec<C64> {
    if i <= tstp {
        less.block(less.idx(i, tstp)).to_vec()
    } else {
        let src = less.block(less.idx(tstp, i));
        let (s1, s2) = (less.size1, less.size2);
        let mut out = vec![C64::ZERO; s1 * s2];
        for a in 0..s1 {
            for b in 0..s2 {
                out[a * s2 + b] = -src[b * s1 + a].conj();
            }
        }
        out
    }
}

/// Serialize a `ContourGreen` (blueprint `show(io, ℱ)`).
pub fn green_to_string(g: &ContourGreen) -> String {
    let mut out = String::new();
    out.push_str("# Standard Contour-Ordered Green's Function:\n");
    out.push_str(&format!("sign  : {}\n", g.sign.as_i64()));
    out.push('\n');
    write_mat_section(&mut out, &g.mat);
    out.push('\n');
    write_ret_section(&mut out, &g.ret);
    out.push('\n');
    write_lmix_section(&mut out, &g.lmix);
    out.push('\n');
    write_less_section(&mut out, &g.less);
    out.push('\n');
    out
}

/// Serialize a `GreenTimeStep` (blueprint `show(io, 𝒻)`).
pub fn timestep_to_string(f: &GreenTimeStep) -> String {
    let mut out = String::new();
    out.push_str("# Standard Contour-Ordered Green's Function:\n");
    out.push_str(&format!("sign  : {}\n", f.sign.as_i64()));
    out.push_str(&format!("tstp  : {}\n", f.tstp));
    out.push('\n');
    write_mat_section(&mut out, &f.mat);
    out.push('\n');
    write_ret_section(&mut out, &f.ret);
    out.push('\n');
    write_lmix_section(&mut out, &f.lmix);
    out.push('\n');
    write_less_section(&mut out, &f.less);
    out.push('\n');
    out
}

/// Serialize a `TimeFunction` (blueprint `show(io, Cf)`): the Matsubara
/// slot is the LAST block (`index = nt + 2`), like Julia's `data[end]`.
pub fn time_function_to_string(cf: &TimeFunction) -> String {
    let mut out = String::new();
    out.push_str("# Contour-Based Function\n");
    out.push_str(&format!("ntime : {}\n", cf.nt + 1));
    out.push_str(&format!("ndim1 : {}\n", cf.size1));
    out.push_str(&format!("ndim2 : {}\n", cf.size2));
    out.push_str("data  : \n");
    for i in 0..(cf.nt + 2) {
        out.push_str(&format!("index :{} \n", fmt_i4(i as i64 + 1)));
        let tstp = if i <= cf.nt { i as i64 } else { crate::types::MATSUBARA };
        write_element_lines(&mut out, cf.block(tstp), cf.size1, cf.size2);
    }
    out
}

/// Serialize a `Contour` (blueprint `show(io, Cn)`); the file keeps the
/// Julia point-count convention.
pub fn contour_to_string(c: &Contour) -> String {
    format!(
        "# Kadanoff-Baym Contour\nntime : {}\nntau  : {}\nndim1 : {}\nndim2 : {}\ntmax  : {}\nbeta  : {}\ndt    : {}\ndtau  : {}\n",
        c.nt + 1,
        c.ntau + 1,
        c.size1,
        c.size2,
        c.tmax,
        c.beta,
        c.dt,
        c.dtau
    )
}

// ------------------------------------------------------------------
// readers
// ------------------------------------------------------------------

struct Lines<'a> {
    lines: std::str::Lines<'a>,
    n: usize,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Lines<'a> {
        Lines {
            lines: text.lines(),
            n: 0,
        }
    }

    fn next(&mut self) -> Result<&'a str, IoError> {
        self.n += 1;
        self.lines
            .next()
            .ok_or_else(|| IoError {
                line: self.n,
                msg: "unexpected end of file".into(),
            })
    }

    /// Skip any single line without checking (the blueprint's bare
    /// `readline(io)`, e.g. the `data  : ` marker).
    fn skip_any(&mut self) -> Result<(), IoError> {
        self.next()?;
        Ok(())
    }

    /// Skip a comment line (`#...`).
    fn skip_comment(&mut self) -> Result<(), IoError> {
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
    fn skip_blank(&mut self) -> Result<(), IoError> {
        let l = self.next()?;
        if !l.trim().is_empty() {
            return Err(IoError {
                line: self.n,
                msg: format!("expected a blank line, got `{l}`"),
            });
        }
        Ok(())
    }

    /// Read a `key : value` line, return the value token.
    fn kv(&mut self) -> Result<&'a str, IoError> {
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

    fn kv_i64(&mut self) -> Result<i64, IoError> {
        self.kv()?
            .parse()
            .map_err(|_| IoError {
                line: self.n,
                msg: "bad integer value".into(),
            })
    }

    fn kv_f64(&mut self) -> Result<f64, IoError> {
        self.kv()?
            .parse()
            .map_err(|_| IoError {
                line: self.n,
                msg: "bad float value".into(),
            })
    }

    /// Skip an `index : ...` line.
    fn skip_index(&mut self) -> Result<(), IoError> {
        let l = self.next()?;
        if !l.starts_with("index") {
            return Err(IoError {
                line: self.n,
                msg: format!("expected an index line, got `{l}`"),
            });
        }
        Ok(())
    }

    /// Read one element line `n m re [im]` and store it at `(n-1, m-1)`
    /// of the block (column-major incoming order).
    fn element(&mut self, block: &mut [C64], s1: usize, s2: usize) -> Result<(), IoError> {
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
        block[(n - 1) * s2 + (m - 1)] = C64::new(re, im);
        Ok(())
    }

    fn block(&mut self, s1: usize, s2: usize) -> Result<Vec<C64>, IoError> {
        self.skip_index()?;
        let mut blk = vec![C64::ZERO; s1 * s2];
        for _m in 0..s2 {
            for _n in 0..s1 {
                self.element(&mut blk, s1, s2)?;
            }
        }
        Ok(blk)
    }
}

/// Parse a `Contour` (file keeps the Julia point-count convention).
pub fn contour_from_str(text: &str) -> Result<Contour, IoError> {
    let mut ls = Lines::new(text);
    ls.skip_comment()?;
    let ntime = ls.kv_i64()?;
    let ntau = ls.kv_i64()?;
    let size1 = ls.kv_i64()?;
    let size2 = ls.kv_i64()?;
    let tmax = ls.kv_f64()?;
    let beta = ls.kv_f64()?;
    let _dt = ls.kv_f64()?;
    let _dtau = ls.kv_f64()?;
    Ok(Contour::new(
        ntime - 1,
        ntau - 1,
        size1,
        size2,
        tmax,
        beta,
    ))
}

/// Parse a `TimeFunction`.
pub fn time_function_from_str(text: &str) -> Result<TimeFunction, IoError> {
    let mut ls = Lines::new(text);
    ls.skip_comment()?;
    let ntime = ls.kv_i64()?;
    let size1 = ls.kv_i64()? as usize;
    let size2 = ls.kv_i64()? as usize;
    ls.skip_any()?; // "data  : " marker
    let nt = (ntime - 1) as usize;
    let mut cf = TimeFunction::new(nt, size1, size2, C64::ZERO);
    for i in 0..(nt + 2) {
        let blk = ls.block(size1, size2)?;
        let tstp = if i <= nt { i as i64 } else { crate::types::MATSUBARA };
        cf.block_mut(tstp).copy_from_slice(&blk);
    }
    Ok(cf)
}

fn read_mat_section(ls: &mut Lines) -> Result<(usize, usize, MatComp), IoError> {
    ls.skip_comment()?;
    ls.kv()?; // type
    let ntau = (ls.kv_i64()? - 1) as usize;
    let s1 = ls.kv_i64()? as usize;
    let s2 = ls.kv_i64()? as usize;
    ls.skip_any()?; // "data  : " marker
    let mut mat = MatComp::new(ntau, s1, s2, C64::ZERO);
    for i in 0..=ntau {
        let blk = ls.block(s1, s2)?;
        mat.block_mut(i).copy_from_slice(&blk);
    }
    Ok((s1, s2, mat))
}

fn read_ret_section(ls: &mut Lines) -> Result<RetComp, IoError> {
    ls.skip_comment()?;
    ls.kv()?; // type
    let nt = (ls.kv_i64()? - 1) as usize;
    let s1 = ls.kv_i64()? as usize;
    let s2 = ls.kv_i64()? as usize;
    ls.skip_any()?; // "data  : " marker
    let mut ret = RetComp::new(nt, s1, s2, C64::ZERO);
    // Full square; only the stored triangle is kept.
    for i in 0..=nt {
        for j in 0..=nt {
            let blk = ls.block(s1, s2)?;
            if j >= i {
                let k = ret.idx(j as i64, i as i64);
                ret.block_mut(k).copy_from_slice(&blk);
            }
        }
    }
    Ok(ret)
}

fn read_lmix_section(ls: &mut Lines) -> Result<LmixComp, IoError> {
    ls.skip_comment()?;
    ls.kv()?; // type
    let nt = (ls.kv_i64()? - 1) as usize;
    let ntau = (ls.kv_i64()? - 1) as usize;
    let s1 = ls.kv_i64()? as usize;
    let s2 = ls.kv_i64()? as usize;
    ls.skip_any()?; // "data  : " marker
    let mut lmix = LmixComp::new(nt, ntau, s1, s2, C64::ZERO);
    for i in 0..=ntau {
        for j in 0..=nt {
            let blk = ls.block(s1, s2)?;
            let k = lmix.idx(j as i64, i as i64);
            lmix.block_mut(k).copy_from_slice(&blk);
        }
    }
    Ok(lmix)
}

fn read_less_section(ls: &mut Lines) -> Result<LessComp, IoError> {
    ls.skip_comment()?;
    ls.kv()?; // type
    let nt = (ls.kv_i64()? - 1) as usize;
    let s1 = ls.kv_i64()? as usize;
    let s2 = ls.kv_i64()? as usize;
    ls.skip_any()?; // "data  : " marker
    let mut less = LessComp::new(nt, s1, s2, C64::ZERO);
    for i in 0..=nt {
        for j in 0..=nt {
            let blk = ls.block(s1, s2)?;
            if j <= i {
                let k = less.idx(j as i64, i as i64);
                less.block_mut(k).copy_from_slice(&blk);
            }
        }
    }
    Ok(less)
}

/// Parse a `ContourGreen`.
pub fn green_from_str(text: &str) -> Result<ContourGreen, IoError> {
    let mut ls = Lines::new(text);
    ls.skip_comment()?;
    let sign = Sign::from_i64(ls.kv_i64()?);
    ls.skip_blank()?;
    let (_, _, mat) = read_mat_section(&mut ls)?;
    ls.skip_blank()?;
    let ret = read_ret_section(&mut ls)?;
    ls.skip_blank()?;
    let lmix = read_lmix_section(&mut ls)?;
    ls.skip_blank()?;
    let less = read_less_section(&mut ls)?;
    Ok(ContourGreen {
        sign,
        nt: ret.nt,
        ntau: mat.ntau,
        size1: mat.size1,
        size2: mat.size2,
        mat,
        ret,
        lmix,
        less,
    })
}

/// Parse a `GreenTimeStep`.
pub fn timestep_from_str(text: &str) -> Result<GreenTimeStep, IoError> {
    let mut ls = Lines::new(text);
    ls.skip_comment()?;
    let sign = Sign::from_i64(ls.kv_i64()?);
    let tstp = ls.kv_i64()? as usize;
    ls.skip_blank()?;
    let (_, _, mat) = read_mat_section(&mut ls)?;
    ls.skip_blank()?;
    let ret = read_ret_section(&mut ls)?;
    ls.skip_blank()?;
    let lmix = read_lmix_section(&mut ls)?;
    ls.skip_blank()?;
    let less = read_less_section(&mut ls)?;
    Ok(GreenTimeStep {
        sign,
        tstp,
        ntau: mat.ntau,
        size1: mat.size1,
        size2: mat.size2,
        mat,
        ret,
        lmix,
        less,
    })
}

/// Read the full text but overwrite only the Matsubara component and
/// the independent real-time domain through `tstp` of `g` (blueprint
/// `read!(io, ℱ, tstp)`).
pub fn green_partial_from_str(text: &str, g: &mut ContourGreen, tstp: i64) -> Result<(), IoError> {
    let src = green_from_str(text)?;
    assert!(src.is_compatible(g), "green_partial: incompatible greens");
    assert!((0..=g.nt as i64).contains(&tstp));
    g.mat.memcpy_from(&src.mat);
    for n in 0..=tstp {
        for j in 0..=n {
            let k = g.ret.idx(n, j);
            g.ret.block_mut(k).copy_from_slice(src.ret.block(k));
        }
        for i in 0..=n {
            let k = g.less.idx(i, n);
            g.less.block_mut(k).copy_from_slice(src.less.block(k));
        }
        for m in 0..=(g.ntau as i64) {
            let k = g.lmix.idx(n, m);
            g.lmix.block_mut(k).copy_from_slice(src.lmix.block(k));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------
// file wrappers
// ------------------------------------------------------------------

pub fn write_green(path: &str, g: &ContourGreen) -> std::io::Result<()> {
    fs::write(path, green_to_string(g))
}

pub fn read_green(path: &str) -> Result<ContourGreen, IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("cannot read {path}: {e}"),
    })?;
    green_from_str(&text)
}

pub fn write_timestep(path: &str, f: &GreenTimeStep) -> std::io::Result<()> {
    fs::write(path, timestep_to_string(f))
}

pub fn read_timestep(path: &str) -> Result<GreenTimeStep, IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("cannot read {path}: {e}"),
    })?;
    timestep_from_str(&text)
}

pub fn write_time_function(path: &str, cf: &TimeFunction) -> std::io::Result<()> {
    fs::write(path, time_function_to_string(cf))
}

pub fn read_time_function(path: &str) -> Result<TimeFunction, IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("cannot read {path}: {e}"),
    })?;
    time_function_from_str(&text)
}

pub fn write_contour(path: &str, c: &Contour) -> std::io::Result<()> {
    fs::write(path, contour_to_string(c))
}

pub fn read_contour(path: &str) -> Result<Contour, IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("cannot read {path}: {e}"),
    })?;
    contour_from_str(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::{Contour, Sign};
    use crate::types::pseudo::PseudoGreen;
    use crate::types::MATSUBARA;

    fn demo_green() -> ContourGreen {
        // all values are exactly representable with 12 decimals (the
        // writer format), so round trips are byte- and value-exact
        let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut v = 1.0;
        for n in 0..=4 {
            g.set_mat(n, 0, 0, C64::new(v, -v / 2.0));
            v += 1.0;
        }
        for i in 0..=3 {
            for j in 0..=i {
                g.set_ret(i, j, 1, 0, C64::new(i as f64, j as f64));
                g.set_less(j, i, 0, 1, C64::new(-(i as f64), j as f64 * 0.25));
            }
        }
        for t in 0..=3 {
            for m in 0..=4 {
                g.set_lmix(t, m, 1, 1, C64::new(t as f64, m as f64));
            }
        }
        g
    }

    #[test]
    fn green_round_trip_exact() {
        let g = demo_green();
        let text = green_to_string(&g);
        let back = green_from_str(&text).expect("parse");
        assert!(back.equal(&g));
        // byte-identical rewrite
        assert_eq!(green_to_string(&back), text);
    }

    #[test]
    fn hermiticity_reconstruction_survives_round_trip() {
        let mut g = demo_green();
        // upper-triangle ret values are reconstructed on write and
        // discarded on read; the stored data still round-trips
        g.set_ret(2, 0, 0, 0, C64::new(0.25, 0.5));
        let text = green_to_string(&g);
        let back = green_from_str(&text).unwrap();
        assert!(back.equal(&g));
        // the file really contains the full square
        assert!(text.contains("index :   1    2 "));
        assert!(text.contains("index :   2    1 "));
    }

    #[test]
    fn time_function_round_trip_with_matsubara_slot() {
        let mut cf = TimeFunction::new(2, 2, 1, C64::ZERO);
        cf.set(0, 0, 0, C64::new(1.0, 1.0));
        cf.set(2, 1, 0, C64::new(2.0, -1.0));
        cf.set(MATSUBARA, 0, 0, C64::new(9.0, 0.5));
        let text = time_function_to_string(&cf);
        let back = time_function_from_str(&text).unwrap();
        assert_eq!(back.get(0, 0, 0), C64::new(1.0, 1.0));
        assert_eq!(back.get(2, 1, 0), C64::new(2.0, -1.0));
        assert_eq!(back.get(MATSUBARA, 0, 0), C64::new(9.0, 0.5));
        assert_eq!(time_function_to_string(&back), text);
    }

    #[test]
    fn contour_round_trip_and_index_translation() {
        let c = Contour::new(10, 20, 2, 3, 5.0, 8.0);
        let text = contour_to_string(&c);
        // the file stores the point-count convention
        assert!(text.contains("ntime : 11"));
        assert!(text.contains("ntau  : 21"));
        let back = contour_from_str(&text).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn timestep_round_trip() {
        let c = Contour::new(3, 4, 1, 1, 1.0, 1.0);
        let mut f = GreenTimeStep::new(&c, 2, Sign::Bose, C64::ZERO);
        f.set_ret(2, 1, 0, 0, C64::new(3.0, 0.0));
        f.set_less(0, 2, 0, 0, C64::new(-1.0, 0.5));
        let text = timestep_to_string(&f);
        let back = timestep_from_str(&text).unwrap();
        assert_eq!(back.tstp, 2);
        assert_eq!(back.sign, Sign::Bose);
        assert!(back.equal(&f));
    }

    #[test]
    fn partial_read_overwrites_domain_only() {
        let mut g = demo_green();
        // corrupt a row beyond tstp = 1
        g.set_ret(3, 2, 0, 0, C64::new(77.0, 0.0));
        let clean = demo_green();
        let text = green_to_string(&clean);
        green_partial_from_str(&text, &mut g, 1).unwrap();
        // rows <= 1 were overwritten with the clean values
        assert!(g.get_ret(1, 0, 0, 0) == clean.get_ret(1, 0, 0, 0));
        // the corrupted row 3 survives
        assert_eq!(g.get_ret(3, 2, 0, 0), C64::new(77.0, 0.0));
    }

    #[test]
    fn real_valued_lines_are_accepted() {
        // Julia's F64 variant writes 3-token lines; accept them (im = 0)
        let text = "# Contour-Based Function\nntime : 1\nndim1 : 1\nndim2 : 1\ndata  : \nindex :   1 \n   1    1     1.500000000000\nindex :   2 \n   1    1     2.500000000000\n";
        let cf = time_function_from_str(text).unwrap();
        assert_eq!(cf.get(0, 0, 0), C64::new(1.5, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), C64::new(2.5, 0.0));
    }

    #[test]
    fn bad_input_reports_line_numbers() {
        let bad = "# Kadanoff-Baym Contour\nntime : X\n";
        let err = contour_from_str(bad).unwrap_err();
        assert_eq!(err.line, 2);
    }

    #[test]
    fn fmt_f12_matches_c_width() {
        // C %16.12f: "1.000000000000" is 14 chars, padded to width 16
        assert_eq!(fmt_f12(1.0), "  1.000000000000");
        assert_eq!(fmt_f12(-1.0), " -1.000000000000");
        assert_eq!(fmt_f12(0.0), "  0.000000000000");
    }
}

// ------------------------------------------------------------------
// Standalone component IO (blueprint `write(fn, comp::G*)`)
// ------------------------------------------------------------------

/// Serialize a single mat component (its file is exactly one section).
pub fn mat_to_string(m: &MatComp) -> String {
    let mut s = String::new();
    write_mat_section(&mut s, m);
    s
}

/// Serialize a single ret component.
pub fn ret_to_string(r: &RetComp) -> String {
    let mut s = String::new();
    write_ret_section(&mut s, r);
    s
}

/// Serialize a single lmix component.
pub fn lmix_to_string(l: &LmixComp) -> String {
    let mut s = String::new();
    write_lmix_section(&mut s, l);
    s
}

/// Serialize a single less component.
pub fn less_to_string(l: &LessComp) -> String {
    let mut s = String::new();
    write_less_section(&mut s, l);
    s
}

/// Parse a single mat component file body.
pub fn mat_from_str(text: &str) -> Result<MatComp, IoError> {
    let mut ls = Lines::new(text);
    let (_, _, m) = read_mat_section(&mut ls)?;
    Ok(m)
}

/// Parse a single ret component file body.
pub fn ret_from_str(text: &str) -> Result<RetComp, IoError> {
    let mut ls = Lines::new(text);
    Ok(read_ret_section(&mut ls)?)
}

/// Parse a single lmix component file body.
pub fn lmix_from_str(text: &str) -> Result<LmixComp, IoError> {
    let mut ls = Lines::new(text);
    Ok(read_lmix_section(&mut ls)?)
}

/// Parse a single less component file body.
pub fn less_from_str(text: &str) -> Result<LessComp, IoError> {
    let mut ls = Lines::new(text);
    Ok(read_less_section(&mut ls)?)
}

// ------------------------------------------------------------------
// moving (truncated) window plain text I/O
// ------------------------------------------------------------------
//
// Blueprint julia/src/io/{show,read}.jl, the `KB_MOVING_V1` format.
// The ring content is written in the LOGICAL order (the physical ring
// order is never written); the file tstp is the KB time step (>= 1
// when initialized), the 0-based tstp of the types converts at this
// boundary.

use crate::types::moving::{CfMoving, MovingGreen};

fn fmt_e16(x: f64) -> String {
    format!("{x:.16e}")
}

/// Parse and validate the header of a moving text file (blueprint
/// `_read_moving_header`).
fn read_moving_header(
    line: &str,
    lineno: usize,
    kind: &str,
    tc: i64,
    ndim: usize,
) -> Result<(bool, i64, i64), IoError> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let bad = |msg: String| IoError {
        line: lineno,
        msg,
    };
    if !(f.len() == 8 && f[0] == "#" && f[1] == "KB_MOVING_V1") {
        return Err(bad("not a KB_MOVING_V1 file".into()));
    }
    if f[2] != kind {
        return Err(bad(format!("wrong kind {}, expected {kind}", f[2])));
    }
    let ftc: i64 = f[3]
        .parse()
        .map_err(|_| bad("bad tc".into()))?;
    let init: i64 = f[4]
        .parse()
        .map_err(|_| bad("bad initialized flag".into()))?;
    let tstp: i64 = f[5]
        .parse()
        .map_err(|_| bad("bad tstp".into()))?;
    let sign: i64 = f[6]
        .parse()
        .map_err(|_| bad("bad sign".into()))?;
    let fd: i64 = f[7]
        .parse()
        .map_err(|_| bad("bad ndim".into()))?;
    if ftc != tc {
        return Err(bad(format!("wrong tc {ftc}, expected {tc}")));
    }
    if fd != ndim as i64 {
        return Err(bad(format!("wrong ndim {fd}, expected {ndim}")));
    }
    if !(init == 0 || init == 1) {
        return Err(bad(format!("bad initialized flag {init}")));
    }
    // an uninitialized object carries the placeholder tstp = 0, an
    // initialized one a valid KB time step
    if !(if init == 0 { tstp == 0 } else { tstp >= 1 }) {
        return Err(bad(format!(
            "inconsistent initialized/tstp: {init}/{tstp}"
        )));
    }
    Ok((init == 1, tstp, sign))
}

/// Write the moving Green's function as plain text (blueprint
/// `Base.show(io, G::Fmov)`).
pub fn moving_green_to_string(g: &MovingGreen) -> String {
    let tc = g.tc;
    let d = g.ret.size1;
    let mut out = String::new();
    let tstp_file = if g.initialized { g.tstp + 1 } else { 0 };
    out.push_str(&format!(
        "# KB_MOVING_V1 F {tc} {} {tstp_file} {} {d}\n",
        g.initialized as i64,
        g.sign.as_i64()
    ));
    for comp in [&g.ret, &g.less] {
        for delti in 0..=tc {
            for deltj in 0..=tc {
                let blk = comp.mblock(delti, deltj);
                for b in 0..d {
                    for a in 0..d {
                        let z = blk[a * d + b];
                        out.push_str(&format!(
                            "{delti} {deltj} {} {} {} {}\n",
                            a + 1,
                            b + 1,
                            fmt_e16(z.re),
                            fmt_e16(z.im)
                        ));
                    }
                }
            }
        }
    }
    out
}

/// Write the moving single-time function as plain text (blueprint
/// `Base.show(io, f::Cfmov)`).
pub fn cf_moving_to_string(f: &CfMoving) -> String {
    let d = f.size1;
    let mut out = String::new();
    let tstp_file = if f.initialized { f.tstp + 1 } else { 0 };
    out.push_str(&format!(
        "# KB_MOVING_V1 Cf {} {} {tstp_file} 0 {d}\n",
        f.tc,
        f.initialized as i64
    ));
    for delti in 0..=f.tc {
        let blk = f.fslot(delti);
        for b in 0..d {
            for a in 0..d {
                let z = blk[a * d + b];
                out.push_str(&format!(
                    "{delti} {} {} {} {}\n",
                    a + 1,
                    b + 1,
                    fmt_e16(z.re),
                    fmt_e16(z.im)
                ));
            }
        }
    }
    out
}

/// Read the moving Green's function from the text produced by
/// `moving_green_to_string`. The text is validated completely before
/// `g` is modified; on success the state is replaced (blueprint
/// `Base.read!(fn, G::Fmov)`).
pub fn moving_green_read_into(text: &str, g: &mut MovingGreen) -> Result<(), IoError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Err(IoError {
            line: 1,
            msg: "empty file".into(),
        });
    }
    let tc = g.tc;
    let d = g.ret.size1;
    let b = d * d;
    let tcl = (tc + 1) as usize;
    let (init, tstp_file, sign) = read_moving_header(lines[0], 1, "F", tc, d)?;
    if sign != g.sign.as_i64() {
        return Err(IoError {
            line: 1,
            msg: format!("wrong sign {sign}, expected {}", g.sign.as_i64()),
        });
    }
    let nrec = 2 * tcl * tcl * b;
    if lines.len() != nrec + 1 {
        return Err(IoError {
            line: 1,
            msg: format!(
                "wrong number of records {}, expected {nrec}",
                lines.len() - 1
            ),
        });
    }
    let mut data = vec![C64::ZERO; 2 * tcl * tcl * b];
    let mut k = 1usize;
    for comp in 0..2usize {
        for delti in 0..=tc {
            for deltj in 0..=tc {
                for bidx in 0..d {
                    for aidx in 0..d {
                        let bad = |msg: &str| IoError {
                            line: k + 1,
                            msg: msg.into(),
                        };
                        let f: Vec<&str> = lines[k].split_whitespace().collect();
                        if f.len() != 6 {
                            return Err(bad("bad record"));
                        }
                        let i0: i64 = f[0].parse().map_err(|_| bad("bad integer"))?;
                        let i1: i64 = f[1].parse().map_err(|_| bad("bad integer"))?;
                        let i2: i64 = f[2].parse().map_err(|_| bad("bad integer"))?;
                        let i3: i64 = f[3].parse().map_err(|_| bad("bad integer"))?;
                        if (i0, i1, i2, i3)
                            != (delti, deltj, aidx as i64 + 1, bidx as i64 + 1)
                        {
                            return Err(bad("unexpected index"));
                        }
                        let re: f64 = f[4].parse().map_err(|_| bad("bad value"))?;
                        let im: f64 = f[5].parse().map_err(|_| bad("bad value"))?;
                        if !re.is_finite() || !im.is_finite() {
                            return Err(bad("non-finite value"));
                        }
                        let slot = (delti as usize * tcl + deltj as usize) * b;
                        data[comp * tcl * tcl * b + slot + aidx * d + bidx] =
                            C64::new(re, im);
                        k += 1;
                    }
                }
            }
        }
    }
    // commit: store in the logical order (identity ring, head = 0)
    g.ret.head = 0;
    g.less.head = 0;
    g.ret.raw_mut().copy_from_slice(&data[..tcl * tcl * b]);
    g.less
        .raw_mut()
        .copy_from_slice(&data[tcl * tcl * b..]);
    g.tstp = if init { tstp_file - 1 } else { 0 };
    g.initialized = init;
    g.generation += 1;
    Ok(())
}

/// Read the moving single-time function (blueprint
/// `Base.read!(fn, f::Cfmov)`).
pub fn cf_moving_read_into(text: &str, f: &mut CfMoving) -> Result<(), IoError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Err(IoError {
            line: 1,
            msg: "empty file".into(),
        });
    }
    let tc = f.tc;
    let d = f.size1;
    let b = d * d;
    let tcl = (tc + 1) as usize;
    let (init, tstp_file, _sign) = read_moving_header(lines[0], 1, "Cf", tc, d)?;
    let nrec = tcl * b;
    if lines.len() != nrec + 1 {
        return Err(IoError {
            line: 1,
            msg: format!(
                "wrong number of records {}, expected {nrec}",
                lines.len() - 1
            ),
        });
    }
    let mut data = vec![C64::ZERO; tcl * b];
    let mut k = 1usize;
    for delti in 0..=tc {
        for bidx in 0..d {
            for aidx in 0..d {
                let bad = |msg: &str| IoError {
                    line: k + 1,
                    msg: msg.into(),
                };
                let fs: Vec<&str> = lines[k].split_whitespace().collect();
                if fs.len() != 5 {
                    return Err(bad("bad record"));
                }
                let i0: i64 = fs[0].parse().map_err(|_| bad("bad integer"))?;
                let i2: i64 = fs[1].parse().map_err(|_| bad("bad integer"))?;
                let i3: i64 = fs[2].parse().map_err(|_| bad("bad integer"))?;
                if (i0, i2, i3) != (delti, aidx as i64 + 1, bidx as i64 + 1) {
                    return Err(bad("unexpected index"));
                }
                let re: f64 = fs[3].parse().map_err(|_| bad("bad value"))?;
                let im: f64 = fs[4].parse().map_err(|_| bad("bad value"))?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(bad("non-finite value"));
                }
                data[delti as usize * b + aidx * d + bidx] = C64::new(re, im);
                k += 1;
            }
        }
    }
    f.head = 0;
    f.raw_mut().copy_from_slice(&data);
    f.tstp = if init { tstp_file - 1 } else { 0 };
    f.initialized = init;
    Ok(())
}

/// Write the moving Green's function to a file.
pub fn write_moving_green(path: &str, g: &MovingGreen) -> std::io::Result<()> {
    std::fs::write(path, moving_green_to_string(g))
}

/// Read the moving Green's function from a file into `g`.
pub fn read_moving_green(path: &str, g: &mut MovingGreen) -> Result<(), IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    moving_green_read_into(&text, g)
}

/// Write the moving single-time function to a file.
pub fn write_cf_moving(path: &str, f: &CfMoving) -> std::io::Result<()> {
    std::fs::write(path, cf_moving_to_string(f))
}

/// Read the moving single-time function from a file into `f`.
pub fn read_cf_moving(path: &str, f: &mut CfMoving) -> Result<(), IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    cf_moving_read_into(&text, f)
}

// ------------------------------------------------------------------
// NESS2 plain text I/O
// ------------------------------------------------------------------
//
// Blueprint julia/src/io/ness.jl. The header line is
//     # FFTArray Nft size1
//     # HermMatrixNESS Nft size1 h dw
// and every other line contains `domain component index i j re im`,
// where `domain` is `time` or `freq`, `component` is `data`
// (NessArray) or `ret` and `les` (NessHermMat), `index` is the 1-based
// storage index and `i, j` are the 1-based orbital indices. All
// matrix elements are written. It is the plain text counterpart of
// `print_to_file` and `read_from_file` in NESSi.

use crate::types::ness::{FftDomain, NessArray, NessGrid, NessHermMat};
use std::collections::HashSet;

/// C `%.17g` formatting of a float.
pub fn fmt_g17(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    let ax = x.abs();
    let exp = ax.log10().floor() as i64;
    let out = if exp < -4 || exp >= 17 {
        // scientific with 17 significant digits
        let raw = format!("{:.16e}", x);
        let (mant, e) = raw.split_once('e').unwrap();
        let mant = if mant.contains('.') {
            mant.trim_end_matches('0').trim_end_matches('.')
        } else {
            mant
        };
        let ev: i64 = e.parse().unwrap();
        let sign = if ev < 0 { '-' } else { '+' };
        let ea = ev.abs();
        format!("{mant}e{sign}{ea:02}")
    } else {
        // fixed with the remaining decimals
        let prec = (16 - exp).max(0) as usize;
        let raw = format!("{x:.prec$}");
        raw.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    out
}

/// Write the time and frequency data of `A` under the component name
/// `comp` (blueprint `_write_fft_rows`).
fn write_fft_rows(out: &mut String, a: &NessArray, comp: &str) {
    let d1 = a.size1;
    let d2 = a.size2;
    for (dom, data) in [("time", &a.time), ("freq", &a.freq)] {
        for n in 1..=(a.nft as usize) {
            for j in 1..=d2 {
                for i in 1..=d1 {
                    let z = data[(n - 1) * d1 * d2 + (i - 1) * d2 + (j - 1)];
                    out.push_str(&format!(
                        "{dom} {comp} {n} {i} {j} {} {}\n",
                        fmt_g17(z.re),
                        fmt_g17(z.im)
                    ));
                }
            }
        }
    }
}

/// Parse the data lines into the given arrays (component name =>
/// NessArray); blueprint `_read_fft_rows!`. The lines are the text
/// after the header line.
fn read_fft_rows(
    lines: &[&str],
    arrays: &mut [(&str, &mut NessArray)],
    ctx: &str,
) -> Result<(), IoError> {
    let nft = arrays[0].1.nft;
    let d1 = arrays[0].1.size1;
    let d2 = arrays[0].1.size2;
    let nexp = 2 * arrays.len() * (nft as usize) * d1 * d2;
    if lines.len() != nexp {
        return Err(IoError {
            line: 0,
            msg: format!("{ctx}: expected {nexp} data lines, got {}", lines.len()),
        });
    }
    let mut seen: HashSet<(String, String, i64, i64, i64)> = HashSet::new();
    for (l, line) in lines.iter().enumerate() {
        let bad = |msg: String| IoError {
            line: l + 2,
            msg,
        };
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 7 {
            return Err(bad(format!("{ctx}: line {} must have 7 columns", l + 2)));
        }
        let dom = f[0];
        let comp = f[1];
        if dom != "time" && dom != "freq" {
            return Err(bad(format!("{ctx}: unknown domain {dom} at line {}", l + 2)));
        }
        let slot = arrays.iter_mut().find(|(name, _)| *name == comp);
        let (_, arr) = match slot {
            Some(x) => x,
            None => {
                return Err(bad(format!("{ctx}: unknown component {comp} at line {}", l + 2)));
            }
        };
        let n: i64 = f[2]
            .parse()
            .map_err(|_| bad(format!("{ctx}: bad index at line {}", l + 2)))?;
        let i: i64 = f[3]
            .parse()
            .map_err(|_| bad(format!("{ctx}: bad index at line {}", l + 2)))?;
        let j: i64 = f[4]
            .parse()
            .map_err(|_| bad(format!("{ctx}: bad index at line {}", l + 2)))?;
        if !(1 <= n && n <= nft && 1 <= i && i <= d1 as i64 && 1 <= j && j <= d2 as i64) {
            return Err(bad(format!("{ctx}: index out of range at line {}", l + 2)));
        }
        let re: f64 = f[5]
            .parse()
            .map_err(|_| bad(format!("{ctx}: bad value at line {}", l + 2)))?;
        let im: f64 = f[6]
            .parse()
            .map_err(|_| bad(format!("{ctx}: bad value at line {}", l + 2)))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(bad(format!("{ctx}: bad value at line {}", l + 2)));
        }
        let key = (dom.to_string(), comp.to_string(), n, i, j);
        if !seen.insert(key) {
            return Err(bad(format!("{ctx}: duplicated entry at line {}", l + 2)));
        }
        let q = ((n - 1) as usize) * d1 * d2 + (i as usize - 1) * d2 + (j as usize - 1);
        let z = C64::new(re, im);
        if dom == "time" {
            arr.time[q] = z;
        } else {
            arr.freq[q] = z;
        }
    }
    Ok(())
}

/// Split a NESS2 text file into the header fields and the data lines
/// (blueprint `_read_ness_header`).
fn read_ness_header<'a>(
    text: &'a str,
    ctx: &str,
    kind: &str,
    nfield: usize,
) -> Result<(Vec<&'a str>, Vec<&'a str>), IoError> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err(IoError {
            line: 0,
            msg: format!("{ctx}: empty file"),
        });
    }
    let h: Vec<&str> = lines[0].split_whitespace().collect();
    if !(h.len() == nfield && h[0] == "#" && h[1] == kind) {
        return Err(IoError {
            line: 1,
            msg: format!("{ctx}: bad header, expected `# {kind} ...`"),
        });
    }
    Ok((h, lines[1..].to_vec()))
}

/// Write an `NessArray` to a plain text file (blueprint
/// `write_fftarray`).
pub fn ness_array_to_string(a: &NessArray) -> String {
    let mut out = String::new();
    out.push_str(&format!("# FFTArray {} {}\n", a.nft, a.size1));
    write_fft_rows(&mut out, a, "data");
    out
}

/// Read an `NessArray` from the text produced by
/// `ness_array_to_string` (blueprint `read_fftarray`).
pub fn ness_array_from_str(text: &str) -> Result<NessArray, IoError> {
    let (h, lines) = read_ness_header(text, "fftarray", "FFTArray", 4)?;
    let nft: i64 = h[2].parse().map_err(|_| IoError {
        line: 1,
        msg: "fftarray: bad header values".into(),
    })?;
    let size1: i64 = h[3].parse().map_err(|_| IoError {
        line: 1,
        msg: "fftarray: bad header values".into(),
    })?;
    if nft < 1 || size1 < 1 {
        return Err(IoError {
            line: 1,
            msg: "fftarray: bad header values".into(),
        });
    }
    let mut a = NessArray::new(nft, size1 as usize);
    read_fft_rows(&lines, &mut [("data", &mut a)], "fftarray")?;
    Ok(a)
}

/// Write a `NessHermMat` to a plain text file (blueprint
/// `write_ness`).
pub fn ness_to_string(g: &NessHermMat) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# HermMatrixNESS {} {} {} {}\n",
        g.grid.nft,
        g.ret.size1,
        fmt_g17(g.grid.h),
        fmt_g17(g.grid.dw)
    ));
    write_fft_rows(&mut out, &g.ret, "ret");
    write_fft_rows(&mut out, &g.les, "les");
    out
}

/// Read a `NessHermMat` from the text produced by `ness_to_string`
/// (blueprint `read_ness`).
pub fn ness_from_str(text: &str) -> Result<NessHermMat, IoError> {
    let (h, lines) = read_ness_header(text, "ness", "HermMatrixNESS", 6)?;
    let nft: i64 = h[2].parse().map_err(|_| IoError {
        line: 1,
        msg: "ness: bad header values".into(),
    })?;
    let size1: i64 = h[3].parse().map_err(|_| IoError {
        line: 1,
        msg: "ness: bad header values".into(),
    })?;
    let hh: f64 = h[4].parse().map_err(|_| IoError {
        line: 1,
        msg: "ness: bad header values".into(),
    })?;
    let dw: f64 = h[5].parse().map_err(|_| IoError {
        line: 1,
        msg: "ness: bad header values".into(),
    })?;
    if size1 < 1 || !(nft >= 4 && nft % 2 == 0) {
        return Err(IoError {
            line: 1,
            msg: "ness: bad header values".into(),
        });
    }
    let mut g = NessHermMat::new(NessGrid::new_full(nft, hh, dw), size1 as usize);
    let mut ret = std::mem::replace(&mut g.ret, NessArray::new(4, 1));
    let mut les = std::mem::replace(&mut g.les, NessArray::new(4, 1));
    read_fft_rows(&lines, &mut [("ret", &mut ret), ("les", &mut les)], "ness")?;
    g.ret = ret;
    g.les = les;
    Ok(g)
}

/// Write an `NessArray` to a file.
pub fn write_fftarray(path: &str, a: &NessArray) -> std::io::Result<()> {
    std::fs::write(path, ness_array_to_string(a))
}

/// Read an `NessArray` from a file.
pub fn read_fftarray(path: &str) -> Result<NessArray, IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    ness_array_from_str(&text)
}

/// Write a `NessHermMat` to a file.
pub fn write_ness(path: &str, g: &NessHermMat) -> std::io::Result<()> {
    std::fs::write(path, ness_to_string(g))
}

/// Read a `NessHermMat` from a file.
pub fn read_ness(path: &str) -> Result<NessHermMat, IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    ness_from_str(&text)
}

/// The domain of a NESS2 block accessor from the string name.
pub fn ness_domain_of(name: &str) -> FftDomain {
    if name == "time" {
        FftDomain::Time
    } else {
        FftDomain::Freq
    }
}


// ------------------------------------------------------------------
// pseudo-particle plain text I/O (blueprint io/show.jl, io/read.jl)
// ------------------------------------------------------------------

/// Serialize a `PseudoGreen` (blueprint `show(io, F^{ps})`); the
/// sections are shared with the standard format, the header marks
/// the pseudo-particle convention.
pub fn pseudo_green_to_string(g: &PseudoGreen) -> String {
    let mut out = String::new();
    out.push_str("# Pseudo-Particle Contour-Ordered Green's Function:\n");
    out.push_str(&format!("sign  : {}\n", g.sign.as_i64()));
    out.push('\n');
    write_mat_section(&mut out, &g.mat);
    out.push('\n');
    write_ret_section(&mut out, &g.ret);
    out.push('\n');
    write_lmix_section(&mut out, &g.lmix);
    out.push('\n');
    write_less_section(&mut out, &g.less);
    out.push('\n');
    out
}

/// Parse a `PseudoGreen` written by [`pseudo_green_to_string`] into
/// the preallocated `g` (the sizes of `g` are never changed, the sign
/// is taken from the file).
pub fn pseudo_green_read_into(text: &str, g: &mut PseudoGreen) -> Result<(), IoError> {
    let first = text.lines().next().unwrap_or("");
    if !first.contains("Pseudo") {
        return Err(IoError {
            line: 1,
            msg: "not a pseudo-particle Green's function file".to_string(),
        });
    }
    let src = green_from_str(text)?;
    if src.nt != g.nt || src.ntau != g.ntau || src.size1 != g.size1 || src.size2 != g.size2 {
        return Err(IoError {
            line: 1,
            msg: format!(
                "dimension mismatch: file {}x{} nt={} ntau={}, target {}x{} nt={} ntau={}",
                src.size1, src.size2, src.nt, src.ntau, g.size1, g.size2, g.nt, g.ntau
            ),
        });
    }
    g.sign = src.sign;
    g.mat = src.mat.clone();
    g.ret = src.ret.clone();
    g.lmix = src.lmix.clone();
    g.less = src.less.clone();
    Ok(())
}

pub fn write_pseudo_green(path: &str, g: &PseudoGreen) -> std::io::Result<()> {
    fs::write(path, pseudo_green_to_string(g))
}

pub fn read_pseudo_green_into(path: &str, g: &mut PseudoGreen) -> Result<(), IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("{path}: {e}"),
    })?;
    pseudo_green_read_into(&text, g)
}
