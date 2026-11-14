//
// Project : KadanoffBaym
// File    : src/io/green.rs
// Purpose : plain-text IO of the standard contour containers
//           (blueprint: julia/src/io/{show,write,read}.jl)
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

use std::fs;

use crate::complex::Complex;
use crate::types::{Contour, Sign};
use crate::types::{
    ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, TimeFunction,
};
use super::{fmt_f12, IoError};

/// C-style `%4i`.
fn fmt_i4(x: i64) -> String {
    format!("{:4}", x)
}

// ------------------------------------------------------------------
// writers (to strings; file wrappers at the bottom)
// ------------------------------------------------------------------

fn write_element_lines(out: &mut String, block: &[Complex], s1: usize, s2: usize) {
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

pub(crate) fn write_mat_section(out: &mut String, mat: &MatComp) {
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

pub(crate) fn write_ret_section(out: &mut String, ret: &RetComp) {
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
                let zeros = vec![Complex::ZERO; ret.size1 * ret.size2];
                write_element_lines(out, &zeros, ret.size1, ret.size2);
            }
        }
    }
}

/// Stored-or-reconstructed ret block (0-based `(tstp, j)`).
fn ret_block_value(ret: &RetComp, tstp: i64, j: i64) -> Vec<Complex> {
    if tstp >= j {
        ret.block(ret.idx(tstp, j)).to_vec()
    } else {
        let src = ret.block(ret.idx(j, tstp));
        let (s1, s2) = (ret.size1, ret.size2);
        let mut out = vec![Complex::ZERO; s1 * s2];
        for a in 0..s1 {
            for b in 0..s2 {
                out[a * s2 + b] = -src[b * s1 + a].conj();
            }
        }
        out
    }
}

pub(crate) fn write_lmix_section(out: &mut String, lmix: &LmixComp) {
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

pub(crate) fn write_less_section(out: &mut String, less: &LessComp) {
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
                let zeros = vec![Complex::ZERO; less.size1 * less.size2];
                write_element_lines(out, &zeros, less.size1, less.size2);
            }
        }
    }
}

/// Stored-or-reconstructed less block (0-based `(i, tstp)`).
fn less_block_value(less: &LessComp, i: i64, tstp: i64) -> Vec<Complex> {
    if i <= tstp {
        less.block(less.idx(i, tstp)).to_vec()
    } else {
        let src = less.block(less.idx(tstp, i));
        let (s1, s2) = (less.size1, less.size2);
        let mut out = vec![Complex::ZERO; s1 * s2];
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
    fn element(&mut self, block: &mut [Complex], s1: usize, s2: usize) -> Result<(), IoError> {
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

    fn block(&mut self, s1: usize, s2: usize) -> Result<Vec<Complex>, IoError> {
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
    let mut cf = TimeFunction::new(nt, size1, size2, Complex::ZERO);
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
    let mut mat = MatComp::new(ntau, s1, s2, Complex::ZERO);
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
    let mut ret = RetComp::new(nt, s1, s2, Complex::ZERO);
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
    let mut lmix = LmixComp::new(nt, ntau, s1, s2, Complex::ZERO);
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
    let mut less = LessComp::new(nt, s1, s2, Complex::ZERO);
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
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut v = 1.0;
        for n in 0..=4 {
            g.set_mat(n, 0, 0, Complex::new(v, -v / 2.0));
            v += 1.0;
        }
        for i in 0..=3 {
            for j in 0..=i {
                g.set_ret(i, j, 1, 0, Complex::new(i as f64, j as f64));
                g.set_less(j, i, 0, 1, Complex::new(-(i as f64), j as f64 * 0.25));
            }
        }
        for t in 0..=3 {
            for m in 0..=4 {
                g.set_lmix(t, m, 1, 1, Complex::new(t as f64, m as f64));
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
        g.set_ret(2, 0, 0, 0, Complex::new(0.25, 0.5));
        let text = green_to_string(&g);
        let back = green_from_str(&text).unwrap();
        assert!(back.equal(&g));
        // the file really contains the full square
        assert!(text.contains("index :   1    2 "));
        assert!(text.contains("index :   2    1 "));
    }

    #[test]
    fn time_function_round_trip_with_matsubara_slot() {
        let mut cf = TimeFunction::new(2, 2, 1, Complex::ZERO);
        cf.set(0, 0, 0, Complex::new(1.0, 1.0));
        cf.set(2, 1, 0, Complex::new(2.0, -1.0));
        cf.set(MATSUBARA, 0, 0, Complex::new(9.0, 0.5));
        let text = time_function_to_string(&cf);
        let back = time_function_from_str(&text).unwrap();
        assert_eq!(back.get(0, 0, 0), Complex::new(1.0, 1.0));
        assert_eq!(back.get(2, 1, 0), Complex::new(2.0, -1.0));
        assert_eq!(back.get(MATSUBARA, 0, 0), Complex::new(9.0, 0.5));
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
        let mut f = GreenTimeStep::new(&c, 2, Sign::Bose, Complex::ZERO);
        f.set_ret(2, 1, 0, 0, Complex::new(3.0, 0.0));
        f.set_less(0, 2, 0, 0, Complex::new(-1.0, 0.5));
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
        g.set_ret(3, 2, 0, 0, Complex::new(77.0, 0.0));
        let clean = demo_green();
        let text = green_to_string(&clean);
        green_partial_from_str(&text, &mut g, 1).unwrap();
        // rows <= 1 were overwritten with the clean values
        assert!(g.get_ret(1, 0, 0, 0) == clean.get_ret(1, 0, 0, 0));
        // the corrupted row 3 survives
        assert_eq!(g.get_ret(3, 2, 0, 0), Complex::new(77.0, 0.0));
    }

    #[test]
    fn real_valued_lines_are_accepted() {
        // Julia's F64 variant writes 3-token lines; accept them (im = 0)
        let text = "# Contour-Based Function\nntime : 1\nndim1 : 1\nndim2 : 1\ndata  : \nindex :   1 \n   1    1     1.500000000000\nindex :   2 \n   1    1     2.500000000000\n";
        let cf = time_function_from_str(text).unwrap();
        assert_eq!(cf.get(0, 0, 0), Complex::new(1.5, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), Complex::new(2.5, 0.0));
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
