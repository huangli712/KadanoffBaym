// Project : KadanoffBaym
// File    : src/io/steady.rs
// Purpose : steady-state plain text I/O; the header lines
//           `# SteadyFunction Nft size1` / `# SteadyGreen Nft size1 h dw`,
//           the data lines `domain component index i j re im`

//!
//! *Remarks* : *Steady-State Text Format*
//!
//! The steady-state data is written as plain text. The header line is
//!
//! ```text
//! # SteadyFunction Nft size1
//! # SteadyGreen Nft size1 h dw
//! ```
//!
//! and every other line contains `domain component index i j re im`, where
//! `domain` is `time` or `freq`, `component` is `data` (SteadyFunction) or `ret`
//! and `les` (SteadyGreen), `index` is the 1-based storage index, and
//! `i, j` are the orbital indices. All matrix elements are written. The
//! reader checks the header, the number of lines, the ranges of the indices,
//! duplicated entries, and finite values.

use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::mem;
use std::str::FromStr;

use crate::complex::Complex;
use crate::types::{SteadyFunction, SteadyGreen, SteadyGrid};

use crate::io::error::IoError;

use crate::io::format::fmt_g17;

// ------------------------------------------------------------------
// string-level serialization: the free helpers stay crate-private, the
// public surface is the standard Display / FromStr traits
// ------------------------------------------------------------------

impl Display for SteadyFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(&steady_function_to_string(self))
    }
}

impl FromStr for SteadyFunction {
    type Err = IoError;
    fn from_str(text: &str) -> Result<Self, IoError> {
        steady_function_from_str(text)
    }
}

impl Display for SteadyGreen {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(&steady_green_to_string(self))
    }
}

impl FromStr for SteadyGreen {
    type Err = IoError;
    fn from_str(text: &str) -> Result<Self, IoError> {
        steady_green_from_str(text)
    }
}

/// Split a steady-state text file into the header fields and the data lines.
fn read_steady_header<'a>(
    text: &'a str,
    ctx: &str,
    kind: &str,
    nfield: usize,
) -> Result<(Vec<&'a str>, Vec<&'a str>), IoError> {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
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

/// Write the time and frequency data of `A` under the component name `comp`.
fn write_fft_rows(out: &mut String, a: &SteadyFunction, comp: &str) {
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
/// SteadyFunction). The lines are the text after the header line.
fn read_fft_rows(
    lines: &[&str],
    arrays: &mut [(&str, &mut SteadyFunction)],
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
        let bad = |msg: String| IoError { line: l + 2, msg };
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 7 {
            return Err(bad(format!("{ctx}: line {} must have 7 columns", l + 2)));
        }
        let dom = f[0];
        let comp = f[1];
        if dom != "time" && dom != "freq" {
            return Err(bad(format!(
                "{ctx}: unknown domain {dom} at line {}",
                l + 2
            )));
        }
        let slot = arrays.iter_mut().find(|(name, _)| *name == comp);
        let (_, arr) = match slot {
            Some(x) => x,
            None => {
                return Err(bad(format!(
                    "{ctx}: unknown component {comp} at line {}",
                    l + 2
                )));
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
        let z = Complex::new(re, im);
        if dom == "time" {
            arr.time[q] = z;
        } else {
            arr.freq[q] = z;
        }
    }
    Ok(())
}

/// Write a `SteadyFunction` to a plain text file.
fn steady_function_to_string(a: &SteadyFunction) -> String {
    let mut out = String::new();
    out.push_str(&format!("# SteadyFunction {} {}\n", a.nft, a.size1));
    write_fft_rows(&mut out, a, "data");
    out
}

/// Read a `SteadyFunction` from the text produced by `steady_function_to_string`.
fn steady_function_from_str(text: &str) -> Result<SteadyFunction, IoError> {
    let (h, lines) = read_steady_header(text, "steady_function", "SteadyFunction", 4)?;
    let nft: i64 = h[2].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_function: bad header values".into(),
    })?;
    let size1: i64 = h[3].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_function: bad header values".into(),
    })?;
    if nft < 1 || size1 < 1 {
        return Err(IoError {
            line: 1,
            msg: "steady_function: bad header values".into(),
        });
    }
    let mut a = SteadyFunction::new(nft, size1 as usize);
    read_fft_rows(&lines, &mut [("data", &mut a)], "steady_function")?;
    Ok(a)
}

/// Write a `SteadyGreen` to a plain text file.
fn steady_green_to_string(g: &SteadyGreen) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# SteadyGreen {} {} {} {}\n",
        g.grid.nft,
        g.ret.size1,
        fmt_g17(g.grid.h),
        fmt_g17(g.grid.dw)
    ));
    write_fft_rows(&mut out, &g.ret, "ret");
    write_fft_rows(&mut out, &g.les, "les");
    out
}

/// Read a `SteadyGreen` from the text produced by `steady_green_to_string`.
fn steady_green_from_str(text: &str) -> Result<SteadyGreen, IoError> {
    let (h, lines) = read_steady_header(text, "steady_green", "SteadyGreen", 6)?;
    let nft: i64 = h[2].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_green: bad header values".into(),
    })?;
    let size1: i64 = h[3].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_green: bad header values".into(),
    })?;
    let hh: f64 = h[4].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_green: bad header values".into(),
    })?;
    let dw: f64 = h[5].parse().map_err(|_| IoError {
        line: 1,
        msg: "steady_green: bad header values".into(),
    })?;
    if size1 < 1 || !(nft >= 4 && nft % 2 == 0) {
        return Err(IoError {
            line: 1,
            msg: "steady_green: bad header values".into(),
        });
    }
    let mut g = SteadyGreen::new(SteadyGrid::new_full(nft, hh, dw), size1 as usize);
    let mut ret = mem::replace(&mut g.ret, SteadyFunction::new(4, 1));
    let mut les = mem::replace(&mut g.les, SteadyFunction::new(4, 1));
    read_fft_rows(
        &lines,
        &mut [("ret", &mut ret), ("les", &mut les)],
        "steady_green",
    )?;
    g.ret = ret;
    g.les = les;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_c64(state: &mut u64) -> Complex {
        let a = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        let b = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        Complex::new(a * 4.0, b * 4.0)
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[test]
    fn steady_plain_text_io() {
        let mut state = 42u64;
        let mut g = SteadyGreen::new(SteadyGrid::new(8, 0.1), 2);
        for data in [
            &mut g.ret.time,
            &mut g.ret.freq,
            &mut g.les.time,
            &mut g.les.freq,
        ] {
            for z in data.iter_mut() {
                *z = rand_c64(&mut state);
            }
        }
        let text = steady_green_to_string(&g);
        let h = steady_green_from_str(&text).unwrap();
        assert_eq!(h.ret.time, g.ret.time);
        assert_eq!(h.ret.freq, g.ret.freq);
        assert_eq!(h.les.time, g.les.time);
        assert_eq!(h.les.freq, g.les.freq);
        assert!((h.grid.h - g.grid.h).abs() < 1e-15);
        assert!((h.grid.dw - g.grid.dw).abs() < 1e-12);

        let mut a = SteadyFunction::new(4, 2);
        for z in a.time.iter_mut() {
            *z = rand_c64(&mut state);
        }
        for z in a.freq.iter_mut() {
            *z = rand_c64(&mut state);
        }
        let text = steady_function_to_string(&a);
        let b = steady_function_from_str(&text).unwrap();
        assert_eq!(b.time, a.time);
        assert_eq!(b.freq, a.freq);

        // corrupted files
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        let bad = |ls: Vec<String>| -> String { ls.join("\n") };
        assert!(steady_function_from_str(&bad(lines[..lines.len() - 1].to_vec())).is_err());
        let mut dup = lines[..lines.len() - 1].to_vec();
        dup.push(lines[1].clone());
        assert!(steady_function_from_str(&bad(dup)).is_err());
        let mut hdr: Vec<String> = vec!["# FFTArray 4".into()];
        hdr.extend(lines[1..].iter().cloned());
        assert!(steady_function_from_str(&bad(hdr.clone())).is_err());
        let mut hdr2: Vec<String> = vec!["# SteadyGreen 4 2 0.1 1.0".into()];
        hdr2.extend(lines[1..].iter().cloned());
        assert!(steady_function_from_str(&bad(hdr2)).is_err());
        let mut tyme = lines.clone();
        tyme[1] = tyme[1].replacen("time", "tyme", 1);
        assert!(steady_function_from_str(&bad(tyme)).is_err());
        let mut oob = lines.clone();
        oob[1] = "time data 9 1 1 0.0 0.0".into();
        assert!(steady_function_from_str(&bad(oob)).is_err());
        let mut nan = lines.clone();
        nan[1] = "time data 1 1 1 NaN 0.0".into();
        assert!(steady_function_from_str(&bad(nan)).is_err());
        let mut cols = lines.clone();
        cols[1] = "time data 1 1 1 0.0".into();
        assert!(steady_function_from_str(&bad(cols)).is_err());
        // ness files
        let text = steady_green_to_string(&g);
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        let mut adv = lines.clone();
        adv[1] = adv[1].replacen("ret", "adv", 1);
        assert!(steady_green_from_str(&bad(adv)).is_err());
        let mut nft: Vec<String> = vec!["# SteadyGreen 7 2 0.1 1.0".into()];
        nft.extend(lines[1..].iter().cloned());
        assert!(steady_green_from_str(&bad(nft)).is_err());
    }
}
