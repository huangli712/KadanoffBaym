//
// Project : KadanoffBaym
// File    : src/io/ness.rs
// Purpose : NESS2 plain text I/O (blueprint julia/src/io/ness.jl).
//           Header lines `# FFTArray Nft size1` / `# HermMatrixNESS
//           Nft size1 h dw`; data lines `domain component index i j
//           re im`. Plain-text counterpart of NESSi print_to_file /
//           read_from_file.
//

use std::collections::HashSet;

use crate::complex::Complex;
use crate::types::ness::{FftDomain, NessArray, NessGrid, NessHermMat};
use super::IoError;

/// C `%.17g` formatting of a float.
pub(crate) fn fmt_g17(x: f64) -> String {
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
        let z = Complex::new(re, im);
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
pub(crate) fn write_fftarray(path: &str, a: &NessArray) -> std::io::Result<()> {
    std::fs::write(path, ness_array_to_string(a))
}

/// Read an `NessArray` from a file.
pub(crate) fn read_fftarray(path: &str) -> Result<NessArray, IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    ness_array_from_str(&text)
}

/// Write a `NessHermMat` to a file.
pub(crate) fn write_ness(path: &str, g: &NessHermMat) -> std::io::Result<()> {
    std::fs::write(path, ness_to_string(g))
}

/// Read a `NessHermMat` from a file.
pub(crate) fn read_ness(path: &str) -> Result<NessHermMat, IoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| IoError { line: 0, msg: e.to_string() })?;
    ness_from_str(&text)
}

/// The domain of a NESS2 block accessor from the string name.
pub(crate) fn ness_domain_of(name: &str) -> FftDomain {
    if name == "time" {
        FftDomain::Time
    } else {
        FftDomain::Freq
    }
}
