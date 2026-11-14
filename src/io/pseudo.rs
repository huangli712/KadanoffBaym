//
// Project : KadanoffBaym
// File    : src/io/pseudo.rs
// Purpose : pseudo-particle plain text I/O (blueprint io/show.jl,
//           io/read.jl); the sections are shared with the standard
//           format, the header marks the pseudo-particle convention
//

use std::fs;

use crate::types::pseudo::PseudoGreen;
use super::IoError;
use super::green::{
    green_from_str, write_less_section, write_lmix_section, write_mat_section,
    write_ret_section,
};

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
