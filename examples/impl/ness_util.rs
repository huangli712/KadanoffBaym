//
// Shared helpers of the NESS steady-state examples (the Rust counterpart
// of the NESSi-style TOML input and text output used by
// julia/tutor/hubbard_bethe/*ness*.jl and *trunc*.jl).
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::toml::{parse_toml_file, TomlValue};
use kadanoffbaym::c64::C64;
use kadanoffbaym::types::ness::{time_at, freq_at, FftDomain, NessHermMat};

/// Read a NESSi-style TOML file and check the sections.
pub fn inp_toml(cfgfile: &str) -> Result<std::collections::BTreeMap<String, TomlValue>, String> {
    // the sections vary between the grid-style and contour-style
    // programs; missing keys are reported by the getters
    parse_toml_file(cfgfile).map_err(|e| format!("{cfgfile}: {e}"))
}

/// Fetch a float key from a section.
pub fn tget_f(
    t: &std::collections::BTreeMap<String, TomlValue>,
    sec: &str,
    key: &str,
) -> Result<f64, String> {
    let v = if sec.is_empty() { &t } else { t[sec].as_table().unwrap() };
    let x = v.get(key).ok_or_else(|| format!("missing [{sec}] {key}"))?;
    x.as_f64().ok_or_else(|| format!("[{sec}] {key} must be a number"))
}

/// Fetch an integer key from a section.
pub fn tget_i(
    t: &std::collections::BTreeMap<String, TomlValue>,
    sec: &str,
    key: &str,
) -> Result<i64, String> {
    let v = if sec.is_empty() { &t } else { t[sec].as_table().unwrap() };
    let x = v.get(key).ok_or_else(|| format!("missing [{sec}] {key}"))?;
    x.as_i64().ok_or_else(|| format!("[{sec}] {key} must be an integer"))
}

/// Fetch a string key from a section.
pub fn tget_s(
    t: &std::collections::BTreeMap<String, TomlValue>,
    sec: &str,
    key: &str,
) -> Result<String, String> {
    let v = if sec.is_empty() { &t } else { t[sec].as_table().unwrap() };
    let x = v.get(key).ok_or_else(|| format!("missing [{sec}] {key}"))?;
    x.as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("[{sec}] {key} must be a string"))
}

/// `%.12e` formatting of a float (C style two-digit exponent).
pub fn fmt_e12(x: f64) -> String {
    let mut s = format!("{x:.12e}");
    // Rust writes e-4 where C writes e-04: pad the exponent to two digits
    if let Some(epos) = s.rfind(['e', 'E']) {
        let (mant, exp) = s.split_at(epos);
        let (sign, digits) = exp[1..].split_at(1);
        if digits.len() < 2 {
            s = format!("{mant}e{sign}0{digits}");
        }
    }
    s
}

/// Write the (1,1) elements of the retarded and lesser components of
/// `G` in the domain `dom` into a text file (blueprint
/// `write_ness_gf`).
pub fn write_ness_gf(fname: &str, g: &NessHermMat, dom: FftDomain) -> Result<(), String> {
    use std::io::Write;
    let grid = &g.grid;
    let mut out = String::new();
    out.push_str(if dom == FftDomain::Time {
        "# t  Re G^R  Im G^R  Re G^<  Im G^<\n"
    } else {
        "# omega  Re G^R  Im G^R  Re G^<  Im G^<\n"
    });
    for n in 0..grid.nft {
        let x = if dom == FftDomain::Time {
            time_at(grid, n)
        } else {
            freq_at(grid, n)
        };
        let r = g.get_ret(n, dom)[0];
        let l = g.get_les(n, dom)[0];
        out.push_str(&format!(
            "{} {} {} {} {}\n",
            fmt_e12(x),
            fmt_e12(r.re),
            fmt_e12(r.im),
            fmt_e12(l.re),
            fmt_e12(l.im)
        ));
    }
    let mut f = std::fs::File::create(fname).map_err(|e| format!("{fname}: {e}"))?;
    f.write_all(out.as_bytes()).map_err(|e| format!("{fname}: {e}"))?;
    Ok(())
}

/// The output directory of a NESSi-style config resolved against the
/// config file directory.
pub fn ness_outdir(
    t: &std::collections::BTreeMap<String, TomlValue>,
    cfgfile: &str,
) -> Result<String, String> {
    let dir = tget_s(t, "output", "dir")?;
    let base = std::path::Path::new(cfgfile)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let d = std::path::Path::new(&dir);
    let d = if d.is_absolute() {
        d.to_path_buf()
    } else {
        base.join(d)
    };
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d.to_string_lossy().into_owned())
}

/// The `%.12e` row for the iterations file.
pub fn ness_iter_row(iter: i64, err: f64, dens: f64) -> String {
    format!("{} {} {}\n", fmt_e12(iter as f64), fmt_e12(err), fmt_e12(dens))
}

/// Placeholder keeping the C64 import used.
#[allow(dead_code)]
fn _c64(_: C64) {}
