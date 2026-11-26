// Project : KadanoffBaym
// File    : util/output.rs
// Purpose : text output of the example programs (tables, green
//           components, NESS-style files)
//

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Result as IoResult;
use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::Path;

use std::fs::{self, remove_file, rename};
use std::process::id;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{FftDomain, SteadyGreen};

use super::config::ExampleConfig;
use super::toml::TomlValue;

use super::format::{fmt_e12, fmt_g17};
use super::toml::get_str;

/// A row of `write_green_components`.
pub struct GreenRow<'a> {
    pub component: &'a str,
    pub i: i64,
    pub j: i64,
    pub a: usize,
    pub b: usize,
    pub value: Complex,
}

/// Write through a temporary file and move it to `path`.
fn atomic_write<F: FnOnce(&mut dyn Write) -> IoResult<()>>(path: &str, f: F) -> IoResult<()> {
    let dir = Path::new(path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    if !dir.is_dir() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("no such directory {}", dir.display()),
        ));
    }
    let tmp = dir.join(format!(
        ".tmp_{}_{:x}",
        Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        id()
    ));
    let write_and_move = || -> IoResult<()> {
        let mut io = BufWriter::new(File::create(&tmp)?);
        f(&mut io)?;
        io.flush()?;
        drop(io);
        rename(&tmp, path)
    };
    match write_and_move() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = remove_file(&tmp);
            Err(e)
        }
    }
}

/// Write a table: `# 1:name1 2:name2 ...` followed by rows of real
/// numbers with 17 significant digits.
pub fn write_table(path: &str, names: &[&str], rows: &[Vec<f64>]) -> Result<(), String> {
    let header = format!(
        "# {}",
        names
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{}:{}", i + 1, n))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut body = String::new();
    body.push_str(&header);
    body.push('\n');
    for r in rows {
        if r.len() != names.len() {
            return Err(format!(
                "a row has {} entries, expected {}",
                r.len(),
                names.len()
            ));
        }
        if !r.iter().all(|x| x.is_finite()) {
            return Err("non-finite value in the output".to_string());
        }
        body.push_str(&r.iter().map(|x| fmt_g17(*x)).collect::<Vec<_>>().join(" "));
        body.push('\n');
    }
    atomic_write(path, |io| io.write_all(body.as_bytes())).map_err(|e| e.to_string())
}

/// Write selected Green's function components as
/// `component i j a b re im` (reference indices).
pub fn write_green_components(path: &str, rows: &[GreenRow<'_>]) -> Result<(), String> {
    let mut body = String::from("# component i j a b re im\n");
    for r in rows {
        if !r.value.re.is_finite() || !r.value.im.is_finite() {
            return Err("non-finite value in the output".to_string());
        }
        body.push_str(&format!(
            "{} {} {} {} {} {} {}\n",
            r.component,
            r.i,
            r.j,
            r.a,
            r.b,
            fmt_g17(r.value.re),
            fmt_g17(r.value.im)
        ));
    }
    atomic_write(path, |io| io.write_all(body.as_bytes())).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------
// NESS-style text output
// ------------------------------------------------------------------

/// Write the (1,1) elements of the retarded and lesser components of
/// `G` in the domain `dom` into a text file (reference
/// `write_ness_gf`).
pub fn write_ness_gf(fname: &str, g: &SteadyGreen, dom: FftDomain) -> Result<(), String> {
    let grid = &g.grid;
    let mut out = String::new();
    out.push_str(if dom == FftDomain::Time {
        "# t  Re G^R  Im G^R  Re G^<  Im G^<\n"
    } else {
        "# omega  Re G^R  Im G^R  Re G^<  Im G^<\n"
    });
    for n in 0..grid.nft {
        let x = if dom == FftDomain::Time {
            grid.time_at(n)
        } else {
            grid.freq_at(n)
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
    let mut f = fs::File::create(fname).map_err(|e| format!("{fname}: {e}"))?;
    f.write_all(out.as_bytes())
        .map_err(|e| format!("{fname}: {e}"))?;
    Ok(())
}

/// The output directory of an example configuration; a relative path is
/// taken relative to the configuration file.
pub fn example_outdir(cfg: &ExampleConfig) -> Result<String, String> {
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The output directory of a reference-style config resolved against the
/// config file directory.
pub fn ness_outdir(t: &BTreeMap<String, TomlValue>, cfgfile: &str) -> Result<String, String> {
    let dir = get_str(t, "output", "dir")?;
    let base = Path::new(cfgfile)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let d = Path::new(&dir);
    let d = if d.is_absolute() {
        d.to_path_buf()
    } else {
        base.join(d)
    };
    fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d.to_string_lossy().into_owned())
}

/// The `%.12e` row for the iterations file.
pub fn ness_iter_row(iter: i64, err: f64, dens: f64) -> String {
    format!(
        "{} {} {}\n",
        fmt_e12(iter as f64),
        fmt_e12(err),
        fmt_e12(dens)
    )
}
