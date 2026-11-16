// Project : KadanoffBaym
// File    : util/output.rs
// Purpose : text output of the example programs (tables, green
//           components)
//

use std::fs;
use std::path::Path;
use std::io::Write;

use super::format::fmt_17g;

/// A row of `write_green_components`.
pub struct GreenRow<'a> {
    pub component: &'a str,
    pub i: i64,
    pub j: i64,
    pub a: usize,
    pub b: usize,
    pub value: kadanoffbaym::complex::Complex,
}

/// Write through a temporary file and move it to `path`.
fn atomic_write<F: FnOnce(&mut dyn Write) -> std::io::Result<()>>(
    path: &str,
    f: F,
) -> std::io::Result<()> {
    let dir = Path::new(path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    if !dir.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no such directory {}", dir.display()),
        ));
    }
    let tmp = dir.join(format!(
        ".tmp_{}_{:x}",
        Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        std::process::id()
    ));
    let write_and_move = || -> std::io::Result<()> {
        let mut io = std::io::BufWriter::new(fs::File::create(&tmp)?);
        f(&mut io)?;
        io.flush()?;
        drop(io);
        fs::rename(&tmp, path)
    };
    match write_and_move() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
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
        body.push_str(&r.iter().map(|x| fmt_17g(*x)).collect::<Vec<_>>().join(" "));
        body.push('\n');
    }
    atomic_write(path, |io| io.write_all(body.as_bytes())).map_err(|e| e.to_string())
}

/// Write selected Green's function components as
/// `component i j a b re im` (NESSi indices).
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
            fmt_17g(r.value.re),
            fmt_17g(r.value.im)
        ));
    }
    atomic_write(path, |io| io.write_all(body.as_bytes())).map_err(|e| e.to_string())
}
