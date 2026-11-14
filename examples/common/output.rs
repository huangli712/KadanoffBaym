//
// Deterministic plain-text output (the Rust port of
// julia/tutor/common/text_output.jl).
//
// Files are written to a temporary file in the same directory and then
// moved to the target, so an interrupted run never leaves a partial
// file.
//

use std::io::Write;
use std::path::Path;

/// C-style `%.17g`.
pub fn fmt_17g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // the decimal exponent from the %.16e form
    let e = format!("{x:.16e}");
    let exp: i32 = e
        .split('e')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if exp < -4 || exp >= 17 {
        // %e style, trailing zeros of the mantissa stripped
        let mut mantissa = e.clone();
        let epos = mantissa.find('e').unwrap();
        let suffix = mantissa.split_off(epos);
        if mantissa.contains('.') {
            while mantissa.ends_with('0') {
                mantissa.pop();
            }
            if mantissa.ends_with('.') {
                mantissa.pop();
            }
        }
        // C prints at least two exponent digits
        let (_, digits) = suffix.split_once('e').unwrap();
        let (sign, mag) = if let Some(m) = digits.strip_prefix('-') {
            ('-', m)
        } else {
            ('+', digits.strip_prefix('+').unwrap_or(digits))
        };
        if mag.len() < 2 {
            format!("{mantissa}e{sign}0{mag}")
        } else {
            format!("{mantissa}e{sign}{mag}")
        }
    } else {
        // %f style with 16 - exp decimals, trailing zeros stripped
        let prec = (16 - exp).max(0) as usize;
        let mut s = format!("{x:.prec$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
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
        let mut io = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f(&mut io)?;
        io.flush()?;
        drop(io);
        std::fs::rename(&tmp, path)
    };
    match write_and_move() {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
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
        body.push_str(
            &r.iter()
                .map(|x| fmt_17g(*x))
                .collect::<Vec<_>>()
                .join(" "),
        );
        body.push('\n');
    }
    atomic_write(path, |io| io.write_all(body.as_bytes()))
        .map_err(|e| e.to_string())
}

/// A row of `write_green_components`.
pub struct GreenRow<'a> {
    pub component: &'a str,
    pub i: i64,
    pub j: i64,
    pub a: usize,
    pub b: usize,
    pub value: kadanoffbaym::c64::C64,
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
    atomic_write(path, |io| io.write_all(body.as_bytes()))
        .map_err(|e| e.to_string())
}
