//
// Project : KadanoffBaym
// File    : src/io/moving.rs
// Purpose : moving (truncated) window plain text I/O, the
//           `KB_MOVING_V1` format (blueprint julia/src/io/{show,read}.jl).
//           The ring content is written in the LOGICAL order; the file
//           tstp is the KB time step, the 0-based tstp of the types
//           converts at this boundary.
//

use crate::complex::Complex;
use crate::types::moving::{CfMoving, MovingGreen};
use super::IoError;

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
    let mut data = vec![Complex::ZERO; 2 * tcl * tcl * b];
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
                            Complex::new(re, im);
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
    let mut data = vec![Complex::ZERO; tcl * b];
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
                data[delti as usize * b + aidx * d + bidx] = Complex::new(re, im);
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
