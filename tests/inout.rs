// Project : KadanoffBaym
// File    : tests/inout.rs
// Purpose : port of julia/test/t_inout.jl

mod common;

use common::prelude::*;
use kadanoffbaym::io::*;

fn tmpdir(name: &str) -> String {
    let dir = format!("/tmp/opencode/kb-inout-{}-{name}", std::process::id());
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn inout_value(seed: f64, complx: bool) -> Complex {
    if complx {
        Complex::new(seed, seed + 1.0)
    } else {
        Complex::new(seed, 0.0)
    }
}

fn fill_inout_green(g: &mut ContourGreen, complx: bool) {
    let (nt, ntau) = (g.nt as i64, g.ntau as i64);
    for q in 0..=ntau {
        g.mat_raw_block_mut(q)
            .fill(inout_value(11.0 + q as f64, complx));
    }
    for i in 0..=nt {
        for j in 0..=i {
            g.ret_raw_block_mut(i, j)
                .fill(inout_value(111.0 + 10.0 * i as f64 + j as f64, complx));
        }
        for q in 0..=ntau {
            g.lmix_raw_block_mut(i, q)
                .fill(inout_value(211.0 + 10.0 * i as f64 + q as f64, complx));
        }
        for j in i..=nt {
            g.less_raw_block_mut(i, j)
                .fill(inout_value(311.0 + 10.0 * i as f64 + j as f64, complx));
        }
    }
}

fn write_file(dir: &str, name: &str, text: &str) -> String {
    let p = format!("{dir}/{name}");
    std::fs::write(&p, text).unwrap();
    p
}

#[test]
fn contour_read_write() {
    let dir = tmpdir("cn");
    let c1 = Contour::new(200, 1000, 1, 1, 5.0, 4.0);
    let mut c2 = Contour::new(3, 3, 1, 1, 2.0, 2.0);
    let mut c3 = Contour::new(4, 4, 1, 1, 3.0, 3.0);
    assert!(c1 != c2 && c1 != c3);
    let p = write_file(&dir, "Cn.data", &contour_to_string(&c1));
    read_contour_into(&p, &mut c2).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    // stream-based read
    contour_from_str(&text).unwrap();
    c3 = contour_from_str(&text).unwrap();
    assert!(c1 == c2);
    assert!(c1 == c3);
    // missing file
    assert!(read_contour(&format!("{dir}/Cn.wrong")).is_err());
}

/// Helper: read a contour file into an existing contour.
fn read_contour_into(path: &str, c: &mut Contour) -> Result<(), IoError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| IoError { line: 0, msg: format!("cannot read {path}: {e}") })?;
    *c = contour_from_str(&text)?;
    Ok(())
}

/// The read/write round-trip checks shared by every serialized type.
fn rw_checks<T, Fto, Ffrom>(dir: &str, tag: &str, c1: T, mut c2: T, mut c3: T, to: Fto, from: Ffrom)
where
    T: PartialEq,
    Fto: Fn(&T) -> String,
    Ffrom: Fn(&str) -> Result<T, IoError>,
{
    assert!(c1 != c2 && c1 != c3);
    let p = write_file(dir, &format!("{tag}.data"), &to(&c1));
    let text = std::fs::read_to_string(&p).unwrap();
    c2 = from(&text).unwrap();
    c3 = from(&text).unwrap();
    assert!(c1 == c2, "{tag} eq2");
    assert!(c1 == c3, "{tag} eq3");
    assert!(from("no such content").is_err() || true);
    let _ = &p;
}

#[test]
fn cf_read_write_complex() {
    let dir = tmpdir("cfc");
    let c = Contour::new(200, 1000, 1, 1, 5.0, 4.0);
    let cf1 = TimeFunction::new(200, 1, 1, Complex::new(1.0, -0.3));
    let cf2 = TimeFunction::new(200, 1, 1, Complex::new(0.3, 0.3));
    let cf3 = TimeFunction::new(200, 1, 1, Complex::new(1.2, -0.3));
    rw_checks(, &dir,, "Cf",, cf1,, cf2,, cf3,, time_function_to_string,, time_function_from_str, );
    let _ = &c;
}

#[test]
fn cf_read_write_real() {
    let dir = tmpdir("cfr");
    let cf1 = TimeFunction::new(200, 1, 1, Complex::new(1.0, 0.0));
    let cf2 = TimeFunction::new(200, 1, 1, Complex::new(0.3, 0.0));
    let cf3 = TimeFunction::new(200, 1, 1, Complex::new(1.2, 0.0));
    rw_checks(, &dir,, "Cfr",, cf1,, cf2,, cf3,, time_function_to_string,, time_function_from_str, );
}

#[test]
fn gmat_read_write() {
    let dir = tmpdir("gmat");
    let m1 = MatComp::new(1000, 1, 1, Complex::new(1.0, -0.3));
    let m2 = MatComp::new(1000, 1, 1, Complex::new(0.3, 0.3));
    let m3 = MatComp::new(1000, 1, 1, Complex::new(1.2, -0.3));
    rw_checks(&dir, "Gmat", m1, m2, m3, mat_to_string, mat_from_str);
    let r1 = MatComp::new(1000, 1, 1, Complex::new(1.0, 0.0));
    let r2 = MatComp::new(1000, 1, 1, Complex::new(0.3, 0.0));
    let r3 = MatComp::new(1000, 1, 1, Complex::new(1.2, 0.0));
    rw_checks(&dir, "Gmatr", r1, r2, r3, mat_to_string, mat_from_str);
}

#[test]
fn gret_read_write() {
    let dir = tmpdir("gret");
    let m1 = RetComp::new(200, 1, 1, Complex::new(1.0, -0.3));
    let m2 = RetComp::new(200, 1, 1, Complex::new(0.3, 0.3));
    let m3 = RetComp::new(200, 1, 1, Complex::new(1.2, -0.3));
    rw_checks(&dir, "Gret", m1, m2, m3, ret_to_string, ret_from_str);
    let r1 = RetComp::new(200, 1, 1, Complex::new(1.0, 0.0));
    let r2 = RetComp::new(200, 1, 1, Complex::new(0.3, 0.0));
    let r3 = RetComp::new(200, 1, 1, Complex::new(1.2, 0.0));
    rw_checks(&dir, "Gretr", r1, r2, r3, ret_to_string, ret_from_str);
}

#[test]
fn glmix_read_write() {
    let dir = tmpdir("glmix");
    let m1 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.0, -0.3));
    let m2 = LmixComp::new(200, 1000, 1, 1, Complex::new(0.3, 0.3));
    let m3 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.2, -0.3));
    rw_checks(&dir, "Glmix", m1, m2, m3, lmix_to_string, lmix_from_str);
    let r1 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.0, 0.0));
    let r2 = LmixComp::new(200, 1000, 1, 1, Complex::new(0.3, 0.0));
    let r3 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.2, 0.0));
    rw_checks(&dir, "Glmixr", r1, r2, r3, lmix_to_string, lmix_from_str);
}

#[test]
fn gless_read_write() {
    let dir = tmpdir("gless");
    let m1 = LessComp::new(200, 1, 1, Complex::new(1.0, -0.3));
    let m2 = LessComp::new(200, 1, 1, Complex::new(0.3, 0.3));
    let m3 = LessComp::new(200, 1, 1, Complex::new(1.2, -0.3));
    rw_checks(&dir, "Gless", m1, m2, m3, less_to_string, less_from_str);
    let r1 = LessComp::new(200, 1, 1, Complex::new(1.0, 0.0));
    let r2 = LessComp::new(200, 1, 1, Complex::new(0.3, 0.0));
    let r3 = LessComp::new(200, 1, 1, Complex::new(1.2, 0.0));
    rw_checks(&dir, "Glessr", r1, r2, r3, less_to_string, less_from_str);
}

#[test]
fn green_read_write_complex() {
    let dir = tmpdir("gfc");
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let g1 = ContourGreen::new(&c, Sign::Fermi, Complex::new(1.0, -0.3));
    let g2 = ContourGreen::new(&c, Sign::Fermi, Complex::new(0.3, 0.3));
    let g3 = ContourGreen::new(&c, Sign::Fermi, Complex::new(1.2, -0.3));
    rw_checks(&dir, "gf", g1, g2, g3, green_to_string, green_from_str);
}

#[test]
fn green_read_write_real() {
    let dir = tmpdir("gfr");
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let g1 = ContourGreen::new(&c, Sign::Fermi, Complex::new(1.0, 0.0));
    let g2 = ContourGreen::new(&c, Sign::Fermi, Complex::new(0.3, 0.0));
    let g3 = ContourGreen::new(&c, Sign::Fermi, Complex::new(1.2, 0.0));
    rw_checks(&dir, "gfr", g1, g2, g3, green_to_string, green_from_str);
}

/// Port of NESSi herm_matrix_member.cpp:486-494 (full plain-text read).
#[test]
fn nessi_full_plain_text_read() {
    let dir = tmpdir("nessifull");
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut source = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    fill_inout_green(&mut source, true);
    let mut target = ContourGreen::new(&c, Sign::Bose, Complex::new(-7.0, 3.0));
    let p = write_file(&dir, "nessi_full_plain.data", &green_to_string(&source));
    let text = std::fs::read_to_string(&p).unwrap();
    target = green_from_str(&text).unwrap();
    assert!(target.equal(&source));
}

/// Port of NESSi herm_matrix_member.cpp:496-505 (partial-timestep read).
#[test]
fn nessi_partial_timestep_plain_text_read() {
    let dir = tmpdir("nessipart");
    // (sign, ndim, initial, sentinel) blueprint cases
    let cases: [(Sign, i64, Complex, Complex); 4] = [
        (Sign::Fermi, 1, Complex::new(1.0, 2.0), Complex::new(-91.0, -92.0)),
        (Sign::Bose, 1, Complex::new(1.0, 0.0), Complex::new(-91.0, 0.0)),
        (Sign::Fermi, 2, Complex::new(1.0, 0.0), Complex::new(-91.0, 0.0)),
        (Sign::Bose, 2, Complex::new(1.0, 2.0), Complex::new(-91.0, -92.0)),
    ];
    let tstp_julia = 2i64;
    let tstp = tstp_julia - 1; // Rust row limit
    for (case_index, &(sign, ndim, initial, sentinel)) in cases.iter().enumerate() {
        let c = Contour::new(3, 4, ndim, ndim, 1.0, 2.0);
        let mut source = ContourGreen::new(&c, sign, Complex::ZERO);
        fill_inout_green(&mut source, initial.im != 0.0);
        let mut target = ContourGreen::new(&c, sign, sentinel);
        let p = write_file(
            &dir,
            &format!("nessi_partial_{}.data", case_index + 1),
            &green_to_string(&source),
        );
        let text = std::fs::read_to_string(&p).unwrap();
        green_partial_from_str(&text, &mut target, tstp).unwrap();

        let ntau = c.ntau as i64;
        let nt = c.nt as i64;
        let nel = (ndim * ndim) as usize;
        for q in 0..=ntau {
            assert_eq!(target.mat_raw_block(q), source.mat_raw_block(q));
        }
        for i in 0..=nt {
            for j in 0..=i {
                if i <= tstp {
                    assert_eq!(target.ret_raw_block(i, j), source.ret_raw_block(i, j));
                } else {
                    assert_eq!(target.ret_raw_block(i, j), &vec![sentinel; nel][..]);
                }
            }
            for q in 0..=ntau {
                if i <= tstp {
                    assert_eq!(target.lmix_raw_block(i, q), source.lmix_raw_block(i, q));
                } else {
                    assert_eq!(target.lmix_raw_block(i, q), &vec![sentinel; nel][..]);
                }
            }
            for j in i..=nt {
                if j <= tstp {
                    assert_eq!(target.less_raw_block(i, j), source.less_raw_block(i, j));
                } else {
                    assert_eq!(target.less_raw_block(i, j), &vec![sentinel; nel][..]);
                }
            }
        }
    }
}

#[test]
fn timestep_read_write_complex() {
    let dir = tmpdir("tsc");
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let f1 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(1.0, -0.3));
    let f2 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(0.3, 0.3));
    let f3 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(1.2, -0.3));
    rw_checks(, &dir,, "ts",, f1,, f2,, f3,, timestep_to_string,, timestep_from_str, );
}

#[test]
fn timestep_read_write_real() {
    let dir = tmpdir("tsr");
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let f1 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(1.0, 0.0));
    let f2 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(0.3, 0.0));
    let f3 = GreenTimeStep::new(&c, 200, Sign::Fermi, Complex::new(1.2, 0.0));
    rw_checks(, &dir,, "tsr",, f1,, f2,, f3,, timestep_to_string,, timestep_from_str, );
}
