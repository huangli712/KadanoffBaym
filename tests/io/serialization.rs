// Project : KadanoffBaym
// File    : tests/io/serialization.rs
// Purpose : the plain-text serialization of the Green's-function types
//

use std::fmt::Display;
use std::str::FromStr;

use std::fs::read_to_string;

use kadanoffbaym::io::IoError;

use kadanoffbaym::io::{green_partial_from_str, read_contour};

use super::util::{tmpdir, write_file};

use super::util::prelude::*;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

fn fill_inout_green(g: &mut ContourGreen, complx: bool) {
    let (nt, ntau) = (g.nt as i64, g.ntau as i64);
    for q in 0..=ntau {
        g.mat_raw_block_mut(q)
            .fill(sample_value(11.0 + q as f64, complx));
    }
    for i in 0..=nt {
        for j in 0..=i {
            g.ret_raw_block_mut(i, j)
                .fill(sample_value(111.0 + 10.0 * i as f64 + j as f64, complx));
        }
        for q in 0..=ntau {
            g.lmix_raw_block_mut(i, q)
                .fill(sample_value(211.0 + 10.0 * i as f64 + q as f64, complx));
        }
        for j in i..=nt {
            g.less_raw_block_mut(i, j)
                .fill(sample_value(311.0 + 10.0 * i as f64 + j as f64, complx));
        }
    }
}

/// Helper: read a contour file into an existing contour.
fn read_contour_into(path: &str, c: &mut Contour) -> Result<(), IoError> {
    let text = read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("cannot read {path}: {e}"),
    })?;
    *c = Contour::from_str(&text)?;
    Ok(())
}

/// The read/write round-trip checks shared by every serialized type.
fn roundtrip_checks<T>(dir: &str, tag: &str, c1: T, mut c2: T, mut c3: T)
where
    T: PartialEq + Display + FromStr<Err = IoError>,
{
    assert!(c1 != c2 && c1 != c3);
    let p = write_file(dir, &format!("{tag}.data"), &c1.to_string());
    let text = read_to_string(&p).unwrap();
    c2 = T::from_str(&text).unwrap();
    c3 = T::from_str(&text).unwrap();
    assert!(c1 == c2, "{tag} eq2");
    assert!(c1 == c3, "{tag} eq3");
}

fn sample_value(seed: f64, complx: bool) -> Complex {
    if complx {
        Complex::new(seed, seed + 1.0)
    } else {
        Complex::new(seed, 0.0)
    }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn contour_read_write() {
    let dir = tmpdir("cn");
    let c1 = Contour::new(200, 1000, 1, 1, 5.0, 4.0);
    let mut c2 = Contour::new(3, 3, 1, 1, 2.0, 2.0);
    let mut c3 = Contour::new(4, 4, 1, 1, 3.0, 3.0);
    assert!(c1 != c2 && c1 != c3);
    let p = write_file(&dir, "Cn.data", &(c1).to_string());
    read_contour_into(&p, &mut c2).unwrap();
    let text = read_to_string(&p).unwrap();
    // stream-based read
    Contour::from_str(&text).unwrap();
    c3 = Contour::from_str(&text).unwrap();
    assert!(c1 == c2);
    assert!(c1 == c3);
    // missing file
    assert!(read_contour(&format!("{dir}/Cn.wrong")).is_err());
}

#[test]
fn gmat_read_write() {
    let dir = tmpdir("gmat");
    let m1 = MatComp::new(1000, 1, 1, Complex::new(1.0, -0.3));
    let m2 = MatComp::new(1000, 1, 1, Complex::new(0.3, 0.3));
    let m3 = MatComp::new(1000, 1, 1, Complex::new(1.2, -0.3));
    roundtrip_checks(&dir, "Gmat", m1, m2, m3);
    let r1 = MatComp::new(1000, 1, 1, Complex::new(1.0, 0.0));
    let r2 = MatComp::new(1000, 1, 1, Complex::new(0.3, 0.0));
    let r3 = MatComp::new(1000, 1, 1, Complex::new(1.2, 0.0));
    roundtrip_checks(&dir, "Gmatr", r1, r2, r3);
}

#[test]
fn gret_read_write() {
    let dir = tmpdir("gret");
    let m1 = RetComp::new(200, 1, 1, Complex::new(1.0, -0.3));
    let m2 = RetComp::new(200, 1, 1, Complex::new(0.3, 0.3));
    let m3 = RetComp::new(200, 1, 1, Complex::new(1.2, -0.3));
    roundtrip_checks(&dir, "Gret", m1, m2, m3);
    let r1 = RetComp::new(200, 1, 1, Complex::new(1.0, 0.0));
    let r2 = RetComp::new(200, 1, 1, Complex::new(0.3, 0.0));
    let r3 = RetComp::new(200, 1, 1, Complex::new(1.2, 0.0));
    roundtrip_checks(&dir, "Gretr", r1, r2, r3);
}

#[test]
fn glmix_read_write() {
    let dir = tmpdir("glmix");
    let m1 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.0, -0.3));
    let m2 = LmixComp::new(200, 1000, 1, 1, Complex::new(0.3, 0.3));
    let m3 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.2, -0.3));
    roundtrip_checks(&dir, "Glmix", m1, m2, m3);
    let r1 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.0, 0.0));
    let r2 = LmixComp::new(200, 1000, 1, 1, Complex::new(0.3, 0.0));
    let r3 = LmixComp::new(200, 1000, 1, 1, Complex::new(1.2, 0.0));
    roundtrip_checks(&dir, "Glmixr", r1, r2, r3);
}

#[test]
fn gless_read_write() {
    let dir = tmpdir("gless");
    let m1 = LessComp::new(200, 1, 1, Complex::new(1.0, -0.3));
    let m2 = LessComp::new(200, 1, 1, Complex::new(0.3, 0.3));
    let m3 = LessComp::new(200, 1, 1, Complex::new(1.2, -0.3));
    roundtrip_checks(&dir, "Gless", m1, m2, m3);
    let r1 = LessComp::new(200, 1, 1, Complex::new(1.0, 0.0));
    let r2 = LessComp::new(200, 1, 1, Complex::new(0.3, 0.0));
    let r3 = LessComp::new(200, 1, 1, Complex::new(1.2, 0.0));
    roundtrip_checks(&dir, "Glessr", r1, r2, r3);
}

#[test]
fn partial_timestep_plain_text_read() {
    let dir = tmpdir("partial");
    // (sign, ndim, initial, sentinel) reference cases
    let cases: [(Sign, i64, Complex, Complex); 4] = [
        (
            Sign::Fermi,
            1,
            Complex::new(1.0, 2.0),
            Complex::new(-91.0, -92.0),
        ),
        (
            Sign::Bose,
            1,
            Complex::new(1.0, 0.0),
            Complex::new(-91.0, 0.0),
        ),
        (
            Sign::Fermi,
            2,
            Complex::new(1.0, 0.0),
            Complex::new(-91.0, 0.0),
        ),
        (
            Sign::Bose,
            2,
            Complex::new(1.0, 2.0),
            Complex::new(-91.0, -92.0),
        ),
    ];
    let tstp_1based = 2i64;
    let tstp = tstp_1based - 1; // row limit here
    for (case_index, &(sign, ndim, initial, sentinel)) in cases.iter().enumerate() {
        let c = Contour::new(3, 4, ndim, ndim, 1.0, 2.0);
        let mut source = ContourGreen::new(&c, sign, Complex::ZERO);
        fill_inout_green(&mut source, initial.im != 0.0);
        let mut target = ContourGreen::new(&c, sign, sentinel);
        let p = write_file(
            &dir,
            &format!("partial_{}.data", case_index + 1),
            &(source).to_string(),
        );
        let text = read_to_string(&p).unwrap();
        green_partial_from_str(&text, &mut target, tstp).unwrap();

        let ntau = c.ntau;
        let nt = c.nt;
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
