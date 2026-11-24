// Project : KadanoffBaym
// File    : tests/pseudo/reference.rs
// Purpose : point-wise comparison of the pseudo-particle algorithms with
//           the reference data
//

use std::collections::BTreeSet;

use std::fs::read_dir;

use kadanoffbaym::convolution::{ConvWorkspace, FOp};
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{
    convolution_density_matrix, incr_less_adv_buf, incr_lmix_rmix_buf, pseudo_convolution,
};
use kadanoffbaym::dyson::pseudo_dyson;
use kadanoffbaym::observables::{pseudo_density, pseudo_density_matrix};
use kadanoffbaym::variants::pseudo_timestep;
use kadanoffbaym::vie2::pseudo_vie2;

use super::util::{RefData, RefRow};

use super::structs::pseudo_density_matrix_step;
use super::util::{c, read_ref};

use super::util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const BETA: f64 = 2.0;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/test/pseudo");

const HH: f64 = 0.02;

const LAM0: f64 = 0.5;

const NT: i64 = 10;

const NTAU: i64 = 30;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Fill all the stored blocks .
fn build_pseudo(g: &mut PseudoGreen, w: i64, s: f64) {
    let d = g.size1 as i64;
    let dtau = BETA / NTAU as f64;
    for q in 0..=NTAU {
        let v = reference_block(w, 0, q as f64 * dtau, 0.0, d, s);
        g.mat.block_mut(q as usize).copy_from_slice(&v);
    }
    for i in 0..=NT {
        for j in 0..=i {
            let v = reference_block(w, 1, i as f64 * HH, j as f64 * HH, d, s);
            g.ret.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
        for q in 0..=NTAU {
            let v = reference_block(w, 2, i as f64 * HH, q as f64 * dtau, d, s);
            g.lmix.raw_block_at_mut(i, q).copy_from_slice(&v);
        }
        for j in i..=NT {
            let v = reference_block(w, 3, i as f64 * HH, j as f64 * HH, d, s);
            g.less.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
    }
}

fn data_files(prefix: &str) -> Vec<String> {
    let mut out: Vec<String> = read_dir(DATA)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|f| f.starts_with(prefix) && f.ends_with(".dat"))
        .collect();
    out.sort();
    out
}

fn hamiltonian_function(d: usize) -> TimeFunction {
    let mut f = TimeFunction::new(NT as usize, d, d, Complex::ZERO);
    f.block_mut(MATSUBARA)
        .copy_from_slice(&reference_hamiltonian(d, 0.0));
    for n in 0..=NT {
        f.block_mut(n)
            .copy_from_slice(&reference_hamiltonian(d, n as f64 * HH));
    }
    f
}

fn middle_function(d: usize) -> TimeFunction {
    let mut f = TimeFunction::new(NT as usize, d, d, Complex::ZERO);
    f.block_mut(MATSUBARA)
        .copy_from_slice(&reference_middle(d, -0.5));
    for n in 0..=NT {
        f.block_mut(n)
            .copy_from_slice(&reference_middle(d, n as f64 * HH));
    }
    f
}

/// The pseudo storage entry , `gc` is the
/// transferred standard green (for the derived `gtr` access).
fn pseudo_value(g: &PseudoGreen, gc: &ContourGreen, r: &RefRow) -> Option<Complex> {
    let d = g.size1;
    let q = r.a as usize * d + r.b as usize;
    match r.component.as_str() {
        "mat" => Some(g.mat.block(r.i as usize)[q]),
        "gtr" => Some(gc.ret_block(r.i, r.j).raw()[q]),
        "tv" => Some(g.lmix.raw_block_at(r.i, r.j)[q]),
        "les" => Some(g.less.raw_block_at(r.i, r.j)[q]),
        _ => None,
    }
}

fn reference_block(w: i64, cc: i64, x: f64, y: f64, d: i64, s: f64) -> Vec<Complex> {
    (0..d * d)
        .map(|q| {
            let (a, b) = (q / d, q % d);
            reference_value(w, cc, x, y, a, b) * c(s, 0.0)
        })
        .collect()
}

fn reference_contour(d: i64) -> Contour {
    Contour::new(NT, NTAU, d, d, NT as f64 * HH, BETA)
}

fn reference_hamiltonian(d: usize, t: f64) -> Vec<Complex> {
    let mut v = vec![c(0.3, 0.0), c(0.2, -0.1), c(0.2, 0.1), c(-0.4, 0.0)];
    for i in 0..2 {
        v[i * 2 + i] += c(0.3 * (2.0 * t).sin(), 0.0);
    }
    v.truncate(d * d);
    v
}

fn reference_meta(refdata: &RefData) -> (usize, i64, Sign) {
    let d: usize = refdata.metadata["size1"].parse().unwrap();
    let k: i64 = refdata.metadata["k"].parse().unwrap();
    let sig: i64 = refdata.metadata["sig"].parse().unwrap();
    (d, k, Sign::from_i64(sig))
}

fn reference_middle(d: usize, t: f64) -> Vec<Complex> {
    let v = vec![
        c(1.0 + 0.2 * t, 0.0),
        c(0.1, 0.3 * t),
        c(-0.2, 0.05),
        c(0.8, 0.0),
    ];
    let mut out = v;
    out.truncate(d * d);
    out
}

/// The closed-form input .
fn reference_value(w: i64, cc: i64, x: f64, y: f64, a: i64, b: i64) -> Complex {
    let amp = 0.3 + 0.1 * w as f64 + 0.05 * cc as f64 + 0.1 * a as f64 - 0.07 * b as f64;
    let env = (-0.2 * x - 0.15 * y).exp();
    c(
        amp * env * (0.7 * x + 0.3 * y + 0.2 * (a + b) as f64 + w as f64).cos(),
        amp * env * (0.5 * x - 0.4 * y + 0.1 * (a * b) as f64 + cc as f64).sin(),
    )
}

/// The lesser start-up source blocks .
fn start_source(
    g: &PseudoGreen,
    a_p: &PseudoGreen,
    acc_p: &PseudoGreen,
    integ: &Integrator,
    alpha: Complex,
    refdata: &RefData,
    nonzero: bool,
) -> f64 {
    let k = integ.k;
    let d = g.size1;
    let rows: Vec<&RefRow> = refdata
        .rows
        .iter()
        .filter(|r| r.component == "les_start_source")
        .collect();
    assert_eq!(rows.len(), d * d * ((k * (k + 1) / 2) as usize));
    let (a_c, acc_c, g_c) = (a_p.to_contour(), acc_p.to_contour(), g.to_contour());
    let mut ws = ConvWorkspace::new(d, d, d, d);
    let mut err = 0.0f64;
    let mut nfail = 0;
    for n in 0..k {
        // the reference tstp = n + 1 (KB) is my time step n
        let tstp = n;
        let mut buf: Vec<Vec<Complex>> = (0..=k as usize)
            .map(|_| vec![Complex::ZERO; d * d])
            .collect();
        incr_lmix_rmix_buf(&mut buf, tstp, &a_c, None, &g_c, integ, g.sign, BETA, alpha);
        incr_less_adv_buf(
            &mut buf,
            tstp,
            &a_c,
            &acc_c,
            &FOp::None,
            &g_c,
            &g_c,
            integ,
            BETA,
            HH,
            alpha,
            &mut ws,
            false,
        );
        for r in &rows {
            if r.j != n {
                continue;
            }
            let v = buf[r.i as usize][r.a as usize * d + r.b as usize];
            err = err.max((v - r.value).abs());
            if (v - r.value).abs() > 1e-10 {
                nfail += 1;
            }
        }
    }
    if nonzero {
        let m = rows.iter().map(|r| r.value.abs()).fold(0.0, f64::max);
        assert!(m > 1e-6);
    }
    assert_eq!(nfail, 0, "start-source mismatches in {}", refdata.path);
    err
}

/// Compare all the storage rows, returns the maximal error.
fn storage_error(g: &PseudoGreen, refdata: &RefData) -> f64 {
    let gc = g.to_contour();
    let mut err = 0.0f64;
    let mut nfail = 0;
    for r in refdata.rows.iter() {
        if !matches!(r.component.as_str(), "mat" | "gtr" | "tv" | "les") {
            continue;
        }
        let v = pseudo_value(g, &gc, r).unwrap();
        err = err.max((v - r.value).abs());
        if (v - r.value).abs() > 1e-10 {
            nfail += 1;
        }
    }
    assert_eq!(nfail, 0, "storage mismatches in {}", refdata.path);
    err
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn pseudo_reference_storage() {
    let refdata = read_ref(&format!("{DATA}/pseudo_storage.dat")).unwrap();
    let mut g = PseudoGreen::new(&reference_contour(2), Sign::Fermi, Complex::ZERO);
    build_pseudo(&mut g, 0, 1.0);
    let err = storage_error(&g, &refdata);
    // derived accessors against their definitions (the reference
    // checks the matm/rmix derived views, which are trivial identities
    // of the accessors; the rmix check is kept via my accessor)
    let gc = g.to_contour();
    let xi = g.sign.as_i64() as f64;
    for q in 0..=NTAU {
        for i in 0..=NT {
            // G^ceil(tau, t) = -xi G^](t, beta - tau)^dagger
            let xb = gc.rmix_block(q, i);
            let x = xb.raw();
            let y = g.lmix.raw_block_at(i, NTAU - q);
            for a in 0..2usize {
                for b in 0..2usize {
                    let want = c(-xi, 0.0) * y[b * 2 + a].conj();
                    assert!((x[a * 2 + b] - want).abs() < 1e-12);
                }
            }
        }
    }
    // G^>(i,j) = -G^>(j,i)^dagger for j > i (through the derived access)
    for i in 0..=NT {
        for j in (i + 1)..=NT {
            for a in 0..2usize {
                for b in 0..2usize {
                    let v = gc.ret_block(i, j).raw()[a * 2 + b];
                    let w = gc.ret_block(j, i).raw()[b * 2 + a];
                    assert!((v + w.conj()).abs() < 1e-12);
                }
            }
        }
    }
    assert!(err < 1e-12, "err {err}");
}

#[test]
fn pseudo_reference_convolution() {
    let files = data_files("pseudo_convolution");
    assert_eq!(files.len(), 5);
    for name in &files {
        let refdata = read_ref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = reference_meta(&refdata);
        let cc = reference_contour(d as i64);
        let mk = |w: i64| {
            let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
            build_pseudo(&mut g, w, 1.0);
            g
        };
        let (a, acc, b, bcc) = (mk(1), mk(2), mk(3), mk(4));
        let f = if refdata.metadata["f"] == "yes" {
            Some(middle_function(d))
        } else {
            None
        };
        let integ = Integrator::new(k);
        let mut x = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_convolution(&mut x, &a, &acc, f.as_ref(), &b, &bcc, &integ, BETA, HH);
        let mut err = storage_error(&x, &refdata);
        // the density-matrix rows
        let (ac, accc, bc, bccc) = (
            a.to_contour(),
            acc.to_contour(),
            b.to_contour(),
            bcc.to_contour(),
        );
        let mut nfail = 0;
        let mut count = 0;
        for r in refdata.rows.iter().filter(|r| r.component == "density") {
            count += 1;
            let tstp = if r.i == -1 { MATSUBARA } else { r.i };
            let rho = convolution_density_matrix(
                tstp,
                &ac,
                &accc,
                f.as_ref(),
                &bc,
                &bccc,
                &integ,
                BETA,
                HH,
            );
            let v = rho.raw()[r.a as usize * d + r.b as usize];
            err = err.max((v - r.value).abs());
            if (v - r.value).abs() > 1e-10 {
                nfail += 1;
            }
        }
        assert_eq!(count, d * d * (NT as usize + 2));
        assert_eq!(nfail, 0, "density mismatches in {name}");
        assert!(err < 1e-12, "{name}: err {err}");
    }
}

#[test]
fn pseudo_reference_dyson() {
    let files = data_files("pseudodyson");
    assert_eq!(files.len(), 7);
    for name in &files {
        let refdata = read_ref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = reference_meta(&refdata);
        let cc = reference_contour(d as i64);
        let mut sigma = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let formula = refdata.metadata["sigma"] == "formula";
        if formula {
            build_pseudo(&mut sigma, 5, 0.2);
        }
        let integ = Integrator::new(k);
        let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_dyson(
            &mut g,
            LAM0,
            &hamiltonian_function(d),
            &sigma,
            &integ,
            BETA,
            HH,
        );
        let mut err = storage_error(&g, &refdata);
        // the density-matrix rows (r.i == -1 is the Matsubara axis)
        let drows: Vec<&RefRow> = refdata
            .rows
            .iter()
            .filter(|r| r.component == "density")
            .collect();
        assert_eq!(drows.len(), d * d * (NT as usize + 2));
        let mut nfail = 0;
        for r in &drows {
            let tstp = if r.i == -1 { MATSUBARA } else { r.i };
            let rho = pseudo_density_matrix(&g, tstp);
            let v = rho[r.a as usize * d + r.b as usize];
            err = err.max((v - r.value).abs());
            if (v - r.value).abs() > 1e-10 {
                nfail += 1;
            }
            // the time slice and the trace agree
            let x = pseudo_timestep(&g, tstp);
            let rho_step = pseudo_density_matrix_step(&x, tstp);
            for q in 0..(d * d) {
                assert_eq!(rho_step[q], rho[q]);
            }
            let trace = pseudo_density(&g, tstp);
            let want: Complex = (0..d).map(|i| rho[i * d + i]).sum();
            assert_eq!(trace, want);
        }
        assert_eq!(nfail, 0, "density mismatches in {name}");
        err = err.max(start_source(
            &g,
            &sigma,
            &sigma,
            &integ,
            Complex::ONE,
            &refdata,
            formula,
        ));
        assert!(err < 1e-10, "{name}: err {err}");
    }
}

#[test]
fn pseudo_reference_vie2() {
    let files = data_files("pseudo_vie2");
    assert_eq!(files.len(), 5);
    for name in &files {
        let refdata = read_ref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = reference_meta(&refdata);
        let cc = reference_contour(d as i64);
        let mut f = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let mut fcc = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let formula = refdata.metadata["kernel"] == "formula";
        if formula {
            build_pseudo(&mut f, 6, 0.2);
            build_pseudo(&mut fcc, 7, 0.2);
        }
        let mut q = PseudoGreen::new(&cc, sig, Complex::ZERO);
        build_pseudo(&mut q, 8, 1.0);
        let integ = Integrator::new(k);
        let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_vie2(&mut g, &f, &fcc, &q, &integ, BETA, HH);
        let mut err = storage_error(&g, &refdata);
        if formula {
            err = err.max(start_source(
                &g,
                &f,
                &fcc,
                &integ,
                -Complex::ONE,
                &refdata,
                true,
            ));
        }
        assert!(err < 1e-10, "{name}: err {err}");
    }
}

#[test]
fn pseudo_reference_all_files_consumed() {
    let all = data_files("");
    assert_eq!(all.len(), 18);
    let prefixes = [
        "pseudo_storage",
        "pseudo_convolution",
        "pseudodyson",
        "pseudo_vie2",
    ];
    let mut consumed = BTreeSet::new();
    for f in &all {
        for p in prefixes {
            if f.starts_with(p) {
                consumed.insert(f.clone());
            }
        }
    }
    assert_eq!(consumed.len(), all.len());
}
