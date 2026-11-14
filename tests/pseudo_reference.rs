mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::convolution_density_matrix;
use kadanoffbaym::convolution::{incr_less_adv_buf, incr_lmix_rmix_buf};
use kadanoffbaym::convolution::pseudo_convolution;
use kadanoffbaym::convolution::{ConvWorkspace, FOp};
use common::kbref_raw::{read_kbref, KbrefData, KbrefRow};
use kadanoffbaym::dyson::pseudo_dyson;
use kadanoffbaym::vie2::pseudo::pseudo_vie2;
use kadanoffbaym::types::Contour;
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::TimeFunction;
use kadanoffbaym::types::MATSUBARA;
use kadanoffbaym::basics::pseudo_timestep;
use kadanoffbaym::observables::{density_matrix_pseudo, density_pseudo};
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

const NT: i64 = 10;
const NTAU: i64 = 30;
const BETA: f64 = 2.0;
const HH: f64 = 0.02;
const LAM0: f64 = 0.5;
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/test/phase6");

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// The closed-form input (blueprint `p6_val`, see pseudo_reference.cpp).
fn p6_val(w: i64, cc: i64, x: f64, y: f64, a: i64, b: i64) -> Complex {
    let amp = 0.3 + 0.1 * w as f64 + 0.05 * cc as f64 + 0.1 * a as f64 - 0.07 * b as f64;
    let env = (-0.2 * x - 0.15 * y).exp();
    c(
        amp * env * (0.7 * x + 0.3 * y + 0.2 * (a + b) as f64 + w as f64).cos(),
        amp * env * (0.5 * x - 0.4 * y + 0.1 * (a * b) as f64 + cc as f64).sin(),
    )
}

fn p6_blk(w: i64, cc: i64, x: f64, y: f64, d: i64, s: f64) -> Vec<Complex> {
    (0..d * d)
        .map(|q| {
            let (a, b) = (q / d, q % d);
            p6_val(w, cc, x, y, a, b) * c(s, 0.0)
        })
        .collect()
}

fn p6_cn(d: i64) -> Contour {
    Contour::new(NT, NTAU, d, d, NT as f64 * HH, BETA)
}

/// Fill all the stored blocks (blueprint `p6_build!`).
fn p6_build(g: &mut PseudoGreen, w: i64, s: f64) {
    let d = g.size1 as i64;
    let dtau = BETA / NTAU as f64;
    for q in 0..=NTAU {
        let v = p6_blk(w, 0, q as f64 * dtau, 0.0, d, s);
        g.mat.block_mut(q as usize).copy_from_slice(&v);
    }
    for i in 0..=NT {
        for j in 0..=i {
            let v = p6_blk(w, 1, i as f64 * HH, j as f64 * HH, d, s);
            g.ret.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
        for q in 0..=NTAU {
            let v = p6_blk(w, 2, i as f64 * HH, q as f64 * dtau, d, s);
            g.lmix.raw_block_at_mut(i, q).copy_from_slice(&v);
        }
        for j in i..=NT {
            let v = p6_blk(w, 3, i as f64 * HH, j as f64 * HH, d, s);
            g.less.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
    }
}

fn p6_ham(d: usize, t: f64) -> Vec<Complex> {
    let mut v = vec![
        c(0.3, 0.0),
        c(0.2, -0.1),
        c(0.2, 0.1),
        c(-0.4, 0.0),
    ];
    for i in 0..2 {
        v[i * 2 + i] += c(0.3 * (2.0 * t).sin(), 0.0);
    }
    v.truncate(d * d);
    v
}

fn p6_middle(d: usize, t: f64) -> Vec<Complex> {
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

fn p6_hcf(d: usize) -> TimeFunction {
    let mut f = TimeFunction::new(NT as usize, d, d, Complex::ZERO);
    f.block_mut(MATSUBARA).copy_from_slice(&p6_ham(d, 0.0));
    for n in 0..=NT {
        f.block_mut(n).copy_from_slice(&p6_ham(d, n as f64 * HH));
    }
    f
}

fn p6_fcf(d: usize) -> TimeFunction {
    let mut f = TimeFunction::new(NT as usize, d, d, Complex::ZERO);
    f.block_mut(MATSUBARA).copy_from_slice(&p6_middle(d, -0.5));
    for n in 0..=NT {
        f.block_mut(n).copy_from_slice(&p6_middle(d, n as f64 * HH));
    }
    f
}

/// The pseudo storage entry (blueprint `kb_value`), `gc` is the
/// transferred standard green (for the derived `gtr` access).
fn kb_value_pseudo(g: &PseudoGreen, gc: &ContourGreen, r: &KbrefRow) -> Option<Complex> {
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

/// Compare all the storage rows, returns the maximal error.
fn p6_storage(g: &PseudoGreen, refdata: &KbrefData) -> f64 {
    let gc = g.to_contour();
    let mut err = 0.0f64;
    let mut nfail = 0;
    for r in refdata.rows.iter() {
        if !matches!(r.component.as_str(), "mat" | "gtr" | "tv" | "les") {
            continue;
        }
        let v = kb_value_pseudo(g, &gc, r).unwrap();
        err = err.max((v - r.value).abs());
        if (v - r.value).abs() > 1e-10 {
            nfail += 1;
        }
    }
    assert_eq!(nfail, 0, "storage mismatches in {}", refdata.path);
    err
}

/// The lesser start-up source blocks (blueprint `p6_start_source`).
fn p6_start_source(
    g: &PseudoGreen,
    a_p: &PseudoGreen,
    acc_p: &PseudoGreen,
    integ: &Integrator,
    alpha: Complex,
    refdata: &KbrefData,
    nonzero: bool,
) -> f64 {
    let k = integ.k as i64;
    let d = g.size1;
    let rows: Vec<&KbrefRow> = refdata
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
        // the blueprint tstp = n + 1 (KB) is my time step n
        let tstp = n;
        let mut buf: Vec<Vec<Complex>> = (0..=k as usize).map(|_| vec![Complex::ZERO; d * d]).collect();
        incr_lmix_rmix_buf(
            &mut buf, tstp, &a_c, None, &g_c, integ, g.sign, BETA, HH, alpha, &mut ws,
        );
        incr_less_adv_buf(
            &mut buf, tstp, &a_c, &acc_c, &FOp::None, &g_c, &g_c, integ, BETA, HH, alpha,
            &mut ws, false,
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

fn data_files(prefix: &str) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(DATA)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|f| f.starts_with(prefix) && f.ends_with(".dat"))
        .collect();
    out.sort();
    out
}

fn p6_meta(refdata: &KbrefData) -> (usize, i64, Sign) {
    let d: usize = refdata.metadata["size1"].parse().unwrap();
    let k: i64 = refdata.metadata["k"].parse().unwrap();
    let sig: i64 = refdata.metadata["sig"].parse().unwrap();
    (d, k, Sign::from_i64(sig))
}

#[test]
fn pseudo_reference_storage() {
    let refdata = read_kbref(&format!("{DATA}/pseudo_storage.dat")).unwrap();
    let mut g = PseudoGreen::new(&p6_cn(2), Sign::Fermi, Complex::ZERO);
    p6_build(&mut g, 0, 1.0);
    let err = p6_storage(&g, &refdata);
    // derived accessors against their definitions (the blueprint
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
        let refdata = read_kbref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = p6_meta(&refdata);
        let cc = p6_cn(d as i64);
        let mk = |w: i64| {
            let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
            p6_build(&mut g, w, 1.0);
            g
        };
        let (a, acc, b, bcc) = (mk(1), mk(2), mk(3), mk(4));
        let f = if refdata.metadata["f"] == "yes" {
            Some(p6_fcf(d))
        } else {
            None
        };
        let integ = Integrator::new(k);
        let mut x = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_convolution(
            &mut x, &a, &acc, f.as_ref(), &b, &bcc, &integ, BETA, HH,
        );
        let mut err = p6_storage(&x, &refdata);
        // the density-matrix rows
        let (ac, accc, bc, bccc) = (a.to_contour(), acc.to_contour(), b.to_contour(), bcc.to_contour());
        let mut nfail = 0;
        let mut count = 0;
        for r in refdata.rows.iter().filter(|r| r.component == "density") {
            count += 1;
            let tstp = if r.i == -1 { MATSUBARA } else { r.i };
            let rho = convolution_density_matrix(
                tstp, &ac, &accc, f.as_ref(), &bc, &bccc, &integ, BETA, HH,
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
        let refdata = read_kbref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = p6_meta(&refdata);
        let cc = p6_cn(d as i64);
        let mut sigma = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let formula = refdata.metadata["sigma"] == "formula";
        if formula {
            p6_build(&mut sigma, 5, 0.2);
        }
        let integ = Integrator::new(k);
        let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_dyson(&mut g, LAM0, &p6_hcf(d), &sigma, &integ, BETA, HH);
        let mut err = p6_storage(&g, &refdata);
        // the density-matrix rows (r.i == -1 is the Matsubara axis)
        let drows: Vec<&KbrefRow> = refdata
            .rows
            .iter()
            .filter(|r| r.component == "density")
            .collect();
        assert_eq!(drows.len(), d * d * (NT as usize + 2));
        let mut nfail = 0;
        for r in &drows {
            let tstp = if r.i == -1 { MATSUBARA } else { r.i };
            let rho = density_matrix_pseudo(&g, tstp);
            let v = rho[r.a as usize * d + r.b as usize];
            err = err.max((v - r.value).abs());
            if (v - r.value).abs() > 1e-10 {
                nfail += 1;
            }
            // the time slice and the trace agree
            let x = pseudo_timestep(&g, tstp);
            let rho_step = kadanoffbaym::observables::density_matrix_pseudo_step(&x, tstp);
            for q in 0..(d * d) {
                assert_eq!(rho_step[q], rho[q]);
            }
            let trace = density_pseudo(&g, tstp);
            let want: Complex = (0..d).map(|i| rho[i * d + i]).sum();
            assert_eq!(trace, want);
        }
        assert_eq!(nfail, 0, "density mismatches in {name}");
        err = err.max(p6_start_source(
            &g, &sigma, &sigma, &integ, Complex::ONE, &refdata, formula,
        ));
        assert!(err < 1e-10, "{name}: err {err}");
    }
}

#[test]
fn pseudo_reference_vie2() {
    let files = data_files("pseudo_vie2");
    assert_eq!(files.len(), 5);
    for name in &files {
        let refdata = read_kbref(&format!("{DATA}/{name}")).unwrap();
        let (d, k, sig) = p6_meta(&refdata);
        let cc = p6_cn(d as i64);
        let mut f = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let mut fcc = PseudoGreen::new(&cc, sig, Complex::ZERO);
        let formula = refdata.metadata["kernel"] == "formula";
        if formula {
            p6_build(&mut f, 6, 0.2);
            p6_build(&mut fcc, 7, 0.2);
        }
        let mut q = PseudoGreen::new(&cc, sig, Complex::ZERO);
        p6_build(&mut q, 8, 1.0);
        let integ = Integrator::new(k);
        let mut g = PseudoGreen::new(&cc, sig, Complex::ZERO);
        pseudo_vie2(&mut g, &f, &fcc, &q, &integ, BETA, HH);
        let mut err = p6_storage(&g, &refdata);
        if formula {
            err = err.max(p6_start_source(
                &g, &f, &fcc, &integ, -Complex::ONE, &refdata, true,
            ));
        }
        assert!(err < 1e-10, "{name}: err {err}");
    }
}

#[test]
fn pseudo_reference_all_files_consumed() {
    let all = data_files("");
    assert_eq!(all.len(), 18);
    let prefixes = ["pseudo_storage", "pseudo_convolution", "pseudodyson", "pseudo_vie2"];
    let mut consumed = std::collections::BTreeSet::new();
    for f in &all {
        for p in prefixes {
            if f.starts_with(p) {
                consumed.insert(f.clone());
            }
        }
    }
    assert_eq!(consumed.len(), all.len());
}
