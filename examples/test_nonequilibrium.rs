//
// Usage:
//   cargo run --example test_nonequilibrium -- julia/tutor/test_nonequilibrium/test_nonequilibrium.toml out.dat
//
// A single level `eps1` coupled to a second level `eps2` (hopping
// `i lambda`) solved by embedding with the Dyson equation and the VIE2
// (the Rust port of julia/tutor/test_nonequilibrium/test_nonequilibrium.jl,
// from NESSi examples/programs/test_nonequilibrium.cpp), both compared
// with the (1,1) element of the exact 2 x 2 solution.
//

#![allow(dead_code)]

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::mat::conv_mat;
use kadanoffbaym::convolution::timestep::convolution_time_step;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::dyson::mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::dyson::start::dyson_start;
use kadanoffbaym::dyson::timestep::dyson_timestep;
use kadanoffbaym::vie2::mat::vie2_mat;
use kadanoffbaym::vie2::start::vie2_start;
use kadanoffbaym::vie2::timestep::vie2_timestep;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::function::TimeFunction;
use kadanoffbaym::basics::MATSUBARA;
use kadanoffbaym::weights::Integrator;
use ness_util::tget_f;
use std::io::Write;

fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// Read the parameters and write the errors of the Dyson and VIE2
/// solutions into `outfile` (blueprint `main`).
pub fn test_nonequilibrium_main(cfgfile: &str, outfile: &str) -> Result<(f64, f64), String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let root = |k: &str| -> Result<f64, String> {
        p.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("{cfgfile}: missing {k}"))
    };
    let nt_i = root("Nt")? as i64;
    let ntau_i = root("Ntau")? as i64;
    let tmax = root("Tmax")?;
    let k = root("SolveOrder")? as i64;
    let (beta, mu) = (root("beta")?, root("mu")?);
    let (eps1, eps2) = (root("eps1")?, root("eps2")?);
    let lambda = root("lambda")?;
    let _ = tget_f(&p, "", "beta");
    if !(ntau_i % 2 == 0 && ntau_i >= 4) {
        return Err(format!("{cfgfile}: Ntau must be even and >= 4"));
    }
    if !((1..=5).contains(&k) && nt_i >= k) {
        return Err(format!("{cfgfile}: invalid solver parameters"));
    }

    // Nt and Ntau count the intervals
    let h = tmax / nt_i as f64;
    let (nt, ntau) = (nt_i, ntau_i);
    let c1 = Contour::new(nt, ntau, 1, 1, tmax, beta);
    let c2 = Contour::new(nt, ntau, 2, 2, tmax, beta);

    // the exact Green's function of the 2 x 2 Hamiltonian, (1,1) part
    let mut g2 = ContourGreen::new(&c2, Sign::Fermi, Complex::ZERO);
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, Complex::new(eps1, 0.0));
    h2.set(0, 1, Complex::new(0.0, lambda));
    h2.set(1, 0, Complex::new(0.0, -lambda));
    h2.set(1, 1, Complex::new(eps2, 0.0));
    from_ham(&mut g2, mu, &h2, beta, h);
    let mut gx = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    for m in 0..=ntau as usize {
        gx.mat.block_mut(m)[0] = g2.mat.block(m)[0];
    }
    for i in 0..=nt {
        for m in 0..=ntau as i64 {
            gx.lmix.raw_block_at_mut(i, m)[0] = g2.lmix.raw_block_at(i, m)[0];
        }
        for j in 0..=i {
            gx.ret.raw_block_at_mut(i, j)[0] = g2.ret.raw_block_at(i, j)[0];
            gx.less.raw_block_at_mut(j, i)[0] = g2.less.raw_block_at(j, i)[0];
        }
    }

    // the embedding self-energy and Hamiltonian
    let mut sigma = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    let mut he = DenseMat::zeros(1);
    he.set(0, 0, Complex::new(eps2, 0.0));
    from_ham(&mut sigma, mu, &he, beta, h);
    for t in MATSUBARA..=nt {
        sigma.smul_at(t, Complex::new(lambda * lambda, 0.0));
    }
    let mut hh = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    for i in MATSUBARA..=nt {
        hh.block_mut(i)[0] = Complex::new(eps1, 0.0);
    }

    let error_of = |g: &ContourGreen| -> f64 {
        let mut e = 0.0;
        for t in 0..=nt {
            e += 2.0 * gx.distance_less(g, t) / (nt_i * nt_i) as f64;
            e += 2.0 * gx.distance_ret(g, t) / (nt_i * nt_i) as f64;
            e += gx.distance_lmix(g, t) / (nt_i * ntau_i) as f64;
        }
        e
    };

    // the Dyson equation
    let mut g = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &hh, beta, k, MatsubaraMethod::Fixpoint, true, None, 1e-12);
    dyson_start(&mut g, mu, &hh, &sigma, beta, h, k);
    for tstp in (k + 2)..=(nt + 1) {
        dyson_timestep(mt(tstp), &mut g, mu, &hh, &sigma, beta, h, k);
    }
    let err_dyson = error_of(&g);

    // the VIE2 [1 + F] G = G0 with F = -(G0 * Sigma), Fcc = -(Sigma * G0)
    let integ = Integrator::new(k);
    let mut g0 = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    let mut h1 = DenseMat::zeros(1);
    h1.set(0, 0, Complex::new(eps1, 0.0));
    from_ham(&mut g0, mu, &h1, beta, h);
    let mut f = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    let mut fcc = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    let kernel = |tstp: i64, f: &mut ContourGreen, fcc: &mut ContourGreen| {
        if tstp == 0 {
            conv_mat(&mut f.mat, &g0.mat, &sigma.mat, &integ, beta, Sign::Fermi);
            conv_mat(&mut fcc.mat, &sigma.mat, &g0.mat, &integ, beta, Sign::Fermi);
        } else {
            convolution_time_step(mt(tstp), f, &g0, &g0, None, &sigma, &sigma, &integ, beta, h);
            convolution_time_step(mt(tstp), fcc, &sigma, &sigma, None, &g0, &g0, &integ, beta, h);
        }
        f.smul_at(mt(tstp), Complex::new(-1.0, 0.0));
        fcc.smul_at(mt(tstp), Complex::new(-1.0, 0.0));
    };
    let mut g = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    kernel(0, &mut f, &mut fcc);
    vie2_mat(&mut g, &f, &fcc, &g0, beta, &integ, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
    for tstp in 1..=k + 1 {
        kernel(tstp, &mut f, &mut fcc);
    }
    vie2_start(&mut g, &f, &fcc, &g0, k, beta, h);
    for tstp in (k + 2)..=(nt + 1) {
        kernel(tstp, &mut f, &mut fcc);
        vie2_timestep(mt(tstp), &mut g, &f, &fcc, &g0, &integ, beta, h, MatsubaraMethod::Fixpoint);
    }
    let err_vie2 = error_of(&g);

    let mut out = std::fs::File::create(outfile).map_err(|e| e.to_string())?;
    writeln!(out, "# err_dyson err_vie2").map_err(|e| e.to_string())?;
    writeln!(out, "{err_dyson:.5e} {err_vie2:.5e}").map_err(|e| e.to_string())?;
    println!(
        "Nt = {nt_i}, Ntau = {ntau_i}, SolveOrder = {k}: err_dyson = {err_dyson:.5e}, err_vie2 = {err_vie2:.5e}"
    );
    Ok((err_dyson, err_vie2))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: test_nonequilibrium <config.toml> <output.dat>");
        std::process::exit(1);
    }
    if let Err(e) = test_nonequilibrium_main(&args[0], &args[1]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
