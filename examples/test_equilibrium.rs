//
// Usage:
//   cargo run --example test_equilibrium -- julia/tutor/test_equilibrium/test_equilibrium.toml out.dat
//
// The Matsubara Dyson equation of a single level `eps1` coupled to a
// second level `eps2` (hopping `i lambda`) solved by embedding (the Rust
// port of julia/tutor/test_equilibrium/test_equilibrium.jl, from NESSi
// examples/programs/test_equilibrium.cpp): the 1 x 1 Green's function
// with Sigma = lambda^2 g2 compared with the (1,1) element of the exact
// 2 x 2 solution.
//

#![allow(dead_code)]

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use ness_util::{tget_f, tget_i};
use std::io::Write;

/// Read the parameters and write the errors of the FIXPOINT and CG
/// methods into `outfile` (blueprint `main`).
pub fn test_equilibrium_main(cfgfile: &str, outfile: &str) -> Result<(f64, f64), String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let root = |k: &str| -> Result<f64, String> {
        p.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("{cfgfile}: missing {k}"))
    };
    let ntau_i = root("Ntau")? as i64;
    let k = root("SolveOrder")? as i64;
    let (beta, mu) = (root("beta")?, root("mu")?);
    let (eps1, eps2) = (root("eps1")?, root("eps2")?);
    let lambda = root("lambda")?;
    let _ = tget_f(&p, "", "beta");
    let _ = tget_i(&p, "", "Ntau");
    if !(ntau_i % 2 == 0 && ntau_i >= 4) {
        return Err(format!("{cfgfile}: Ntau must be even and >= 4"));
    }
    if !(1..=5).contains(&k) {
        return Err(format!("{cfgfile}: SolveOrder must be in 1..=5"));
    }

    // Ntau counts the intervals; a minimal real time grid suffices
    let h = 1.0;
    let ntau = ntau_i;
    let c1 = Contour::new(1, ntau, 1, 1, h, beta);
    let c2 = Contour::new(1, ntau, 2, 2, h, beta);

    // the exact Green's function of the 2 x 2 Hamiltonian
    let mut g2 = ContourGreen::new(&c2, Sign::Fermi, C64::ZERO);
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, C64::new(eps1, 0.0));
    h2.set(0, 1, C64::new(0.0, lambda));
    h2.set(1, 0, C64::new(0.0, -lambda));
    h2.set(1, 1, C64::new(eps2, 0.0));
    from_ham(&mut g2, mu, &h2, beta, h);
    let mut gx = ContourGreen::new(&c1, Sign::Fermi, C64::ZERO);
    for m in 0..=ntau as usize {
        gx.mat.block_mut(m)[0] = g2.mat.block(m)[0];
    }

    // the embedding self-energy and Hamiltonian
    let mut sigma = ContourGreen::new(&c1, Sign::Fermi, C64::ZERO);
    let mut he = DenseMat::zeros(1);
    he.set(0, 0, C64::new(eps2, 0.0));
    from_ham(&mut sigma, mu, &he, beta, h);
    sigma.smul_at(MATSUBARA, C64::new(lambda * lambda, 0.0));
    let mut hh = TimeFunction::new(1, 1, 1, C64::ZERO);
    for i in MATSUBARA..=1 {
        hh.block_mut(i)[0] = C64::new(eps1, 0.0);
    }

    let mut errs = Vec::new();
    for method in [MatsubaraMethod::Fixpoint, MatsubaraMethod::Cg] {
        let mut g = ContourGreen::new(&c1, Sign::Fermi, C64::ZERO);
        dyson_mat(&mut g, &sigma, mu, &hh, beta, k, method, true, None, 1e-12);
        errs.push(g.distance_at(&gx, MATSUBARA) / ntau_i as f64);
    }

    let mut f = std::fs::File::create(outfile).map_err(|e| e.to_string())?;
    writeln!(f, "# err_fixpoint err_cg").map_err(|e| e.to_string())?;
    writeln!(f, "{:.5e} {:.5e}", errs[0], errs[1]).map_err(|e| e.to_string())?;
    println!(
        "Ntau = {ntau_i}, SolveOrder = {k}: err_fixpoint = {:.5e}, err_cg = {:.5e}",
        errs[0], errs[1]
    );
    Ok((errs[0], errs[1]))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: test_equilibrium <config.toml> <output.dat>");
        std::process::exit(1);
    }
    if let Err(e) = test_equilibrium_main(&args[0], &args[1]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
