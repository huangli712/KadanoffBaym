//
// Usage:
//   cargo run --example anderson_ness -- julia/tutor/anderson_ness/anderson_ness.toml
//
// Steady state of the single impurity Anderson model (the Rust port of
// julia/tutor/anderson_ness/anderson_ness.jl, from NESSi
// examples/programs/ness2_siam.cpp; no HDF5). The impurity level `epsd`
// is coupled to two leads with smooth box densities of states and the
// chemical potentials +/-V/2; the self-energy is the bath plus the
// second order diagram of G0.
//

#![allow(dead_code)]

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::{bubble1_ness_scalar, bubble2_ness_scalar};
use kadanoffbaym::equilibrium::from_dos_ness;
use kadanoffbaym::fourier::{integral_transform_to_freq, FftIntegralMethod};
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::observables::{convolution_density_matrix_ness, density_matrix_ness};
use kadanoffbaym::dyson::{dyson_ness, DysonBathType};
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::{SteadyGrid, SteadyGreen};
use ness_util::{fmt_e12, ness_outdir, tget_f, tget_i, tget_s, write_ness_gf};
use std::io::Write;

/// Smooth box density of states of the leads,
/// `A(w) = 1/[pi (1 + e^{nu (w - wc)}) (1 + e^{-nu (wc + w)})]`.
fn box_dos(w: f64) -> f64 {
    let (wc, nu) = (10.0, 3.0);
    1.0 / ((1.0 + (nu * (w - wc)).exp()) * (1.0 + (-nu * (wc + w)).exp()))
        / std::f64::consts::PI
}

/// The result of `anderson_ness_main`.
pub struct AndersonResult {
    pub density: f64,
    pub eint: f64,
    pub current: f64,
    pub current_gsl: f64,
    pub g: SteadyGreen,
}

/// Read the parameters, solve the steady state, and write the results
/// (blueprint `main`).
pub fn anderson_ness_main(cfgfile: &str) -> Result<AndersonResult, String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let (u, beta) = (tget_f(&p, "model", "U")?, tget_f(&p, "model", "beta")?);
    let (v, epsd) = (tget_f(&p, "model", "V")?, tget_f(&p, "model", "epsd")?);
    let eta = tget_f(&p, "model", "eta")?;
    let (nft, h) = (tget_i(&p, "grid", "Nft")?, tget_f(&p, "grid", "h")?);
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = tget_s(&p, "output", "prefix")?;
    if !(nft % 2 == 0 && nft >= 4) {
        return Err(format!("{cfgfile}: [grid] Nft must be even and >= 4"));
    }
    if !(h > 0.0 && beta > 0.0 && eta >= 0.0) {
        return Err(format!("{cfgfile}: invalid parameters"));
    }

    let g = SteadyGrid::new(nft, h);
    let mut g0 = SteadyGreen::new(g.clone(), 1);
    let mut gg = SteadyGreen::new(g.clone(), 1);
    let mut sl = SteadyGreen::new(g.clone(), 1);
    let mut sr = SteadyGreen::new(g.clone(), 1);
    let mut su = SteadyGreen::new(g.clone(), 1);
    let mut w = SteadyGreen::new(g.clone(), 1);

    // the bath self-energies of the two leads
    from_dos_ness(
        &mut sl,
        Sign::Fermi,
        &box_dos,
        beta,
        0.5 * v,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    from_dos_ness(
        &mut sr,
        Sign::Fermi,
        &box_dos,
        beta,
        -0.5 * v,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let mut sbath = sl.clone();
    sbath.incr(&sr, Complex::ONE, kadanoffbaym::types::FftDomain::Time);

    // the noninteracting Dyson equation
    let mut eps = DenseMat::zeros(1);
    eps.set(0, 0, Complex::new(epsd, 0.0));
    let mu = 0.0;
    dyson_ness(
        &mut g0,
        mu,
        &eps,
        &sbath,
        FftIntegralMethod::Trapez,
        DysonBathType::Gauss,
        eta,
        beta,
        mu,
    );

    // the second order diagram
    bubble1_ness_scalar(&mut w, &g0, &g0);
    w.smul(Complex::new(u * u, 0.0),
        kadanoffbaym::types::FftDomain::Time,
    );
    bubble2_ness_scalar(&mut su, &g0, &w);
    su.smul(Complex::new(-1.0, 0.0),
        kadanoffbaym::types::FftDomain::Time,
    );

    // the interacting Dyson equation
    let mut s = sbath.clone();
    s.incr(&su, Complex::ONE, kadanoffbaym::types::FftDomain::Time);
    dyson_ness(
        &mut gg,
        mu,
        &eps,
        &s,
        FftIntegralMethod::Trapez,
        DysonBathType::Gauss,
        eta,
        beta,
        mu,
    );

    // the observables
    let slg = convolution_density_matrix_ness(&sl, &gg, Sign::Fermi);
    let gsl = convolution_density_matrix_ness(&gg, &sl, Sign::Fermi);
    let sug = convolution_density_matrix_ness(&su, &gg, Sign::Fermi);
    let rho = density_matrix_ness(&gg, Sign::Fermi);
    let current = 4.0 * slg[0].im;
    let current1 = -4.0 * gsl[0].im;
    let eint = sug[0].re;
    let dens = 2.0 * rho[0].re;

    let mut f = std::fs::File::create(format!("{dir}/{prefix}_observables.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# density  Eint  current  current_GSL - current_SLG").map_err(|e| e.to_string())?;
    writeln!(
        f,
        "{} {} {} {}",
        fmt_e12(dens),
        fmt_e12(eint),
        fmt_e12(current),
        fmt_e12(current1 - current)
    )
    .map_err(|e| e.to_string())?;

    // the spectra
    integral_transform_to_freq(&mut gg, FftIntegralMethod::Trapez);
    write_ness_gf(&format!("{dir}/{prefix}_G_time.dat"), &gg, kadanoffbaym::types::FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_G_freq.dat"), &gg, kadanoffbaym::types::FftDomain::Freq)?;

    println!(
        "density = {dens:.8}, Eint = {eint:.8}, current = {current:.8}, hermiticity of current: {:.3e}",
        current1 - current
    );
    Ok(AndersonResult {
        density: dens,
        eint,
        current,
        current_gsl: current1,
        g: gg,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: anderson_ness <config.toml>");
        std::process::exit(1);
    }
    if let Err(e) = anderson_ness_main(&args[0]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
