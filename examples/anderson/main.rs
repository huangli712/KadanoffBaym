// Project : KadanoffBaym
// File    : examples/anderson/main.rs
// Purpose : steady state of the single-impurity Anderson model with two box-DOS leads and a second-order self-energy.
//
// Usage:
//   cargo run --example anderson -- examples/anderson/anderson.toml
//

#[path = "../../util/mod.rs"]
mod util;

use std::fs::File;
use std::io::Write;

use std::f64::consts::PI;

use std::env::args;
use std::process::exit;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::DysonBathType;
use kadanoffbaym::fourier::FftIntegralMethod;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::{FftDomain, Sign, SteadyGreen, SteadyGrid};

use kadanoffbaym::convolution::{bubble1_steady_scalar, bubble2_steady_scalar};
use kadanoffbaym::dyson::dyson_steady;
use kadanoffbaym::equilibrium::from_dos_steady;
use kadanoffbaym::fourier::integral_transform_to_freq;
use kadanoffbaym::observables::{steady_convolution_density_matrix, steady_density};

use util::{fmt_e12, get_f64, get_i64, get_str, load_toml_config, ness_outdir, write_ness_gf};

/// Smooth box density of states of the leads,
/// `A(w) = 1/[pi (1 + e^{nu (w - wc)}) (1 + e^{-nu (wc + w)})]`.
fn box_density_of_states(w: f64) -> f64 {
    let (wc, nu) = (10.0, 3.0);
    1.0 / ((1.0 + (nu * (w - wc)).exp()) * (1.0 + (-nu * (wc + w)).exp())) / PI
}

/// Read the parameters, solve the steady state, and write the results.
pub(crate) fn run_anderson(cfgfile: &str) -> Result<(), String> {
    let p = load_toml_config(cfgfile)?;
    let (u, beta) = (get_f64(&p, "model", "U")?, get_f64(&p, "model", "beta")?);
    let (v, epsd) = (get_f64(&p, "model", "V")?, get_f64(&p, "model", "epsd")?);
    let eta = get_f64(&p, "model", "eta")?;
    let (nft, h) = (get_i64(&p, "grid", "Nft")?, get_f64(&p, "grid", "h")?);
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = get_str(&p, "output", "prefix")?;
    if !(nft % 2 == 0 && nft >= 4) {
        return Err(format!("{cfgfile}: [grid] Nft must be even and >= 4"));
    }
    if !(h > 0.0 && beta > 0.0 && eta >= 0.0) {
        return Err(format!("{cfgfile}: invalid parameters"));
    }

    let g = SteadyGrid::new(nft, h);
    let mut g0 = SteadyGreen::new(g, 1);
    let mut gg = SteadyGreen::new(g, 1);
    let mut sl = SteadyGreen::new(g, 1);
    let mut sr = SteadyGreen::new(g, 1);
    let mut su = SteadyGreen::new(g, 1);
    let mut w = SteadyGreen::new(g, 1);

    // the bath self-energies of the two leads
    from_dos_steady(
        &mut sl,
        Sign::Fermi,
        &box_density_of_states,
        beta,
        0.5 * v,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    from_dos_steady(
        &mut sr,
        Sign::Fermi,
        &box_density_of_states,
        beta,
        -0.5 * v,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );
    let mut sbath = sl.clone();
    sbath.incr(&sr, Complex::ONE, FftDomain::Time);

    // the noninteracting Dyson equation
    let mut eps = DenseMat::zeros(1);
    eps.set(0, 0, Complex::new(epsd, 0.0));
    let mu = 0.0;
    dyson_steady(
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
    bubble1_steady_scalar(&mut w, &g0, &g0);
    w.smul(Complex::new(u * u, 0.0), FftDomain::Time);
    bubble2_steady_scalar(&mut su, &g0, &w);
    su.smul(Complex::new(-1.0, 0.0), FftDomain::Time);

    // the interacting Dyson equation
    let mut s = sbath.clone();
    s.incr(&su, Complex::ONE, FftDomain::Time);
    dyson_steady(
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
    let slg = steady_convolution_density_matrix(&sl, &gg, Sign::Fermi);
    let gsl = steady_convolution_density_matrix(&gg, &sl, Sign::Fermi);
    let sug = steady_convolution_density_matrix(&su, &gg, Sign::Fermi);
    let dens = 2.0 * steady_density(&gg, Sign::Fermi).re;
    let current = 4.0 * slg[0].im;
    let current1 = -4.0 * gsl[0].im;
    let eint = sug[0].re;

    let mut f =
        File::create(format!("{dir}/{prefix}_observables.dat")).map_err(|e| e.to_string())?;
    writeln!(f, "# density  Eint  current  current_GSL - current_SLG")
        .map_err(|e| e.to_string())?;
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
    write_ness_gf(&format!("{dir}/{prefix}_G_time.dat"), &gg, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_G_freq.dat"), &gg, FftDomain::Freq)?;

    println!(
        "density = {dens:.8}, Eint = {eint:.8}, current = {current:.8}, hermiticity of current: {:.3e}",
        current1 - current
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: anderson <config.toml>");
        exit(1);
    }
    if let Err(e) = run_anderson(&args[0]) {
        eprintln!("{e}");
        exit(1);
    }
}
