//
// Usage:
//   cargo run --example hubbard_bethe_ness2 -- julia/tutor/hubbard_bethe/hubbard_bethe_ness2.toml
//
// Steady-state DMFT of the Hubbard model on the Bethe lattice with the
// bandwidth 4 (the Rust port of
// julia/tutor/hubbard_bethe/hubbard_bethe_ness2.jl, from NESSi
// examples/programs/ness2_bethe.cpp; no HDF5). The local Green's
// function solves `G = [i d/dt + mu - Sigma - Delta]^-1` with
// `Delta = G` and the second-order self-energy of the field G (PT) or
// the Weiss field (IPT), with linear mixing.
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1_ness_scalar, bubble2_ness_scalar};
use kadanoffbaym::equilibrium::from_dos_ness;
use kadanoffbaym::fourier::transforms::FftIntegralMethod;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::observables::{convolution_density_matrix_ness, density_matrix_ness};
use kadanoffbaym::dyson::ness::{dyson_ness, DysonBathType};
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::steady::{FftDomain, SteadyGrid, SteadyGreen};
use ness_util::{ness_iter_row, ness_outdir, tget_f, tget_i, tget_s, write_ness_gf};
use std::io::Write;

/// The result of `ness2_main`.
#[allow(dead_code)]
pub struct Ness2Result {
    pub density: f64,
    pub eint: f64,
    pub iterations: i64,
    pub errors: Vec<f64>,
    pub g: kadanoffbaym::types::steady::SteadyGreen,
    pub s: kadanoffbaym::types::steady::SteadyGreen,
    pub gweiss: kadanoffbaym::types::steady::SteadyGreen,
}

/// Read the parameters from `cfgfile`, run the DMFT loop, and write the
/// results (blueprint `main`).
pub fn ness2_main(cfgfile: &str) -> Result<Ness2Result, String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let (nft, h) = (tget_i(&p, "grid", "Nft")?, tget_f(&p, "grid", "h")?);
    let (beta, u, mu) = (
        tget_f(&p, "model", "beta")?,
        tget_f(&p, "model", "U")?,
        tget_f(&p, "model", "mu")?,
    );
    let (niter, errmax, mix) = (
        tget_i(&p, "solver", "Niter")?,
        tget_f(&p, "solver", "errmax")?,
        tget_f(&p, "solver", "mixing")?,
    );
    let (ipt, out_every) = (
        tget_i(&p, "solver", "ipt_flag")?,
        tget_i(&p, "solver", "out_every")?,
    );
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = tget_s(&p, "output", "prefix")?;
    if !(nft % 2 == 0 && nft >= 4) {
        return Err(format!("{cfgfile}: [grid] Nft must be even and >= 4"));
    }
    if !(0.0 < mix && mix <= 1.0) {
        return Err(format!("{cfgfile}: [solver] mixing must be in (0, 1]"));
    }
    if !(h > 0.0 && beta > 0.0 && niter >= 1 && out_every >= 0) {
        return Err(format!("{cfgfile}: invalid solver/grid parameters"));
    }

    let g = SteadyGrid::new(nft, h);
    let mut gg = SteadyGreen::new(g.clone(), 1);
    let mut su = SteadyGreen::new(g.clone(), 1);
    let mut w = SteadyGreen::new(g.clone(), 1);
    let mut k = SteadyGreen::new(g.clone(), 1);
    let mut gweiss = SteadyGreen::new(g.clone(), 1);
    let ham = DenseMat::zeros(1);

    // initial guess, the noninteracting Green's function
    let dos = |eps: f64| {
        if eps.abs() < 2.0 {
            (2.0 * (1.0 - (eps / 2.0) * (eps / 2.0))).sqrt() / (2.0 * std::f64::consts::PI)
        } else {
            0.0
        }
    };
    from_dos_ness(
        &mut gg,
        Sign::Fermi,
        &dos,
        beta,
        mu,
        FftIntegralMethod::Trapez,
        None,
        None,
        100,
        20,
    );

    // the second-order self-energy of the field G
    let sigma = |ge: &SteadyGreen, w: &mut SteadyGreen, su: &mut SteadyGreen| {
        bubble1_ness_scalar(w, ge, ge);
        w.smul(Complex::new(u * u, 0.0), FftDomain::Time);
        bubble2_ness_scalar(su, ge, w);
        su.smul(Complex::new(-1.0, 0.0), FftDomain::Time);
    };

    let mut errs: Vec<f64> = Vec::new();
    let mut itermax = niter;
    let mut fit = std::fs::File::create(format!("{dir}/{prefix}_iterations.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(fit, "# iteration  error  density").map_err(|e| e.to_string())?;
    for iter in 1..=niter {
        let gold = gg.clone();
        // the self-energy
        if ipt != 0 {
            dyson_ness(
                &mut gweiss,
                mu,
                &ham,
                &gg,
                FftIntegralMethod::Trapez,
                DysonBathType::None,
                0.0,
                1.0,
                0.0,
            );
        } else {
            gweiss = gg.clone();
        }
        sigma(&gweiss, &mut w, &mut su);
        // the kernel K = Sigma + G and the Dyson equation
        k.memset(FftDomain::Time, Complex::ZERO);
        k.incr(&su, Complex::ONE, FftDomain::Time);
        k.incr(&gg, Complex::ONE, FftDomain::Time);
        dyson_ness(
            &mut gg,
            mu,
            &ham,
            &k,
            FftIntegralMethod::Trapez,
            DysonBathType::None,
            0.0,
            1.0,
            0.0,
        );
        let err = gg.distance(&gold, FftDomain::Time);
        // linear mixing
        gg.smul(Complex::new(mix, 0.0), FftDomain::Time);
        gg.incr(&gold, Complex::new(1.0 - mix, 0.0), FftDomain::Time);

        errs.push(err);
        let dens = 2.0 * density_matrix_ness(&gg, Sign::Fermi)[0].re;
        write!(fit, "{}", ness_iter_row(iter, err, dens)).map_err(|e| e.to_string())?;
        println!("iteration {iter}: error = {err:.6e}");
        if out_every > 0 && iter % out_every == 0 {
            write_ness_gf(&format!("{dir}/{prefix}_G_it{iter}.dat"), &gg, FftDomain::Time)?;
            write_ness_gf(&format!("{dir}/{prefix}_S_it{iter}.dat"), &su, FftDomain::Time)?;
        }
        if err < errmax && iter > 3 {
            itermax = iter;
            println!("converged after {itermax} iterations");
            break;
        }
    }
    drop(fit);

    // observables
    let sug = convolution_density_matrix_ness(&su, &gg, Sign::Fermi);
    let eint = sug[0].re;
    let dens = 2.0 * density_matrix_ness(&gg, Sign::Fermi)[0].re;
    let mut f = std::fs::File::create(format!("{dir}/{prefix}_observables.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# density  Eint  iterations").map_err(|e| e.to_string())?;
    write!(f, "{} {} {}\n", ness_util::fmt_e12(dens), ness_util::fmt_e12(eint), itermax)
        .map_err(|e| e.to_string())?;

    // spectra
    kadanoffbaym::fourier::transforms::integral_transform_to_freq(
        &mut gg,
        FftIntegralMethod::Trapez,
    );
    kadanoffbaym::fourier::transforms::integral_transform_to_freq(
        &mut su,
        FftIntegralMethod::Trapez,
    );
    write_ness_gf(&format!("{dir}/{prefix}_G_time.dat"), &gg, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_G_freq.dat"), &gg, FftDomain::Freq)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_time.dat"), &su, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_freq.dat"), &su, FftDomain::Freq)?;

    println!("density = {dens:.8}, Eint = {eint:.8}, iterations = {itermax}");
    Ok(Ness2Result {
        density: dens,
        eint,
        iterations: itermax,
        errors: errs,
        g: gg,
        s: su,
        gweiss,
    })
}

#[allow(dead_code)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: hubbard_bethe_ness2 <config.toml>");
        std::process::exit(1);
    }
    if let Err(e) = ness2_main(&args[0]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
