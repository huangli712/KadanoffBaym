//
// Usage:
//   cargo run --example hubbard -- bethe_steady examples/hubbard/hubbard_bethe_steady.toml
//
// Steady-state DMFT of the Hubbard model on the Bethe lattice with the
// bandwidth 4, from NESSi examples/programs/ness2_bethe.cpp; no HDF5).
// The local Green's
// function solves `G = [i d/dt + mu - Sigma - Delta]^-1` with
// `Delta = G` and the second-order self-energy of the field G (PT) or
// the Weiss field (IPT), with linear mixing.
//

// Shared example sources are included by several binaries.

use crate::util::{ness_iter_row, ness_outdir, tget_f, tget_i, tget_s, write_ness_gf};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::{bubble1_steady_scalar, bubble2_steady_scalar};
use kadanoffbaym::dyson::{DysonBathType, dyson_steady};
use kadanoffbaym::equilibrium::from_dos_steady;
use kadanoffbaym::fourier::FftIntegralMethod;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::observables::{convolution_density_matrix_steady, density_matrix_steady};
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::{FftDomain, SteadyGreen, SteadyGrid};
use std::io::Write;

/// The result of `steady_main`.
#[allow(dead_code)]
pub struct SteadyResult {
    pub density: f64,
    pub eint: f64,
    pub iterations: i64,
    pub errors: Vec<f64>,
    pub g: kadanoffbaym::types::SteadyGreen,
    pub s: kadanoffbaym::types::SteadyGreen,
    pub gweiss: kadanoffbaym::types::SteadyGreen,
}

/// Read the parameters from `cfgfile`, run the DMFT loop, and write the
/// results (blueprint `main`).
pub fn steady_main(cfgfile: &str) -> Result<SteadyResult, String> {
    let p = crate::util::inp_toml(cfgfile)?;
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
    let mut gg = SteadyGreen::new(g, 1);
    let mut su = SteadyGreen::new(g, 1);
    let mut w = SteadyGreen::new(g, 1);
    let mut k = SteadyGreen::new(g, 1);
    let mut gweiss = SteadyGreen::new(g, 1);
    let ham = DenseMat::zeros(1);

    // initial guess, the noninteracting Green's function
    let dos = |eps: f64| {
        if eps.abs() < 2.0 {
            (2.0 * (1.0 - (eps / 2.0) * (eps / 2.0))).sqrt() / (2.0 * std::f64::consts::PI)
        } else {
            0.0
        }
    };
    from_dos_steady(
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
        bubble1_steady_scalar(w, ge, ge);
        w.smul(Complex::new(u * u, 0.0), FftDomain::Time);
        bubble2_steady_scalar(su, ge, w);
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
            dyson_steady(
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
        dyson_steady(
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
        let dens = 2.0 * density_matrix_steady(&gg, Sign::Fermi)[0].re;
        write!(fit, "{}", ness_iter_row(iter, err, dens)).map_err(|e| e.to_string())?;
        println!("iteration {iter}: error = {err:.6e}");
        if out_every > 0 && iter % out_every == 0 {
            write_ness_gf(
                &format!("{dir}/{prefix}_G_it{iter}.dat"),
                &gg,
                FftDomain::Time,
            )?;
            write_ness_gf(
                &format!("{dir}/{prefix}_S_it{iter}.dat"),
                &su,
                FftDomain::Time,
            )?;
        }
        if err < errmax && iter > 3 {
            itermax = iter;
            println!("converged after {itermax} iterations");
            break;
        }
    }
    drop(fit);

    // observables
    let sug = convolution_density_matrix_steady(&su, &gg, Sign::Fermi);
    let eint = sug[0].re;
    let dens = 2.0 * density_matrix_steady(&gg, Sign::Fermi)[0].re;
    let mut f = std::fs::File::create(format!("{dir}/{prefix}_observables.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# density  Eint  iterations").map_err(|e| e.to_string())?;
    writeln!(
        f,
        "{} {} {}",
        crate::util::fmt_e12(dens),
        crate::util::fmt_e12(eint),
        itermax
    )
    .map_err(|e| e.to_string())?;

    // spectra
    kadanoffbaym::fourier::integral_transform_to_freq(&mut gg, FftIntegralMethod::Trapez);
    kadanoffbaym::fourier::integral_transform_to_freq(&mut su, FftIntegralMethod::Trapez);
    write_ness_gf(&format!("{dir}/{prefix}_G_time.dat"), &gg, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_G_freq.dat"), &gg, FftDomain::Freq)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_time.dat"), &su, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_freq.dat"), &su, FftDomain::Freq)?;

    println!("density = {dens:.8}, Eint = {eint:.8}, iterations = {itermax}");
    Ok(SteadyResult {
        density: dens,
        eint,
        iterations: itermax,
        errors: errs,
        g: gg,
        s: su,
        gweiss,
    })
}
