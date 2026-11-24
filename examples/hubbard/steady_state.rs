// Project : KadanoffBaym
// File    : examples/hubbard/steady_state.rs
// Purpose : steady-state DMFT of the Hubbard model on the Bethe lattice.
//
// Usage:
//   cargo run --example hubbard -- bethe_steady examples/hubbard/bethe_steady.toml
//

use std::fs::File;
use std::io::Write;

use std::f64::consts::PI;

use kadanoffbaym::dyson::DysonBathType;
use kadanoffbaym::fourier::FftIntegralMethod;
use kadanoffbaym::types::{FftDomain, SteadyGreen, SteadyGrid};

use kadanoffbaym::convolution::{bubble1_steady_scalar, bubble2_steady_scalar};
use kadanoffbaym::dyson::dyson_steady;
use kadanoffbaym::equilibrium::from_dos_steady;
use kadanoffbaym::fourier::integral_transform_to_freq;
use kadanoffbaym::observables::{steady_convolution_density_matrix, steady_density};

use super::util::{
    fmt_e12, get_f64, get_i64, get_str, load_toml_config, ness_iter_row, ness_outdir, write_ness_gf,
};

use super::util::prelude::*;

/// The result of `run_steady_state`.
pub(crate) struct SteadyStateResult {
    pub(crate) g: SteadyGreen,
    pub(crate) s: SteadyGreen,
    pub(crate) gweiss: SteadyGreen,
}

/// Read the parameters from `cfgfile`, run the DMFT loop, and write the
/// results .
pub(crate) fn run_steady_state(cfgfile: &str) -> Result<SteadyStateResult, String> {
    let p = load_toml_config(cfgfile)?;
    let (nft, h) = (get_i64(&p, "grid", "Nft")?, get_f64(&p, "grid", "h")?);
    let (beta, u, mu) = (
        get_f64(&p, "model", "beta")?,
        get_f64(&p, "model", "U")?,
        get_f64(&p, "model", "mu")?,
    );
    let (niter, errmax, mix) = (
        get_i64(&p, "solver", "Niter")?,
        get_f64(&p, "solver", "errmax")?,
        get_f64(&p, "solver", "mixing")?,
    );
    let (ipt, out_every) = (
        get_i64(&p, "solver", "ipt_flag")?,
        get_i64(&p, "solver", "out_every")?,
    );
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = get_str(&p, "output", "prefix")?;
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
            (2.0 * (1.0 - (eps / 2.0) * (eps / 2.0))).sqrt() / (2.0 * PI)
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
    let mut fit =
        File::create(format!("{dir}/{prefix}_iterations.dat")).map_err(|e| e.to_string())?;
    writeln!(fit, "# iteration  error  density").map_err(|e| e.to_string())?;
    for iter in 1..=niter {
        let g_old = gg.clone();
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
        let err = gg.distance(&g_old, FftDomain::Time);
        // linear mixing
        gg.smul(Complex::new(mix, 0.0), FftDomain::Time);
        gg.incr(&g_old, Complex::new(1.0 - mix, 0.0), FftDomain::Time);

        errs.push(err);
        let dens = 2.0 * steady_density(&gg, Sign::Fermi).re;
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
    let sug = steady_convolution_density_matrix(&su, &gg, Sign::Fermi);
    let eint = sug[0].re;
    let dens = 2.0 * steady_density(&gg, Sign::Fermi).re;
    let mut f =
        File::create(format!("{dir}/{prefix}_observables.dat")).map_err(|e| e.to_string())?;
    writeln!(f, "# density  Eint  iterations").map_err(|e| e.to_string())?;
    writeln!(f, "{} {} {}", fmt_e12(dens), fmt_e12(eint), itermax).map_err(|e| e.to_string())?;

    // spectra
    integral_transform_to_freq(&mut gg, FftIntegralMethod::Trapez);
    integral_transform_to_freq(&mut su, FftIntegralMethod::Trapez);
    write_ness_gf(&format!("{dir}/{prefix}_G_time.dat"), &gg, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_G_freq.dat"), &gg, FftDomain::Freq)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_time.dat"), &su, FftDomain::Time)?;
    write_ness_gf(&format!("{dir}/{prefix}_S_freq.dat"), &su, FftDomain::Freq)?;

    println!("density = {dens:.8}, Eint = {eint:.8}, iterations = {itermax}");
    Ok(SteadyStateResult {
        g: gg,
        s: su,
        gweiss,
    })
}
