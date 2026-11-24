//
// Usage:
//   cargo run --example hubbard -- bethe_prop examples/hubbard/hubbard_bethe_steady_prop.toml
//
// Real-time propagation of the DMFT equations of the Hubbard model on
// the Bethe lattice (bandwidth 4) in equilibrium, from NESSi
// examples/programs/steady_bethe_prop.cpp; no HDF5). The last time slice
// is converted to a steady-state Green's function and back; in
// equilibrium both densities agree and the deviation of G^R from the
// time translation invariant form measures the quality of the
// propagation.
//

#[path = "../../util/mod.rs"]
mod util;
use std::fs::File;
use std::io::Write;

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::linalg::Adjoint;
use kadanoffbaym::math::BetheDos;
use kadanoffbaym::types::{Contour, ContourGreen, Sign, SteadyGreen, SteadyGrid, TimeFunction};
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::types::MATSUBARA;

use kadanoffbaym::convolution::{bubble1, bubble2};
use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::{from_dos_mat, from_mat};
use kadanoffbaym::fourier::{contour_to_steady, steady_to_contour};
use kadanoffbaym::math::extrapolate_timestep;
use kadanoffbaym::observables::{density, steady_density};

use crate::util::{fmt_e12, get_f64, get_i64, get_str, load_toml_config, ness_outdir};
use util::add_green_step;

fn mt(n: i64) -> i64 {
    if n == 0 { MATSUBARA } else { n - 1 }
}

/// Second-order self-energy at the KB time step `tstp`:
/// `W = U [i G G] U` (bosonic) and `Sigma = -i G W`.
fn sigma_timestep(
    tstp: i64,
    sigma: &mut ContourGreen,
    ge: &ContourGreen,
    w: &mut ContourGreen,
    uf: &TimeFunction,
) {
    bubble1(mt(tstp), w, 0, 0, ge, ge, 0, 0, ge, ge, 0, 0);
    w.rmul_cf_at(mt(tstp), uf, Adjoint::No);
    w.lmul_cf_at(mt(tstp), uf, Adjoint::No);
    bubble2(mt(tstp), sigma, 0, 0, ge, ge, 0, 0, w, w, 0, 0);
    sigma.smul_at(mt(tstp), Complex::new(-1.0, 0.0));
}

/// The result of the propagation.
#[allow(dead_code)]
pub struct PropResult {
    pub density: Vec<f64>,
    pub density_ness: f64,
    pub tti: f64,
    pub mat_iterations: i64,
    pub errors: Vec<f64>,
    pub error_log: Vec<(i64, i64, i64, f64)>,
    pub g: ContourGreen,
    pub gness: SteadyGreen,
}

/// Read the parameters, run the propagation, and write the results.
pub fn prop_main(cfgfile: &str) -> Result<PropResult, String> {
    let p = load_toml_config(cfgfile)?;
    let nt = get_i64(&p, "contour", "nt")?;
    // the TOML follows the blueprint point convention (dtau = beta /
    // (ntau - 1)); the library counts the imaginary-time intervals
    let ntau = get_i64(&p, "contour", "ntau")? - 1;
    let (h, beta) = (
        get_f64(&p, "contour", "h")?,
        get_f64(&p, "contour", "beta")?,
    );
    let (u, mu) = (get_f64(&p, "model", "U")?, get_f64(&p, "model", "mu")?);
    let (k, ipt) = (
        get_i64(&p, "solver", "SolveOrder")?,
        get_i64(&p, "solver", "ipt_flag")?,
    );
    let method = match get_str(&p, "solver", "mat_method")?.as_str() {
        "fixpoint" => MatsubaraMethod::Fixpoint,
        "fourier" => MatsubaraMethod::Fourier,
        s => return Err(format!("{cfgfile}: [solver] mat_method {s} not supported")),
    };
    let (maxiter, maxerr) = (
        get_i64(&p, "solver", "BootstrapMaxIter")?,
        get_f64(&p, "solver", "BootstrapMaxErr")?,
    );
    let ncorr = get_i64(&p, "solver", "CorrectorSteps")?;
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = get_str(&p, "output", "prefix")?;
    if !(ntau >= 2 && (1..=5).contains(&k) && nt >= k) {
        return Err(format!("{cfgfile}: invalid contour parameters"));
    }
    if !(h > 0.0 && beta > 0.0 && maxiter >= 0 && ncorr >= 0) {
        return Err(format!("{cfgfile}: invalid solver parameters"));
    }

    let ntime = nt + 1;
    let cc = Contour::new(nt, ntau, 1, 1, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut g = fermion();
    let mut gweiss = fermion();
    let mut sigma = fermion();
    let mut kker = fermion();
    let mut w = ContourGreen::new(&cc, Sign::Bose, Complex::ZERO);
    let mut uf = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    let mut eloc = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    for n in MATSUBARA..=nt {
        uf.block_mut(n)[0] = Complex::new(u, 0.0);
        eloc.block_mut(n)[0] = Complex::ZERO;
    }
    let mut gtemp = fermion();

    // K = Sigma + G at the time step tstp
    let kernel = |sigma: &ContourGreen, g: &ContourGreen, k: &mut ContourGreen, t: i64| {
        k.memcpy_from_at(t, sigma);
        add_green_step(k, g, t, Complex::ONE);
    };

    let mut ferr = File::create(format!("{dir}/{prefix}_errors.dat")).map_err(|e| e.to_string())?;
    writeln!(ferr, "# stage  tstp  iteration  error").map_err(|e| e.to_string())?;
    let mut errors: Vec<f64> = Vec::new();
    let mut error_log: Vec<(i64, i64, i64, f64)> = Vec::new();
    let mut record = |stage: i64, tstp: i64, iter: i64, err: f64| -> Result<(), String> {
        errors.push(err);
        error_log.push((stage, tstp, iter, err));
        writeln!(ferr, "{} {} {} {}", stage, tstp, iter, fmt_e12(err)).map_err(|e| e.to_string())
    };

    // Matsubara: the noninteracting Bethe Green's function as the guess
    let rho_dos = BetheDos::new(-2.0, 2.0);
    from_dos_mat(
        &mut g,
        &|eps: f64| rho_dos.at(eps),
        rho_dos.lo,
        rho_dos.hi,
        beta,
        0.0,
        100,
        20,
    );
    gweiss.memcpy_from_at(MATSUBARA, &g);
    for q in sigma.mat.raw_mut() {
        *q = Complex::ZERO;
    }
    kernel(&sigma, &g, &mut kker, MATSUBARA);
    let mut itermax = maxiter;
    for iter in 0..=maxiter {
        gtemp.memcpy_from_at(MATSUBARA, &g);
        if ipt != 0 {
            dyson_mat(
                &mut gweiss,
                &g,
                mu,
                &eloc,
                beta,
                k,
                method,
                true,
                None,
                1e-12,
            );
        } else {
            gweiss.memcpy_from_at(MATSUBARA, &g);
        }
        sigma_timestep(0, &mut sigma, &gweiss, &mut w, &uf);
        kernel(&sigma, &g, &mut kker, MATSUBARA);
        dyson_mat(&mut g, &kker, mu, &eloc, beta, k, method, true, None, 1e-12);
        let err = g.distance_at(&gtemp, MATSUBARA);
        record(0, 0, iter, err)?;
        if err < maxerr && iter > 3 {
            itermax = iter;
            break;
        }
    }

    // Bootstrapping: the KB time steps 1..k+1
    let nb = k + 1;
    from_mat(&mut g);
    for iter in 0..=maxiter {
        gtemp.memcpy_from_at(nb - 1, &g);
        if ipt != 0 {
            dyson_start(&mut gweiss, mu, &eloc, &g, beta, h, k);
        } else {
            for n in 1..=nb {
                gweiss.memcpy_from_at(n - 1, &g);
            }
        }
        for n in 1..=nb {
            sigma_timestep(n, &mut sigma, &gweiss, &mut w, &uf);
            kernel(&sigma, &g, &mut kker, n - 1);
        }
        dyson_start(&mut g, mu, &eloc, &kker, beta, h, k);
        let err = g.distance_at(&gtemp, nb - 1);
        record(1, nb, iter, err)?;
        if err < maxerr && iter > 3 {
            break;
        }
    }

    // Time stepping
    for tstp in (nb + 1)..=ntime {
        gtemp.memcpy_from_at(tstp - 1, &g);
        extrapolate_timestep(tstp - 2, &mut g, &Integrator::new(k));
        let d_integ = Integrator::new(k);
        let mut dws = DysonWorkspace::new(k, g.size1, g.ntau);
        for iter in 0..=ncorr {
            gtemp.memcpy_from_at(tstp - 1, &g);
            if ipt != 0 {
                // `g` (as sigma) evolves per iteration
                dws.reset();
                dyson_timestep_ws(
                    tstp - 1,
                    &mut gweiss,
                    mu,
                    &eloc,
                    &g,
                    beta,
                    h,
                    &d_integ,
                    &mut dws,
                );
            } else {
                gweiss.memcpy_from_at(tstp - 1, &g);
            }
            sigma_timestep(tstp, &mut sigma, &gweiss, &mut w, &uf);
            kernel(&sigma, &g, &mut kker, tstp - 1);
            // a different sigma (`kker`)
            dws.reset();
            dyson_timestep_ws(
                tstp - 1,
                &mut g,
                mu,
                &eloc,
                &kker,
                beta,
                h,
                &d_integ,
                &mut dws,
            );
            let err = g.distance_at(&gtemp, tstp - 1);
            record(2, tstp, iter, err)?;
        }
    }
    drop(ferr);

    // Densities, n = n_up + n_down (the KB time steps 0..ntime)
    let dens: Vec<f64> = (0..=ntime)
        .map(|tstp| 2.0 * density(&g, mt(tstp)).re)
        .collect();
    let mut f = File::create(format!("{dir}/{prefix}_density.dat")).map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  density").map_err(|e| e.to_string())?;
    writeln!(f, "0 {} {}", fmt_e12(0.0), fmt_e12(dens[0])).map_err(|e| e.to_string())?;
    for tstp in 1..=ntime {
        writeln!(
            f,
            "{} {} {}",
            tstp,
            fmt_e12((tstp - 1) as f64 * h),
            fmt_e12(dens[tstp as usize])
        )
        .map_err(|e| e.to_string())?;
    }

    // Steady-state view of the last time slice and back
    let gn_grid = SteadyGrid::new(2 * ntime, h);
    let mut gn = SteadyGreen::new(gn_grid, 1);
    contour_to_steady(&mut gn, &g, nt); // my step of the KB ntime
    let dens_ness = 2.0 * steady_density(&gn, Sign::Fermi).re;
    let mut gtti = fermion();
    steady_to_contour(&mut gtti, &gn);
    let mut tti = 0.0f64;
    for n in 1..=ntime {
        for m in 1..=n {
            let d = (g.ret.raw_block_at(n - 1, m - 1)[0] - gtti.ret.raw_block_at(n - 1, m - 1)[0])
                .abs();
            tti = tti.max(d);
        }
    }
    let mut f = File::create(format!("{dir}/{prefix}_ness.dat")).map_err(|e| e.to_string())?;
    writeln!(f, "# density_contour  density_ness  max|G^R - G^R_tti|")
        .map_err(|e| e.to_string())?;
    writeln!(
        f,
        "{} {} {}",
        fmt_e12(dens[ntime as usize]),
        fmt_e12(dens_ness),
        fmt_e12(tti)
    )
    .map_err(|e| e.to_string())?;

    println!(
        "Matsubara iterations = {itermax}, density(t_max) = {:.8}, density(NESS) = {dens_ness:.8}, TTI deviation = {tti:.3e}",
        dens[ntime as usize]
    );
    Ok(PropResult {
        density: dens,
        density_ness: dens_ness,
        tti,
        mat_iterations: itermax,
        errors,
        error_log,
        g,
        gness: gn,
    })
}
