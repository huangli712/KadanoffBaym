//
// Usage:
//   cargo run --example hubbard_bethe_ness2_prop -- julia/tutor/hubbard_bethe/hubbard_bethe_ness2_prop.toml
//
// Real-time propagation of the DMFT equations of the Hubbard model on
// the Bethe lattice (bandwidth 4) in equilibrium (the Rust port of
// julia/tutor/hubbard_bethe/hubbard_bethe_ness2_prop.jl, from NESSi
// examples/programs/ness2_bethe_prop.cpp; no HDF5). The last time slice
// is converted to a steady-state Green's function and back; in
// equilibrium both densities agree and the deviation of G^R from the
// time translation invariant form measures the quality of the
// propagation.
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1, bubble2};
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::math::dos::BetheDos;
use kadanoffbaym::observables::{density_matrix_green, density_matrix_ness};
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::equilibrium::from_mat::from_mat;
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::fourier::convert::{cntr2ness, ness2cntr};
use kadanoffbaym::types::ness::{NessGrid, NessHermMat};
use ness_util::{fmt_e12, tget_f, tget_i, tget_s};
use std::io::Write;

fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// Second-order self-energy at the KB time step `tstp`:
/// `W = U [i G G] U` (bosonic) and `Sigma = -i G W`.
fn sigma_timestep(tstp: i64, sigma: &mut ContourGreen, ge: &ContourGreen, w: &mut ContourGreen, uf: &TimeFunction) {
    bubble1(mt(tstp), w, 0, 0, ge, ge, 0, 0, ge, ge, 0, 0);
    w.rmul_cf_at(mt(tstp), uf, kadanoffbaym::basics::Mode::Normal);
    w.lmul_cf_at(mt(tstp), uf, kadanoffbaym::basics::Mode::Normal);
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
    pub gness: NessHermMat,
}

/// Read the parameters, run the propagation, and write the results.
pub fn prop_main(cfgfile: &str) -> Result<PropResult, String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let nt = tget_i(&p, "contour", "nt")?;
    // the TOML follows the blueprint point convention (dtau = beta /
    // (ntau - 1)); the library counts the imaginary-time intervals
    let ntau = tget_i(&p, "contour", "ntau")? - 1;
    let (h, beta) = (
        tget_f(&p, "contour", "h")?,
        tget_f(&p, "contour", "beta")?,
    );
    let (u, mu) = (
        tget_f(&p, "model", "U")?,
        tget_f(&p, "model", "mu")?,
    );
    let (k, ipt) = (
        tget_i(&p, "solver", "SolveOrder")?,
        tget_i(&p, "solver", "ipt_flag")?,
    );
    let method = match tget_s(&p, "solver", "mat_method")?.as_str() {
        "fixpoint" => MatsubaraMethod::Fixpoint,
        "fourier" => MatsubaraMethod::Fourier,
        s => return Err(format!("{cfgfile}: [solver] mat_method {s} not supported")),
    };
    let (maxiter, maxerr) = (
        tget_i(&p, "solver", "BootstrapMaxIter")?,
        tget_f(&p, "solver", "BootstrapMaxErr")?,
    );
    let ncorr = tget_i(&p, "solver", "CorrectorSteps")?;
    let dir = ness_util::ness_outdir(&p, cfgfile)?;
    let prefix = tget_s(&p, "output", "prefix")?;
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
        copy_step_zs(sigma, k, t);
        kadanoffbaym_add_step(k, g, t, Complex::ONE);
    };

    let mut ferr = std::fs::File::create(format!("{dir}/{prefix}_errors.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(ferr, "# stage  tstp  iteration  error").map_err(|e| e.to_string())?;
    let mut errors: Vec<f64> = Vec::new();
    let mut error_log: Vec<(i64, i64, i64, f64)> = Vec::new();
    let mut record = |stage: i64, tstp: i64, iter: i64, err: f64| -> Result<(), String> {
        errors.push(err);
        error_log.push((stage, tstp, iter, err));
        writeln!(ferr, "{} {} {} {}", stage, tstp, iter, fmt_e12(err))
            .map_err(|e| e.to_string())
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
    copy_step_zs(&g, &mut gweiss, MATSUBARA);
    for q in sigma.mat.raw_mut() {
        *q = Complex::ZERO;
    }
    kernel(&sigma, &g, &mut kker, MATSUBARA);
    let mut itermax = maxiter;
    for iter in 0..=maxiter {
        copy_step_zs(&g, &mut gtemp, MATSUBARA);
        if ipt != 0 {
            dyson_mat(&mut gweiss, &g, mu, &eloc, beta, k, method, true, None, 1e-12);
        } else {
            copy_step_zs(&g, &mut gweiss, MATSUBARA);
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
        copy_step_zs(&g, &mut gtemp, nb - 1);
        if ipt != 0 {
            dyson_start(&mut gweiss, mu, &eloc, &g, beta, h, k);
        } else {
            for n in 1..=nb {
                copy_step_zs(&g, &mut gweiss, n - 1);
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
        copy_step_zs(&g, &mut gtemp, tstp - 1);
        kadanoffbaym::math::extrap::extrapolate_timestep(
            tstp - 2,
            &mut g,
            &kadanoffbaym::weights::Integrator::new(k),
        );
        for iter in 0..=ncorr {
            copy_step_zs(&g, &mut gtemp, tstp - 1);
            if ipt != 0 {
                dyson_timestep(tstp - 1, &mut gweiss, mu, &eloc, &g, beta, h, k);
            } else {
                copy_step_zs(&g, &mut gweiss, tstp - 1);
            }
            sigma_timestep(tstp, &mut sigma, &gweiss, &mut w, &uf);
            kernel(&sigma, &g, &mut kker, tstp - 1);
            dyson_timestep(tstp - 1, &mut g, mu, &eloc, &kker, beta, h, k);
            let err = g.distance_at(&gtemp, tstp - 1);
            record(2, tstp, iter, err)?;
        }
    }
    drop(ferr);

    // Densities, n = n_up + n_down (the KB time steps 0..ntime)
    let dens: Vec<f64> = (0..=ntime)
        .map(|tstp| 2.0 * density_matrix_green(&g, mt(tstp)).get(0, 0).re)
        .collect();
    let mut f = std::fs::File::create(format!("{dir}/{prefix}_density.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  density").map_err(|e| e.to_string())?;
    write!(f, "0 {} {}\n", fmt_e12(0.0), fmt_e12(dens[0])).map_err(|e| e.to_string())?;
    for tstp in 1..=ntime {
        write!(
            f,
            "{} {} {}\n",
            tstp,
            fmt_e12((tstp - 1) as f64 * h),
            fmt_e12(dens[tstp as usize])
        )
        .map_err(|e| e.to_string())?;
    }

    // Steady-state view of the last time slice and back
    let gn_grid = NessGrid::new(2 * ntime, h);
    let mut gn = NessHermMat::new(gn_grid, 1);
    cntr2ness(&mut gn, &g, nt);  // my step of the KB ntime
    let dens_ness = 2.0 * density_matrix_ness(&gn, Sign::Fermi)[0].re;
    let mut gtti = fermion();
    ness2cntr(&mut gtti, &gn);
    let mut tti = 0.0f64;
    for n in 1..=ntime {
        for m in 1..=n {
            let d = (g.ret.raw_block_at(n - 1, m - 1)[0] - gtti.ret.raw_block_at(n - 1, m - 1)[0])
                .abs();
            tti = tti.max(d);
        }
    }
    let mut f = std::fs::File::create(format!("{dir}/{prefix}_ness.dat"))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# density_contour  density_ness  max|G^R - G^R_tti|").map_err(|e| e.to_string())?;
    write!(
        f,
        "{} {} {}\n",
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

/// Copy a single step of `src` into `dst` (blueprint `memcpy!`).
fn copy_step_zs(src: &ContourGreen, dst: &mut ContourGreen, t: i64) {
    if t == MATSUBARA {
        dst.mat.memcpy_from(&src.mat);
    } else {
        for j in 0..=t {
            let v = src.ret.raw_block_at(t, j).to_vec();
            dst.ret.raw_block_at_mut(t, j).copy_from_slice(&v);
        }
        for i in 0..=t {
            let v = src.less.raw_block_at(i, t).to_vec();
            dst.less.raw_block_at_mut(i, t).copy_from_slice(&v);
        }
        for m in 0..=src.ntau as i64 {
            let v = src.lmix.raw_block_at(t, m).to_vec();
            dst.lmix.raw_block_at_mut(t, m).copy_from_slice(&v);
        }
    }
}

/// `dst += alpha * src` at a single step (the local `add_step`).
fn kadanoffbaym_add_step(dst: &mut ContourGreen, src: &ContourGreen, t: i64, alpha: Complex) {
    if t == MATSUBARA {
        for q in 0..dst.mat.raw().len() {
            dst.mat.raw_mut()[q] += alpha * src.mat.raw()[q];
        }
    } else {
        for j in 0..=t {
            for q in 0..dst.ret.raw_block_at(t, j).len() {
                dst.ret.raw_block_at_mut(t, j)[q] += alpha * src.ret.raw_block_at(t, j)[q];
            }
        }
        for i in 0..=t {
            for q in 0..dst.less.raw_block_at(i, t).len() {
                dst.less.raw_block_at_mut(i, t)[q] += alpha * src.less.raw_block_at(i, t)[q];
            }
        }
        for m in 0..=dst.ntau as i64 {
            for q in 0..dst.lmix.raw_block_at(t, m).len() {
                dst.lmix.raw_block_at_mut(t, m)[q] += alpha * src.lmix.raw_block_at(t, m)[q];
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: hubbard_bethe_ness2_prop <config.toml>");
        std::process::exit(1);
    }
    if let Err(e) = prop_main(&args[0]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
