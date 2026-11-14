//
// Interaction quench U0 -> U1 of the Falicov-Kimball model on the Bethe
// lattice in DMFT (the Rust port of julia/tutor/fkm_bethe/fkm_bethe.jl,
// from NESSi examples/programs/fkm_bethe_quench.cpp):
//
//   G0 = [i d/dt + mu - f0 - Delta]^-1,  f0 = -U/2
//   G1 = [i d/dt + mu - f1 - Delta]^-1,  f1 = +U/2
//   G  = (G0 + G1) / 2,  Delta = G,  mu = 0
//
// U = U0 on the Matsubara branch and U = U1 at t >= 0. The Matsubara
// Dyson equations use the Fourier method. Output: <prefix>_density.dat
// with t, den0, den1, total (real time steps).
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::config::{
    contour_params, load_example_config, parse_example_args, solver_params, ExampleConfig,
};
use crate::common::contour_loop::iterate_bootstrap;
use crate::common::output::write_table;
use kadanoffbaym::c64::C64;
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::math::dos::BetheDos;
use kadanoffbaym::observables::density_matrix_green;
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::weights::Integrator;

pub const FKM_MODEL_KEYS: [&str; 2] = ["U0", "U1"];

/// Map a blueprint KB step onto the library step.
fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// Copy a single step of `src` into `dst` (blueprint `memcpy!`; the
/// whole-component copy of `elemcpy_green_at` would clobber the other
/// steps of the destination).
pub fn copy_step(src: &ContourGreen, dst: &mut ContourGreen, t: i64) {
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

/// The state of the Falicov-Kimball model.
pub struct FkmState {
    pub g0: ContourGreen,
    pub g1: ContourGreen,
    pub g: ContourGreen,
    pub delta: ContourGreen,
    pub gold: ContourGreen,
    pub f0: TimeFunction,
    pub f1: TimeFunction,
    pub beta: f64,
    pub h: f64,
    pub k: i64,
    pub nt: i64,
    pub integ: Integrator,
}

/// Read and validate the TOML file; `U0` and `U1` must be finite and
/// non-negative.
pub fn load_fkm_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    let cfg = load_example_config(cfgfile, profile, &FKM_MODEL_KEYS, &[], &[], &[])?;
    for key in FKM_MODEL_KEYS {
        let v = cfg.model[key].as_f64().unwrap();
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!("{cfgfile}: [model] {key} must be a non-negative number"));
        }
    }
    solver_params(&cfg)?;
    Ok(cfg)
}

/// Allocate the Green's functions, `Delta` = the Matsubara Green's
/// function of the Bethe lattice, and `f0`, `f1`.
pub fn fkm_state(cfg: &ExampleConfig) -> Result<FkmState, String> {
    let c = contour_params(cfg)?;
    let cc = Contour::new(c.nt, c.ntau, 1, 1, c.nt as f64 * c.h, c.beta);
    let mut delta = ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
    // the Bethe lattice defaults of `from_dos_mat_bethe!` (V = 1, E0 = 0)
    let rho = BetheDos::new(-2.0, 2.0);
    let (lo, hi) = (rho.lo, rho.hi);
    from_dos_mat(&mut delta, &|eps: f64| rho.at(eps), lo, hi, c.beta, 0.0, 100, 20);
    let u0 = cfg.model["U0"].as_f64().unwrap();
    let u1 = cfg.model["U1"].as_f64().unwrap();
    let mut f0 = TimeFunction::new(c.nt as usize, 1, 1, C64::ZERO);
    let mut f1 = TimeFunction::new(c.nt as usize, 1, 1, C64::ZERO);
    f0.block_mut(MATSUBARA)[0] = C64::new(-0.5 * u0, 0.0);
    f1.block_mut(MATSUBARA)[0] = C64::new(0.5 * u0, 0.0);
    for i in 0..=c.nt {
        f0.block_mut(i)[0] = C64::new(-0.5 * u1, 0.0);
        f1.block_mut(i)[0] = C64::new(0.5 * u1, 0.0);
    }
    Ok(FkmState {
        g0: ContourGreen::new(&cc, Sign::Fermi, C64::ZERO),
        g1: ContourGreen::new(&cc, Sign::Fermi, C64::ZERO),
        g: ContourGreen::new(&cc, Sign::Fermi, C64::ZERO),
        delta,
        gold: ContourGreen::new(&cc, Sign::Fermi, C64::ZERO),
        f0,
        f1,
        beta: c.beta,
        h: c.h,
        k: c.k,
        nt: c.nt,
        integ: Integrator::new(c.k),
    })
}

/// `G = (G0 + G1) / 2` at the KB step `n`.
fn fkm_average(st: &mut FkmState, n: i64) {
    let t = mt(n);
    let half = C64::new(0.5, 0.0);
    if t == MATSUBARA {
        for m in 0..=st.g.ntau as usize {
            let v = st.g0.mat.block(m)[0] * half + st.g1.mat.block(m)[0] * half;
            st.g.mat.block_mut(m)[0] = v;
        }
    } else {
        for j in 0..=t {
            let v = st.g0.ret.raw_block_at(t, j)[0] * half + st.g1.ret.raw_block_at(t, j)[0] * half;
            st.g.ret.raw_block_at_mut(t, j)[0] = v;
        }
        for i in 0..=t {
            let v =
                st.g0.less.raw_block_at(i, t)[0] * half + st.g1.less.raw_block_at(i, t)[0] * half;
            st.g.less.raw_block_at_mut(i, t)[0] = v;
        }
        for m in 0..=st.g.ntau as i64 {
            let v =
                st.g0.lmix.raw_block_at(t, m)[0] * half + st.g1.lmix.raw_block_at(t, m)[0] * half;
            st.g.lmix.raw_block_at_mut(t, m)[0] = v;
        }
    }
}

/// Matsubara iteration, bootstrapping, and propagation (blueprint
/// `solve_fkm!`).
pub fn solve_fkm(st: &mut FkmState, cfg: &ExampleConfig) -> Result<(), String> {
    let s = solver_params(cfg)?;
    let c = contour_params(cfg)?;
    let (k, mu) = (st.k, 0.0);
    // Matsubara (the FOURIER method, at least four iterations as in
    // NESSi): the previous iterate and Delta are updated only if the
    // iteration did not converge
    let mut err = f64::NAN;
    for iter in 0..=s.mats_max_iter {
        dyson_mat(
            &mut st.g0,
            &st.delta,
            mu,
            &st.f0,
            st.beta,
            k,
            MatsubaraMethod::Fourier,
            true,
            None,
            1e-12,
        );
        dyson_mat(
            &mut st.g1,
            &st.delta,
            mu,
            &st.f1,
            st.beta,
            k,
            MatsubaraMethod::Fourier,
            true,
            None,
            1e-12,
        );
        fkm_average(st, 0);
        err = st.g.distance_at(&st.gold, MATSUBARA);
        println!("matsubara iteration {iter}  error = {err:.6e}");
        if err < s.mats_max_error && iter > 2 {
            break;
        }
        copy_step(&st.g, &mut st.gold, MATSUBARA);
        copy_step(&st.g, &mut st.delta, MATSUBARA);
    }
    if !(err < s.mats_max_error) {
        return Err(format!(
            "ConvergenceError: the matsubara iteration did not converge after {} iterations (error = {err})",
            s.mats_max_iter + 1
        ));
    }
    // Bootstrapping (exit at the first iteration below the threshold)
    iterate_bootstrap(
        |_iter| {
            dyson_start(&mut st.g0, mu, &st.f0, &st.delta, st.beta, st.h, k);
            dyson_start(&mut st.g1, mu, &st.f1, &st.delta, st.beta, st.h, k);
            let mut err = 0.0;
            for n in 1..=k + 1 {
                fkm_average(st, n);
                err += st.g.distance_at(&st.gold, n - 1);
            }
            if err >= s.bootstrap_max_error {
                for n in 1..=k + 1 {
                    copy_step(&st.g, &mut st.gold, n - 1);
                    copy_step(&st.g, &mut st.delta, n - 1);
                }
            }
            err
        },
        s.bootstrap_max_iter,
        s.bootstrap_max_error,
        0,
        |e| println!("{} iteration {}  error = {:.6e}", e.stage, e.iter, e.err),
    )
    .map_err(|e| e.to_string())?;
    // Propagation: the predictor extrapolates Delta, the correctors solve
    // both Dyson equations and update Delta = G
    for n in (k + 2)..=(c.nt + 1) {
        // the blueprint extrapolates the KB row n of Delta from the rows
        // n-1, .., n-k-1; in my steps the argument is n - 2
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut st.delta, &st.integ);
        for _iter in 0..s.corrector_steps {
            dyson_timestep(n - 1, &mut st.g0, mu, &st.f0, &st.delta, st.beta, st.h, k);
            dyson_timestep(n - 1, &mut st.g1, mu, &st.f1, &st.delta, st.beta, st.h, k);
            fkm_average(st, n);
            copy_step(&st.g, &mut st.delta, n - 1);
        }
    }
    Ok(())
}

/// `(n, den0, den1)` for the KB time steps `n = 0, .., ntime` (`n = 0`
/// is the Matsubara branch), `den = Re rho`.
pub fn fkm_densities(st: &FkmState) -> Vec<(i64, f64, f64)> {
    (0..=st.nt + 1)
        .map(|n| {
            let d0 = density_matrix_green(&st.g0, mt(n)).get(0, 0).re;
            let d1 = density_matrix_green(&st.g1, mt(n)).get(0, 0).re;
            (n, d0, d1)
        })
        .collect()
}

/// The result of `fkm_main`.
#[allow(dead_code)]
pub struct FkmResult {
    pub config: ExampleConfig,
    pub state: FkmState,
    pub densities: Option<Vec<(i64, f64, f64)>>,
}

/// Entry of the example.
pub fn fkm_main(args: &[String]) -> Result<FkmResult, String> {
    let a = parse_example_args(args)?;
    let cfg = load_fkm_config(&a.cfgfile, &a.profile)?;
    let mut st = fkm_state(&cfg)?;
    if a.validate_only {
        return Ok(FkmResult {
            config: cfg,
            state: st,
            densities: None,
        });
    }
    solve_fkm(&mut st, &cfg)?;
    let dens = fkm_densities(&st);
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if std::path::Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let rows: Vec<Vec<f64>> = dens
        .iter()
        .filter(|(n, _, _)| *n >= 1)
        .map(|(n, d0, d1)| vec![(n - 1) as f64 * st.h, *d0, *d1, d0 + d1])
        .collect();
    write_table(
        &format!(
            "{}/{}_density.dat",
            dir,
            cfg.output["prefix"].as_str().unwrap()
        ),
        &["t", "den0", "den1", "total"],
        &rows,
    )?;
    Ok(FkmResult {
        config: cfg,
        state: st,
        densities: Some(dens),
    })
}
