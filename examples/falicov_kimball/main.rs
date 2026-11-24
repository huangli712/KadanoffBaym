// Project : KadanoffBaym
// File    : examples/falicov_kimball/main.rs
// Purpose : interaction quench of the Falicov-Kimball model on the Bethe lattice.
//
// Usage:
//   cargo run --example falicov_kimball -- examples/falicov_kimball/falicov_kimball.toml [--profile quick|release] [--validate-only]
//

use std::path::Path;

use std::env::args;
use std::fs::create_dir_all;
use std::process::exit;

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::math::BetheDos;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::math::extrapolate_timestep;
use kadanoffbaym::observables::density;

use util::ExampleConfig;

use util::{
    contour_params, iterate_bootstrap, load_example_config, matsubara_step, parse_example_args,
    solver_params, write_table,
};

use util::prelude::*;

#[path = "../../util/mod.rs"]
mod util;

pub(crate) const MODEL_KEYS: [&str; 2] = ["U0", "U1"];

/// The state of the Falicov-Kimball model.
pub(crate) struct FalicovKimballState {
    pub(crate) g0: ContourGreen,
    pub(crate) g1: ContourGreen,
    pub(crate) g: ContourGreen,
    pub(crate) delta: ContourGreen,
    pub(crate) g_old: ContourGreen,
    pub(crate) f0: TimeFunction,
    pub(crate) f1: TimeFunction,
    pub(crate) beta: f64,
    pub(crate) h: f64,
    pub(crate) k: i64,
    pub(crate) nt: i64,
    pub(crate) integ: Integrator,
}

/// Read and validate the TOML file; `U0` and `U1` must be finite and
/// non-negative.
pub(crate) fn load_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    let cfg = load_example_config(cfgfile, profile, &MODEL_KEYS, &[], &[], &[])?;
    for key in MODEL_KEYS {
        let v = cfg.model[key].as_f64().unwrap();
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!(
                "{cfgfile}: [model] {key} must be a non-negative number"
            ));
        }
    }
    solver_params(&cfg)?;
    Ok(cfg)
}

/// Allocate the Green's functions, `Delta` = the Matsubara Green's
/// function of the Bethe lattice, and `f0`, `f1`.
pub(crate) fn build_state(cfg: &ExampleConfig) -> Result<FalicovKimballState, String> {
    let c = contour_params(cfg)?;
    let cc = Contour::new(c.nt, c.ntau, 1, 1, c.nt as f64 * c.h, c.beta);
    let mut delta = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    // the Bethe lattice defaults of `from_dos_mat_bethe!` (V = 1, E0 = 0)
    let rho = BetheDos::new(-2.0, 2.0);
    let (lo, hi) = (rho.lo, rho.hi);
    from_dos_mat(
        &mut delta,
        &|eps: f64| rho.at(eps),
        lo,
        hi,
        c.beta,
        0.0,
        100,
        20,
    );
    let u0 = cfg.model["U0"].as_f64().unwrap();
    let u1 = cfg.model["U1"].as_f64().unwrap();
    let mut f0 = TimeFunction::new(c.nt as usize, 1, 1, Complex::ZERO);
    let mut f1 = TimeFunction::new(c.nt as usize, 1, 1, Complex::ZERO);
    f0.block_mut(MATSUBARA)[0] = Complex::new(-0.5 * u0, 0.0);
    f1.block_mut(MATSUBARA)[0] = Complex::new(0.5 * u0, 0.0);
    for i in 0..=c.nt {
        f0.block_mut(i)[0] = Complex::new(-0.5 * u1, 0.0);
        f1.block_mut(i)[0] = Complex::new(0.5 * u1, 0.0);
    }
    Ok(FalicovKimballState {
        g0: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        g1: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        g: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        delta,
        g_old: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
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
fn average_density(st: &mut FalicovKimballState, n: i64) {
    let t = matsubara_step(n);
    let half = Complex::new(0.5, 0.0);
    if t == MATSUBARA {
        for m in 0..=st.g.ntau {
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

/// Matsubara iteration, bootstrapping, and propagation.
pub(crate) fn solve(st: &mut FalicovKimballState, cfg: &ExampleConfig) -> Result<(), String> {
    let s = solver_params(cfg)?;
    let c = contour_params(cfg)?;
    let (k, mu) = (st.k, 0.0);
    // Matsubara (the FOURIER method, at least four iterations as in
    // iterations): the previous iterate and Delta are updated only if the
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
        average_density(st, 0);
        err = st.g.distance_at(&st.g_old, MATSUBARA);
        println!("matsubara iteration {iter}  error = {err:.6e}");
        if err < s.mats_max_error && iter > 2 {
            break;
        }
        st.g_old.memcpy_from_at(MATSUBARA, &st.g);
        st.delta.memcpy_from_at(MATSUBARA, &st.g);
    }
    if err.is_nan() || err >= s.mats_max_error {
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
                average_density(st, n);
                err += st.g.distance_at(&st.g_old, n - 1);
            }
            if err >= s.bootstrap_max_error {
                for n in 1..=k + 1 {
                    st.g_old.memcpy_from_at(n - 1, &st.g);
                    st.delta.memcpy_from_at(n - 1, &st.g);
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
    let d_integ = Integrator::new(k);
    let mut dws = DysonWorkspace::new(k, st.g0.size1, st.g0.ntau);
    for n in (k + 2)..=(c.nt + 1) {
        // the extrapolates the KB row n of Delta from the rows
        // n-1, .., n-k-1; in my steps the argument is n - 2
        extrapolate_timestep(n - 2, &mut st.delta, &st.integ);
        for _iter in 0..s.corrector_steps {
            // both solves share the same sigma (`delta`), which is
            // refreshed per iteration, hence the reset
            dws.reset();
            dyson_timestep_ws(
                n - 1,
                &mut st.g0,
                mu,
                &st.f0,
                &st.delta,
                st.beta,
                st.h,
                &d_integ,
                &mut dws,
            );
            dyson_timestep_ws(
                n - 1,
                &mut st.g1,
                mu,
                &st.f1,
                &st.delta,
                st.beta,
                st.h,
                &d_integ,
                &mut dws,
            );
            average_density(st, n);
            st.delta.memcpy_from_at(n - 1, &st.g);
        }
    }
    Ok(())
}

/// `(n, den0, den1)` for the KB time steps `n = 0, .., ntime` (`n = 0`
/// is the Matsubara branch), `den = Re rho`.
pub(crate) fn densities(st: &FalicovKimballState) -> Vec<(i64, f64, f64)> {
    (0..=st.nt + 1)
        .map(|n| {
            let d0 = density(&st.g0, matsubara_step(n)).re;
            let d1 = density(&st.g1, matsubara_step(n)).re;
            (n, d0, d1)
        })
        .collect()
}

/// Entry of the example.
pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_config(&a.cfgfile, &a.profile)?;
    let mut st = build_state(&cfg)?;
    if a.validate_only {
        return Ok(());
    }
    solve(&mut st, &cfg)?;
    let dens = densities(&st);
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    create_dir_all(&dir).map_err(|e| e.to_string())?;
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
    Ok(())
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("{e}");
        exit(1);
    }
}
