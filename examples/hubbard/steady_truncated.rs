// Project : KadanoffBaym
// File    : examples/hubbard/steady_truncated.rs
// Purpose : truncated moving-window propagation started from the steady state.
//
// Usage:
//   cargo run --example hubbard -- bethe_steady_trunc examples/hubbard/bethe_steady_truncated.toml
//

use std::fs::File;
use std::io::Write;

use kadanoffbaym::dyson::MovingDysonWorkspace;
use kadanoffbaym::math::MovingStageWorkspace;
use kadanoffbaym::types::{MovingGreen, MovingTimeFunction, SteadyGreen, SteadyGrid};
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::dyson::dyson_timestep_moving;
use kadanoffbaym::fourier::downsample;
use kadanoffbaym::math::extrapolate_timestep_moving;
use kadanoffbaym::variants::{
    copy_timestep, distance_moving, forward, incr_step, init_from_function, init_from_steady,
};

use super::steady_state::SteadyStateResult;

use super::bethe_trunc::{moving_density, update_kernel, update_selfenergy};
use super::steady_state::run_steady_state;
use super::util::{fmt_e12, get_f64, get_i64, get_str, load_toml_config, ness_outdir};

use super::util::prelude::*;

/// One propagation record.
pub(crate) struct TruncStep {
    pub(crate) rel: i64,
    pub(crate) tstp: i64,
    pub(crate) density: f64,
    pub(crate) err: f64,
}

/// Moving window (cut-off `tc`) of the steady-state Green's function
/// `gness` downsampled by `factor`; initialized at the KB time step 1.
fn initialize_from_steady(gness: &SteadyGreen, tc: i64, factor: i64) -> MovingGreen {
    let mut gm = MovingGreen::new(tc, 1, Sign::Fermi);
    init_from_steady(&mut gm, &downsample(gness, factor));
    gm
}

pub(crate) fn run_steady_truncated(cfgfile: &str) -> Result<(), String> {
    let p = load_toml_config(cfgfile)?;
    let t = p["trunc"].as_table().unwrap();
    let (tc, tmax) = (
        t.get("tc")
            .and_then(|v| v.as_i64())
            .ok_or("missing [trunc] tc")?,
        t.get("tmax")
            .and_then(|v| v.as_i64())
            .ok_or("missing [trunc] tmax")?,
    );
    let factor = t
        .get("downsampling")
        .and_then(|v| v.as_i64())
        .ok_or("missing [trunc] downsampling")?;
    let (ncorr, k) = (
        t.get("CorrectorSteps")
            .and_then(|v| v.as_i64())
            .ok_or("missing [trunc] CorrectorSteps")?,
        t.get("SolveOrder")
            .and_then(|v| v.as_i64())
            .ok_or("missing [trunc] SolveOrder")?,
    );
    let (u, mu) = (get_f64(&p, "model", "U")?, get_f64(&p, "model", "mu")?);
    let ipt = get_i64(&p, "solver", "ipt_flag")?;
    let h = get_f64(&p, "grid", "h")? * factor as f64;
    let dir = ness_outdir(&p, cfgfile)?;
    let prefix = get_str(&p, "output", "prefix")?;
    if !((1..=5).contains(&k) && tc >= 2 * k + 2 && tmax >= 1 && factor >= 1 && ncorr >= 0) {
        return Err(format!("{cfgfile}: invalid [trunc] parameters"));
    }

    // The steady state computed by steady_state.rs
    let steady = run_steady_state(cfgfile)?;

    let mut g = initialize_from_steady(&steady_green(&steady), tc, factor);
    let mut sigma = initialize_from_steady(&steady_sigma(&steady), tc, factor);
    let mut gw = initialize_from_steady(&steady_interaction(&steady), tc, factor);
    let mut kk = initialize_from_steady(&steady_green(&steady), tc, factor);
    for delti in 0..=tc {
        let mut krow = copy_timestep(&kk, delti);
        let srow = copy_timestep(&sigma, delti);
        incr_step(&mut krow, &srow, Complex::ONE);
        kk.ret.mrow_mut(delti).copy_from_slice(krow.ret.raw());
        kk.less.mrow_mut(delti).copy_from_slice(krow.less.raw());
    }
    // an initialized zero bosonic window
    let mut w = MovingGreen::new(tc, 1, Sign::Bose);
    {
        let empty = SteadyGreen::new(SteadyGrid::new(4 * (tc + 1), h), 1);
        init_from_steady(&mut w, &empty);
    }
    let nt_cf = tc + 1;
    let mut ucf = TimeFunction::new(nt_cf as usize, 1, 1, Complex::ZERO);
    for n in MATSUBARA..=nt_cf {
        ucf.block_mut(n)[0] = Complex::new(u, 0.0);
    }
    let mut uf = MovingTimeFunction::new(tc, 1);
    init_from_function(&mut uf, &ucf, tc);
    let mut eloc = MovingTimeFunction::new(tc, 1);
    let zcf = TimeFunction::new(nt_cf as usize, 1, 1, Complex::ZERO);
    init_from_function(&mut eloc, &zcf, tc);
    let density0 = moving_density(&g, 0);
    // the constant functions are aligned with the steady-state windows (KB step 1)
    uf.tstp = 0;
    eloc.tstp = 0;

    let integ = Integrator::new(k);
    let mut steps: Vec<TruncStep> = Vec::new();
    for rel in 1..=tmax {
        for x in [&mut sigma, &mut kk, &mut g, &mut gw, &mut w] {
            forward(x);
        }
        {
            let v = uf.fslot(1)[0];
            uf.forward();
            uf.fslot_mut(0)[0] = v;
        }
        {
            let v = eloc.fslot(1)[0];
            eloc.forward();
            eloc.fslot_mut(0)[0] = v;
        }
        let mut mws = MovingDysonWorkspace::new(k, gw.ret.size1);
        let mut sws = MovingStageWorkspace::new(k, g.tc, g.ret.size1);
        let n = g.tstp; // my step; the KB time step is n + 1
        extrapolate_timestep_moving(&mut g, k, &mut sws);
        let mut err = 0.0;
        for _ in 0..=ncorr {
            let gtmp = copy_timestep(&g, 0);
            if ipt != 0 {
                dyson_timestep_moving(n, &mut gw, mu, &eloc, &g, h, &integ, &mut mws);
            } else {
                g.ret.mrow_mut(0).copy_from_slice(gw.ret.mrow(0));
                g.less.mrow_mut(0).copy_from_slice(gw.less.mrow(0));
            }
            update_selfenergy(&mut sigma, &gw, &mut w, &uf);
            update_kernel(&mut kk, &sigma, &g);
            dyson_timestep_moving(n, &mut g, mu, &eloc, &kk, h, &integ, &mut mws);
            err = distance_moving(&copy_timestep(&g, 0), &gtmp);
        }
        steps.push(TruncStep {
            rel,
            tstp: n + 1,
            density: moving_density(&g, 0),
            err,
        });
    }

    let mut f =
        File::create(format!("{dir}/{prefix}_trunc_density.dat")).map_err(|e| e.to_string())?;
    writeln!(f, "# rel  tstp  density  corrector_error").map_err(|e| e.to_string())?;
    writeln!(f, "0 1 {} {}", fmt_e12(density0), fmt_e12(0.0)).map_err(|e| e.to_string())?;
    for s in &steps {
        writeln!(
            f,
            "{} {} {} {}",
            s.rel,
            s.tstp,
            fmt_e12(s.density),
            fmt_e12(s.err)
        )
        .map_err(|e| e.to_string())?;
    }
    println!(
        "density(steady) = {:.10}, density(t_max) = {:.10}",
        density0,
        steps.last().map(|s| s.density).unwrap_or(density0)
    );
    Ok(())
}

// the final Green's functions are carried by the steady result
fn steady_green(r: &SteadyStateResult) -> SteadyGreen {
    r.g.clone()
}

fn steady_sigma(r: &SteadyStateResult) -> SteadyGreen {
    r.s.clone()
}

fn steady_interaction(r: &SteadyStateResult) -> SteadyGreen {
    r.gweiss.clone()
}
