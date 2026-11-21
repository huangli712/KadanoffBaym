//
// Usage:
//   cargo run --example hubbard -- bethe_steady_trunc examples/hubbard/hubbard_bethe_steady_trunc.toml
//
// Truncated (moving window) propagation of the Hubbard model on the
// Bethe lattice, started from the steady state of hubbard_bethe_steady,
// from NESSi examples/programs/ness2_bethe_trunc.cpp; no HDF5).
//

// Shared example sources are included by several binaries.

use crate::trunc_model::moving_density;
use crate::util::fmt_e12;
use crate::util::{tget_f, tget_i, tget_s};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::{MovingDysonWorkspace, dyson_timestep_moving_ws};
use kadanoffbaym::fourier::downsample;
use kadanoffbaym::types::MATSUBARA;
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::TimeFunction;
use kadanoffbaym::types::{MovingGreen, MovingTimeFunction};
use kadanoffbaym::variants::copy_timestep;
use kadanoffbaym::variants::moving::init_from_steady;
use kadanoffbaym::variants::moving::{MovingStageWorkspace, extrapolate_timestep_moving_ws};
use kadanoffbaym::variants::moving::{forward, init_from_function};
use kadanoffbaym::variants::{distance_moving, incr_step};
use kadanoffbaym::weights::Integrator;

/// Moving window (cut-off `tc`) of the steady-state Green's function
/// `gness` downsampled by `factor`; initialized at the KB time step 1.
fn initialize_from_ness(
    gness: &kadanoffbaym::types::SteadyGreen,
    tc: i64,
    factor: i64,
) -> MovingGreen {
    let mut gm = MovingGreen::new(tc, 1, Sign::Fermi);
    init_from_steady(&mut gm, &downsample(gness, factor));
    gm
}

/// One propagation record.
#[allow(dead_code)]
pub struct TruncStep {
    pub rel: i64,
    pub tstp: i64,
    pub density: f64,
    pub err: f64,
}

/// The result of `steady_trunc_main`.
#[allow(dead_code)]
pub struct SteadyTruncResult {
    pub density0: f64,
    pub steps: Vec<TruncStep>,
}

pub fn steady_trunc_main(cfgfile: &str) -> Result<SteadyTruncResult, String> {
    let p = crate::util::inp_toml(cfgfile)?;
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
    let (u, mu) = (tget_f(&p, "model", "U")?, tget_f(&p, "model", "mu")?);
    let ipt = tget_i(&p, "solver", "ipt_flag")?;
    let h = tget_f(&p, "grid", "h")? * factor as f64;
    let dir = crate::util::ness_outdir(&p, cfgfile)?;
    let prefix = tget_s(&p, "output", "prefix")?;
    if !((1..=5).contains(&k) && tc >= 2 * k + 2 && tmax >= 1 && factor >= 1 && ncorr >= 0) {
        return Err(format!("{cfgfile}: invalid [trunc] parameters"));
    }

    // The steady state (tutor/hubbard_bethe_steady)
    let ness = crate::steady::steady_main(cfgfile)?;

    let mut g = initialize_from_ness(&ness_result_g(&ness), tc, factor);
    let mut sigma = initialize_from_ness(&ness_result_s(&ness), tc, factor);
    let mut gw = initialize_from_ness(&ness_result_w(&ness), tc, factor);
    let mut kk = initialize_from_ness(&ness_result_g(&ness), tc, factor);
    for delti in 0..=tc {
        let mut krow = copy_timestep(&kk, delti);
        let srow = copy_timestep(&sigma, delti);
        incr_step(&mut krow, &srow, Complex::ONE);
        kk.ret.mrow_mut(delti).copy_from_slice(krow.ret.raw());
        kk.less.mrow_mut(delti).copy_from_slice(krow.less.raw());
    }
    // an initialized zero bosonic window (blueprint init_from_steady! of an
    // empty steady-state object)
    let mut w = MovingGreen::new(tc, 1, Sign::Bose);
    {
        use kadanoffbaym::types::{SteadyGreen, SteadyGrid};
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
    // the constant functions are aligned with the NESS windows (KB step 1)
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
        extrapolate_timestep_moving_ws(&mut g, &integ, &mut sws);
        let mut err = 0.0;
        for _ in 0..=ncorr {
            let gtmp = copy_timestep(&g, 0);
            if ipt != 0 {
                dyson_timestep_moving_ws(n, &mut gw, mu, &eloc, &g, h, &integ, &mut mws);
            } else {
                g.ret.mrow_mut(0).copy_from_slice(gw.ret.mrow(0));
                g.less.mrow_mut(0).copy_from_slice(gw.less.mrow(0));
            }
            crate::trunc_model::update_selfenergy_public(&mut sigma, &gw, &mut w, &uf);
            crate::trunc_model::update_kernel_public(&mut kk, &sigma, &g);
            dyson_timestep_moving_ws(n, &mut g, mu, &eloc, &kk, h, &integ, &mut mws);
            err = distance_moving(&copy_timestep(&g, 0), &gtmp);
        }
        steps.push(TruncStep {
            rel,
            tstp: n + 1,
            density: moving_density(&g, 0),
            err,
        });
    }

    use std::io::Write;
    let mut f = std::fs::File::create(format!("{dir}/{prefix}_trunc_density.dat"))
        .map_err(|e| e.to_string())?;
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
        "density(NESS) = {:.10}, density(t_max) = {:.10}",
        density0,
        steps.last().map(|s| s.density).unwrap_or(density0)
    );
    Ok(SteadyTruncResult { density0, steps })
}

// the final Green's functions are carried by the steady result
fn ness_result_g(r: &crate::steady::SteadyResult) -> kadanoffbaym::types::SteadyGreen {
    r.g.clone()
}
fn ness_result_s(r: &crate::steady::SteadyResult) -> kadanoffbaym::types::SteadyGreen {
    r.s.clone()
}
fn ness_result_w(r: &crate::steady::SteadyResult) -> kadanoffbaym::types::SteadyGreen {
    r.gweiss.clone()
}
