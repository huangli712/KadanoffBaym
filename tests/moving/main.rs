// Project : KadanoffBaym
// File    : tests/moving/main.rs
// Purpose : moving-window family: containers, algebra, solvers (dyson, vie2)

use kadanoffbaym::dyson::MovingDysonWorkspace;
use kadanoffbaym::dyson::dyson_timestep_moving_ws;
use kadanoffbaym::types::{MovingGreen, MovingTimeFunction};
use kadanoffbaym::vie2::MovingVie2Workspace;
use kadanoffbaym::vie2::vie2_timestep_moving_ws;
use kadanoffbaym::weights::Integrator;

mod dyson_moving;
mod moving;
mod moving_algebra;
mod moving_solvers;
#[path = "../../util/mod.rs"]
mod util;
mod vie2_moving;

/// One-shot moving VIE2 step: builds a fresh `Integrator` and a fresh
/// workspace for the call.
fn vie2_moving_step(
    tstp: i64,
    g: &mut MovingGreen,
    f: &MovingGreen,
    fcc: &MovingGreen,
    q: &MovingGreen,
    hstep: f64,
    k: i64,
) {
    let integ = Integrator::new(k);
    let mut ws = MovingVie2Workspace::new(k, g.ret.size1);
    vie2_timestep_moving_ws(tstp, g, f, fcc, q, hstep, &integ, &mut ws);
}

/// One-shot moving Dyson step: builds a fresh `Integrator` and a fresh
/// workspace for the call.
#[allow(clippy::too_many_arguments)]
fn dyson_moving_step(
    tstp: i64,
    g: &mut MovingGreen,
    mu: f64,
    h: &MovingTimeFunction,
    sigma: &MovingGreen,
    hstep: f64,
    k: i64,
) {
    let integ = Integrator::new(k);
    let mut ws = MovingDysonWorkspace::new(k, g.ret.size1);
    dyson_timestep_moving_ws(tstp, g, mu, h, sigma, hstep, &integ, &mut ws);
}
