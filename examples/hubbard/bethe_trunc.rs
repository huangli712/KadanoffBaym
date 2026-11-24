// Project : KadanoffBaym
// File    : examples/hubbard/bethe_trunc.rs
// Purpose : model functions of the truncated moving-window propagation on the Bethe lattice.
//

use std::fs::{create_dir_all, write};
use std::iter::once;

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::dyson::{DysonWorkspace, MovingDysonWorkspace};
use kadanoffbaym::linalg::Adjoint;
use kadanoffbaym::math::{BetheDos, MovingStageWorkspace};
use kadanoffbaym::types::{MovingGreen, MovingTimeFunction};
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{bubble1, bubble1_moving_herm, bubble2, bubble2_moving_herm};
use kadanoffbaym::dyson::{dyson, dyson_start, dyson_timestep_moving, dyson_timestep_ws};
use kadanoffbaym::equilibrium::{from_dos_mat, from_mat};
use kadanoffbaym::io::{
    read_moving_green, read_moving_time_function, write_moving_green, write_moving_time_function,
};
use kadanoffbaym::math::{extrapolate_timestep, extrapolate_timestep_moving};
use kadanoffbaym::observables::kinetic_energy_trunc;
use kadanoffbaym::variants::{
    copy_timestep, distance_moving, forward, incr_step, init_from_function, init_from_green_herm,
    lmul_step, rmul_step, smul_step,
};

use super::util::TomlValue;

use super::util::{
    add_green_step, get_f64, get_i64, get_str, load_toml_config, matsubara_step, ness_outdir,
    parse_toml_file,
};

use super::util::prelude::*;

/// The parameters of the truncated examples .
pub(crate) struct TruncatedParams {
    pub(crate) nt: i64,
    pub(crate) ntau: i64,
    pub(crate) h: f64,
    pub(crate) beta: f64,
    pub(crate) tc: i64,
    pub(crate) tmax: i64,
    pub(crate) u: f64,
    pub(crate) epsk: Vec<f64>,
    pub(crate) solve_order: i64,
    pub(crate) bootstrap_max_iter: i64,
    pub(crate) bootstrap_max_err: f64,
    pub(crate) corrector_steps: i64,
    pub(crate) dir: String,
    pub(crate) prefix: String,
}

/// The full start result.
pub(crate) struct BetheStart {
    pub(crate) g: ContourGreen,
    pub(crate) sigma: ContourGreen,
    pub(crate) gk: Vec<ContourGreen>,
    pub(crate) u: TimeFunction,
    pub(crate) eloc: TimeFunction,
    pub(crate) epsk: Vec<TimeFunction>,
}

/// The moving state of the truncated propagation.
pub(crate) struct MovingState {
    pub(crate) g: MovingGreen,
    pub(crate) sigma: MovingGreen,
    pub(crate) k: MovingGreen,
    pub(crate) w: MovingGreen,
    pub(crate) gk: Vec<MovingGreen>,
    pub(crate) u: MovingTimeFunction,
    pub(crate) eloc: MovingTimeFunction,
    pub(crate) epsk: Vec<MovingTimeFunction>,
}

/// One per-step record of the propagation.
pub(crate) struct StepRecord {
    pub(crate) tstp: i64,
    pub(crate) density: f64,
    pub(crate) densk: Vec<f64>,
    pub(crate) ekin: f64,
    pub(crate) err: f64,
}

/// Read the parameters; `ntau` counts the imaginary-time POINTS (KB
/// convention) and is converted to the interval count here.
pub(crate) fn load_params(cfgfile: &str) -> Result<TruncatedParams, String> {
    let p = load_toml_config(cfgfile)?;
    let q = TruncatedParams {
        nt: get_i64(&p, "contour", "nt")?,
        ntau: get_i64(&p, "contour", "ntau")? - 1,
        h: get_f64(&p, "contour", "h")?,
        beta: get_f64(&p, "contour", "beta")?,
        tc: get_i64(&p, "contour", "tc")?,
        tmax: get_i64(&p, "contour", "tmax")?,
        u: get_f64(&p, "model", "U")?,
        epsk: {
            let m = p["model"].as_table().unwrap();
            match m.get("epsk") {
                Some(TomlValue::Array(a)) => a
                    .iter()
                    .map(|v| v.as_f64().ok_or_else(|| "epsk must be numbers".to_string()))
                    .collect::<Result<Vec<f64>, String>>()?,
                _ => return Err(format!("{cfgfile}: [model] epsk must be an array")),
            }
        },
        solve_order: get_i64(&p, "solver", "SolveOrder")?,
        bootstrap_max_iter: get_i64(&p, "solver", "BootstrapMaxIter")?,
        bootstrap_max_err: get_f64(&p, "solver", "BootstrapMaxErr")?,
        corrector_steps: get_i64(&p, "solver", "CorrectorSteps")?,
        dir: String::new(),
        prefix: get_str(&p, "output", "prefix")?,
    };
    if !(1..=5).contains(&q.solve_order) {
        return Err(format!("{cfgfile}: SolveOrder must be in 1..=5"));
    }
    if q.tc < 2 * q.solve_order + 2 {
        return Err(format!("{cfgfile}: tc must be >= 2 SolveOrder + 2"));
    }
    if q.nt < q.tc || q.tmax < q.tc {
        return Err(format!("{cfgfile}: nt and tmax must be >= tc"));
    }
    if !(q.h > 0.0 && q.beta > 0.0 && q.ntau >= 2) {
        return Err(format!("{cfgfile}: invalid contour parameters"));
    }
    let dir = ness_outdir(&p, cfgfile)?;
    Ok(TruncatedParams { dir, ..q })
}

/// Second-order self-energy at the KB time step `tstp` of the full
/// contour functions.
fn update_bethe_selfenergy(
    tstp: i64,
    sigma: &mut ContourGreen,
    g: &ContourGreen,
    w: &mut ContourGreen,
    uf: &TimeFunction,
) {
    bubble1(matsubara_step(tstp), w, 0, 0, g, g, 0, 0, g, g, 0, 0);
    w.rmul_cf_at(matsubara_step(tstp), uf, Adjoint::No);
    w.lmul_cf_at(matsubara_step(tstp), uf, Adjoint::No);
    bubble2(matsubara_step(tstp), sigma, 0, 0, g, g, 0, 0, w, w, 0, 0);
    sigma.smul_at(matsubara_step(tstp), Complex::new(-1.0, 0.0));
}

/// Kernel `K = Sigma + G` at the KB time step `tstp` (full).
fn update_bethe_kernel(tstp: i64, k: &mut ContourGreen, sigma: &ContourGreen, g: &ContourGreen) {
    k.memcpy_from_at(matsubara_step(tstp), sigma);
    add_green_step(k, g, matsubara_step(tstp), Complex::ONE);
}

/// Full Kadanoff-Baym start : the
/// bootstrapping of the first SolveOrder + 1 steps and the time stepping
/// up to the KB time step nt + 1, plus the lattice Green's functions.
pub(crate) fn solve_start(p: &TruncatedParams) -> Result<BetheStart, String> {
    let (nt, ntau, h, beta) = (p.nt, p.ntau, p.h, p.beta);
    let (k, mu) = (p.solve_order, 0.0);
    let cc = Contour::new(nt, ntau, 1, 1, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let (mut g, mut sigma, mut kk) = (fermion(), fermion(), fermion());
    let mut w = ContourGreen::new(&cc, Sign::Bose, Complex::ZERO);
    // U = 0 on the Matsubara branch (quench at t = 0)
    let mut u = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    u.block_mut(MATSUBARA)[0] = Complex::ZERO;
    for n in 0..=nt {
        u.block_mut(n)[0] = Complex::new(p.u, 0.0);
    }
    let eloc = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);

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
    for q in sigma.mat.raw_mut() {
        *q = Complex::ZERO;
    }
    update_bethe_kernel(0, &mut kk, &sigma, &g);

    // Bootstrapping, the KB time steps 1..k+1
    let nb = k + 1;
    let mut gtemp = fermion();
    from_mat(&mut g);
    let mut converged = false;
    for iter in 0..=p.bootstrap_max_iter {
        gtemp.memcpy_from_at(nb - 1, &g);
        for n in 1..=nb {
            update_bethe_selfenergy(n, &mut sigma, &g, &mut w, &u);
            update_bethe_kernel(n, &mut kk, &sigma, &g);
        }
        dyson_start(&mut g, mu, &eloc, &kk, beta, h, k);
        let err = g.distance_at(&gtemp, nb - 1);
        if err < p.bootstrap_max_err && iter > 3 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("bootstrapping not converged".to_string());
    }

    // Time stepping
    for tstp in (nb + 1)..=(nt + 1) {
        gtemp.memcpy_from_at(tstp - 1, &g);
        extrapolate_timestep(tstp - 2, &mut g, &Integrator::new(k));
        let d_integ = Integrator::new(k);
        let mut dws = DysonWorkspace::new(k, g.size1, g.ntau);
        for _ in 0..=p.corrector_steps {
            gtemp.memcpy_from_at(tstp - 1, &g);
            update_bethe_selfenergy(tstp, &mut sigma, &g, &mut w, &u);
            update_bethe_kernel(tstp, &mut kk, &sigma, &g);
            // the kernel `kk` is rebuilt per iteration
            dws.reset();
            dyson_timestep_ws(
                tstp - 1,
                &mut g,
                mu,
                &eloc,
                &kk,
                beta,
                h,
                &d_integ,
                &mut dws,
            );
        }
    }

    // Lattice Green's functions
    let mut gk = Vec::new();
    let mut epsk = Vec::new();
    for eps in &p.epsk {
        let mut e = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        for n in MATSUBARA..=nt {
            e.block_mut(n)[0] = Complex::new(*eps, 0.0);
        }
        let mut ggk = fermion();
        dyson(
            &mut ggk,
            mu,
            &e,
            &sigma,
            beta,
            h,
            k,
            MatsubaraMethod::Fixpoint,
            true,
        );
        gk.push(ggk);
        epsk.push(e);
    }

    Ok(BetheStart {
        g,
        sigma,
        gk,
        u,
        eloc,
        epsk,
    })
}

/// Moving copy of the full function at the KB time step `n0`.
fn moving_window(x: &ContourGreen, tc: i64, n0: i64) -> MovingGreen {
    let mut m = MovingGreen::new(tc, x.size1, x.sign);
    init_from_green_herm(&mut m, x, n0 - 1);
    m
}

/// Moving windows of the full start at the KB time step `tc + 1`
/// ; the kernel is set to
/// `K = G + Sigma` on all rows.
pub(crate) fn initialize_moving_state(start: &BetheStart, tc: i64) -> MovingState {
    let n0 = tc + 1;
    let g = moving_window(&start.g, tc, n0);
    let mut sigma = moving_window(&start.sigma, tc, n0);
    let mut k = moving_window(&start.g, tc, n0);
    for delti in 0..=tc {
        let mut krow = copy_timestep(&k, delti);
        let srow = copy_timestep(&sigma, delti);
        incr_step(&mut krow, &srow, Complex::ONE);
        k.ret.mrow_mut(delti).copy_from_slice(krow.ret.raw());
        k.less.mrow_mut(delti).copy_from_slice(krow.less.raw());
    }
    let _ = &mut sigma;
    // the bosonic work function starts as an initialized zero window
    let ccw = Contour::new(start.g.nt as i64, start.g.ntau as i64, 1, 1, 1.0, 1.0);
    let wzero = ContourGreen::new(&ccw, Sign::Bose, Complex::ZERO);
    let w = moving_window(&wzero, tc, n0);
    let gk = start.gk.iter().map(|g| moving_window(g, tc, n0)).collect();
    let u = moving_window_function(&start.u, tc, n0);
    let eloc = moving_window_function(&start.eloc, tc, n0);
    let epsk = start
        .epsk
        .iter()
        .map(|e| moving_window_function(e, tc, n0))
        .collect();
    MovingState {
        g,
        sigma,
        k,
        w,
        gk,
        u,
        eloc,
        epsk,
    }
}

fn moving_window_function(f: &TimeFunction, tc: i64, n0: i64) -> MovingTimeFunction {
    let mut m = MovingTimeFunction::new(tc, 1);
    init_from_function(&mut m, f, n0 - 1);
    m
}

/// Density `i xi G^<(t,t)` of the row `delti` (one spin,).
pub(crate) fn moving_density(g: &MovingGreen, delti: i64) -> f64 {
    let row = copy_timestep(g, delti);
    let xi = if g.sign == Sign::Fermi { -1.0 } else { 1.0 };
    (Complex::new(0.0, xi) * row.less.raw()[0]).re
}

/// Write the moving state as plain text (`KB_MOVING_V1`) into the
/// directory `dir`, together with `meta.toml`.
pub(crate) fn write_checkpoint(dir: &str, state: &MovingState, h: f64) -> Result<(), String> {
    create_dir_all(dir).map_err(|e| e.to_string())?;
    write_moving_green(&format!("{dir}/G.dat"), &state.g).map_err(|e| e.to_string())?;
    write_moving_green(&format!("{dir}/S.dat"), &state.sigma).map_err(|e| e.to_string())?;
    write_moving_green(&format!("{dir}/K.dat"), &state.k).map_err(|e| e.to_string())?;
    write_moving_green(&format!("{dir}/W.dat"), &state.w).map_err(|e| e.to_string())?;
    for (i, g) in state.gk.iter().enumerate() {
        write_moving_green(&format!("{dir}/Gk_{}.dat", i + 1), g).map_err(|e| e.to_string())?;
    }
    write_moving_time_function(&format!("{dir}/U.dat"), &state.u).map_err(|e| e.to_string())?;
    write_moving_time_function(&format!("{dir}/eloc.dat"), &state.eloc)
        .map_err(|e| e.to_string())?;
    for (i, e) in state.epsk.iter().enumerate() {
        write_moving_time_function(&format!("{dir}/epsk_{}.dat", i + 1), e)
            .map_err(|e| e.to_string())?;
    }
    write(
        format!("{dir}/meta.toml"),
        format!(
            "tc = {}\ntstp = {}\nh = {}\nnk = {}\n",
            state.g.tc,
            state.g.tstp,
            h,
            state.gk.len()
        ),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Read the moving state written by `write_checkpoint`.
pub(crate) fn read_checkpoint(dir: &str, h: f64) -> Result<MovingState, String> {
    let meta =
        parse_toml_file(&format!("{dir}/meta.toml")).map_err(|e| format!("checkpoint: {e}"))?;
    let root = |key: &str| -> Result<f64, String> {
        meta.get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("checkpoint: missing {key}"))
    };
    let tc = root("tc")? as i64;
    let tstp = root("tstp")? as i64;
    let nk = root("nk")? as i64;
    let hmeta = root("h")?;
    if hmeta != h {
        return Err("checkpoint: h differs".to_string());
    }
    let read_g = |path: &str, sign: Sign| -> Result<MovingGreen, String> {
        let mut g = MovingGreen::new(tc, 1, sign);
        read_moving_green(path, &mut g).map_err(|e| e.to_string())?;
        Ok(g)
    };
    let g = read_g(&format!("{dir}/G.dat"), Sign::Fermi)?;
    let sigma = read_g(&format!("{dir}/S.dat"), Sign::Fermi)?;
    let k = read_g(&format!("{dir}/K.dat"), Sign::Fermi)?;
    let w = read_g(&format!("{dir}/W.dat"), Sign::Bose)?;
    let mut gk = Vec::new();
    for i in 1..=nk {
        gk.push(read_g(&format!("{dir}/Gk_{i}.dat"), Sign::Fermi)?);
    }
    let read_cf = |path: &str| -> Result<MovingTimeFunction, String> {
        let mut f = MovingTimeFunction::new(tc, 1);
        read_moving_time_function(path, &mut f).map_err(|e| e.to_string())?;
        Ok(f)
    };
    let u = read_cf(&format!("{dir}/U.dat"))?;
    let eloc = read_cf(&format!("{dir}/eloc.dat"))?;
    let mut epsk = Vec::new();
    for i in 1..=nk {
        epsk.push(read_cf(&format!("{dir}/epsk_{i}.dat"))?);
    }
    let state = MovingState {
        g,
        sigma,
        k,
        w,
        gk,
        u,
        eloc,
        epsk,
    };
    for x in once(&state.g)
        .chain(once(&state.sigma))
        .chain(once(&state.k))
        .chain(once(&state.w))
        .chain(state.gk.iter())
    {
        if !(x.is_initialized() && x.tstp == tstp) {
            return Err("checkpoint: inconsistent state".to_string());
        }
    }
    Ok(state)
}

/// Forward the constant single-time function and set the leading slot.
fn forward_constant(f: &mut MovingTimeFunction) {
    let v = f.fslot(1)[0];
    f.forward();
    f.fslot_mut(0)[0] = v;
}

/// Truncated propagation  up to the KB time step
/// `tmax + 1`. Every step: forward of all windows, the predictor
/// (extrapolation), CorrectorSteps + 1 corrector iterations, the lattice
/// moving Dyson equations, and the observables.
pub(crate) fn propagate_bethe_moving(
    state: &mut MovingState,
    p: &TruncatedParams,
) -> Vec<StepRecord> {
    let (k, h, mu) = (p.solve_order, p.h, 0.0);
    let integ = Integrator::new(k);
    let mut mws = MovingDysonWorkspace::new(k, state.g.ret.size1);
    let mut sws = MovingStageWorkspace::new(k, state.g.tc, state.g.ret.size1);
    let mut out = Vec::new();
    let n_start = state.g.tstp + 1;
    for n in n_start..=(p.tmax + 1) {
        // (my steps; the KB time steps tstp + 1..tmax + 1)
        forward(&mut state.sigma);
        forward(&mut state.k);
        forward(&mut state.g);
        forward(&mut state.w);
        forward_constant(&mut state.u);
        forward_constant(&mut state.eloc);
        extrapolate_timestep_moving(&mut state.g, k, &mut sws);
        let mut err = 0.0;
        for _ in 0..=p.corrector_steps {
            let gtmp = copy_timestep(&state.g, 0);
            update_bethe_selfenergy_moving(&mut state.sigma, &state.g, &mut state.w, &state.u);
            update_bethe_kernel_moving(&mut state.k, &state.sigma, &state.g);
            dyson_timestep_moving(
                n,
                &mut state.g,
                mu,
                &state.eloc,
                &state.k,
                h,
                &integ,
                &mut mws,
            );
            err = distance_moving(&copy_timestep(&state.g, 0), &gtmp);
        }
        let mut densk = Vec::new();
        for i in 0..state.gk.len() {
            forward(&mut state.gk[i]);
            forward_constant(&mut state.epsk[i]);
            dyson_timestep_moving(
                n,
                &mut state.gk[i],
                mu,
                &state.epsk[i],
                &state.sigma,
                h,
                &integ,
                &mut mws,
            );
            densk.push(moving_density(&state.gk[i], 0));
        }
        let ekin = kinetic_energy_trunc(n, &state.g, &state.g, &integ, h);
        out.push(StepRecord {
            tstp: n + 1,
            density: moving_density(&state.g, 0),
            densk,
            ekin,
            err,
        });
    }
    out
}

/// Second-order self-energy at the leading row of the moving functions.
fn update_bethe_selfenergy_moving(
    sigma: &mut MovingGreen,
    g: &MovingGreen,
    w: &mut MovingGreen,
    uf: &MovingTimeFunction,
) {
    let n = g.tstp;
    bubble1_moving_herm(n, w, 0, 0, g, 0, 0, g, 0, 0);
    let mut wrow = copy_timestep(w, 0);
    lmul_step(&mut wrow, uf, false);
    rmul_step(&mut wrow, uf, false);
    w.ret.mrow_mut(0).copy_from_slice(wrow.ret.raw());
    w.less.mrow_mut(0).copy_from_slice(wrow.less.raw());
    bubble2_moving_herm(n, sigma, 0, 0, g, 0, 0, w, 0, 0);
    let mut srow = copy_timestep(sigma, 0);
    smul_step(&mut srow, Complex::new(-1.0, 0.0));
    sigma.ret.mrow_mut(0).copy_from_slice(srow.ret.raw());
    sigma.less.mrow_mut(0).copy_from_slice(srow.less.raw());
}

/// Wrapper used by the steady truncated example.
pub(crate) fn update_selfenergy(
    sigma: &mut MovingGreen,
    g: &MovingGreen,
    w: &mut MovingGreen,
    uf: &MovingTimeFunction,
) {
    update_bethe_selfenergy_moving(sigma, g, w, uf)
}

/// Wrapper used by the steady truncated example.
pub(crate) fn update_kernel(k: &mut MovingGreen, sigma: &MovingGreen, g: &MovingGreen) {
    update_bethe_kernel_moving(k, sigma, g)
}

/// Kernel `K = Sigma + G` at the leading row (moving).
fn update_bethe_kernel_moving(k: &mut MovingGreen, sigma: &MovingGreen, g: &MovingGreen) {
    // K <- Sigma (overwrite), then K <- K + G
    let mut krow = copy_timestep(sigma, 0);
    let grow = copy_timestep(g, 0);
    incr_step(&mut krow, &grow, Complex::ONE);
    k.ret.mrow_mut(0).copy_from_slice(krow.ret.raw());
    k.less.mrow_mut(0).copy_from_slice(krow.less.raw());
}
