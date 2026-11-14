//
// Model functions of the truncated (moving window) propagation of the
// Hubbard model on the Bethe lattice (the Rust port of
// julia/tutor/hubbard_bethe/hubbard_bethe_trunc_model.jl, from NESSi
// examples/programs/trunc_bethe_start.cpp and trunc_bethe.cpp; no HDF5).
//
//   W(t,t') = U(t) [i G(t,t') G(t',t)] U(t'),  Sigma = -i G W,
//   K = Sigma + G,  G = [i d/dt + mu - K]^-1.
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::toml::TomlValue;
use crate::ness_util::{tget_f, tget_i, tget_s};

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1, bubble2, bubble1_moving_herm, bubble2_moving_herm};
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::math::dos::BetheDos;
use kadanoffbaym::moving::algebra::{
    distance_moving, extrapolate_timestep_moving, incr_step, lmul_step, rmul_step, smul_step,
};
use kadanoffbaym::moving::window::{copy_timestep, forward, init_from_function, init_from_g_herm};
use kadanoffbaym::observables::kinetic_energy_trunc;
use kadanoffbaym::solvers::dyson::dyson;
use kadanoffbaym::solvers::dyson_mat::MatsubaraMethod;
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::solvers::dyson_moving::dyson_timestep_moving;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::equilibrium::from_mat::from_mat;
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::types::moving::{CfMoving, MovingGreen};
use kadanoffbaym::weights::Integrator;

fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// Copy a single step of `src` into `dst` (blueprint `memcpy!`).
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

/// The parameters of the truncated examples (blueprint `read_params`).
pub struct TruncParams {
    pub nt: i64,
    pub ntau: i64,
    pub h: f64,
    pub beta: f64,
    pub tc: i64,
    pub tmax: i64,
    pub u: f64,
    pub epsk: Vec<f64>,
    pub solve_order: i64,
    pub bootstrap_max_iter: i64,
    pub bootstrap_max_err: f64,
    pub corrector_steps: i64,
    pub dir: String,
    pub prefix: String,
}

/// Read the parameters; `ntau` counts the imaginary-time POINTS (KB
/// convention) and is converted to the interval count here.
pub fn read_trunc_params(cfgfile: &str) -> Result<TruncParams, String> {
    let p = crate::ness_util::inp_toml(cfgfile)?;
    let q = TruncParams {
        nt: tget_i(&p, "contour", "nt")?,
        ntau: tget_i(&p, "contour", "ntau")? - 1,
        h: tget_f(&p, "contour", "h")?,
        beta: tget_f(&p, "contour", "beta")?,
        tc: tget_i(&p, "contour", "tc")?,
        tmax: tget_i(&p, "contour", "tmax")?,
        u: tget_f(&p, "model", "U")?,
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
        solve_order: tget_i(&p, "solver", "SolveOrder")?,
        bootstrap_max_iter: tget_i(&p, "solver", "BootstrapMaxIter")?,
        bootstrap_max_err: tget_f(&p, "solver", "BootstrapMaxErr")?,
        corrector_steps: tget_i(&p, "solver", "CorrectorSteps")?,
        dir: String::new(),
        prefix: tget_s(&p, "output", "prefix")?,
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
    let dir = crate::ness_util::ness_outdir(&p, cfgfile)?;
    Ok(TruncParams { dir, ..q })
}

/// Second-order self-energy at the KB time step `tstp` of the full
/// contour functions.
fn update_bethe_selfenergy(tstp: i64, sigma: &mut ContourGreen, g: &ContourGreen, w: &mut ContourGreen, uf: &TimeFunction) {
    bubble1(mt(tstp), w, 0, 0, g, g, 0, 0, g, g, 0, 0);
    w.rmul_cf_at(mt(tstp), uf, kadanoffbaym::basics::Mode::Normal);
    w.lmul_cf_at(mt(tstp), uf, kadanoffbaym::basics::Mode::Normal);
    bubble2(mt(tstp), sigma, 0, 0, g, g, 0, 0, w, w, 0, 0);
    sigma.smul_at(mt(tstp), Complex::new(-1.0, 0.0));
}

/// Kernel `K = Sigma + G` at the KB time step `tstp` (full).
fn update_bethe_kernel(tstp: i64, k: &mut ContourGreen, sigma: &ContourGreen, g: &ContourGreen) {
    copy_step(sigma, k, mt(tstp));
    kadanoffbaym_add_step(k, g, mt(tstp), Complex::ONE);
}

/// `dst += alpha * src` at a single step.
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

/// The full start result.
pub struct BetheStart {
    pub g: ContourGreen,
    pub sigma: ContourGreen,
    pub k: ContourGreen,
    pub gk: Vec<ContourGreen>,
    pub u: TimeFunction,
    pub eloc: TimeFunction,
    pub epsk: Vec<TimeFunction>,
    pub errors: Vec<(i64, i64, i64, f64)>,
}

/// Full Kadanoff-Baym start (NESSi trunc_bethe_start.cpp): the
/// bootstrapping of the first SolveOrder + 1 steps and the time stepping
/// up to the KB time step nt + 1, plus the lattice Green's functions.
pub fn solve_bethe_start(p: &TruncParams) -> Result<BetheStart, String> {
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
    let mut errlog: Vec<(i64, i64, i64, f64)> = Vec::new();

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
        copy_step(&g, &mut gtemp, nb - 1);
        for n in 1..=nb {
            update_bethe_selfenergy(n, &mut sigma, &g, &mut w, &u);
            update_bethe_kernel(n, &mut kk, &sigma, &g);
        }
        dyson_start(&mut g, mu, &eloc, &kk, beta, h, k);
        let err = g.distance_at(&gtemp, nb - 1);
        errlog.push((1, nb, iter, err));
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
        copy_step(&g, &mut gtemp, tstp - 1);
        kadanoffbaym::math::extrap::extrapolate_timestep(tstp - 2, &mut g, &Integrator::new(k));
        for iter in 0..=p.corrector_steps {
            copy_step(&g, &mut gtemp, tstp - 1);
            update_bethe_selfenergy(tstp, &mut sigma, &g, &mut w, &u);
            update_bethe_kernel(tstp, &mut kk, &sigma, &g);
            dyson_timestep(tstp - 1, &mut g, mu, &eloc, &kk, beta, h, k);
            errlog.push((2, tstp, iter, g.distance_at(&gtemp, tstp - 1)));
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
        dyson(&mut ggk, mu, &e, &sigma, beta, h, k, MatsubaraMethod::Fixpoint, true);
        gk.push(ggk);
        epsk.push(e);
    }

    Ok(BetheStart {
        g,
        sigma,
        k: kk,
        gk,
        u,
        eloc,
        epsk,
        errors: errlog,
    })
}

/// The moving state of the truncated propagation.
pub struct MovingState {
    pub g: MovingGreen,
    pub sigma: MovingGreen,
    pub k: MovingGreen,
    pub w: MovingGreen,
    pub gk: Vec<MovingGreen>,
    pub u: CfMoving,
    pub eloc: CfMoving,
    pub epsk: Vec<CfMoving>,
}

/// Moving copy of the full function at the KB time step `n0`.
fn moving_copy(x: &ContourGreen, tc: i64, n0: i64) -> MovingGreen {
    let mut m = MovingGreen::new(tc, x.size1, x.sign);
    init_from_g_herm(&mut m, x, n0 - 1);
    m
}

/// Moving windows of the full start at the KB time step `tc + 1`
/// (blueprint `initialize_moving_state`); the kernel is set to
/// `K = G + Sigma` on all rows.
pub fn initialize_moving_state(start: &BetheStart, tc: i64) -> MovingState {
    let n0 = tc + 1;
    let g = moving_copy(&start.g, tc, n0);
    let mut sigma = moving_copy(&start.sigma, tc, n0);
    let mut k = moving_copy(&start.g, tc, n0);
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
    let w = moving_copy(&wzero, tc, n0);
    let gk = start.gk.iter().map(|g| moving_copy(g, tc, n0)).collect();
    let u = moving_copy_cf(&start.u, tc, n0);
    let eloc = moving_copy_cf(&start.eloc, tc, n0);
    let epsk = start.epsk.iter().map(|e| moving_copy_cf(e, tc, n0)).collect();
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

fn moving_copy_cf(f: &TimeFunction, tc: i64, n0: i64) -> CfMoving {
    let mut m = CfMoving::new(tc, 1);
    init_from_function(&mut m, f, n0 - 1);
    m
}

/// Density `i xi G^<(t,t)` of the row `delti` (one spin, as in NESSi).
pub fn moving_density(g: &MovingGreen, delti: i64) -> f64 {
    let row = copy_timestep(g, delti);
    let xi = if g.sign == Sign::Fermi { -1.0 } else { 1.0 };
    (Complex::new(0.0, xi) * row.less.raw()[0]).re
}

/// Write the moving state as plain text (`KB_MOVING_V1`) into the
/// directory `dir`, together with `meta.toml`.
pub fn write_checkpoint(dir: &str, state: &MovingState, h: f64) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    kadanoffbaym::io::write_moving_green(&format!("{dir}/G.dat"), &state.g)
        .map_err(|e| e.to_string())?;
    kadanoffbaym::io::write_moving_green(&format!("{dir}/S.dat"), &state.sigma)
        .map_err(|e| e.to_string())?;
    kadanoffbaym::io::write_moving_green(&format!("{dir}/K.dat"), &state.k)
        .map_err(|e| e.to_string())?;
    kadanoffbaym::io::write_moving_green(&format!("{dir}/W.dat"), &state.w)
        .map_err(|e| e.to_string())?;
    for (i, g) in state.gk.iter().enumerate() {
        kadanoffbaym::io::write_moving_green(&format!("{dir}/Gk_{}.dat", i + 1), g)
            .map_err(|e| e.to_string())?;
    }
    kadanoffbaym::io::write_cf_moving(&format!("{dir}/U.dat"), &state.u)
        .map_err(|e| e.to_string())?;
    kadanoffbaym::io::write_cf_moving(&format!("{dir}/eloc.dat"), &state.eloc)
        .map_err(|e| e.to_string())?;
    for (i, e) in state.epsk.iter().enumerate() {
        kadanoffbaym::io::write_cf_moving(&format!("{dir}/epsk_{}.dat", i + 1), e)
            .map_err(|e| e.to_string())?;
    }
    std::fs::write(
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
pub fn read_checkpoint(dir: &str, h: f64) -> Result<MovingState, String> {
    let meta = crate::common::toml::parse_toml_file(&format!("{dir}/meta.toml"))
        .map_err(|e| format!("checkpoint: {e}"))?;
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
        kadanoffbaym::io::read_moving_green(path, &mut g).map_err(|e| e.to_string())?;
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
    let read_cf = |path: &str| -> Result<CfMoving, String> {
        let mut f = CfMoving::new(tc, 1);
        kadanoffbaym::io::read_cf_moving(path, &mut f).map_err(|e| e.to_string())?;
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
    for x in std::iter::once(&state.g)
        .chain(std::iter::once(&state.sigma))
        .chain(std::iter::once(&state.k))
        .chain(std::iter::once(&state.w))
        .chain(state.gk.iter())
    {
        if !(x.initialized && x.tstp == tstp) {
            return Err("checkpoint: inconsistent state".to_string());
        }
    }
    Ok(state)
}

/// Forward the constant single-time function and set the leading slot.
fn forward_const(f: &mut CfMoving) {
    let v = f.fslot(1)[0];
    f.forward();
    f.fslot_mut(0)[0] = v;
}

/// One per-step record of the propagation.
pub struct StepRecord {
    pub tstp: i64,
    pub density: f64,
    pub densk: Vec<f64>,
    pub ekin: f64,
    pub err: f64,
}

/// Truncated propagation (NESSi trunc_bethe.cpp) up to the KB time step
/// `tmax + 1`. Every step: forward of all windows, the predictor
/// (extrapolation), CorrectorSteps + 1 corrector iterations, the lattice
/// moving Dyson equations, and the observables.
pub fn propagate_bethe_moving(state: &mut MovingState, p: &TruncParams) -> Vec<StepRecord> {
    let (k, h, mu) = (p.solve_order, p.h, 0.0);
    let integ = Integrator::new(k);
    let mut out = Vec::new();
    let n_start = state.g.tstp + 1;
    for n in n_start..=(p.tmax + 1) {
        // (my steps; the KB time steps tstp + 1..tmax + 1)
        forward(&mut state.sigma);
        forward(&mut state.k);
        forward(&mut state.g);
        forward(&mut state.w);
        forward_const(&mut state.u);
        forward_const(&mut state.eloc);
        extrapolate_timestep_moving(&mut state.g, &integ);
        let mut err = 0.0;
        for _ in 0..=p.corrector_steps {
            let gtmp = copy_timestep(&state.g, 0);
            update_bethe_selfenergy_moving(&mut state.sigma, &state.g, &mut state.w, &state.u);
            update_bethe_kernel_moving(&mut state.k, &state.sigma, &state.g);
            dyson_timestep_moving(n, &mut state.g, mu, &state.eloc, &state.k, h, k);
            err = distance_moving(&copy_timestep(&state.g, 0), &gtmp);
        }
        let mut densk = Vec::new();
        for i in 0..state.gk.len() {
            forward(&mut state.gk[i]);
            forward_const(&mut state.epsk[i]);
            dyson_timestep_moving(
                n,
                &mut state.gk[i],
                mu,
                &state.epsk[i],
                &state.sigma,
                h,
                k,
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
fn update_bethe_selfenergy_moving(sigma: &mut MovingGreen, g: &MovingGreen, w: &mut MovingGreen, uf: &CfMoving) {
    let n = g.tstp;
    bubble1_moving_herm(n, w, 0, 0, g, 0, 0, g, 0, 0);
    let mut wrow = copy_timestep(w, 0);
    lmul_step(uf, &mut wrow, false);
    rmul_step(uf, &mut wrow, false);
    w.ret.mrow_mut(0).copy_from_slice(wrow.ret.raw());
    w.less.mrow_mut(0).copy_from_slice(wrow.less.raw());
    bubble2_moving_herm(n, sigma, 0, 0, g, 0, 0, w, 0, 0);
    let mut srow = copy_timestep(sigma, 0);
    smul_step(&mut srow, Complex::new(-1.0, 0.0));
    sigma.ret.mrow_mut(0).copy_from_slice(srow.ret.raw());
    sigma.less.mrow_mut(0).copy_from_slice(srow.less.raw());
}

/// Public wrapper used by hubbard_bethe_ness2_trunc.
pub fn update_selfenergy_public(sigma: &mut MovingGreen, g: &MovingGreen, w: &mut MovingGreen, uf: &CfMoving) {
    update_bethe_selfenergy_moving(sigma, g, w, uf)
}

/// Public wrapper used by hubbard_bethe_ness2_trunc.
pub fn update_kernel_public(k: &mut MovingGreen, sigma: &MovingGreen, g: &MovingGreen) {
    update_bethe_kernel_moving(k, sigma, g)
}

/// Kernel `K = Sigma + G` at the leading row (moving).
fn update_bethe_kernel_moving(k: &mut MovingGreen, sigma: &MovingGreen, g: &MovingGreen) {
    // K <- Sigma (overwrite), then K <- K + G (blueprint memcpy! + incr!)
    let mut krow = copy_timestep(sigma, 0);
    let grow = copy_timestep(g, 0);
    incr_step(&mut krow, &grow, Complex::ONE);
    k.ret.mrow_mut(0).copy_from_slice(krow.ret.raw());
    k.less.mrow_mut(0).copy_from_slice(krow.less.raw());
}
