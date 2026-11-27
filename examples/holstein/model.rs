// Project : KadanoffBaym
// File    : examples/holstein/model.rs
// Purpose : shared Holstein kernels (phonon propagator, polarization, self-energies,
//           displacement) and the configuration helpers of the examples.
//

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::convolution::ConvWorkspace;
use kadanoffbaym::linalg::Adjoint;
use kadanoffbaym::vie2::Vie2Workspace;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{bubble1, bubble2, conv_mat, convolution_timestep};
use kadanoffbaym::equilibrium::green_single_pole;
use kadanoffbaym::vie2::{vie2_mat, vie2_start, vie2_timestep, vie2_timestep_ws};

use super::util::ExampleConfig;

use super::util::{
    add_green_step, contour_params, example_outdir, load_tvector, matsubara_step, zero_green_step,
};

use super::util::prelude::*;

/// Phonon propagators and work space.
pub(crate) struct HolsteinPhonon {
    pub(crate) d0: ContourGreen,
    pub(crate) d: ContourGreen,
    pub(crate) pi: ContourGreen,
    pub(crate) d0_pi: ContourGreen,
    pub(crate) pi_d0: ContourGreen,
    pub(crate) pol: ContourGreen,
    pub(crate) ggg: ContourGreen,
    pub(crate) w0: f64,
    pub(crate) beta: f64,
    pub(crate) h: f64,
    pub(crate) k: i64,
    pub(crate) integ: Integrator,
    /// VIE2 scratch of the phonon propagator, one instance for the
    /// whole evolution (the kernel Matsubara blocks stay fixed)
    pub(crate) d_ws: Vie2Workspace,
}

/// State of the two-stage Matsubara iteration of the Holstein examples:
/// the FOURIER method until the error is below the threshold, then the
/// FIXPOINT method until it is below the threshold again.
pub(crate) struct MatsubaraStage {
    pub(crate) fixpoint: bool,
    pub(crate) converged: bool,
}

impl MatsubaraStage {
    pub fn new() -> Self {
        Self {
            fixpoint: false,
            converged: false,
        }
    }

    pub fn method(&self) -> MatsubaraMethod {
        if self.fixpoint {
            MatsubaraMethod::Fixpoint
        } else {
            MatsubaraMethod::Fourier
        }
    }

    /// Update the stage after an iteration with the error `err`; returns
    /// true if the previous iterate must be updated.
    pub fn update(&mut self, err: f64, tol: f64) -> bool {
        if err < tol && self.fixpoint {
            self.converged = true;
            false
        } else if err < tol {
            self.fixpoint = true;
            true
        } else {
            true
        }
    }
}

/// The coupling constant with its drive, as values at the KB time steps and
/// as a time function, and the hopping as a time function.
pub(crate) struct Couplings {
    pub(crate) gv: Vec<Complex>,
    pub(crate) gf: TimeFunction,
    pub(crate) jf: TimeFunction,
}

/// Allocate the phonon propagators on the contour, `D0 = 2 D_XX` (the
/// single-pole propagator scaled by two).
pub(crate) fn new_phonon(
    nt: i64,
    ntau: i64,
    size: usize,
    w0: f64,
    beta: f64,
    h: f64,
    k: i64,
) -> HolsteinPhonon {
    let tmax = nt as f64 * h;
    let c1 = Contour::new(nt, ntau, 1, 1, tmax, beta);
    let cs = Contour::new(nt, ntau, size as i64, size as i64, tmax, beta);
    let boson1 = || ContourGreen::new(&c1, Sign::Bose, Complex::ZERO);
    let mut d0 = boson1();
    green_single_pole(&mut d0, w0, beta, h);
    for n in MATSUBARA..=nt {
        d0.smul_at(n, Complex::new(2.0, 0.0));
    }
    HolsteinPhonon {
        d0,
        d: boson1(),
        pi: boson1(),
        d0_pi: boson1(),
        pi_d0: boson1(),
        pol: boson1(),
        ggg: ContourGreen::new(&cs, Sign::Fermi, Complex::ZERO),
        w0,
        beta,
        h,
        k,
        integ: Integrator::new(k),
        d_ws: Vie2Workspace::new(k, 1, ntau as usize),
    }
}

/// 1x1 single-time function from `values[n]` (my time step `n`).
pub(crate) fn scalar_time_function(nt: i64, values: &[Complex]) -> TimeFunction {
    let mut f = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    for n in MATSUBARA..=nt {
        f.block_mut(n)[0] = values[(n + 1) as usize];
    }
    f
}

/// `gGg = g G g` at the KB time step `n`.
pub(crate) fn form_g_g_g(n: i64, g: &ContourGreen, gf: &TimeFunction, ph: &mut HolsteinPhonon) {
    ph.ggg.memcpy_from_at(matsubara_step(n), g);
    ph.ggg.rmul_cf_at(matsubara_step(n), gf, Adjoint::No);
    ph.ggg.lmul_cf_at(matsubara_step(n), gf, Adjoint::No);
}

/// `Pi = -2 i gGg(t,t') G(t',t)` at the KB time step `n`.
fn phonon_polarization(n: i64, g: &ContourGreen, ph: &mut HolsteinPhonon) {
    bubble1(
        matsubara_step(n),
        &mut ph.pol,
        0,
        0,
        &ph.ggg,
        &ph.ggg,
        0,
        0,
        g,
        g,
        0,
        0,
    );
    ph.pi.memcpy_from_at(matsubara_step(n), &ph.pol);
    ph.pi.smul_at(matsubara_step(n), Complex::new(-2.0, 0.0));
}

/// `C = A * B` at the KB time step `n >= 1` (the time step is cleared
/// first, the lmix and less components accumulate).
#[allow(clippy::too_many_arguments)]
fn convolve_step(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    ws: &mut ConvWorkspace,
) {
    zero_green_step(c, matsubara_step(n));
    convolution_timestep(
        matsubara_step(n),
        c,
        a,
        a,
        None,
        b,
        b,
        integ,
        beta,
        h,
        Some(ws),
    );
}

/// Solve `[1 - D0*Pi] * D = D0` at the KB time step `n` (`n = 0`:
/// Matsubara, `n > k + 1`: time step), `Hols::step_D`.
pub(crate) fn phonon_timestep(n: i64, ph: &mut HolsteinPhonon, method: MatsubaraMethod) {
    if n == 0 {
        conv_mat(
            &mut ph.d0_pi.mat,
            &ph.d0.mat,
            &ph.pi.mat,
            &ph.integ,
            ph.beta,
            Sign::Bose,
        );
        conv_mat(
            &mut ph.pi_d0.mat,
            &ph.pi.mat,
            &ph.d0.mat,
            &ph.integ,
            ph.beta,
            Sign::Bose,
        );
    } else {
        let (beta, h, ki) = (ph.beta, ph.h, &ph.integ);
        let mut ws1 = ConvWorkspace::new(ph.d0.size1, ph.d0.size2, ph.pi.size1, ph.pi.size2);
        let mut ws2 = ConvWorkspace::new(ph.pi.size1, ph.pi.size2, ph.d0.size1, ph.d0.size2);
        convolve_step(n, &mut ph.d0_pi, &ph.d0, &ph.pi, ki, beta, h, &mut ws1);
        convolve_step(n, &mut ph.pi_d0, &ph.pi, &ph.d0, ki, beta, h, &mut ws2);
    }
    ph.d0_pi.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
    ph.pi_d0.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
    if n == 0 {
        vie2_mat(
            &mut ph.d, &ph.d0_pi, &ph.pi_d0, &ph.d0, ph.beta, &ph.integ, method, 6, 40, 1e-12,
        );
    } else if matsubara_step(n) > ph.k {
        // stepping: the fast path with the persistent workspace
        vie2_timestep_ws(
            matsubara_step(n),
            &mut ph.d,
            &ph.d0_pi,
            &ph.pi_d0,
            &ph.d0,
            &ph.integ,
            ph.beta,
            ph.h,
            &mut ph.d_ws,
        );
    } else {
        vie2_timestep(
            matsubara_step(n),
            &mut ph.d,
            &ph.d0_pi,
            &ph.pi_d0,
            &ph.d0,
            &ph.integ,
            ph.beta,
            ph.h,
            MatsubaraMethod::Fixpoint,
        );
    }
}

/// Start-up block (KB time steps `1, .., k + 1`) of the phonon Dyson
/// equation, `Hols::start_D`.
pub(crate) fn phonon_start(ph: &mut HolsteinPhonon) {
    // one workspace per (a, b) pair: the `f_0 * B^M` cache of each
    // stays valid across the whole loop (same pair every step)
    let mut ws1 = ConvWorkspace::new(ph.d0.size1, ph.d0.size2, ph.pi.size1, ph.pi.size2);
    let mut ws2 = ConvWorkspace::new(ph.pi.size1, ph.pi.size2, ph.d0.size1, ph.d0.size2);
    for n in 1..=ph.k + 1 {
        let (beta, h, ki) = (ph.beta, ph.h, &ph.integ);
        convolve_step(n, &mut ph.d0_pi, &ph.d0, &ph.pi, ki, beta, h, &mut ws1);
        ph.d0_pi.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
        convolve_step(n, &mut ph.pi_d0, &ph.pi, &ph.d0, ki, beta, h, &mut ws2);
        ph.pi_d0.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
    }
    vie2_start(&mut ph.d, &ph.d0_pi, &ph.pi_d0, &ph.d0, ph.k, ph.beta, ph.h);
}

/// Self-consistent Migdal self-energy at the KB time step `n` (`n = 0`
/// or `n > k + 1`), `Hols::Sigma_Mig`: `Pi`, the phonon Dyson
/// equation, and `Sigma = i D gGg`.
pub(crate) fn migdal_self_energy(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
    method: MatsubaraMethod,
) {
    form_g_g_g(n, g, gf, ph);
    phonon_polarization(n, g, ph);
    phonon_timestep(n, ph, method);
    bubble2(
        matsubara_step(n),
        sigma,
        0,
        0,
        &ph.d,
        &ph.d,
        0,
        0,
        &ph.ggg,
        &ph.ggg,
        0,
        0,
    );
}

/// Migdal self-energy of the start-up time steps (bulk form).
pub(crate) fn migdal_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        form_g_g_g(n, g, gf, ph);
        phonon_polarization(n, g, ph);
    }
    phonon_start(ph);
    for n in 1..=ph.k + 1 {
        bubble2(
            matsubara_step(n),
            sigma,
            0,
            0,
            &ph.d,
            &ph.d,
            0,
            0,
            &ph.ggg,
            &ph.ggg,
            0,
            0,
        );
    }
}

/// Unrenormalized Migdal self-energy `Sigma = i D0 gGg` at the KB time
/// step `n`.
pub(crate) fn umigdal_self_energy(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    form_g_g_g(n, g, gf, ph);
    bubble2(
        matsubara_step(n),
        sigma,
        0,
        0,
        &ph.d0,
        &ph.d0,
        0,
        0,
        &ph.ggg,
        &ph.ggg,
        0,
        0,
    );
}

/// uMig self-energy of the start-up time steps.
pub(crate) fn umigdal_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        umigdal_self_energy(n, g, sigma, gf, ph);
    }
}

/// `sum_j w_{nj} f_j` with the Gregory weights; `f[j]` is the value at
/// the index `j`.
pub(crate) fn integrate_gregory(integ: &Integrator, f: &[Complex], n: i64) -> Complex {
    if n == 0 {
        return Complex::ZERO;
    }
    let n1 = if n <= integ.k { integ.k } else { n };
    let mut s = Complex::ZERO;
    for j in 0..=n1 {
        s += integ.giw.at(n as usize, j as usize) * f[j as usize];
    }
    s
}

/// Phonon displacement at the KB time step `n`:
/// `X(t) = X0 + int_0^t D0^R(t,t') [g n(t') - g n(0^-)] dt'` with
/// `X0 = -2 g n(0^-) / w0`. `X`, `ntot`, and `g` are indexed by `n`.
pub(crate) fn phonon_displacement(
    n: i64,
    x: &mut [Complex],
    ntot: &[Complex],
    g: &[Complex],
    ph: &HolsteinPhonon,
) {
    // the arrays are indexed by the KB step number (0 = Matsubara)
    let ng = |m: usize| g[m] * ntot[m];
    let x0 = -2.0 * ng(0) / Complex::new(ph.w0, 0.0);
    if n == 0 {
        x[0] = x0;
    } else {
        let t = n - 1;
        let mut f = vec![Complex::ZERO; n as usize];
        for it in 0..n {
            // the source at the real time t_it = KB it + 1
            f[it as usize] = ph.d0.ret.raw_block_at(t, it)[0] * (ng(it as usize + 1) - ng(0));
        }
        x[n as usize] = integrate_gregory(&ph.integ, &f, t) * Complex::new(ph.h, 0.0) + x0;
    }
}

/// Displacement of the start-up time steps (bulk form): the integral of the
/// time step `k` with `g n(t') = g n(0^-)` for `t' < 0`.
pub(crate) fn phonon_displacement_start(
    x: &mut [Complex],
    ntot: &[Complex],
    g: &[Complex],
    ph: &HolsteinPhonon,
) {
    let k = ph.k;
    let ng = |m: usize| g[m] * ntot[m];
    let x0 = -2.0 * ng(0) / Complex::new(ph.w0, 0.0);
    for nn in 0..=k {
        let mut f = vec![Complex::ZERO; (k + 1) as usize];
        for it in 0..=k {
            // m is a real-time index (the value sits at the KB m + 1);
            // before the quench g n is frozen at the Matsubara value
            let m = it + nn - k;
            let ngm = if m >= 0 { ng(m as usize + 1) } else { ng(0) };
            f[it as usize] = ph.d0.ret.raw_block_at(k, it)[0] * (ngm - ng(0));
        }
        x[nn as usize + 1] = integrate_gregory(&ph.integ, &f, k) * Complex::new(ph.h, 0.0) + x0;
    }
}

/// `Delta(t,t') = J(t) G(t,t') J(t')` at the KB time step `n` (right
/// multiplication first,).
pub(crate) fn hybridization(n: i64, hyb: &mut ContourGreen, g: &ContourGreen, j: &TimeFunction) {
    hyb.memcpy_from_at(matsubara_step(n), g);
    hyb.rmul_cf_at(matsubara_step(n), j, Adjoint::No);
    hyb.lmul_cf_at(matsubara_step(n), j, Adjoint::No);
}

/// `HS = Hyb + Sigma` at the KB time step `n`.
pub(crate) fn hybridization_self_energy(
    n: i64,
    hs: &mut ContourGreen,
    hyb: &ContourGreen,
    sigma: &ContourGreen,
) {
    hs.memcpy_from_at(matsubara_step(n), hyb);
    add_green_step(hs, sigma, matsubara_step(n), Complex::ONE);
}

/// Values of a drive at the KB time steps (my time steps).
pub(crate) fn drive_series(cfg: &ExampleConfig, name: &str) -> Result<Vec<Complex>, String> {
    let c = contour_params(cfg)?;
    let f = load_tvector(cfg, name)?;
    let mut v = vec![Complex::ZERO; (c.nt + 2) as usize];
    for n in MATSUBARA..=c.nt {
        v[(n + 1) as usize] = f.block(n)[0];
    }
    Ok(v)
}

/// The model keys must be numbers and the phonon frequency must be positive.
pub(crate) fn check_model(cfg: &ExampleConfig, cfgfile: &str, keys: &[&str]) -> Result<(), String> {
    for key in keys {
        if cfg.model[*key].as_f64().is_none() {
            return Err(format!("{cfgfile}: [model] {key} must be a number"));
        }
    }
    if cfg.model["Phfreq_w0"].as_f64().unwrap().is_nan()
        || cfg.model["Phfreq_w0"].as_f64().unwrap() <= 0.0
    {
        return Err(format!("{cfgfile}: [model] Phfreq_w0 must be > 0"));
    }
    Ok(())
}

/// The coupling constant and the hopping with their drives.
pub(crate) fn couplings(cfg: &ExampleConfig, nt: i64) -> Result<Couplings, String> {
    let m = &cfg.model;
    let gd = drive_series(cfg, "dEl_Ph_g")?;
    let jd = drive_series(cfg, "dHopping")?;
    let gv: Vec<Complex> = (0..=(nt + 1) as usize)
        .map(|q| Complex::new(m["El_Ph_g"].as_f64().unwrap(), 0.0) + gd[q])
        .collect();
    let jv: Vec<Complex> = (0..=(nt + 1) as usize)
        .map(|q| Complex::new(m["Hopping"].as_f64().unwrap(), 0.0) + jd[q])
        .collect();
    let gf = scalar_time_function(nt, &gv);
    let jf = scalar_time_function(nt, &jv);
    Ok(Couplings { gv, gf, jf })
}

/// The output table `<prefix>_obs.dat` of a configuration.
pub(crate) fn obs_path(cfg: &ExampleConfig) -> Result<String, String> {
    Ok(format!(
        "{}/{}_obs.dat",
        example_outdir(cfg)?,
        cfg.output["prefix"].as_str().unwrap()
    ))
}
