//
// Shared helpers of the Holstein examples (the Rust port of
// julia/tutor/holstein_bethe/holstein_impl.jl, from NESSi
// examples/programs/Holstein_impurity_impl.cpp): normal-state
// self-energies in the self-consistent (Migdal) and unrenormalized
// (uMig) Migdal approximations, the Dyson equation of the phonon
// propagator, and the phonon displacement.
//
//   Pi(t,t')    = -2 i g G(t,t') g G(t',t)
//   D           = D0 + D0 * Pi * D        ([1 - D0*Pi] * D = D0)
//   Sigma(t,t') =  i D(t,t') g G(t,t') g   (Migdal, D0 for uMig)
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::config::{contour_params, ExampleConfig};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1, bubble2};
use kadanoffbaym::convolution::mat::conv_mat;
use kadanoffbaym::convolution::timestep::convolution_time_step;
use kadanoffbaym::equilibrium::green_single_pole;
use kadanoffbaym::dyson::mat::MatsubaraMethod;
use kadanoffbaym::vie2::mat::vie2_mat;
use kadanoffbaym::vie2::start::vie2_start;
use kadanoffbaym::vie2::timestep::vie2_timestep;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::function::TimeFunction;
use kadanoffbaym::basics::MATSUBARA;
use kadanoffbaym::weights::Integrator;

/// Map a blueprint KB step onto the library step.
pub fn mt(n: i64) -> i64 {
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

/// `dst += alpha * src` at the single step `t` (blueprint `incr!`).
pub fn add_step(dst: &mut ContourGreen, src: &ContourGreen, t: i64, alpha: Complex) {
    if t == MATSUBARA {
        for q in 0..src.mat.raw().len() {
            dst.mat.raw_mut()[q] += alpha * src.mat.raw()[q];
        }
    } else {
        for j in 0..=t {
            for q in 0..src.ret.raw_block_at(t, j).len() {
                dst.ret.raw_block_at_mut(t, j)[q] += alpha * src.ret.raw_block_at(t, j)[q];
            }
        }
        for i in 0..=t {
            for q in 0..src.less.raw_block_at(i, t).len() {
                dst.less.raw_block_at_mut(i, t)[q] += alpha * src.less.raw_block_at(i, t)[q];
            }
        }
        for m in 0..=src.ntau as i64 {
            for q in 0..src.lmix.raw_block_at(t, m).len() {
                dst.lmix.raw_block_at_mut(t, m)[q] += alpha * src.lmix.raw_block_at(t, m)[q];
            }
        }
    }
}

/// Zero the single step `t` of `g` (blueprint `zeros!`).
pub fn zero_step(g: &mut ContourGreen, t: i64) {
    if t == MATSUBARA {
        for q in g.mat.raw_mut() {
            *q = Complex::ZERO;
        }
    } else {
        for j in 0..=t {
            for q in g.ret.raw_block_at_mut(t, j) {
                *q = Complex::ZERO;
            }
        }
        for i in 0..=t {
            for q in g.less.raw_block_at_mut(i, t) {
                *q = Complex::ZERO;
            }
        }
        for m in 0..=g.ntau as i64 {
            for q in g.lmix.raw_block_at_mut(t, m) {
                *q = Complex::ZERO;
            }
        }
    }
}

/// Phonon propagators and work space.
pub struct HolsteinPhonon {
    pub d0: ContourGreen,
    pub d: ContourGreen,
    pub pi: ContourGreen,
    pub d0_pi: ContourGreen,
    pub pi_d0: ContourGreen,
    pub pol: ContourGreen,
    pub ggg: ContourGreen,
    pub w0: f64,
    pub beta: f64,
    pub h: f64,
    pub k: i64,
    pub integ: Integrator,
}

/// Allocate the phonon propagators on the contour, `D0 = 2 D_XX` (the
/// single-pole propagator scaled by two).
pub fn holstein_phonon(
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
    }
}

/// 1x1 single-time function from `values[n]` (my time step `n`).
#[allow(dead_code)]
pub fn holstein_scalar_cf(nt: i64, values: &[Complex]) -> TimeFunction {
    let mut f = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
    for n in MATSUBARA..=nt {
        f.block_mut(n)[0] = values[(n + 1) as usize];
    }
    f
}

/// `gGg = g G g` at the KB time step `n`.
pub fn gg_g(n: i64, g: &ContourGreen, gf: &TimeFunction, ph: &mut HolsteinPhonon) {
    copy_step(g, &mut ph.ggg, mt(n));
    ph.ggg.rmul_cf_at(mt(n), gf, kadanoffbaym::basics::Mode::Normal);
    ph.ggg.lmul_cf_at(mt(n), gf, kadanoffbaym::basics::Mode::Normal);
}

/// `Pi = -2 i gGg(t,t') G(t',t)` at the KB time step `n`.
fn pi_normal(n: i64, g: &ContourGreen, ph: &mut HolsteinPhonon) {
    bubble1(mt(n), &mut ph.pol, 0, 0, &ph.ggg, &ph.ggg, 0, 0, g, g, 0, 0);
    copy_step(&ph.pol, &mut ph.pi, mt(n));
    ph.pi.smul_at(mt(n), Complex::new(-2.0, 0.0));
}

/// `C = A * B` at the KB time step `n >= 1` (the time step is cleared
/// first, the lmix and less components accumulate).
fn conv_step(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    zero_step(c, mt(n));
    convolution_time_step(mt(n), c, a, a, None, b, b, integ, beta, h);
}

/// Solve `[1 - D0*Pi] * D = D0` at the KB time step `n` (`n = 0`:
/// Matsubara, `n > k + 1`: time step), NESSi `Hols::step_D`.
pub fn step_d(n: i64, ph: &mut HolsteinPhonon, method: MatsubaraMethod) {
    if n == 0 {
        conv_mat(&mut ph.d0_pi.mat, &ph.d0.mat, &ph.pi.mat, &ph.integ, ph.beta, Sign::Bose);
        conv_mat(&mut ph.pi_d0.mat, &ph.pi.mat, &ph.d0.mat, &ph.integ, ph.beta, Sign::Bose);
    } else {
        let (beta, h, ki) = (ph.beta, ph.h, &ph.integ);
        conv_step(n, &mut ph.d0_pi, &ph.d0, &ph.pi, ki, beta, h);
        conv_step(n, &mut ph.pi_d0, &ph.pi, &ph.d0, ki, beta, h);
    }
    ph.d0_pi.smul_at(mt(n), Complex::new(-1.0, 0.0));
    ph.pi_d0.smul_at(mt(n), Complex::new(-1.0, 0.0));
    if n == 0 {
        vie2_mat(
            &mut ph.d, &ph.d0_pi, &ph.pi_d0, &ph.d0, ph.beta, &ph.integ, method, 6, 40, 1e-12,
        );
    } else {
        vie2_timestep(
            mt(n), &mut ph.d, &ph.d0_pi, &ph.pi_d0, &ph.d0, &ph.integ, ph.beta, ph.h,
            MatsubaraMethod::Fixpoint,
        );
    }
}

/// Start-up block (KB time steps `1, .., k + 1`) of the phonon Dyson
/// equation, NESSi `Hols::start_D`.
pub fn start_d(ph: &mut HolsteinPhonon) {
    for n in 1..=ph.k + 1 {
        let (beta, h, ki) = (ph.beta, ph.h, &ph.integ);
        conv_step(n, &mut ph.d0_pi, &ph.d0, &ph.pi, ki, beta, h);
        ph.d0_pi.smul_at(mt(n), Complex::new(-1.0, 0.0));
        conv_step(n, &mut ph.pi_d0, &ph.pi, &ph.d0, ki, beta, h);
        ph.pi_d0.smul_at(mt(n), Complex::new(-1.0, 0.0));
    }
    vie2_start(&mut ph.d, &ph.d0_pi, &ph.pi_d0, &ph.d0, ph.k, ph.beta, ph.h);
}

/// Self-consistent Migdal self-energy at the KB time step `n` (`n = 0`
/// or `n > k + 1`), NESSi `Hols::Sigma_Mig`: `Pi`, the phonon Dyson
/// equation, and `Sigma = i D gGg`.
pub fn sigma_mig(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
    method: MatsubaraMethod,
) {
    gg_g(n, g, gf, ph);
    pi_normal(n, g, ph);
    step_d(n, ph, method);
    bubble2(mt(n), sigma, 0, 0, &ph.d, &ph.d, 0, 0, &ph.ggg, &ph.ggg, 0, 0);
}

/// Migdal self-energy of the start-up time steps, NESSi
/// `Hols::Sigma_Mig` (the bulk form).
pub fn sigma_mig_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        gg_g(n, g, gf, ph);
        pi_normal(n, g, ph);
    }
    start_d(ph);
    for n in 1..=ph.k + 1 {
        bubble2(mt(n), sigma, 0, 0, &ph.d, &ph.d, 0, 0, &ph.ggg, &ph.ggg, 0, 0);
    }
}

/// Unrenormalized Migdal self-energy `Sigma = i D0 gGg` at the KB time
/// step `n`.
pub fn sigma_umig(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    gg_g(n, g, gf, ph);
    bubble2(mt(n), sigma, 0, 0, &ph.d0, &ph.d0, 0, 0, &ph.ggg, &ph.ggg, 0, 0);
}

/// uMig self-energy of the start-up time steps.
pub fn sigma_umig_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        sigma_umig(n, g, sigma, gf, ph);
    }
}

/// `sum_j w_{nj} f_j` with the Gregory weights (NESSi
/// `Integrator::integrate`); `f[j]` is the value at the index `j`.
pub fn gregory_integrate(integ: &Integrator, f: &[Complex], n: i64) -> Complex {
    if n == 0 {
        return Complex::ZERO;
    }
    let n1 = if n <= integ.k as i64 {
        integ.k as i64
    } else {
        n
    };
    let mut s = Complex::ZERO;
    for j in 0..=n1 {
        s += integ.giw.at(n as usize, j as usize) * f[j as usize];
    }
    s
}

/// Phonon displacement at the KB time step `n` (NESSi
/// `Hols::get_phonon_displace`):
/// `X(t) = X0 + int_0^t D0^R(t,t') [g n(t') - g n(0^-)] dt'` with
/// `X0 = -2 g n(0^-) / w0`. `X`, `ntot`, and `g` are indexed by `n`.
pub fn phonon_displacement(n: i64, x: &mut [Complex], ntot: &[Complex], g: &[Complex], ph: &HolsteinPhonon) {
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
            f[it as usize] =
                ph.d0.ret.raw_block_at(t, it)[0] * (ng(it as usize + 1) - ng(0));
        }
        x[n as usize] = gregory_integrate(&ph.integ, &f, t) * Complex::new(ph.h, 0.0) + x0;
    }
}

/// Displacement of the start-up time steps (NESSi
/// `Hols::get_phonon_displace` in the bulk form): the integral of the
/// time step `k` with `g n(t') = g n(0^-)` for `t' < 0`.
pub fn phonon_displacement_start(x: &mut [Complex], ntot: &[Complex], g: &[Complex], ph: &HolsteinPhonon) {
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
        x[nn as usize + 1] = gregory_integrate(&ph.integ, &f, k) * Complex::new(ph.h, 0.0) + x0;
    }
}

/// `Delta(t,t') = J(t) G(t,t') J(t')` at the KB time step `n` (right
/// multiplication first, as in NESSi).
pub fn holstein_hybridization(n: i64, hyb: &mut ContourGreen, g: &ContourGreen, j: &TimeFunction) {
    copy_step(g, hyb, mt(n));
    hyb.rmul_cf_at(mt(n), j, kadanoffbaym::basics::Mode::Normal);
    hyb.lmul_cf_at(mt(n), j, kadanoffbaym::basics::Mode::Normal);
}

/// `HS = Hyb + Sigma` at the KB time step `n`.
pub fn hyb_sigma(n: i64, hs: &mut ContourGreen, hyb: &ContourGreen, sigma: &ContourGreen) {
    copy_step(hyb, hs, mt(n));
    add_step(hs, sigma, mt(n), Complex::ONE);
}

/// State of the two-stage Matsubara iteration of the Holstein examples:
/// the FOURIER method until the error is below the threshold, then the
/// FIXPOINT method until it is below the threshold again.
pub struct MatsubaraTwoStage {
    pub fixpoint: bool,
    pub converged: bool,
}

impl MatsubaraTwoStage {
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

/// Values of a drive at the KB time steps (my time steps).
pub fn holstein_drive(cfg: &ExampleConfig, name: &str) -> Result<Vec<Complex>, String> {
    let c = contour_params(cfg)?;
    let f = crate::common::config::load_tvector(cfg, name)?;
    let mut v = vec![Complex::ZERO; (c.nt + 2) as usize];
    for n in MATSUBARA..=c.nt {
        v[(n + 1) as usize] = f.block(n)[0];
    }
    Ok(v)
}
