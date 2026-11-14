//
// Usage:
//   cargo run --example gw -- julia/tutor/gw/gw.toml [--current-mode=nessi|peierls_corrected]
//
// One-dimensional Hubbard model with nearest neighbour interaction in
// the GW approximation, driven by an electric field pulse (Peierls
// phase) (the Rust port of julia/tutor/gw/{gw,gw_lattice,gw_kpoints,
// gw_selfenergy}.jl, from NESSi examples/programs/gw*.cpp; serial, no
// MPI/HDF5). Every k point carries G_k, the GW self-energy, the
// polarization P_q, and the screened interaction W_q = V_q + V_q P_q W_q
// (VIE2 with singular kernel).
//

#![allow(dead_code)]

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1, bubble2};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::observables::{correlation_energy, density_matrix_green};
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::solvers::vie2_sin::{vie2_timestep_sin, Vie2SinWorkspace};
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::weights::Integrator;
use ness_util::{fmt_e12, tget_f, tget_i, tget_s};
use std::io::Write;

fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// The parameters of the GW example.
pub struct GwParams {
    pub nk: i64,
    pub hopping_t: f64,
    pub hubbard_u: f64,
    pub v: f64,
    pub mu_chem: f64,
    pub beta: f64,
    pub nt: i64,
    pub ntau: i64,
    pub h: f64,
    pub solve_order: i64,
    pub mats_max_iter: i64,
    pub mats_max_err: f64,
    pub bootstrap_max_iter: i64,
    pub bootstrap_max_err: f64,
    pub time_max_err: f64,
    pub corrector_steps: i64,
    pub current_mode: String,
    pub e0: f64,
    pub omega: f64,
    pub t0: f64,
    pub width: f64,
    pub dir: String,
    pub prefix: String,
}

fn read_gw_params(cfgfile: &str) -> Result<GwParams, String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let q = GwParams {
        nk: tget_i(&p, "model", "Nk")?,
        hopping_t: tget_f(&p, "model", "HoppingT")?,
        hubbard_u: tget_f(&p, "model", "HubbardU")?,
        v: tget_f(&p, "model", "V")?,
        mu_chem: tget_f(&p, "model", "MuChem")?,
        beta: tget_f(&p, "model", "beta")?,
        nt: tget_i(&p, "contour", "Nt")?,
        // the TOML counts the imaginary time POINTS (KB convention);
        // the library counts the intervals
        ntau: tget_i(&p, "contour", "Ntau")? - 1,
        h: tget_f(&p, "contour", "h")?,
        solve_order: tget_i(&p, "solver", "SolveOrder")?,
        mats_max_iter: tget_i(&p, "solver", "MatsMaxIter")?,
        mats_max_err: tget_f(&p, "solver", "MatsMaxErr")?,
        bootstrap_max_iter: tget_i(&p, "solver", "BootstrapMaxIter")?,
        bootstrap_max_err: tget_f(&p, "solver", "BootstrapMaxErr")?,
        time_max_err: tget_f(&p, "solver", "TimeMaxErr")?,
        corrector_steps: tget_i(&p, "solver", "CorrectorSteps")?,
        current_mode: {
            let m = tget_s(&p, "solver", "current_mode")?;
            if m != "nessi" && m != "peierls_corrected" {
                return Err(format!("{cfgfile}: current_mode must be nessi or peierls_corrected"));
            }
            m
        },
        e0: tget_f(&p, "pulse", "E0")?,
        omega: tget_f(&p, "pulse", "omega")?,
        t0: tget_f(&p, "pulse", "t0")?,
        width: tget_f(&p, "pulse", "width")?,
        dir: String::new(),
        prefix: tget_s(&p, "output", "prefix")?,
    };
    if q.nk <= 1 {
        return Err(format!("{cfgfile}: Nk must be > 1"));
    }
    if !(1..=5).contains(&q.solve_order) {
        return Err(format!("{cfgfile}: SolveOrder must be in 1..=5"));
    }
    if q.nt < q.solve_order {
        return Err(format!("{cfgfile}: Nt must be >= SolveOrder"));
    }
    if !(q.h > 0.0 && q.beta > 0.0 && q.ntau >= 2) {
        return Err(format!("{cfgfile}: invalid contour parameters"));
    }
    let dir = ness_util::ness_outdir(&p, cfgfile)?;
    Ok(GwParams { dir, ..q })
}

/// Electric field at the KB time steps (index = the KB step),
/// `E(t) = E0 exp(-((t - t0)/w)^2) sin(omega (t - t0))`.
fn electric_field(p: &GwParams) -> Vec<f64> {
    let mut e = vec![0.0; (p.nt + 2) as usize];
    for n in 1..=p.nt + 1 {
        let t = (n - 1) as f64 * p.h;
        e[n as usize] = p.e0
            * (-((t - p.t0) / p.width) * ((t - p.t0) / p.width)).exp()
            * (p.omega * (t - p.t0)).sin();
    }
    e
}

/// The one-dimensional lattice.
pub struct GwLattice {
    pub nk: i64,
    pub g: i64,
    pub kpoints: Vec<f64>,
    pub kweights: Vec<f64>,
    pub th: f64,
    pub u: f64,
    pub v: f64,
    pub afield: Vec<f64>,
}

/// `A(t) = -int E dt` by Gregory integration of the order `min(k, Nt)`
/// (blueprint `efield_to_afield`).
fn efield_to_afield(e: &[f64], h: f64, k: i64) -> Vec<f64> {
    let nt = (e.len() as i64) - 2;
    let kt1 = if nt >= k { k } else { nt };
    let integ = Integrator::new(kt1);
    let mut a = vec![0.0; e.len()];
    for tstp in 0..=nt {
        let n1 = if tstp < kt1 { kt1 } else { tstp };
        let mut acc = 0.0;
        for n in 0..=n1 {
            acc += integ.giw.at(tstp as usize, n as usize) * e[(n + 1) as usize];
        }
        a[(tstp + 1) as usize] = -h * acc;
    }
    a
}

fn build_kgrid(nk: i64) -> (i64, Vec<f64>, Vec<f64>) {
    let g = nk / 2;
    let kp = (0..nk).map(|j| (j - g) as f64 * 2.0 * std::f64::consts::PI / nk as f64).collect();
    (g, kp, vec![1.0 / nk as f64; nk as usize])
}

impl GwLattice {
    fn new(nk: i64, th: f64, u: f64, v: f64, e: &[f64], h: f64, k: i64) -> Self {
        let (g, kp, kw) = build_kgrid(nk);
        GwLattice {
            nk,
            g,
            kpoints: kp,
            kweights: kw,
            th,
            u,
            v,
            afield: efield_to_afield(e, h, k),
        }
    }

    /// Index (0-based) of the momentum `s1 k1 + s2 k2` folded back.
    fn add_kpoints(&self, k1: i64, s1: i64, k2: i64, s2: i64) -> i64 {
        let k12 = s1 * (k1 - self.g) + s2 * (k2 - self.g) + self.g;
        k12.rem_euclid(self.nk)
    }

    /// `eps_k(t) = -2 th cos(k - A(t))` at the KB step `n`.
    fn dispersion(&self, k: f64, n: i64) -> f64 {
        -2.0 * self.th * (k - self.afield[n as usize]).cos()
    }

    fn velocity_nessi(&self, k: f64) -> f64 {
        2.0 * self.th * k.sin()
    }

    fn velocity_peierls(&self, k: f64, n: i64) -> f64 {
        2.0 * self.th * (k - self.afield[n as usize]).sin()
    }

    fn interaction_vertex(&self, q: f64) -> f64 {
        self.u + 2.0 * self.v * q.cos()
    }
}

/// Hartree self-energy (the same for all k points).
fn sigma_hartree(lat: &GwLattice, dens: &[Complex]) -> Complex {
    let vloc = lat.u;
    let vnl = lat.interaction_vertex(lat.kpoints[lat.g as usize]) - vloc;
    let mut s = Complex::ZERO;
    for k in 0..lat.nk as usize {
        let (wk, rho) = (lat.kweights[k], dens[k]);
        s += Complex::new(vloc * rho.re * wk, 0.0);
        s += Complex::new(vnl * rho.re * wk * 2.0, 0.0);
    }
    s
}

/// Fock self-energy of the k point `k` (0-based).
fn sigma_fock(k: i64, lat: &GwLattice, dens: &[Complex]) -> Complex {
    let mut s = Complex::ZERO;
    for q in 0..lat.nk {
        let wq = lat.kweights[q as usize];
        let kq = lat.add_kpoints(k, 1, q, -1);
        let vnl = lat.interaction_vertex(lat.kpoints[q as usize]) - lat.u;
        s -= Complex::new(vnl, 0.0) * dens[kq as usize] * Complex::new(wq, 0.0);
    }
    s
}

/// One k point of the GW example.
pub struct GwKPoint {
    pub k: f64,
    pub g: ContourGreen,
    pub sigma: ContourGreen,
    pub p: ContourGreen,
    pub pv: ContourGreen,
    pub vp: ContourGreen,
    pub w: ContourGreen,
    pub wsin: TimeFunction,
    pub sh: TimeFunction,
    pub sf: TimeFunction,
    pub rho: TimeFunction,
    pub hk: TimeFunction,
    pub hkeff: TimeFunction,
    pub vertex: TimeFunction,
    pub ws: Vie2SinWorkspace,
}

impl GwKPoint {
    fn new(cc: &Contour, nt: i64, lat: &GwLattice, j: i64) -> Self {
        let k = lat.kpoints[j as usize];
        let mut hk = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        let mut vertex = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        for kb in 0..=nt + 1 {
            // dispersion takes the KB time step; the vertex is constant
            hk.block_mut(mt(kb))[0] = Complex::new(lat.dispersion(k, kb), 0.0);
            vertex.block_mut(mt(kb))[0] = Complex::new(lat.interaction_vertex(k), 0.0);
        }
        let hkeff = hk.clone();
        let w = ContourGreen::new(cc, Sign::Bose, Complex::ZERO);
        GwKPoint {
            k,
            g: ContourGreen::new(cc, Sign::Fermi, Complex::ZERO),
            sigma: ContourGreen::new(cc, Sign::Fermi, Complex::ZERO),
            p: ContourGreen::new(cc, Sign::Bose, Complex::ZERO),
            pv: ContourGreen::new(cc, Sign::Bose, Complex::ZERO),
            vp: ContourGreen::new(cc, Sign::Bose, Complex::ZERO),
            w: w.clone(),
            wsin: TimeFunction::new(nt as usize, 1, 1, Complex::ZERO),
            sh: TimeFunction::new(nt as usize, 1, 1, Complex::ZERO),
            sf: TimeFunction::new(nt as usize, 1, 1, Complex::ZERO),
            rho: TimeFunction::new(nt as usize, 1, 1, Complex::ZERO),
            hk,
            hkeff,
            vertex,
            ws: Vie2SinWorkspace::new(&w),
        }
    }
}

/// Time steps touched at the KB step `n` (the start-up steps are solved
/// together).
fn steprange(n: i64, order: i64) -> std::ops::RangeInclusive<i64> {
    if n == 0 || n > order + 1 {
        n..=n
    } else {
        1..=order + 1
    }
}

/// Solve `W = V + V P W` as the VIE2 with singular kernel
/// `[1 - V P] W = V` (blueprint `step_w!`).
fn step_w(kp: &mut GwKPoint, n: i64, order: i64, beta: f64, h: f64, fsin: &TimeFunction, q: &ContourGreen) {
    for m in steprange(n, order) {
        // PV = -(P vertex), VP = -(vertex P) at the step m
        copy_green_step(&kp.p, &mut kp.pv, mt(m));
        let vx = kp.vertex.block(mt(m))[0];
        kp.pv.smul_at(mt(m), vx * Complex::new(-1.0, 0.0));
        copy_green_step(&kp.p, &mut kp.vp, mt(m));
        kp.vp.lmul_scalar_at(mt(m), vx * Complex::new(-1.0, 0.0));
    }
    vie2_timestep_sin(
        mt(n),
        &mut kp.w,
        &mut kp.wsin,
        &kp.vp,
        &kp.pv,
        fsin,
        q,
        &kp.vertex,
        &Integrator::new(order),
        beta,
        h,
        MatsubaraMethod::Fixpoint,
        &mut kp.ws,
    );
}

/// Solve the Dyson equation and store the densities (blueprint
/// `step_dyson!`); the second Matsubara iteration uses the
/// noninteracting Green's function of `hkeff`.
fn step_dyson(kp: &mut GwKPoint, n: i64, iter: i64, lat: &GwLattice, mu: f64, beta: f64, h: f64, order: i64) {
    let r = steprange(n, order);
    let m0 = *r.start();
    let m1 = *r.end();
    for m in r {
        kp.hk.block_mut(mt(m))[0] = Complex::new(lat.dispersion(kp.k, m), 0.0);
        kp.hkeff.block_mut(mt(m))[0] =
            kp.sh.block(mt(m))[0] + kp.sf.block(mt(m))[0] + kp.hk.block(mt(m))[0];
    }
    if n == 0 && iter == 1 {
        let cc1 = Contour::new(1, kp.g.ntau as i64, 1, 1, h, beta);
        let mut t = ContourGreen::new(&cc1, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, kp.hkeff.block(MATSUBARA)[0]);
        from_ham(&mut t, mu, &h0, beta, h);
        kp.g.mat.memcpy_from(&t.mat);
    } else if n == 0 {
        dyson_mat(&mut kp.g, &kp.sigma, mu, &kp.hkeff, beta, order, MatsubaraMethod::Fixpoint, true, None, 1e-12);
    } else if n <= order + 1 {
        dyson_start(&mut kp.g, mu, &kp.hkeff, &kp.sigma, beta, h, order);
    } else {
        dyson_timestep(mt(n), &mut kp.g, mu, &kp.hkeff, &kp.sigma, beta, h, order);
    }
    for m in m0..=m1 {
        kp.rho.block_mut(mt(m))[0] = density_matrix_green(&kp.g, mt(m)).get(0, 0);
    }
}

/// Copy a single step (the local helper).
fn copy_green_step(src: &ContourGreen, dst: &mut ContourGreen, t: i64) {
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

/// The GW state.
pub struct GwState {
    pub p: GwParams,
    pub lat: GwLattice,
    pub kps: Vec<GwKPoint>,
    pub dens: Vec<Vec<Complex>>,
    pub q: ContourGreen,
    pub fsin: TimeFunction,
}

/// One iteration at the KB step `n` (blueprint `gw_iteration!`).
fn gw_iteration(n: i64, iter: i64, st: &mut GwState) -> (f64, f64) {
    let (nk, k) = (st.lat.nk, st.p.solve_order);
    let nmu = nk as usize;
    // the density matrix is hermitized
    for j in 0..nmu {
        let rho = density_matrix_green(&st.kps[j].g, mt(n)).get(0, 0);
        st.dens[j][n as usize] = Complex::new(rho.re, 0.0);
    }
    let dn: Vec<Complex> = (0..nmu).map(|j| st.dens[j][n as usize]).collect();
    let sh = sigma_hartree(&st.lat, &dn);
    for j in 0..nmu {
        st.kps[j].sh.block_mut(mt(n))[0] = sh;
        st.kps[j].sf.block_mut(mt(n))[0] = sigma_fock(j as i64, &st.lat, &dn);
        // GW self-energy: Sigma_k = sum_q w_q i G_{k-q} W_q (Bubble2);
        // a local work object avoids aliasing kps[j] with kps[kq]
        let mut acc = st.kps[j].sigma.clone();
        zero_green_at(&mut acc, mt(n));
        let mut work = st.kps[j].sigma.clone();
        for q in 0..st.lat.nk {
            let kq = st.lat.add_kpoints(j as i64, 1, q, -1);
            zero_green_at(&mut work, mt(n));
            bubble2(
                mt(n),
                &mut work,
                0,
                0,
                &st.kps[kq as usize].g,
                &st.kps[kq as usize].g,
                0,
                0,
                &st.kps[q as usize].w,
                &st.kps[q as usize].w,
                0,
                0,
            );
            add_green_at(&mut acc, &work, mt(n), Complex::new(st.lat.kweights[q as usize], 0.0));
        }
        copy_green_step(&acc, &mut st.kps[j].sigma, mt(n));
    }
    for q in 0..st.lat.nk {
        // polarization: P_q = -sum_k w_k i G_{k-q} G_k (Bubble1)
        let mut acc = st.kps[q as usize].p.clone();
        zero_green_at(&mut acc, mt(n));
        let mut work = st.kps[q as usize].p.clone();
        for kk in 0..st.lat.nk {
            let kq = st.lat.add_kpoints(kk, 1, q, -1);
            zero_green_at(&mut work, mt(n));
            bubble1(
                mt(n),
                &mut work,
                0,
                0,
                &st.kps[kq as usize].g,
                &st.kps[kq as usize].g,
                0,
                0,
                &st.kps[kk as usize].g,
                &st.kps[kk as usize].g,
                0,
                0,
            );
            add_green_at(&mut acc, &work, mt(n), Complex::new(-st.lat.kweights[kk as usize], 0.0));
        }
        copy_green_step(&acc, &mut st.kps[q as usize].p, mt(n));
    }
    let mut e1 = 0.0f64;
    let mut e2 = 0.0f64;
    for j in 0..nmu {
        // the Dyson step with the error
        let mut gsave = st.kps[j].g.clone();
        copy_green_step(&st.kps[j].g, &mut gsave, mt(n));
        step_dyson(&mut st.kps[j], n, iter, &st.lat, st.p.mu_chem, st.p.beta, st.p.h, k);
        e1 += gsave.distance_at(&st.kps[j].g, mt(n));
    }
    for j in 0..nmu {
        let mut wsave = st.kps[j].w.clone();
        copy_green_step(&st.kps[j].w, &mut wsave, mt(n));
        step_w(&mut st.kps[j], n, k, st.p.beta, st.p.h, &st.fsin, &st.q);
        e2 += wsave.distance_at(&st.kps[j].w, mt(n));
    }
    (e1, e2)
}

fn zero_green_at(g: &mut ContourGreen, t: i64) {
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

fn add_green_at(dst: &mut ContourGreen, src: &ContourGreen, t: i64, alpha: Complex) {
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

/// `dst[t] = scalar * dst[t]` at one step (the left scalar multiply).
trait ScalarAt {
    fn lmul_scalar_at(&mut self, t: i64, a: Complex);
}

impl ScalarAt for ContourGreen {
    fn lmul_scalar_at(&mut self, t: i64, a: Complex) {
        // scalar multiplication is side independent
        self.smul_at(t, a);
    }
}

/// Matsubara iterations, bootstrapping, and time stepping (blueprint
/// `solve_gw!`).
pub fn solve_gw(p: &GwParams) -> Result<GwState, String> {
    let ntime = p.nt + 1;
    let k = p.solve_order;
    let cc = Contour::new(p.nt, p.ntau, 1, 1, p.nt as f64 * p.h, p.beta);
    let lat = GwLattice::new(p.nk, p.hopping_t, p.hubbard_u, p.v, &electric_field(p), p.h, k);
    let kps: Vec<GwKPoint> = (0..p.nk).map(|j| GwKPoint::new(&cc, p.nt, &lat, j)).collect();
    let mut st = GwState {
        p: GwParams { dir: p.dir.clone(), ..read_gw_params_clone(p) },
        lat,
        kps,
        dens: vec![vec![Complex::ZERO; (ntime + 1) as usize]; p.nk as usize],
        q: ContourGreen::new(&cc, Sign::Bose, Complex::ZERO),
        fsin: TimeFunction::new(p.nt as usize, 1, 1, Complex::ZERO),
    };

    // noninteracting start
    for kp in &mut st.kps {
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, kp.hk.block(MATSUBARA)[0]);
        from_ham(&mut kp.g, p.mu_chem, &h0, p.beta, p.h);
    }

    // Matsubara iterations
    let mut converged = false;
    for iter in 0..=p.mats_max_iter {
        let (e1, e2) = gw_iteration(0, iter, &mut st);
        println!("tstp = 0 iter = {iter}  err ele = {e1:.6e}  err bos = {e2:.6e}");
        if e1 < p.mats_max_err && e2 < p.mats_max_err {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("Matsubara iteration not converged".to_string());
    }

    // Bootstrapping
    for tb in 1..=k + 1 {
        for kp in &mut st.kps {
            if tb == 1 {
                kadanoffbaym::equilibrium::from_mat::from_mat(&mut kp.g);
                kadanoffbaym::equilibrium::from_mat::from_mat(&mut kp.w);
            } else {
                let ord = k.min(tb - 2);
                kadanoffbaym::math::extrap::extrapolate_timestep_order(tb - 2, &mut kp.g, ord);
                kadanoffbaym::math::extrap::extrapolate_timestep_order(tb - 2, &mut kp.w, ord);
            }
        }
        let mut converged = false;
        for iter in 0..=p.bootstrap_max_iter {
            let mut last = (0.0, 0.0);
            for n in 1..=tb {
                last = gw_iteration(n, iter, &mut st);
            }
            let (e1, e2) = last;
            if e1 < p.bootstrap_max_err && e2 < p.bootstrap_max_err && iter > 2 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(format!("bootstrapping not converged at the time step {tb}"));
        }
    }

    // Time stepping
    for tb in (k + 2)..=ntime {
        for kp in &mut st.kps {
            kadanoffbaym::math::extrap::extrapolate_timestep_order(tb - 2, &mut kp.g, k);
            kadanoffbaym::math::extrap::extrapolate_timestep_order(tb - 2, &mut kp.w, k);
        }
        for iter in 0..p.corrector_steps {
            let (e1, e2) = gw_iteration(tb, iter, &mut st);
            if e1 < p.time_max_err && e2 < p.time_max_err && iter > 1 {
                break;
            }
        }
    }
    Ok(st)
}

/// Clone helper for the parameters kept in the state.
fn read_gw_params_clone(p: &GwParams) -> GwParams {
    GwParams {
        nk: p.nk,
        hopping_t: p.hopping_t,
        hubbard_u: p.hubbard_u,
        v: p.v,
        mu_chem: p.mu_chem,
        beta: p.beta,
        nt: p.nt,
        ntau: p.ntau,
        h: p.h,
        solve_order: p.solve_order,
        mats_max_iter: p.mats_max_iter,
        mats_max_err: p.mats_max_err,
        bootstrap_max_iter: p.bootstrap_max_iter,
        bootstrap_max_err: p.bootstrap_max_err,
        time_max_err: p.time_max_err,
        corrector_steps: p.corrector_steps,
        current_mode: p.current_mode.clone(),
        e0: p.e0,
        omega: p.omega,
        t0: p.t0,
        width: p.width,
        dir: p.dir.clone(),
        prefix: p.prefix.clone(),
    }
}

/// One observables row.
pub struct GwRow {
    pub tstp: i64,
    pub t: f64,
    pub e: f64,
    pub a: f64,
    pub ekin: f64,
    pub eint: f64,
    pub density: f64,
    pub current_nessi: f64,
    pub current_corrected: f64,
}

/// Observables at the KB time steps (blueprint `gw_observables`).
pub fn gw_observables(st: &GwState) -> Vec<GwRow> {
    let p = &st.p;
    let lat = &st.lat;
    let efield = electric_field(p);
    let integ = Integrator::new(p.solve_order);
    let mut rows = Vec::new();
    for n in 0..=p.nt + 1 {
        let dn: Vec<Complex> = (0..lat.nk as usize).map(|j| st.dens[j][n as usize]).collect();
        // kinetic energy
        let mut ekin = 0.0;
        for kk in 0..lat.nk as usize {
            let hk = lat.dispersion(lat.kpoints[kk], n);
            ekin += (hk * dn[kk].re) * lat.kweights[kk];
        }
        // interaction energy
        let mut eint = 0.0f64;
        for kk in 0..lat.nk as usize {
            let kp = &st.kps[kk];
            let wk = lat.kweights[kk];
            let hf = (kp.sh.block(mt(n))[0] + kp.sf.block(mt(n))[0]) * kp.rho.block(mt(n))[0];
            eint += hf.re * wk;
            eint += correlation_energy(mt(n), &kp.g, &kp.sigma, &integ, p.beta, p.h) * wk;
        }
        // local density and currents
        let mut loc = Complex::ZERO;
        for j in 0..lat.nk as usize {
            loc += Complex::new(lat.kweights[j], 0.0) * dn[j];
        }
        let mut cur_n = 0.0;
        let mut cur_p = 0.0;
        for j in 0..lat.nk as usize {
            let kk = lat.kpoints[j];
            cur_n += lat.kweights[j] * lat.velocity_nessi(kk) * dn[j].re;
            cur_p += lat.kweights[j] * lat.velocity_peierls(kk, n) * dn[j].re;
        }
        rows.push(GwRow {
            tstp: n,
            t: if n == 0 { 0.0 } else { (n - 1) as f64 * p.h },
            e: efield[n as usize],
            a: lat.afield[n as usize],
            ekin,
            eint,
            density: loc.re,
            current_nessi: cur_n,
            current_corrected: cur_p,
        });
    }
    rows
}

/// Entry point (blueprint `main`).
pub fn gw_main(cfgfile: &str) -> Result<Vec<GwRow>, String> {
    let p = read_gw_params(cfgfile)?;
    let st = solve_gw(&p)?;
    let rows = gw_observables(&st);
    let corrected = p.current_mode == "peierls_corrected";
    let mut f = std::fs::File::create(format!("{}/{}_observables.dat", p.dir, p.prefix))
        .map_err(|e| e.to_string())?;
    write!(
        f,
        "# tstp  t  E  A  Ekin  Eint  Etot  density  current_nessi{}\n",
        if corrected { "  current_peierls_corrected" } else { "" }
    )
    .map_err(|e| e.to_string())?;
    for r in &rows {
        let mut line = format!(
            "{} {} {} {} {} {} {} {} {}",
            r.tstp,
            fmt_e12(r.t),
            fmt_e12(r.e),
            fmt_e12(r.a),
            fmt_e12(r.ekin),
            fmt_e12(r.eint),
            fmt_e12(r.ekin + r.eint),
            fmt_e12(r.density),
            fmt_e12(r.current_nessi)
        );
        if corrected {
            line.push_str(&format!(" {}", fmt_e12(r.current_corrected)));
        }
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    let last = rows.last().unwrap();
    println!(
        "GW: Ekin(t_max) = {:.10}, Eint(t_max) = {:.10}, current(t_max) = {:.10}",
        last.ekin, last.eint, last.current_nessi
    );
    Ok(rows)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gw <config.toml> [--current-mode=nessi|peierls_corrected]");
        std::process::exit(1);
    }
    let mode = args.iter().find_map(|a| a.strip_prefix("--current-mode=").map(|s| s.to_string()));
    let cfg = if args[0].starts_with("--") {
        args.iter().find(|a| !a.starts_with("--")).cloned()
    } else {
        Some(args[0].clone())
    }
    .ok_or_else(|| "missing config".to_string())
    .unwrap();
    let _ = mode;
    if let Err(e) = gw_main(&cfg) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
