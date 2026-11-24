// Project : KadanoffBaym
// File    : examples/gw/main.rs
// Purpose : one-dimensional Hubbard model in the GW approximation under a Peierls electric-field pulse.
//
// Usage:
//   cargo run --example gw -- examples/gw/gw.toml
//

use std::fs::File;
use std::io::Write;
use std::ops::RangeInclusive;

use std::f64::consts::PI;

use std::env::args;
use std::process::exit;

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::{Contour, ContourGreen, Sign, TimeFunction};
use kadanoffbaym::vie2::Vie2SingularWorkspace;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::types::MATSUBARA;

use kadanoffbaym::convolution::{bubble1, bubble2};
use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::{from_ham, from_mat};
use kadanoffbaym::math::extrapolate_timestep_of_order;
use kadanoffbaym::observables::{correlation_energy, density};
use kadanoffbaym::vie2::vie2_timestep_singular;

use util::{
    add_green_step, fmt_e12, get_f64, get_i64, get_str, load_toml_config, matsubara_step,
    ness_outdir, zero_green_step,
};

#[path = "../../util/mod.rs"]
mod util;

/// The parameters of the GW example.
#[derive(Clone)]
pub(crate) struct GwParams {
    pub(crate) nk: i64,
    pub(crate) hopping_t: f64,
    pub(crate) hubbard_u: f64,
    pub(crate) v: f64,
    pub(crate) mu_chem: f64,
    pub(crate) beta: f64,
    pub(crate) nt: i64,
    pub(crate) ntau: i64,
    pub(crate) h: f64,
    pub(crate) solve_order: i64,
    pub(crate) mats_max_iter: i64,
    pub(crate) mats_max_err: f64,
    pub(crate) bootstrap_max_iter: i64,
    pub(crate) bootstrap_max_err: f64,
    pub(crate) time_max_err: f64,
    pub(crate) corrector_steps: i64,
    pub(crate) current_mode: String,
    pub(crate) e0: f64,
    pub(crate) omega: f64,
    pub(crate) t0: f64,
    pub(crate) width: f64,
    pub(crate) dir: String,
    pub(crate) prefix: String,
}

/// The one-dimensional lattice.
pub(crate) struct GwLattice {
    pub(crate) nk: i64,
    pub(crate) g: i64,
    pub(crate) kpoints: Vec<f64>,
    pub(crate) kweights: Vec<f64>,
    pub(crate) th: f64,
    pub(crate) u: f64,
    pub(crate) v: f64,
    pub(crate) afield: Vec<f64>,
}

impl GwLattice {
    fn new(nk: i64, th: f64, u: f64, v: f64, e: &[f64], h: f64, k: i64) -> Self {
        let (g, kp, kw) = momentum_grid(nk);
        GwLattice {
            nk,
            g,
            kpoints: kp,
            kweights: kw,
            th,
            u,
            v,
            afield: electric_field_to_vector_potential(e, h, k),
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

    fn velocity_plain(&self, k: f64) -> f64 {
        2.0 * self.th * k.sin()
    }

    fn velocity_peierls(&self, k: f64, n: i64) -> f64 {
        2.0 * self.th * (k - self.afield[n as usize]).sin()
    }

    fn interaction_vertex(&self, q: f64) -> f64 {
        self.u + 2.0 * self.v * q.cos()
    }
}

/// One k point of the GW example.
pub(crate) struct GwKPoint {
    pub(crate) k: f64,
    pub(crate) g: ContourGreen,
    pub(crate) sigma: ContourGreen,
    pub(crate) p: ContourGreen,
    pub(crate) pv: ContourGreen,
    pub(crate) vp: ContourGreen,
    pub(crate) w: ContourGreen,
    pub(crate) wsin: TimeFunction,
    pub(crate) sh: TimeFunction,
    pub(crate) sf: TimeFunction,
    pub(crate) rho: TimeFunction,
    pub(crate) hk: TimeFunction,
    pub(crate) hkeff: TimeFunction,
    pub(crate) vertex: TimeFunction,
    pub(crate) ws: Vie2SingularWorkspace,
}

impl GwKPoint {
    fn new(cc: &Contour, nt: i64, lat: &GwLattice, j: i64) -> Self {
        let k = lat.kpoints[j as usize];
        let mut hk = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        let mut vertex = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        for kb in 0..=nt + 1 {
            // dispersion takes the KB time step; the vertex is constant
            hk.block_mut(matsubara_step(kb))[0] = Complex::new(lat.dispersion(k, kb), 0.0);
            vertex.block_mut(matsubara_step(kb))[0] = Complex::new(lat.interaction_vertex(k), 0.0);
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
            ws: Vie2SingularWorkspace::new(&w),
        }
    }
}

/// The GW state.
pub(crate) struct GwState {
    pub(crate) p: GwParams,
    pub(crate) lat: GwLattice,
    pub(crate) kps: Vec<GwKPoint>,
    pub(crate) dens: Vec<Vec<Complex>>,
    pub(crate) q: ContourGreen,
    pub(crate) fsing: TimeFunction,
}

/// One observables row.
pub(crate) struct GwRow {
    pub(crate) tstp: i64,
    pub(crate) t: f64,
    pub(crate) e: f64,
    pub(crate) a: f64,
    pub(crate) ekin: f64,
    pub(crate) eint: f64,
    pub(crate) density: f64,
    pub(crate) current: f64,
    pub(crate) current_corrected: f64,
}

fn load_params(cfgfile: &str) -> Result<GwParams, String> {
    let p = load_toml_config(cfgfile)?;
    let q = GwParams {
        nk: get_i64(&p, "model", "Nk")?,
        hopping_t: get_f64(&p, "model", "HoppingT")?,
        hubbard_u: get_f64(&p, "model", "HubbardU")?,
        v: get_f64(&p, "model", "V")?,
        mu_chem: get_f64(&p, "model", "MuChem")?,
        beta: get_f64(&p, "model", "beta")?,
        nt: get_i64(&p, "contour", "Nt")?,
        // the TOML counts the imaginary time POINTS (KB convention);
        // the library counts the intervals
        ntau: get_i64(&p, "contour", "Ntau")? - 1,
        h: get_f64(&p, "contour", "h")?,
        solve_order: get_i64(&p, "solver", "SolveOrder")?,
        mats_max_iter: get_i64(&p, "solver", "MatsMaxIter")?,
        mats_max_err: get_f64(&p, "solver", "MatsMaxErr")?,
        bootstrap_max_iter: get_i64(&p, "solver", "BootstrapMaxIter")?,
        bootstrap_max_err: get_f64(&p, "solver", "BootstrapMaxErr")?,
        time_max_err: get_f64(&p, "solver", "TimeMaxErr")?,
        corrector_steps: get_i64(&p, "solver", "CorrectorSteps")?,
        current_mode: {
            let m = get_str(&p, "solver", "current_mode")?;
            if m != "plain" && m != "peierls_corrected" {
                return Err(format!(
                    "{cfgfile}: current_mode must be plain or peierls_corrected"
                ));
            }
            m
        },
        e0: get_f64(&p, "pulse", "E0")?,
        omega: get_f64(&p, "pulse", "omega")?,
        t0: get_f64(&p, "pulse", "t0")?,
        width: get_f64(&p, "pulse", "width")?,
        dir: String::new(),
        prefix: get_str(&p, "output", "prefix")?,
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
    let dir = ness_outdir(&p, cfgfile)?;
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

/// `A(t) = -int E dt` by Gregory integration of the order `min(k, Nt)`
/// .
fn electric_field_to_vector_potential(e: &[f64], h: f64, k: i64) -> Vec<f64> {
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

fn momentum_grid(nk: i64) -> (i64, Vec<f64>, Vec<f64>) {
    let g = nk / 2;
    let kp = (0..nk)
        .map(|j| (j - g) as f64 * 2.0 * PI / nk as f64)
        .collect();
    (g, kp, vec![1.0 / nk as f64; nk as usize])
}

/// Hartree self-energy (the same for all k points).
fn sigma_hartree(lat: &GwLattice, dens: &[Complex]) -> Complex {
    let vloc = lat.u;
    let vnl = lat.interaction_vertex(lat.kpoints[lat.g as usize]) - vloc;
    let mut s = Complex::ZERO;
    for (wk, rho) in lat.kweights.iter().zip(dens.iter()).take(lat.nk as usize) {
        s += Complex::new(vloc * rho.re * *wk, 0.0);
        s += Complex::new(vnl * rho.re * *wk * 2.0, 0.0);
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

/// Time steps touched at the KB step `n` (the start-up steps are solved
/// together).
fn step_range(n: i64, order: i64) -> RangeInclusive<i64> {
    if n == 0 || n > order + 1 {
        n..=n
    } else {
        1..=order + 1
    }
}

/// Solve `W = V + V P W` as the VIE2 with singular kernel
/// `[1 - V P] W = V` .
fn screened_interaction_step(
    kp: &mut GwKPoint,
    n: i64,
    order: i64,
    beta: f64,
    h: f64,
    fsing: &TimeFunction,
    q: &ContourGreen,
) {
    for m in step_range(n, order) {
        // PV = -(P vertex), VP = -(vertex P) at the step m
        kp.pv.memcpy_from_at(matsubara_step(m), &kp.p);
        let vx = kp.vertex.block(matsubara_step(m))[0];
        kp.pv
            .smul_at(matsubara_step(m), vx * Complex::new(-1.0, 0.0));
        kp.vp.memcpy_from_at(matsubara_step(m), &kp.p);
        kp.vp
            .smul_at(matsubara_step(m), vx * Complex::new(-1.0, 0.0));
    }
    vie2_timestep_singular(
        matsubara_step(n),
        &mut kp.w,
        &mut kp.wsin,
        &kp.vp,
        &kp.pv,
        fsing,
        q,
        &kp.vertex,
        &Integrator::new(order),
        beta,
        h,
        MatsubaraMethod::Fixpoint,
        &mut kp.ws,
    );
}

/// Solve the Dyson equation and store the densities; the second
/// Matsubara iteration uses the noninteracting Green's function of
/// `hkeff`.
#[allow(clippy::too_many_arguments)]
fn dyson_step(
    kp: &mut GwKPoint,
    n: i64,
    iter: i64,
    lat: &GwLattice,
    mu: f64,
    beta: f64,
    h: f64,
    order: i64,
    integ: &Integrator,
    dws: &mut DysonWorkspace,
) {
    let r = step_range(n, order);
    let m0 = *r.start();
    let m1 = *r.end();
    for m in r {
        kp.hk.block_mut(matsubara_step(m))[0] = Complex::new(lat.dispersion(kp.k, m), 0.0);
        kp.hkeff.block_mut(matsubara_step(m))[0] = kp.sh.block(matsubara_step(m))[0]
            + kp.sf.block(matsubara_step(m))[0]
            + kp.hk.block(matsubara_step(m))[0];
    }
    if n == 0 && iter == 1 {
        let cc1 = Contour::new(1, kp.g.ntau as i64, 1, 1, h, beta);
        let mut t = ContourGreen::new(&cc1, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, kp.hkeff.block(MATSUBARA)[0]);
        from_ham(&mut t, mu, &h0, beta, h);
        kp.g.mat.memcpy_from(&t.mat);
    } else if n == 0 {
        dyson_mat(
            &mut kp.g,
            &kp.sigma,
            mu,
            &kp.hkeff,
            beta,
            order,
            MatsubaraMethod::Fixpoint,
            true,
            None,
            1e-12,
        );
    } else if n <= order + 1 {
        dyson_start(&mut kp.g, mu, &kp.hkeff, &kp.sigma, beta, h, order);
    } else {
        dyson_timestep_ws(
            matsubara_step(n),
            &mut kp.g,
            mu,
            &kp.hkeff,
            &kp.sigma,
            beta,
            h,
            integ,
            dws,
        );
    }
    for m in m0..=m1 {
        kp.rho.block_mut(matsubara_step(m))[0] = density(&kp.g, matsubara_step(m));
    }
}

/// One iteration at the KB step `n` .
fn gw_iteration(n: i64, iter: i64, st: &mut GwState) -> (f64, f64) {
    let (nk, k) = (st.lat.nk, st.p.solve_order);
    let nmu = nk as usize;
    // the density matrix is hermitized
    for j in 0..nmu {
        let rho = density(&st.kps[j].g, matsubara_step(n));
        st.dens[j][n as usize] = Complex::new(rho.re, 0.0);
    }
    let dn: Vec<Complex> = (0..nmu).map(|j| st.dens[j][n as usize]).collect();
    let sh = sigma_hartree(&st.lat, &dn);
    for j in 0..nmu {
        st.kps[j].sh.block_mut(matsubara_step(n))[0] = sh;
        st.kps[j].sf.block_mut(matsubara_step(n))[0] = sigma_fock(j as i64, &st.lat, &dn);
        // GW self-energy: Sigma_k = sum_q w_q i G_{k-q} W_q (Bubble2);
        // a local work object avoids aliasing kps[j] with kps[kq]
        let mut acc = st.kps[j].sigma.clone();
        zero_green_step(&mut acc, matsubara_step(n));
        let mut work = st.kps[j].sigma.clone();
        for q in 0..st.lat.nk {
            let kq = st.lat.add_kpoints(j as i64, 1, q, -1);
            zero_green_step(&mut work, matsubara_step(n));
            bubble2(
                matsubara_step(n),
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
            add_green_step(
                &mut acc,
                &work,
                matsubara_step(n),
                Complex::new(st.lat.kweights[q as usize], 0.0),
            );
        }
        st.kps[j].sigma.memcpy_from_at(matsubara_step(n), &acc);
    }
    for q in 0..st.lat.nk {
        // polarization: P_q = -sum_k w_k i G_{k-q} G_k (Bubble1)
        let mut acc = st.kps[q as usize].p.clone();
        zero_green_step(&mut acc, matsubara_step(n));
        let mut work = st.kps[q as usize].p.clone();
        for kk in 0..st.lat.nk {
            let kq = st.lat.add_kpoints(kk, 1, q, -1);
            zero_green_step(&mut work, matsubara_step(n));
            bubble1(
                matsubara_step(n),
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
            add_green_step(
                &mut acc,
                &work,
                matsubara_step(n),
                Complex::new(-st.lat.kweights[kk as usize], 0.0),
            );
        }
        st.kps[q as usize].p.memcpy_from_at(matsubara_step(n), &acc);
    }
    let mut e1 = 0.0f64;
    let mut e2 = 0.0f64;
    let integ = Integrator::new(k);
    let mut dws = DysonWorkspace::new(k, st.kps[0].g.size1, st.kps[0].g.ntau);
    for j in 0..nmu {
        // the Dyson step with the error (each k point carries its own
        // sigma, hence the reset of the Matsubara cache)
        let mut gsave = st.kps[j].g.clone();
        dws.reset();
        gsave.memcpy_from_at(matsubara_step(n), &st.kps[j].g);
        dyson_step(
            &mut st.kps[j],
            n,
            iter,
            &st.lat,
            st.p.mu_chem,
            st.p.beta,
            st.p.h,
            k,
            &integ,
            &mut dws,
        );
        e1 += gsave.distance_at(&st.kps[j].g, matsubara_step(n));
    }
    for j in 0..nmu {
        let mut wsave = st.kps[j].w.clone();
        wsave.memcpy_from_at(matsubara_step(n), &st.kps[j].w);
        screened_interaction_step(&mut st.kps[j], n, k, st.p.beta, st.p.h, &st.fsing, &st.q);
        e2 += wsave.distance_at(&st.kps[j].w, matsubara_step(n));
    }
    (e1, e2)
}

/// Matsubara iterations, bootstrapping, and time stepping.
pub(crate) fn solve_gw(p: &GwParams) -> Result<GwState, String> {
    let ntime = p.nt + 1;
    let k = p.solve_order;
    let cc = Contour::new(p.nt, p.ntau, 1, 1, p.nt as f64 * p.h, p.beta);
    let lat = GwLattice::new(
        p.nk,
        p.hopping_t,
        p.hubbard_u,
        p.v,
        &electric_field(p),
        p.h,
        k,
    );
    let kps: Vec<GwKPoint> = (0..p.nk)
        .map(|j| GwKPoint::new(&cc, p.nt, &lat, j))
        .collect();
    let mut st = GwState {
        p: GwParams {
            dir: p.dir.clone(),
            ..(*p).clone()
        },
        lat,
        kps,
        dens: vec![vec![Complex::ZERO; (ntime + 1) as usize]; p.nk as usize],
        q: ContourGreen::new(&cc, Sign::Bose, Complex::ZERO),
        fsing: TimeFunction::new(p.nt as usize, 1, 1, Complex::ZERO),
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
                from_mat(&mut kp.g);
                from_mat(&mut kp.w);
            } else {
                let ord = k.min(tb - 2);
                extrapolate_timestep_of_order(tb - 2, &mut kp.g, ord);
                extrapolate_timestep_of_order(tb - 2, &mut kp.w, ord);
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
            extrapolate_timestep_of_order(tb - 2, &mut kp.g, k);
            extrapolate_timestep_of_order(tb - 2, &mut kp.w, k);
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

/// Observables at the KB time steps .
pub(crate) fn gw_observables(st: &GwState) -> Vec<GwRow> {
    let p = &st.p;
    let lat = &st.lat;
    let efield = electric_field(p);
    let integ = Integrator::new(p.solve_order);
    let mut rows = Vec::new();
    for n in 0..=p.nt + 1 {
        let dn: Vec<Complex> = (0..lat.nk as usize)
            .map(|j| st.dens[j][n as usize])
            .collect();
        // kinetic energy
        let mut ekin = 0.0;
        for ((kp, dnk), wk) in lat
            .kpoints
            .iter()
            .zip(dn.iter())
            .zip(lat.kweights.iter())
            .take(lat.nk as usize)
        {
            let hk = lat.dispersion(*kp, n);
            ekin += (hk * dnk.re) * *wk;
        }
        // interaction energy
        let mut eint = 0.0f64;
        for kk in 0..lat.nk as usize {
            let kp = &st.kps[kk];
            let wk = lat.kweights[kk];
            let hf = (kp.sh.block(matsubara_step(n))[0] + kp.sf.block(matsubara_step(n))[0])
                * kp.rho.block(matsubara_step(n))[0];
            eint += hf.re * wk;
            eint +=
                correlation_energy(matsubara_step(n), &kp.g, &kp.sigma, &integ, p.beta, p.h) * wk;
        }
        // local density and currents
        let mut loc = Complex::ZERO;
        for (wk, dnj) in lat.kweights.iter().zip(dn.iter()).take(lat.nk as usize) {
            loc += Complex::new(*wk, 0.0) * *dnj;
        }
        let mut cur_n = 0.0;
        let mut cur_p = 0.0;
        for ((kk, wk), dnj) in lat
            .kpoints
            .iter()
            .zip(lat.kweights.iter())
            .zip(dn.iter())
            .take(lat.nk as usize)
        {
            cur_n += wk * lat.velocity_plain(*kk) * dnj.re;
            cur_p += wk * lat.velocity_peierls(*kk, n) * dnj.re;
        }
        rows.push(GwRow {
            tstp: n,
            t: if n == 0 { 0.0 } else { (n - 1) as f64 * p.h },
            e: efield[n as usize],
            a: lat.afield[n as usize],
            ekin,
            eint,
            density: loc.re,
            current: cur_n,
            current_corrected: cur_p,
        });
    }
    rows
}

/// Entry point .
pub(crate) fn run(cfgfile: &str) -> Result<Vec<GwRow>, String> {
    let p = load_params(cfgfile)?;
    let st = solve_gw(&p)?;
    let rows = gw_observables(&st);
    let corrected = p.current_mode == "peierls_corrected";
    let mut f = File::create(format!("{}/{}_observables.dat", p.dir, p.prefix))
        .map_err(|e| e.to_string())?;
    writeln!(
        f,
        "# tstp  t  E  A  Ekin  Eint  Etot  density  current{}",
        if corrected {
            "  current_peierls_corrected"
        } else {
            ""
        }
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
            fmt_e12(r.current)
        );
        if corrected {
            line.push_str(&format!(" {}", fmt_e12(r.current_corrected)));
        }
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    let last = rows.last().unwrap();
    println!(
        "GW: Ekin(t_max) = {:.10}, Eint(t_max) = {:.10}, current(t_max) = {:.10}",
        last.ekin, last.eint, last.current
    );
    Ok(rows)
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gw <config.toml> ");
        exit(1);
    }
    let cfg = if args[0].starts_with("--") {
        args.iter().find(|a| !a.starts_with("--")).cloned()
    } else {
        Some(args[0].clone())
    }
    .ok_or_else(|| "missing config".to_string())
    .unwrap();
    if let Err(e) = run(&cfg) {
        eprintln!("{e}");
        exit(1);
    }
}
