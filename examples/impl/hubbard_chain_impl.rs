//
// Open Hubbard chain with an on-site quench in the second Born (2B), GW,
// and particle-particle T-matrix (TPP) approximations (the Rust port of
// julia/tutor/hubbard_chain/hubbard_chain_impl.jl, from NESSi
// examples/programs/hubbard_chain_{2b,gw,tpp}.cpp).
//
// The bubbles of the library contain the factor i (bubble1: i A(t,t')
// B(t',t), bubble2: i A(t,t') B(t,t')), so
//
//   P(t,t')    = -i G(t,t') G(t',t)
//   Sigma_2B   =  i G(t,t') [U P U](t,t')
//   (1 - P U) * Chi = P,  deltaW = U Chi U
//   Sigma_GW   =  i G(t,t') deltaW(t,t')
//   Phi(t,t')  = -i G(t,t') G(t,t')
//   (1 + Phi U) * TPP = Phi
//   Sigma_TPP  =  i [U TPP U](t,t') G(t',t)
//
// The public steps follow the blueprint's KB convention (n = 0 is the
// Matsubara axis, n >= 1 the real steps); `mt()` maps them onto the
// library steps.
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::config::{
    contour_params, load_example_config, parse_example_args, solver_params, ContourParams,
    ExampleConfig,
};
use crate::common::contour_loop::{iterate_bootstrap, iterate_matsubara, LoopEvent};
use crate::common::output::{write_green_components, write_table, GreenRow};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{bubble1, bubble2};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::observables::{correlation_energy, density_matrix_green};
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::solvers::vie2_mat::vie2_mat;
use kadanoffbaym::solvers::vie2_start::vie2_start;
use kadanoffbaym::solvers::vie2_timestep::vie2_timestep;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::basics::MATSUBARA;
use kadanoffbaym::weights::Integrator;

pub const HUBBARD_APPROX: [&str; 3] = ["2B", "GW", "TPP"];
pub const HUBBARD_MODEL_KEYS: [&str; 6] =
    ["Nsites", "HoppingT", "HubbardU", "MuChem", "RampSite", "RampW0"];

/// Map a blueprint KB step onto the library step.
fn mt(n: i64) -> i64 {
    if n == 0 {
        MATSUBARA
    } else {
        n - 1
    }
}

/// The state of the Hubbard chain.
#[allow(dead_code)]
pub struct HubbardChain {
    pub approx: String,
    pub g: ContourGreen,
    pub sigma: ContourGreen,
    pub gold: ContourGreen,
    /// polarization (GW) or Phi (TPP), BOSE
    pub x: ContourGreen,
    pub xu: ContourGreen,
    pub ux: ContourGreen,
    pub w: ContourGreen,
    /// chi (GW) or T (TPP), BOSE
    pub s: ContourGreen,
    pub eps0: DenseMat,
    pub eps_mf: TimeFunction,
    pub ut: TimeFunction,
    pub mu: f64,
    pub beta: f64,
    pub h: f64,
    pub k: i64,
    pub nt: i64,
    pub integ: Integrator,
    pub cparams: ContourParams,
}

/// Copy the whole KB step `n` of `src` into `dst` (blueprint
/// `memcpy!(src, dst, n)`); a pure per-step copy, the whole-component
/// copy of `elemcpy_green_at` would clobber the other steps.
pub fn memcpy_step(src: &ContourGreen, dst: &mut ContourGreen, n: i64) {
    let t = mt(n);
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

fn id_block(st: &HubbardChain, n: i64) -> DenseMat {
    let d = st.g.size1;
    DenseMat::from_vec(d, d, st.ut.block(mt(n)).to_vec())
}

/// Construct the chain and the Hartree-shifted chemical potential
/// (blueprint `hubbard_chain`).
pub fn hubbard_chain(approx: &str, cfg: &ExampleConfig) -> Result<HubbardChain, String> {
    if !HUBBARD_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    let c = contour_params(cfg)?;
    let m = &cfg.model;
    let n_sites = m["Nsites"]
        .as_i64()
        .filter(|n| *n >= 1)
        .ok_or("Nsites must be a positive integer")?;
    let rs = m["RampSite"]
        .as_i64()
        .filter(|r| (1..=n_sites).contains(r))
        .ok_or("RampSite must be in 1:Nsites")?;
    let _ = rs;
    let u = m["HubbardU"].as_f64().unwrap();
    let n = n_sites as usize;
    let cc = Contour::new(c.nt, c.ntau, n_sites, n_sites, c.nt as f64 * c.h, c.beta);
    let mut eps0 = DenseMat::zeros(n);
    let t_hop = m["HoppingT"].as_f64().unwrap();
    for i in 0..(n - 1) {
        eps0.set(i, i + 1, Complex::new(-t_hop, 0.0));
        eps0.set(i + 1, i, Complex::new(-t_hop, 0.0));
    }
    let mut ut = TimeFunction::new(c.nt as usize, n, n, Complex::ZERO);
    let uid: Vec<Complex> = (0..(n * n))
        .map(|q| {
            if q % (n + 1) == 0 {
                Complex::new(u, 0.0)
            } else {
                Complex::ZERO
            }
        })
        .collect();
    for i in MATSUBARA..=c.nt {
        ut.block_mut(i).copy_from_slice(&uid);
    }
    let mut g = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut mu = m["MuChem"].as_f64().unwrap();
    from_ham(&mut g, mu, &eps0, c.beta, c.h);
    let rho = density_matrix_green(&g, MATSUBARA);
    let mut npart = 0.0;
    for i in 0..n {
        npart += rho.get(i, i).re;
    }
    mu += u * npart / n_sites as f64;
    let boson = |cc: &Contour| ContourGreen::new(cc, Sign::Bose, Complex::ZERO);
    Ok(HubbardChain {
        approx: approx.to_string(),
        g,
        sigma: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        gold: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        x: boson(&cc),
        xu: boson(&cc),
        ux: boson(&cc),
        w: boson(&cc),
        s: boson(&cc),
        eps0,
        eps_mf: TimeFunction::new(c.nt as usize, n, n, Complex::ZERO),
        ut,
        mu,
        beta: c.beta,
        h: c.h,
        k: c.k,
        nt: c.nt,
        integ: Integrator::new(c.k),
        cparams: c,
    })
}

/// Mean-field Hamiltonian `h_MF = h_0 + diag(U n)` at the KB step `n`
/// (blueprint `hartree_hamiltonian!`).
pub fn hartree_hamiltonian(n: i64, st: &mut HubbardChain) {
    let rho = density_matrix_green(&st.g, mt(n));
    let u = st.ut.block(mt(n));
    let d = st.eps0.nrows();
    let mut hmf = st.eps0.clone();
    for i in 0..d {
        hmf.set(i, i, hmf.get(i, i) + u[i * d + i] * Complex::new(rho.get(i, i).re, 0.0));
    }
    st.eps_mf.block_mut(mt(n)).copy_from_slice(hmf.raw());
}

/// `P(t,t') = -i G(t,t') G(t',t)` at the KB step `n`.
pub fn polarization(n: i64, g: &ContourGreen, p: &mut ContourGreen) {
    let d = g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble1(mt(n), p, i, j, g, g, i, j, g, g, i, j);
        }
    }
    p.smul_at(mt(n), Complex::new(-1.0, 0.0));
}

/// Second Born self-energy at the KB step `n` (the polarization is
/// stored in `st.w`).
pub fn sigma_2b(n: i64, st: &mut HubbardChain) {
    polarization(n, &st.g, &mut st.w);
    let um = id_block(st, n);
    st.w.rmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    st.w.lmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(
                mt(n), &mut st.sigma, i, j, &st.g, &st.g, i, j, &st.w, &st.w, i, j,
            );
        }
    }
}

/// `XU = s X U`, `UX = s U X` at the KB step `n`.
fn vie2_kernels(n: i64, st: &mut HubbardChain, s: f64) {
    // copy ONLY the KB step n of X into XU/UX; a whole-component copy
    // would overwrite the previously scaled steps of the kernel
    if n == 0 {
        st.xu.mat.memcpy_from(&st.x.mat);
        st.ux.mat.memcpy_from(&st.x.mat);
    } else {
        let t = n - 1;
        for j in 0..=t {
            let v = st.x.ret.raw_block_at(t, j).to_vec();
            st.xu.ret.raw_block_at_mut(t, j).copy_from_slice(&v);
            st.ux.ret.raw_block_at_mut(t, j).copy_from_slice(&v);
        }
        for i in 0..=t {
            let v = st.x.less.raw_block_at(i, t).to_vec();
            st.xu.less.raw_block_at_mut(i, t).copy_from_slice(&v);
            st.ux.less.raw_block_at_mut(i, t).copy_from_slice(&v);
        }
        for m in 0..=st.x.ntau as i64 {
            let v = st.x.lmix.raw_block_at(t, m).to_vec();
            st.xu.lmix.raw_block_at_mut(t, m).copy_from_slice(&v);
            st.ux.lmix.raw_block_at_mut(t, m).copy_from_slice(&v);
        }
    }
    let um = id_block(st, n);
    st.xu.rmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    st.ux.lmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    if s != 1.0 {
        st.xu.smul_at(mt(n), Complex::new(s, 0.0));
        st.ux.smul_at(mt(n), Complex::new(s, 0.0));
    }
}

/// Solve the VIE2 for `st.s` at the KB step `n`.
fn vie2_step(n: i64, st: &mut HubbardChain) {
    if n == 0 {
        vie2_mat(
            &mut st.s,
            &st.xu,
            &st.ux,
            &st.x,
            st.beta,
            &st.integ,
            MatsubaraMethod::Fixpoint,
            6,
            40,
            1e-12,
        );
    } else {
        vie2_timestep(
            n - 1,
            &mut st.s,
            &st.xu,
            &st.ux,
            &st.x,
            &st.integ,
            st.beta,
            st.h,
            MatsubaraMethod::Fixpoint,
        );
    }
}

/// `(1 - P U) Chi = P` at the KB step `n` (`n = 0` or `n > k + 1`).
pub fn gen_chi(n: i64, st: &mut HubbardChain) {
    vie2_kernels(n, st, -1.0);
    vie2_step(n, st);
}

/// Start-up block of the chi ladder (KB steps `1, .., k + 1`).
pub fn gen_chi_start(st: &mut HubbardChain) {
    for n in 1..=st.k + 1 {
        vie2_kernels(n, st, -1.0);
    }
    vie2_start(&mut st.s, &st.xu, &st.ux, &st.x, st.k, st.beta, st.h);
}

/// GW self-energy `Sigma = i G U Chi U` at the KB step `n`.
pub fn sigma_gw(n: i64, st: &mut HubbardChain) {
    memcpy_step(&st.s, &mut st.w, n);
    let um = id_block(st, n);
    st.w.rmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    st.w.lmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(
                mt(n), &mut st.sigma, i, j, &st.g, &st.g, i, j, &st.w, &st.w, i, j,
            );
        }
    }
}

/// `Phi(t,t') = -i G G` at the KB step `n`, stored in `st.x`.
fn phi(n: i64, st: &mut HubbardChain) {
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(mt(n), &mut st.x, i, j, &st.g, &st.g, i, j, &st.g, &st.g, i, j);
        }
    }
    st.x.smul_at(mt(n), Complex::new(-1.0, 0.0));
}

/// Phi and `(1 + Phi U) T = Phi` at the KB step `n`.
pub fn gen_tpp(n: i64, st: &mut HubbardChain) {
    phi(n, st);
    vie2_kernels(n, st, 1.0);
    vie2_step(n, st);
}

/// Start-up block of the T-matrix.
pub fn gen_tpp_start(st: &mut HubbardChain) {
    for n in 1..=st.k + 1 {
        phi(n, st);
        vie2_kernels(n, st, 1.0);
    }
    vie2_start(&mut st.s, &st.xu, &st.ux, &st.x, st.k, st.beta, st.h);
}

/// T-matrix self-energy `Sigma = i [U T U] G` at the KB step `n`.
pub fn sigma_tpp(n: i64, st: &mut HubbardChain) {
    memcpy_step(&st.s, &mut st.w, n);
    let um = id_block(st, n);
    st.w.rmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    st.w.lmul_at(mt(n), &um, kadanoffbaym::basics::Mode::Normal);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble1(
                mt(n), &mut st.sigma, i, j, &st.w, &st.w, i, j, &st.g, &st.g, i, j,
            );
        }
    }
}

/// Self-energy of the approximation at the KB step `n`.
pub fn update_sigma(n: i64, st: &mut HubbardChain) {
    match st.approx.as_str() {
        "2B" => sigma_2b(n, st),
        "GW" => {
            polarization(n, &st.g, &mut st.x);
            gen_chi(n, st);
            sigma_gw(n, st);
        }
        _ => {
            gen_tpp(n, st);
            sigma_tpp(n, st);
        }
    }
}

/// Self-energy of the start-up steps `1, .., k + 1`.
pub fn update_sigma_start(st: &mut HubbardChain) {
    let r = 1..=st.k + 1;
    match st.approx.as_str() {
        "2B" => {
            for n in r {
                sigma_2b(n, st);
            }
        }
        "GW" => {
            for n in r {
                polarization(n, &st.g, &mut st.x);
            }
            gen_chi_start(st);
            for n in 1..=st.k + 1 {
                sigma_gw(n, st);
            }
        }
        _ => {
            gen_tpp_start(st);
            for n in 1..=st.k + 1 {
                sigma_tpp(n, st);
            }
        }
    }
}

/// Kinetic, potential, and total energies at the KB step `n`.
pub fn hubbard_energies(n: i64, st: &HubbardChain) -> (f64, f64, f64) {
    let rho = density_matrix_green(&st.g, mt(n));
    let d = st.eps0.nrows();
    let mut ekin = 0.0;
    for i in 0..d {
        for j in 0..d {
            ekin += (st.eps0.get(i, j) * rho.get(j, i)).re;
        }
    }
    let hmf = st.eps_mf.block(mt(n));
    let mut epot = 0.0;
    for i in 0..d {
        for j in 0..d {
            epot += ((hmf[i * d + j] - st.eps0.get(i, j)) * rho.get(j, i)).re;
        }
    }
    epot *= 0.5;
    epot += correlation_energy(mt(n), &st.g, &st.sigma, &st.integ, st.beta, st.h);
    (ekin, epot, ekin + epot)
}

/// Matsubara iteration, quench, bootstrapping, and predictor/corrector
/// propagation (blueprint `solve_hubbard!`).
pub fn solve_hubbard(st: &mut HubbardChain, cfg: &ExampleConfig) -> Result<(), String> {
    let s = solver_params(cfg)?;
    let c = contour_params(cfg)?;
    let k = st.k;
    let logf = |e: &LoopEvent| {
        if e.stage == "matsubara" || e.stage == "bootstrap" {
            println!("{} iteration {}  error = {:.6e}", e.stage, e.iter, e.err);
        }
    };
    // Matsubara
    memcpy_step(&st.g, &mut st.gold, 0);
    iterate_matsubara(
        |_iter| {
            hartree_hamiltonian(0, st);
            update_sigma(0, st);
            dyson_mat(
                &mut st.g,
                &st.sigma,
                st.mu,
                &st.eps_mf,
                st.beta,
                k,
                MatsubaraMethod::Fixpoint,
                true,
                None,
                1e-12,
            );
            let err = st.g.distance_at(&st.gold, MATSUBARA);
            memcpy_step(&st.g, &mut st.gold, 0);
            err
        },
        s.mats_max_iter,
        s.mats_max_error,
        logf,
    )
    .map_err(|e| e.to_string())?;
    // Quench of the on-site energy of RampSite
    let rs = cfg.model["RampSite"].as_i64().unwrap();
    st.eps0.set(
        (rs - 1) as usize,
        (rs - 1) as usize,
        Complex::new(cfg.model["RampW0"].as_f64().unwrap(), 0.0),
    );
    // Bootstrapping
    for n in 1..=k + 1 {
        memcpy_step(&st.g, &mut st.gold, n);
    }
    iterate_bootstrap(
        |_iter| {
            for n in 1..=k + 1 {
                hartree_hamiltonian(n, st);
            }
            update_sigma_start(st);
            dyson_start(
                &mut st.g,
                st.mu,
                &st.eps_mf,
                &st.sigma,
                st.beta,
                st.h,
                k,
            );
            let mut err = 0.0;
            for n in 1..=k + 1 {
                err += st.g.distance_at(&st.gold, n - 1);
                memcpy_step(&st.g, &mut st.gold, n);
            }
            err
        },
        s.bootstrap_max_iter,
        s.bootstrap_max_error,
        3,
        logf,
    )
    .map_err(|e| e.to_string())?;
    // Propagation (the equivalent of `propagate_timesteps!`; inlined so
    // that the predictor and the corrector can share `&mut st`)
    for n in (k + 2)..=(c.nt + 1) {
        let t = (n - 1) as f64 * st.h;
        // the blueprint extrapolates the KB row n from the KB rows
        // n-1, .., n-k-1; in my steps that is my row n-1 from my rows
        // n-2, .., so the argument (the last solved my-row) is n - 2
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut st.g, &st.integ);

        logf(&LoopEvent {
            stage: "predictor",
            kb_step: n,
            nessi_step: n - 1,
            t,
            iter: -1,
            err: f64::NAN,
        });
        for iter in 0..s.corrector_steps {
            hartree_hamiltonian(n, st);
            update_sigma(n, st);
            dyson_timestep(
                n - 1,
                &mut st.g,
                st.mu,
                &st.eps_mf,
                &st.sigma,
                st.beta,
                st.h,
                k,
            );
            logf(&LoopEvent {
                stage: "corrector",
                kb_step: n,
                nessi_step: n - 1,
                t,
                iter,
                err: f64::NAN,
            });
        }
    }
    Ok(())
}

/// Occupations and energies at the real time steps, `t = (n - 1) h`.
pub fn hubbard_observables(st: &HubbardChain) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let d = st.g.size1;
    let mut occ = Vec::new();
    let mut en = Vec::new();
    for n in 1..=st.nt + 1 {
        let t = (n - 1) as f64 * st.h;
        let rho = density_matrix_green(&st.g, n - 1);
        let mut row = vec![t];
        for i in 0..d {
            row.push(rho.get(i, i).re);
        }
        occ.push(row);
        let (ekin, epot, etot) = hubbard_energies(n, st);
        en.push(vec![t, ekin, epot, etot]);
    }
    (occ, en)
}

/// The four selected Green's function components (NESSi indices).
pub fn hubbard_selected_green(st: &HubbardChain) -> Vec<GreenRow<'_>> {
    let g = &st.g;
    vec![
        GreenRow {
            component: "mat",
            i: 0,
            j: -1,
            a: 0,
            b: 0,
            value: g.mat.block(0)[0],
        },
        GreenRow {
            component: "ret",
            i: 3,
            j: 1,
            a: 0,
            b: 0,
            value: g.ret.raw_block_at(3, 1)[0],
        },
        GreenRow {
            component: "les",
            i: 1,
            j: 3,
            a: 0,
            b: 0,
            value: g.less.raw_block_at(1, 3)[0],
        },
        GreenRow {
            component: "tv",
            i: 3,
            j: 2,
            a: 0,
            b: 0,
            value: g.lmix.raw_block_at(3, 2)[0],
        },
    ]
}

/// Read and validate the TOML file of the Hubbard chain.
pub fn load_hubbard_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    load_example_config(cfgfile, profile, &HUBBARD_MODEL_KEYS, &[], &[], &[])
}

/// The result of `hubbard_main`.
#[allow(dead_code)]
pub struct HubbardResult {
    pub config: ExampleConfig,
    pub state: HubbardChain,
    pub occupation: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>,
}

/// Entry of the examples (blueprint `hubbard_main`).
pub fn hubbard_main(args: &[String], approx: &str) -> Result<HubbardResult, String> {
    let a = parse_example_args(args)?;
    let cfg = load_hubbard_config(&a.cfgfile, &a.profile)?;
    solver_params(&cfg)?;
    let mut st = hubbard_chain(approx, &cfg)?;
    if a.validate_only {
        return Ok(HubbardResult {
            config: cfg,
            state: st,
            occupation: None,
        });
    }
    solve_hubbard(&mut st, &cfg)?;
    let obs = hubbard_observables(&st);
    // output
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if std::path::Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let pre = format!(
        "{}/{}_{}",
        dir,
        cfg.output["prefix"].as_str().unwrap(),
        approx.to_lowercase()
    );
    let d = st.g.size1;
    let mut names = vec!["t".to_string()];
    for i in 0..d {
        names.push(format!("n{}", i + 1));
    }
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    write_table(&format!("{pre}_occupation.dat"), &name_refs, &obs.0)?;
    write_table(
        &format!("{pre}_energy.dat"),
        &["t", "Ekin", "Epot", "Etot"],
        &obs.1,
    )?;
    if cfg.output["save_green_components"].as_bool().unwrap() {
        write_green_components(&format!("{pre}_green.dat"), &hubbard_selected_green(&st))?;
    }
    Ok(HubbardResult {
        config: cfg,
        state: st,
        occupation: Some(obs),
    })
}
