// Project : KadanoffBaym
// File    : examples/hubbard/chain.rs
// Purpose : open Hubbard chain in the second Born, GW, and particle-particle T-matrix approximations.
//

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::linalg::Adjoint;
use kadanoffbaym::vie2::Vie2Workspace;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{bubble1, bubble2};
use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::math::extrapolate_timestep;
use kadanoffbaym::observables::{correlation_energy, density, density_matrix};
use kadanoffbaym::vie2::{vie2_mat, vie2_start, vie2_timestep_ws};

use super::util::{ExampleConfig, GreenRow, LoopEvent};

use super::util::{
    contour_params, example_outdir, iterate_bootstrap, iterate_matsubara, load_example_config,
    matsubara_step, parse_example_args, solver_params, write_green_components, write_table,
};

use super::util::prelude::*;

pub(crate) const APPROXIMATIONS: [&str; 3] = ["2B", "GW", "TPP"];

pub(crate) const MODEL_KEYS: [&str; 6] = [
    "Nsites", "HoppingT", "HubbardU", "MuChem", "RampSite", "RampW0",
];

/// The state of the Hubbard chain.
pub(crate) struct ChainState {
    pub(crate) approx: String,
    pub(crate) g: ContourGreen,
    pub(crate) sigma: ContourGreen,
    pub(crate) g_old: ContourGreen,
    /// polarization (GW) or Phi (TPP), BOSE
    pub(crate) polarization: ContourGreen,
    pub(crate) polarization_u: ContourGreen,
    pub(crate) u_polarization: ContourGreen,
    pub(crate) screened: ContourGreen,
    /// chi (GW) or T (TPP), BOSE
    pub(crate) susceptibility: ContourGreen,
    pub(crate) eps0: DenseMat,
    pub(crate) mean_field: TimeFunction,
    pub(crate) interaction: TimeFunction,
    pub(crate) mu: f64,
    pub(crate) beta: f64,
    pub(crate) h: f64,
    pub(crate) k: i64,
    pub(crate) nt: i64,
    pub(crate) integrator: Integrator,
    /// VIE2 scratch, one instance for the whole evolution
    pub(crate) susceptibility_workspace: Vie2Workspace,
}

fn interaction_block(st: &ChainState, n: i64) -> DenseMat {
    let d = st.g.size1;
    DenseMat::from_vec(d, d, st.interaction.block(matsubara_step(n)).to_vec())
}

/// Construct the chain and the Hartree-shifted chemical potential
/// .
pub(crate) fn build_chain(approx: &str, cfg: &ExampleConfig) -> Result<ChainState, String> {
    if !APPROXIMATIONS.contains(&approx) {
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
    let npart = density(&g, MATSUBARA).re;
    mu += u * npart / n_sites as f64;
    let boson = |cc: &Contour| ContourGreen::new(cc, Sign::Bose, Complex::ZERO);
    Ok(ChainState {
        approx: approx.to_string(),
        g,
        sigma: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        g_old: ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO),
        polarization: boson(&cc),
        polarization_u: boson(&cc),
        u_polarization: boson(&cc),
        screened: boson(&cc),
        susceptibility: boson(&cc),
        eps0,
        mean_field: TimeFunction::new(c.nt as usize, n, n, Complex::ZERO),
        interaction: ut,
        mu,
        beta: c.beta,
        h: c.h,
        k: c.k,
        nt: c.nt,
        integrator: Integrator::new(c.k),
        susceptibility_workspace: Vie2Workspace::new(c.k, n_sites as usize, c.ntau as usize),
    })
}

/// Mean-field Hamiltonian `h_MF = h_0 + diag(U n)` at the KB step `n`
/// .
pub(crate) fn hartree_hamiltonian(n: i64, st: &mut ChainState) {
    let rho = density_matrix(&st.g, matsubara_step(n));
    let u = st.interaction.block(matsubara_step(n));
    let d = st.eps0.nrows();
    let mut hmf = st.eps0.clone();
    for i in 0..d {
        hmf.set(
            i,
            i,
            hmf.get(i, i) + u[i * d + i] * Complex::new(rho.get(i, i).re, 0.0),
        );
    }
    st.mean_field
        .block_mut(matsubara_step(n))
        .copy_from_slice(hmf.raw());
}

/// `P(t,t') = -i G(t,t') G(t',t)` at the KB step `n`.
pub(crate) fn polarization(n: i64, g: &ContourGreen, p: &mut ContourGreen) {
    let d = g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble1(matsubara_step(n), p, i, j, g, g, i, j, g, g, i, j);
        }
    }
    p.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
}

/// Second Born self-energy at the KB step `n` (the polarization is
/// stored in `st.screened`).
pub(crate) fn second_born_self_energy(n: i64, st: &mut ChainState) {
    polarization(n, &st.g, &mut st.screened);
    let um = interaction_block(st, n);
    st.screened.rmul_at(matsubara_step(n), &um, Adjoint::No);
    st.screened.lmul_at(matsubara_step(n), &um, Adjoint::No);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(
                matsubara_step(n),
                &mut st.sigma,
                i,
                j,
                &st.g,
                &st.g,
                i,
                j,
                &st.screened,
                &st.screened,
                i,
                j,
            );
        }
    }
}

/// `XU = s X U`, `UX = s U X` at the KB step `n`.
fn vie2_kernels(n: i64, st: &mut ChainState, s: f64) {
    // copy ONLY the KB step n of X into XU/UX; a whole-component copy
    // would overwrite the previously scaled steps of the kernel
    if n == 0 {
        st.polarization_u.mat.memcpy_from(&st.polarization.mat);
        st.u_polarization.mat.memcpy_from(&st.polarization.mat);
    } else {
        let t = n - 1;
        for j in 0..=t {
            let v = st.polarization.ret.raw_block_at(t, j).to_vec();
            st.polarization_u
                .ret
                .raw_block_at_mut(t, j)
                .copy_from_slice(&v);
            st.u_polarization
                .ret
                .raw_block_at_mut(t, j)
                .copy_from_slice(&v);
        }
        for i in 0..=t {
            let v = st.polarization.less.raw_block_at(i, t).to_vec();
            st.polarization_u
                .less
                .raw_block_at_mut(i, t)
                .copy_from_slice(&v);
            st.u_polarization
                .less
                .raw_block_at_mut(i, t)
                .copy_from_slice(&v);
        }
        for m in 0..=st.polarization.ntau as i64 {
            let v = st.polarization.lmix.raw_block_at(t, m).to_vec();
            st.polarization_u
                .lmix
                .raw_block_at_mut(t, m)
                .copy_from_slice(&v);
            st.u_polarization
                .lmix
                .raw_block_at_mut(t, m)
                .copy_from_slice(&v);
        }
    }
    let um = interaction_block(st, n);
    st.polarization_u
        .rmul_at(matsubara_step(n), &um, Adjoint::No);
    st.u_polarization
        .lmul_at(matsubara_step(n), &um, Adjoint::No);
    if s != 1.0 {
        st.polarization_u
            .smul_at(matsubara_step(n), Complex::new(s, 0.0));
        st.u_polarization
            .smul_at(matsubara_step(n), Complex::new(s, 0.0));
    }
}

/// Solve the VIE2 for `st.susceptibility` at the KB step `n`.
fn vie2_step(n: i64, st: &mut ChainState) {
    if n == 0 {
        vie2_mat(
            &mut st.susceptibility,
            &st.polarization_u,
            &st.u_polarization,
            &st.polarization,
            st.beta,
            &st.integrator,
            MatsubaraMethod::Fixpoint,
            6,
            40,
            1e-12,
        );
    } else {
        // the kernel Matsubara blocks are refreshed outside, hence
        // the reset
        st.susceptibility_workspace.reset();
        vie2_timestep_ws(
            n - 1,
            &mut st.susceptibility,
            &st.polarization_u,
            &st.u_polarization,
            &st.polarization,
            &st.integrator,
            st.beta,
            st.h,
            &mut st.susceptibility_workspace,
        )
    }
}

/// `(1 - P U) Chi = P` at the KB step `n` (`n = 0` or `n > k + 1`).
pub(crate) fn polarization_kernel(n: i64, st: &mut ChainState) {
    vie2_kernels(n, st, -1.0);
    vie2_step(n, st);
}

/// Start-up block of the chi ladder (KB steps `1, .., k + 1`).
pub(crate) fn polarization_start(st: &mut ChainState) {
    for n in 1..=st.k + 1 {
        vie2_kernels(n, st, -1.0);
    }
    vie2_start(
        &mut st.susceptibility,
        &st.polarization_u,
        &st.u_polarization,
        &st.polarization,
        st.k,
        st.beta,
        st.h,
    );
}

/// GW self-energy `Sigma = i G U Chi U` at the KB step `n`.
pub(crate) fn gw_self_energy(n: i64, st: &mut ChainState) {
    st.screened
        .memcpy_from_at(matsubara_step(n), &st.susceptibility);
    let um = interaction_block(st, n);
    st.screened.rmul_at(matsubara_step(n), &um, Adjoint::No);
    st.screened.lmul_at(matsubara_step(n), &um, Adjoint::No);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(
                matsubara_step(n),
                &mut st.sigma,
                i,
                j,
                &st.g,
                &st.g,
                i,
                j,
                &st.screened,
                &st.screened,
                i,
                j,
            );
        }
    }
}

/// `Phi(t,t') = -i G G` at the KB step `n`, stored in `st.polarization`.
fn phi(n: i64, st: &mut ChainState) {
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble2(
                matsubara_step(n),
                &mut st.polarization,
                i,
                j,
                &st.g,
                &st.g,
                i,
                j,
                &st.g,
                &st.g,
                i,
                j,
            );
        }
    }
    st.polarization
        .smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
}

/// Phi and `(1 + Phi U) T = Phi` at the KB step `n`.
pub(crate) fn tpp_kernel(n: i64, st: &mut ChainState) {
    phi(n, st);
    vie2_kernels(n, st, 1.0);
    vie2_step(n, st);
}

/// Start-up block of the T-matrix.
pub(crate) fn tpp_start(st: &mut ChainState) {
    for n in 1..=st.k + 1 {
        phi(n, st);
        vie2_kernels(n, st, 1.0);
    }
    vie2_start(
        &mut st.susceptibility,
        &st.polarization_u,
        &st.u_polarization,
        &st.polarization,
        st.k,
        st.beta,
        st.h,
    );
}

/// T-matrix self-energy `Sigma = i [U T U] G` at the KB step `n`.
pub(crate) fn tpp_self_energy(n: i64, st: &mut ChainState) {
    st.screened
        .memcpy_from_at(matsubara_step(n), &st.susceptibility);
    let um = interaction_block(st, n);
    st.screened.rmul_at(matsubara_step(n), &um, Adjoint::No);
    st.screened.lmul_at(matsubara_step(n), &um, Adjoint::No);
    let d = st.g.size1;
    for i in 0..d {
        for j in 0..d {
            bubble1(
                matsubara_step(n),
                &mut st.sigma,
                i,
                j,
                &st.screened,
                &st.screened,
                i,
                j,
                &st.g,
                &st.g,
                i,
                j,
            );
        }
    }
}

/// Self-energy of the approximation at the KB step `n`.
pub(crate) fn update_sigma(n: i64, st: &mut ChainState) {
    match st.approx.as_str() {
        "2B" => second_born_self_energy(n, st),
        "GW" => {
            polarization(n, &st.g, &mut st.polarization);
            polarization_kernel(n, st);
            gw_self_energy(n, st);
        }
        _ => {
            tpp_kernel(n, st);
            tpp_self_energy(n, st);
        }
    }
}

/// Self-energy of the start-up steps `1, .., k + 1`.
pub(crate) fn update_sigma_start(st: &mut ChainState) {
    let r = 1..=st.k + 1;
    match st.approx.as_str() {
        "2B" => {
            for n in r {
                second_born_self_energy(n, st);
            }
        }
        "GW" => {
            for n in r {
                polarization(n, &st.g, &mut st.polarization);
            }
            polarization_start(st);
            for n in 1..=st.k + 1 {
                gw_self_energy(n, st);
            }
        }
        _ => {
            tpp_start(st);
            for n in 1..=st.k + 1 {
                tpp_self_energy(n, st);
            }
        }
    }
}

/// Kinetic, potential, and total energies at the KB step `n`.
pub(crate) fn energies(n: i64, st: &ChainState) -> (f64, f64, f64) {
    let rho = density_matrix(&st.g, matsubara_step(n));
    let d = st.eps0.nrows();
    let mut ekin = 0.0;
    for i in 0..d {
        for j in 0..d {
            ekin += (st.eps0.get(i, j) * rho.get(j, i)).re;
        }
    }
    let hmf = st.mean_field.block(matsubara_step(n));
    let mut epot = 0.0;
    for i in 0..d {
        for j in 0..d {
            epot += ((hmf[i * d + j] - st.eps0.get(i, j)) * rho.get(j, i)).re;
        }
    }
    epot *= 0.5;
    epot += correlation_energy(
        matsubara_step(n),
        &st.g,
        &st.sigma,
        &st.integrator,
        st.beta,
        st.h,
    );
    (ekin, epot, ekin + epot)
}

/// Matsubara iteration, quench, bootstrapping, and predictor/corrector
/// propagation .
pub(crate) fn solve(st: &mut ChainState, cfg: &ExampleConfig) -> Result<(), String> {
    let s = solver_params(cfg)?;
    let c = contour_params(cfg)?;
    let k = st.k;
    let logf = |e: &LoopEvent| {
        if e.stage == "matsubara" || e.stage == "bootstrap" {
            println!("{} iteration {}  error = {:.6e}", e.stage, e.iter, e.err);
        }
    };
    // Matsubara
    st.g_old.memcpy_from_at(matsubara_step(0), &st.g);
    iterate_matsubara(
        |_iter| {
            hartree_hamiltonian(0, st);
            update_sigma(0, st);
            dyson_mat(
                &mut st.g,
                &st.sigma,
                st.mu,
                &st.mean_field,
                st.beta,
                k,
                MatsubaraMethod::Fixpoint,
                true,
                None,
                1e-12,
            );
            let err = st.g.distance_at(&st.g_old, MATSUBARA);
            st.g_old.memcpy_from_at(matsubara_step(0), &st.g);
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
        st.g_old.memcpy_from_at(matsubara_step(n), &st.g);
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
                &st.mean_field,
                &st.sigma,
                st.beta,
                st.h,
                k,
            );
            let mut err = 0.0;
            for n in 1..=k + 1 {
                err += st.g.distance_at(&st.g_old, n - 1);
                st.g_old.memcpy_from_at(matsubara_step(n), &st.g);
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
        // the extrapolates the KB row n from the KB rows
        // n-1, .., n-k-1; in my steps that is my row n-1 from my rows
        // n-2, .., so the argument (the last solved my-row) is n - 2
        extrapolate_timestep(n - 2, &mut st.g, &st.integrator);

        logf(&LoopEvent {
            stage: "predictor",
            kb_step: n,
            step: n - 1,
            t,
            iter: -1,
            err: f64::NAN,
        });
        let integ = Integrator::new(k);
        let mut dws = DysonWorkspace::new(k, st.g.size1, st.g.ntau);
        for iter in 0..s.corrector_steps {
            hartree_hamiltonian(n, st);
            update_sigma(n, st);
            dws.reset();
            dyson_timestep_ws(
                n - 1,
                &mut st.g,
                st.mu,
                &st.mean_field,
                &st.sigma,
                st.beta,
                st.h,
                &integ,
                &mut dws,
            );
            logf(&LoopEvent {
                stage: "corrector",
                kb_step: n,
                step: n - 1,
                t,
                iter,
                err: f64::NAN,
            });
        }
    }
    Ok(())
}

/// Occupations and energies at the real time steps, `t = (n - 1) h`.
pub(crate) fn observables(st: &ChainState) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let d = st.g.size1;
    let mut occ = Vec::new();
    let mut en = Vec::new();
    for n in 1..=st.nt + 1 {
        let t = (n - 1) as f64 * st.h;
        let rho = density_matrix(&st.g, n - 1);
        let mut row = vec![t];
        for i in 0..d {
            row.push(rho.get(i, i).re);
        }
        occ.push(row);
        let (ekin, epot, etot) = energies(n, st);
        en.push(vec![t, ekin, epot, etot]);
    }
    (occ, en)
}

/// The four selected Green's function components .
pub(crate) fn selected_green(st: &ChainState) -> Vec<GreenRow<'_>> {
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
pub(crate) fn load_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    load_example_config(cfgfile, profile, &MODEL_KEYS, &[], &[], &[])
}

/// Entry of the chain examples.
pub(crate) fn run(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_config(&a.cfgfile, &a.profile)?;
    solver_params(&cfg)?;
    let mut st = build_chain(approx, &cfg)?;
    if a.validate_only {
        return Ok(());
    }
    solve(&mut st, &cfg)?;
    let obs = observables(&st);
    // output
    let dir = example_outdir(&cfg)?;
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
        write_green_components(&format!("{pre}_green.dat"), &selected_green(&st))?;
    }
    Ok(())
}
