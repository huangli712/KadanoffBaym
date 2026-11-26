// Project : KadanoffBaym
// File    : examples/holstein/impurity.rs
// Purpose : Holstein impurity coupled to a single bath site with Migdal or uMig self-energies.
//

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::linalg::Adjoint;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::observables::{correlation_energy, density};

use super::model::Couplings;
use super::solver::{Kernels, Loop, MeanField};
use super::util::ExampleConfig;

use super::energy::{phonon_energy_classical, phonon_energy_quantum, phonon_momentum};
use super::model::{
    check_model, couplings, drive_series, hybridization, new_phonon, obs_path, phonon_displacement,
    scalar_time_function,
};
use super::solver::{bootstrap_loop, matsubara_loop, propagate};
use super::util::{
    contour_params, load_example_config, parse_example_args, solver_params, write_table,
};

use super::util::prelude::*;

pub(crate) const APPROXIMATIONS: [&str; 2] = ["Migdal", "uMig"];

pub(crate) const MODEL_KEYS: [&str; 5] = ["Hopping", "El_Ph_g", "Phfreq_w0", "Eps0_MF", "Eps1"];

/// The result of the impurity run, indexed by the KB step number.
pub(crate) struct ImpurityResult {
    pub(crate) g00: ContourGreen,
    pub(crate) rho00: Vec<Complex>,
    pub(crate) rho11: Vec<Complex>,
    pub(crate) xph: Vec<Complex>,
    pub(crate) pph: Vec<Complex>,
    pub(crate) xx: Vec<Complex>,
    pub(crate) ekin: Vec<Complex>,
    pub(crate) enx_mf: Vec<Complex>,
    pub(crate) enx_corr: Vec<Complex>,
    pub(crate) eph_cl: Vec<Complex>,
    pub(crate) eph_corr: Vec<Complex>,
    pub(crate) etot: Vec<Complex>,
}

/// Read and validate the TOML file of the impurity examples.
pub(crate) fn load_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    let drives: [&str; 2] = ["dHopping", "dEl_Ph_g"];
    let cfg = load_example_config(cfgfile, profile, &MODEL_KEYS, &[], &drives, &drives)?;
    solver_params(&cfg)?;
    check_model(&cfg, cfgfile, &MODEL_KEYS)?;
    drive_series(&cfg, "dHopping")?;
    drive_series(&cfg, "dEl_Ph_g")?;
    Ok(cfg)
}

/// Matsubara (FIXPOINT), start-up, and time steps of a Dyson equation,
/// as a post-processing step.
fn solve_dyson(
    g: &mut ContourGreen,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    k: i64,
) {
    dyson_mat(
        g,
        sigma,
        0.0,
        h,
        beta,
        k,
        MatsubaraMethod::Fixpoint,
        true,
        None,
        1e-12,
    );
    dyson_start(g, 0.0, h, sigma, beta, hstep, k);
    // the kernel `sigma` is fixed within the pass
    let integ = Integrator::new(k);
    let mut dws = DysonWorkspace::new(k, g.size1, g.ntau);
    for n in (k + 2)..=(g.nt as i64 + 1) {
        dyson_timestep_ws(n - 1, g, 0.0, h, sigma, beta, hstep, &integ, &mut dws);
    }
}

/// Solve the impurity problem.
pub(crate) fn run_impurity(approx: &str, cfg: &ExampleConfig) -> Result<ImpurityResult, String> {
    let kernels = Kernels::of(approx)?;
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
    let (eps0mf, eps1) = (m["Eps0_MF"].as_f64().unwrap(), m["Eps1"].as_f64().unwrap());
    let Couplings { gv, gf, jf } = couplings(cfg, nt)?;
    let cc = Contour::new(nt, c.ntau, 1, 1, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut g011 = fermion();
    let mut g00 = fermion();
    let ph = new_phonon(nt, c.ntau, 1, w0, beta, h, k);
    let h11 = scalar_time_function(nt, &vec![Complex::new(eps1, 0.0); (nt + 2) as usize]);
    let mut eps = DenseMat::zeros(1);
    eps.set(0, 0, Complex::new(eps1, 0.0));
    from_ham(&mut g011, 0.0, &eps, beta, h);
    eps.set(0, 0, Complex::new(eps0mf, 0.0));
    from_ham(&mut g00, 0.0, &eps, beta, h);
    let mut hyb = fermion();
    // the KB time steps 0..ntime; hybridization maps them itself
    for n in 0..=nt + 1 {
        hybridization(n, &mut hyb, &g011, &jf);
    }
    let mut l = Loop {
        g: g00,
        g_old: fermion(),
        sigma: fermion(),
        hs: fermion(),
        hyb,
        h0: scalar_time_function(nt, &vec![Complex::new(eps0mf, 0.0); (nt + 2) as usize]),
        gf,
        jf,
        ph,
        dmft: false,
        nt,
        k,
        beta,
        h,
    };
    matsubara_loop(&mut l, kernels, &s, |_, _| {})?;
    // the Matsubara density and the phonon displacement set the Hartree term
    let mut mf = MeanField {
        gv,
        x: vec![Complex::ZERO; (nt + 2) as usize],
        ntot: vec![Complex::ZERO; (nt + 2) as usize],
        h0const: Complex::ZERO,
    };
    mf.ntot[0] = Complex::new(2.0 * density(&l.g, MATSUBARA).re, 0.0);
    phonon_displacement(0, &mut mf.x, &mf.ntot, &mf.gv, &l.ph);
    let h00 = eps0mf - (mf.x[0] * mf.gv[0]).re;
    mf.h0const = Complex::new(h00, 0.0);
    bootstrap_loop(&mut l, Some(&mut mf), kernels, &s)?;
    propagate(&mut l, Some(&mut mf), kernels, &s);
    // G'00 = [i d/dt - h00_MF - Sigma]^-1, G11 = [i d/dt - h11 - J G'00 J]^-1
    let mut jg00j = fermion();
    let mut g11 = fermion();
    solve_dyson(&mut jg00j, &l.h0, &l.sigma, beta, h, k);
    for n in MATSUBARA..=nt {
        jg00j.rmul_cf_at(n, &l.jf, Adjoint::No);
        jg00j.lmul_cf_at(n, &l.jf, Adjoint::No);
    }
    solve_dyson(&mut g11, &h11, &jg00j, beta, h, k);
    let MeanField {
        gv,
        x,
        ntot,
        h0const: _,
    } = mf;
    let Loop {
        g: g00,
        hyb,
        sigma,
        mut ph,
        ..
    } = l;
    let p = phonon_momentum(&ntot, &gv, &ph);
    // Energies
    if !mig {
        for n in MATSUBARA..=nt {
            ph.pi.smul_at(n, Complex::new(0.0, 0.0));
        }
    }
    let dout_ref = &ph.d0;
    let _ = dout_ref;
    let dout = if mig { &ph.d } else { &ph.d0 };
    let integ = Integrator::new(k);
    let mut ekin = vec![Complex::ZERO; (nt + 2) as usize];
    let mut emf = vec![Complex::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![Complex::ZERO; (nt + 2) as usize];
    let mut etot = vec![Complex::ZERO; (nt + 2) as usize];
    let mut xx = vec![Complex::ZERO; (nt + 2) as usize];
    let mut rho00 = vec![Complex::ZERO; (nt + 2) as usize];
    let mut rho11 = vec![Complex::ZERO; (nt + 2) as usize];
    let ecl = phonon_energy_classical(&x, &p, w0);
    let equ = phonon_energy_quantum(dout, &ph.pi, &ph);
    for n in MATSUBARA + 1..=nt + 1 {
        let q = n as usize;
        let rho10j = correlation_energy(n - 1, &g00, &hyb, &integ, beta, h);
        let n1 = Complex::new(2.0, 0.0) * density(&g11, n - 1);
        ekin[q] = Complex::new(8.0 * rho10j, 0.0)
            + Complex::new(eps1, 0.0) * n1
            + Complex::new(h00, 0.0) * ntot[q];
        emf[q] = gv[q] * x[q] * ntot[q];
        ecorr[q] = Complex::new(
            4.0 * correlation_energy(n - 1, &g00, &sigma, &integ, beta, h),
            0.0,
        );
        xx[q] = if n == 0 {
            Complex::ZERO
        } else {
            Complex::new(0.0, 1.0) * dout.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho00[q] = density(&g00, n - 1);
        rho11[q] = density(&g11, n - 1);
    }
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + ecl[n] + equ[n];
    }
    Ok(ImpurityResult {
        g00,
        rho00,
        rho11,
        xph: x,
        pph: p,
        xx,
        ekin,
        enx_mf: emf,
        enx_corr: ecorr,
        eph_cl: ecl,
        eph_corr: equ,
        etot,
    })
}

/// Entry of the impurity examples. Writes `<prefix>_obs.dat` with the
/// columns `t, rho00.re/im, rho11.re/im, Xph, Pph, XX, Ekin,
/// Enx_MF, Enx_corr, Eph_cl, Eph_corr, Etot`.
pub(crate) fn run(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_config(&a.cfgfile, &a.profile)?;
    if !APPROXIMATIONS.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_impurity(approx, &cfg)?;
    let h = contour_params(&cfg)?.h;
    let rows: Vec<Vec<f64>> = (1..=r.g00.nt as i64 + 1)
        .map(|n| {
            let q = n as usize;
            vec![
                (n - 1) as f64 * h,
                r.rho00[q].re,
                r.rho00[q].im,
                r.rho11[q].re,
                r.rho11[q].im,
                r.xph[q].re,
                r.pph[q].re,
                r.xx[q].re,
                r.ekin[q].re,
                r.enx_mf[q].re,
                r.enx_corr[q].re,
                r.eph_cl[q].re,
                r.eph_corr[q].re,
                r.etot[q].re,
            ]
        })
        .collect();
    write_table(
        &obs_path(&cfg)?,
        &[
            "time", "rho00.re", "rho00.im", "rho11.re", "rho11.im", "Xph", "Pph", "XX", "Ekin",
            "Enx_MF", "Enx_corr", "Eph_cl", "Eph_corr", "Etot",
        ],
        &rows,
    )?;
    Ok(())
}
