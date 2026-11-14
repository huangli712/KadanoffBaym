//
// Holstein impurity coupled to a single bath site, self-consistent
// (Migdal) or unrenormalized (uMig) Migdal self-energy (the Rust port
// of julia/tutor/holstein_impurity/holstein_impurity_impl.jl, from NESSi
// examples/programs/Holstein_impurity_singlebath_{Migdal,uMig}.cpp, no
// HDF5). The hybridization `Delta = J G0_11 J` of the free bath site is
// fixed.
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::common::config::{
    contour_params, load_example_config, parse_example_args, solver_params, ExampleConfig,
};
use crate::common::output::write_table;
use crate::holstein_energy::{
    phonon_energy_classical, phonon_energy_quantum, phonon_momentum,
};
use crate::holstein_impl as hi;
use hi::{copy_step, MatsubaraTwoStage};
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::mat::DenseMat;
use kadanoffbaym::observables::{correlation_energy, density_matrix_green};
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::from_mat::from_mat_k;
use kadanoffbaym::types::green::ContourGreen;
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::weights::Integrator;

pub const IMPURITY_APPROX: [&str; 2] = ["Migdal", "uMig"];
pub const IMPURITY_KEYS: [&str; 5] = ["Hopping", "El_Ph_g", "Phfreq_w0", "Eps0_MF", "Eps1"];

/// Read and validate the TOML file of the impurity examples.
pub fn load_impurity_config(cfgfile: &str, profile: &str) -> Result<ExampleConfig, String> {
    let drives: [&str; 2] = ["dHopping", "dEl_Ph_g"];
    let cfg = load_example_config(cfgfile, profile, &IMPURITY_KEYS, &[], &drives, &drives)?;
    solver_params(&cfg)?;
    for key in IMPURITY_KEYS {
        if cfg.model[key].as_f64().is_none() {
            return Err(format!("{cfgfile}: [model] {key} must be a number"));
        }
    }
    if !(cfg.model["Phfreq_w0"].as_f64().unwrap() > 0.0) {
        return Err(format!("{cfgfile}: [model] Phfreq_w0 must be > 0"));
    }
    hi::holstein_drive(&cfg, "dHopping")?;
    hi::holstein_drive(&cfg, "dEl_Ph_g")?;
    Ok(cfg)
}

/// Matsubara (FIXPOINT), start-up, and time steps of a Dyson equation,
/// as the NESSi post-processing.
fn dyson_all(g: &mut ContourGreen, h: &kadanoffbaym::types::green::TimeFunction, sigma: &ContourGreen, beta: f64, hstep: f64, k: i64) {
    dyson_mat(g, sigma, 0.0, h, beta, k, MatsubaraMethod::Fixpoint, true, None, 1e-12);
    dyson_start(g, 0.0, h, sigma, beta, hstep, k);
    for n in (k + 2)..=(g.nt as i64 + 1) {
        dyson_timestep(n - 1, g, 0.0, h, sigma, beta, hstep, k);
    }
}

/// The result of the impurity run, indexed by the KB step number.
#[allow(dead_code)]
pub struct ImpurityResult {
    pub g00: ContourGreen,
    pub g11: ContourGreen,
    pub sigma: ContourGreen,
    pub phonon: hi::HolsteinPhonon,
    pub rho00: Vec<C64>,
    pub rho11: Vec<C64>,
    pub xph: Vec<C64>,
    pub pph: Vec<C64>,
    pub xx: Vec<C64>,
    pub ekin: Vec<C64>,
    pub enx_mf: Vec<C64>,
    pub enx_corr: Vec<C64>,
    pub eph_cl: Vec<C64>,
    pub eph_corr: Vec<C64>,
    pub etot: Vec<C64>,
}

/// Solve the impurity problem.
pub fn run_holstein_impurity(approx: &str, cfg: &ExampleConfig) -> Result<ImpurityResult, String> {
    if !IMPURITY_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
    let (eps0mf, eps1) = (
        m["Eps0_MF"].as_f64().unwrap(),
        m["Eps1"].as_f64().unwrap(),
    );
    let gd = hi::holstein_drive(cfg, "dEl_Ph_g")?;
    let jd = hi::holstein_drive(cfg, "dHopping")?;
    let gv: Vec<C64> = (0..=(nt + 1) as usize)
        .map(|q| C64::new(m["El_Ph_g"].as_f64().unwrap(), 0.0) + gd[q])
        .collect();
    let jv: Vec<C64> = (0..=(nt + 1) as usize)
        .map(|q| C64::new(m["Hopping"].as_f64().unwrap(), 0.0) + jd[q])
        .collect();
    let gf = hi::holstein_scalar_cf(nt, &gv);
    let jf = hi::holstein_scalar_cf(nt, &jv);
    let cc = Contour::new(nt, c.ntau, 1, 1, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
    let (mut g00, mut hyb, mut sigma, mut hs) = (fermion(), fermion(), fermion(), fermion());
    let (mut gold, mut g011, mut jg00j, mut g11) = (fermion(), fermion(), fermion(), fermion());
    let mut ph = hi::holstein_phonon(nt, c.ntau, 1, w0, beta, h, k);
    let mut h00mf = hi::holstein_scalar_cf(nt, &vec![C64::new(eps0mf, 0.0); (nt + 2) as usize]);
    let h11 = hi::holstein_scalar_cf(nt, &vec![C64::new(eps1, 0.0); (nt + 2) as usize]);
    let mut ntot = vec![C64::ZERO; (nt + 2) as usize];
    let mut x = vec![C64::ZERO; (nt + 2) as usize];
    let mut eps = DenseMat::zeros(1);
    eps.set(0, 0, C64::new(eps1, 0.0));
    from_ham(&mut g011, 0.0, &eps, beta, h);
    eps.set(0, 0, C64::new(eps0mf, 0.0));
    from_ham(&mut g00, 0.0, &eps, beta, h);
    // the KB time steps 0..ntime; holstein_hybridization maps them itself
    for n in 0..=nt + 1 {
        hi::holstein_hybridization(n, &mut hyb, &g011, &jf);
    }
    // Matsubara
    copy_step(&g00, &mut gold, MATSUBARA);
    {
        let mut stage = MatsubaraTwoStage::new();
        let mut err = f64::NAN;
        for iter in 0..=s.mats_max_iter {
            let meth = stage.method();
            if mig {
                hi::sigma_mig(0, &g00, &mut sigma, &gf, &mut ph, meth);
            } else {
                hi::sigma_umig(0, &g00, &mut sigma, &gf, &mut ph);
            }
            hi::hyb_sigma(0, &mut hs, &hyb, &sigma);
            dyson_mat(&mut g00, &hs, 0.0, &h00mf, beta, k, meth, true, None, 1e-12);
            g00.smul_at(MATSUBARA, C64::new(0.5, 0.0));
            hi::add_step(&mut g00, &gold, MATSUBARA, C64::new(0.5, 0.0));
            err = g00.distance_at(&gold, MATSUBARA);
            let name = if stage.fixpoint { "fixpoint" } else { "fourier" };
            println!("matsubara ({name}) iteration {iter}  error = {err:.6e}");
            if stage.update(err, s.mats_max_error) {
                copy_step(&g00, &mut gold, MATSUBARA);
            }
            if stage.converged {
                break;
            }
        }
        if !stage.converged {
            return Err(format!(
                "ConvergenceError: the matsubara iteration did not converge (error = {err})"
            ));
        }
    }
    ntot[0] = C64::new(2.0 * density_matrix_green(&g00, MATSUBARA).get(0, 0).re, 0.0);
    hi::phonon_displacement(0, &mut x, &ntot, &gv, &ph);
    let h00 = eps0mf - (x[0] * gv[0]).re;
    from_mat_k(&mut g00, k);
    // Bootstrapping
    for n in 1..=k + 1 {
        copy_step(&g00, &mut gold, n - 1);
    }
    {
        let mut err = f64::NAN;
        for iter in 0..=s.bootstrap_max_iter {
            if mig {
                hi::sigma_mig_start(&g00, &mut sigma, &gf, &mut ph);
            } else {
                hi::sigma_umig_start(&g00, &mut sigma, &gf, &mut ph);
            }
            for n in 1..=k + 1 {
                ntot[n as usize] =
                    C64::new(2.0 * density_matrix_green(&g00, n - 1).get(0, 0).re, 0.0);
            }
            hi::phonon_displacement_start(&mut x, &ntot, &gv, &ph);
            for n in 1..=k + 1 {
                h00mf.block_mut(n - 1)[0] =
                    C64::new(h00, 0.0) + x[n as usize] * gv[n as usize];
                hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            }
            dyson_start(&mut g00, 0.0, &h00mf, &hs, beta, h, k);
            err = 0.0;
            for n in 1..=k + 1 {
                err += g00.distance_at(&gold, n - 1);
            }
            println!("bootstrap iteration {iter}  error = {err:.6e}");
            if err < s.bootstrap_max_error && iter > 2 {
                break;
            }
            for n in 1..=k + 1 {
                copy_step(&g00, &mut gold, n - 1);
            }
        }
        if !(err < s.bootstrap_max_error) {
            return Err(format!(
                "ConvergenceError: the bootstrap iteration did not converge (error = {err})"
            ));
        }
    }
    // Propagation
    for n in (k + 2)..=(nt + 1) {
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut g00, &ph.integ);
        for _iter in 0..s.corrector_steps {
            if mig {
                hi::sigma_mig(n, &g00, &mut sigma, &gf, &mut ph, MatsubaraMethod::Fixpoint);
            } else {
                hi::sigma_umig(n, &g00, &mut sigma, &gf, &mut ph);
            }
            ntot[n as usize] =
                C64::new(2.0 * density_matrix_green(&g00, n - 1).get(0, 0).re, 0.0);
            hi::phonon_displacement(n, &mut x, &ntot, &gv, &ph);
            h00mf.block_mut(n - 1)[0] = C64::new(h00, 0.0) + x[n as usize] * gv[n as usize];
            hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            dyson_timestep(n - 1, &mut g00, 0.0, &h00mf, &hs, beta, h, k);
        }
    }
    // G'00 = [i d/dt - h00_MF - Sigma]^-1, G11 = [i d/dt - h11 - J G'00 J]^-1
    dyson_all(&mut jg00j, &h00mf, &sigma, beta, h, k);
    for n in MATSUBARA..=nt {
        jg00j.rmul_cf_at(n, &jf, kadanoffbaym::ops::Mode::Normal);
        jg00j.lmul_cf_at(n, &jf, kadanoffbaym::ops::Mode::Normal);
    }
    dyson_all(&mut g11, &h11, &jg00j, beta, h, k);
    let p = phonon_momentum(&ntot, &gv, &ph);
    // Energies
    if !mig {
        for n in MATSUBARA..=nt {
            ph.pi.smul_at(n, C64::new(0.0, 0.0));
        }
    }
    let dout_ref = &ph.d0;
    let _ = dout_ref;
    let dout = if mig { &ph.d } else { &ph.d0 };
    let integ = Integrator::new(k);
    let mut ekin = vec![C64::ZERO; (nt + 2) as usize];
    let mut emf = vec![C64::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![C64::ZERO; (nt + 2) as usize];
    let mut etot = vec![C64::ZERO; (nt + 2) as usize];
    let mut xx = vec![C64::ZERO; (nt + 2) as usize];
    let mut rho00 = vec![C64::ZERO; (nt + 2) as usize];
    let mut rho11 = vec![C64::ZERO; (nt + 2) as usize];
    let ecl = phonon_energy_classical(&x, &p, w0);
    let equ = phonon_energy_quantum(dout, &ph.pi, &ph);
    for n in MATSUBARA + 1..=nt + 1 {
        let q = n as usize;
        let rho10j = correlation_energy(n - 1, &g00, &hyb, &integ, beta, h);
        let n1 = C64::new(2.0, 0.0) * density_matrix_green(&g11, n - 1).get(0, 0);
        ekin[q] = C64::new(8.0 * rho10j, 0.0) + C64::new(eps1, 0.0) * n1
            + C64::new(h00, 0.0) * ntot[q];
        emf[q] = gv[q] * x[q] * ntot[q];
        ecorr[q] = C64::new(
            4.0 * correlation_energy(n - 1, &g00, &sigma, &integ, beta, h),
            0.0,
        );
        xx[q] = if n == 0 {
            C64::ZERO
        } else {
            C64::new(0.0, 1.0) * dout.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho00[q] = density_matrix_green(&g00, n - 1).get(0, 0);
        rho11[q] = density_matrix_green(&g11, n - 1).get(0, 0);
    }
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + ecl[n] + equ[n];
    }
    Ok(ImpurityResult {
        g00,
        g11,
        sigma,
        phonon: ph,
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
/// NESSi columns `t, rho00.re/im, rho11.re/im, Xph, Pph, XX, Ekin,
/// Exn_MF, Exn_corr, Eph_cl, Eph_corr, Etot`.
pub fn holstein_impurity_main(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_impurity_config(&a.cfgfile, &a.profile)?;
    if !IMPURITY_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_holstein_impurity(approx, &cfg)?;
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
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if std::path::Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_table(
        &format!(
            "{}/{}_obs.dat",
            dir,
            cfg.output["prefix"].as_str().unwrap()
        ),
        &[
            "time",
            "rho00.re",
            "rho00.im",
            "rho11.re",
            "rho11.im",
            "Xph",
            "Pph",
            "XX",
            "Ekin",
            "Exn_MF",
            "Exn_corr",
            "Eph_cl",
            "Eph_corr",
            "Etot",
        ],
        &rows,
    )?;
    Ok(())
}
