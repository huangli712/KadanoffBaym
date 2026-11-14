//
// Holstein model on the Bethe lattice in DMFT, normal state and Nambu
// (superconducting) form, with the self-consistent (Migdal) or the
// unrenormalized (uMig) Migdal self-energy (the Rust port of
// julia/tutor/holstein_bethe/holstein_bethe_impl.jl, from NESSi
// examples/programs/Holstein_bethe_{Migdal,uMig,Nambu_Migdal,
// Nambu_uMig}.cpp, no HDF5). The mean-field terms use the real part of
// the density.
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
use crate::holstein_nambu_impl as hn;
use hi::{copy_step, MatsubaraTwoStage};
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::math::dos::BetheDos;
use kadanoffbaym::observables::{correlation_energy, density_matrix_green};
use kadanoffbaym::solvers::dyson_mat::{dyson_mat, MatsubaraMethod};
use kadanoffbaym::solvers::dyson_start::dyson_start;
use kadanoffbaym::solvers::dyson_timestep::dyson_timestep;
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::types::from_mat::from_mat_k;
use kadanoffbaym::weights::Integrator;

pub const HOLSTEIN_APPROX: [&str; 2] = ["Migdal", "uMig"];
pub const HOLSTEIN_BETHE_KEYS: [&str; 4] = ["Hopping", "El_Ph_g", "Phfreq_w0", "MuChem_MF"];
pub const HOLSTEIN_NAMBU_KEYS: [&str; 6] = [
    "Hopping",
    "El_Ph_g",
    "Phfreq_w0",
    "MuChem_MF",
    "sc_seed",
    "sc_seed_iterations",
];

/// Read and validate the TOML file of the Bethe Holstein examples.
pub fn load_holstein_bethe_config(
    cfgfile: &str,
    profile: &str,
    nambu: bool,
) -> Result<ExampleConfig, String> {
    let keys: &[&str] = if nambu {
        &HOLSTEIN_NAMBU_KEYS
    } else {
        &HOLSTEIN_BETHE_KEYS
    };
    let drives: &[&str] = if nambu {
        &["dHopping", "sc_field"]
    } else {
        &["dHopping", "dEl_Ph_g"]
    };
    let cfg = load_example_config(cfgfile, profile, keys, &[], drives, drives)?;
    solver_params(&cfg)?;
    for key in keys {
        if cfg.model[*key].as_f64().is_none() {
            return Err(format!("{cfgfile}: [model] {key} must be a number"));
        }
    }
    if !(cfg.model["Phfreq_w0"].as_f64().unwrap() > 0.0) {
        return Err(format!("{cfgfile}: [model] Phfreq_w0 must be > 0"));
    }
    if nambu {
        let it = cfg.model["sc_seed_iterations"].as_i64();
        if !(matches!(it, Some(v) if v >= 0)) {
            return Err(format!(
                "{cfgfile}: [model] sc_seed_iterations must be a non-negative integer"
            ));
        }
    }
    for d in drives {
        hi::holstein_drive(&cfg, d)?;
    }
    Ok(cfg)
}

/// The observables of the normal-state run, indexed by `KB step + 1`.
#[allow(dead_code)]
pub struct HolsteinBetheResult {
    pub g: ContourGreen,
    pub sigma: ContourGreen,
    pub hyb: ContourGreen,
    pub phonon: hi::HolsteinPhonon,
    pub rho: Vec<DenseMatAlias>,
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

/// Alias to keep the struct definition readable.
pub type DenseMatAlias = kadanoffbaym::mat::DenseMat;

/// Normal-state Holstein model on the Bethe lattice (NESSi
/// holstein_bethe_migdal.cpp / holstein_bethe_umig.cpp).
pub fn run_holstein_bethe(
    approx: &str,
    cfg: &ExampleConfig,
) -> Result<HolsteinBetheResult, String> {
    if !HOLSTEIN_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
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
    let (mut g, mut hyb, mut sigma, mut hs) = (fermion(), fermion(), fermion(), fermion());
    let mut gold = fermion();
    let mut ph = hi::holstein_phonon(nt, c.ntau, 1, w0, beta, h, k);
    let mut h0mf = hi::holstein_scalar_cf(
        nt,
        &vec![C64::new(-m["MuChem_MF"].as_f64().unwrap(), 0.0); (nt + 2) as usize],
    );
    let mut ntot = vec![C64::ZERO; (nt + 2) as usize];
    let mut x = vec![C64::ZERO; (nt + 2) as usize];
    // the Bethe lattice Matsubara Green's function (V = 1, E0 = 0)
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
    hi::holstein_hybridization(0, &mut hyb, &g, &jf);
    // Matsubara
    copy_step(&g, &mut gold, MATSUBARA);
    {
        let mut stage = MatsubaraTwoStage::new();
        let mut err = f64::NAN;
        for iter in 0..=s.mats_max_iter {
            let meth = stage.method();
            if mig {
                hi::sigma_mig(0, &g, &mut sigma, &gf, &mut ph, meth);
            } else {
                hi::sigma_umig(0, &g, &mut sigma, &gf, &mut ph);
            }
            hi::hyb_sigma(0, &mut hs, &hyb, &sigma);
            dyson_mat(&mut g, &hs, 0.0, &h0mf, beta, k, meth, true, None, 1e-12);
            g.smul_at(MATSUBARA, C64::new(0.5, 0.0));
            hi::add_step(&mut g, &gold, MATSUBARA, C64::new(0.5, 0.0));
            hi::holstein_hybridization(0, &mut hyb, &g, &jf);
            err = g.distance_at(&gold, MATSUBARA);
            let name = if stage.fixpoint { "fixpoint" } else { "fourier" };
            println!("matsubara ({name}) iteration {iter}  error = {err:.6e}");
            if stage.update(err, s.mats_max_error) {
                copy_step(&g, &mut gold, MATSUBARA);
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
    ntot[0] = C64::new(2.0 * density_matrix_green(&g, MATSUBARA).get(0, 0).re, 0.0);
    hi::phonon_displacement(0, &mut x, &ntot, &gv, &ph);
    let h0imp = -(m["MuChem_MF"].as_f64().unwrap() + (x[0] * gv[0]).re);
    // Start-up time steps from the Matsubara Green's function
    from_mat_k(&mut g, k);
    from_mat_k(&mut hyb, k);
    for n in 1..=k + 1 {
        copy_step(&g, &mut gold, n - 1);
    }
    {
        let mut err = f64::NAN;
        for iter in 0..=s.bootstrap_max_iter {
            if mig {
                hi::sigma_mig_start(&g, &mut sigma, &gf, &mut ph);
            } else {
                hi::sigma_umig_start(&g, &mut sigma, &gf, &mut ph);
            }
            for n in 1..=k + 1 {
                ntot[n as usize] =
                    C64::new(2.0 * density_matrix_green(&g, n - 1).get(0, 0).re, 0.0);
            }
            hi::phonon_displacement_start(&mut x, &ntot, &gv, &ph);
            for n in 1..=k + 1 {
                h0mf.block_mut(n - 1)[0] =
                    C64::new(h0imp, 0.0) + x[n as usize] * gv[n as usize];
                hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            }
            dyson_start(&mut g, 0.0, &h0mf, &hs, beta, h, k);
            err = 0.0;
            for n in 1..=k + 1 {
                hi::holstein_hybridization(n, &mut hyb, &g, &jf);
                err += g.distance_at(&gold, n - 1);
            }
            println!("bootstrap iteration {iter}  error = {err:.6e}");
            if err < s.bootstrap_max_error && iter > 2 {
                break;
            }
            for n in 1..=k + 1 {
                copy_step(&g, &mut gold, n - 1);
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
        // the predictor extrapolates the KB rows n of G and Hyb
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut g, &ph.integ);
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut hyb, &ph.integ);
        for _iter in 0..s.corrector_steps {
            if mig {
                hi::sigma_mig(n, &g, &mut sigma, &gf, &mut ph, MatsubaraMethod::Fixpoint);
            } else {
                hi::sigma_umig(n, &g, &mut sigma, &gf, &mut ph);
            }
            ntot[n as usize] =
                C64::new(2.0 * density_matrix_green(&g, n - 1).get(0, 0).re, 0.0);
            hi::phonon_displacement(n, &mut x, &ntot, &gv, &ph);
            h0mf.block_mut(n - 1)[0] =
                C64::new(h0imp, 0.0) + x[n as usize] * gv[n as usize];
            hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            dyson_timestep(n - 1, &mut g, 0.0, &h0mf, &hs, beta, h, k);
            hi::holstein_hybridization(n, &mut hyb, &g, &jf);
        }
    }
    // Energies and momentum
    let dout = if mig { &ph.d } else { &ph.d0 };
    if !mig {
        for n in MATSUBARA..=nt {
            ph.pi.smul_at(n, C64::new(0.0, 0.0));
        }
    }
    let p = phonon_momentum(&ntot, &gv, &ph);
    let ecl = phonon_energy_classical(&x, &p, w0);
    let equ = phonon_energy_quantum(dout, &ph.pi, &ph);
    let integ = Integrator::new(k);
    let mut ekin = vec![C64::ZERO; (nt + 2) as usize];
    let mut emf = vec![C64::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![C64::ZERO; (nt + 2) as usize];
    let mut etot = vec![C64::ZERO; (nt + 2) as usize];
    let mut xx = vec![C64::ZERO; (nt + 2) as usize];
    let mut rho = vec![kadanoffbaym::mat::DenseMat::zeros(1); (nt + 2) as usize];
    for n in MATSUBARA + 1..=nt + 1 {
        ekin[n as usize] = C64::new(
            4.0 * correlation_energy(n - 1, &g, &hyb, &integ, beta, h),
            0.0,
        );
        emf[n as usize] = gv[n as usize] * x[n as usize] * ntot[n as usize];
        ecorr[n as usize] = C64::new(
            4.0 * correlation_energy(n - 1, &g, &sigma, &integ, beta, h),
            0.0,
        );
        xx[n as usize] = if n == 0 {
            C64::ZERO
        } else {
            C64::new(0.0, 1.0) * dout.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho[n as usize] = density_matrix_green(&g, n - 1);
    }
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + ecl[n] + equ[n];
    }
    Ok(HolsteinBetheResult {
        g,
        sigma,
        hyb,
        phonon: ph,
        rho,
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

/// The observables of the Nambu run, indexed by `KB step + 1`.
#[allow(dead_code)]
pub struct HolsteinNambuResult {
    pub g: ContourGreen,
    pub sigma: ContourGreen,
    pub hyb: ContourGreen,
    pub phonon: hi::HolsteinPhonon,
    pub rho: Vec<kadanoffbaym::mat::DenseMat>,
    pub xx: Vec<C64>,
    pub ekin: Vec<C64>,
    pub enx_mf: Vec<C64>,
    pub enx_corr: Vec<C64>,
    pub eph: Vec<C64>,
    pub etot: Vec<C64>,
}

/// Holstein model on the Bethe lattice in the Nambu form (NESSi
/// Holstein_bethe_Nambu_{Migdal,uMig}.cpp). The superconducting seed
/// field `sc_seed sigma_1` is added to the Matsubara Hamiltonian in the
/// first `sc_seed_iterations` iterations.
pub fn run_holstein_nambu(
    approx: &str,
    cfg: &ExampleConfig,
) -> Result<HolsteinNambuResult, String> {
    if !HOLSTEIN_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
    let half = C64::new(0.5, 0.0);
    let mut s3 = kadanoffbaym::mat::DenseMat::zeros(2);
    s3.set(0, 0, C64::ONE);
    s3.set(1, 1, -C64::ONE);
    let mut s1 = kadanoffbaym::mat::DenseMat::zeros(2);
    s1.set(0, 1, C64::ONE);
    s1.set(1, 0, C64::ONE);
    let dj = hi::holstein_drive(cfg, "dHopping")?;
    let scf = hi::holstein_drive(cfg, "sc_field")?;
    let mut gf = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut jf = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let mut h0 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
    let h0imp = kadanoffbaym::mat::scale_c(&s3, C64::new(-m["MuChem_MF"].as_f64().unwrap(), 0.0));
    for n in MATSUBARA..=nt {
        let q = (n + 1) as usize;
        gf.block_mut(n)
            .copy_from_slice(kadanoffbaym::mat::scale_c(&s3, C64::new(m["El_Ph_g"].as_f64().unwrap(), 0.0)).raw());
        jf.block_mut(n)
            .copy_from_slice(kadanoffbaym::mat::scale_c(&s3, C64::new(m["Hopping"].as_f64().unwrap(), 0.0) + dj[q]).raw());
        let mut hb = h0imp.clone();
        let sc = kadanoffbaym::mat::scale_c(&s1, scf[q]);
        for i in 0..2 {
            for j in 0..2 {
                hb.set(i, j, hb.get(i, j) + sc.get(i, j));
            }
        }
        h0.block_mut(n).copy_from_slice(hb.raw());
    }
    let cc = Contour::new(nt, c.ntau, 2, 2, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
    let (mut g, mut hyb, mut sigma, mut hs) = (fermion(), fermion(), fermion(), fermion());
    let mut gold = fermion();
    let mut ph = hi::holstein_phonon(nt, c.ntau, 2, w0, beta, h, k);
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
    hi::holstein_hybridization(0, &mut hyb, &g, &jf);
    let seed = m["sc_seed"].as_f64().unwrap();
    let nseed = m["sc_seed_iterations"].as_i64().unwrap();
    copy_step(&g, &mut gold, MATSUBARA);
    {
        let mut stage = MatsubaraTwoStage::new();
        let mut err = f64::NAN;
        for iter in 0..=s.mats_max_iter {
            let meth = stage.method();
            if mig {
                hn::sigma_mig_sc(0, &g, &mut sigma, &gf, &mut ph, meth);
            } else {
                hn::sigma_umig_sc(0, &g, &mut sigma, &gf, &mut ph);
            }
            hi::hyb_sigma(0, &mut hs, &hyb, &sigma);
            // the seed field acts on the Matsubara Hamiltonian
            let mut hb = h0imp.clone();
            let amp = if iter < nseed { seed } else { 0.0 };
            let sc = kadanoffbaym::mat::scale_c(&s1, C64::new(amp, 0.0));
            for i in 0..2 {
                for j in 0..2 {
                    hb.set(i, j, hb.get(i, j) + sc.get(i, j));
                }
            }
            h0.block_mut(MATSUBARA).copy_from_slice(hb.raw());
            dyson_mat(&mut g, &hs, 0.0, &h0, beta, k, meth, true, None, 1e-12);
            g.smul_at(MATSUBARA, half);
            hi::add_step(&mut g, &gold, MATSUBARA, half);
            hi::holstein_hybridization(0, &mut hyb, &g, &jf);
            err = g.distance_at(&gold, MATSUBARA);
            let name = if stage.fixpoint { "fixpoint" } else { "fourier" };
            println!("matsubara ({name}) iteration {iter}  error = {err:.6e}");
            if stage.update(err, s.mats_max_error) {
                copy_step(&g, &mut gold, MATSUBARA);
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
    from_mat_k(&mut g, k);
    from_mat_k(&mut hyb, k);
    for n in 1..=k + 1 {
        copy_step(&g, &mut gold, n - 1);
    }
    {
        let mut err = f64::NAN;
        for iter in 0..=s.bootstrap_max_iter {
            if mig {
                hn::sigma_mig_sc_start(&g, &mut sigma, &gf, &mut ph);
            } else {
                hn::sigma_umig_sc_start(&g, &mut sigma, &gf, &mut ph);
            }
            for n in 1..=k + 1 {
                hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            }
            dyson_start(&mut g, 0.0, &h0, &hs, beta, h, k);
            err = 0.0;
            for n in 1..=k + 1 {
                hi::holstein_hybridization(n, &mut hyb, &g, &jf);
                err += g.distance_at(&gold, n - 1);
            }
            println!("bootstrap iteration {iter}  error = {err:.6e}");
            if err < s.bootstrap_max_error && iter > 2 {
                break;
            }
            for n in 1..=k + 1 {
                copy_step(&g, &mut gold, n - 1);
            }
        }
        if !(err < s.bootstrap_max_error) {
            return Err(format!(
                "ConvergenceError: the bootstrap iteration did not converge (error = {err})"
            ));
        }
    }
    for n in (k + 2)..=(nt + 1) {
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut g, &ph.integ);
        kadanoffbaym::math::extrap::extrapolate_timestep(n - 2, &mut hyb, &ph.integ);
        for _iter in 0..s.corrector_steps {
            if mig {
                hn::sigma_mig_sc(n, &g, &mut sigma, &gf, &mut ph, MatsubaraMethod::Fixpoint);
            } else {
                hn::sigma_umig_sc(n, &g, &mut sigma, &gf, &mut ph);
            }
            hi::hyb_sigma(n, &mut hs, &hyb, &sigma);
            dyson_timestep(n - 1, &mut g, 0.0, &h0, &hs, beta, h, k);
            hi::holstein_hybridization(n, &mut hyb, &g, &jf);
        }
    }
    // uMig: D = D0 and Pi = 0
    if !mig {
        for n in MATSUBARA..=nt {
            hi::copy_step(&ph.d0, &mut ph.d, n);
            ph.pi.smul_at(n, C64::new(0.0, 0.0));
        }
    }
    let integ = Integrator::new(k);
    let mut ekin = vec![C64::ZERO; (nt + 2) as usize];
    let emf = vec![C64::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![C64::ZERO; (nt + 2) as usize];
    let mut etot = vec![C64::ZERO; (nt + 2) as usize];
    let mut xx = vec![C64::ZERO; (nt + 2) as usize];
    let mut rho = vec![kadanoffbaym::mat::DenseMat::zeros(2); (nt + 2) as usize];
    for n in MATSUBARA + 1..=nt + 1 {
        ekin[n as usize] = C64::new(
            2.0 * correlation_energy(n - 1, &g, &hyb, &integ, beta, h),
            0.0,
        );
        ecorr[n as usize] = C64::new(
            2.0 * correlation_energy(n - 1, &g, &sigma, &integ, beta, h),
            0.0,
        );
        xx[n as usize] = if n == 0 {
            C64::ZERO
        } else {
            C64::new(0.0, 1.0) * ph.d.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho[n as usize] = density_matrix_green(&g, n - 1);
    }
    let eph = phonon_energy_quantum(&ph.d, &ph.pi, &ph);
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + eph[n];
    }
    Ok(HolsteinNambuResult {
        g,
        sigma,
        hyb,
        phonon: ph,
        rho,
        xx,
        ekin,
        enx_mf: emf,
        enx_corr: ecorr,
        eph,
        etot,
    })
}

/// Output directory of a configuration.
fn holstein_outdir(cfg: &ExampleConfig) -> Result<String, String> {
    let dir = cfg.output["dir"].as_str().unwrap().to_string();
    let dir = if std::path::Path::new(&dir).is_absolute() {
        dir
    } else {
        format!("{}/{}", cfg.dir, dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Entry of the normal-state Bethe examples. Writes `<prefix>_obs.dat`
/// with the NESSi columns `t, rho.re, rho.im, Xph, Pph, XX, Ekin,
/// Exn_MF, Exn_corr, Eph_cl, Eph_corr, Etot`.
pub fn holstein_bethe_main(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_holstein_bethe_config(&a.cfgfile, &a.profile, false)?;
    if !HOLSTEIN_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_holstein_bethe(approx, &cfg)?;
    let h = contour_params(&cfg)?.h;
    let rows: Vec<Vec<f64>> = (1..=r.g.nt + 1)
        .map(|n| {
            let q = n as usize;
            vec![
                (n - 1) as f64 * h,
                r.rho[q].get(0, 0).re,
                r.rho[q].get(0, 0).im,
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
        &format!(
            "{}/{}_obs.dat",
            holstein_outdir(&cfg)?,
            cfg.output["prefix"].as_str().unwrap()
        ),
        &[
            "time",
            "rho(0,0).re",
            "rho(0,0).im",
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

/// Entry of the Nambu Bethe examples. Writes `<prefix>_obs.dat` with
/// the NESSi columns `t, rho(a,b).re/im, XX, Ekin, Exn_MF, Exn_corr,
/// Eph, Etot`.
pub fn holstein_nambu_main(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_holstein_bethe_config(&a.cfgfile, &a.profile, true)?;
    if !HOLSTEIN_APPROX.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_holstein_nambu(approx, &cfg)?;
    let h = contour_params(&cfg)?.h;
    let mut names: Vec<String> = vec!["time".to_string()];
    for a in 0..2usize {
        for b in 0..2usize {
            names.push(format!("rho({a},{b}).re"));
            names.push(format!("rho({a},{b}).im"));
        }
    }
    for n in ["XX", "Ekin", "Exn_MF", "Exn_corr", "Eph", "Etot"] {
        names.push(n.to_string());
    }
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let rows: Vec<Vec<f64>> = (1..=r.g.nt + 1)
        .map(|n| {
            let q = n as usize;
            let mut v = vec![(n - 1) as f64 * h];
            for a in 0..2usize {
                for b in 0..2usize {
                    v.push(r.rho[q].get(a, b).re);
                    v.push(r.rho[q].get(a, b).im);
                }
            }
            for val in [
                r.xx[q],
                r.ekin[q],
                r.enx_mf[q],
                r.enx_corr[q],
                r.eph[q],
                r.etot[q],
            ] {
                v.push(val.re);
            }
            v
        })
        .collect();
    write_table(
        &format!(
            "{}/{}_obs.dat",
            holstein_outdir(&cfg)?,
            cfg.output["prefix"].as_str().unwrap()
        ),
        &name_refs,
        &rows,
    )?;
    Ok(())
}
