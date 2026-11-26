// Project : KadanoffBaym
// File    : examples/holstein/bethe.rs
// Purpose : Holstein model on the Bethe lattice in DMFT, normal and Nambu form, with Migdal or uMig self-energies.
//

use kadanoffbaym::math::BetheDos;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::equilibrium::from_dos_mat;
use kadanoffbaym::linalg::scale_c;
use kadanoffbaym::observables::{correlation_energy, density, density_matrix};

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

pub(crate) const MODEL_KEYS: [&str; 4] = ["Hopping", "El_Ph_g", "Phfreq_w0", "MuChem_MF"];

pub(crate) const NAMBU_MODEL_KEYS: [&str; 6] = [
    "Hopping",
    "El_Ph_g",
    "Phfreq_w0",
    "MuChem_MF",
    "sc_seed",
    "sc_seed_iterations",
];

/// The observables of the normal-state run, indexed by `KB step + 1`.
pub(crate) struct BetheResult {
    pub(crate) g: ContourGreen,
    pub(crate) rho: Vec<DenseMat>,
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

/// The observables of the Nambu run, indexed by `KB step + 1`.
pub(crate) struct NambuResult {
    pub(crate) g: ContourGreen,
    pub(crate) rho: Vec<DenseMat>,
    pub(crate) xx: Vec<Complex>,
    pub(crate) ekin: Vec<Complex>,
    pub(crate) enx_mf: Vec<Complex>,
    pub(crate) enx_corr: Vec<Complex>,
    pub(crate) eph: Vec<Complex>,
    pub(crate) etot: Vec<Complex>,
}

/// Read and validate the TOML file of the Bethe Holstein examples.
pub(crate) fn load_config(
    cfgfile: &str,
    profile: &str,
    nambu: bool,
) -> Result<ExampleConfig, String> {
    let keys: &[&str] = if nambu {
        &NAMBU_MODEL_KEYS
    } else {
        &MODEL_KEYS
    };
    let drives: &[&str] = if nambu {
        &["dHopping", "sc_field"]
    } else {
        &["dHopping", "dEl_Ph_g"]
    };
    let cfg = load_example_config(cfgfile, profile, keys, &[], drives, drives)?;
    solver_params(&cfg)?;
    check_model(&cfg, cfgfile, keys)?;
    if nambu {
        let it = cfg.model["sc_seed_iterations"].as_i64();
        if !(matches!(it, Some(v) if v >= 0)) {
            return Err(format!(
                "{cfgfile}: [model] sc_seed_iterations must be a non-negative integer"
            ));
        }
    }
    for d in drives {
        drive_series(&cfg, d)?;
    }
    Ok(cfg)
}

/// Normal-state Holstein model on the Bethe lattice.
pub(crate) fn run_bethe(approx: &str, cfg: &ExampleConfig) -> Result<BetheResult, String> {
    let kernels = Kernels::of(approx)?;
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
    let Couplings { gv, gf, jf } = couplings(cfg, nt)?;
    let cc = Contour::new(nt, c.ntau, 1, 1, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut g = fermion();
    let ph = new_phonon(nt, c.ntau, 1, w0, beta, h, k);
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
    let mut hyb = fermion();
    hybridization(0, &mut hyb, &g, &jf);
    let mut l = Loop {
        g,
        g_old: fermion(),
        sigma: fermion(),
        hs: fermion(),
        hyb,
        h0: scalar_time_function(
            nt,
            &vec![Complex::new(-m["MuChem_MF"].as_f64().unwrap(), 0.0); (nt + 2) as usize],
        ),
        gf,
        jf,
        ph,
        dmft: true,
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
    mf.h0const = Complex::new(
        -(m["MuChem_MF"].as_f64().unwrap() + (mf.x[0] * mf.gv[0]).re),
        0.0,
    );
    bootstrap_loop(&mut l, Some(&mut mf), kernels, &s)?;
    propagate(&mut l, Some(&mut mf), kernels, &s);
    let MeanField {
        gv,
        x,
        ntot,
        h0const: _,
    } = mf;
    let Loop {
        g,
        hyb,
        sigma,
        mut ph,
        ..
    } = l;
    // Energies and momentum
    let dout = if mig { &ph.d } else { &ph.d0 };
    if !mig {
        for n in MATSUBARA..=nt {
            ph.pi.smul_at(n, Complex::new(0.0, 0.0));
        }
    }
    let p = phonon_momentum(&ntot, &gv, &ph);
    let ecl = phonon_energy_classical(&x, &p, w0);
    let equ = phonon_energy_quantum(dout, &ph.pi, &ph);
    let integ = Integrator::new(k);
    let mut ekin = vec![Complex::ZERO; (nt + 2) as usize];
    let mut emf = vec![Complex::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![Complex::ZERO; (nt + 2) as usize];
    let mut etot = vec![Complex::ZERO; (nt + 2) as usize];
    let mut xx = vec![Complex::ZERO; (nt + 2) as usize];
    let mut rho = vec![DenseMat::zeros(1); (nt + 2) as usize];
    for n in MATSUBARA + 1..=nt + 1 {
        ekin[n as usize] = Complex::new(
            4.0 * correlation_energy(n - 1, &g, &hyb, &integ, beta, h),
            0.0,
        );
        emf[n as usize] = gv[n as usize] * x[n as usize] * ntot[n as usize];
        ecorr[n as usize] = Complex::new(
            4.0 * correlation_energy(n - 1, &g, &sigma, &integ, beta, h),
            0.0,
        );
        xx[n as usize] = if n == 0 {
            Complex::ZERO
        } else {
            Complex::new(0.0, 1.0) * dout.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho[n as usize] = density_matrix(&g, n - 1);
    }
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + ecl[n] + equ[n];
    }
    Ok(BetheResult {
        g,
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

/// Holstein model on the Bethe lattice in the Nambu form. The
/// superconducting seed
/// field `sc_seed sigma_1` is added to the Matsubara Hamiltonian in the
/// first `sc_seed_iterations` iterations.
pub(crate) fn run_nambu(approx: &str, cfg: &ExampleConfig) -> Result<NambuResult, String> {
    let kernels = Kernels::nambu(approx)?;
    let mig = approx == "Migdal";
    let (c, s, m) = (contour_params(cfg)?, solver_params(cfg)?, &cfg.model);
    let (nt, k) = (c.nt, c.k);
    let (beta, h) = (c.beta, c.h);
    let w0 = m["Phfreq_w0"].as_f64().unwrap();
    let mut s3 = DenseMat::zeros(2);
    s3.set(0, 0, Complex::ONE);
    s3.set(1, 1, -Complex::ONE);
    let mut s1 = DenseMat::zeros(2);
    s1.set(0, 1, Complex::ONE);
    s1.set(1, 0, Complex::ONE);
    let dj = drive_series(cfg, "dHopping")?;
    let scf = drive_series(cfg, "sc_field")?;
    let mut gf = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut jf = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut h0 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let h0imp = scale_c(&s3, Complex::new(-m["MuChem_MF"].as_f64().unwrap(), 0.0));
    for n in MATSUBARA..=nt {
        let q = (n + 1) as usize;
        gf.block_mut(n)
            .copy_from_slice(scale_c(&s3, Complex::new(m["El_Ph_g"].as_f64().unwrap(), 0.0)).raw());
        jf.block_mut(n).copy_from_slice(
            scale_c(
                &s3,
                Complex::new(m["Hopping"].as_f64().unwrap(), 0.0) + dj[q],
            )
            .raw(),
        );
        let mut hb = h0imp.clone();
        let sc = scale_c(&s1, scf[q]);
        for i in 0..2 {
            for j in 0..2 {
                hb.set(i, j, hb.get(i, j) + sc.get(i, j));
            }
        }
        h0.block_mut(n).copy_from_slice(hb.raw());
    }
    let cc = Contour::new(nt, c.ntau, 2, 2, nt as f64 * h, beta);
    let fermion = || ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut g = fermion();
    let ph = new_phonon(nt, c.ntau, 2, w0, beta, h, k);
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
    let mut hyb = fermion();
    hybridization(0, &mut hyb, &g, &jf);
    let seed = m["sc_seed"].as_f64().unwrap();
    let nseed = m["sc_seed_iterations"].as_i64().unwrap();
    let mut l = Loop {
        g,
        g_old: fermion(),
        sigma: fermion(),
        hs: fermion(),
        hyb,
        h0,
        gf,
        jf,
        ph,
        dmft: true,
        nt,
        k,
        beta,
        h,
    };
    // the seed field acts on the Matsubara Hamiltonian
    matsubara_loop(&mut l, kernels, &s, |iter, h0| {
        let mut hb = h0imp.clone();
        let amp = if iter < nseed { seed } else { 0.0 };
        let sc = scale_c(&s1, Complex::new(amp, 0.0));
        for i in 0..2 {
            for j in 0..2 {
                hb.set(i, j, hb.get(i, j) + sc.get(i, j));
            }
        }
        h0.block_mut(MATSUBARA).copy_from_slice(hb.raw());
    })?;
    bootstrap_loop(&mut l, None, kernels, &s)?;
    propagate(&mut l, None, kernels, &s);
    let Loop {
        g,
        hyb,
        sigma,
        mut ph,
        ..
    } = l;
    // uMig: D = D0 and Pi = 0
    if !mig {
        for n in MATSUBARA..=nt {
            ph.d.memcpy_from_at(n, &ph.d0);
            ph.pi.smul_at(n, Complex::new(0.0, 0.0));
        }
    }
    let integ = Integrator::new(k);
    let mut ekin = vec![Complex::ZERO; (nt + 2) as usize];
    let emf = vec![Complex::ZERO; (nt + 2) as usize];
    let mut ecorr = vec![Complex::ZERO; (nt + 2) as usize];
    let mut etot = vec![Complex::ZERO; (nt + 2) as usize];
    let mut xx = vec![Complex::ZERO; (nt + 2) as usize];
    let mut rho = vec![DenseMat::zeros(2); (nt + 2) as usize];
    for n in MATSUBARA + 1..=nt + 1 {
        ekin[n as usize] = Complex::new(
            2.0 * correlation_energy(n - 1, &g, &hyb, &integ, beta, h),
            0.0,
        );
        ecorr[n as usize] = Complex::new(
            2.0 * correlation_energy(n - 1, &g, &sigma, &integ, beta, h),
            0.0,
        );
        xx[n as usize] = if n == 0 {
            Complex::ZERO
        } else {
            Complex::new(0.0, 1.0) * ph.d.less.raw_block_at(n - 1, n - 1)[0]
        };
        rho[n as usize] = density_matrix(&g, n - 1);
    }
    let eph = phonon_energy_quantum(&ph.d, &ph.pi, &ph);
    for n in 0..((nt + 2) as usize) {
        etot[n] = ekin[n] + emf[n] + ecorr[n] + eph[n];
    }
    Ok(NambuResult {
        g,
        rho,
        xx,
        ekin,
        enx_mf: emf,
        enx_corr: ecorr,
        eph,
        etot,
    })
}

/// Entry of the normal-state Bethe examples. Writes `<prefix>_obs.dat`
/// with the output columns `t, rho.re, rho.im, Xph, Pph, XX, Ekin,
/// Enx_MF, Enx_corr, Eph_cl, Eph_corr, Etot`.
pub(crate) fn run_bethe_case(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_config(&a.cfgfile, &a.profile, false)?;
    if !APPROXIMATIONS.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_bethe(approx, &cfg)?;
    let h = contour_params(&cfg)?.h;
    let rows: Vec<Vec<f64>> = (1..=r.g.nt + 1)
        .map(|n| {
            let q = n;
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
        &obs_path(&cfg)?,
        &[
            "time",
            "rho(0,0).re",
            "rho(0,0).im",
            "Xph",
            "Pph",
            "XX",
            "Ekin",
            "Enx_MF",
            "Enx_corr",
            "Eph_cl",
            "Eph_corr",
            "Etot",
        ],
        &rows,
    )?;
    Ok(())
}

/// Entry of the Nambu Bethe examples. Writes `<prefix>_obs.dat` with
/// the output columns `t, rho(a,b).re/im, XX, Ekin, Enx_MF, Enx_corr,
/// Eph, Etot`.
pub(crate) fn run_nambu_case(args: &[String], approx: &str) -> Result<(), String> {
    let a = parse_example_args(args)?;
    let cfg = load_config(&a.cfgfile, &a.profile, true)?;
    if !APPROXIMATIONS.contains(&approx) {
        return Err(format!("unknown approximation {approx}"));
    }
    if a.validate_only {
        return Ok(());
    }
    let r = run_nambu(approx, &cfg)?;
    let h = contour_params(&cfg)?.h;
    let mut names: Vec<String> = vec!["time".to_string()];
    for a in 0..2usize {
        for b in 0..2usize {
            names.push(format!("rho({a},{b}).re"));
            names.push(format!("rho({a},{b}).im"));
        }
    }
    for n in ["XX", "Ekin", "Enx_MF", "Enx_corr", "Eph", "Etot"] {
        names.push(n.to_string());
    }
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let rows: Vec<Vec<f64>> = (1..=r.g.nt + 1)
        .map(|n| {
            let q = n;
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
    write_table(&obs_path(&cfg)?, &name_refs, &rows)?;
    Ok(())
}
