// Project : KadanoffBaym
// File    : examples/holstein/solver.rs
// Purpose : the three self-consistency loops of the Holstein examples (the
//           Matsubara pass, the start-up pass, and the time steps) shared by
//           the Bethe, impurity, and Nambu models.
//

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::dyson::DysonWorkspace;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::dyson::{dyson_mat, dyson_start, dyson_timestep_ws};
use kadanoffbaym::equilibrium::from_mat_k;
use kadanoffbaym::math::extrapolate_timestep;
use kadanoffbaym::observables::density;

use super::model::{HolsteinPhonon, MatsubaraStage};
use super::util::SolverParams;

use super::model::{
    hybridization, hybridization_self_energy, migdal_self_energy, migdal_start,
    phonon_displacement, phonon_displacement_start, umigdal_self_energy, umigdal_start,
};
use super::nambu::{
    migdal_self_energy_nambu, migdal_start_nambu, umigdal_self_energy_nambu, umigdal_start_nambu,
};
use super::util::add_green_step;

use super::util::prelude::*;

/// The self-energy of one model and approximation.
#[derive(Clone, Copy)]
pub(crate) enum Kernels {
    Migdal,
    Umigdal,
    MigdalNambu,
    UmigdalNambu,
}

impl Kernels {
    /// The approximation `approx` of the normal-state models.
    pub(crate) fn of(approx: &str) -> Result<Self, String> {
        match approx {
            "Migdal" => Ok(Self::Migdal),
            "uMig" => Ok(Self::Umigdal),
            _ => Err(format!("unknown approximation {approx}")),
        }
    }

    /// The approximation `approx` of the Nambu form.
    pub(crate) fn nambu(approx: &str) -> Result<Self, String> {
        match approx {
            "Migdal" => Ok(Self::MigdalNambu),
            "uMig" => Ok(Self::UmigdalNambu),
            _ => Err(format!("unknown approximation {approx}")),
        }
    }

    /// The self-energy at the Matsubara slot (`t = 0`) or at the real-time
    /// step `t`.
    fn step(
        self,
        t: i64,
        g: &ContourGreen,
        sigma: &mut ContourGreen,
        gf: &TimeFunction,
        ph: &mut HolsteinPhonon,
        method: MatsubaraMethod,
    ) {
        match self {
            Self::Migdal => migdal_self_energy(t, g, sigma, gf, ph, method),
            Self::Umigdal => umigdal_self_energy(t, g, sigma, gf, ph),
            Self::MigdalNambu => migdal_self_energy_nambu(t, g, sigma, gf, ph, method),
            Self::UmigdalNambu => umigdal_self_energy_nambu(t, g, sigma, gf, ph),
        }
    }

    /// The self-energy of the start-up steps.
    fn start(
        self,
        g: &ContourGreen,
        sigma: &mut ContourGreen,
        gf: &TimeFunction,
        ph: &mut HolsteinPhonon,
    ) {
        match self {
            Self::Migdal => migdal_start(g, sigma, gf, ph),
            Self::Umigdal => umigdal_start(g, sigma, gf, ph),
            Self::MigdalNambu => migdal_start_nambu(g, sigma, gf, ph),
            Self::UmigdalNambu => umigdal_start_nambu(g, sigma, gf, ph),
        }
    }
}

/// The state carried through the loops.
pub(crate) struct Loop {
    pub(crate) g: ContourGreen,
    pub(crate) g_old: ContourGreen,
    pub(crate) sigma: ContourGreen,
    /// `Hyb + Sigma`, the self-energy of the Dyson equation
    pub(crate) hs: ContourGreen,
    pub(crate) hyb: ContourGreen,
    /// the Hamiltonian of the Dyson equation without `hs`
    pub(crate) h0: TimeFunction,
    pub(crate) gf: TimeFunction,
    pub(crate) jf: TimeFunction,
    pub(crate) ph: HolsteinPhonon,
    /// whether the hybridization follows `g` (the impurity model keeps it
    /// fixed)
    pub(crate) dmft: bool,
    pub(crate) nt: i64,
    pub(crate) k: i64,
    pub(crate) beta: f64,
    pub(crate) h: f64,
}

/// The mean field of the normal-state models: the coupling constant, the
/// phonon displacement, the total density, and the constant term of the
/// Hartree Hamiltonian, indexed by the KB step.
pub(crate) struct MeanField {
    pub(crate) gv: Vec<Complex>,
    pub(crate) x: Vec<Complex>,
    pub(crate) ntot: Vec<Complex>,
    pub(crate) h0const: Complex,
}

/// The Matsubara self-consistency pass. `before_solve` runs before each
/// Dyson solve, with the iteration and the Hamiltonian `h0`.
pub(crate) fn matsubara_loop(
    l: &mut Loop,
    kernels: Kernels,
    s: &SolverParams,
    mut before_solve: impl FnMut(i64, &mut TimeFunction),
) -> Result<(), String> {
    l.g_old.memcpy_from_at(MATSUBARA, &l.g);
    let mut stage = MatsubaraStage::new();
    let mut err = f64::NAN;
    for iter in 0..=s.mats_max_iter {
        let meth = stage.method();
        kernels.step(0, &l.g, &mut l.sigma, &l.gf, &mut l.ph, meth);
        hybridization_self_energy(0, &mut l.hs, &l.hyb, &l.sigma);
        before_solve(iter, &mut l.h0);
        dyson_mat(
            &mut l.g, &l.hs, 0.0, &l.h0, l.beta, l.k, meth, true, None, 1e-12,
        );
        l.g.smul_at(MATSUBARA, Complex::new(0.5, 0.0));
        add_green_step(&mut l.g, &l.g_old, MATSUBARA, Complex::new(0.5, 0.0));
        if l.dmft {
            hybridization(0, &mut l.hyb, &l.g, &l.jf);
        }
        err = l.g.distance_at(&l.g_old, MATSUBARA);
        let name = if stage.fixpoint {
            "fixpoint"
        } else {
            "fourier"
        };
        println!("matsubara ({name}) iteration {iter}  error = {err:.6e}");
        if stage.update(err, s.mats_max_error) {
            l.g_old.memcpy_from_at(MATSUBARA, &l.g);
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
    Ok(())
}

/// The start-up pass on the steps `0..k` from the Matsubara Green's
/// function; `mf` is the mean field where the model has one.
pub(crate) fn bootstrap_loop(
    l: &mut Loop,
    mut mf: Option<&mut MeanField>,
    kernels: Kernels,
    s: &SolverParams,
) -> Result<(), String> {
    from_mat_k(&mut l.g, l.k);
    if l.dmft {
        from_mat_k(&mut l.hyb, l.k);
    }
    for n in 1..=l.k + 1 {
        l.g_old.memcpy_from_at(n - 1, &l.g);
    }
    let mut err = f64::NAN;
    for iter in 0..=s.bootstrap_max_iter {
        kernels.start(&l.g, &mut l.sigma, &l.gf, &mut l.ph);
        if let Some(mf) = mf.as_deref_mut() {
            for n in 1..=l.k + 1 {
                mf.ntot[n as usize] = Complex::new(2.0 * density(&l.g, n - 1).re, 0.0);
            }
            phonon_displacement_start(&mut mf.x, &mf.ntot, &mf.gv, &l.ph);
        }
        for n in 1..=l.k + 1 {
            if let Some(mf) = mf.as_deref_mut() {
                l.h0.block_mut(n - 1)[0] = mf.h0const + mf.x[n as usize] * mf.gv[n as usize];
            }
            hybridization_self_energy(n, &mut l.hs, &l.hyb, &l.sigma);
        }
        dyson_start(&mut l.g, 0.0, &l.h0, &l.hs, l.beta, l.h, l.k);
        err = 0.0;
        for n in 1..=l.k + 1 {
            if l.dmft {
                hybridization(n, &mut l.hyb, &l.g, &l.jf);
            }
            err += l.g.distance_at(&l.g_old, n - 1);
        }
        println!("bootstrap iteration {iter}  error = {err:.6e}");
        if err < s.bootstrap_max_error && iter > 2 {
            break;
        }
        for n in 1..=l.k + 1 {
            l.g_old.memcpy_from_at(n - 1, &l.g);
        }
    }
    if err.is_nan() || err >= s.bootstrap_max_error {
        return Err(format!(
            "ConvergenceError: the bootstrap iteration did not converge (error = {err})"
        ));
    }
    Ok(())
}

/// The time steps after the start-up, `s.corrector_steps` correctors each;
/// `mf` is the mean field where the model has one.
pub(crate) fn propagate(
    l: &mut Loop,
    mut mf: Option<&mut MeanField>,
    kernels: Kernels,
    s: &SolverParams,
) {
    let integ = Integrator::new(l.k);
    let mut dws = DysonWorkspace::new(l.k, l.g.size1, l.g.ntau);
    for n in (l.k + 2)..=(l.nt + 1) {
        // the predictor extrapolates the KB rows n of G and Hyb
        extrapolate_timestep(n - 2, &mut l.g, &l.ph.integ);
        if l.dmft {
            extrapolate_timestep(n - 2, &mut l.hyb, &l.ph.integ);
        }
        for _iter in 0..s.corrector_steps {
            kernels.step(
                n,
                &l.g,
                &mut l.sigma,
                &l.gf,
                &mut l.ph,
                MatsubaraMethod::Fixpoint,
            );
            if let Some(mf) = mf.as_deref_mut() {
                mf.ntot[n as usize] = Complex::new(2.0 * density(&l.g, n - 1).re, 0.0);
                phonon_displacement(n, &mut mf.x, &mf.ntot, &mf.gv, &l.ph);
                l.h0.block_mut(n - 1)[0] = mf.h0const + mf.x[n as usize] * mf.gv[n as usize];
            }
            hybridization_self_energy(n, &mut l.hs, &l.hyb, &l.sigma);
            // `hs` is refreshed every step
            dws.reset();
            dyson_timestep_ws(
                n - 1,
                &mut l.g,
                0.0,
                &l.h0,
                &l.hs,
                l.beta,
                l.h,
                &integ,
                &mut dws,
            );
            if l.dmft {
                hybridization(n, &mut l.hyb, &l.g, &l.jf);
            }
        }
    }
}
