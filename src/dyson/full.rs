// Project : KadanoffBaym
// File    : src/dyson/full.rs
// Purpose : Dyson equation: the public drivers (the full contour,
//           the single time step, and its workspace-reusing form)

use crate::base::MatsubaraMethod;
use crate::types::{ContourGreen, TimeFunction};
use crate::weights::Integrator;

use crate::dyson::workspace::DysonWorkspace;

use crate::dyson::check::check_dyson_rt;
use crate::dyson::mat::dyson_mat;
use crate::dyson::start::dyson_start;
use crate::dyson::timestep::dyson_timestep_less_ws;
use crate::dyson::timestep::dyson_timestep_lmix_ws;
use crate::dyson::timestep::dyson_timestep_ret_ws;

///
/// *Remarks* : *Dyson Equation*
///
/// The Dyson equation for the contour-ordered Green's function reads
///
/// ```math
/// \begin{equation}
/// [i\partial_t + \mu - H(t)] G(t,t') - [\Sigma * G](t,t')
///     = \delta_{\mathcal{C}}(t,t').
/// \end{equation}
/// ```
///
/// It is a Volterra integro-differential equation (VIDE). It is solved in
/// three steps:
///
/// 1. Matsubara component, `dyson_mat`.
///
/// 2. Start-up of the real time components at the first 𝑘 + 1 time steps,
///    `dyson_start`.
///
/// 3. Time stepping of the real time components, `dyson_timestep`.
///
///
/// Solve the Dyson equation for all the components of `g`: Matsubara,
/// start-up, and then all the remaining real time steps. `h` is the
/// Hamiltonian, `sigma` the self-energy.
#[allow(clippy::too_many_arguments)]
pub fn dyson(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
    matsubara_method: MatsubaraMethod,
    force_hermitian: bool,
) {
    let k = solve_order;
    check_dyson_rt(g, sigma, h, k, g.nt as i64);

    dyson_mat(
        g,
        sigma,
        mu,
        h,
        beta,
        solve_order,
        matsubara_method,
        force_hermitian,
        None,
        1e-12,
    );
    dyson_start(g, mu, h, sigma, beta, hstep, solve_order);
    let integ = Integrator::new(k);
    let mut dws = DysonWorkspace::new(k, g.size1, g.ntau);
    for tstp in (k + 1)..=(g.nt as i64) {
        dyson_timestep_ws(tstp, g, mu, h, sigma, beta, hstep, &integ, &mut dws);
    }
}

/// Solve the Dyson equation at the real time step `tstp > solve_order`.
/// The previous time steps of `g` must be known; the Matsubara component
/// is not recalculated.
///
/// Convenience wrapper: builds a fresh workspace on every call. In
/// time-step loops build a [`DysonWorkspace`] (and an `Integrator`)
/// once and call [`dyson_timestep_ws`] instead.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    assert!(
        tstp > k,
        "dyson_timestep: time step {tstp} must be larger than {k}"
    );
    check_dyson_rt(g, sigma, h, k, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    let mut ws = DysonWorkspace::new(k, g.size1, g.ntau);
    dyson_timestep_ws(tstp, g, mu, h, sigma, beta, hstep, &integ, &mut ws);
}

/// Workspace-reusing variant of [`dyson_timestep`]: pass one
/// `DysonWorkspace` (and one `Integrator`) built before the time-step
/// loop. Reuse is valid while `sigma` is unchanged; the lesser-path
/// convolution cache in the workspace assumes the same input sequence.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWorkspace,
) {
    let k = integ.k;
    assert!(
        tstp > k,
        "dyson_timestep: time step {tstp} must be larger than {k}"
    );
    check_dyson_rt(g, sigma, h, k, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    dyson_timestep_ret_ws(tstp, g, mu, h, sigma, hstep, integ, ws);
    dyson_timestep_lmix_ws(tstp, g, mu, h, sigma, beta, hstep, integ, ws);
    dyson_timestep_less_ws(tstp, g, mu, h, sigma, beta, hstep, integ, ws);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::complex::Complex;
    use crate::linalg::DenseMat;
    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

    fn const_hamiltonian(nt: usize) -> (TimeFunction, DenseMat) {
        let mut h = TimeFunction::new(nt, 2, 2, Complex::ZERO);
        let m = DenseMat::from_rows(&[
            vec![Complex::new(-0.4, 0.0), Complex::new(0.0, 0.1)],
            vec![Complex::new(0.0, -0.1), Complex::new(0.6, 0.0)],
        ]);
        for t in -1..=(nt as i64) {
            for i in 0..2 {
                for j in 0..2 {
                    h.set(t, i, j, m.get(i, j));
                }
            }
        }
        (h, m)
    }

    /// Sigma = 0: the full `dyson` driver must reproduce the free
    /// propagator of from_ham in all four components.
    #[test]
    fn sigma_zero_full_dyson_matches_from_ham() {
        let (nt, ntau) = (20usize, 100);
        let (beta, hstep, mu, k) = (4.0f64, 0.05, 0.0, 5i64);
        let c = Contour::new(nt as i64, ntau as i64, 2, 2, hstep * nt as f64, beta);
        let (h, h0) = const_hamiltonian(nt);
        let sigma = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        dyson(
            &mut g,
            mu,
            &h,
            &sigma,
            beta,
            hstep,
            k,
            MatsubaraMethod::Fixpoint,
            true,
        );

        let mut want = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut want, mu, &h0, beta, hstep);

        let mut e_mat = 0.0f64;
        for q in 0..want.mat.raw().len() {
            e_mat = e_mat.max((g.mat.raw()[q] - want.mat.raw()[q]).abs());
        }
        let mut e_ret = 0.0f64;
        let mut e_lmix = 0.0f64;
        let mut e_less = 0.0f64;
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for q in 0..4 {
                    e_ret = e_ret
                        .max((g.ret.raw_block_at(i, j)[q] - want.ret.raw_block_at(i, j)[q]).abs());
                    e_less = e_less.max(
                        (g.less.raw_block_at(j, i)[q] - want.less.raw_block_at(j, i)[q]).abs(),
                    );
                }
            }
            for m in 0..=(ntau as i64) {
                for q in 0..4 {
                    e_lmix = e_lmix.max(
                        (g.lmix.raw_block_at(i, m)[q] - want.lmix.raw_block_at(i, m)[q]).abs(),
                    );
                }
            }
        }
        assert!(e_mat < 1e-10, "mat {e_mat}");
        assert!(e_ret < 1e-6, "ret {e_ret}");
        assert!(e_lmix < 1e-6, "lmix {e_lmix}");
        assert!(e_less < 1e-6, "less {e_less}");
    }
}
