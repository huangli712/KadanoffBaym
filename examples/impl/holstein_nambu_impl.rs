//
// Nambu (superconducting) self-energies of the Holstein model (the Rust
// port of julia/tutor/holstein_bethe/holstein_nambu_impl.jl, from NESSi
// examples/programs/Holstein_impurity_impl.cpp, Sigma_Mig_sc /
// Sigma_uMig_sc). The coupling is `g sigma_3` (2x2):
//
//   Pi(t,t')    = -i sum_ab gGg_ab(t,t') G_ab(t',t)
//   Sigma_ab    =  i D(t,t') gGg_ab(t,t')   (D0 for uMig)
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::holstein_impl as hi;
use hi::{mt, HolsteinPhonon};
use kadanoffbaym::convolution::bubble::{bubble1, bubble2};
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::solvers::dyson_mat::MatsubaraMethod;
use kadanoffbaym::core::green::{ContourGreen, TimeFunction};

/// `Pi = -sum_ab i gGg_ab G_ba` at the KB time step `n`.
fn pi_nambu(n: i64, g: &ContourGreen, ph: &mut HolsteinPhonon) {
    let mut first = true;
    for a in 0..2usize {
        for b in 0..2usize {
            bubble1(
                mt(n), &mut ph.pol, 0, 0, &ph.ggg, &ph.ggg, a, b, g, g, a, b,
            );
            if first {
                hi::copy_step(&ph.pol, &mut ph.pi, mt(n));
                ph.pi.smul_at(mt(n), C64::new(-1.0, 0.0));
                first = false;
            } else {
                hi::add_step(&mut ph.pi, &ph.pol, mt(n), C64::new(-1.0, 0.0));
            }
        }
    }
}

fn sigma_nambu(n: i64, sigma: &mut ContourGreen, d: &ContourGreen, ph: &HolsteinPhonon) {
    for a in 0..2usize {
        for b in 0..2usize {
            bubble2(mt(n), sigma, a, b, d, d, 0, 0, &ph.ggg, &ph.ggg, a, b);
        }
    }
}

/// Nambu Migdal self-energy at the KB time step `n` (NESSi
/// `Hols::Sigma_Mig_sc`).
pub fn sigma_mig_sc(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
    method: MatsubaraMethod,
) {
    hi::gg_g(n, g, gf, ph);
    pi_nambu(n, g, ph);
    hi::step_d(n, ph, method);
    sigma_nambu(n, sigma, &ph.d, ph);
}

/// Nambu Migdal self-energy of the start-up time steps.
pub fn sigma_mig_sc_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        hi::gg_g(n, g, gf, ph);
        pi_nambu(n, g, ph);
    }
    hi::start_d(ph);
    for n in 1..=ph.k + 1 {
        sigma_nambu(n, sigma, &ph.d, ph);
    }
}

/// Nambu uMig self-energy at the KB time step `n` (NESSi
/// `Hols::Sigma_uMig_sc`).
pub fn sigma_umig_sc(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    hi::gg_g(n, g, gf, ph);
    sigma_nambu(n, sigma, &ph.d0, ph);
}

/// Nambu uMig self-energy of the start-up time steps.
pub fn sigma_umig_sc_start(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        sigma_umig_sc(n, g, sigma, gf, ph);
    }
}
