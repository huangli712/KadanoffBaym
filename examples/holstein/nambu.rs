// Project : KadanoffBaym
// File    : examples/holstein/nambu.rs
// Purpose : Nambu (superconducting) self-energies of the Holstein model.
//

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{ContourGreen, TimeFunction};

use kadanoffbaym::convolution::{bubble1, bubble2};

use super::model::HolsteinPhonon;

use super::model::{form_g_g_g, phonon_start, phonon_timestep};
use super::util::{add_green_step, matsubara_step};

/// `Pi = -sum_ab i gGg_ab G_ba` at the KB time step `n`.
fn phonon_polarization_nambu(n: i64, g: &ContourGreen, ph: &mut HolsteinPhonon) {
    let mut first = true;
    for a in 0..2usize {
        for b in 0..2usize {
            bubble1(
                matsubara_step(n),
                &mut ph.pol,
                0,
                0,
                &ph.ggg,
                &ph.ggg,
                a,
                b,
                g,
                g,
                a,
                b,
            );
            if first {
                ph.pi.memcpy_from_at(matsubara_step(n), &ph.pol);
                ph.pi.smul_at(matsubara_step(n), Complex::new(-1.0, 0.0));
                first = false;
            } else {
                add_green_step(
                    &mut ph.pi,
                    &ph.pol,
                    matsubara_step(n),
                    Complex::new(-1.0, 0.0),
                );
            }
        }
    }
}

fn nambu_self_energy(n: i64, sigma: &mut ContourGreen, d: &ContourGreen, ph: &HolsteinPhonon) {
    for a in 0..2usize {
        for b in 0..2usize {
            bubble2(
                matsubara_step(n),
                sigma,
                a,
                b,
                d,
                d,
                0,
                0,
                &ph.ggg,
                &ph.ggg,
                a,
                b,
            );
        }
    }
}

/// Nambu Migdal self-energy at the KB time step `n`.
pub(crate) fn migdal_self_energy_nambu(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
    method: MatsubaraMethod,
) {
    form_g_g_g(n, g, gf, ph);
    phonon_polarization_nambu(n, g, ph);
    phonon_timestep(n, ph, method);
    nambu_self_energy(n, sigma, &ph.d, ph);
}

/// Nambu Migdal self-energy of the start-up time steps.
pub(crate) fn migdal_start_nambu(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        form_g_g_g(n, g, gf, ph);
        phonon_polarization_nambu(n, g, ph);
    }
    phonon_start(ph);
    for n in 1..=ph.k + 1 {
        nambu_self_energy(n, sigma, &ph.d, ph);
    }
}

/// Nambu uMig self-energy at the KB time step `n`.
pub(crate) fn umigdal_self_energy_nambu(
    n: i64,
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    form_g_g_g(n, g, gf, ph);
    nambu_self_energy(n, sigma, &ph.d0, ph);
}

/// Nambu uMig self-energy of the start-up time steps.
pub(crate) fn umigdal_start_nambu(
    g: &ContourGreen,
    sigma: &mut ContourGreen,
    gf: &TimeFunction,
    ph: &mut HolsteinPhonon,
) {
    for n in 1..=ph.k + 1 {
        umigdal_self_energy_nambu(n, g, sigma, gf, ph);
    }
}
