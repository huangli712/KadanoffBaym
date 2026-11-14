// Project : KadanoffBaym
// File    : src/dyson/ness.rs
// Purpose : the steady-state Dyson equation in the frequency domain
//           (blueprint julia/src/solvers/dyson_ness.jl)

use crate::complex::Complex;
use crate::fourier::transforms::{
    integral_transform_to_freq, integral_transform_to_time, FftIntegralMethod,
};
use crate::linalg::mat::DenseMat;
use crate::linalg::lu::mat_inv;
use crate::types::steady::SteadyGreen;

const I: Complex = Complex::new(0.0, 1.0);

/// Type of the bath regularization of the steady-state Dyson equation
/// (blueprint `DysonBathType`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DysonBathType {
    None,
    Const,
    Gauss,
    Ohmic,
}

/// Fermi function `1/(e^x + 1)` with the cut-offs `x > 100` (0) and
/// `x < -100` (1) (blueprint `_fermi_cut`).
pub(crate) fn fermi_cut(x: f64) -> f64 {
    if x > 100.0 {
        0.0
    } else if x < -100.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Retarded and lesser self-energy of the bath that regularizes the
/// steady-state Dyson equation at the frequency `omega` (blueprint
/// `sigma_bath`).
///
/// NESSi writes the damping factor as `exp(-omega*omega/omegac*omegac)`
/// which evaluates to `e^{-omega^2}`; this is reproduced exactly. `h`
/// is only checked.
///
/// Retarded and lesser self-energy of the bath that regularizes the
/// steady-state Dyson equation, at the frequency `omega`:
///
/// * `BATH_NONE`  : ``\\Sigma^R = \\Sigma^< = 0``.
/// * `BATH_CONST` : ``\\Sigma^R = -i\\eta``, ``\\Sigma^< = 2i\\eta f(\\omega-\\mu)``.
/// * `BATH_GAUSS` : as `BATH_CONST`, multiplied by ``e^{-\\omega^2}``.
/// * `BATH_OHMIC` : ``\\Sigma^R = -i\\eta\\omega e^{-\\omega^2}``, the lesser
///   component uses the Bose factor of ``\\beta\\omega``, with the limit
///   ``-2i\\eta e^{-\\omega^2}/\\beta`` for ``|\\beta\\omega| < 10^{-6}``.
///
/// NESSi writes the damping factor as `exp(-omega*omega/omegac*omegac)` with
/// `omegac = π/(10h)`. Since the division and the multiplication are
/// evaluated from the left, the factor is ``e^{-\\omega^2}`` and does not
/// depend on `h`; this is reproduced exactly. `h` is only checked.
///
/// ### Arguments
/// * omega     -> Frequency ω.
/// * h         -> Time step.
/// * bath_type -> Type of the bath.
/// * eta       -> Coupling η ≥ 0.
/// * beta      -> Inverse temperature of the bath, β > 0.
/// * mu        -> Chemical potential of the bath (not used by `BATH_OHMIC`).
///
/// ### Returns
/// * ret -> Retarded self-energy, a C64.
/// * les -> Lesser self-energy, a C64.
///
/// *References* :
///
/// Please see [`NESSi`] `ness2_dyson_impl.hpp`, `sigma_bath`.
pub fn sigma_bath(
    omega: f64,
    h: f64,
    bath_type: DysonBathType,
    eta: f64,
    beta: f64,
    mu: f64,
) -> (Complex, Complex) {
    assert!(omega.is_finite(), "omega must be finite");
    if bath_type == DysonBathType::None {
        return (Complex::ZERO, Complex::ZERO);
    }
    assert!(h > 0.0, "h must be positive, got {h}");
    assert!(eta >= 0.0, "eta must be >= 0, got {eta}");
    assert!(beta > 0.0, "beta must be positive, got {beta}");

    match bath_type {
        DysonBathType::Const => {
            let fm = fermi_cut(beta * (omega - mu));
            (
                Complex::new(0.0, -eta),
                Complex::new(0.0, 2.0 * fm * eta),
            )
        }
        DysonBathType::Gauss => {
            let expfac = (-omega * omega).exp();
            let fm = fermi_cut(beta * (omega - mu));
            (
                Complex::new(0.0, -eta * expfac),
                Complex::new(0.0, 2.0 * fm * eta * expfac),
            )
        }
        DysonBathType::Ohmic => {
            let expfac = (-omega * omega).exp();
            let ret = Complex::new(0.0, -eta * omega * expfac);
            let arg = beta * omega;
            let les = if arg > 100.0 {
                Complex::ZERO
            } else if arg < -100.0 {
                Complex::new(0.0, 2.0 * eta * omega * expfac)
            } else if arg.abs() < 1.0e-6 {
                Complex::new(0.0, -2.0 * eta * expfac / beta)
            } else {
                Complex::new(0.0, -2.0 * eta * omega * expfac / ((arg).exp_m1()))
            };
            (ret, les)
        }
        DysonBathType::None => unreachable!(),
    }
}

/// Solve the steady-state Dyson equation in the frequency domain
/// (blueprint `dyson_ness!`),
///
/// `G^R(w) = [w + mu - H - S^R(w) - S^R_b(w)]^{-1}`,
/// `G^<(w) = G^R(w) [S^<(w) + S^<_b(w)] G^R(w)^dagger`.
///
/// The self-energy is transformed by the trapezoidal rule (a local copy
/// is used, `sigma` is not changed). Only the central band
/// `|m| < (Nft/2 - 1)/3` of the frequency modes is filled. Finally `G`
/// is transformed back to the time domain.
#[allow(clippy::too_many_arguments)]
///
/// Solve the steady-state Dyson equation in the frequency domain,
///
/// ```math
/// G^R(\\omega) = [\\omega + \\mu - H - \\Sigma^R(\\omega) - \\Sigma^R_b(\\omega)]^{-1},\\quad
/// G^<(\\omega) = G^R(\\omega)[\\Sigma^<(\\omega) + \\Sigma^<_b(\\omega)]G^R(\\omega)^\\dagger,
/// ```
///
/// where ``\\Sigma_b`` is the bath self-energy, see `sigma_bath()`. The
/// self-energy is transformed by the trapezoidal rule. Only the central band
/// ``|m| < (N_{ft}/2-1)/3`` of the frequency modes is filled, all the other
/// modes of `G` are set to zero. Finally `G` is transformed back to the time
/// domain. `Sigma` is not changed (NESSi transforms it in place).
///
/// ### Arguments
/// * G         -> On return, the Green's function (both domains).
/// * mu        -> Chemical potential.
/// * H         -> Hamiltonian, a square matrix.
/// * Sigma     -> Self-energy, the time data is the input.
/// * method    -> Only FFT_TRAPEZ.
/// * bath_type -> Type of the bath, see `sigma_bath()`.
/// * bath_eta  -> Bath coupling.
/// * bath_beta -> Bath inverse temperature.
/// * bath_mu   -> Bath chemical potential.
///
/// ### Returns
/// * G -> The Green's function.
///
/// *References* :
///
/// Please see [`NESSi`] `ness2_dyson_impl.hpp`, `dyson` and `dyson_fft`.
pub fn dyson_ness(
    g: &mut SteadyGreen,
    mu: f64,
    h: &DenseMat,
    sigma: &SteadyGreen,
    method: FftIntegralMethod,
    bath_type: DysonBathType,
    bath_eta: f64,
    bath_beta: f64,
    bath_mu: f64,
) {
    assert!(
        method == FftIntegralMethod::Trapez,
        "dyson_ness: only FFT_TRAPEZ is implemented"
    );
    assert!(
        g.is_compatible(sigma),
        "G and Sigma must have the same grid and size"
    );
    assert!(
        h.nrows() == g.ret.size1 && h.ncols() == g.ret.size2,
        "H must have the size {}x{}",
        g.ret.size1,
        g.ret.size2
    );

    let grid = g.grid;
    let nft = grid.nft;
    let d = g.ret.size1;
    let b = d * d;
    let mut id = DenseMat::zeros(d);
    for q in 0..d {
        id.set(q, q, Complex::ONE);
    }

    // Local copy, the input self-energy is not changed
    let mut s = sigma.clone();
    integral_transform_to_freq(&mut s, FftIntegralMethod::Trapez);

    g.ret.freq.fill(Complex::ZERO);
    g.les.freq.fill(Complex::ZERO);
    let nfreq = (nft / 2 - 1) / 3;
    for w in (-nfreq + 1)..=(nfreq - 1) {
        let n = w.rem_euclid(nft);
        let omega = grid.freq_mode(w);
        let (bret, bles) = if bath_type != DysonBathType::None {
            sigma_bath(omega, grid.h, bath_type, bath_eta, bath_beta, bath_mu)
        } else {
            (Complex::ZERO, Complex::ZERO)
        };
        // ginv = (w + mu) Id - H - S^R(n) - bret Id
        let mut ginv = crate::linalg::mat::scale_c(h, Complex::new(-1.0, 0.0));
        let sret = s.ret.freq[n as usize * b..(n as usize + 1) * b].to_vec();
        let mut sret_dm = DenseMat::zeros(d);
        for q in 0..b {
            *sret_dm.raw_mut().get_mut(q).unwrap() = sret[q];
        }
        crate::linalg::mat::axpy_c(&mut ginv, Complex::new(-1.0, 0.0), &sret_dm);
        for q in 0..d {
            let v = ginv.get(q, q) + Complex::new(omega + mu, 0.0) - bret;
            ginv.set(q, q, v);
        }
        let gret = mat_inv(&ginv);
        g.ret.freq[n as usize * b..(n as usize + 1) * b].copy_from_slice(gret.raw());
        // G^< = gret (S^< + bles Id) gret'
        let mut tmp = DenseMat::zeros(d);
        let sles = s.les.freq[n as usize * b..(n as usize + 1) * b].to_vec();
        let mut sles_dm = DenseMat::zeros(d);
        for q in 0..b {
            *sles_dm.raw_mut().get_mut(q).unwrap() = sles[q];
        }
        crate::linalg::mat::axpy_c(&mut tmp, Complex::ONE, &sles_dm);
        for q in 0..d {
            let v = tmp.get(q, q) + bles;
            tmp.set(q, q, v);
        }
        let prod = gret.mat_mul(&tmp.mat_mul(&gret.adjoint()));
        g.les.freq[n as usize * b..(n as usize + 1) * b].copy_from_slice(prod.raw());
        let _ = I;
    }

    integral_transform_to_time(g, FftIntegralMethod::Trapez);
}
