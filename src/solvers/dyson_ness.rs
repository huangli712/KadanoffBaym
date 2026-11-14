// Project : KadanoffBaym
// File    : src/solvers/dyson_ness.rs
// Purpose : the steady-state Dyson equation in the frequency domain
//           (blueprint julia/src/solvers/dyson_ness.jl)

use crate::complex::c64::C64;
use crate::fourier::transforms::{
    integral_transform_to_freq, integral_transform_to_time, FftIntegralMethod,
};
use crate::linalg::mat::DenseMat;
use crate::linalg::lu::mat_inv;
use crate::core::ness::{freq_mode, NessHermMat};

const I: C64 = C64::new(0.0, 1.0);

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
pub fn fermi_cut(x: f64) -> f64 {
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
pub fn sigma_bath(
    omega: f64,
    h: f64,
    bath_type: DysonBathType,
    eta: f64,
    beta: f64,
    mu: f64,
) -> (C64, C64) {
    assert!(omega.is_finite(), "omega must be finite");
    if bath_type == DysonBathType::None {
        return (C64::ZERO, C64::ZERO);
    }
    assert!(h > 0.0, "h must be positive, got {h}");
    assert!(eta >= 0.0, "eta must be >= 0, got {eta}");
    assert!(beta > 0.0, "beta must be positive, got {beta}");

    match bath_type {
        DysonBathType::Const => {
            let fm = fermi_cut(beta * (omega - mu));
            (
                C64::new(0.0, -eta),
                C64::new(0.0, 2.0 * fm * eta),
            )
        }
        DysonBathType::Gauss => {
            let expfac = (-omega * omega).exp();
            let fm = fermi_cut(beta * (omega - mu));
            (
                C64::new(0.0, -eta * expfac),
                C64::new(0.0, 2.0 * fm * eta * expfac),
            )
        }
        DysonBathType::Ohmic => {
            let expfac = (-omega * omega).exp();
            let ret = C64::new(0.0, -eta * omega * expfac);
            let arg = beta * omega;
            let les = if arg > 100.0 {
                C64::ZERO
            } else if arg < -100.0 {
                C64::new(0.0, 2.0 * eta * omega * expfac)
            } else if arg.abs() < 1.0e-6 {
                C64::new(0.0, -2.0 * eta * expfac / beta)
            } else {
                C64::new(0.0, -2.0 * eta * omega * expfac / ((arg).exp_m1()))
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
pub fn dyson_ness(
    g: &mut NessHermMat,
    mu: f64,
    h: &DenseMat,
    sigma: &NessHermMat,
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
        id.set(q, q, C64::ONE);
    }

    // Local copy, the input self-energy is not changed
    let mut s = sigma.clone();
    integral_transform_to_freq(&mut s, FftIntegralMethod::Trapez);

    g.ret.freq.fill(C64::ZERO);
    g.les.freq.fill(C64::ZERO);
    let nfreq = (nft / 2 - 1) / 3;
    for w in (-nfreq + 1)..=(nfreq - 1) {
        let n = w.rem_euclid(nft);
        let omega = freq_mode(&grid, w);
        let (bret, bles) = if bath_type != DysonBathType::None {
            sigma_bath(omega, grid.h, bath_type, bath_eta, bath_beta, bath_mu)
        } else {
            (C64::ZERO, C64::ZERO)
        };
        // ginv = (w + mu) Id - H - S^R(n) - bret Id
        let mut ginv = crate::linalg::mat::scale_c(h, C64::new(-1.0, 0.0));
        let sret = s.ret.freq[n as usize * b..(n as usize + 1) * b].to_vec();
        let mut sret_dm = DenseMat::zeros(d);
        for q in 0..b {
            *sret_dm.raw_mut().get_mut(q).unwrap() = sret[q];
        }
        crate::linalg::mat::axpy_c(&mut ginv, C64::new(-1.0, 0.0), &sret_dm);
        for q in 0..d {
            let v = ginv.get(q, q) + C64::new(omega + mu, 0.0) - bret;
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
        crate::linalg::mat::axpy_c(&mut tmp, C64::ONE, &sles_dm);
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
