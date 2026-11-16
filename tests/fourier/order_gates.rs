// Project : KadanoffBaym
// File    : tests/fourier/order_gates.rs
// Purpose : release-only convergence-order gates of the Matsubara
//           Fourier transforms

use super::util::prelude::*;
use super::util::{check_order_gate, is_release};
use kadanoffbaym::fourier::dftcorr_cubic;
use kadanoffbaym::fourier::dftcorr_linear;
use kadanoffbaym::fourier::get_omega;
use kadanoffbaym::fourier::{matsubara_dft, matsubara_ft};

#[test]
fn release_order_gates() {
    if !is_release() {
        return;
    }
    let (beta, eps) = (2.0f64, 0.7);
    let ns: Vec<i64> = vec![32, 64, 128, 256];
    for sign in [Sign::Fermi, Sign::Bose] {
        let errs = |order: i64| -> Vec<f64> {
            ns.iter()
                .map(|&n| mf_err(&mf_pole(eps, n, beta, sign), eps, beta, sign, order))
                .collect()
        };
        assert!(
            check_order_gate(&ns, &errs(3), 4.0, 1e-10, 1e-13, 0.3).ok,
            "cubic {sign:?}"
        );
        assert!(
            check_order_gate(&ns, &errs(1), 2.0, 1e-5, 1e-13, 0.3).ok,
            "linear {sign:?}"
        );
        assert!(
            check_order_gate(&ns, &errs(0), 1.0, 0.5, 1e-13, 0.3).ok,
            "raw {sign:?}"
        );

        // mutant: the corrected transform without the endpoint terms
        let mutant = |n: i64, order: i64| -> f64 {
            let g = mf_pole(eps, n, beta, sign);
            let d = matsubara_dft(&g, sign);
            let dtau = beta / n as f64;
            let one = if sign == Sign::Fermi { 1i64 } else { 0 };
            let mut worst = 0.0f64;
            for m in [-7i64, -1, 0, 3, 11] {
                let theta = (2 * m + one) as f64 * std::f64::consts::PI / n as f64;
                let w = if order == 3 {
                    dftcorr_cubic(theta).0
                } else {
                    dftcorr_linear(theta).0
                };
                let x = Complex::new(dtau * w, 0.0) * d[m.rem_euclid(n) as usize][0];
                let om = get_omega(m, beta, sign);
                let ex = Complex::ONE / Complex::new(-eps, om);
                worst = worst.max((x - ex).abs() / ex.abs());
            }
            worst
        };
        let m3: Vec<f64> = ns.iter().map(|&n| mutant(n, 3)).collect();
        assert!(
            !check_order_gate(&ns, &m3, 4.0, 1e-10, 1e-13, 0.3).ok,
            "cubic mutant {sign:?}"
        );
        let m1: Vec<f64> = ns.iter().map(|&n| mutant(n, 1)).collect();
        assert!(
            !check_order_gate(&ns, &m1, 2.0, 1e-5, 1e-13, 0.3).ok,
            "linear mutant {sign:?}"
        );

        // the raw transform with the wrong statistics
        let wrong = if sign == Sign::Fermi {
            Sign::Bose
        } else {
            Sign::Fermi
        };
        let e0: Vec<f64> = ns
            .iter()
            .map(|&n| {
                let g = mf_pole(eps, n, beta, sign);
                let d = matsubara_dft(&g, wrong);
                let mut worst = 0.0f64;
                for m in [-1i64, 3] {
                    let x = matsubara_ft(m, &g, &d, wrong, beta, 0);
                    let om = get_omega(m, beta, sign);
                    let ex = Complex::ONE / Complex::new(-eps, om);
                    worst = worst.max((x[0] - ex).abs() / ex.abs());
                }
                worst
            })
            .collect();
        assert!(
            !check_order_gate(&ns, &e0, 1.0, 0.5, 1e-13, 0.3).ok,
            "wrong stats {sign:?}"
        );
    }
}

/// Fermionic/bosonic single pole at `eps` (blueprint `mf_pole`).
fn mf_pole(eps: f64, n: i64, beta: f64, sign: Sign) -> MatComp {
    let mut g = MatComp::new(n as usize, 1, 1, Complex::ZERO);
    let z = if sign == Sign::Fermi {
        1.0 + (-beta * eps).exp()
    } else {
        1.0 - (-beta * eps).exp()
    };
    for r in 0..=n {
        g.block_mut(r as usize)[0] =
            Complex::new(-(-(eps * r as f64 * beta / n as f64)).exp() / z, 0.0);
    }
    g
}

/// Maximal relative error at selected frequencies (blueprint `mf_err`).
fn mf_err(g: &MatComp, eps: f64, beta: f64, sign: Sign, order: i64) -> f64 {
    let raw = matsubara_dft(g, sign);
    let mut err = 0.0f64;
    for m in [-7i64, -1, 0, 3, 11] {
        let x = matsubara_ft(m, g, &raw, sign, beta, order);
        let om = get_omega(m, beta, sign);
        let ex = Complex::ONE / Complex::new(-eps, om);
        err = err.max((x[0] - ex).abs() / ex.abs());
    }
    err
}
