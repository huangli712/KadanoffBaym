// Project : KadanoffBaym
// File    : tests/matsubara_fourier.rs
// Purpose : port of julia/test/t_matsubara_fourier.jl — Matsubara
//           Fourier transform of single-pole and matrix functions

mod common;

use common::prelude::*;
use kadanoffbaym::fourier::{
    dftcorr_cubic, dftcorr_linear, first_order_tail, get_omega, get_tau, matsubara_dft,
    matsubara_ft,
};
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::types::green::MatComp;

fn cis(x: f64) -> C64 {
    C64::from_polar(1.0, x)
}

/// Fermionic/bosonic single pole at `eps` (blueprint `mf_pole`).
fn mf_pole(eps: f64, n: i64, beta: f64, sign: Sign) -> MatComp {
    let mut g = MatComp::new(n as usize, 1, 1, C64::ZERO);
    let z = if sign == Sign::Fermi {
        1.0 + (-beta * eps).exp()
    } else {
        1.0 - (-beta * eps).exp()
    };
    for r in 0..=n {
        g.block_mut(r as usize)[0] = C64::new(-(-(eps * r as f64 * beta / n as f64)).exp() / z, 0.0);
    }
    g
}

/// U = [[1, i], [i, 1]] / sqrt(2)
fn mf_u() -> DenseMat {
    let s = C64::new(1.0 / 2.0f64.sqrt(), 0.0);
    DenseMat::from_vec(
        2,
        2,
        vec![s, C64::new(0.0, 1.0) * s, C64::new(0.0, 1.0) * s, s],
    )
}

/// 2x2 non-diagonal matrix U diag(g(e1), g(e2)) U^dag (blueprint
/// `mf_matrix`).
fn mf_matrix(eps: &[f64], n: i64, beta: f64, sign: Sign) -> MatComp {
    let g1 = mf_pole(eps[0], n, beta, sign);
    let g2 = mf_pole(eps[1], n, beta, sign);
    let u = mf_u();
    let mut g = MatComp::new(n as usize, 2, 2, C64::ZERO);
    for r in 0..=(n as usize) {
        let mut diag = DenseMat::zeros(2);
        diag.set(0, 0, g1.block(r)[0]);
        diag.set(1, 1, g2.block(r)[0]);
        let m = u.mat_mul(&diag).mat_mul(&u.adjoint());
        let blk = g.block_mut(r);
        for q in 0..4 {
            blk[q] = m.raw()[q];
        }
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
        let ex = C64::ONE / C64::new(-eps, om);
        err = err.max((x[0] - ex).abs() / ex.abs());
    }
    err
}

#[test]
fn grids() {
    assert_eq!(get_tau(8, 2.0, 17), 1.0);
    assert_eq!(get_tau(0, 2.0, 17), 0.0);
    assert_eq!(get_tau(16, 2.0, 17), 2.0);
    assert!((get_omega(0, 2.0, Sign::Fermi) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!(get_omega(0, 2.0, Sign::Bose).abs() < 1e-15);
    assert!((get_omega(-1, 2.0, Sign::Fermi) + std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!((get_omega(-3, 2.0, Sign::Bose) + 3.0 * std::f64::consts::PI).abs() < 1e-14);
}

#[test]
#[should_panic]
fn fail_get_tau_negative_beta() {
    let _ = get_tau(1, -1.0, 17);
}

#[test]
#[should_panic]
fn fail_get_tau_one_point() {
    let _ = get_tau(1, 1.0, 1);
}

#[test]
#[should_panic]
fn fail_get_omega_zero_beta() {
    let _ = get_omega(1, 0.0, Sign::Fermi);
}

#[test]
fn first_order_tail_fermi_and_bose() {
    let n = 16usize;
    let (beta, eps) = (2.0f64, 0.7);
    for sign in [Sign::Fermi, Sign::Bose] {
        let c = vec![C64::new(2.0, 0.0)];
        let mut d = vec![vec![C64::ZERO; 1]; n + 1];
        first_order_tail(&mut d, &c, 4.0, sign);
        assert_eq!(c, vec![C64::new(2.0, 0.0)], "coeff changed");
        if sign == Sign::Fermi {
            for blk in &d {
                assert_eq!(blk[0], C64::new(-1.0, 0.0));
            }
        } else {
            assert_eq!(d[0][0], C64::new(-1.0, 0.0));
            assert_eq!(d[n][0], C64::new(1.0, 0.0));
            assert!(d[n / 2][0].abs() < 1e-15);
        }
        // the tail removes the jump: G - c0 is (anti)periodic
        let g = mf_pole(eps, n as i64, beta, sign);
        let sg = C64::new(sign.as_i64() as f64, 0.0);
        let a0 = sg * g.block(n)[0] - g.block(0)[0];
        let mut t = vec![vec![C64::ZERO; 1]; n + 1];
        first_order_tail(&mut t, &[a0], beta, sign);
        let lhs = g.block(0)[0] - t[0][0];
        let rhs = sg * (g.block(n)[0] - t[n][0]);
        assert!((lhs - rhs).abs() <= 1e-12, "{sign:?}");

        // the transform of the tail is a0/(i omega), exact for the
        // cubic order
        let mut tt = MatComp::new(n, 1, 1, C64::ZERO);
        for r in 0..=n {
            tt.block_mut(r)[0] = t[r][0];
        }
        let raw = matsubara_dft(&tt, sign);
        for m in [-3i64, 1, 5] {
            let x = matsubara_ft(m, &tt, &raw, sign, beta, 3);
            let om = get_omega(m, beta, sign);
            let want = a0 / C64::new(0.0, om);
            assert!((x[0] - want).abs() <= 1e-12, "m {m}: {} vs {want}", x[0]);
        }
    }
    // matrix coefficients
    for sign in [Sign::Fermi, Sign::Bose] {
        let cm = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 2.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
        ];
        let mut dm = vec![vec![C64::ZERO; 4]; n + 1];
        first_order_tail(&mut dm, &cm, 2.0, sign);
        for q in 0..4 {
            let want = cm[q].scale(-0.5);
            assert!((dm[0][q] - want).abs() < 1e-15, "q {q}");
        }
    }
}

#[test]
#[should_panic]
fn fail_tail_coeff_size_mismatch() {
    let mut dm = vec![vec![C64::ZERO; 4]; 17];
    let _ = first_order_tail(&mut dm, &[C64::ONE], 2.0, Sign::Fermi);
}

#[test]
#[should_panic]
fn fail_tail_negative_beta() {
    let mut dm = vec![vec![C64::ZERO; 1]; 17];
    let _ = first_order_tail(&mut dm, &[C64::ONE], -1.0, Sign::Fermi);
}

#[test]
fn raw_dft_delta_and_matrix() {
    // a delta at tau = 0 transforms to all ones (boson)
    let mut g = MatComp::new(16, 1, 1, C64::ZERO);
    g.block_mut(0)[0] = C64::ONE;
    let d = matsubara_dft(&g, Sign::Bose);
    for m in 0..=16usize {
        assert!((d[m][0] - C64::ONE).abs() < 1e-14, "m {m}");
    }
    // direct sum for a 2x2 matrix function, m = 0..N
    let (n, beta) = (8i64, 2.0);
    let g = mf_matrix(&[0.3, 1.1], n, beta, Sign::Fermi);
    let d = matsubara_dft(&g, Sign::Fermi);
    for m in 0..=n {
        let mut s = [C64::ZERO; 4];
        for r in 0..=n {
            let arg = std::f64::consts::PI * (2 * m + 1) as f64 * r as f64 / n as f64;
            let w = cis(arg);
            let blk = g.block(r as usize);
            for q in 0..4 {
                s[q] += w * blk[q];
            }
        }
        for q in 0..4 {
            assert!(
                (d[m as usize][q] - s[q]).abs() < 1e-13,
                "m {m} q {q}: {} vs {}",
                d[m as usize][q],
                s[q]
            );
        }
    }
}

#[test]
fn corrected_transform_poles_and_matrix() {
    let (beta, eps) = (2.0f64, 0.7);
    for sign in [Sign::Fermi, Sign::Bose] {
        // cubic order on a moderate grid, negative and high modes
        assert!(mf_err(&mf_pole(eps, 256, beta, sign), eps, beta, sign, 3) <= 1e-10);
        assert!(mf_err(&mf_pole(eps, 64, beta, sign), eps, beta, sign, 1) <= 1e-4);
        assert!(mf_err(&mf_pole(eps, 64, beta, sign), eps, beta, sign, 0) <= 1.0);
        // periodic mapping of the raw transform: m and m + N share the
        // raw value; with no correction the results are identical
        let n = 32i64;
        let g = mf_pole(eps, n, beta, sign);
        let d = matsubara_dft(&g, sign);
        let x0 = matsubara_ft(2, &g, &d, sign, beta, 0);
        let x1 = matsubara_ft(2 + n, &g, &d, sign, beta, 0);
        assert_eq!(x0, x1);
        // 2x2 non-diagonal matrix
        let n = 256i64;
        let g = mf_matrix(&[0.3, 1.1], n, beta, sign);
        let d = matsubara_dft(&g, sign);
        let u = mf_u();
        for m in [-5i64, 0, 4] {
            let x = matsubara_ft(m, &g, &d, sign, beta, 3);
            let iom = C64::new(0.0, get_omega(m, beta, sign));
            let mut exd = DenseMat::zeros(2);
            exd.set(0, 0, C64::ONE / (iom - C64::new(0.3, 0.0)));
            exd.set(1, 1, C64::ONE / (iom - C64::new(1.1, 0.0)));
            let ex = u.mat_mul(&exd).mat_mul(&u.adjoint());
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for q in 0..4 {
                let dz = x[q] - ex.raw()[q];
                num += dz.abs() * dz.abs();
                den += ex.raw()[q].abs() * ex.raw()[q].abs();
            }
            assert!((num / den).sqrt() <= 1e-10, "m {m}");
        }
    }
}

#[test]
#[should_panic]
fn fail_ft_order_two() {
    let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
    let d = matsubara_dft(&g, Sign::Fermi);
    let _ = matsubara_ft(0, &g, &d, Sign::Fermi, 2.0, 2);
}

#[test]
#[should_panic]
fn fail_ft_negative_beta() {
    let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
    let d = matsubara_dft(&g, Sign::Fermi);
    let _ = matsubara_ft(0, &g, &d, Sign::Fermi, -2.0, 3);
}

#[test]
#[should_panic]
fn fail_ft_raw_len_mismatch() {
    let g = mf_pole(0.7, 32, 2.0, Sign::Fermi);
    let mut d = matsubara_dft(&g, Sign::Fermi);
    d.pop();
    let _ = matsubara_ft(0, &g, &d, Sign::Fermi, 2.0, 3);
}

#[test]
fn release_order_gates() {
    if !common::is_release() {
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
            common::check_order_gate(&ns, &errs(3), 4.0, 1e-10, 1e-13, 0.3).ok,
            "cubic {sign:?}"
        );
        assert!(
            common::check_order_gate(&ns, &errs(1), 2.0, 1e-5, 1e-13, 0.3).ok,
            "linear {sign:?}"
        );
        assert!(
            common::check_order_gate(&ns, &errs(0), 1.0, 0.5, 1e-13, 0.3).ok,
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
                let x = C64::new(dtau * w, 0.0) * d[m.rem_euclid(n) as usize][0];
                let om = get_omega(m, beta, sign);
                let ex = C64::ONE / C64::new(-eps, om);
                worst = worst.max((x - ex).abs() / ex.abs());
            }
            worst
        };
        let m3: Vec<f64> = ns.iter().map(|&n| mutant(n, 3)).collect();
        assert!(
            !common::check_order_gate(&ns, &m3, 4.0, 1e-10, 1e-13, 0.3).ok,
            "cubic mutant {sign:?}"
        );
        let m1: Vec<f64> = ns.iter().map(|&n| mutant(n, 1)).collect();
        assert!(
            !common::check_order_gate(&ns, &m1, 2.0, 1e-5, 1e-13, 0.3).ok,
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
                    let ex = C64::ONE / C64::new(-eps, om);
                    worst = worst.max((x[0] - ex).abs() / ex.abs());
                }
                worst
            })
            .collect();
        assert!(
            !common::check_order_gate(&ns, &e0, 1.0, 0.5, 1e-13, 0.3).ok,
            "wrong stats {sign:?}"
        );
    }
}

