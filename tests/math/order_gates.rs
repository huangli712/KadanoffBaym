// Project : KadanoffBaym
// File    : tests/math/order_gates.rs
// Purpose : release-only convergence-order gates of the
//           extrapolation and interpolation schemes

use super::util::prelude::*;
use super::util::{assert_order_gate, check_order_gate, is_release};
use kadanoffbaym::math::{extrapolate_timestep_cf, interpolation};
use kadanoffbaym::weights::Integrator;

/// Release-only order gate.
#[test]
fn release_extrapolation_interpolation_order_gate() {
    if !is_release() {
        return;
    }
    let grids = &[(1i64, 12..16), (2, 6..10), (3, 5..9), (4, 4..8), (5, 3..7)];
    let extrap_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
        let h = 1.0 / n_grid as f64;
        let tplus = 1.3;
        let mut f = TimeFunction::new((k + 1) as usize, 2, 2, Complex::ZERO);
        for i in 0..=(k + 1) {
            let t = tplus - (k + 2 - (i + 1)) as f64 * h;
            f.block_mut(i).copy_from_slice(&ut_f(t));
        }
        extrapolate_timestep_cf(k, &mut f, ord);
        let mut e = 0.0;
        for (x, y) in f.block(k + 1).iter().zip(ut_f(tplus)) {
            e += (*x - y).abs();
        }
        e
    };
    let interp_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
        let h = 1.0 / n_grid as f64;
        let tplus = 1.3;
        let mut f = TimeFunction::new(k as usize, 2, 2, Complex::ZERO);
        for i in 0..=k {
            let t = tplus - (k + 1 - (i + 1)) as f64 * h;
            f.block_mut(i).copy_from_slice(&ut_f(t));
        }
        let x = interpolation(k, k as f64 - 0.5, &f, &Integrator::new(ord));
        let want = ut_f(tplus - 0.5 * h);
        let mut e = 0.0;
        for (q, wq) in want.iter().enumerate() {
            e += (x.get(q / 2, q % 2) - *wq).abs();
        }
        e
    };
    for (k, range) in grids {
        let ns: Vec<i64> = range.clone().map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, *k)).collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, *k)).collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, k - 1)).collect();
        assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        if *k >= 2 {
            let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, k - 1)).collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
    }
}

/// Blueprint `ut_f`.
fn ut_f(t: f64) -> [Complex; 4] {
    let e = (0.3 * t).exp();
    [
        Complex::new((-t).cos() * e, 0.0),
        Complex::new(0.0, -0.2 * (0.7 * t).sin() * e),
        Complex::new(0.0, 0.2 * (0.7 * t).sin() * e),
        Complex::new((0.8 * t).cos() * e, 0.0),
    ]
}
