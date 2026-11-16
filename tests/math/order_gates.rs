// Project : KadanoffBaym
// File    : tests/math/order_gates.rs
// Purpose : release-only convergence-order gates of the differentiation
//           and extrapolation/interpolation schemes

use super::util::prelude::*;
use super::util::{assert_order_gate, check_order_gate, is_release};
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::linalg::expm;
use kadanoffbaym::math::{
    deriv1_element, deriv1_matsubara, deriv2_element, extrapolate_timestep_cf, interpolation,
};
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
/// Release-only: convergence order gates (blueprint
/// `kb_release()` section).
#[test]
fn release_differentiation_order_gate() {
    if !is_release() {
        return;
    }
    // H, X, Z matrices and U(t) = exp(-i H t) products; derivatives are
    // compared against the analytic i d/dt relations
    let mut hm = DenseMat::zeros(2);
    hm.set(0, 0, Complex::new(-0.4, 0.0));
    hm.set(1, 1, Complex::new(0.6, 0.0));
    hm.set(0, 1, Complex::new(0.0, 0.3));
    hm.set(1, 0, Complex::new(0.0, -0.3));
    let u = |t: f64| -> DenseMat {
        // U(t) = exp(-i H t)
        let mut arg = DenseMat::zeros_shape(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                arg.set(i, j, Complex::new(0.0, -t) * hm.get(i, j));
            }
        }
        expm(&arg)
    };
    let z = block_z();

    // error of the left derivative at grid index i (0-based `i_rust`,
    // blueprint index = 0-based index + 1)
    let deriv_error = |k: i64, ord: i64, h: f64, i_1based: i64| -> f64 {
        let i_rust = i_1based - 1;
        let ntime = i_1based.max(k + 1) + 1; // points
        let nt = ntime - 1;
        let ntau = 2 * k + 1; // intervals (2k+2 points)
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // Fr(t,s) = -i U(t+0.3) U(s+0.3)†; Fl(t,s) = i U(t+0.3) X U(s+0.3)†;
        // Acc entries carry the SWAPPED-argument conjugate (blueprint
        // `Acc.ret[i,j] = -Fr(t_j, t_i)†`, same for less)
        for i in 0..=nt {
            for j in 0..=i {
                let v = mul_c(
                    -C64_I,
                    &u(i as f64 * h + 0.3).mat_mul(&u(j as f64 * h + 0.3).adjoint()),
                );
                a.ret_raw_block_mut(i, j).copy_from_slice(&v);
                let vsw = mul_c(
                    -C64_I,
                    &u(j as f64 * h + 0.3).mat_mul(&u(i as f64 * h + 0.3).adjoint()),
                );
                acc.ret_raw_block_mut(i, j).copy_from_slice(&neg_adj(&vsw));
            }
            for j in i..=nt {
                let x = block_x();
                let v = mul_c(
                    C64_I,
                    &u(i as f64 * h + 0.3)
                        .mat_mul(&x)
                        .mat_mul(&u(j as f64 * h + 0.3).adjoint()),
                );
                a.less_raw_block_mut(i, j).copy_from_slice(&v);
                let vsw = mul_c(
                    C64_I,
                    &u(j as f64 * h + 0.3)
                        .mat_mul(&x)
                        .mat_mul(&u(i as f64 * h + 0.3).adjoint()),
                );
                acc.less_raw_block_mut(i, j).copy_from_slice(&neg_adj(&vsw));
            }
        }
        for i in 0..=nt {
            for m in 0..=ntau {
                let v = to_block(&u(i as f64 * h).mat_mul(&z));
                a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
            }
        }
        for m in 0..=ntau {
            a.mat_raw_block_mut(m).copy_from_slice(&to_block(&z));
            acc.mat_raw_block_mut(m).copy_from_slice(&to_block(&z));
        }
        let integ = Integrator::new(ord);
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut db = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut err = 0.0f64;
        for j in 0..=(i_1based - 1) {
            deriv1_element(i_rust, j, &mut da, &a, Some(&acc), &integ, h);
            deriv2_element(j, i_rust, &mut db, &a, Some(&acc), &integ, h);
            let t = i_rust as f64 * h;
            let s = j as f64 * h;
            // i d/dt Fr = H Fr ; i d/ds Fl = ... Fl H
            let want_l = hm.mat_mul(&u(t + 0.3).mat_mul(&u(s + 0.3).adjoint()));
            let got_l = DenseMat::from_vec(2, 2, da.ret_raw_block(i_rust, j).to_vec());
            let want_l = DenseMat::from_vec(2, 2, mul_c(-C64_I, &want_l));
            for q in 0..4 {
                err = err.max((dm_at(&got_l, q) - dm_at(&want_l, q)).abs());
            }
            let x = block_x();
            let fl = u(s + 0.3).mat_mul(&x).mat_mul(&u(t + 0.3).adjoint());
            let want_r = fl.mat_mul(&hm);
            let want_r = DenseMat::from_vec(2, 2, mul_c(C64_I, &want_r));
            let got_r = DenseMat::from_vec(2, 2, db.less_raw_block(j, i_rust).to_vec());
            for q in 0..4 {
                err = err.max((dm_at(&got_r, q) - dm_at(&want_r, q)).abs());
            }
        }
        err
    };

    // matsubara error: -d/dtau of Z e^{-lambda tau} at tau_0
    let mat_error = |k: i64, ord: i64, hv: f64| -> f64 {
        let ntau = 2 * k + 2; // intervals (2k+3 points)
        let beta = hv * ntau as f64;
        let c = Contour::new(1, ntau, 2, 2, 1.0, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let lam = 0.8f64;
        for m in 0..=ntau {
            let v = to_block(&mat_scale(&z, (-lam * m as f64 * hv).exp()));
            a.mat_raw_block_mut(m).copy_from_slice(&v);
        }
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        deriv1_matsubara(&mut da, &a, &Integrator::new(ord), beta);
        let want = mat_scale(&z, lam);
        let got = da.mat_raw_block(0);
        let wantb = to_block(&want);
        got.iter()
            .zip(&wantb)
            .fold(0.0, |e, (g, w)| e.max((*g - *w).abs()))
    };

    let grids: Vec<(i64, i64, std::ops::Range<i64>)> = vec![
        (1, 0, 11..15),
        (1, 2, 11..15),
        (1, 4, 11..15),
        (2, 1, 11..15),
        (2, 2, 6..10),
        (2, 4, 6..10),
        (3, 1, 6..10),
        (3, 2, 4..8),
        (3, 4, 4..8),
        (4, 1, 4..8),
        (4, 2, 3..7),
        (4, 4, 3..7),
        (5, 1, 3..7),
        (5, 2, 2..6),
        (5, 4, 2..6),
    ];
    // (k, tag): tag 0 = start-up, 1 = stable, 2 = matsubara
    for k in 1..=5i64 {
        if k >= 2 {
            // start-up (i = 1): order k (UT_GRIDS `:s`)
            let srange = grids
                .iter()
                .find(|g| g.0 == k && g.1 == 1)
                .unwrap()
                .2
                .clone();
            let ns: Vec<i64> = srange.map(|m| 1 << m).collect();
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k, 1.0 / n as f64, 1))
                .collect();
            assert_order_gate(&ns, &errs, k as f64, 1e-8);
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k - 1, 1.0 / n as f64, 1))
                .collect();
            assert!(!check_order_gate(&ns, &errs, k as f64, 1e-8, 1e-13, 0.3).ok);
        }
        // stable (i = k+2): order k+1
        let mrange = grids
            .iter()
            .find(|g| g.0 == k && g.1 == 2)
            .unwrap()
            .2
            .clone();
        let ns: Vec<i64> = mrange.map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns
            .iter()
            .map(|&n| deriv_error(k, k, 1.0 / n as f64, k + 2))
            .collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        if k >= 2 {
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k - 1, 1.0 / n as f64, k + 2))
                .collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
        // matsubara: order k+1
        let mrange = grids
            .iter()
            .find(|g| g.0 == k && g.1 == 4)
            .unwrap()
            .2
            .clone();
        let ns: Vec<i64> = mrange.map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns
            .iter()
            .map(|&n| mat_error(k, k, 1.0 / n as f64))
            .collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        if k >= 2 {
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| mat_error(k, k - 1, 1.0 / n as f64))
                .collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
    }
}

const C64_I: Complex = Complex::new(0.0, 1.0);

fn mul_c(c: Complex, m: &DenseMat) -> Vec<Complex> {
    to_block(m).iter().map(|z| c * *z).collect()
}

fn to_block(m: &DenseMat) -> Vec<Complex> {
    (0..m.nrows() * m.ncols()).map(|q| dm_at(m, q)).collect()
}

fn block_x() -> DenseMat {
    let mut x = DenseMat::zeros(2);
    x.set(0, 0, Complex::new(0.3, 0.0));
    x.set(0, 1, Complex::new(0.1, 0.0));
    x.set(1, 0, Complex::new(0.1, 0.0));
    x.set(1, 1, Complex::new(0.2, 0.0));
    x
}

fn block_z() -> DenseMat {
    let mut z = DenseMat::zeros(2);
    z.set(0, 0, Complex::new(0.5, 0.0));
    z.set(0, 1, Complex::new(0.0, 0.2));
    z.set(1, 0, Complex::new(0.0, -0.2));
    z.set(1, 1, Complex::new(0.4, 0.0));
    z
}

fn mat_scale(m: &DenseMat, s: f64) -> DenseMat {
    let mut out = DenseMat::zeros_shape(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.set(i, j, m.get(i, j).scale(s));
        }
    }
    out
}

fn dm_at(m: &DenseMat, q: usize) -> Complex {
    m.get(q / m.ncols(), q % m.ncols())
}

/// -(block)† for square blocks.
fn neg_adj(blk: &[Complex]) -> Vec<Complex> {
    let n = (blk.len() as f64).sqrt() as usize;
    let mut out = vec![Complex::ZERO; blk.len()];
    for a in 0..n {
        for b in 0..n {
            out[b * n + a] = -blk[a * n + b].conj();
        }
    }
    out
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
