//! Port of julia/test/s_weights.jl (7 testsets).

mod common;

use kadanoffbaym::math::fermi_tau;
use kadanoffbaym::weights::{
    BackwardDifferentiationWeights, BoundaryConvolutionWeights, GregoryIntegrationWeights,
    Integrator, PolynomialDifferentiationWeights, PolynomialIntegrationWeights,
    PolynomialInterpolationWeights,
};

fn isclose(got: f64, want: f64, atol: f64, rtol: f64) -> bool {
    (got - want).abs() <= atol + rtol * want.abs()
}

#[test]
fn polynomial_interpolation_weights() {
    let k = 5usize;
    let h = 0.1;
    let eps = 1.0e-4;
    let piw = PolynomialInterpolationWeights::new(k as i64);
    let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
    let mut err = 0.0;
    for i in 0..=k {
        let t = (i as f64 + 0.5) * h;
        let mut fint = 0.0;
        for l in 0..=k {
            let mut t1 = 1.0;
            let mut weight = piw.at(0, l);
            for n in 1..=k {
                t1 *= i as f64 + 0.5;
                weight += t1 * piw.at(n, l);
            }
            fint += ft[l] * weight;
        }
        err += (t.cos() - fint).abs();
    }
    assert!(err < eps, "err = {err}");
}

#[test]
fn polynomial_differentiation_weights() {
    let k = 5usize;
    let h = 0.1;
    let eps = 1.0e-4;
    let pdw = PolynomialDifferentiationWeights::new(k as i64);
    let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
    let mut err = 0.0;
    for i in 0..=k {
        let df_exact = -(h * i as f64).sin();
        let mut df_approx = 0.0;
        for l in 0..=k {
            df_approx += (1.0 / h) * pdw.at(i, l) * ft[l];
        }
        err += (df_exact - df_approx).abs();
    }
    assert!(err < eps, "err = {err}");
}

#[test]
fn polynomial_integration_weights() {
    let k = 5usize;
    let h = 0.1;
    let eps = 1.0e-4;
    let xiw = PolynomialIntegrationWeights::new(k as i64);
    let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
    let mut err = 0.0;
    for i in 1..=k {
        for j in 0..i {
            let mut i_approx = 0.0;
            for l in 0..=k {
                i_approx += h * xiw.at(i, j, l) * ft[l];
            }
            let i_exact = (h * j as f64).sin() - (h * i as f64).sin();
            err += (i_exact - i_approx).abs();
        }
    }
    assert!(err < eps, "err = {err}");
}

#[test]
fn backward_differentiation_weights() {
    let k = 5usize;
    let h = 0.1;
    let eps = 1.0e-4;
    let bdw = BackwardDifferentiationWeights::new(k as i64);
    let t1 = 0.5;
    let mut df_approx = 0.0;
    for l in 0..=k + 1 {
        df_approx += bdw.at(l) * (t1 - l as f64 * h).cos() / h;
    }
    let df_exact = -t1.sin();
    assert!(
        (df_exact - df_approx).abs() < eps,
        "err = {}",
        (df_exact - df_approx).abs()
    );
}

#[test]
fn boundary_convolution_weights() {
    let k = 5usize;
    let ntau = 100usize;
    let eps = 1.0e-4;
    let eps1 = -0.1;
    let eps2 = 0.2;
    let beta = 5.0;
    let dtau = beta / ntau as f64;
    let bcw = BoundaryConvolutionWeights::new(k as i64);
    let mut a = vec![0.0; ntau + 1];
    let mut b = vec![0.0; ntau + 1];
    for m in 0..=ntau {
        a[m] = -fermi_tau(beta, m as f64 * dtau, -eps1);
        b[m] = -fermi_tau(beta, m as f64 * dtau, -eps2);
    }
    // Julia R_exact[1..4] -> 0-based 0..3 (m = 1..4)
    let r_exact = [0.0, 0.0137659, 0.0274638, 0.0410948, 0.0546598];
    let mut err = 0.0;
    // Julia: for i = 1:k-1 -> m = i, BCW[i-1, j, l] (0-based m-1)
    for i in 1..=k - 1 {
        let mut r_approx = 0.0;
        for j in 0..=k {
            for l in 0..=k {
                r_approx += dtau * bcw.at(i, j, l) * a[j] * b[l];
            }
        }
        err += (r_exact[i] - r_approx).abs();
    }
    assert!(err < eps, "err = {err}");
}

#[test]
fn gregory_integration_weights() {
    let k = 5usize;
    let nt = 100usize;
    let tmax = 10.0;
    let h = tmax / nt as f64;
    let eps = 1.0e-4;
    let giw = GregoryIntegrationWeights::new(k as i64);
    let ft1: Vec<f64> = (0..=nt).map(|x| (h * x as f64).cos()).collect();
    let ft2: Vec<kadanoffbaym::C64> = (0..=nt)
        .map(|i| kadanoffbaym::C64::from_polar(1.0, h * i as f64))
        .collect();
    let mut err1 = 0.0;
    let mut err2 = 0.0;
    for i in (k + 1)..=nt {
        let i_exact1 = (h * i as f64).sin();
        let i_exact2 = kadanoffbaym::C64::from_polar(1.0, h * i as f64) - kadanoffbaym::C64::ONE;
        let i_exact2 = kadanoffbaym::C64::new(0.0, -1.0) * i_exact2;
        let mut i_approx1 = 0.0;
        let mut i_approx2 = kadanoffbaym::C64::ZERO;
        for j in 0..=i {
            i_approx1 += giw.at(i, j) * ft1[j] * h;
            i_approx2 = i_approx2 + ft2[j].scale(giw.at(i, j)).scale(h);
        }
        err1 += (i_exact1 - i_approx1).abs();
        err2 += (i_exact2 - i_approx2).abs();
    }
    assert!(err1 < eps, "err1 = {err1}");
    assert!(err2 < eps, "err2 = {err2}");
}

/// NESSi reference: libcntr/cntr/integration.{hpp,cpp}. The order `k` of
/// KB's `Integrator` is identical to the `k` of NESSi's
/// `integration::I<T>(k)`.
#[test]
fn integrator_nessi_order_mapping_k_1_to_5() {
    let tol = 1e-14;
    // Gregory boundary weights omega_j (NESSi integration.cpp)
    let greg: Vec<Vec<f64>> = vec![
        vec![5.0 / 12.0, 13.0 / 12.0],
        vec![3.0 / 8.0, 7.0 / 6.0, 23.0 / 24.0],
        vec![251.0 / 720.0, 299.0 / 240.0, 211.0 / 240.0, 739.0 / 720.0],
        vec![
            95.0 / 288.0,
            317.0 / 240.0,
            23.0 / 30.0,
            793.0 / 720.0,
            157.0 / 160.0,
        ],
        vec![
            19087.0 / 60480.0,
            84199.0 / 60480.0,
            18869.0 / 30240.0,
            37621.0 / 30240.0,
            55031.0 / 60480.0,
            61343.0 / 60480.0,
        ],
    ];
    // Backward differentiation weights of order k + 1 (BDF(k+1))
    let bdf: Vec<Vec<f64>> = vec![
        vec![3.0 / 2.0, -2.0, 1.0 / 2.0],
        vec![11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0],
        vec![25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0],
        vec![137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
        vec![
            147.0 / 60.0,
            -6.0,
            15.0 / 2.0,
            -20.0 / 3.0,
            15.0 / 4.0,
            -6.0 / 5.0,
            1.0 / 6.0,
        ],
    ];
    for k in 1..=5i64 {
        let integ = Integrator::new(k);
        assert_eq!(integ.k, k);
        assert_eq!(integ.giw.k, k);
        assert_eq!(integ.bdw.k, k + 1);
        // BCW dims: (k-1, k+1, k+1)
        assert_eq!(integ.bcw.w.len(), ((k - 1) * (k + 1) * (k + 1)) as usize);
        // Gregory tail weights
        for (i, want) in greg[(k - 1) as usize].iter().enumerate() {
            let got = integ.giw.omega[i];
            assert!(
                isclose(got, *want, tol, tol),
                "k={k} omega[{i}] = {got}, want {want}"
            );
        }
        // Interior Gregory weights are unity
        let n = (4 * k + 4) as usize;
        let j = (2 * k + 2) as usize;
        assert!(isclose(integ.giw.at(n, j), 1.0, tol, tol));
        // BDW (round-off grows slightly with k, about 4e-14 for k = 5)
        let btol = if k <= 3 { tol } else { 1e-12 };
        for (l, want) in bdf[(k - 1) as usize].iter().enumerate() {
            let got = integ.bdw.at(l);
            assert!(
                isclose(got, *want, btol, btol),
                "k={k} bdw[{l}] = {got}, want {want}"
            );
        }
    }
    // k = 1: rcorr is not allocated in NESSi, so W is empty.
    let integ = Integrator::new(1);
    assert_eq!(integ.bcw.w.len(), 0);
    // BCW.k stays 1, and the (k+1, k+1) block dims are implied by the
    // empty first dimension (Julia size (0, 2, 2)).
    assert_eq!(integ.bcw.k, 1);
}

// keep the shared helper referenced (profile module compiles for later phases)
#[test]
fn order_gate_self_test() {
    let g = common::check_order_gate(
        &[10, 20, 40, 80],
        &[1e-2, 1e-3, 1e-4, 1e-5],
        1.0,
        2e-5,
        1e-13,
        0.3,
    );
    assert!(g.ok, "{}", g.msg);
}
