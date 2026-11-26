// Project : KadanoffBaym
// File    : tests/steady/matsubara.rs
// Purpose : the Fourier method of the Matsubara Dyson and VIE2 solvers
//
// and cntr_vie2_impl.hpp, vie2_mat_fourier_dispatch (pcf = 20).
//
// The frequency sums are truncated at (2pcf + 1) N/2 modes and the
// remaining tail decays as 1/(i w)^2, so that the Fourier method
// converges only as N^{-3/2} towards the fixpoint solution.

use kadanoffbaym::base::MatsubaraMethod;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::base::matsubara_seed;
use kadanoffbaym::convolution::conv_mat;
use kadanoffbaym::dyson::dyson_mat;
use kadanoffbaym::vie2::vie2_mat;

use super::util::c;
use super::util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const BETA: f64 = 2.0;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// `dyson_mat` in the Fourier method with the vanishing self-energy
/// `zero` at `mu = 0` (the order 5).
fn fourier_dyson_zero(g: &mut ContourGreen, zero: &ContourGreen, h: &TimeFunction) {
    dyson_mat(
        g,
        zero,
        0.0,
        h,
        BETA,
        5,
        MatsubaraMethod::Fourier,
        true,
        None,
        1e-12,
    );
}

fn reldiff(x: &ContourGreen, y: &ContourGreen) -> f64 {
    let mut dn = 0.0;
    let mut dd = 0.0;
    for q in 0..x.mat.raw().len() {
        let d = x.mat.raw()[q] - y.mat.raw()[q];
        dn += d.abs() * d.abs();
        let v = y.mat.raw()[q];
        dd += v.abs() * v.abs();
    }
    (dn / dd).sqrt()
}

/// Hamiltonian, self-energy, and the VIE2 kernels on a 2 x 2 grid
/// ; `ntau` is the number of intervals.
fn setup(
    sign: Sign,
    ntau: i64,
    d: i64,
) -> (
    Contour,
    TimeFunction,
    f64,
    ContourGreen,
    ContourGreen,
    ContourGreen,
    ContourGreen,
    Integrator,
) {
    let contour = Contour::new(1, ntau, d, d, 0.1, BETA);
    let h = if d == 2 {
        time_function2(0.3, c(0.2, -0.1), c(0.2, 0.1), -0.4)
    } else {
        time_function1(0.3)
    };
    let hs = if d == 2 {
        time_function2(0.5, c(0.0, 0.1), c(0.0, -0.1), -0.2)
    } else {
        time_function1(0.5)
    };
    let mu = if sign == Sign::Fermi { 0.1 } else { -1.5 };
    let mu_s = if sign == Sign::Fermi { 0.0 } else { -1.0 };

    let mut sigma = ContourGreen::new(&contour, sign, Complex::ZERO);
    let zero = ContourGreen::new(&contour, sign, Complex::ZERO);
    dyson_mat(
        &mut sigma,
        &zero,
        mu_s,
        &hs,
        BETA,
        5,
        MatsubaraMethod::Fixpoint,
        true,
        None,
        1e-12,
    );
    for q in 0..sigma.mat.raw().len() {
        sigma.mat.raw_mut()[q] *= c(0.3, 0.0);
    }

    let mut g0 = ContourGreen::new(&contour, sign, Complex::ZERO);
    dyson_mat(
        &mut g0,
        &zero,
        mu,
        &h,
        BETA,
        5,
        MatsubaraMethod::Fixpoint,
        true,
        None,
        1e-12,
    );

    let i5 = Integrator::new(5);
    let mut f = ContourGreen::new(&contour, sign, Complex::ZERO);
    let mut fcc = ContourGreen::new(&contour, sign, Complex::ZERO);
    conv_mat(&mut f.mat, &g0.mat, &sigma.mat, &i5, BETA, sign);
    conv_mat(&mut fcc.mat, &sigma.mat, &g0.mat, &i5, BETA, sign);
    for q in 0..f.mat.raw().len() {
        f.mat.raw_mut()[q] = -f.mat.raw()[q];
        fcc.mat.raw_mut()[q] = -fcc.mat.raw()[q];
    }
    (contour, h, mu, sigma, g0, f, fcc, i5)
}

fn sign_factor(sign: Sign) -> f64 {
    sign.as_i64() as f64
}

fn time_function1(v: f64) -> TimeFunction {
    let mut h = TimeFunction::new(1, 1, 1, Complex::ZERO);
    h.set(MATSUBARA, 0, 0, c(v, 0.0));
    h
}

fn time_function2(m00: f64, m01: Complex, m10: Complex, m11: f64) -> TimeFunction {
    let mut h = TimeFunction::new(1, 2, 2, Complex::ZERO);
    h.set(MATSUBARA, 0, 0, c(m00, 0.0));
    h.set(MATSUBARA, 0, 1, m01);
    h.set(MATSUBARA, 1, 0, m10);
    h.set(MATSUBARA, 1, 1, c(m11, 0.0));
    h
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn dyson_fourier_free_particle() {
    // analytic G(tau) = -e^{-eps tau} / (1 - sign_factor e^{-beta eps})
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let eps = if sign == Sign::Fermi { 0.4 } else { 0.6 };
        let sign_factor = sign.as_i64() as f64;
        let mut errs = Vec::new();
        for &ntau_p in &[33i64, 65, 129] {
            let ntau = ntau_p - 1; // intervals
            let contour = Contour::new(1, ntau, 1, 1, 0.1, BETA);
            let h = time_function1(eps);
            let mut g = ContourGreen::new(&contour, sign, Complex::ZERO);
            let zero = ContourGreen::new(&contour, sign, Complex::ZERO);
            dyson_mat(
                &mut g,
                &zero,
                0.0,
                &h,
                BETA,
                5,
                MatsubaraMethod::Fourier,
                true,
                None,
                1e-12,
            );
            for z in g.mat.raw() {
                assert!(z.re.is_finite() && z.im.is_finite());
            }
            let denom = 1.0 - sign_factor * (-BETA * eps).exp();
            let mut err = 0.0f64;
            for m in 0..=ntau {
                let tau = m as f64 * BETA / ntau as f64;
                let want = -(-eps * tau).exp() / denom;
                err = err.max((g.get_mat(m, 0, 0) - c(want, 0.0)).abs());
            }
            errs.push(err);
        }
        assert!(errs[0] < 2e-3, "{sign:?}: {errs:?}");
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "{sign:?}: {errs:?}");
    }
}

#[test]
fn dyson_fourier_vs_fixpoint() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        for &d in &[1i64, 2] {
            let mut diffs = Vec::new();
            for &ntau_p in &[33i64, 65, 129] {
                let ntau = ntau_p - 1;
                let (contour, h, mu, sigma, _g0, _f, _fcc, _i5) = setup(sign, ntau, d);
                let mut gf = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat(
                    &mut gf,
                    &sigma,
                    mu,
                    &h,
                    BETA,
                    5,
                    MatsubaraMethod::Fixpoint,
                    true,
                    None,
                    1e-12,
                );
                let mut go = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat(
                    &mut go,
                    &sigma,
                    mu,
                    &h,
                    BETA,
                    5,
                    MatsubaraMethod::Fourier,
                    false,
                    None,
                    1e-12,
                );
                diffs.push(reldiff(&go, &gf));

                // hermiticity of G(tau) without the symmetrization; the
                // bosonic frequency window is not symmetric
                let tol = if sign == Sign::Fermi { 1e-10 } else { 1e-8 };
                for m in 0..=ntau {
                    let blk = go.mat.block(m as usize);
                    // hermitian norm: max |M - M^dagger|
                    let mut hmax = 0.0f64;
                    for i in 0..d as usize {
                        for j in 0..d as usize {
                            let v = blk[i * d as usize + j];
                            let w = blk[j * d as usize + i].conj();
                            hmax = hmax.max((v - w).abs());
                        }
                    }
                    assert!(hmax < tol, "hermitian at m = {m}: {hmax}");
                }
                // KMS: the jump G(0) - sign_factor G(beta) is -1 (the free tail)
                let mut jump = 0.0f64;
                for i in 0..d as usize {
                    for j in 0..d as usize {
                        let v = go.mat.block(0)[i * d as usize + j]
                            - go.mat.block(ntau as usize)[i * d as usize + j]
                                * c(sign_factor(sign), 0.0)
                            + if i == j { Complex::ONE } else { Complex::ZERO };
                        jump = jump.max(v.abs());
                    }
                }
                assert!(jump < 1e-10, "KMS jump: {jump}");

                // the public method uses pcf = 10 and the cubic correction
                let mut gp = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat(
                    &mut gp,
                    &sigma,
                    mu,
                    &h,
                    BETA,
                    5,
                    MatsubaraMethod::Fourier,
                    false,
                    None,
                    1e-12,
                );
                let dmax = (0..gp.mat.raw().len())
                    .map(|q| (gp.mat.raw()[q] - go.mat.raw()[q]).abs())
                    .fold(0.0, f64::max);
                assert!(dmax == 0.0, "public pcf=10 path: {dmax}");
            }
            assert!(diffs[0] < 5e-4, "{sign:?} d{d}: {diffs:?}");
            assert!(
                diffs[1] < diffs[0] && diffs[2] < diffs[1],
                "{sign:?} d{d}: {diffs:?}"
            );
        }
    }
}

#[test]
fn vie2_fourier_vs_fixpoint() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let mut diffs = Vec::new();
        for &ntau_p in &[33i64, 65, 129] {
            let ntau = ntau_p - 1;
            let (contour, _h, _mu, _sigma, g0, f, fcc, i5) = setup(sign, ntau, 2);
            let mut xf = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(
                &mut xf,
                &f,
                &fcc,
                &g0,
                BETA,
                &i5,
                MatsubaraMethod::Fixpoint,
                6,
                40,
                1e-12,
            );
            let mut xo = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(
                &mut xo,
                &f,
                &fcc,
                &g0,
                BETA,
                &i5,
                MatsubaraMethod::Fourier,
                6,
                40,
                1e-12,
            );
            diffs.push(reldiff(&xo, &xf));

            // F = 0 gives G = Q up to the truncated tail
            let z = ContourGreen::new(&contour, sign, Complex::ZERO);
            let mut x0 = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(
                &mut x0,
                &z,
                &z,
                &g0,
                BETA,
                &i5,
                MatsubaraMethod::Fourier,
                6,
                40,
                1e-12,
            );
            assert!(reldiff(&x0, &g0) < 1e-3, "zero kernel {sign:?}");

            // Fcc is not used by the frequency-domain solver
            let mut xc = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(
                &mut xc,
                &f,
                &z,
                &g0,
                BETA,
                &i5,
                MatsubaraMethod::Fourier,
                6,
                40,
                1e-12,
            );
            let dmax = (0..xc.mat.raw().len())
                .map(|q| (xc.mat.raw()[q] - xo.mat.raw()[q]).abs())
                .fold(0.0, f64::max);
            assert!(dmax == 0.0, "Fcc unused: {dmax}");

            // pcf = 20, not the Dyson window pcf = 10
            let mut x20 = ContourGreen::new(&contour, sign, Complex::ZERO);
            let mut x10 = ContourGreen::new(&contour, sign, Complex::ZERO);
            matsubara_seed(&mut x20.mat, &f.mat, &g0.mat, BETA, sign, 20, 3);
            matsubara_seed(&mut x10.mat, &f.mat, &g0.mat, BETA, sign, 10, 3);
            let dmax = (0..x20.mat.raw().len())
                .map(|q| (x20.mat.raw()[q] - xo.mat.raw()[q]).abs())
                .fold(0.0, f64::max);
            assert!(dmax == 0.0, "pcf=20 path: {dmax}");
            assert!(
                reldiff(&x10, &x20) > 1e-7,
                "pcf=10 must differ: {}",
                reldiff(&x10, &x20)
            );
        }
        assert!(diffs[0] < 1e-4, "{sign:?}: {diffs:?}");
        assert!(
            diffs[1] < diffs[0] && diffs[2] < diffs[1],
            "{sign:?}: {diffs:?}"
        );
    }
}

#[test]
#[should_panic]
fn fourier_rejects_odd_intervals_dyson() {
    // 34 points = 33 intervals (odd)
    let contour = Contour::new(1, 33, 1, 1, 0.1, BETA);
    let h = time_function1(0.3);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let zero = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    fourier_dyson_zero(&mut g, &zero, &h);
}

#[test]
#[should_panic]
fn fourier_rejects_odd_intervals_vie2() {
    let contour = Contour::new(1, 33, 1, 1, 0.1, BETA);
    let g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let i5 = Integrator::new(5);
    vie2_mat(
        &mut g.clone(),
        &g,
        &g,
        &g,
        BETA,
        &i5,
        MatsubaraMethod::Fourier,
        6,
        40,
        1e-12,
    );
}

#[test]
#[should_panic(expected = "singular")]
fn fourier_rejects_singular_bosonic_zero_mode() {
    // H - mu + Sigma(0) = 0 for the bosonic zero mode
    let contour = Contour::new(1, 32, 1, 1, 0.1, BETA);
    let h = time_function1(0.0);
    let mut g = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    let zero = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    fourier_dyson_zero(&mut g, &zero, &h);
}
