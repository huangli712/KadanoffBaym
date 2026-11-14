// Project : KadanoffBaym
// File    : tests/matsubara_solvers_fourier.rs
// Purpose : blueprint t_matsubara_solvers_fourier.jl: the Fourier
//           method of dyson_mat and vie2_mat
//
// NESSi: cntr_dyson_impl.hpp, dyson_mat_fourier_dispatch (pcf = 10),
// and cntr_vie2_impl.hpp, vie2_mat_fourier_dispatch (pcf = 20).
//
// The frequency sums are truncated at (2pcf + 1) N/2 modes and the
// remaining tail decays as 1/(i w)^2, so that the Fourier method
// converges only as N^{-3/2} towards the fixpoint solution.

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::conv_mat;
use kadanoffbaym::fourier::matsubara_seed;
use kadanoffbaym::dyson::{dyson_mat, dyson_mat_fourier, MatsubaraMethod};
use kadanoffbaym::vie2::mat::vie2_mat;
use kadanoffbaym::types::Contour;
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::TimeFunction;
use kadanoffbaym::types::MATSUBARA;
use kadanoffbaym::weights::Integrator;

const BETA: f64 = 2.0;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn tf2(m00: f64, m01: Complex, m10: Complex, m11: f64) -> TimeFunction {
    let mut h = TimeFunction::new(1, 2, 2, Complex::ZERO);
    h.set(MATSUBARA, 0, 0, c(m00, 0.0));
    h.set(MATSUBARA, 0, 1, m01);
    h.set(MATSUBARA, 1, 0, m10);
    h.set(MATSUBARA, 1, 1, c(m11, 0.0));
    h
}

fn tf1(v: f64) -> TimeFunction {
    let mut h = TimeFunction::new(1, 1, 1, Complex::ZERO);
    h.set(MATSUBARA, 0, 0, c(v, 0.0));
    h
}

/// Hamiltonian, self-energy, and the VIE2 kernels on a 2 x 2 grid
/// (blueprint `msf_setup`); `ntau` is the number of intervals.
#[allow(clippy::type_complexity)]
fn msf_setup(sign: Sign, ntau: i64, d: i64) -> (Contour, TimeFunction, f64, ContourGreen, ContourGreen, ContourGreen, ContourGreen, Integrator) {
    let contour = Contour::new(1, ntau, d, d, 0.1, BETA);
    let h = if d == 2 {
        tf2(0.3, c(0.2, -0.1), c(0.2, 0.1), -0.4)
    } else {
        tf1(0.3)
    };
    let hs = if d == 2 {
        tf2(0.5, c(0.0, 0.1), c(0.0, -0.1), -0.2)
    } else {
        tf1(0.5)
    };
    let mu = if sign == Sign::Fermi { 0.1 } else { -1.5 };
    let mu_s = if sign == Sign::Fermi { 0.0 } else { -1.0 };

    let mut sigma = ContourGreen::new(&contour, sign, Complex::ZERO);
    let zero = ContourGreen::new(&contour, sign, Complex::ZERO);
    dyson_mat(&mut sigma, &zero, mu_s, &hs, BETA, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
    for q in 0..sigma.mat.raw().len() {
        sigma.mat.raw_mut()[q] *= c(0.3, 0.0);
    }

    let mut g0 = ContourGreen::new(&contour, sign, Complex::ZERO);
    dyson_mat(&mut g0, &zero, mu, &h, BETA, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);

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

#[test]
fn dyson_fourier_free_particle() {
    // analytic G(tau) = -e^{-eps tau} / (1 - xi e^{-beta eps})
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let eps = if sign == Sign::Fermi { 0.4 } else { 0.6 };
        let xi = sign.as_i64() as f64;
        let mut errs = Vec::new();
        for &ntau_p in &[33i64, 65, 129] {
            let ntau = ntau_p - 1; // intervals
            let contour = Contour::new(1, ntau, 1, 1, 0.1, BETA);
            let h = tf1(eps);
            let mut g = ContourGreen::new(&contour, sign, Complex::ZERO);
            let zero = ContourGreen::new(&contour, sign, Complex::ZERO);
            dyson_mat(&mut g, &zero, 0.0, &h, BETA, 5, MatsubaraMethod::Fourier, true, None, 1e-12);
            for z in g.mat.raw() {
                assert!(z.re.is_finite() && z.im.is_finite());
            }
            let denom = 1.0 - xi * (-BETA * eps).exp();
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
                let (contour, h, mu, sigma, _g0, _f, _fcc, _i5) = msf_setup(sign, ntau, d);
                let mut gf = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat(&mut gf, &sigma, mu, &h, BETA, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
                let mut go = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat(&mut go, &sigma, mu, &h, BETA, 5, MatsubaraMethod::Fourier, false, None, 1e-12);
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
                // KMS: the jump G(0) - xi G(beta) is -1 (the free tail)
                let mut jump = 0.0f64;
                for i in 0..d as usize {
                    for j in 0..d as usize {
                        let v = go.mat.block(0)[i * d as usize + j]
                            - go.mat.block(ntau as usize)[i * d as usize + j] * c(xi(sign), 0.0)
                            + if i == j { Complex::ONE } else { Complex::ZERO };
                        jump = jump.max(v.abs());
                    }
                }
                assert!(jump < 1e-10, "KMS jump: {jump}");

                // the public method uses pcf = 10 and the cubic correction
                let mut gp = ContourGreen::new(&contour, sign, Complex::ZERO);
                dyson_mat_fourier(&mut gp, &sigma, mu, &h, BETA, 10, 3);
                let dmax = (0..gp.mat.raw().len())
                    .map(|q| (gp.mat.raw()[q] - go.mat.raw()[q]).abs())
                    .fold(0.0, f64::max);
                assert!(dmax == 0.0, "public pcf=10 path: {dmax}");
            }
            assert!(diffs[0] < 5e-4, "{sign:?} d{d}: {diffs:?}");
            assert!(diffs[1] < diffs[0] && diffs[2] < diffs[1], "{sign:?} d{d}: {diffs:?}");
        }
    }
}

fn xi(sign: Sign) -> f64 {
    sign.as_i64() as f64
}

#[test]
fn vie2_fourier_vs_fixpoint() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let mut diffs = Vec::new();
        for &ntau_p in &[33i64, 65, 129] {
            let ntau = ntau_p - 1;
            let (contour, _h, _mu, _sigma, g0, f, fcc, i5) = msf_setup(sign, ntau, 2);
            let mut xf = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(&mut xf, &f, &fcc, &g0, BETA, &i5, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
            let mut xo = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(&mut xo, &f, &fcc, &g0, BETA, &i5, MatsubaraMethod::Fourier, 6, 40, 1e-12);
            diffs.push(reldiff(&xo, &xf));

            // F = 0 gives G = Q up to the truncated tail
            let z = ContourGreen::new(&contour, sign, Complex::ZERO);
            let mut x0 = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(&mut x0, &z, &z, &g0, BETA, &i5, MatsubaraMethod::Fourier, 6, 40, 1e-12);
            assert!(reldiff(&x0, &g0) < 1e-3, "zero kernel {sign:?}");

            // Fcc is not used by the frequency-domain solver
            let mut xc = ContourGreen::new(&contour, sign, Complex::ZERO);
            vie2_mat(&mut xc, &f, &z, &g0, BETA, &i5, MatsubaraMethod::Fourier, 6, 40, 1e-12);
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
        assert!(diffs[1] < diffs[0] && diffs[2] < diffs[1], "{sign:?}: {diffs:?}");
    }
}

#[test]
#[should_panic]
fn fourier_rejects_odd_intervals_dyson() {
    // 34 points = 33 intervals (odd)
    let contour = Contour::new(1, 33, 1, 1, 0.1, BETA);
    let h = tf1(0.3);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let zero = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &zero, 0.0, &h, BETA, 5, MatsubaraMethod::Fourier, true, None, 1e-12);
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
    let h = tf1(0.0);
    let mut g = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    let zero = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    dyson_mat(&mut g, &zero, 0.0, &h, BETA, 5, MatsubaraMethod::Fourier, true, None, 1e-12);
}
