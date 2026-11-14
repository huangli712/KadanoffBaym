// Project : KadanoffBaym
// File    : tests/dyson_mat.rs
// Purpose : blueprint t_dyson_mat.jl: the Matsubara Dyson solver and
//           the corrected-DFT core
//
// NESSi Catch tests ported here:
//   libcntr/test/matsubara.cpp:34-104, sections Fixpoint and Conjugate
//   gradient (the section Dyson uses CNTR_MAT_FOURIER, which belongs to
//   Phase 7)

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::expm::expm;
use kadanoffbaym::fourier::dftcor::dftcorr_cubic;
use kadanoffbaym::fourier::matsubara::{matsubara_dft, matsubara_ft_direct, matsubara_seed};
use kadanoffbaym::linalg::lu::mat_inv;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::solvers::dyson_mat::{
    dyson_mat, dyson_mat_cg, dyson_mat_fixpoint, dyson_mat_kernel, free_mat, mat_inner,
    MatsubaraMethod,
};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::{ContourGreen, MatComp, TimeFunction};
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::weights::Integrator;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn tf_of(m: &DenseMat) -> TimeFunction {
    let d = m.nrows();
    let mut h = TimeFunction::new(1, d, d, Complex::ZERO);
    for i in 0..d {
        for j in 0..d {
            h.set(MATSUBARA, i, j, m.get(i, j));
        }
    }
    h
}

/// Matsubara Green's function of the free hermitian Hamiltonian `h0`
/// (blueprint `dm_free_mat`); `ntau` is the number of imaginary time
/// intervals.
fn dm_free_mat(h0: &DenseMat, mu: f64, beta: f64, ntau: usize, sign: Sign) -> MatComp {
    let d = h0.nrows();
    let mut g = MatComp::new(ntau, d, d, Complex::ZERO);
    let h = tf_of(h0);
    free_mat(&mut g, mu, &h, beta, sign);
    g
}

/// Embedding model: the (a,a) block of the free Green's function of
/// `Hfull = [Ha V; V' Hb]` solves the Dyson equation for Ha with
/// `Sigma = V gb V'` (blueprint `dm_embedding`).
#[allow(clippy::type_complexity)]
fn dm_embedding(
    ha: &DenseMat,
    hb: &DenseMat,
    v: &DenseMat,
    mu: f64,
    beta: f64,
    ntau: usize,
    sign: Sign,
) -> (Contour, ContourGreen, ContourGreen, TimeFunction) {
    let (da, db) = (ha.nrows(), hb.nrows());
    let dfull = da + db;
    let mut hfull = DenseMat::zeros(dfull);
    for i in 0..dfull {
        for j in 0..dfull {
            let x = if i < da && j < da {
                ha.get(i, j)
            } else if i < da && j >= da {
                v.get(i, j - da)
            } else if i >= da && j < da {
                v.get(j, i - da).conj()
            } else {
                hb.get(i - da, j - da)
            };
            hfull.set(i, j, x);
        }
    }
    let gfull = dm_free_mat(&hfull, mu, beta, ntau, sign);
    let gb = dm_free_mat(hb, mu, beta, ntau, sign);

    let contour = Contour::new(1, ntau as i64, da as i64, da as i64, 1.0, beta);
    let mut sigma = ContourGreen::new(&contour, sign, Complex::ZERO);
    let mut g_exact = ContourGreen::new(&contour, sign, Complex::ZERO);
    let vd = DenseMat::from_vec(da, db, v.raw().to_vec());
    for m in 0..=(ntau as i64) {
        // Sigma = V gb V^dagger
        let gbm = DenseMat::from_vec(db, db, gb.block(m as usize).to_vec());
        let s = vd.mat_mul(&gbm).mat_mul(&vd.adjoint());
        for i in 0..da {
            for j in 0..da {
                sigma.set_mat(m, i, j, s.get(i, j));
                g_exact.set_mat(
                    m,
                    i,
                    j,
                    gfull.block(m as usize)[i * (da + db) + j],
                );
            }
        }
    }
    let h = tf_of(ha);
    (contour, sigma, g_exact, h)
}

fn max_abs(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
}

#[test]
fn corrected_dft_core() {
    // boundary terms at theta = 0
    let (corfac, endcor) = dftcorr_cubic(0.0);
    assert!((corfac - 1.0).abs() < 1e-15);
    assert!((endcor[0] - c(-2.0 / 3.0, 0.0)).abs() < 1e-15);
    assert!((endcor[1] - c(7.0 / 24.0, 0.0)).abs() < 1e-15);
    assert!((endcor[2] - c(-1.0 / 6.0, 0.0)).abs() < 1e-15);
    assert!((endcor[3] - c(1.0 / 24.0, 0.0)).abs() < 1e-15);

    // continuity of the series and trigonometric branches
    for s in [1.0f64, -1.0] {
        let (c1, e1) = dftcorr_cubic(s * (0.05 - 1e-12));
        let (c2, e2) = dftcorr_cubic(s * (0.05 + 1e-12));
        assert!((c1 - c2).abs() < 1e-9);
        for k in 0..4 {
            assert!((e1[k] - e2[k]).abs() < 1e-9);
        }
    }

    // Fourier transform of A(tau) = exp(-tau K) with a non-hermitian,
    // non-normal 2 x 2 K:
    //     int_0^beta dtau e^{i w tau} A(tau) = (i w - K)^{-1} (xi e^{-beta K} - I)
    let mut kmat = DenseMat::zeros(2);
    kmat.set(0, 0, c(0.7, 0.0));
    kmat.set(0, 1, c(0.3, 0.2));
    kmat.set(1, 0, c(-0.1, 0.4));
    kmat.set(1, 1, c(-0.5, 0.0));
    let beta = 4.0f64;
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let xi = sign.as_i64() as f64;
        let one: i64 = if sign == Sign::Fermi { 1 } else { 0 };
        for &n_int in &[64usize, 128] {
            let mut a = MatComp::new(n_int, 2, 2, Complex::ZERO);
            for m in 0..=(n_int as i64) {
                let t = m as f64 * (beta / n_int as f64);
                let mut arg = kmat.clone();
                for q in 0..4 {
                    arg.raw_mut()[q] *= c(-t, 0.0);
                }
                a.block_mut(m as usize).copy_from_slice(expm(&arg).raw());
            }
            let mdft = matsubara_dft(&a, sign);
            let mut err = 0.0f64;
            for &n in &[
                -3 * (n_int as i64) - 1,
                -(n_int as i64),
                -2,
                0,
                1,
                7,
                (n_int as i64) / 2,
                2 * (n_int as i64) + 3,
            ] {
                let w = (2 * n + one) as f64 * std::f64::consts::PI / beta;
                // exact = (i w I - K)^{-1} (xi e^{-beta K} - I)
                let mut ebk = kmat.clone();
                for q in 0..4 {
                    ebk.raw_mut()[q] *= c(-beta, 0.0);
                }
                let ebk = expm(&ebk);
                let mut rhs = DenseMat::zeros(2);
                for i in 0..2 {
                    for j in 0..2 {
                        let v = ebk.get(i, j) * c(xi, 0.0);
                        rhs.set(i, j, v - if i == j { Complex::ONE } else { Complex::ZERO });
                    }
                }
                let mut lw = DenseMat::zeros(2);
                for i in 0..2 {
                    for j in 0..2 {
                        let diag = if i == j { c(0.0, w) } else { Complex::ZERO };
                        lw.set(i, j, diag - kmat.get(i, j));
                    }
                }
                let exact = mat_inv(&lw).mat_mul(&rhs);
                let num = matsubara_ft_direct(n, &a, &mdft, sign, beta, 3);
                let d: f64 = (0..4).map(|q| (num[q] - exact.raw()[q]).abs().powi(2)).sum();
                err = err.max(d.sqrt());
            }
            // fourth order corrected DFT
            let tol = if n_int == 64 { 2e-6 } else { 2e-7 };
            assert!(err < tol, "sign {xi:?} N {n_int}: {err}");

            // without the boundary correction the error is much larger
            let w1 = (2 + one) as f64 * std::f64::consts::PI / beta;
            let mut ebk = kmat.clone();
            for q in 0..4 {
                ebk.raw_mut()[q] *= c(-beta, 0.0);
            }
            let ebk = expm(&ebk);
            let mut rhs = DenseMat::zeros(2);
            for i in 0..2 {
                for j in 0..2 {
                    let v = ebk.get(i, j) * c(xi, 0.0);
                    rhs.set(i, j, v - if i == j { Complex::ONE } else { Complex::ZERO });
                }
            }
            let mut lw = DenseMat::zeros(2);
            for i in 0..2 {
                for j in 0..2 {
                    let diag = if i == j { c(0.0, w1) } else { Complex::ZERO };
                    lw.set(i, j, diag - kmat.get(i, j));
                }
            }
            let exact1 = mat_inv(&lw).mat_mul(&rhs);
            let num0 = matsubara_ft_direct(1, &a, &mdft, sign, beta, 0);
            let d: f64 = (0..4)
                .map(|q| (num0[q] - exact1.raw()[q]).abs().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d > 1e-3, "uncorrected error {d}");
        }
    }
}

#[test]
#[should_panic]
fn ft_direct_rejects_order_two() {
    let a = MatComp::new(64, 1, 1, Complex::ONE);
    let mdft = matsubara_dft(&a, Sign::Fermi);
    let _ = matsubara_ft_direct(1, &a, &mdft, Sign::Fermi, 4.0, 2);
}

#[test]
fn matsubara_seed_embedding() {
    let beta = 5.0f64;
    let ntau = 200usize; // 201 points
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let mu = if sign == Sign::Fermi { 0.0 } else { -2.0 };
        let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
        let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
        let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
        let (_c, sigma, g_exact, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, sign);
        let i5 = Integrator::new(5);
        let (g0, f) = dyson_mat_kernel(&g_exact, &sigma, mu, &h, beta, &i5);

        // (1 + F) X = G0: the seed is limited by the truncated sum
        let mut x = MatComp::new(ntau, 2, 2, Complex::ZERO);
        matsubara_seed(&mut x, &f, &g0, beta, sign, 5, 3);
        let d = max_abs(x.raw(), g_exact.mat.raw());
        assert!(d < 2e-3, "seed vs exact: {d}");

        // F = 0: the seed reproduces B
        let z = MatComp::new(ntau, 2, 2, Complex::ZERO);
        matsubara_seed(&mut x, &z, &g0, beta, sign, 5, 3);
        let d = max_abs(x.raw(), g0.raw());
        assert!(d < 1e-3, "zero kernel: {d}");
    }
}

#[test]
#[should_panic]
fn seed_rejects_odd_intervals() {
    let mut x = MatComp::new(41, 2, 2, Complex::ZERO);
    let f = MatComp::new(41, 2, 2, Complex::ZERO);
    let b = MatComp::new(41, 2, 2, Complex::ONE);
    matsubara_seed(&mut x, &f, &b, 5.0, Sign::Fermi, 5, 3);
}

#[test]
#[should_panic]
fn seed_rejects_mismatched_grids() {
    let mut x = MatComp::new(41, 2, 2, Complex::ZERO);
    let f = MatComp::new(40, 2, 2, Complex::ZERO);
    let b = MatComp::new(40, 2, 2, Complex::ONE);
    matsubara_seed(&mut x, &f, &b, 5.0, Sign::Fermi, 5, 3);
}

#[test]
#[should_panic]
fn seed_rejects_nan() {
    let mut x = MatComp::new(40, 2, 2, Complex::ZERO);
    let f = MatComp::new(40, 2, 2, Complex::ZERO);
    let mut b = MatComp::new(40, 2, 2, Complex::ONE);
    b.block_mut(3).fill(Complex::new(f64::NAN, 0.0));
    matsubara_seed(&mut x, &f, &b, 5.0, Sign::Fermi, 5, 3);
}

#[test]
fn nessi_catch_matsubara() {
    let (eps1, eps2, lam) = (-1.0f64, 1.0f64, 0.2f64);
    let (mu, beta, ntau) = (0.0f64, 20.0, 400usize); // 401 points
    let mut h2x2 = DenseMat::zeros(2);
    h2x2.set(0, 0, c(eps1, 0.0));
    h2x2.set(0, 1, c(0.0, lam));
    h2x2.set(1, 0, c(0.0, -lam));
    h2x2.set(1, 1, c(eps2, 0.0));
    let g2x2 = dm_free_mat(&h2x2, mu, beta, ntau, Sign::Fermi);
    let mut h1 = DenseMat::zeros(1);
    h1.set(0, 0, c(eps2, 0.0));
    let g22 = dm_free_mat(&h1, mu, beta, ntau, Sign::Fermi);

    let contour = Contour::new(1, ntau as i64, 1, 1, 1.0, beta);
    let mut g_exact = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let mut sigma = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    for m in 0..=(ntau as i64) {
        g_exact.set_mat(m, 0, 0, g2x2.block(m as usize)[0]);
        sigma.set_mat(m, 0, 0, c(lam * lam, 0.0) * g22.block(m as usize)[0]);
    }
    let mut he = DenseMat::zeros(1);
    he.set(0, 0, c(eps1, 0.0));
    let h = tf_of(&he);

    for &method in &[MatsubaraMethod::Fixpoint, MatsubaraMethod::Cg] {
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        dyson_mat(&mut g, &sigma, mu, &h, beta, 5, method, true, None, 1e-12);
        let d = g.distance_at(&g_exact, MATSUBARA);
        assert!(d < 1e-7, "{method:?}: {d}");
    }
}

#[test]
fn embedding_model() {
    let beta = 5.0f64;
    let ntau = 160usize; // 161 points
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let mu = if sign == Sign::Fermi { 0.1 } else { -2.0 };
        let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
        let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.1)], vec![c(0.2, -0.1), c(1.5, 0.0)]]);
        let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
        let (contour, sigma, g_exact, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, sign);

        let mut results = Vec::new();
        for &method in &[MatsubaraMethod::Fixpoint, MatsubaraMethod::Cg] {
            let mut g = ContourGreen::new(&contour, sign, Complex::ZERO);
            dyson_mat(&mut g, &sigma, mu, &h, beta, 5, method, true, None, 1e-12);
            let d = g.distance_at(&g_exact, MATSUBARA);
            assert!(d / (ntau as f64) < 1e-7, "{method:?} sign {sign:?}: {d}");
            // hermitian Matsubara component
            for m in 0..=(ntau as i64) {
                let blk = g.mat.block(m as usize);
                let dm = (blk[0] - blk[0].conj())
                    .abs()
                    .max((blk[1] - blk[2].conj()).abs())
                    .max((blk[3] - blk[3].conj()).abs());
                assert!(dm < 1e-14, "hermitian at m = {m}");
            }
            results.push(g);
        }
        let d = max_abs(results[0].mat.raw(), results[1].mat.raw());
        assert!(d < 1e-9, "fixpoint vs cg: {d}");

        // the accuracy depends on the integration order
        let mut g1 = ContourGreen::new(&contour, sign, Complex::ZERO);
        dyson_mat(&mut g1, &sigma, mu, &h, beta, 1, MatsubaraMethod::Fixpoint, true, None, 1e-12);
        assert!(
            g1.distance_at(&g_exact, MATSUBARA) > 100.0 * results[0].distance_at(&g_exact, MATSUBARA),
            "order 1 must be worse"
        );

        // without the symmetrization the result is still hermitian up to
        // the discretization error
        let mut gn = ContourGreen::new(&contour, sign, Complex::ZERO);
        dyson_mat(&mut gn, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, false, None, 1e-12);
        let d = max_abs(gn.mat.raw(), results[0].mat.raw());
        assert!(d < 1e-8, "force_hermitian: {d}");
    }
}

#[test]
fn fixpoint_and_cg_iterations() {
    let (beta, ntau, mu) = (5.0f64, 80usize, 0.0); // 81 points
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g_exact, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let i5 = Integrator::new(5);

    // fixpoint: maxiter = 0 is the seed, the corrections converge
    let mut gref = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat_fixpoint(&mut gref, &sigma, mu, &h, beta, &i5, 30);
    let mut errs = Vec::new();
    for n in 0..=6 {
        let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
        dyson_mat_fixpoint(&mut g, &sigma, mu, &h, beta, &i5, n);
        errs.push(g.distance_at(&gref, MATSUBARA));
    }
    assert!(errs[1] < errs[0], "first correction must improve");
    assert!(errs[6] < 1e-10, "converged: {}", errs[6]);

    // CG: no iteration
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    assert_eq!(dyson_mat_cg(&mut g, &sigma, mu, &h, beta, &i5, 0, 1e-12), 0);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    assert_eq!(dyson_mat_cg(&mut g, &sigma, mu, &h, beta, &i5, 40, 1e3), 0);
    // CG: the iteration limit is exhausted
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    assert_eq!(dyson_mat_cg(&mut g, &sigma, mu, &h, beta, &i5, 2, 0.0), 2);
    // CG: stop inside the loop
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    let niter = dyson_mat_cg(&mut g, &sigma, mu, &h, beta, &i5, 40, 1e-8);
    assert!((1..40).contains(&niter), "niter = {niter}");
    assert!(g.distance_at(&gref, MATSUBARA) < 1e-6);
    // CG: the default parameters
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat_cg(&mut g, &sigma, mu, &h, beta, &i5, 40, 1e-12);
    assert!(g.distance_at(&gref, MATSUBARA) < 1e-7);

    // the inner product sums the real parts of all elements of X'Y
    let mut x = MatComp::new(ntau, 2, 2, Complex::ZERO);
    let mut y = MatComp::new(ntau, 2, 2, Complex::ZERO);
    let xb = [c(1.0, 0.0), c(0.0, 2.0), c(0.5, 0.0), c(-1.0, 0.0)];
    let yb = [c(0.3, 0.0), c(1.0, 0.0), c(0.0, -2.0), c(0.7, 0.0)];
    x.block_mut(39).copy_from_slice(&xb);
    y.block_mut(39).copy_from_slice(&yb);
    // sum of the real parts of all elements of X^dagger Y
    let prod = [
        xb[0].conj() * yb[0] + xb[2].conj() * yb[2],
        xb[0].conj() * yb[1] + xb[2].conj() * yb[3],
        xb[1].conj() * yb[0] + xb[3].conj() * yb[2],
        xb[1].conj() * yb[1] + xb[3].conj() * yb[3],
    ];
    let sum_reals: f64 = prod.iter().map(|z| z.re).sum();
    let tr = prod[0].re + prod[3].re;
    let got = mat_inner(&x, &y, &i5);
    assert!((got - sum_reals).abs() < 1e-12, "{got} vs {sum_reals}");
    assert!((got - tr).abs() > 1e-12);

    // Gregory end weights
    let e00 = [c(1.0, 0.0), Complex::ZERO, Complex::ZERO, Complex::ZERO];
    let e11 = [Complex::ZERO, Complex::ZERO, Complex::ZERO, c(1.0, 0.0)];
    x.block_mut(0).copy_from_slice(&e00);
    y.block_mut(0).copy_from_slice(&e00);
    x.block_mut(79).copy_from_slice(&e11);
    y.block_mut(79).copy_from_slice(&e11);
    let want = sum_reals + i5.giw.omega[0] + i5.giw.omega[1];
    let got = mat_inner(&x, &y, &i5);
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

#[test]
fn fourier_method_is_finite() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fourier, true, None, 1e-12);
    for z in g.mat.raw() {
        assert!(z.re.is_finite() && z.im.is_finite());
    }
}

#[test]
#[should_panic]
fn reject_solve_order_zero() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 0, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_solve_order_six() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 6, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_statistics_mismatch() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut g = ContourGreen::new(&contour, Sign::Bose, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_dimension_mismatch() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (_c, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let c1 = Contour::new(1, ntau as i64, 1, 1, 1.0, beta);
    let mut g = ContourGreen::new(&c1, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_grid_mismatch() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (_c, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let c2 = Contour::new(1, (ntau + 2) as i64, 2, 2, 1.0, beta);
    let mut g = ContourGreen::new(&c2, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_odd_intervals_in_solver() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (_c, _sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    // ntau + 1 points = odd number of intervals: the seed rejects it
    let c3 = Contour::new(1, (ntau + 1) as i64, 2, 2, 1.0, beta);
    let mut sigma3 = ContourGreen::new(&c3, Sign::Fermi, Complex::ZERO);
    for m in 0..=(ntau as i64 + 1) {
        sigma3.set_mat(m, 0, 0, Complex::new(0.01, 0.0));
        sigma3.set_mat(m, 1, 1, Complex::new(0.01, 0.0));
    }
    let mut g = ContourGreen::new(&c3, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma3, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_non_hermitian_hamiltonian() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, _h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut hn = DenseMat::zeros(2);
    hn.set(0, 0, c(-1.0, 0.0));
    hn.set(0, 1, c(0.3, 0.0));
    hn.set(1, 0, Complex::ZERO);
    hn.set(1, 1, c(0.5, 0.0));
    let h = tf_of(&hn);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_nan_hamiltonian() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, _h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut hn = DenseMat::zeros(2);
    hn.set(0, 0, c(f64::NAN, 0.0));
    hn.set(1, 1, c(0.5, 0.0));
    let h = tf_of(&hn);
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigma, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
#[should_panic]
fn reject_nan_self_energy() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, _sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut sigman = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    for m in 0..=(ntau as i64) {
        sigman.set_mat(m, 0, 0, Complex::new(0.01, 0.0));
        sigman.set_mat(m, 1, 1, Complex::new(0.01, 0.0));
    }
    sigman.set_mat(2, 0, 0, c(f64::NAN, 0.0));
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigman, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
}

#[test]
fn non_hermitian_self_energy_is_accepted() {
    let (beta, ntau, mu) = (5.0f64, 40usize, 0.0);
    let ha = DenseMat::from_rows(&[vec![c(-1.0, 0.0), c(0.0, 0.3)], vec![c(0.0, -0.3), c(0.5, 0.0)]]);
    let hb = DenseMat::from_rows(&[vec![c(1.0, 0.0), c(0.2, 0.0)], vec![c(0.2, 0.0), c(1.5, 0.0)]]);
    let v = DenseMat::from_rows(&[vec![c(0.4, 0.0), c(0.0, 0.1)], vec![c(-0.2, 0.0), c(0.3, 0.0)]]);
    let (contour, sigma, _g, h) = dm_embedding(&ha, &hb, &v, mu, beta, ntau, Sign::Fermi);
    let mut sigmah = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    for m in 0..=(ntau as i64) {
        for i in 0..2 {
            for j in 0..2 {
                let mut v2 = sigma.get_mat(m, i, j);
                if i == 0 && j == 1 {
                    v2 += c(0.0, 0.01);
                }
                sigmah.set_mat(m, i, j, v2);
            }
        }
    }
    let mut g = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
    dyson_mat(&mut g, &sigmah, mu, &h, beta, 5, MatsubaraMethod::Fixpoint, true, None, 1e-12);
    for z in g.mat.raw() {
        assert!(z.re.is_finite() && z.im.is_finite());
    }
}
