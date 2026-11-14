mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::dyson::{
    pseudo_dyson, pseudo_dyson_mat, pseudo_dyson_start, pseudo_dyson_timestep,
};
use kadanoffbaym::types::Contour;
use kadanoffbaym::types::Sign;
use kadanoffbaym::types::TimeFunction;
use kadanoffbaym::types::MATSUBARA;
use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

const LAMBDA0: f64 = 0.5;
const BETA: f64 = 2.0;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn pd_h0_2() -> DenseMat {
    DenseMat::from_vec(
        2,
        2,
        vec![c(0.3, 0.0), c(0.2, -0.1), c(0.2, 0.1), c(-0.4, 0.0)],
    )
}

fn pd_a(t: f64) -> f64 {
    0.3 * (2.0 * t).sin()
}

fn pd_aa(t: f64) -> f64 {
    0.15 * (1.0 - (2.0 * t).cos())
}

/// H[MATSUBARA] = H0, H[n] = H0 + a(t) Id (the blueprint `pd_ham`).
fn pd_ham(nt: i64, h: f64, h0: &DenseMat) -> TimeFunction {
    let d = h0.nrows();
    let mut f = TimeFunction::new(nt as usize, d, d, Complex::ZERO);
    f.block_mut(MATSUBARA).copy_from_slice(h0.raw());
    let mut v = vec![Complex::ZERO; d * d];
    for n in 0..=nt {
        for i in 0..d {
            v[i * d + i] = h0.get(i, i) + c(pd_a(n as f64 * h), 0.0);
            for j in 0..d {
                if i != j {
                    v[i * d + j] = h0.get(i, j);
                }
            }
        }
        f.block_mut(n).copy_from_slice(&v);
    }
    f
}

fn norm_of(m: &DenseMat) -> f64 {
    m.raw().iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt()
}

/// The four component errors against the atomic limit (blueprint
/// `pd_errors`); all the indices are my 0-based ones.
fn pd_errors(
    g: &PseudoGreen,
    h0: &DenseMat,
    lambda0: f64,
    beta: f64,
    h: f64,
) -> (f64, f64, f64, f64) {
    let (nt, ntau) = (g.nt as i64, g.ntau as i64);
    let d = g.size1;
    let xi = g.sign.as_i64() as f64;
    let dtau = beta / ntau as f64;
    let _id = DenseMat::identity(d);
    // A = H0 - lambda0 I
    let mut a_m = h0.clone();
    for i in 0..d {
        a_m.set(i, i, h0.get(i, i) - c(lambda0, 0.0));
    }
    let u_of = |t: f64| -> DenseMat {
        // U(t) = exp(-i A t - i int a) = exp(-i (A t + aa(t) I))
        let mut m = kadanoffbaym::linalg::scale_c(&a_m, Complex::new(t, 0.0));
        for i in 0..d {
            m.set(i, i, m.get(i, i) + c(pd_aa(t), 0.0));
        }
        kadanoffbaym::linalg::expm(&kadanoffbaym::linalg::scale_c(
            &m,
            Complex::new(0.0, -1.0),
        ))
    };
    let t_of = |n: i64| n as f64 * h;
    let mut errm = 0.0f64;
    for m in 0..=ntau {
        let want = kadanoffbaym::linalg::expm(&kadanoffbaym::linalg::scale_c(
            &a_m,
            Complex::new(-m as f64 * dtau, 0.0),
        ));
        let mut diff = g.mat.block(m as usize).to_vec();
        for q in 0..diff.len() {
            diff[q] += want.raw()[q];
        }
        errm = errm.max(norm_of(&DenseMat::from_vec(d, d, diff)));
    }
    let mut errg = 0.0f64;
    for n in 0..=nt {
        let un = u_of(t_of(n));
        for j in 0..=n {
            let uj = u_of(t_of(j));
            let want = kadanoffbaym::linalg::scale_c(&un.mat_mul(&uj.adjoint()), Complex::new(0.0, -1.0));
            let mut diff = g.ret.raw_block_at(n, j).to_vec();
            for q in 0..diff.len() {
                diff[q] -= want.raw()[q];
            }
            errg = errg.max(norm_of(&DenseMat::from_vec(d, d, diff)));
        }
    }
    let mut errt = 0.0f64;
    for n in 0..=nt {
        let g0 = DenseMat::from_vec(d, d, g.ret.raw_block_at(n, 0).to_vec());
        for m in 0..=ntau {
            let want = kadanoffbaym::linalg::scale_c(
                &g0.mat_mul(&kadanoffbaym::linalg::expm(&kadanoffbaym::linalg::scale_c(
                    &a_m,
                    Complex::new(-(beta - m as f64 * dtau), 0.0),
                ))),
                Complex::new(xi, 0.0),
            );
            let mut diff = g.lmix.raw_block_at(n, m).to_vec();
            for q in 0..diff.len() {
                diff[q] -= want.raw()[q];
            }
            errt = errt.max(norm_of(&DenseMat::from_vec(d, d, diff)));
        }
    }
    let mut errl = 0.0f64;
    for n in 0..=nt {
        let un = u_of(t_of(n));
        for j in 0..=n {
            let uj = u_of(t_of(j));
            let want = kadanoffbaym::linalg::scale_c(
                &uj.mat_mul(&kadanoffbaym::linalg::expm(&kadanoffbaym::linalg::scale_c(
                    &a_m,
                    Complex::new(-beta, 0.0),
                )))
                .mat_mul(&un.adjoint()),
                Complex::new(0.0, -xi),
            );
            let mut diff = g.less.raw_block_at(j, n).to_vec();
            for q in 0..diff.len() {
                diff[q] -= want.raw()[q];
            }
            errl = errl.max(norm_of(&DenseMat::from_vec(d, d, diff)));
        }
    }
    (errm, errg, errt, errl)
}

#[test]
fn pseudo_dyson_atomic_limit() {
    for sign in [Sign::Fermi, Sign::Bose] {
        for k in [1i64, 3, 5] {
            for d in [1usize, 2] {
                let (nt, ntau, h) = (11i64, 60i64, 0.01f64);
                let h0 = if d == 1 {
                    DenseMat::from_vec(1, 1, vec![c(0.3, 0.0)])
                } else {
                    pd_h0_2()
                };
                let cc = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, BETA);
                let mut g = PseudoGreen::new(&cc, sign, Complex::ZERO);
                let sigma = PseudoGreen::new(&cc, sign, Complex::ZERO);
                let hf = pd_ham(nt, h, &h0);
                pseudo_dyson(&mut g, LAMBDA0, &hf, &sigma, &Integrator::new(k), BETA, h);
                let (errm, errg, errt, errl) = pd_errors(&g, &h0, LAMBDA0, BETA, h);
                let tol = match k {
                    1 => 2e-2,
                    3 => 2e-5,
                    _ => 1e-7,
                };
                assert!(errm < tol, "sign={:?} k={k} d={d} errm {errm}", sign);
                assert!(errg < tol, "sign={:?} k={k} d={d} errg {errg}", sign);
                assert!(errt < tol, "sign={:?} k={k} d={d} errt {errt}", sign);
                assert!(errl < tol, "sign={:?} k={k} d={d} errl {errl}", sign);
                // G^{>}(t, t) = -i
                for n in 0..=nt {
                    for i in 0..d {
                        for j in 0..d {
                            let want = if i == j { c(0.0, -1.0) } else { Complex::ZERO };
                            assert!(
                                (g.ret.raw_block_at(n, n)[i * d + j] - want).abs() < 1e-9
                            );
                        }
                    }
                }
                assert!(g.mat.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite()));
                assert!(g.lmix.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite()));
            }
        }
    }
}

#[test]
fn pseudo_dyson_mat_grid_convergence() {
    for k in [2i64, 4] {
        let mut err = Vec::new();
        for ntau in [20i64, 40, 80] {
            let cc = Contour::new(5, ntau, 2, 2, 0.05, BETA);
            let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
            let sigma = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
            pseudo_dyson_mat(
                &mut g,
                LAMBDA0,
                &pd_h0_2(),
                &sigma,
                &Integrator::new(k),
                BETA,
            );
            let dtau = BETA / ntau as f64;
            let mut a_m = pd_h0_2();
            for i in 0..2 {
                a_m.set(i, i, a_m.get(i, i) - c(LAMBDA0, 0.0));
            }
            a_m.scale(Complex::new(-1.0, 0.0).re);
            let mut e = 0.0f64;
            for m in 0..=ntau {
                let want = kadanoffbaym::linalg::expm(&kadanoffbaym::linalg::scale_c(
                    &a_m,
                    Complex::new(m as f64 * dtau, 0.0),
                ));
                let mut diff = g.mat.block(m as usize).to_vec();
                for q in 0..4 {
                    diff[q] += want.raw()[q];
                }
                e = e.max(norm_of(&DenseMat::from_vec(2, 2, diff)));
            }
            err.push(e);
        }
        assert!(err[1] < err[0] / 2f64.powi(k as i32), "k={k}");
        assert!(err[2] < err[1] / 2f64.powi(k as i32), "k={k}");
    }
}

#[test]
fn pseudo_dyson_startup_timestep_overall() {
    let (nt, ntau, h, k) = (13i64, 40i64, 0.02f64, 4i64);
    let cc = Contour::new(nt, ntau, 2, 2, nt as f64 * h, BETA);
    let i4 = Integrator::new(k);
    let hf = pd_ham(nt, h, &pd_h0_2());
    // a nonzero self-energy, built by a pseudo-particle Dyson solve
    let mut sigma = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let h2 = DenseMat::from_vec(2, 2, vec![c(0.1, 0.0), c(0.05, 0.0), c(0.05, 0.0), c(0.2, 0.0)]);
    let hf2 = pd_ham(nt, h, &h2);
    let zero = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    pseudo_dyson(&mut sigma, 0.2, &hf2, &zero, &i4, BETA, h);
    for t in MATSUBARA..=nt {
        sigma.smul_step(t, c(0.3, 0.0));
    }
    let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    pseudo_dyson(&mut g, LAMBDA0, &hf, &sigma, &i4, BETA, h);
    let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let h0 = DenseMat::from_vec(2, 2, hf.block(MATSUBARA).to_vec());
    pseudo_dyson_mat(&mut x, LAMBDA0, &h0, &sigma, &i4, BETA);
    pseudo_dyson_start(&mut x, LAMBDA0, &hf, &sigma, &i4, BETA, h);
    for tstp in (k + 1)..=nt {
        pseudo_dyson_timestep(tstp, &mut x, LAMBDA0, &hf, &sigma, &i4, BETA, h);
    }
    for t in MATSUBARA..=nt {
        assert_eq!(g.distance_at(&x, t), 0.0);
    }
    // the self-energy changes the result
    let mut y = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let zero2 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    pseudo_dyson(&mut y, LAMBDA0, &hf, &zero2, &i4, BETA, h);
    assert!(g.distance_at(&y, MATSUBARA) > 1e-3);
    assert!(g.distance_at(&y, 1) > 1e-4);
    // the start-up time step is rejected by the time stepping
    assert!(panics(|| {
        let mut xx = x.clone();
        pseudo_dyson_timestep(k, &mut xx, LAMBDA0, &hf, &sigma, &i4, BETA, h);
    }));
}

#[test]
fn pseudo_dyson_hamiltonian_adapter() {
    let (nt, ntau, h, k) = (9i64, 30i64, 0.02f64, 3i64);
    let i3 = Integrator::new(k);
    let cc = Contour::new(nt, ntau, 2, 2, nt as f64 * h, BETA);
    let hcf = pd_ham(nt, h, &pd_h0_2());
    // an identically assembled time function
    let mut hv = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    hv.block_mut(MATSUBARA).copy_from_slice(hcf.block(MATSUBARA));
    for n in 0..=nt {
        hv.block_mut(n).copy_from_slice(hcf.block(n));
    }
    let mut g1 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut g2 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let sigma = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    pseudo_dyson(&mut g1, LAMBDA0, &hcf, &sigma, &i3, BETA, h);
    pseudo_dyson(&mut g2, LAMBDA0, &hv, &sigma, &i3, BETA, h);
    for t in MATSUBARA..=nt {
        assert!(g1.distance_at(&g2, t) < 1e-12);
    }
    // a wrong-size Hamiltonian is rejected
    assert!(panics(|| {
        let hbig = pd_ham(nt, h, &DenseMat::identity(3));
        let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_dyson(&mut g, LAMBDA0, &hbig, &sigma, &i3, BETA, h);
    }));
}

#[test]
fn pseudo_dyson_argument_checks() {
    let cc = Contour::new(9, 30, 2, 2, 0.2, BETA);
    let hf = pd_ham(9, 0.02, &pd_h0_2());
    let sigma_bose = PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO);
    assert!(panics(|| {
        let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_dyson(&mut g, LAMBDA0, &hf, &sigma_bose, &Integrator::new(3), BETA, 0.02);
    }));
    assert!(panics(|| {
        let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        let sigma = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_dyson(&mut g, LAMBDA0, &hf, &sigma, &Integrator::new(6), BETA, 0.02);
    }));
    // a non-hermitian Hamiltonian is rejected
    assert!(panics(|| {
        let hn = pd_ham(9, 0.02, &DenseMat::from_vec(2, 2, vec![c(0.3, 0.0), c(0.2, 0.0), c(0.1, 0.0), c(-0.4, 0.0)]));
        let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        let sigma = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_dyson(&mut g, LAMBDA0, &hn, &sigma, &Integrator::new(3), BETA, 0.02);
    }));
}
