// Project : KadanoffBaym
// File    : tests/base.rs
// Purpose : blueprint t_base.jl: initialization of contour-ordered
//           Green's functions (from_ham) against closed-form free
//           solutions
//
// NESSi reference: libcntr/cntr/cntr_equilibrium_impl.hpp (green_from_H)

use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::expm::expm;
use kadanoffbaym::linalg::eigh::eigh;
use kadanoffbaym::linalg::mat::DenseMat;
use kadanoffbaym::core::contour::{Contour, Sign};
use kadanoffbaym::core::green::ContourGreen;

const NT: i64 = 100; // intervals = points - 1
const NTAU: i64 = 100;
const TMAX: f64 = 1.0;
const BETA: f64 = 5.0;
const HSTEP: f64 = TMAX / 100.0;
const DTAU: f64 = BETA / 100.0;
const MU: f64 = 0.3;
const TOL: f64 = 1e-12;

fn approx(a: C64, b: C64) -> bool {
    (a - b).abs() <= TOL * (1.0 + b.abs())
}

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[test]
fn from_ham_scalar_fermion_and_boson() {
    let cases = [(Sign::Fermi, 1.0f64), (Sign::Bose, 1.0f64)];
    for &(sign, eps_band) in &cases {
        let cont = Contour::new(NT, NTAU, 1, 1, TMAX, BETA);
        let mut g = ContourGreen::new(&cont, sign, C64::ZERO);
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, c(eps_band, 0.0));
        from_ham(&mut g, MU, &h0, BETA, HSTEP);

        let xi = sign.as_i64() as f64;
        let e = eps_band - MU;
        let n = if sign == Sign::Fermi {
            1.0 / ((BETA * e).exp() + 1.0)
        } else {
            1.0 / ((BETA * e).exp() - 1.0)
        };

        // Matsubara: G(tau) = -e^{-eps tau} (1 + xi n)
        for m in 0..=NTAU {
            let tau = m as f64 * DTAU;
            let want = c(-(-e * tau).exp() * (1.0 + xi * n), 0.0);
            assert!(
                approx(g.get_mat(m, 0, 0), want),
                "mat({m}) = {:?} vs {want:?}",
                g.get_mat(m, 0, 0)
            );
        }

        // Retarded: G(t,t') = -i e^{-i eps (t-t')}
        for i in 0..=NT {
            for j in 0..=i {
                let want =
                    C64::new(0.0, -1.0) * C64::from_polar(1.0, -e * (i - j) as f64 * HSTEP);
                assert!(
                    approx(g.get_ret(i, j, 0, 0), want),
                    "ret({i},{j})"
                );
            }
        }

        // Left-mixing: G(t,tau) = -i xi n e^{eps tau} e^{-i eps t}
        for i in 0..=NT {
            for m in 0..=NTAU {
                let tau = m as f64 * DTAU;
                let want = -C64::new(0.0, xi * n)
                    * c((e * tau).exp(), 0.0)
                    * C64::from_polar(1.0, -e * i as f64 * HSTEP);
                assert!(
                    approx(g.get_lmix(i, m, 0, 0), want),
                    "lmix({i},{m})"
                );
            }
        }

        // Lesser: G(t',t) = -i xi n e^{-i eps (t-t')}
        for j in 0..=NT {
            for i in 0..=j {
                // e^{-i eps (i-j) h} with i-j <= 0, i.e. e^{+i eps (j-i) h}
                let want = C64::new(0.0, -xi * n)
                    * C64::from_polar(1.0, e * (j - i) as f64 * HSTEP);
                assert!(
                    approx(g.get_less(i, j, 0, 0), want),
                    "less({i},{j})"
                );
            }
        }

        // KMS boundary: G(0+) - xi G(beta-) = -1
        assert!(
            approx(
                g.get_mat(0, 0, 0) - g.get_mat(NTAU, 0, 0) * c(xi, 0.0),
                c(-1.0, 0.0)
            ),
            "KMS"
        );
        // Boundary: Grangle(0,tau) = i xi G^M(beta-tau)
        for m in 0..=NTAU {
            let want = C64::new(0.0, 1.0) * g.get_mat(NTAU - m, 0, 0) * c(xi, 0.0);
            assert!(
                approx(g.get_lmix(0, m, 0, 0), want),
                "lmix boundary at m = {m}"
            );
        }
        // Equal-time: G<(0,0) = Grangle(0,0)
        assert!(approx(g.get_less(0, 0, 0, 0), g.get_lmix(0, 0, 0, 0)));
    }
}

#[test]
fn from_ham_matrix_hermitian_fermion() {
    let cont = Contour::new(NT, NTAU, 2, 2, TMAX, BETA);
    let mut g = ContourGreen::new(&cont, Sign::Fermi, C64::ZERO);
    let mut h0 = DenseMat::zeros(2);
    let s = 2.0f64.sqrt();
    h0.set(0, 0, c(s, 0.0));
    h0.set(0, 1, c(0.0, s));
    h0.set(1, 0, c(0.0, -s));
    h0.set(1, 1, c(-s, 0.0));
    from_ham(&mut g, MU, &h0, BETA, HSTEP);

    // H_mu = H - mu I and rho = f(H - mu)
    let mut hmu = h0.clone();
    for i in 0..2 {
        let v = hmu.get(i, i);
        hmu.set(i, i, v - c(MU, 0.0));
    }
    let (vals, vecs) = eigh(&hmu);
    let mut rho = DenseMat::zeros(2);
    for a in 0..2 {
        for b in 0..2 {
            let mut v = C64::ZERO;
            for l in 0..2 {
                v += vecs.get(a, l)
                    * c(1.0 / ((BETA * vals[l]).exp() + 1.0), 0.0)
                    * vecs.get(b, l).conj();
            }
            rho.set(a, b, v);
        }
    }

    let eye = |i: usize, j: usize| if i == j { c(1.0, 0.0) } else { C64::ZERO };

    for i in 0..=NT {
        // G^R(t,t) = -i I
        for a in 0..2 {
            for b in 0..2 {
                assert!(
                    approx(
                        g.get_ret(i, i, a, b),
                        eye(a, b) * c(0.0, -1.0)
                    ),
                    "ret({i},{i})[{a},{b}]"
                );
                // G^<(t,t) = i rho, anti-hermitian
                let v = g.get_less(i, i, a, b);
                assert!(approx(v, rho.get(a, b) * c(0.0, 1.0)), "less diag");
                assert!(approx(v, -g.get_less(i, i, b, a).conj()), "anti-herm");
            }
        }
    }

    // Retarded: G(t,t') = -i exp(-i (H-mu) (t-t'))
    for i in 0..=NT {
        for j in 0..=i {
            let step = (i - j) as f64 * HSTEP;
            let mut arg = hmu.clone();
            for q in 0..arg.raw().len() {
                arg.raw_mut()[q] *= c(0.0, -step);
            }
            let want = {
                let e = expm(&arg);
                let mut w = DenseMat::zeros(2);
                for a in 0..2 {
                    for b in 0..2 {
                        w.set(a, b, e.get(a, b) * c(0.0, -1.0));
                    }
                }
                w
            };
            for a in 0..2 {
                for b in 0..2 {
                    assert!(
                        approx(g.get_ret(i, j, a, b), want.get(a, b)),
                        "ret({i},{j})[{a},{b}]"
                    );
                }
            }
        }
    }

    // Lesser: G(t',t) = i e^{-i(H-mu)t'} rho e^{i(H-mu)t}
    // (stored less(i,j) with i <= j, t_i = i h)
    for j in 0..=NT {
        for i in 0..=j {
            let ti = i as f64 * HSTEP;
            let tj = j as f64 * HSTEP;
            let mut argi = hmu.clone();
            let mut argj = hmu.clone();
            for q in 0..4 {
                argi.raw_mut()[q] *= c(0.0, -ti);
                argj.raw_mut()[q] *= c(0.0, tj);
            }
            let ei = expm(&argi);
            let ej = expm(&argj);
            let prod = ei.mat_mul(&rho).mat_mul(&ej);
            for a in 0..2 {
                for b in 0..2 {
                    assert!(
                        approx(g.get_less(i, j, a, b), prod.get(a, b) * c(0.0, 1.0)),
                        "less({i},{j})[{a},{b}] = {:?} vs {:?}",
                        g.get_less(i, j, a, b),
                        prod.get(a, b) * c(0.0, 1.0)
                    );
                }
            }
        }
    }

    // Matsubara: hermitian, G(beta-) = -rho, G(0+) = rho - I
    for m in 0..=NTAU {
        for a in 0..2 {
            for b in 0..2 {
                assert!(
                    approx(g.get_mat(m, a, b), g.get_mat(m, b, a).conj()),
                    "mat hermitian at m = {m}"
                );
            }
        }
    }
    for a in 0..2 {
        for b in 0..2 {
            assert!(
                approx(g.get_mat(NTAU, a, b), -rho.get(a, b)),
                "mat(beta)[{a},{b}]"
            );
            assert!(
                approx(
                    g.get_mat(0, a, b),
                    rho.get(a, b) - eye(a, b)
                ),
                "mat(0)[{a},{b}]"
            );
        }
    }

    // Boundary: Grangle(0,tau) = -i G^M(beta-tau)
    for m in 0..=NTAU {
        for a in 0..2 {
            for b in 0..2 {
                let want = g.get_mat(NTAU - m, a, b) * c(0.0, -1.0);
                assert!(
                    approx(g.get_lmix(0, m, a, b), want),
                    "lmix boundary at m = {m}"
                );
            }
        }
    }
}
