// Project : KadanoffBaym
// File    : tests/bubble.rs
// Purpose : port of julia/test/t_bubble.jl — bubble diagrams verified
//           against the closed forms of free Green's functions

mod common;

use common::prelude::*;
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::convolution::bubble::{bubble1, bubble1_herm, bubble2, bubble2_herm};
use kadanoffbaym::linalg::eigh;
use kadanoffbaym::types::MATSUBARA;

/// Closed form of a free Green's function of a hermitian H (blueprint
/// `BBFreeGF`): H - mu = R diag(eps) R^dag, n = 1/(exp(beta eps) - xi).
#[derive(Clone)]
struct BbFreeGf {
    r: DenseMat,
    eps: Vec<f64>,
    n: Vec<f64>,
    xi: i64,
}

impl BbFreeGf {
    fn new(h: &DenseMat, mu: f64, beta: f64, xi: i64) -> Self {
        let d = h.nrows();
        let mut hm = h.clone();
        for i in 0..d {
            hm.set(i, i, hm.get(i, i) - Complex::new(mu, 0.0));
        }
        let (eps, r) = eigh(&hm);
        let n = eps
            .iter()
            .map(|&e| 1.0 / ((beta * e).exp() - xi as f64))
            .collect();
        BbFreeGf { r, eps, n, xi }
    }

    fn mk(&self, d: &[Complex]) -> DenseMat {
        let n = self.eps.len();
        let mut dm = DenseMat::zeros(n);
        for i in 0..n {
            dm.set(i, i, d[i]);
        }
        self.r.mat_mul(&dm).mat_mul(&self.r.adjoint())
    }

    fn les(&self, t: f64, tp: f64) -> DenseMat {
        let d: Vec<Complex> = self
            .eps
            .iter()
            .zip(&self.n)
            .map(|(&e, &nv)| {
                Complex::new(0.0, -(self.xi as f64) * nv) * Complex::from_polar(1.0, -e * (t - tp))
            })
            .collect();
        self.mk(&d)
    }

    fn gtr(&self, t: f64, tp: f64) -> DenseMat {
        let d: Vec<Complex> = self
            .eps
            .iter()
            .zip(&self.n)
            .map(|(&e, &nv)| {
                Complex::new(0.0, -(1.0 + self.xi as f64 * nv)) * Complex::from_polar(1.0, -e * (t - tp))
            })
            .collect();
        self.mk(&d)
    }

    fn mat(&self, tau: f64) -> DenseMat {
        let d: Vec<Complex> = self
            .eps
            .iter()
            .zip(&self.n)
            .map(|(&e, &nv)| -(1.0 + self.xi as f64 * nv) * Complex::new((-e * tau).exp(), 0.0))
            .collect();
        self.mk(&d)
    }

    fn tv(&self, t: f64, tau: f64) -> DenseMat {
        let d: Vec<Complex> = self
            .eps
            .iter()
            .zip(&self.n)
            .map(|(&e, &nv)| {
                Complex::new(0.0, -(self.xi as f64) * nv)
                    * Complex::from_polar(1.0, -e * t)
                    * Complex::new((e * tau).exp(), 0.0)
            })
            .collect();
        self.mk(&d)
    }

    fn vt(&self, tau: f64, t: f64) -> DenseMat {
        let d: Vec<Complex> = self
            .eps
            .iter()
            .zip(&self.n)
            .map(|(&e, &nv)| {
                Complex::new(0.0, -(1.0 + self.xi as f64 * nv))
                    * Complex::from_polar(1.0, e * t)
                    * Complex::new((-e * tau).exp(), 0.0)
            })
            .collect();
        self.mk(&d)
    }
}

/// Contour function from the closed forms scaled by c (blueprint `bb_gf`).
fn bb_gf(c: &Contour, g: &BbFreeGf, cc: Complex, h: f64, beta: f64) -> ContourGreen {
    let (nt, ntau) = (c.nt as i64, c.ntau as i64);
    let mut out = ContourGreen::new(c, sign_of(g.xi), Complex::ZERO);
    let dtau = beta / ntau as f64;
    for m in 0..=ntau {
        let m0 = g.mat(m as f64 * dtau);
        for i in 0..(c.size1 as usize) {
            for j in 0..(c.size2 as usize) {
                out.set_mat(m, i, j, cc * m0.get(i, j));
            }
        }
    }
    for i in 0..=nt {
        for m in 0..=ntau {
            let m0 = g.tv(i as f64 * h, m as f64 * dtau);
            for a in 0..(c.size1 as usize) {
                for b in 0..(c.size2 as usize) {
                    out.set_lmix(i, m, a, b, cc * m0.get(a, b));
                }
            }
        }
    }
    for i in 0..=nt {
        for j in 0..=i {
            let (t, tp) = (i as f64 * h, j as f64 * h);
            let gtr = g.gtr(t, tp);
            let les = g.les(t, tp);
            let les_sw = g.les(tp, t);
            for a in 0..(c.size1 as usize) {
                for b in 0..(c.size2 as usize) {
                    out.set_ret(i, j, a, b, cc * (gtr.get(a, b) - les.get(a, b)));
                    out.set_less(j, i, a, b, cc * les_sw.get(a, b));
                }
            }
        }
    }
    out
}

fn sign_of(xi: i64) -> Sign {
    if xi == -1 {
        Sign::Fermi
    } else {
        Sign::Bose
    }
}


/// Maximum deviation of the element (c1, c2) from the expected values
/// (blueprint `bb_check`); the expected-value closures borrow the
/// closed forms directly.
fn bb_check<E1, E2, E3, E4>(
    c: &ContourGreen,
    c1: usize,
    c2: usize,
    h: f64,
    beta: f64,
    emat: E1,
    eret: E2,
    eles: E3,
    etv: E4,
) -> f64
where
    E1: Fn(f64) -> Complex,
    E2: Fn(f64, f64) -> Complex,
    E3: Fn(f64, f64) -> Complex,
    E4: Fn(f64, f64) -> Complex,
{
    let (nt, ntau) = (c.nt as i64, c.ntau as i64);
    let dtau = beta / ntau as f64;
    let mut err = 0.0f64;
    for m in 0..=ntau {
        err = err.max((c.get_mat(m, c1, c2) - emat(m as f64 * dtau)).abs());
    }
    for i in 0..=nt {
        let t = i as f64 * h;
        for j in 0..=i {
            let tp = j as f64 * h;
            err = err.max((c.get_ret(i, j, c1, c2) - eret(t, tp)).abs());
            err = err.max((c.get_less(j, i, c1, c2) - eles(tp, t)).abs());
        }
        for m in 0..=ntau {
            err = err.max((c.get_lmix(i, m, c1, c2) - etv(t, m as f64 * dtau)).abs());
        }
    }
    err
}

fn elem(m: &DenseMat, a: usize, b: usize) -> Complex {
    m.get(a, b)
}

#[allow(clippy::too_many_arguments)]
fn bb_run(nt: i64, ntau: i64, beta: f64, h: f64, d: usize, xa: i64, xb: i64, ca: Complex, cb: Complex) -> f64 {
    let ha = if d == 1 {
        let mut m = DenseMat::zeros(1);
        m.set(0, 0, Complex::new(0.4, 0.0));
        m
    } else {
        let mut m = DenseMat::zeros(2);
        m.set(0, 0, Complex::new(0.4, 0.0));
        m.set(0, 1, Complex::new(0.3, -0.2));
        m.set(1, 0, Complex::new(0.3, 0.2));
        m.set(1, 1, Complex::new(-0.6, 0.0));
        m
    };
    let hb = if d == 1 {
        let mut m = DenseMat::zeros(1);
        m.set(0, 0, Complex::new(1.2, 0.0));
        m
    } else {
        let mut m = DenseMat::zeros(2);
        m.set(0, 0, Complex::new(1.2, 0.0));
        m.set(0, 1, Complex::new(0.0, 0.1));
        m.set(1, 0, Complex::new(0.0, -0.1));
        m.set(1, 1, Complex::new(1.9, 0.0));
        m
    };
    let ga = BbFreeGf::new(&ha, if xa == -1 { 0.1 } else { -0.5 }, beta, xa);
    let gb = BbFreeGf::new(&hb, if xb == -1 { 0.2 } else { -0.3 }, beta, xb);
    let c = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, beta);
    let a = bb_gf(&c, &ga, ca, h, beta);
    let acc = bb_gf(&c, &ga, ca.conj(), h, beta);
    let b = bb_gf(&c, &gb, cb, h, beta);
    let bcc = bb_gf(&c, &gb, cb.conj(), h, beta);
    let cc = ca * cb;
    let im = Complex::new(0.0, 1.0);
    let mut errmax = 0.0f64;
    let (c1, c2) = if d == 1 { (0usize, 0usize) } else { (0usize, d - 1) };
    for a1 in 0..d {
        for a2 in 0..d {
            for b1 in 0..d {
                for b2 in 0..d {
                    // Bubble1: C(t,t') = i A_{a1a2}(t,t') B_{b2b1}(t',t)
                    let mut c1g = ContourGreen::new(&c, sign_of(xa * xb), Complex::ZERO);
                    bubble1(MATSUBARA, &mut c1g, c1, c2, &a, &acc, a1, a2, &b, &bcc, b1, b2);
                    for tstp in 0..=nt {
                        bubble1(tstp, &mut c1g, c1, c2, &a, &acc, a1, a2, &b, &bcc, b1, b2);
                    }
                    let e = bb_check(
                        &c1g,
                        c1,
                        c2,
                        h,
                        beta,
                        |tau| -cc * (xb as f64) * elem(&ga.mat(tau), a1, a2) * elem(&gb.mat(beta - tau), b2, b1),
                        |t, tp| {
                            im * cc
                                * (elem(&ga.gtr(t, tp), a1, a2) * elem(&gb.les(tp, t), b2, b1)
                                    - elem(&ga.les(t, tp), a1, a2) * elem(&gb.gtr(tp, t), b2, b1))
                        },
                        |tp, t| im * cc * elem(&ga.les(tp, t), a1, a2) * elem(&gb.gtr(t, tp), b2, b1),
                        |t, tau| im * cc * elem(&ga.tv(t, tau), a1, a2) * elem(&gb.vt(tau, t), b2, b1),
                    );
                    errmax = errmax.max(e);
                    // Bubble2: C(t,t') = i A_{a1a2}(t,t') B_{b1b2}(t,t')
                    let mut c2g = ContourGreen::new(&c, sign_of(xa * xb), Complex::ZERO);
                    bubble2(MATSUBARA, &mut c2g, c1, c2, &a, &acc, a1, a2, &b, &bcc, b1, b2);
                    for tstp in 0..=nt {
                        bubble2(tstp, &mut c2g, c1, c2, &a, &acc, a1, a2, &b, &bcc, b1, b2);
                    }
                    let e = bb_check(
                        &c2g,
                        c1,
                        c2,
                        h,
                        beta,
                        |tau| -cc * elem(&ga.mat(tau), a1, a2) * elem(&gb.mat(tau), b1, b2),
                        |t, tp| {
                            im * cc
                                * (elem(&ga.gtr(t, tp), a1, a2) * elem(&gb.gtr(t, tp), b1, b2)
                                    - elem(&ga.les(t, tp), a1, a2) * elem(&gb.les(t, tp), b1, b2))
                        },
                        |tp, t| im * cc * elem(&ga.les(tp, t), a1, a2) * elem(&gb.les(tp, t), b1, b2),
                        |t, tau| im * cc * elem(&ga.tv(t, tau), a1, a2) * elem(&gb.tv(t, tau), b1, b2),
                    );
                    errmax = errmax.max(e);
                }
            }
        }
    }
    errmax
}

fn grid() -> (i64, i64, f64, f64) {
    if common::is_release() {
        (101, 201, 5.0, 0.02)
    } else {
        (11, 21, 2.0, 0.1)
    }
}

#[test]
fn closed_forms_vs_from_ham() {
    let (nt, ntau, beta, h) = grid();
    let mut h0 = DenseMat::zeros(2);
    h0.set(0, 0, Complex::new(0.4, 0.0));
    h0.set(0, 1, Complex::new(0.3, -0.2));
    h0.set(1, 0, Complex::new(0.3, 0.2));
    h0.set(1, 1, Complex::new(-0.6, 0.0));
    for xi in [-1i64, 1] {
        let mu = if xi == -1 { 0.1 } else { -1.5 };
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let mut g = ContourGreen::new(&c, sign_of(xi), Complex::ZERO);
        from_ham(&mut g, mu, &h0, beta, h);
        let gx = bb_gf(&c, &BbFreeGf::new(&h0, mu, beta, xi), Complex::ONE, h, beta);
        let mut dmax = 0.0f64;
        for n in 0..=nt {
            dmax = dmax.max(g.distance_at(&gx, n));
        }
        dmax = dmax.max(g.distance_at(&gx, MATSUBARA));
        assert!(dmax < 1e-10, "xi {xi}: {dmax}");
    }
}

#[test]
fn bubble_all_combinations() {
    let (nt, ntau, beta, h) = grid();
    for d in [1usize, 2] {
        for xa in [-1i64, 1] {
            for xb in [-1i64, 1] {
                let e = bb_run(nt, ntau, beta, h, d, xa, xb, Complex::ONE, Complex::ONE);
                assert!(e < 1e-10, "d {d} xa {xa} xb {xb}: {e}");
            }
        }
    }
}

#[test]
fn bubble_explicit_non_hermitian_partners() {
    let (nt, ntau, beta, h) = grid();
    for d in [1usize, 2] {
        for xa in [-1i64, 1] {
            for xb in [-1i64, 1] {
                let e = bb_run(
                    nt,
                    ntau,
                    beta,
                    h,
                    d,
                    xa,
                    xb,
                    Complex::new(0.6, 0.8),
                    Complex::new(-0.3, 1.1),
                );
                assert!(e < 1e-10, "d {d} xa {xa} xb {xb}: {e}");
            }
        }
    }
}

#[test]
fn overload_equivalence_and_untouched_elements() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let mut h0 = DenseMat::zeros(2);
    h0.set(0, 0, Complex::new(0.4, 0.0));
    h0.set(0, 1, Complex::new(0.3, -0.2));
    h0.set(1, 0, Complex::new(0.3, 0.2));
    h0.set(1, 1, Complex::new(-0.6, 0.0));
    let a = bb_gf(&c, &BbFreeGf::new(&h0, 0.1, beta, -1), Complex::ONE, h, beta);
    for variant in 0..2 {
        let mut x = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        let mut y = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        for m in 0..=(c.ntau as i64) {
            x.set_mat(m, 0, 0, Complex::new(7.0, 0.0));
            x.set_mat(m, 0, 1, Complex::new(7.0, 0.0));
            x.set_mat(m, 1, 0, Complex::new(7.0, 0.0));
            x.set_mat(m, 1, 1, Complex::new(7.0, 0.0));
        }
        // the Matsubara step writes the mat element
        if variant == 0 {
            bubble1(MATSUBARA, &mut x, 1, 0, &a, &a, 0, 1, &a, &a, 1, 1);
            bubble1_herm(MATSUBARA, &mut y, 1, 0, &a, 0, 1, &a, 1, 1);
        } else {
            bubble2(MATSUBARA, &mut x, 1, 0, &a, &a, 0, 1, &a, &a, 1, 1);
            bubble2_herm(MATSUBARA, &mut y, 1, 0, &a, 0, 1, &a, 1, 1);
        }
        for tstp in 0..=(c.nt as i64) {
            if variant == 0 {
                bubble1(tstp, &mut x, 1, 0, &a, &a, 0, 1, &a, &a, 1, 1);
                bubble1_herm(tstp, &mut y, 1, 0, &a, 0, 1, &a, 1, 1);
            } else {
                bubble2(tstp, &mut x, 1, 0, &a, &a, 0, 1, &a, &a, 1, 1);
                bubble2_herm(tstp, &mut y, 1, 0, &a, 0, 1, &a, 1, 1);
            }
        }
        for n in 0..=(c.nt as i64) {
            assert_eq!(x.distance_at(&y, n), 0.0, "variant {variant} n {n}");
        }
        for m in 0..=(c.ntau as i64) {
            assert_eq!(x.get_mat(m, 0, 0), Complex::new(7.0, 0.0));
            assert_eq!(x.get_mat(m, 0, 1), Complex::new(7.0, 0.0));
            assert_eq!(x.get_mat(m, 1, 1), Complex::new(7.0, 0.0));
            assert_eq!(x.get_mat(m, 1, 0), y.get_mat(m, 1, 0));
        }
    }
}

#[test]
#[should_panic]
fn error_path_sign_mismatch1() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ONE);
    let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    bubble1_herm(1, &mut x, 0, 0, &a, 0, 0, &a, 0, 0);
}

#[test]
#[should_panic]
fn error_path_sign_mismatch2() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ONE);
    let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    bubble2_herm(1, &mut x, 0, 0, &a, 0, 0, &a, 0, 0);
}

#[test]
#[should_panic]
fn error_path_partner_stats_mismatch() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ONE);
    let ab = ContourGreen::new(&c, Sign::Bose, Complex::ONE);
    let mut x = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    bubble1(1, &mut x, 0, 0, &a, &ab, 0, 0, &a, &a, 0, 0);
}

#[test]
#[should_panic]
fn error_path_orbital_out_of_bounds() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ONE);
    let mut x = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    bubble1_herm(1, &mut x, 3, 0, &a, 0, 0, &a, 0, 0);
}

#[test]
#[should_panic]
fn error_path_tstp_out_of_bounds() {
    let (nt, ntau, beta, h) = grid();
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ONE);
    let mut x = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    bubble2_herm(nt + 1, &mut x, 0, 0, &a, 0, 0, &a, 0, 0);
}

