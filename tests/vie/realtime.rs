// Project : KadanoffBaym
// File    : tests/vie/realtime.rs
// Purpose : real-time VIE2 drivers and the timestep/start argument
//           validation
//

use super::setup::*;

#[test]
fn vie2_realtime_routing_and_drivers() {
    for &k in &[1i64, 3, 5] {
        for &sign in &[Sign::Fermi, Sign::Bose] {
            let s = setup(12, k, sign);
            let ik = Integrator::new(k);
            // full driver
            let mut g1 = ContourGreen::new(&s.contour, sign, Complex::ZERO);
            vie2(
                &mut g1,
                &s.f,
                &s.fcc,
                &s.q,
                &ik,
                s.beta,
                s.h,
                MatsubaraMethod::Fixpoint,
            );
            // step by step, including the reduced orders at the start
            let mut g2 = ContourGreen::new(&s.contour, sign, Complex::ZERO);
            for tstp in MATSUBARA..=s.nt {
                vie2_timestep(
                    tstp,
                    &mut g2,
                    &s.f,
                    &s.fcc,
                    &s.q,
                    &ik,
                    s.beta,
                    s.h,
                    MatsubaraMethod::Fixpoint,
                );
                if tstp == 0 {
                    assert_eq!(g2.ret.raw_block_at(0, 0), s.q.ret.raw_block_at(0, 0));
                } else if tstp >= 1 && tstp <= k {
                    let mut g3 = g2.clone();
                    vie2_start(&mut g3, &s.f, &s.fcc, &s.q, tstp, s.beta, s.h);
                    for n in MATSUBARA..=tstp {
                        assert_eq!(g2.distance_at(&g3, n), 0.0, "row {n}");
                    }
                }
            }
            for n in MATSUBARA..=s.nt {
                assert_eq!(g1.distance_at(&g2, n), 0.0, "driver row {n}");
            }
            // accuracy, the sum of the block norms at each time step
            let mut dmax = 0.0f64;
            for n in MATSUBARA..=s.nt {
                dmax = dmax.max(g1.distance_at(&s.g, n));
            }
            let tol = if k == 1 { 1e-1 } else { 2e-2 };
            assert!(dmax < tol, "k = {k}: {dmax}");
            // both F and Fcc are used: exchanging them changes the result
            let mut g4 = ContourGreen::new(&s.contour, sign, Complex::ZERO);
            vie2(
                &mut g4,
                &s.fcc,
                &s.f,
                &s.q,
                &ik,
                s.beta,
                s.h,
                MatsubaraMethod::Fixpoint,
            );
            let mut dsw = 0.0f64;
            for n in 0..=s.nt {
                dsw = dsw.max(g1.distance_at(&g4, n));
            }
            assert!(dsw > 1e-6, "swap must change the result: {dsw}");
        }
    }
}

#[test]
fn vie2_realtime_sentinels() {
    let k = 3i64;
    let s = setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    for tstp in MATSUBARA..=(s.nt - 1) {
        vie2_timestep(
            tstp,
            &mut g,
            &s.f,
            &s.fcc,
            &s.q,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
        );
    }
    let tstp = s.nt;
    let mut gref = g.clone();
    vie2_timestep(
        tstp,
        &mut gref,
        &s.f,
        &s.fcc,
        &s.q,
        &ik,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
    );
    let ntau = g.ntau as i64;

    // the current left-mixing row is cleared before the convolution
    for val in [1e6f64, f64::NAN] {
        let mut x = g.clone();
        vie2_timestep_ret(tstp, &mut x, &s.fcc, &s.f, &s.q, s.h, &ik);
        for m in 0..=ntau {
            x.lmix.raw_block_at_mut(tstp, m).fill(c(val, 0.0));
        }
        vie2_timestep_lmix(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for m in 0..=ntau {
            assert_eq!(
                x.lmix.raw_block_at(tstp, m),
                gref.lmix.raw_block_at(tstp, m)
            );
        }
    }

    // the current lesser column does not enter the source term
    for val in [1e6f64, f64::NAN] {
        let mut x = g.clone();
        vie2_timestep_ret(tstp, &mut x, &s.fcc, &s.f, &s.q, s.h, &ik);
        vie2_timestep_lmix(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for j in 0..=tstp {
            x.less.raw_block_at_mut(j, tstp).fill(c(val, 0.0));
        }
        vie2_timestep_less(tstp, &mut x, &s.f, &s.fcc, &s.q, s.beta, s.h, &ik);
        for j in 0..=tstp {
            assert_eq!(
                x.less.raw_block_at(j, tstp),
                gref.less.raw_block_at(j, tstp)
            );
        }
    }
}

#[test]
#[should_panic]
fn vie2_rejects_timestep_out_of_range() {
    let s = setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_timestep(
        s.nt + 1,
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        &i2,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
    );
}

#[test]
#[should_panic]
fn vie2_rejects_negative_timestep() {
    let s = setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_timestep(
        -2,
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        &i2,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
    );
}

#[test]
#[should_panic]
fn vie2_rejects_zero_step() {
    let s = setup(8, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_timestep(
        1,
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        &i2,
        s.beta,
        0.0,
        MatsubaraMethod::Fixpoint,
    );
}

#[test]
#[should_panic]
fn vie2_start_rejects_negative_beta() {
    let s = setup(8, 2, Sign::Fermi);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_start(&mut g, &s.f, &s.fcc, &s.q, 2, s.beta, -0.1);
}

#[test]
#[should_panic]
fn vie2_start_rejects_order_six() {
    let s = setup(8, 2, Sign::Fermi);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_start(&mut g, &s.f, &s.fcc, &s.q, 6, s.beta, 0.25);
}

#[test]
#[should_panic]
fn vie2_start_rejects_short_grid() {
    // 3 points = 2 intervals, solve order 3 needs more
    let s = setup(2, 2, Sign::Fermi);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_start(&mut g, &s.f, &s.fcc, &s.q, 3, s.beta, s.h);
}
