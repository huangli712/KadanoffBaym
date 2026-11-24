// Project : KadanoffBaym
// File    : tests/vie/mats.rs
// Purpose : Matsubara VIE2 solvers: fixpoint, CG dispatch, and argument
//           validation
//

use super::setup::*;

#[test]
fn vie2_matsubara_fixpoint_and_cg() {
    for &sign in &[Sign::Fermi, Sign::Bose] {
        let s = setup(10, 5, sign);
        let i5 = Integrator::new(5);
        let ntau = s.g.ntau as i64;
        let mut g1 = ContourGreen::new(&s.contour, sign, Complex::ZERO);
        vie2_mat(
            &mut g1,
            &s.f,
            &s.fcc,
            &s.q,
            s.beta,
            &i5,
            MatsubaraMethod::Fixpoint,
            6,
            40,
            1e-12,
        );
        let mut g2 = ContourGreen::new(&s.contour, sign, Complex::ZERO);
        vie2_mat(
            &mut g2,
            &s.f,
            &s.fcc,
            &s.q,
            s.beta,
            &i5,
            MatsubaraMethod::Cg,
            6,
            40,
            1e-12,
        );
        assert!(g1.distance_at(&s.g, MATSUBARA) / (ntau as f64) < 1e-12);
        assert!(g2.distance_at(&s.g, MATSUBARA) / (ntau as f64) < 1e-12);

        // residual of the discrete equation G + F * G - Q
        for g in [&g1, &g2] {
            let mut r = MatComp::new(ntau as usize, 2, 2, Complex::ZERO);
            conv_mat(&mut r, &s.f.mat, &g.mat, &i5, s.beta, sign);
            let mut m = 0.0f64;
            for q in 0..r.raw().len() {
                m = m.max((r.raw()[q] + g.mat.raw()[q] - s.q.mat.raw()[q]).abs());
            }
            assert!(m < 1e-10, "residual {m}");
        }
        // only the Matsubara component is written
        assert!(g1.ret.raw().iter().all(|z| *z == Complex::ZERO));
        assert!(g1.less.raw().iter().all(|z| *z == Complex::ZERO));
    }
}

#[test]
fn vie2_matsubara_dispatch_and_errors() {
    let s = setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_mat(
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        s.beta,
        &i2,
        MatsubaraMethod::Fourier,
        6,
        40,
        1e-12,
    );
    for z in g.mat.raw() {
        assert!(z.re.is_finite() && z.im.is_finite());
    }
}

#[test]
#[should_panic]
fn vie2_mat_rejects_negative_beta() {
    let s = setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_mat(
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        -1.0,
        &i2,
        MatsubaraMethod::Fixpoint,
        6,
        40,
        1e-12,
    );
}

#[test]
#[should_panic]
fn vie2_mat_rejects_order_six() {
    let s = setup(6, 2, Sign::Fermi);
    let i6 = Integrator::new(6);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    vie2_mat(
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        s.beta,
        &i6,
        MatsubaraMethod::Fixpoint,
        6,
        40,
        1e-12,
    );
}

#[test]
#[should_panic]
fn vie2_mat_rejects_statistics_mismatch() {
    let s = setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let mut g = ContourGreen::new(&s.contour, Sign::Bose, Complex::ZERO);
    vie2_mat(
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        s.beta,
        &i2,
        MatsubaraMethod::Fixpoint,
        6,
        40,
        1e-12,
    );
}

#[test]
#[should_panic]
fn vie2_mat_rejects_grid_mismatch() {
    let s = setup(6, 2, Sign::Fermi);
    let i2 = Integrator::new(2);
    let c7 = Contour::new(7, 11, 2, 2, 1.0, 1.0);
    let mut g = ContourGreen::new(&c7, Sign::Fermi, Complex::ZERO);
    vie2_mat(
        &mut g,
        &s.f,
        &s.fcc,
        &s.q,
        s.beta,
        &i2,
        MatsubaraMethod::Fixpoint,
        6,
        40,
        1e-12,
    );
}
