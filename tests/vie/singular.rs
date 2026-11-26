// Project : KadanoffBaym
// File    : tests/vie/singular.rs
// Purpose : singular-kernel VIE2 solver, workspaces, and validation
//

use super::setup::*;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// The singular VIE2 step 0 on the setup `s` with the singular kernel
/// `fs`, a vanishing `qs`, and the workspace `ws`.
fn singular_step0(s: &VtSetup, k: i64, fs: &TimeFunction, ws: &mut Vie2SingularWorkspace) {
    let ik = Integrator::new(k);
    let qs = TimeFunction::new(s.nt as usize, 2, 2, Complex::ZERO);
    let mut g = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut gs = TimeFunction::new(s.nt as usize, 2, 2, Complex::ZERO);
    vie2_timestep_singular(
        0,
        &mut g,
        &mut gs,
        &s.f,
        &s.fcc,
        fs,
        &s.q,
        &qs,
        &ik,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
        ws,
    );
}

#[test]
fn vie2_singular_kernel() {
    let k = 3i64;
    let s = setup(12, k, Sign::Fermi);
    let ik = Integrator::new(k);
    let nt = s.nt;
    let id = DenseMat::identity(2);

    // zero singular parts: the ordinary VIE2
    let mut fs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut g1 = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut g2 = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    for tstp in MATSUBARA..=nt {
        vie2_timestep(
            tstp,
            &mut g1,
            &s.f,
            &s.fcc,
            &s.q,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
        );
        let mut ws = Vie2SingularWorkspace::new(&g2);
        vie2_timestep_singular(
            tstp,
            &mut g2,
            &mut gs,
            &s.f,
            &s.fcc,
            &fs,
            &s.q,
            &qs,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
            &mut ws,
        );
    }
    for n in MATSUBARA..=nt {
        assert_eq!(g1.distance_at(&g2, n), 0.0, "row {n}");
    }

    // constructed problem: D = 1 + Fsing, F' = D F, Fcc' = Fcc D,
    // Q' = D Q + F' Gsing
    let mut dm = DenseMat::zeros(2);
    dm.set(0, 0, c(1.3, 0.0));
    dm.set(0, 1, c(0.0, 0.1));
    dm.set(1, 0, c(0.0, -0.1));
    dm.set(1, 1, c(0.8, 0.0));
    let dc = constant_time_function(nt, &dm);
    let mut gsx = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fblk = [c(0.5, 0.0), c(0.2, 0.0), c(0.2, 0.0), c(-0.4, 0.0)];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, dm.get(a, b) - id.get(a, b));
                qs.set(t, a, b, fblk[a * 2 + b] * c(1.0 + 0.1 * ijl, 0.0));
            }
        }
        // Gsx = D^-1 Qs
        let di = mat_inv(&dm);
        let qsm = DenseMat::from_vec(2, 2, qs.block(t).to_vec());
        let gx = di.mat_mul(&qsm);
        for a in 0..2usize {
            for b in 0..2usize {
                gsx.set(t, a, b, gx.get(a, b));
            }
        }
    }
    let mut fd = s.f.clone();
    let mut fccd = s.fcc.clone();
    let mut qd = s.q.clone();
    for t in MATSUBARA..=nt {
        let dm2 = DenseMat::from_vec(2, 2, dc.block(t).to_vec());
        fd.lmul_at(t, &dm2, Adjoint::No);
        fccd.rmul_at(t, &dm2, Adjoint::No);
        qd.lmul_at(t, &dm2, Adjoint::No);
    }
    let mut fg = fd.clone();
    for t in MATSUBARA..=nt {
        let gsm = DenseMat::from_vec(2, 2, gsx.block(t).to_vec());
        fg.rmul_at(t, &gsm, Adjoint::No);
        let step = timestep_at(&fg, t.max(0));
        qd.incr_by_step(&step, t, Complex::ONE);
    }
    let mut g3 = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut gs3 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    for tstp in MATSUBARA..=nt {
        let mut ws = Vie2SingularWorkspace::new(&g3);
        vie2_timestep_singular(
            tstp,
            &mut g3,
            &mut gs3,
            &fd,
            &fccd,
            &fs,
            &qd,
            &qs,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
            &mut ws,
        );
    }
    let mut dmax = 0.0f64;
    for n in MATSUBARA..=nt {
        dmax = dmax.max(g1.distance_at(&g3, n));
    }
    assert!(dmax < 1e-12, "regular part: {dmax}");
    let mut gerr = 0.0f64;
    for t in MATSUBARA..=nt {
        for q in 0..4 {
            gerr = gerr.max((gs3.block(t)[q] - gsx.block(t)[q]).abs());
        }
    }
    assert!(gerr < 1e-13, "singular part: {gerr}");
}

#[test]
#[should_panic(expected = "singular")]
fn vie2_singular_rejects_singular_kernel() {
    let k = 3i64;
    let s = setup(12, k, Sign::Fermi);
    let mut fs = TimeFunction::new(s.nt as usize, 2, 2, Complex::ZERO);
    let id = DenseMat::identity(2);
    for t in MATSUBARA..=s.nt {
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, -id.get(a, b));
            }
        }
    }
    let own = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut ws = Vie2SingularWorkspace::new(&own);
    singular_step0(&s, k, &fs, &mut ws);
}

#[test]
#[should_panic(expected = "hermitian")]
fn vie2_singular_rejects_non_hermitian_kernel() {
    let k = 3i64;
    let s = setup(12, k, Sign::Fermi);
    let mut fs = TimeFunction::new(s.nt as usize, 2, 2, Complex::ZERO);
    for t in MATSUBARA..=s.nt {
        fs.set(t, 0, 0, c(0.1, 0.0));
        fs.set(t, 0, 1, c(0.2, 0.0));
        fs.set(t, 1, 0, c(0.0, 0.0));
        fs.set(t, 1, 1, c(0.1, 0.0));
    }
    let own = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut ws = Vie2SingularWorkspace::new(&own);
    singular_step0(&s, k, &fs, &mut ws);
}

#[test]
fn vie2_singular_kernel_workspace() {
    let cases: Vec<(Sign, i64)> = if is_release() {
        vec![
            (Sign::Fermi, 1),
            (Sign::Fermi, 2),
            (Sign::Fermi, 3),
            (Sign::Fermi, 4),
            (Sign::Fermi, 5),
            (Sign::Bose, 1),
            (Sign::Bose, 2),
            (Sign::Bose, 3),
            (Sign::Bose, 4),
            (Sign::Bose, 5),
        ]
    } else {
        vec![(Sign::Fermi, 1), (Sign::Bose, 3), (Sign::Fermi, 5)]
    };
    for &(sign, k) in &cases {
        let (gr, gsr) = sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            sin_reference(tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h);
        });
        // without a persistent workspace
        let (g1, gs1) = sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            let mut ws = Vie2SingularWorkspace::new(g);
            vie2_timestep_singular(
                tstp,
                g,
                gs,
                f,
                fcc,
                fs,
                q,
                qs,
                ik,
                beta,
                h,
                MatsubaraMethod::Fixpoint,
                &mut ws,
            );
        });
        assert!(same_green(&g1, &gr), "fresh ws, k = {k}");
        for t in MATSUBARA..=(gr.nt as i64) {
            assert_eq!(gs1.block(t), gsr.block(t), "gs k = {k} at {t}");
        }
        // one workspace for all the calls
        let mut wsopt: Option<Vie2SingularWorkspace> = None;
        let (g2, gs2) = sin_pattern(k, sign, |tstp, g, gs, f, fcc, fs, q, qs, ik, beta, h| {
            if wsopt.is_none() {
                wsopt = Some(Vie2SingularWorkspace::new(g));
            }
            vie2_timestep_singular(
                tstp,
                g,
                gs,
                f,
                fcc,
                fs,
                q,
                qs,
                ik,
                beta,
                h,
                MatsubaraMethod::Fixpoint,
                wsopt.as_mut().unwrap(),
            );
        });
        assert!(same_green(&g2, &gr), "shared ws, k = {k}");
        for t in MATSUBARA..=(gr.nt as i64) {
            assert_eq!(gs2.block(t), gsr.block(t));
        }
    }
}

#[test]
fn vie2_singular_workspace_reset_and_invalidation() {
    let k = 2i64;
    let s = setup(10, k, Sign::Fermi);
    let nt = s.nt;
    let ik = Integrator::new(k);
    let mut fs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fblk = [
            c(0.2 * (1.0 + 0.02 * ijl), 0.0),
            c(0.0, 0.05),
            c(0.0, -0.05),
            c(-0.1 * (1.0 + 0.02 * ijl), 0.0),
        ];
        let qblk = [c(0.4, 0.0), c(0.1, 0.0), c(0.1, 0.0), c(-0.3, 0.0)];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, fblk[a * 2 + b]);
                qs.set(t, a, b, qblk[a * 2 + b]);
            }
        }
    }
    let mut f2 = s.f.clone();
    for t in MATSUBARA..=nt {
        f2.smul_at(t, c(0.9, 0.0));
    }
    let mut gr = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut gsr = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut gw = ContourGreen::new(&s.contour, Sign::Fermi, Complex::ZERO);
    let mut gsw = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut ws = Vie2SingularWorkspace::new(&gw);
    for tstp in MATSUBARA..=nt {
        sin_reference(
            tstp, &mut gr, &mut gsr, &s.f, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h,
        );
        vie2_timestep_singular(
            tstp,
            &mut gw,
            &mut gsw,
            &s.f,
            &s.fcc,
            &fs,
            &s.q,
            &qs,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
            &mut ws,
        );
    }
    assert!(same_green(&gw, &gr));
    assert_eq!(ws.ndone(), nt);

    // another kernel object at the last time step
    sin_reference(
        nt, &mut gr, &mut gsr, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h,
    );
    vie2_timestep_singular(
        nt,
        &mut gw,
        &mut gsw,
        &f2,
        &s.fcc,
        &fs,
        &s.q,
        &qs,
        &ik,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
        &mut ws,
    );
    assert!(same_green(&gw, &gr));

    // back to an earlier time step with modified inputs, then forward
    let id = DenseMat::identity(2);
    let t0 = k + 3;
    for t in t0..=nt {
        for a in 0..2usize {
            for b in 0..2usize {
                let v = fs.block(t)[a * 2 + b] + c(0.01, 0.0) * id.get(a, b);
                fs.set(t, a, b, v);
            }
        }
        f2.smul_at(t, c(1.1, 0.0));
    }
    for tstp in t0..=nt {
        sin_reference(
            tstp, &mut gr, &mut gsr, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h,
        );
        vie2_timestep_singular(
            tstp,
            &mut gw,
            &mut gsw,
            &f2,
            &s.fcc,
            &fs,
            &s.q,
            &qs,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
            &mut ws,
        );
    }
    assert!(same_green(&gw, &gr));
    for t in MATSUBARA..=nt {
        assert_eq!(gsw.block(t), gsr.block(t));
    }

    // another Gsing object at a later time step: the workspace resets
    let mut gsn = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut gsr2 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    sin_reference(
        nt, &mut gr, &mut gsr2, &f2, &s.fcc, &fs, &s.q, &qs, &ik, s.beta, s.h,
    );
    vie2_timestep_singular(
        nt,
        &mut gw,
        &mut gsn,
        &f2,
        &s.fcc,
        &fs,
        &s.q,
        &qs,
        &ik,
        s.beta,
        s.h,
        MatsubaraMethod::Fixpoint,
        &mut ws,
    );
    assert!(same_green(&gw, &gr));
    for t in MATSUBARA..=nt {
        assert_eq!(gsn.block(t), gsr2.block(t));
    }
}

#[test]
#[should_panic(expected = "workspace")]
fn vie2_singular_rejects_incompatible_workspace() {
    let k = 2i64;
    let s = setup(10, k, Sign::Fermi);
    let fs = TimeFunction::new(s.nt as usize, 2, 2, Complex::ZERO);
    let s2 = setup(6, k, Sign::Fermi);
    let mut bad =
        Vie2SingularWorkspace::new(&ContourGreen::new(&s2.contour, Sign::Fermi, Complex::ZERO));
    singular_step0(&s, k, &fs, &mut bad);
}
