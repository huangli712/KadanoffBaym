// Project : KadanoffBaym
// File    : tests/pseudo/convolution.rs
// Purpose : the pseudo-particle convolution kernels
//

use kadanoffbaym::types::PseudoGreen;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{
    convolution_density_matrix, incr_convolution_less, incr_convolution_lmix, incr_convolution_ret,
    pseudo_convolution, pseudo_convolution_mat, pseudo_convolution_time_step,
    pseudo_incr_convolution, pseudo_incr_convolution_lmix, pseudo_incr_convolution_mat,
    pseudo_incr_convolution_step,
};
use kadanoffbaym::variants::pseudo_timestep;

use super::util::{block_norm, c, panics, rand_c64_scaled};

use super::util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const BETA: f64 = 2.0;

const H: f64 = 0.05;

const NT: i64 = 11; // 12 points

const NTAU: i64 = 20; // 21 points

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

fn block_difference(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
    a.iter().zip(b).map(|(x, y)| *x - *y).collect()
}

fn block_of(m: &DenseMat) -> Vec<Complex> {
    m.raw().to_vec()
}

/// Fill with constant blocks .
fn constant_inputs(
    g: &mut PseudoGreen,
    m: &[Complex],
    r: &[Complex],
    x: &[Complex],
    l: &[Complex],
) {
    for mi in 0..=NTAU {
        g.mat.block_mut(mi as usize).copy_from_slice(m);
    }
    for i in 0..=NT {
        for j in 0..=i {
            g.ret.raw_block_at_mut(i, j).copy_from_slice(r);
        }
        for mi in 0..=NTAU {
            g.lmix.raw_block_at_mut(i, mi).copy_from_slice(x);
        }
        for j in i..=NT {
            g.less.raw_block_at_mut(i, j).copy_from_slice(l);
        }
    }
}

fn make_contour(d: i64) -> Contour {
    Contour::new(NT, NTAU, d, d, NT as f64 * H, BETA)
}

fn matrix_of(blk: &[Complex], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, blk.to_vec())
}

/// Anti-hermitian random block .
fn random_antihermitian(d: usize, state: &mut u64) -> Vec<Complex> {
    let x = random_matrix(d, state);
    (0..d * d)
        .map(|q| {
            let i = q / d;
            let j = q % d;
            x[i * d + j] - x[j * d + i].conj()
        })
        .collect()
}

/// Random data in the stored triangles .
fn random_inputs(g: &mut PseudoGreen, state: &mut u64) {
    let d = g.size1;
    for mi in 0..=NTAU {
        let v = random_matrix(d, state);
        g.mat.block_mut(mi as usize).copy_from_slice(&v);
    }
    for i in 0..=NT {
        for j in 0..=i {
            let v = random_matrix(d, state);
            g.ret.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
        for mi in 0..=NTAU {
            let v = random_matrix(d, state);
            g.lmix.raw_block_at_mut(i, mi).copy_from_slice(&v);
        }
        for j in i..=NT {
            let v = random_matrix(d, state);
            g.less.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
    }
}

fn random_matrix(d: usize, state: &mut u64) -> Vec<Complex> {
    (0..d * d).map(|_| rand_c64_scaled(state, 4.0)).collect()
}

fn tau_of(m: i64) -> f64 {
    m as f64 * (BETA / NTAU as f64)
}

fn time_of(n: i64) -> f64 {
    n as f64 * H
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn pseudo_conv_constant_kernels() {
    for k in 1..=5i64 {
        for d in [1usize, 2] {
            let integ = Integrator::new(k);
            let cc = make_contour(d as i64);
            let mut state = (k * 10 + d as i64) as u64;
            let ma = random_matrix(d, &mut state);
            let mb = random_matrix(d, &mut state);
            let ra = random_antihermitian(d, &mut state);
            let rb = random_antihermitian(d, &mut state);
            let xa = random_matrix(d, &mut state);
            let xb = random_matrix(d, &mut state);
            let la = random_antihermitian(d, &mut state);
            let lb = random_antihermitian(d, &mut state);
            let f0 = random_matrix(d, &mut state);
            let f1 = random_matrix(d, &mut state);
            let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
            constant_inputs(&mut a, &ma, &ra, &xa, &la);
            let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
            constant_inputs(&mut b, &mb, &rb, &xb, &lb);
            let mut f = TimeFunction::new(NT as usize, d, d, Complex::ZERO);
            f.block_mut(MATSUBARA).copy_from_slice(&f0);
            for i in 0..=NT {
                f.block_mut(i).copy_from_slice(&f1);
            }
            let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
            pseudo_convolution(&mut x, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
            // restricted Matsubara integral: tau_of * Ma * F0 * Mb
            let mfm = matrix_of(&ma, d)
                .mat_mul(&matrix_of(&f0, d))
                .mat_mul(&matrix_of(&mb, d));
            let xfm = matrix_of(&xa, d)
                .mat_mul(&matrix_of(&f0, d))
                .mat_mul(&matrix_of(&mb, d));
            let rfx = matrix_of(&ra, d)
                .mat_mul(&matrix_of(&f1, d))
                .mat_mul(&matrix_of(&xb, d));
            let scale = 1.0
                + BETA * block_norm(&block_of(&mfm))
                + BETA * block_norm(&block_of(&xfm))
                + time_of(NT) * block_norm(&block_of(&rfx));
            for mi in 0..=NTAU {
                let want: Vec<Complex> = block_of(&mfm).iter().map(|z| *z * tau_of(mi)).collect();
                assert!(
                    block_norm(&block_difference(x.mat.block(mi as usize), &want)) < 1e-12 * scale,
                    "k={k} d={d} m={mi}"
                );
            }
            // left mixing: t * Ra * F1 * Xb + (beta - tau_of) * Xa * F0 * Mb
            let mut err = 0.0f64;
            for n in 0..=NT {
                for mi in 0..=NTAU {
                    let want: Vec<Complex> = block_of(&rfx)
                        .iter()
                        .zip(block_of(&xfm))
                        .map(|(p, q)| *p * time_of(n) + q * (BETA - tau_of(mi)))
                        .collect();
                    err = err.max(block_norm(&block_difference(
                        x.lmix.raw_block_at(n, mi),
                        &want,
                    )));
                }
            }
            assert!(err < 1e-12 * scale, "k={k} d={d} lmix err {err}");
            // the end point tau_of = beta of the imaginary time integral vanishes
            assert!(block_norm(x.lmix.raw_block_at(0, NTAU)) < 1e-14);
            if d == 2 {
                let other = matrix_of(&mb, d)
                    .mat_mul(&matrix_of(&f0, d))
                    .mat_mul(&matrix_of(&ma, d));
                assert!(block_norm(&block_difference(&block_of(&mfm), &block_of(&other))) > 1e-3);
            }
        }
    }
}

#[test]
fn pseudo_conv_linear_kernels() {
    for k in 2..=5i64 {
        let integ = Integrator::new(k);
        let cc = make_contour(1);
        let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        for mi in 0..=NTAU {
            a.mat.block_mut(mi as usize)[0] = c(1.0, 0.0);
            b.mat.block_mut(mi as usize)[0] = c(tau_of(mi), 0.0);
        }
        for n in 0..=NT {
            for mi in 0..=NTAU {
                a.lmix.raw_block_at_mut(n, mi)[0] = c(tau_of(mi), 0.0);
            }
        }
        let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_convolution_mat(&mut x, &a, None, &b, &integ, BETA);
        // int_0^tau_of x dx = tau_of^2 / 2
        for mi in 0..=NTAU {
            let v = x.mat.block(mi as usize)[0];
            assert!(
                (v - c(tau_of(mi) * tau_of(mi) / 2.0, 0.0)).abs() < 1e-12,
                "k={k} m={mi}"
            );
        }
        for n in 0..=NT {
            pseudo_convolution_time_step(n, &mut x, &a, &a, None, &b, &b, &integ, BETA, H);
        }
        // int_tau^beta x (x - tau_of) dx = (beta - tau_of)^2 (2 beta + tau_of) / 6
        let mut err = 0.0f64;
        for n in 0..=NT {
            for mi in 0..=NTAU {
                let want =
                    (BETA - tau_of(mi)) * (BETA - tau_of(mi)) * (2.0 * BETA + tau_of(mi)) / 6.0;
                err = err.max((x.lmix.raw_block_at(n, mi)[0] - c(want, 0.0)).abs());
            }
        }
        assert!(err < 1e-12, "k={k} lmix err {err}");
    }
}

#[test]
fn pseudo_conv_greater_less_are_standard() {
    let integ = Integrator::new(4);
    let cc = make_contour(2);
    let mut state = 21u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut a, &mut state);
    let mut acc = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut acc, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut b, &mut state);
    let mut bcc = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut bcc, &mut state);
    let mut f = TimeFunction::new(NT as usize, 2, 2, Complex::ZERO);
    for i in MATSUBARA..=NT {
        let v = random_matrix(2, &mut state);
        f.block_mut(i).copy_from_slice(&v);
    }
    let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let mut y = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let (fa, fc, fb, fd) = (
        a.to_contour(),
        acc.to_contour(),
        b.to_contour(),
        bcc.to_contour(),
    );
    let mut err = 0.0f64;
    for n in 1..=NT {
        pseudo_incr_convolution(
            n,
            &mut x,
            &a,
            &acc,
            Some(&f),
            &b,
            &bcc,
            &integ,
            BETA,
            H,
            c(0.7, 0.0),
        );
        incr_convolution_ret(
            n,
            &mut y,
            &fa,
            &fc,
            Some(&f),
            &fb,
            &fd,
            &integ,
            H,
            c(0.7, 0.0),
        );
        incr_convolution_less(
            n,
            &mut y,
            &fa,
            &fc,
            Some(&f),
            &fb,
            &fd,
            &integ,
            BETA,
            H,
            c(0.7, 0.0),
        );
        for j in 0..=n {
            err = err
                .max(block_norm(&block_difference(
                    x.ret.raw_block_at(n, j),
                    y.ret.raw_block_at(n, j),
                )))
                .max(block_norm(&block_difference(
                    x.less.raw_block_at(j, n),
                    y.less.raw_block_at(j, n),
                )));
        }
    }
    assert_eq!(err, 0.0);
    // the left-mixing component differs from the standard one
    incr_convolution_lmix(
        3,
        &mut y,
        &fa,
        &fc,
        Some(&f),
        &fb,
        &fd,
        &integ,
        BETA,
        H,
        c(0.7, 0.0),
    );
    let mut dmax = 0.0f64;
    for mi in 0..=NTAU {
        dmax = dmax.max(block_norm(&block_difference(
            x.lmix.raw_block_at(3, mi),
            y.lmix.raw_block_at(3, mi),
        )));
    }
    assert!(dmax > 1e-3);
}

#[test]
fn pseudo_conv_masks() {
    let integ = Integrator::new(3);
    let cc = make_contour(2);
    let mut state = 31u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut a, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut b, &mut state);
    let mut x0 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut x0, &mut state);
    let mask: Vec<bool> = (0..=NTAU).map(|m| m % 2 == 0).collect();
    // Matsubara
    let mut x = x0.clone();
    let mut y = x0.clone();
    pseudo_incr_convolution_mat(
        &mut x,
        &a,
        None,
        &b,
        &integ,
        BETA,
        c(1.5, 0.0),
        Some(mask.as_slice()),
    );
    pseudo_incr_convolution_mat(&mut y, &a, None, &b, &integ, BETA, c(1.5, 0.0), None);
    for (mi, masked) in mask.iter().enumerate().take(NTAU as usize + 1) {
        let want = if *masked {
            y.mat.block(mi)
        } else {
            x0.mat.block(mi)
        };
        assert_eq!(x.mat.block(mi), want);
    }
    // left-mixing
    let mut x = x0.clone();
    let mut y = x0.clone();
    pseudo_incr_convolution_lmix(
        5,
        &mut x,
        &a,
        &a,
        None,
        &b,
        &b,
        &integ,
        BETA,
        H,
        c(1.5, 0.0),
        Some(mask.as_slice()),
    );
    pseudo_incr_convolution_lmix(
        5,
        &mut y,
        &a,
        &a,
        None,
        &b,
        &b,
        &integ,
        BETA,
        H,
        c(1.5, 0.0),
        None,
    );
    for (mi, masked) in mask.iter().enumerate().take(NTAU as usize + 1) {
        let want = if *masked {
            y.lmix.raw_block_at(5, mi as i64)
        } else {
            x0.lmix.raw_block_at(5, mi as i64)
        };
        assert_eq!(x.lmix.raw_block_at(5, mi as i64), want);
    }
    assert_eq!(x.distance_at(&x0, 4), 0.0);
    // a wrong-length mask is rejected
    let bad = [true, true, true];
    assert!(panics(|| {
        let mut xx = x0.clone();
        pseudo_incr_convolution_mat(
            &mut xx,
            &a,
            None,
            &b,
            &integ,
            BETA,
            c(1.0, 0.0),
            Some(&bad[..]),
        );
    }));
}

#[test]
fn pseudo_conv_overall_stepwise_and_slice() {
    let integ = Integrator::new(5);
    let cc = make_contour(2);
    let mut state = 41u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut a, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut b, &mut state);
    let mut f = TimeFunction::new(NT as usize, 2, 2, Complex::ZERO);
    for i in MATSUBARA..=NT {
        let v = random_matrix(2, &mut state);
        f.block_mut(i).copy_from_slice(&v);
    }
    let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    pseudo_convolution(&mut x, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
    let mut y = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    random_inputs(&mut y, &mut state);
    for n in MATSUBARA..=NT {
        // the old values are discarded, repeated calls give the same result
        pseudo_convolution_time_step(n, &mut y, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
        pseudo_convolution_time_step(n, &mut y, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
        assert_eq!(x.distance_at(&y, n), 0.0);
    }
    // a time slice target gives the same increment
    let zp = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    for n in 0..=NT {
        let mut g = pseudo_timestep(&zp, n);
        pseudo_incr_convolution_step(
            n,
            &mut g,
            &a,
            &a,
            Some(&f),
            &b,
            &b,
            &integ,
            BETA,
            H,
            Complex::ONE,
        );
        // compare against the full driver's step
        for j in 0..=n {
            for q in 0..4 {
                assert_eq!(g.ret.raw_block_at(n, j)[q], x.ret.raw_block_at(n, j)[q]);
                assert_eq!(g.less.raw_block_at(j, n)[q], x.less.raw_block_at(j, n)[q]);
            }
        }
        for mi in 0..=NTAU {
            for q in 0..4 {
                assert_eq!(g.lmix.raw_block_at(0, mi)[q], x.lmix.raw_block_at(n, mi)[q]);
            }
        }
    }
}

#[test]
fn pseudo_conv_time_diagonal_density() {
    for k in [1i64, 3, 5] {
        let integ = Integrator::new(k);
        let cc = make_contour(2);
        let mut state = (51 + k) as u64;
        let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        random_inputs(&mut a, &mut state);
        let mut acc = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        random_inputs(&mut acc, &mut state);
        let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        random_inputs(&mut b, &mut state);
        let mut bcc = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        random_inputs(&mut bcc, &mut state);
        let mut f = TimeFunction::new(NT as usize, 2, 2, Complex::ZERO);
        for i in MATSUBARA..=NT {
            let v = random_matrix(2, &mut state);
            f.block_mut(i).copy_from_slice(&v);
        }
        let mut x = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        pseudo_convolution(&mut x, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, H);
        let (fa, fc, fb, fd) = (
            a.to_contour(),
            acc.to_contour(),
            b.to_contour(),
            bcc.to_contour(),
        );
        let rho0 =
            convolution_density_matrix(MATSUBARA, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H);
        let mut err = block_norm(&block_difference(rho0.raw(), &{
            let v = x.mat.block(NTAU as usize);
            let mut out = vec![Complex::ZERO; 4];
            for q in 0..4 {
                out[q] = -v[q];
            }
            out
        }));
        for n in 0..=NT {
            let rho = convolution_density_matrix(n, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H);
            // the reference start-up correction for n <= 2k + 2
            let mut l = x.less.raw_block_at(n, n).to_vec();
            if n <= 2 * k + 2 {
                let delta: Vec<Complex> = (0..4)
                    .map(|q| {
                        let i = q / 2;
                        let j = q % 2;
                        -(acc.less.raw_block_at(n, n)[j * 2 + i].conj())
                            - a.less.raw_block_at(n, n)[q]
                    })
                    .collect();
                let dm = matrix_of(&delta, 2);
                let fm = matrix_of(f.block(n), 2);
                let bg = matrix_of(bcc.ret.raw_block_at(n, n), 2);
                let bg_adj = bg.adjoint();
                let mut corr = dm.mat_mul(&fm).mat_mul(&bg_adj);
                corr.scale(H * integ.giw.at(n as usize, n as usize));
                for (lq, cq) in l.iter_mut().zip(corr.raw().iter()) {
                    *lq -= *cq;
                }
            }
            let want: Vec<Complex> = l.iter().map(|z| Complex::new(0.0, -1.0) * *z).collect();
            err = err.max(block_norm(&block_difference(rho.raw(), &want)));
        }
        assert!(err < 1e-10, "k={k} err {err}");
        // the correction is not trivial for these (non-partner) inputs
        let d0: Vec<Complex> = (0..4)
            .map(|q| {
                let i = q / 2;
                let j = q % 2;
                acc.less.raw_block_at(0, 0)[j * 2 + i].conj() + a.less.raw_block_at(0, 0)[q]
            })
            .collect();
        assert!(block_norm(&d0) > 1e-3);
    }
}
