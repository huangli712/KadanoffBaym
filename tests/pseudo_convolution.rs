mod common;

use common::*;
use kadanoffbaym::c64::C64;
use kadanoffbaym::convolution::density_matrix::convolution_density_matrix;
use kadanoffbaym::convolution::less::incr_convolution_les;
use kadanoffbaym::convolution::pseudo::{
    pseudo_convolution, pseudo_convolution_mat, pseudo_convolution_timestep,
    pseudo_incr_convolution, pseudo_incr_convolution_lmix, pseudo_incr_convolution_mat,
    pseudo_incr_convolution_step,
};
use kadanoffbaym::convolution::ret::incr_convolution_ret;
use kadanoffbaym::mat::DenseMat;
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::TimeFunction;
use kadanoffbaym::types::layout::MATSUBARA;
use kadanoffbaym::types::pseudo::{
    pseudo_distance_at, pseudo_timestep, PseudoGreen,
};
use kadanoffbaym::weights::Integrator;

const NT: i64 = 11; // 12 points
const NTAU: i64 = 20; // 21 points
const BETA: f64 = 2.0;
const H: f64 = 0.05;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_c64(state: &mut u64) -> C64 {
    let a = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let b = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    C64::new(a * 4.0, b * 4.0)
}

fn contour(d: i64) -> Contour {
    Contour::new(NT, NTAU, d, d, NT as f64 * H, BETA)
}

fn tau(m: i64) -> f64 {
    m as f64 * (BETA / NTAU as f64)
}

fn t_of(n: i64) -> f64 {
    n as f64 * H
}

fn norm_blk(blk: &[C64]) -> f64 {
    blk.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt()
}

fn blk_sub(a: &[C64], b: &[C64]) -> Vec<C64> {
    a.iter().zip(b).map(|(x, y)| *x - *y).collect()
}

fn mat_of(blk: &[C64], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, blk.to_vec())
}

fn blk_of(m: &DenseMat) -> Vec<C64> {
    m.raw().to_vec()
}

fn rand_mat(d: usize, state: &mut u64) -> Vec<C64> {
    (0..d * d).map(|_| rand_c64(state)).collect()
}

/// Anti-hermitian random block (blueprint `pc_ah`).
fn rand_ah(d: usize, state: &mut u64) -> Vec<C64> {
    let x = rand_mat(d, state);
    (0..d * d)
        .map(|q| {
            let i = q / d;
            let j = q % d;
            x[i * d + j] - x[j * d + i].conj()
        })
        .collect()
}

/// Fill with constant blocks (blueprint `pc_const!`).
fn pc_const(g: &mut PseudoGreen, m: &[C64], r: &[C64], x: &[C64], l: &[C64]) {
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

/// Random data in the stored triangles (blueprint `pc_rand!`).
fn pc_rand(g: &mut PseudoGreen, state: &mut u64) {
    let d = g.size1;
    for mi in 0..=NTAU {
        let v = rand_mat(d, state);
        g.mat.block_mut(mi as usize).copy_from_slice(&v);
    }
    for i in 0..=NT {
        for j in 0..=i {
            let v = rand_mat(d, state);
            g.ret.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
        for mi in 0..=NTAU {
            let v = rand_mat(d, state);
            g.lmix.raw_block_at_mut(i, mi).copy_from_slice(&v);
        }
        for j in i..=NT {
            let v = rand_mat(d, state);
            g.less.raw_block_at_mut(i, j).copy_from_slice(&v);
        }
    }
}

#[test]
fn pseudo_conv_constant_kernels() {
    for k in 1..=5i64 {
        for d in [1usize, 2] {
            let integ = Integrator::new(k);
            let cc = contour(d as i64);
            let mut state = (k * 10 + d as i64) as u64;
            let ma = rand_mat(d, &mut state);
            let mb = rand_mat(d, &mut state);
            let ra = rand_ah(d, &mut state);
            let rb = rand_ah(d, &mut state);
            let xa = rand_mat(d, &mut state);
            let xb = rand_mat(d, &mut state);
            let la = rand_ah(d, &mut state);
            let lb = rand_ah(d, &mut state);
            let f0 = rand_mat(d, &mut state);
            let f1 = rand_mat(d, &mut state);
            let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
            pc_const(&mut a, &ma, &ra, &xa, &la);
            let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
            pc_const(&mut b, &mb, &rb, &xb, &lb);
            let mut f = TimeFunction::new(NT as usize, d, d, C64::ZERO);
            f.block_mut(MATSUBARA).copy_from_slice(&f0);
            for i in 0..=NT {
                f.block_mut(i).copy_from_slice(&f1);
            }
            let mut x = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
            pseudo_convolution(&mut x, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
            // restricted Matsubara integral: tau * Ma * F0 * Mb
            let mfm = mat_of(&ma, d).mat_mul(&mat_of(&f0, d)).mat_mul(&mat_of(&mb, d));
            let xfm = mat_of(&xa, d).mat_mul(&mat_of(&f0, d)).mat_mul(&mat_of(&mb, d));
            let rfx = mat_of(&ra, d).mat_mul(&mat_of(&f1, d)).mat_mul(&mat_of(&xb, d));
            let scale = 1.0
                + BETA * norm_blk(&blk_of(&mfm))
                + BETA * norm_blk(&blk_of(&xfm))
                + t_of(NT) * norm_blk(&blk_of(&rfx));
            for mi in 0..=NTAU {
                let want: Vec<C64> = blk_of(&mfm).iter().map(|z| *z * tau(mi)).collect();
                assert!(
                    norm_blk(&blk_sub(x.mat.block(mi as usize), &want)) < 1e-12 * scale,
                    "k={k} d={d} m={mi}"
                );
            }
            // left mixing: t * Ra * F1 * Xb + (beta - tau) * Xa * F0 * Mb
            let mut err = 0.0f64;
            for n in 0..=NT {
                for mi in 0..=NTAU {
                    let want: Vec<C64> = blk_of(&rfx)
                        .iter()
                        .zip(blk_of(&xfm))
                        .map(|(p, q)| *p * t_of(n) + q * (BETA - tau(mi)))
                        .collect();
                    err = err.max(norm_blk(&blk_sub(x.lmix.raw_block_at(n, mi), &want)));
                }
            }
            assert!(err < 1e-12 * scale, "k={k} d={d} lmix err {err}");
            // the end point tau = beta of the imaginary time integral vanishes
            assert!(norm_blk(x.lmix.raw_block_at(0, NTAU)) < 1e-14);
            if d == 2 {
                let other = mat_of(&mb, d).mat_mul(&mat_of(&f0, d)).mat_mul(&mat_of(&ma, d));
                assert!(norm_blk(&blk_sub(&blk_of(&mfm), &blk_of(&other))) > 1e-3);
            }
        }
    }
}

#[test]
fn pseudo_conv_linear_kernels() {
    for k in 2..=5i64 {
        let integ = Integrator::new(k);
        let cc = contour(1);
        let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        for mi in 0..=NTAU {
            a.mat.block_mut(mi as usize)[0] = c(1.0, 0.0);
            b.mat.block_mut(mi as usize)[0] = c(tau(mi), 0.0);
        }
        for n in 0..=NT {
            for mi in 0..=NTAU {
                a.lmix.raw_block_at_mut(n, mi)[0] = c(tau(mi), 0.0);
            }
        }
        let mut x = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pseudo_convolution_mat(&mut x, &a, None, &b, &integ, BETA);
        // int_0^tau x dx = tau^2 / 2
        for mi in 0..=NTAU {
            let v = x.mat.block(mi as usize)[0];
            assert!((v - c(tau(mi) * tau(mi) / 2.0, 0.0)).abs() < 1e-12, "k={k} m={mi}");
        }
        for n in 0..=NT {
            pseudo_convolution_timestep(n, &mut x, &a, &a, None, &b, &b, &integ, BETA, H);
        }
        // int_tau^beta x (x - tau) dx = (beta - tau)^2 (2 beta + tau) / 6
        let mut err = 0.0f64;
        for n in 0..=NT {
            for mi in 0..=NTAU {
                let want = (BETA - tau(mi)) * (BETA - tau(mi)) * (2.0 * BETA + tau(mi)) / 6.0;
                err = err.max((x.lmix.raw_block_at(n, mi)[0] - c(want, 0.0)).abs());
            }
        }
        assert!(err < 1e-12, "k={k} lmix err {err}");
    }
}

#[test]
fn pseudo_conv_greater_less_are_standard() {
    let integ = Integrator::new(4);
    let cc = contour(2);
    let mut state = 21u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut a, &mut state);
    let mut acc = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut acc, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut b, &mut state);
    let mut bcc = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut bcc, &mut state);
    let mut f = TimeFunction::new(NT as usize, 2, 2, C64::ZERO);
    for i in MATSUBARA..=NT {
        let v = rand_mat(2, &mut state);
        f.block_mut(i).copy_from_slice(&v);
    }
    let mut x = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    let mut y = kadanoffbaym::types::green::ContourGreen::new(&cc, Sign::Fermi, C64::ZERO);
    let (fa, fc, fb, fd) = (a.to_contour(), acc.to_contour(), b.to_contour(), bcc.to_contour());
    let mut err = 0.0f64;
    for n in 1..=NT {
        pseudo_incr_convolution(
            n, &mut x, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, H, c(0.7, 0.0),
        );
        incr_convolution_ret(n, &mut y, &fa, &fc, Some(&f), &fb, &fd, &integ, H, c(0.7, 0.0));
        incr_convolution_les(n, &mut y, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H, c(0.7, 0.0));
        for j in 0..=n {
            err = err
                .max(norm_blk(&blk_sub(x.ret.raw_block_at(n, j), y.ret.raw_block_at(n, j))))
                .max(norm_blk(&blk_sub(
                    x.less.raw_block_at(j, n),
                    y.less.raw_block_at(j, n),
                )));
        }
    }
    assert_eq!(err, 0.0);
    // the left-mixing component differs from the standard one
    kadanoffbaym::convolution::lmix::incr_convolution_lmix(
        3, &mut y, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H, c(0.7, 0.0),
    );
    let mut dmax = 0.0f64;
    for mi in 0..=NTAU {
        dmax = dmax.max(norm_blk(&blk_sub(
            x.lmix.raw_block_at(3, mi),
            y.lmix.raw_block_at(3, mi),
        )));
    }
    assert!(dmax > 1e-3);
}

#[test]
fn pseudo_conv_masks() {
    let integ = Integrator::new(3);
    let cc = contour(2);
    let mut state = 31u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut a, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut b, &mut state);
    let mut x0 = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut x0, &mut state);
    let mask: Vec<bool> = (0..=NTAU).map(|m| m % 2 == 0).collect();
    // Matsubara
    let mut x = x0.clone();
    let mut y = x0.clone();
    pseudo_incr_convolution_mat(&mut x, &a, None, &b, &integ, BETA, c(1.5, 0.0), Some(mask.as_slice()));
    pseudo_incr_convolution_mat(&mut y, &a, None, &b, &integ, BETA, c(1.5, 0.0), None);
    for mi in 0..=NTAU as usize {
        let want = if mask[mi] {
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
        5, &mut x, &a, &a, None, &b, &b, &integ, BETA, H, c(1.5, 0.0), Some(mask.as_slice()),
    );
    pseudo_incr_convolution_lmix(
        5, &mut y, &a, &a, None, &b, &b, &integ, BETA, H, c(1.5, 0.0), None,
    );
    for mi in 0..=NTAU as usize {
        let want = if mask[mi] {
            y.lmix.raw_block_at(5, mi as i64)
        } else {
            x0.lmix.raw_block_at(5, mi as i64)
        };
        assert_eq!(x.lmix.raw_block_at(5, mi as i64), want);
    }
    assert_eq!(pseudo_distance_at(&x, &x0, 4), 0.0);
    // a wrong-length mask is rejected
    let bad = [true, true, true];
    assert!(panics(|| {
        let mut xx = x0.clone();
        pseudo_incr_convolution_mat(&mut xx, &a, None, &b, &integ, BETA, c(1.0, 0.0), Some(&bad[..]));
    }));
}

#[test]
fn pseudo_conv_overall_stepwise_and_slice() {
    let integ = Integrator::new(5);
    let cc = contour(2);
    let mut state = 41u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut a, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut b, &mut state);
    let mut f = TimeFunction::new(NT as usize, 2, 2, C64::ZERO);
    for i in MATSUBARA..=NT {
        let v = rand_mat(2, &mut state);
        f.block_mut(i).copy_from_slice(&v);
    }
    let mut x = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pseudo_convolution(&mut x, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
    let mut y = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    pc_rand(&mut y, &mut state);
    for n in MATSUBARA..=NT {
        // the old values are discarded, repeated calls give the same result
        pseudo_convolution_timestep(n, &mut y, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
        pseudo_convolution_timestep(n, &mut y, &a, &a, Some(&f), &b, &b, &integ, BETA, H);
        assert_eq!(pseudo_distance_at(&x, &y, n), 0.0);
    }
    // a time slice target gives the same increment
    let zp = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
    for n in 0..=NT {
        let mut g = pseudo_timestep(&zp, n);
        pseudo_incr_convolution_step(
            n, &mut g, &a, &a, Some(&f), &b, &b, &integ, BETA, H, C64::ONE,
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
        let cc = contour(2);
        let mut state = (51 + k) as u64;
        let mut a = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pc_rand(&mut a, &mut state);
        let mut acc = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pc_rand(&mut acc, &mut state);
        let mut b = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pc_rand(&mut b, &mut state);
        let mut bcc = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pc_rand(&mut bcc, &mut state);
        let mut f = TimeFunction::new(NT as usize, 2, 2, C64::ZERO);
        for i in MATSUBARA..=NT {
            let v = rand_mat(2, &mut state);
            f.block_mut(i).copy_from_slice(&v);
        }
        let mut x = PseudoGreen::new(&cc, Sign::Fermi, C64::ZERO);
        pseudo_convolution(&mut x, &a, &acc, Some(&f), &b, &bcc, &integ, BETA, H);
        let (fa, fc, fb, fd) = (a.to_contour(), acc.to_contour(), b.to_contour(), bcc.to_contour());
        let rho0 = convolution_density_matrix(
            MATSUBARA, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H,
        );
        let mut err = norm_blk(&blk_sub(
            rho0.raw(),
            &{
                let v = x.mat.block(NTAU as usize);
                let mut out = vec![C64::ZERO; 4];
                for q in 0..4 {
                    out[q] = -v[q];
                }
                out
            },
        ));
        for n in 0..=NT {
            let rho = convolution_density_matrix(n, &fa, &fc, Some(&f), &fb, &fd, &integ, BETA, H);
            // the NESSi start-up correction for n <= 2k + 2
            let mut l = x.less.raw_block_at(n, n).to_vec();
            if n <= 2 * k + 2 {
                let delta: Vec<C64> = (0..4)
                    .map(|q| {
                        let i = q / 2;
                        let j = q % 2;
                        -(acc.less.raw_block_at(n, n)[j * 2 + i].conj())
                            - a.less.raw_block_at(n, n)[q]
                    })
                    .collect();
                let dm = mat_of(&delta, 2);
                let fm = mat_of(f.block(n), 2);
                let bg = mat_of(bcc.ret.raw_block_at(n, n), 2);
                let bg_adj = bg.adjoint();
                let mut corr = dm.mat_mul(&fm).mat_mul(&bg_adj);
                corr.scale(H * integ.giw.at(n as usize, n as usize));
                for q in 0..4 {
                    l[q] -= corr.raw()[q];
                }
            }
            let want: Vec<C64> = l.iter().map(|z| C64::new(0.0, -1.0) * *z).collect();
            err = err.max(norm_blk(&blk_sub(rho.raw(), &want)));
        }
        assert!(err < 1e-10, "k={k} err {err}");
        // the correction is not trivial for these (non-partner) inputs
        let d0: Vec<C64> = (0..4)
            .map(|q| {
                let i = q / 2;
                let j = q % 2;
                acc.less.raw_block_at(0, 0)[j * 2 + i].conj() + a.less.raw_block_at(0, 0)[q]
            })
            .collect();
        assert!(norm_blk(&d0) > 1e-3);
    }
}
