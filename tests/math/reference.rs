// Project : KadanoffBaym
// File    : tests/math/reference.rs
// Purpose : the reference-data comparisons of the math utilities
//

use kadanoffbaym::weights::Integrator;

use kadanoffbaym::algebra::frobenius_distance;
use kadanoffbaym::equilibrium::{from_ham, from_mat, from_mat_k};
use kadanoffbaym::math::{extrapolate_timestep, extrapolate_timestep_cf, interpolation};
use kadanoffbaym::storage::timestep_at as timestep_slice;

use super::util::prelude::*;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Reference `lagrange_weights`.
fn lagrange_weights(p: usize, h: f64, x: f64) -> Vec<f64> {
    let mut w = vec![1.0f64; p + 1];
    for (i, wi) in w.iter_mut().enumerate() {
        for j in 0..=p {
            if i != j {
                *wi *= (x - h * j as f64) / (i as f64 * h - h * j as f64);
            }
        }
    }
    w
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn utilities_reference() {
    let (nt, ntau, kt) = (20i64, 100i64, 5i64);
    let (beta, h) = (5.0f64, 0.05f64);
    let ntauk = ntau; // intervals here == ntau+1 points
    let tmax = nt as f64 * h;
    let (w1, w2, w3) = (-1.0f64, 0.8, 0.7);
    let (nu1, nu2, nu3) = (-2.0f64, 1.6, 1.4);
    let mut ham1 = DenseMat::zeros(2);
    ham1.set(0, 0, Complex::new(w1, 0.0));
    ham1.set(1, 1, Complex::new(w2, 0.0));
    ham1.set(0, 1, Complex::new(0.0, -w3));
    ham1.set(1, 0, Complex::new(0.0, w3));
    let mut ham2 = DenseMat::zeros(2);
    ham2.set(0, 0, Complex::new(nu1, 0.0));
    ham2.set(1, 1, Complex::new(nu2, 0.0));
    ham2.set(0, 1, Complex::new(0.0, -nu3));
    ham2.set(1, 0, Complex::new(0.0, nu3));

    // extrapolate timestep (Lagrange reference)
    {
        let c = Contour::new(nt, ntauk, 1, 1, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, Complex::new(1.0, 0.0));
        from_ham(&mut g1, 0.0, &h0, beta, h);
        let mut g2 = g1.clone();
        let mut g3 = g1.clone();
        let p = kt as usize;
        let wl = lagrange_weights(p, h, (p + 1) as f64 * h);
        let mut err = 0.0;
        for tstp in kt..nt {
            // reference n = tstp+1 -> row r = tstp here
            let r = tstp;
            g2.memcpy_from_at(r, &g1);
            g3.memcpy_from_at(r, &g1);
            extrapolate_timestep(r, &mut g2, &Integrator::new(kt));
            // tv
            for m in 0..=ntauk {
                let mut x = Complex::ZERO;
                for q in 0..=p {
                    x += g3.get_lmix(r - q as i64, m, 0, 0) * wl[p - q];
                }
                g3.set_lmix(r + 1, m, 0, 0, x);
            }
            // les
            for j in 0..=r {
                let mut x = Complex::ZERO;
                for q in 0..=p {
                    x += g3.get_less(j, r - q as i64, 0, 0) * wl[p - q];
                }
                g3.set_less(j, r + 1, 0, 0, x);
            }
            {
                let mut x = Complex::ZERO;
                for q in 0..=p {
                    x += g3.get_less(r - q as i64, r + 1, 0, 0) * wl[p - q];
                }
                g3.set_less(r + 1, r + 1, 0, 0, x);
            }
            // ret
            for j in 0..=r {
                let mut x = Complex::ZERO;
                for q in 0..=p {
                    x += g3.get_ret(r - q as i64, j, 0, 0) * wl[p - q];
                }
                g3.set_ret(r + 1, j, 0, 0, x);
            }
            err += g2.distance_less(&g3, r + 1);
        }
        assert!(err < 1e-6, "extrapolate timestep err {err}");
    }

    // extrapolate function
    let fv = |t: f64| -> [Complex; 4] {
        [
            Complex::new((w1 * t).cos(), 0.0),
            Complex::new(0.0, -0.2 * (w3 * t).sin()),
            Complex::new(0.0, 0.2 * (w3 * t).sin()),
            Complex::new((w2 * t).cos(), 0.0),
        ]
    };
    {
        let mut func1 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        let mut func2 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        let mut func3 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        for t in 0..=nt {
            for f in [&mut func1, &mut func2, &mut func3] {
                f.block_mut(t).copy_from_slice(&fv(t as f64 * h));
            }
        }
        let p = kt as usize;
        let wl = lagrange_weights(p, h, (p + 1) as f64 * h);
        let mut err = 0.0;
        for tstp in kt..nt {
            let r = tstp;
            extrapolate_timestep_cf(r, &mut func1, kt);
            let mut fref = [Complex::ZERO; 4];
            for q in 0..=p {
                for (x, s) in fref.iter_mut().zip(func3.block(r - q as i64)) {
                    *x += *s * wl[p - q];
                }
            }
            for (x, f) in func1.block(r + 1).iter().zip(fref.iter()) {
                err += (*x - *f).abs();
            }
            func1.block_mut(r + 1).copy_from_slice(func2.block(r + 1));
            func3.block_mut(r + 1).copy_from_slice(func2.block(r + 1));
        }
        assert!(err < 1e-6, "extrapolate function err {err}");
    }

    // interpolate function
    {
        let mut func1 = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        for t in 0..=nt {
            func1.block_mut(t).copy_from_slice(&fv(t as f64 * h));
        }
        let p = kt as usize;
        let mut err = 0.0;
        for tstp in kt..nt {
            let tinter = tstp as f64 + 0.5;
            let f = interpolation(tstp + 1, tinter, &func1, &Integrator::new(kt));
            let ts = tstp + 1; // 0-based window end
            let wl = lagrange_weights(p, h, h * tinter - h * (ts - p as i64) as f64);
            let mut fref = [Complex::ZERO; 4];
            for q in 0..=p {
                for (x, s) in fref.iter_mut().zip(func1.block(ts - q as i64)) {
                    *x += *s * wl[p - q];
                }
            }
            for (q, fq) in fref.iter().enumerate() {
                err += (f.get(q / 2, q % 2) - *fq).abs();
            }
        }
        assert!(err < 1e-6, "interpolate function err {err}");
    }

    // from_mat
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        g2.memcpy_from_at(MATSUBARA, &g1);
        from_mat(&mut g2);
        assert!(g1.distance_at(&g2, 0) < 1e-6);
    }

    // from_mat_k
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        g2.memcpy_from_at(MATSUBARA, &g1);
        from_mat_k(&mut g2, kt);
        let mut err = 0.0;
        for n in 0..=kt {
            for m in 0..=ntauk {
                err += frobenius_distance(g1.lmix_raw_block(0, m), g2.lmix_raw_block(n, m));
            }
            err += frobenius_distance(g1.less_raw_block(n, n), g2.less_raw_block(n, n));
            err += frobenius_distance(g1.ret_raw_block(n, n), g2.ret_raw_block(n, n));
        }
        assert!(err < 1e-6, "from_mat_k err {err}");
    }

    // distance_norm2
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        from_ham(&mut g2, 0.0, &ham2, beta, h);
        let dist_green = |tstp: i64, a: &ContourGreen, b: &ContourGreen| -> f64 {
            let mut d = 0.0;
            if tstp == MATSUBARA {
                for m in 0..=ntauk {
                    d += frobenius_distance(a.mat_raw_block(m), b.mat_raw_block(m));
                }
            } else {
                for m in 0..=ntauk {
                    d += frobenius_distance(a.lmix_raw_block(tstp, m), b.lmix_raw_block(tstp, m));
                }
                for j in 0..=tstp {
                    d += frobenius_distance(a.less_raw_block(j, tstp), b.less_raw_block(j, tstp));
                    d += frobenius_distance(a.ret_raw_block(tstp, j), b.ret_raw_block(tstp, j));
                }
            }
            d
        };
        let mut err = 0.0;
        for tstp in -1..=nt {
            err += (g1.distance_at(&g2, tstp) - dist_green(tstp, &g1, &g2)).abs();
        }
        assert!(err < 1e-6, "distance_norm2 err {err}");
    }

    // force_matsubara_hermitian (mixed matsubara)
    {
        let mut ham3 = DenseMat::zeros(2);
        ham3.set(0, 0, Complex::new(w2, 0.0));
        ham3.set(1, 1, Complex::new(w1, 0.0));
        ham3.set(0, 1, Complex::new(w3, 0.0));
        ham3.set(1, 0, Complex::new(w3, 0.0));
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        from_ham(&mut g2, 0.0, &ham3, beta, h);
        let s2 = timestep_slice(&g2, 0);
        g1.incr_by_step(MATSUBARA, &s2, Complex::new(0.0, 1.0));
        g1.mat.force_matsubara_hermitian();
        let mut err = 0.0;
        for m in 0..=ntauk {
            let blk = g1.mat_raw_block(m);
            for a in 0..2 {
                for b in 0..2 {
                    err += (blk[a * 2 + b] - blk[b * 2 + a].conj()).abs();
                }
            }
        }
        assert!(err < 1e-6);
    }
}

#[test]
fn function_reference() {
    let (nt, size1) = (50i64, 2usize);
    let (eps, eps2, h, kt) = (1e-7f64, 1e-3, 0.01, 5i64);
    let mut a = TimeFunction::new(nt as usize, size1, size1, Complex::ZERO);
    let s2 = 2.0f64.sqrt();
    let av = [
        Complex::new(s2, 0.0),
        Complex::new(0.0, s2),
        Complex::new(0.0, -s2),
        Complex::new(-s2, 0.0),
    ];
    let mut toterr = 0.0;
    for m in 0..nt {
        a.block_mut(m).copy_from_slice(&av);
        for (x, y) in av.iter().zip(a.block(m).iter()) {
            toterr += (*x - *y).abs();
        }
    }
    assert!(toterr < eps);

    let cv = |t: f64| -> [Complex; 4] {
        let cs = t.cos();
        [
            Complex::new(2.0 * cs, 0.0),
            Complex::new(0.5 * cs, 0.0),
            Complex::new(0.5 * cs, 0.0),
            Complex::new(3.0 * cs, 0.0),
        ]
    };
    a.block_mut(MATSUBARA).copy_from_slice(&cv(0.0));
    for t in 0..=nt {
        a.block_mut(t).copy_from_slice(&cv(t as f64 * h));
    }

    // extrapolation
    {
        let mut b = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        let mut toterr = 0.0;
        for tstp in 0..nt {
            let n1 = if tstp <= kt { tstp } else { kt };
            for m in -1..=tstp {
                b.block_mut(m).copy_from_slice(a.block(m));
            }
            extrapolate_timestep_cf(tstp, &mut b, n1);
            for q in 0..4 {
                toterr += (a.block(tstp + 1)[q] - b.block(tstp + 1)[q]).abs();
            }
        }
        assert!(toterr < 1e-3, "extrapolation toterr {toterr}");
    }

    // interpolation
    {
        let mut b = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        let mut toterr0 = 0.0;
        let mut toterr1 = 0.0;
        for tstp in 1..nt {
            let n1 = if tstp <= kt { tstp } else { kt };
            for m in -1..=tstp {
                b.block_mut(m).copy_from_slice(a.block(m));
            }
            let b0 = interpolation(tstp, tstp as f64, &b, &Integrator::new(n1));
            for q in 0..4 {
                toterr0 += (a.block(tstp)[q] - b0.get(q / 2, q % 2)).abs();
            }
            let b1 = interpolation(tstp, tstp as f64 - 0.5, &b, &Integrator::new(n1));
            let a1 = cv((tstp as f64 - 0.5) * h);
            for (q, aq) in a1.iter().enumerate() {
                toterr1 += (*aq - b1.get(q / 2, q % 2)).abs();
            }
        }
        assert!(toterr0 < eps, "interp0 {toterr0}");
        assert!(toterr1 < eps2, "interp1 {toterr1}");
    }
}
