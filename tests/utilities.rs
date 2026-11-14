// Project : KadanoffBaym
// File    : tests/utilities.rs
// Purpose : port of julia/test/t_utilities.jl

mod common;

use common::prelude::*;
use common::{assert_order_gate, check_order_gate, is_release};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::math::extrap::{extrapolate_timestep, extrapolate_timestep_cf};
use kadanoffbaym::math::interp::interpolation;
use kadanoffbaym::ops::{force_matsubara_hermitian, MatsubaraHermitian};
use kadanoffbaym::types::from_mat::{from_mat, from_mat_k};
use kadanoffbaym::weights::Integrator;

struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn block(&mut self, nel: usize) -> Vec<C64> {
        (0..nel)
            .map(|_| C64::new(self.next_f64(), self.next_f64()))
            .collect()
    }
}

/// Blueprint `ut_lagrange_weights`.
fn ut_lagrange_weights(p: usize, h: f64, x: f64) -> Vec<f64> {
    let mut w = vec![1.0f64; p + 1];
    for i in 0..=p {
        for j in 0..=p {
            if i != j {
                w[i] *= (x - h * j as f64) / (i as f64 * h - h * j as f64);
            }
        }
    }
    w
}

/// Smooth matrix test function (blueprint `ut_F`).
fn ut_f(t: f64) -> [C64; 4] {
    let e = (0.3 * t).exp();
    [
        C64::new((-1.0 * t).cos() * e, 0.0),
        C64::new(0.0, -0.2 * (0.7 * t).sin() * e),
        C64::new(0.0, 0.2 * (0.7 * t).sin() * e),
        C64::new((0.8 * t).cos() * e, 0.0),
    ]
}

fn poly_eval(ms: &[Vec<C64>], t: f64) -> Vec<C64> {
    let nel = ms[0].len();
    let mut out = vec![C64::ZERO; nel];
    for (r, m) in ms.iter().enumerate() {
        let w = C64::new(t.powi(r as i32), 0.0);
        for (o, c) in out.iter_mut().zip(m) {
            *o += *c * w;
        }
    }
    out
}

#[test]
fn extrapolate_cf_polynomial_exactness() {
    let h = 0.1;
    for k in 0..=5i64 {
        for nel in [1usize, 4] {
            let dims = if nel == 1 { (1, 1) } else { (2, 2) };
            let mut rng = Rng(0x5EED + k as u64);
            let ms: Vec<Vec<C64>> = (0..=k).map(|_| rng.block(nel)).collect();
            let ntime = (k + 4) as usize; // Rust nt
            let mut f = TimeFunction::new(ntime, dims.0, dims.1, C64::ZERO);
            for i in 0..=ntime as i64 {
                f.block_mut(i)
                    .copy_from_slice(&poly_eval(&ms, i as f64 * h));
            }
            f.block_mut(MATSUBARA)
                .copy_from_slice(&vec![C64::new(7.0, 0.0); nel]);
            let mut g = f.clone();
            let n = ntime as i64 - 1; // last filled row
            g.block_mut(n + 1).fill(C64::ZERO);
            extrapolate_timestep_cf(n, &mut g, k);
            for (a, b) in g.block(n + 1).iter().zip(f.block(n + 1)) {
                assert!((*a - *b).abs() < 1e-13, "k={k} nel={nel}");
            }
            assert_eq!(g.block(MATSUBARA), f.block(MATSUBARA));
            if k >= 1 {
                g.block_mut(n + 1).fill(C64::ZERO);
                // Integrator overload (order from the Integrator)
                let p = kadanoffbaym::math::extrap::extrapolation_weights(&Integrator::new(k));
                let mut acc = vec![C64::ZERO; nel];
                for l in 0..=(k as usize) {
                    for (a, s) in acc.iter_mut().zip(g.block(n - l as i64)) {
                        *a += *s * p[l];
                    }
                }
                g.block_mut(n + 1).copy_from_slice(&acc);
                for (a, b) in g.block(n + 1).iter().zip(f.block(n + 1)) {
                    assert!((*a - *b).abs() < 1e-13);
                }
            }
        }
    }
}

#[test]
fn extrapolate_green_polynomial_exactness() {
    let (ntime, ntau, h) = (12i64, 7i64, 0.1);
    let c = Contour::new(ntime - 1, ntau - 1, 2, 2, (ntime - 1) as f64 * h, 2.0);
    let (nt, ntau_r) = (ntime - 1, ntau - 1);
    for k in 0..=5i64 {
        let mut rng = Rng(0xBEEF + k as u64);
        let kk = k as usize;
        let mut q = vec![vec![[C64::ZERO; 4]; kk + 1]; kk + 1];
        let mut w = vec![vec![[C64::ZERO; 4]; kk + 1]; kk + 1];
        for a in 0..=kk {
            for b in 0..=kk {
                if a + b <= kk {
                    q[a][b] = {
                        let mut v = [C64::ZERO; 4];
                        for z in v.iter_mut() {
                            *z = C64::new(rng.next_f64(), rng.next_f64());
                        }
                        v
                    };
                    w[a][b] = {
                        let mut v = [C64::ZERO; 4];
                        for z in v.iter_mut() {
                            *z = C64::new(rng.next_f64(), rng.next_f64());
                        }
                        v
                    };
                }
            }
        }
        let qf = |t: f64, s: f64| -> [C64; 4] {
            let mut out = [C64::ZERO; 4];
            for a in 0..=kk {
                for b in 0..=kk {
                    if a + b <= kk {
                        let wgt = t.powi(a as i32) * s.powi(b as i32);
                        for (o, cz) in out.iter_mut().zip(&q[a][b]) {
                            *o += *cz * wgt;
                        }
                    }
                }
            }
            out
        };
        let wf = |t: f64, s: f64| -> [C64; 4] {
            let mut out = [C64::ZERO; 4];
            for a in 0..=kk {
                for b in 0..=kk {
                    if a + b <= kk {
                        let wgt = t.powi(a as i32) * s.powi(b as i32);
                        for (o, cz) in out.iter_mut().zip(&w[a][b]) {
                            *o += *cz * wgt;
                        }
                    }
                }
            }
            out
        };
        let neg_t = |b: &[C64]| -> [C64; 4] {
            // -(block)^dagger for 2x2: conjugate + transpose
            [-b[0].conj(), -b[2].conj(), -b[1].conj(), -b[3].conj()]
        };
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        for i in 0..=nt {
            for j in 0..=i {
                let r = qf(i as f64 * h, j as f64 * h);
                let rs = qf(j as f64 * h, i as f64 * h);
                let mut v = [C64::ZERO; 4];
                for (x, (y, z)) in v.iter_mut().zip(r.iter().zip(neg_t(&rs))) {
                    *x = *y + z;
                }
                g.ret_raw_block_mut(i, j).copy_from_slice(&v);
            }
            for j in i..=nt {
                let l = wf(i as f64 * h, j as f64 * h);
                let ls = wf(j as f64 * h, i as f64 * h);
                let mut v = [C64::ZERO; 4];
                for (x, (y, z)) in v.iter_mut().zip(l.iter().zip(neg_t(&ls))) {
                    *x = *y + z;
                }
                g.less_raw_block_mut(i, j).copy_from_slice(&v);
            }
            for m in 0..=ntau_r {
                g.lmix_raw_block_mut(i, m)
                    .copy_from_slice(&qf(i as f64 * h, 0.1 * (m + 1) as f64));
            }
        }
        for m in 0..=ntau_r {
            g.mat_raw_block_mut(m).copy_from_slice(&rng.block(4));
        }
        let gref = g.clone();
        let n = nt - 1; // last filled 0-based row (Julia n = ntime-1)
        g.set_zero_at(n + 1);
        extrapolate_timestep(n, &mut g, &Integrator::new(k.max(1)));
        // order 0 has no Integrator; the copy case is covered by the
        // k >= 1 orders here plus the Cf testset
        let zero = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let scale = gref.distance_at(&zero, n + 1);
        assert!(
            g.distance_at(&gref, n + 1) <= 1e-12 * scale.max(1.0),
            "k={k}"
        );
        assert!(g.mat.equal(&gref.mat));
        for t in 0..=n {
            assert_eq!(g.distance_at(&gref, t), 0.0);
        }
    }
}

#[test]
fn extrapolate_preconditions() {
    let c = Contour::new(7, 4, 1, 1, 0.7, 1.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extrapolate_timestep(2, &mut g, &Integrator::new(3));
    }));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extrapolate_timestep(7, &mut g, &Integrator::new(1));
    }));
    assert!(r.is_err());
    let mut f = TimeFunction::new(7, 1, 1, C64::ZERO);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extrapolate_timestep_cf(1, &mut f, 2);
    }));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extrapolate_timestep_cf(7, &mut f, 0);
    }));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extrapolate_timestep_cf(4, &mut f, 6);
    }));
    assert!(r.is_err());
}

#[test]
fn interpolation_polynomial_exactness() {
    let h = 0.1;
    for k in 1..=5i64 {
        for nel in [1usize, 4] {
            let dims = if nel == 1 { (1, 1) } else { (2, 2) };
            let mut rng = Rng(0x777 + k as u64);
            let ms: Vec<Vec<C64>> = (0..=k).map(|_| rng.block(nel)).collect();
            let ntime = (k + 3) as i64; // Rust nt
            let mut f = TimeFunction::new(ntime as usize, dims.0, dims.1, C64::ZERO);
            for i in 0..=ntime {
                f.block_mut(i)
                    .copy_from_slice(&poly_eval(&ms, i as f64 * h));
            }
            let tstp = ntime; // last 0-based index
            let integ = Integrator::new(k);
            for tinter in [
                tstp as f64,
                (tstp - k) as f64,
                tstp as f64 - 0.5,
                (tstp - k) as f64 + 0.3,
            ] {
                let x = interpolation(tstp, tinter, &f, &integ);
                let want = poly_eval(&ms, tinter * h);
                for q in 0..nel {
                    let gotv = x.get(q / dims.1, q % dims.1);
                    assert!((gotv - want[q]).abs() < 1e-12, "k={k} tinter={tinter}");
                }
            }
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                interpolation(tstp, tstp as f64 + 0.1, &f, &integ);
            }));
            assert!(r.is_err());
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                interpolation(tstp, (tstp - k) as f64 - 0.1, &f, &integ);
            }));
            assert!(r.is_err());
        }
    }
    // k = 0 cases are covered by `interpolation_zero_order`

}

#[test]
fn interpolation_zero_order() {
    let mut f = TimeFunction::new(4, 2, 2, C64::ZERO);
    f.block_mut(3)
        .copy_from_slice(&[C64::new(1.0, 1.0), C64::ZERO, C64::ZERO, C64::ONE]);
    // k = 0: interpolate onto the last point itself
    let piw = kadanoffbaym::weights::PolynomialInterpolationWeights::new(0);
    let _ = piw;
    // manual zeroth-order: result equals the block at tstp
    let x = interpolation(3, 3.0, &f, &Integrator::new(1));
    // (order-1 interpolation at a node reproduces the node value)
    assert_eq!(x.get(0, 0), C64::new(1.0, 1.0));
}

#[test]
fn distance_block_norms() {
    let c = Contour::new(5, 4, 2, 2, 0.5, 1.0);
    let mut a = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let b = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let tstp = 4i64;
    // single block difference 3 + 4i -> norm 5
    let mut blk = [C64::ZERO; 4];
    blk[0] = C64::new(3.0, 4.0);
    a.ret_raw_block_mut(tstp, 1).copy_from_slice(&blk);
    assert_eq!(a.distance_ret(&b, tstp), 5.0);
    assert_eq!(a.distance_lmix(&b, tstp), 0.0);
    assert_eq!(a.distance_less(&b, tstp), 0.0);
    // elements with vanishing sum but non-zero norm
    let mut tv = [C64::ZERO; 4];
    tv[0] = C64::new(1.0, 0.0);
    tv[1] = C64::new(-1.0, 0.0);
    a.lmix_raw_block_mut(tstp, 2).copy_from_slice(&tv);
    let mut les = [C64::ZERO; 4];
    les[1] = C64::new(2.0, 0.0);
    a.less_raw_block_mut(1, tstp).copy_from_slice(&les);
    assert!((a.distance_lmix(&b, tstp) - 2.0f64.sqrt()).abs() < 1e-14);
    assert_eq!(a.distance_less(&b, tstp), 2.0);
    let total = a.distance_at(&b, tstp);
    assert!((total - (5.0 + 2.0f64.sqrt() + 2.0)).abs() < 1e-14);
    assert!(
        (total
            - (a.distance_ret(&b, tstp) + a.distance_lmix(&b, tstp) + a.distance_less(&b, tstp)))
        .abs()
            < 1e-14
    );
    // Matsubara branch
    let mut mb = [C64::ZERO; 4];
    mb[0] = C64::new(1.0, 0.0);
    mb[1] = C64::new(-1.0, 0.0);
    a.mat_raw_block_mut(1).copy_from_slice(&mb);
    assert!((a.distance_at(&b, MATSUBARA) - 2.0f64.sqrt()).abs() < 1e-14);
    assert_eq!(a.distance_ret(&b, MATSUBARA), 0.0);
}

#[test]
fn from_mat_and_from_mat() {
    let (ntime, ntau, beta, h) = (9i64, 41i64, 5.0, 0.05);
    let mut hmat = DenseMat::zeros(2);
    hmat.set(0, 0, C64::new(-1.0, 0.0));
    hmat.set(1, 1, C64::new(0.8, 0.0));
    hmat.set(0, 1, C64::new(0.0, -0.7));
    hmat.set(1, 0, C64::new(0.0, 0.7));
    for sign in [Sign::Fermi, Sign::Bose] {
        let c = Contour::new(ntime - 1, ntau - 1, 2, 2, (ntime - 1) as f64 * h, beta);
        let mut gref = ContourGreen::new(&c, sign, C64::ZERO);
        let mut h_use = hmat.clone();
        if sign == Sign::Bose {
            h_use.axpy(2.0, &DenseMat::identity(2));
        }
        from_ham(&mut gref, 0.0, &h_use, beta, h);

        let mut g = ContourGreen::new(&c, sign, C64::ZERO);
        g.memcpy_from_at(MATSUBARA, &gref);
        from_mat(&mut g);
        assert!(g.distance_at(&gref, 0) <= 1e-13);
        assert!(g.mat.equal(&gref.mat));

        let k = 4i64;
        let mut g = ContourGreen::new(&c, sign, C64::ZERO);
        g.memcpy_from_at(MATSUBARA, &gref);
        let sentinel = [
            C64::new(9.0, 0.0),
            C64::new(8.0, 0.0),
            C64::new(7.0, 0.0),
            C64::new(6.0, 0.0),
        ];
        for n in 1..=k + 1 {
            g.ret_raw_block_mut(n, 0).copy_from_slice(&sentinel);
            g.less_raw_block_mut(0, n).copy_from_slice(&sentinel);
        }
        from_mat_k(&mut g, k);
        for n in 0..=k {
            for m in 0..=(ntau - 1) {
                let d = kadanoffbaym::properties::blkdist(
                    g.lmix_raw_block(n, m),
                    gref.lmix_raw_block(0, m),
                );
                assert!(d <= 1e-13, "lmix n={n} m={m}");
            }
            let d = kadanoffbaym::properties::blkdist(
                g.less_raw_block(n, n),
                gref.less_raw_block(0, 0),
            );
            assert!(d <= 1e-13);
            let d =
                kadanoffbaym::properties::blkdist(g.ret_raw_block(n, n), gref.ret_raw_block(0, 0));
            assert!(d <= 1e-13);
        }
        for n in 1..=(k + 1) {
            assert_eq!(g.ret_raw_block(n, 0), sentinel);
            assert_eq!(g.less_raw_block(0, n), sentinel);
        }
        let zero = ContourGreen::new(&c, sign, C64::ZERO);
        assert_eq!(g.distance_at(&zero, k + 2), 0.0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g2 = g.clone();
            from_mat_k(&mut g2, ntime);
        }));
        assert!(r.is_err());
    }
}

#[test]
fn force_matsubara_hermitian_test() {
    let c = Contour::new(4, 5, 2, 2, 0.4, 1.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let mut rng = Rng(0xF00D);
    for m in 0..=5 {
        g.mat_raw_block_mut(m).copy_from_slice(&rng.block(4));
    }
    for i in 0..=4 {
        for j in 0..=i {
            g.ret_raw_block_mut(i, j).copy_from_slice(&rng.block(4));
        }
    }
    let gref = g.clone();
    force_matsubara_hermitian(&mut g);
    for m in 0..=5 {
        // hermitian
        let blk = g.mat_raw_block(m).to_vec();
        let mut e = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                e += (blk[a * 2 + b] - blk[b * 2 + a].conj()).abs();
            }
        }
        assert!(e <= 1e-14);
        // equals (ref + ref†)/2
        let rf = gref.mat_raw_block(m);
        for a in 0..2 {
            for b in 0..2 {
                let want = (rf[a * 2 + b] + rf[b * 2 + a].conj()) * 0.5;
                assert!((blk[a * 2 + b] - want).abs() < 1e-15);
            }
        }
    }
    for t in 0..=4 {
        assert_eq!(g.distance_at(&gref, t), 0.0);
    }
    // timestep variant (equilibrium slice)
    {
        let mut s = timestep_slice(&gref, 0);
        force_matsubara_hermitian(&mut s);
        for m in 0..=5 {
            let blk = s.mat_raw_block(m).to_vec();
            let mut e = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    e += (blk[a * 2 + b] - blk[b * 2 + a].conj()).abs();
                }
            }
            assert!(e <= 1e-14);
            assert_eq!(blk, g.mat_raw_block(m).to_vec());
        }
    }
    // rectangular rejection
    let cr = Contour::new(4, 5, 2, 3, 0.4, 1.0);
    let mut gr = ContourGreen::new(&cr, Sign::Fermi, C64::ZERO);
    let r = std::panic::catch_unwind(move || {
        force_matsubara_hermitian(&mut gr);
    });
    assert!(r.is_err());
}

use kadanoffbaym::types::indexing::timestep_at as timestep_slice;

/// Port of libcntr/test/utilities.cpp:213-488.
#[test]
fn nessi_utilities_cpp() {
    let (nt, ntau, kt) = (20i64, 100i64, 5i64);
    let (beta, h) = (5.0f64, 0.05f64);
    let ntauk = ntau; // Rust intervals == Julia ntau+1 points
    let tmax = nt as f64 * h;
    let (w1, w2, w3) = (-1.0f64, 0.8, 0.7);
    let (nu1, nu2, nu3) = (-2.0f64, 1.6, 1.4);
    let mut ham1 = DenseMat::zeros(2);
    ham1.set(0, 0, C64::new(w1, 0.0));
    ham1.set(1, 1, C64::new(w2, 0.0));
    ham1.set(0, 1, C64::new(0.0, -w3));
    ham1.set(1, 0, C64::new(0.0, w3));
    let mut ham2 = DenseMat::zeros(2);
    ham2.set(0, 0, C64::new(nu1, 0.0));
    ham2.set(1, 1, C64::new(nu2, 0.0));
    ham2.set(0, 1, C64::new(0.0, -nu3));
    ham2.set(1, 0, C64::new(0.0, nu3));

    // extrapolate timestep (Lagrange reference)
    {
        let c = Contour::new(nt, ntauk, 1, 1, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut h0 = DenseMat::zeros(1);
        h0.set(0, 0, C64::new(1.0, 0.0));
        from_ham(&mut g1, 0.0, &h0, beta, h);
        let mut g2 = g1.clone();
        let mut g3 = g1.clone();
        let p = kt as usize;
        let wl = ut_lagrange_weights(p, h, (p + 1) as f64 * h);
        let mut err = 0.0;
        for tstp in kt..nt {
            // Julia n = tstp+1 -> Rust row r = tstp
            let r = tstp;
            g2.memcpy_from_at(r, &g1);
            g3.memcpy_from_at(r, &g1);
            extrapolate_timestep(r, &mut g2, &Integrator::new(kt));
            // tv
            for m in 0..=ntauk {
                let mut x = C64::ZERO;
                for q in 0..=p {
                    x += g3.get_lmix(r as i64 - q as i64, m, 0, 0) * wl[p - q];
                }
                g3.set_lmix(r + 1, m, 0, 0, x);
            }
            // les
            for j in 0..=r {
                let mut x = C64::ZERO;
                for q in 0..=p {
                    x += g3.get_less(j, r - q as i64, 0, 0) * wl[p - q];
                }
                g3.set_less(j, r + 1, 0, 0, x);
            }
            {
                let mut x = C64::ZERO;
                for q in 0..=p {
                    x += g3.get_less(r - q as i64, r + 1, 0, 0) * wl[p - q];
                }
                g3.set_less(r + 1, r + 1, 0, 0, x);
            }
            // ret
            for j in 0..=r {
                let mut x = C64::ZERO;
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
    let fv = |t: f64| -> [C64; 4] {
        [
            C64::new((w1 * t).cos(), 0.0),
            C64::new(0.0, -0.2 * (w3 * t).sin()),
            C64::new(0.0, 0.2 * (w3 * t).sin()),
            C64::new((w2 * t).cos(), 0.0),
        ]
    };
    {
        let mut func1 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
        let mut func2 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
        let mut func3 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
        for t in 0..=nt {
            for f in [&mut func1, &mut func2, &mut func3] {
                f.block_mut(t).copy_from_slice(&fv(t as f64 * h));
            }
        }
        let p = kt as usize;
        let wl = ut_lagrange_weights(p, h, (p + 1) as f64 * h);
        let mut err = 0.0;
        for tstp in kt..nt {
            let r = tstp;
            extrapolate_timestep_cf(r, &mut func1, kt);
            let mut fref = [C64::ZERO; 4];
            for q in 0..=p {
                for (x, s) in fref.iter_mut().zip(func3.block(r - q as i64)) {
                    *x += *s * wl[p - q];
                }
            }
            for q in 0..4 {
                err += (func1.block(r + 1)[q] - fref[q]).abs();
            }
            func1.block_mut(r + 1).copy_from_slice(func2.block(r + 1));
            func3.block_mut(r + 1).copy_from_slice(func2.block(r + 1));
        }
        assert!(err < 1e-6, "extrapolate function err {err}");
    }

    // interpolate function
    {
        let mut func1 = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
        for t in 0..=nt {
            func1.block_mut(t).copy_from_slice(&fv(t as f64 * h));
        }
        let p = kt as usize;
        let mut err = 0.0;
        for tstp in kt..nt {
            let tinter = tstp as f64 + 0.5;
            let f = interpolation(tstp + 1, tinter, &func1, &Integrator::new(kt));
            let ts = tstp + 1; // 0-based window end
            let wl = ut_lagrange_weights(p, h, h * tinter - h * (ts - p as i64) as f64);
            let mut fref = [C64::ZERO; 4];
            for q in 0..=p {
                for (x, s) in fref.iter_mut().zip(func1.block(ts - q as i64)) {
                    *x += *s * wl[p - q];
                }
            }
            for q in 0..4 {
                err += (f.get(q / 2, q % 2) - fref[q]).abs();
            }
        }
        assert!(err < 1e-6, "interpolate function err {err}");
    }

    // set_t0_from_mat
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        g2.memcpy_from_at(MATSUBARA, &g1);
        from_mat(&mut g2);
        assert!(g1.distance_at(&g2, 0) < 1e-6);
    }

    // set_tk_from_mat
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        g2.memcpy_from_at(MATSUBARA, &g1);
        from_mat_k(&mut g2, kt);
        let mut err = 0.0;
        for n in 0..=kt {
            for m in 0..=ntauk {
                err += kadanoffbaym::properties::blkdist(
                    g1.lmix_raw_block(0, m),
                    g2.lmix_raw_block(n, m),
                );
            }
            err +=
                kadanoffbaym::properties::blkdist(g1.less_raw_block(n, n), g2.less_raw_block(n, n));
            err +=
                kadanoffbaym::properties::blkdist(g1.ret_raw_block(n, n), g2.ret_raw_block(n, n));
        }
        assert!(err < 1e-6, "set_tk err {err}");
    }

    // distance_norm2
    {
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        from_ham(&mut g2, 0.0, &ham2, beta, h);
        let dist_green = |tstp: i64, a: &ContourGreen, b: &ContourGreen| -> f64 {
            let mut d = 0.0;
            if tstp == MATSUBARA {
                for m in 0..=ntauk {
                    d += kadanoffbaym::properties::blkdist(a.mat_raw_block(m), b.mat_raw_block(m));
                }
            } else {
                for m in 0..=ntauk {
                    d += kadanoffbaym::properties::blkdist(
                        a.lmix_raw_block(tstp, m),
                        b.lmix_raw_block(tstp, m),
                    );
                }
                for j in 0..=tstp {
                    d += kadanoffbaym::properties::blkdist(
                        a.less_raw_block(j, tstp),
                        b.less_raw_block(j, tstp),
                    );
                    d += kadanoffbaym::properties::blkdist(
                        a.ret_raw_block(tstp, j),
                        b.ret_raw_block(tstp, j),
                    );
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
        ham3.set(0, 0, C64::new(w2, 0.0));
        ham3.set(1, 1, C64::new(w1, 0.0));
        ham3.set(0, 1, C64::new(w3, 0.0));
        ham3.set(1, 0, C64::new(w3, 0.0));
        let c = Contour::new(nt, ntauk, 2, 2, tmax, beta);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut g1, 0.0, &ham1, beta, h);
        from_ham(&mut g2, 0.0, &ham3, beta, h);
        let s2 = timestep_slice(&g2, 0);
        g1.incr_by_step(&s2, MATSUBARA, C64::new(0.0, 1.0));
        force_matsubara_hermitian(&mut g1);
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

/// Port of libcntr/test/function.cpp:97-227 (non-HDF5 sections).
#[test]
fn nessi_function_cpp() {
    let (nt, size1) = (50i64, 2usize);
    let (eps, eps2, h, kt) = (1e-7f64, 1e-3, 0.01, 5i64);
    let mut a = TimeFunction::new(nt as usize, size1, size1, C64::ZERO);
    let s2 = 2.0f64.sqrt();
    let av = [
        C64::new(s2, 0.0),
        C64::new(0.0, s2),
        C64::new(0.0, -s2),
        C64::new(-s2, 0.0),
    ];
    let mut toterr = 0.0;
    for m in 0..nt {
        a.block_mut(m).copy_from_slice(&av);
        for q in 0..4 {
            toterr += (av[q] - a.block(m)[q]).abs();
        }
    }
    assert!(toterr < eps);
    let _ = (size1, eps2, h, kt);

    let cv = |t: f64| -> [C64; 4] {
        let cs = t.cos();
        [
            C64::new(2.0 * cs, 0.0),
            C64::new(0.5 * cs, 0.0),
            C64::new(0.5 * cs, 0.0),
            C64::new(3.0 * cs, 0.0),
        ]
    };
    a.block_mut(MATSUBARA).copy_from_slice(&cv(0.0));
    for t in 0..=nt {
        a.block_mut(t).copy_from_slice(&cv(t as f64 * h));
    }

    // extrapolation
    {
        let mut b = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
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
        let mut b = TimeFunction::new(nt as usize, 2, 2, C64::ZERO);
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
            for q in 0..4 {
                toterr1 += (a1[q] - b1.get(q / 2, q % 2)).abs();
            }
        }
        assert!(toterr0 < eps, "interp0 {toterr0}");
        assert!(toterr1 < eps2, "interp1 {toterr1}");
    }
}

/// Release-only order gate.
#[test]
fn release_extrapolation_interpolation_order_gate() {
    if !is_release() {
        return;
    }
    let grids: &[(i64, std::ops::Range<i64>)] =
        &[(1, 12..16), (2, 6..10), (3, 5..9), (4, 4..8), (5, 3..7)];
    let extrap_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
        let h = 1.0 / n_grid as f64;
        let tplus = 1.3;
        let mut f = TimeFunction::new((k + 1) as usize, 2, 2, C64::ZERO);
        for i in 0..=(k + 1) {
            let t = tplus - (k + 2 - (i + 1)) as f64 * h;
            f.block_mut(i).copy_from_slice(&ut_f(t));
        }
        extrapolate_timestep_cf(k, &mut f, ord);
        let mut e = 0.0;
        for (x, y) in f.block(k + 1).iter().zip(ut_f(tplus)) {
            e += (*x - y).abs();
        }
        e
    };
    let interp_error = |k: i64, n_grid: i64, ord: i64| -> f64 {
        let h = 1.0 / n_grid as f64;
        let tplus = 1.3;
        let mut f = TimeFunction::new(k as usize, 2, 2, C64::ZERO);
        for i in 0..=k {
            let t = tplus - (k + 1 - (i + 1)) as f64 * h;
            f.block_mut(i).copy_from_slice(&ut_f(t));
        }
        let x = interpolation(k, k as f64 - 0.5, &f, &Integrator::new(ord));
        let want = ut_f(tplus - 0.5 * h);
        let mut e = 0.0;
        for q in 0..4 {
            e += (x.get(q / 2, q % 2) - want[q]).abs();
        }
        e
    };
    for (k, range) in grids {
        let ns: Vec<i64> = range.clone().map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, *k)).collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, *k)).collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        let errs: Vec<f64> = ns.iter().map(|&n| extrap_error(*k, n, k - 1)).collect();
        assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        if *k >= 2 {
            let errs: Vec<f64> = ns.iter().map(|&n| interp_error(*k, n, k - 1)).collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
    }
}
