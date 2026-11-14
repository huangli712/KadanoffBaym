// Project : KadanoffBaym
// File    : tests/differentiation.rs
// Purpose : port of julia/test/t_differentiation.jl

mod common;

use common::prelude::*;
use common::{assert_order_gate, check_order_gate, is_release};
use kadanoffbaym::math::deriv::{
    deriv1_element, deriv1_lmix, deriv1_matsubara, deriv1_timestep, deriv2_element, deriv2_lmix,
    deriv2_timestep,
};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::expm;
use kadanoffbaym::types::indexing::timestep_at;
use kadanoffbaym::weights::Integrator;

/// Deterministic pseudo-random complex stream (splitmix64).
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

    fn c64(&mut self) -> Complex {
        Complex::new(self.next_f64(), self.next_f64())
    }
}

/// Two-variable matrix polynomial of degree `deg` with `a + b <= deg`
/// terms (blueprint `df_poly2` with deterministic coefficients).
struct Poly2 {
    deg: usize,
    nel: usize,
    c: Vec<Vec<Vec<Complex>>>, // c[a][b] = block
}

impl Poly2 {
    fn new(deg: usize, nel: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut c = vec![vec![vec![Complex::ZERO; nel]; deg + 1]; deg + 1];
        for a in 0..=deg {
            for b in 0..=deg {
                if a + b <= deg {
                    for z in c[a][b].iter_mut() {
                        *z = rng.c64();
                    }
                }
            }
        }
        Self { deg, nel, c }
    }

    fn f(&self, t: f64, s: f64) -> Vec<Complex> {
        let mut out = vec![Complex::ZERO; self.nel];
        for a in 0..=self.deg {
            for b in 0..=self.deg {
                if a + b <= self.deg {
                    let w = Complex::new(t.powi(a as i32) * s.powi(b as i32), 0.0);
                    for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                        *o += *cz * w;
                    }
                }
            }
        }
        out
    }

    /// d/dt.
    fn d1(&self, t: f64, s: f64) -> Vec<Complex> {
        let mut out = vec![Complex::ZERO; self.nel];
        for a in 1..=self.deg {
            for b in 0..=self.deg {
                if a + b <= self.deg {
                    let w = Complex::new((a as f64) * t.powi(a as i32 - 1) * s.powi(b as i32), 0.0);
                    for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                        *o += *cz * w;
                    }
                }
            }
        }
        out
    }

    /// d/ds.
    fn d2(&self, t: f64, s: f64) -> Vec<Complex> {
        let mut out = vec![Complex::ZERO; self.nel];
        for a in 0..=self.deg {
            for b in 1..=self.deg {
                if a + b <= self.deg {
                    let w = Complex::new((b as f64) * t.powi(a as i32) * s.powi(b as i32 - 1), 0.0);
                    for (o, cz) in out.iter_mut().zip(&self.c[a][b]) {
                        *o += *cz * w;
                    }
                }
            }
        }
        out
    }
}

/// -(block)† for square blocks.
fn neg_adj(blk: &[Complex]) -> Vec<Complex> {
    let n = (blk.len() as f64).sqrt() as usize;
    let mut out = vec![Complex::ZERO; blk.len()];
    for a in 0..n {
        for b in 0..n {
            out[b * n + a] = -blk[a * n + b].conj();
        }
    }
    out
}

fn df_fill(
    a: &mut ContourGreen,
    acc: &mut ContourGreen,
    h: f64,
    fr: &Poly2,
    fl: &Poly2,
    fm: &Poly2,
    fx: &Poly2,
    hv: f64,
) {
    let (nt, ntau) = (a.nt as i64, a.ntau as i64);
    for i in 0..=nt {
        for j in 0..=i {
            a.ret_raw_block_mut(i, j)
                .copy_from_slice(&fr.f(i as f64 * h, j as f64 * h));
            acc.ret_raw_block_mut(i, j)
                .copy_from_slice(&neg_adj(&fr.f(j as f64 * h, i as f64 * h)));
        }
    }
    for j in 0..=nt {
        for i in 0..=j {
            a.less_raw_block_mut(i, j)
                .copy_from_slice(&fl.f(i as f64 * h, j as f64 * h));
            acc.less_raw_block_mut(i, j)
                .copy_from_slice(&neg_adj(&fl.f(j as f64 * h, i as f64 * h)));
        }
    }
    for i in 0..=nt {
        for m in 0..=ntau {
            let v = fm.f(i as f64 * h, m as f64 * hv);
            a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
            acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
        }
    }
    for m in 0..=ntau {
        let v = fx.f(m as f64 * hv, 0.0);
        a.mat_raw_block_mut(m).copy_from_slice(&v);
        acc.mat_raw_block_mut(m).copy_from_slice(&v);
    }
}

fn dm_at(m: &DenseMat, q: usize) -> Complex {
    m.get(q / m.ncols(), q % m.ncols())
}

fn max_blk_err(err: f64, got: &[Complex], want: &[Complex]) -> f64 {
    got.iter()
        .zip(want)
        .fold(err, |e, (g, w)| e.max((*g - *w).abs()))
}

#[test]
fn polynomial_exactness() {
    let ic = Complex::new(0.0, 1.0);
    for k in 1..=5i64 {
        for nel in [1usize, 4] {
            let dims = if nel == 1 { (1, 1) } else { (2, 2) };
            // blueprint: ntime = 2k+5 points, ntau = 2k+4 points
            let (nt, ntau) = (2 * k + 4, 2 * k + 3);
            let (h, beta) = (0.1f64, 0.7f64);
            let hv = beta / ntau as f64;
            let c = Contour::new(nt, ntau, dims.0, dims.1, nt as f64 * h, beta);
            let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            let fr = Poly2::new(k as usize, nel, 0x1234 + k as u64);
            let fl = Poly2::new(k as usize, nel, 0x2345 + k as u64);
            let fm = Poly2::new(k as usize, nel, 0x3456 + k as u64);
            let fx = Poly2::new(k as usize + 1, nel, 0x4567 + k as u64);
            df_fill(&mut a, &mut acc, h, &fr, &fl, &fm, &fx, hv);

            // left derivative, all time steps (MATSUBARA + real)
            let integ = Integrator::new(k);
            let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            for tstp in -1..=nt {
                deriv1_timestep(tstp, &mut da, &a, Some(&acc), &integ, beta, h);
            }
            let mut err = 0.0f64;
            for i in 0..=nt {
                for j in 0..=i {
                    let want = fr
                        .d1(i as f64 * h, j as f64 * h)
                        .iter()
                        .map(|z| ic * *z)
                        .collect::<Vec<_>>();
                    err = max_blk_err(err, da.ret_raw_block(i, j), &want);
                }
                for j in i..=nt {
                    let want = fl
                        .d1(i as f64 * h, j as f64 * h)
                        .iter()
                        .map(|z| ic * *z)
                        .collect::<Vec<_>>();
                    err = max_blk_err(err, da.less_raw_block(i, j), &want);
                }
                for m in 0..=ntau {
                    let want = fm
                        .d1(i as f64 * h, m as f64 * hv)
                        .iter()
                        .map(|z| ic * *z)
                        .collect::<Vec<_>>();
                    err = max_blk_err(err, da.lmix_raw_block(i, m), &want);
                }
            }
            for m in 0..=ntau {
                // blueprint: dA.mat + Dx1 == 0  =>  dA.mat = -Dx1
                let want: Vec<Complex> = fx.d1(m as f64 * hv, 0.0).iter().map(|z| -*z).collect();
                err = max_blk_err(err, da.mat_raw_block(m), &want);
            }
            assert!(err <= 1e-10, "k={k} nel={nel} left err={err}");

            // right derivative
            let mut db = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            for tstp in -1..=nt {
                deriv2_timestep(tstp, &mut db, &a, Some(&acc), &integ, beta, h);
            }
            let mut err = 0.0f64;
            for i in 0..=nt {
                for j in 0..=i {
                    let want = fr
                        .d2(i as f64 * h, j as f64 * h)
                        .iter()
                        .map(|z| -(ic * *z))
                        .collect::<Vec<_>>();
                    err = max_blk_err(err, db.ret_raw_block(i, j), &want);
                }
                for j in i..=nt {
                    let want = fl
                        .d2(i as f64 * h, j as f64 * h)
                        .iter()
                        .map(|z| -(ic * *z))
                        .collect::<Vec<_>>();
                    err = max_blk_err(err, db.less_raw_block(i, j), &want);
                }
                for m in 0..=ntau {
                    let want = fm.d2(i as f64 * h, m as f64 * hv);
                    err = max_blk_err(err, db.lmix_raw_block(i, m), &want);
                }
            }
            assert!(err <= 1e-10, "k={k} nel={nel} right err={err}");
            assert!(db.mat.equal(&da.mat));
        }
    }
}

#[test]
fn hermitian_gf_left_and_right_derivatives() {
    // blueprint: ntime = 40 points, ntau = 30 points
    let (nt, ntau) = (39i64, 29i64);
    let (beta, h, mu) = (2.0f64, 0.01f64, 0.1f64);
    let mut hm = DenseMat::zeros(2);
    hm.set(0, 0, Complex::new(-0.4, 0.0));
    hm.set(1, 1, Complex::new(0.6, 0.0));
    hm.set(0, 1, Complex::new(0.0, 0.3));
    hm.set(1, 0, Complex::new(0.0, -0.3));
    let ident = DenseMat::identity(2);
    for sign in [Sign::Fermi, Sign::Bose] {
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let mut g = ContourGreen::new(&c, sign, Complex::ZERO);
        // H + 2I for bosons (blueprint shift)
        let mut h_use = hm.clone();
        if sign == Sign::Bose {
            h_use.axpy(2.0, &ident);
        }
        from_ham(&mut g, mu, &h_use, beta, h);
        // Hs = H - mu I (+ 2I for bosons)
        let mut hs = hm.clone();
        if sign == Sign::Bose {
            hs.axpy(2.0, &ident);
        }
        let mut hs2 = hs.clone();
        hs2.axpy(-mu, &ident);
        let k = 5;
        let integ = Integrator::new(k);
        let mut da = ContourGreen::new(&c, sign, Complex::ZERO);
        let mut db = ContourGreen::new(&c, sign, Complex::ZERO);
        for tstp in 0..=nt {
            deriv1_timestep(tstp, &mut da, &g, None, &integ, beta, h);
            deriv2_timestep(tstp, &mut db, &g, None, &integ, beta, h);
        }
        let mut err = 0.0f64;
        for i in 0..=nt {
            for j in 0..=i {
                let gr = DenseMat::from_vec(2, 2, g.ret_raw_block(i, j).to_vec());
                let want_l = hs2.mat_mul(&gr);
                let want_r = gr.mat_mul(&hs2);
                let dr = DenseMat::from_vec(2, 2, da.ret_raw_block(i, j).to_vec());
                let drr = DenseMat::from_vec(2, 2, db.ret_raw_block(i, j).to_vec());
                for q in 0..4 {
                    err = err.max((dm_at(&dr, q) - dm_at(&want_l, q)).abs());
                    err = err.max((dm_at(&drr, q) - dm_at(&want_r, q)).abs());
                }
            }
            for j in i..=nt {
                let gl = DenseMat::from_vec(2, 2, g.less_raw_block(i, j).to_vec());
                let want_ll = hs2.mat_mul(&gl);
                let want_lr = gl.mat_mul(&hs2);
                let dl = DenseMat::from_vec(2, 2, da.less_raw_block(i, j).to_vec());
                let dlr = DenseMat::from_vec(2, 2, db.less_raw_block(i, j).to_vec());
                for q in 0..4 {
                    err = err.max((dm_at(&dl, q) - dm_at(&want_ll, q)).abs());
                    err = err.max((dm_at(&dlr, q) - dm_at(&want_lr, q)).abs());
                }
            }
            for m in 0..=ntau {
                let gm = DenseMat::from_vec(2, 2, g.lmix_raw_block(i, m).to_vec());
                let want = hs2.mat_mul(&gm);
                let dm = DenseMat::from_vec(2, 2, da.lmix_raw_block(i, m).to_vec());
                for q in 0..4 {
                    err = err.max((dm_at(&dm, q) - dm_at(&want, q)).abs());
                }
            }
        }
        assert!(err <= 1e-8, "sign={:?} err={err}", sign);
    }
}

#[test]
fn timestep_entry_equals_element_calls() {
    let (nt, ntau) = (9i64, 8i64);
    let (h, beta) = (0.1f64, 1.0f64);
    let hv = beta / ntau as f64;
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    // the blueprint reuses a degree-4 polynomial with tau = 0.1 * m
    let fm = Poly2::new(4, 4, 0x9999);
    let fr = Poly2::new(4, 4, 0xAAAA);
    let fl = Poly2::new(4, 4, 0xBBBB);
    // fm-only fill with the blueprint's tau rule (t, 0.1*(m+1)) etc.
    let mut fill = |a: &mut ContourGreen, acc: &mut ContourGreen| {
        for i in 0..=nt {
            for j in 0..=i {
                a.ret_raw_block_mut(i, j)
                    .copy_from_slice(&fr.f(i as f64 * h, j as f64 * h));
                acc.ret_raw_block_mut(i, j)
                    .copy_from_slice(&neg_adj(&fr.f(j as f64 * h, i as f64 * h)));
            }
            for j in i..=nt {
                a.less_raw_block_mut(i, j)
                    .copy_from_slice(&fl.f(i as f64 * h, j as f64 * h));
                acc.less_raw_block_mut(i, j)
                    .copy_from_slice(&neg_adj(&fl.f(j as f64 * h, i as f64 * h)));
            }
        }
        for i in 0..=nt {
            for m in 0..=ntau {
                let v = fm.f(i as f64 * h, 0.1 * (m + 1) as f64);
                a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
            }
        }
        for m in 0..=ntau {
            let v = fm.f(0.1 * (m + 1) as f64, 0.0);
            a.mat_raw_block_mut(m).copy_from_slice(&v);
            acc.mat_raw_block_mut(m).copy_from_slice(&v);
        }
    };
    fill(&mut a, &mut acc);

    let integ = Integrator::new(3);
    let sentinel = Complex::new(3.0, -1.0);
    for tstp in [1i64, 3, 6] {
        // left
        let mut da = ContourGreen::new(&c, Sign::Fermi, sentinel);
        let mut de = ContourGreen::new(&c, Sign::Fermi, sentinel);
        deriv1_timestep(tstp, &mut da, &a, Some(&acc), &integ, beta, h);
        for j in 0..=nt {
            deriv1_element(tstp, j, &mut de, &a, Some(&acc), &integ, h);
        }
        deriv1_lmix(tstp, &mut de, &a, &integ, h);
        assert!(da.equal(&de), "tstp {tstp} left");
        // only the target time step is touched
        for i in 0..=nt {
            if i == tstp {
                continue;
            }
            for j in 0..=i {
                assert!(da.ret_raw_block(i, j).iter().all(|z| *z == sentinel));
            }
            for j in i..=nt {
                assert!(da.less_raw_block(i, j).iter().all(|z| *z == sentinel));
            }
            for m in 0..=ntau {
                assert!(da.lmix_raw_block(i, m).iter().all(|z| *z == sentinel));
            }
        }
        assert!(da.mat.raw().iter().all(|z| *z == sentinel));
        // right
        let mut db = ContourGreen::new(&c, Sign::Fermi, sentinel);
        let mut de = ContourGreen::new(&c, Sign::Fermi, sentinel);
        deriv2_timestep(tstp, &mut db, &a, Some(&acc), &integ, beta, h);
        for j in 0..=nt {
            deriv2_element(tstp, j, &mut de, &a, Some(&acc), &integ, h);
        }
        deriv2_lmix(tstp, &mut de, &a, &integ, beta);
        assert!(db.equal(&de), "tstp {tstp} right");
    }
}

#[test]
fn matsubara_grid_lower_bound() {
    for k in 1..=5i64 {
        let integ = Integrator::new(k);
        // minimal valid grid: ntau = 2k+1 intervals (2k+2 points)
        let c = Contour::new(k + 1, 2 * k + 1, 1, 1, 1.0, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for m in 0..=(c.ntau as i64) {
            a.set_mat(m, 0, 0, Complex::new((m + 1) as f64 * (m + 1) as f64, 0.0));
        }
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        deriv1_matsubara(&mut da, &a, &integ, 1.0);
        assert!(da
            .mat
            .raw()
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite()));
    }
    // one point less must be rejected
    {
        let result = std::panic::catch_unwind(|| {
            let k = 3i64;
            let integ = Integrator::new(k);
            let c = Contour::new(k + 1, 2 * k, 1, 1, 1.0, 1.0);
            let a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            deriv1_matsubara(&mut da, &a, &integ, 1.0);
        });
        assert!(result.is_err(), "ntau = 2k must be rejected");
    }
    // other preconditions (blueprint Cn(6, 6) -> Rust (5, 5))
    let c = Contour::new(5, 5, 1, 1, 1.0, 1.0);
    let a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let acc_g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut out = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let integ2 = Integrator::new(2);
    // invalid tstp
    let r = std::panic::catch_unwind(|| {
        let mut o = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        deriv1_element(-1, 0, &mut o, &a, Some(&acc_g), &integ2, 0.1);
    });
    assert!(r.is_err());
    let r = std::panic::catch_unwind(|| {
        let mut o = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        deriv1_lmix(6, &mut o, &a, &integ2, 0.1);
    });
    assert!(r.is_err());
    // NOTE (deviation): the blueprint also asserts that using the output
    // object as input A or Acc is rejected. In safe Rust the aliased call
    // `deriv1_element(.., &mut g, &g, ..)` cannot even be constructed
    // (E0502), so the library-side `ptr::eq` guards stand untested here;
    // the compiler enforces the same guarantee statically.
}

/// Release-only: convergence order gates (blueprint
/// `kb_release()` section).
#[test]
fn release_differentiation_order_gate() {
    if !is_release() {
        return;
    }
    // H, X, Z matrices and U(t) = exp(-i H t) products; derivatives are
    // compared against the analytic i d/dt relations
    let mut hm = DenseMat::zeros(2);
    hm.set(0, 0, Complex::new(-0.4, 0.0));
    hm.set(1, 1, Complex::new(0.6, 0.0));
    hm.set(0, 1, Complex::new(0.0, 0.3));
    hm.set(1, 0, Complex::new(0.0, -0.3));
    let u = |t: f64| -> DenseMat {
        // U(t) = exp(-i H t)
        let mut arg = DenseMat::zeros_shape(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                arg.set(i, j, Complex::new(0.0, -t) * hm.get(i, j));
            }
        }
        expm(&arg)
    };
    let z = block_z();

    // error of the left derivative at grid index i (0-based `i_rust`,
    // blueprint i_julia = i_rust + 1)
    let deriv_error = |k: i64, ord: i64, h: f64, i_julia: i64| -> f64 {
        let i_rust = i_julia - 1;
        let ntime = (i_julia.max(k + 1) + 1) as i64; // points
        let nt = ntime - 1;
        let ntau = 2 * k + 1; // intervals (2k+2 points)
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut acc = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // Fr(t,s) = -i U(t+0.3) U(s+0.3)†; Fl(t,s) = i U(t+0.3) X U(s+0.3)†;
        // Acc entries carry the SWAPPED-argument conjugate (blueprint
        // `Acc.ret[i,j] = -Fr(t_j, t_i)†`, same for less)
        for i in 0..=nt {
            for j in 0..=i {
                let v = mul_c(
                    -C64_I,
                    &u(i as f64 * h + 0.3).mat_mul(&u(j as f64 * h + 0.3).adjoint()),
                );
                a.ret_raw_block_mut(i, j).copy_from_slice(&v);
                let vsw = mul_c(
                    -C64_I,
                    &u(j as f64 * h + 0.3).mat_mul(&u(i as f64 * h + 0.3).adjoint()),
                );
                acc.ret_raw_block_mut(i, j).copy_from_slice(&neg_adj(&vsw));
            }
            for j in i..=nt {
                let x = block_x();
                let v = mul_c(
                    C64_I,
                    &u(i as f64 * h + 0.3)
                        .mat_mul(&x)
                        .mat_mul(&u(j as f64 * h + 0.3).adjoint()),
                );
                a.less_raw_block_mut(i, j).copy_from_slice(&v);
                let vsw = mul_c(
                    C64_I,
                    &u(j as f64 * h + 0.3)
                        .mat_mul(&x)
                        .mat_mul(&u(i as f64 * h + 0.3).adjoint()),
                );
                acc.less_raw_block_mut(i, j).copy_from_slice(&neg_adj(&vsw));
            }
        }
        for i in 0..=nt {
            for m in 0..=ntau {
                let v = to_block(&u(i as f64 * h).mat_mul(&z));
                a.lmix_raw_block_mut(i, m).copy_from_slice(&v);
                acc.lmix_raw_block_mut(i, m).copy_from_slice(&v);
            }
        }
        for m in 0..=ntau {
            a.mat_raw_block_mut(m).copy_from_slice(&to_block(&z));
            acc.mat_raw_block_mut(m).copy_from_slice(&to_block(&z));
        }
        let integ = Integrator::new(ord);
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut db = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut err = 0.0f64;
        for j in 0..=(i_julia - 1) {
            deriv1_element(i_rust, j, &mut da, &a, Some(&acc), &integ, h);
            deriv2_element(j, i_rust, &mut db, &a, Some(&acc), &integ, h);
            let t = i_rust as f64 * h;
            let s = j as f64 * h;
            // i d/dt Fr = H Fr ; i d/ds Fl = ... Fl H
            let want_l = hm.mat_mul(&u(t + 0.3).mat_mul(&u(s + 0.3).adjoint()));
            let got_l = DenseMat::from_vec(2, 2, da.ret_raw_block(i_rust, j).to_vec());
            let want_l = DenseMat::from_vec(2, 2, mul_c(-C64_I, &want_l));
            for q in 0..4 {
                err = err.max((dm_at(&got_l, q) - dm_at(&want_l, q)).abs());
            }
            let x = block_x();
            let fl = u(s + 0.3).mat_mul(&x).mat_mul(&u(t + 0.3).adjoint());
            let want_r = fl.mat_mul(&hm);
            let want_r = DenseMat::from_vec(2, 2, mul_c(C64_I, &want_r));
            let got_r = DenseMat::from_vec(2, 2, db.less_raw_block(j, i_rust).to_vec());
            for q in 0..4 {
                err = err.max((dm_at(&got_r, q) - dm_at(&want_r, q)).abs());
            }
        }
        err
    };

    // matsubara error: -d/dtau of Z e^{-lambda tau} at tau_0
    let mat_error = |k: i64, ord: i64, hv: f64| -> f64 {
        let ntau = (2 * k + 2) as i64; // intervals (2k+3 points)
        let beta = hv * ntau as f64;
        let c = Contour::new(1, ntau, 2, 2, 1.0, beta);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let lam = 0.8f64;
        for m in 0..=ntau {
            let v = to_block(&mat_scale(&z, (-lam * m as f64 * hv).exp()));
            a.mat_raw_block_mut(m).copy_from_slice(&v);
        }
        let mut da = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        deriv1_matsubara(&mut da, &a, &Integrator::new(ord), beta);
        let want = mat_scale(&z, lam);
        let got = da.mat_raw_block(0);
        let wantb = to_block(&want);
        got.iter()
            .zip(&wantb)
            .fold(0.0, |e, (g, w)| e.max((*g - *w).abs()))
    };

    let grids: Vec<(i64, i64, std::ops::Range<i64>)> = vec![
        (1, 0, 11..15),
        (1, 2, 11..15),
        (1, 4, 11..15),
        (2, 1, 11..15),
        (2, 2, 6..10),
        (2, 4, 6..10),
        (3, 1, 6..10),
        (3, 2, 4..8),
        (3, 4, 4..8),
        (4, 1, 4..8),
        (4, 2, 3..7),
        (4, 4, 3..7),
        (5, 1, 3..7),
        (5, 2, 2..6),
        (5, 4, 2..6),
    ];
    // (k, tag): tag 0 = start-up, 1 = stable, 2 = matsubara
    for k in 1..=5i64 {
        if k >= 2 {
            // start-up (i = 1): order k (UT_GRIDS `:s`)
            let srange = grids
                .iter()
                .find(|g| g.0 == k && g.1 == 1)
                .unwrap()
                .2
                .clone();
            let ns: Vec<i64> = srange.map(|m| 1 << m).collect();
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k, 1.0 / n as f64, 1))
                .collect();
            assert_order_gate(&ns, &errs, k as f64, 1e-8);
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k - 1, 1.0 / n as f64, 1))
                .collect();
            assert!(!check_order_gate(&ns, &errs, k as f64, 1e-8, 1e-13, 0.3).ok);
        }
        // stable (i = k+2): order k+1
        let mrange = grids
            .iter()
            .find(|g| g.0 == k && g.1 == 2)
            .unwrap()
            .2
            .clone();
        let ns: Vec<i64> = mrange.map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns
            .iter()
            .map(|&n| deriv_error(k, k, 1.0 / n as f64, k + 2))
            .collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        if k >= 2 {
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| deriv_error(k, k - 1, 1.0 / n as f64, k + 2))
                .collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
        // matsubara: order k+1
        let mrange = grids
            .iter()
            .find(|g| g.0 == k && g.1 == 4)
            .unwrap()
            .2
            .clone();
        let ns: Vec<i64> = mrange.map(|m| 1 << m).collect();
        let errs: Vec<f64> = ns
            .iter()
            .map(|&n| mat_error(k, k, 1.0 / n as f64))
            .collect();
        assert_order_gate(&ns, &errs, (k + 1) as f64, 1e-8);
        if k >= 2 {
            let errs: Vec<f64> = ns
                .iter()
                .map(|&n| mat_error(k, k - 1, 1.0 / n as f64))
                .collect();
            assert!(!check_order_gate(&ns, &errs, (k + 1) as f64, 1e-8, 1e-13, 0.3).ok);
        }
    }
}

const C64_I: Complex = Complex::new(0.0, 1.0);

fn block_x() -> DenseMat {
    let mut x = DenseMat::zeros(2);
    x.set(0, 0, Complex::new(0.3, 0.0));
    x.set(0, 1, Complex::new(0.1, 0.0));
    x.set(1, 0, Complex::new(0.1, 0.0));
    x.set(1, 1, Complex::new(0.2, 0.0));
    x
}

fn block_z() -> DenseMat {
    let mut z = DenseMat::zeros(2);
    z.set(0, 0, Complex::new(0.5, 0.0));
    z.set(0, 1, Complex::new(0.0, 0.2));
    z.set(1, 0, Complex::new(0.0, -0.2));
    z.set(1, 1, Complex::new(0.4, 0.0));
    z
}

fn to_block(m: &DenseMat) -> Vec<Complex> {
    (0..m.nrows() * m.ncols()).map(|q| dm_at(&m, q)).collect()
}

fn mul_c(c: Complex, m: &DenseMat) -> Vec<Complex> {
    to_block(m).iter().map(|z| c * *z).collect()
}

fn mat_scale(m: &DenseMat, s: f64) -> DenseMat {
    let mut out = DenseMat::zeros_shape(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.set(i, j, m.get(i, j).scale(s));
        }
    }
    out
}
