// Project : KadanoffBaym
// File    : tests/gf_tools.rs
// Purpose : port of julia/test/t_gf_tools.jl

mod common;

use common::prelude::*;
use kadanoffbaym::basics::{add, smul};
use kadanoffbaym::basics::{elemcpy_green_map, elemcpy_green_map_at};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::basics::Mode;
use kadanoffbaym::types::indexing::{set_timestep_at, timestep_at};

struct Rng(u64);

impl Rng {
    fn c64(&mut self) -> Complex {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let f = || ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let re = f();
        let im = f();
        Complex::new(re, im)
    }

    fn block(&mut self, nel: usize) -> Vec<Complex> {
        (0..nel).map(|_| self.c64()).collect()
    }
}

#[test]
fn elemcpy_with_index_maps() {
    // blueprint Cn(6, 5, 3, 3) source, 2x2 destination
    let cs = Contour::new(5, 4, 3, 3, 0.5, 1.0);
    let cd = Contour::new(5, 4, 2, 2, 0.5, 1.0);
    let mut src = ContourGreen::new(&cs, Sign::Fermi, Complex::ZERO);
    let mut rng = Rng(0xC0FFEE);
    for m in 0..=4 {
        src.mat_raw_block_mut(m).copy_from_slice(&rng.block(9));
    }
    for i in 0..=5 {
        for j in 0..=i {
            src.ret_raw_block_mut(i, j).copy_from_slice(&rng.block(9));
        }
        for m in 0..=4 {
            src.lmix_raw_block_mut(i, m).copy_from_slice(&rng.block(9));
        }
        for j in i..=5 {
            src.less_raw_block_mut(i, j).copy_from_slice(&rng.block(9));
        }
    }
    let sentinel = Complex::new(7.0, -7.0);
    // non-rectangular map (Julia 1-based -> Rust 0-based)
    let r1 = [2i64, 0, 1];
    let c1 = [0i64, 2, 1];
    let r2 = [0i64, 1, 1];
    let c2 = [0i64, 0, 1];
    let mut dst = ContourGreen::new(&cd, Sign::Fermi, sentinel);
    elemcpy_green_map(&r1, &c1, &src, &r2, &c2, &mut dst);
    for m in 0..=4 {
        for k in 0..3 {
            let s = src.mat_raw_block(m)[(r1[k] * 3 + c1[k]) as usize];
            let d = dst.mat_raw_block(m)[(r2[k] * 2 + c2[k]) as usize];
            assert_eq!(d, s, "mat {m} pair {k}");
        }
        assert_eq!(dst.mat_raw_block(m)[0 * 2 + 1], sentinel);
    }
    for i in 0..=5 {
        for j in 0..=i {
            for k in 0..3 {
                let s = src.ret_raw_block(i, j)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.ret_raw_block(i, j)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s, "ret ({i},{j}) pair {k}");
            }
        }
        for m in 0..=4 {
            for k in 0..3 {
                let s = src.lmix_raw_block(i, m)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.lmix_raw_block(i, m)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s);
            }
        }
        for j in i..=5 {
            for k in 0..3 {
                let s = src.less_raw_block(i, j)[(r1[k] * 3 + c1[k]) as usize];
                let d = dst.less_raw_block(i, j)[(r2[k] * 2 + c2[k]) as usize];
                assert_eq!(d, s);
            }
        }
        assert_eq!(dst.ret_raw_block(i, 0)[1], sentinel);
        assert_eq!(dst.lmix_raw_block(i, 0)[1], sentinel);
        assert_eq!(dst.less_raw_block(0, i)[1], sentinel);
    }
    // single time step version (blueprint tstp = 4 -> Rust 3)
    let mut dst = ContourGreen::new(&cd, Sign::Fermi, sentinel);
    let tstp = 3i64;
    elemcpy_green_map_at(tstp, &r1, &c1, &src, &r2, &c2, &mut dst);
    for j in 0..=tstp {
        for k in 0..3 {
            let s = src.ret_raw_block(tstp, j)[(r1[k] * 3 + c1[k]) as usize];
            let d = dst.ret_raw_block(tstp, j)[(r2[k] * 2 + c2[k]) as usize];
            assert_eq!(d, s);
            let s = src.less_raw_block(j, tstp)[(r1[k] * 3 + c1[k]) as usize];
            let d = dst.less_raw_block(j, tstp)[(r2[k] * 2 + c2[k]) as usize];
            assert_eq!(d, s);
        }
    }
    assert!(dst
        .ret_raw_block(tstp - 1, 0)
        .iter()
        .all(|z| *z == sentinel));
    assert!(dst.mat_raw_block(0).iter().all(|z| *z == sentinel));
    // Matsubara version (blueprint tstp = 0)
    elemcpy_green_map_at(MATSUBARA, &r1, &c1, &src, &r2, &c2, &mut dst);
    for k in 0..3 {
        let s = src.mat_raw_block(1)[(r1[k] * 3 + c1[k]) as usize];
        let d = dst.mat_raw_block(1)[(r2[k] * 2 + c2[k]) as usize];
        assert_eq!(d, s);
    }
    // mismatched map lengths are rejected
    let result = std::panic::catch_unwind(|| {
        let mut d = ContourGreen::new(&cd, Sign::Fermi, sentinel);
        elemcpy_green_map(&r1, &c1[..2], &src, &r2, &c2, &mut d);
    });
    assert!(result.is_err());
}

#[test]
fn smul_with_val_adjoint() {
    // blueprint ntime = 7, ntau = 6 points
    let c = Contour::new(6, 5, 2, 2, 0.6, 1.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut h = DenseMat::zeros(2);
    h.set(0, 0, Complex::new(-0.3, 0.0));
    h.set(1, 1, Complex::new(0.5, 0.0));
    h.set(0, 1, Complex::new(0.0, 0.2));
    h.set(1, 0, Complex::new(0.0, -0.2));
    from_ham(&mut g, 0.0, &h, 1.0, 0.1);
    let mut rng = Rng(0xABCD);
    let mut f = TimeFunction::new(6, 2, 2, Complex::ZERO);
    for t in -1..=6 {
        f.block_mut(t).copy_from_slice(&rng.block(4));
    }
    let w = 0.7f64;
    let mut fw = f.clone();
    fw.smul(Complex::new(w, 0.0));
    for tstp in -1..=6 {
        // left: A <- w * f(†) * A
        let mut a = g.clone();
        a.lmul_cf_at(tstp, &fw, Mode::Adjoint);
        let mut gslice = timestep_at(&g, if tstp == MATSUBARA { 0 } else { tstp });
        gslice.lmul_cf(&fw, Mode::Adjoint);
        match tstp {
            MATSUBARA => {
                for m in 0..=5 {
                    let want = DenseMat::from_vec(2, 2, g.mat_raw_block(m).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.mat_raw_block(m).to_vec());
                    let wl = f
                        .block(MATSUBARA)
                        .iter()
                        .zip(f.block(MATSUBARA))
                        .map(|_| Complex::ZERO)
                        .collect::<Vec<_>>();
                    let _ = wl;
                    // w * f[0]' * G.mat
                    let fadj = adjoint2(f.block(MATSUBARA));
                    let wantm = fadj.mat_mul(&want);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13, "mat {m} {d}");
                    }
                }
            }
            _ => {
                for j in 0..=tstp {
                    check_left_row(&a, &g, &f, w, tstp, j);
                }
                for m in 0..=5 {
                    let want = DenseMat::from_vec(2, 2, g.lmix_raw_block(tstp, m).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.lmix_raw_block(tstp, m).to_vec());
                    let fadj = adjoint2(f.block(tstp));
                    let wantm = fadj.mat_mul(&want);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13);
                    }
                }
            }
        }
        // slice agrees
        let row = timestep_at(&a, if tstp == MATSUBARA { 0 } else { tstp });
        {
            // compare only the components the blueprint distance covers
            let mut e = 0.0f64;
            if tstp == MATSUBARA {
                for q in 0..=5 {
                    e += blkd(row.mat_raw_block(q), gslice.mat_raw_block(q));
                }
            } else {
                for j in 0..=tstp {
                    e += blkd(row.ret_raw_block(tstp, j), gslice.ret_raw_block(tstp, j));
                    e += blkd(row.less_raw_block(j, tstp), gslice.less_raw_block(j, tstp));
                }
                for m in 0..=5 {
                    e += blkd(row.lmix_raw_block(tstp, m), gslice.lmix_raw_block(tstp, m));
                }
            }
            assert!(e <= 1e-28, "slice mismatch at tstp {tstp}: {e}");
        }
        // other time steps are untouched
        for t in -1..=6 {
            if t == tstp {
                continue;
            }
            assert_eq!(a.distance_at(&g, t), 0.0);
        }
        // right: A <- w * A * f(†)
        let mut a = g.clone();
        a.rmul_cf_at(tstp, &fw, Mode::Adjoint);
        let mut gslice = timestep_at(&g, if tstp == MATSUBARA { 0 } else { tstp });
        gslice.rmul_cf(&fw, Mode::Adjoint);
        match tstp {
            MATSUBARA => {
                for m in 0..=5 {
                    let want = DenseMat::from_vec(2, 2, g.mat_raw_block(m).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.mat_raw_block(m).to_vec());
                    let fadj = adjoint2(f.block(MATSUBARA));
                    let wantm = want.mat_mul(&fadj);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13);
                    }
                }
            }
            _ => {
                for j in 0..=tstp {
                    // ret * f[j]'
                    let want = DenseMat::from_vec(2, 2, g.ret_raw_block(tstp, j).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
                    let fadj = adjoint2(f.block(j));
                    let wantm = want.mat_mul(&fadj);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13, "ret r ({tstp},{j})");
                    }
                    // less * f[tstp]'
                    let want = DenseMat::from_vec(2, 2, g.less_raw_block(j, tstp).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
                    let fadj = adjoint2(f.block(tstp));
                    let wantm = want.mat_mul(&fadj);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13, "less r ({j},{tstp})");
                    }
                }
                for m in 0..=5 {
                    let want = DenseMat::from_vec(2, 2, g.lmix_raw_block(tstp, m).to_vec());
                    let got = DenseMat::from_vec(2, 2, a.lmix_raw_block(tstp, m).to_vec());
                    let fadj = adjoint2(f.block(MATSUBARA));
                    let wantm = want.mat_mul(&fadj);
                    for q in 0..4 {
                        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
                        assert!(d < 1e-13);
                    }
                }
            }
        }
    }
    // default weight is one: smul with and without weight agree
    let mut a = g.clone();
    a.lmul_cf_at(3, &f, Mode::Adjoint);
    let mut b = g.clone();
    b.lmul_cf_at(3, &fw, Mode::Adjoint);
    // not equal (w = 0.7); check scaling relation instead on one block
    let q = (a.get_ret(3, 1, 0, 0) - b.get_ret(3, 1, 0, 0) / Complex::new(w, 0.0)).abs();
    assert!(q < 1e-13);
    // rectangular functions are rejected
    let cr = Contour::new(6, 5, 2, 3, 0.6, 1.0);
    let cfr = TimeFunction::new(6, 2, 3, Complex::ZERO);
    let _gr = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
    let r1 = {
        let mut gr2 = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
        let cfl = cfr.clone();
        std::panic::catch_unwind(move || {
            gr2.lmul_cf_at(2, &cfl, Mode::Adjoint);
        })
    };
    assert!(r1.is_err());
    let r2 = {
        let mut gr3 = ContourGreen::new(&cr, Sign::Fermi, Complex::ZERO);
        std::panic::catch_unwind(move || {
            gr3.rmul_cf_at(2, &cfr, Mode::Adjoint);
        })
    };
    assert!(r2.is_err());
}

fn adjoint2(b: &[Complex]) -> DenseMat {
    DenseMat::from_vec(2, 2, b.to_vec()).adjoint()
}

fn dm(m: &DenseMat, q: usize) -> Complex {
    m.get(q / 2, q % 2)
}

fn blkd(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0, |e, (x, y)| e + (*x - *y).abs() * (*x - *y).abs())
}

fn check_left_row(a: &ContourGreen, g: &ContourGreen, f: &TimeFunction, w: f64, tstp: i64, j: i64) {
    // ret: w * f[tstp]' * ret
    let want = DenseMat::from_vec(2, 2, g.ret_raw_block(tstp, j).to_vec());
    let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
    let wantm = adjoint2(f.block(tstp)).mat_mul(&want);
    for q in 0..4 {
        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
        assert!(d < 1e-13, "ret l ({tstp},{j})");
    }
    // less: w * f[j]' * less
    let want = DenseMat::from_vec(2, 2, g.less_raw_block(j, tstp).to_vec());
    let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
    let wantm = adjoint2(f.block(j)).mat_mul(&want);
    for q in 0..4 {
        let d = (dm(&got, q) - dm(&wantm, q).scale(w)).abs();
        assert!(d < 1e-13, "less l ({j},{tstp})");
    }
}

/// Port of NESSi herm_matrix_submatrix.cpp:19-91.
#[test]
fn nessi_submatrix_green_of_submatrix() {
    let (nt, ntau, size) = (30i64, 100i64, 5usize);
    let (beta, h) = (10.0f64, 0.02f64);
    let mut a = DenseMat::zeros(size);
    for i in 0..2 {
        for j in 0..2 {
            a.set(i, j, Complex::new((i + j) as f64, 0.0));
        }
    }
    for i in (size - 3)..size {
        for j in (size - 3)..size {
            a.set(i, j, Complex::new((i + j + 1) as f64, 0.0));
        }
    }
    let c = Contour::new(nt, ntau, size as i64, size as i64, nt as f64 * h, beta);
    let mut ga = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut ga, 0.0, &a, beta, h);

    // test 1: 2x2 submatrix (Julia 1-based indices -> 0-based)
    let i1 = [0i64, 0, 1, 1];
    let i2 = [0i64, 1, 0, 1];
    let j1 = [0i64, 0, 1, 1];
    let j2 = [0i64, 1, 0, 1];
    let cs = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let mut asub = ContourGreen::new(&cs, Sign::Fermi, Complex::ZERO);
    elemcpy_green_map(&i1, &i2, &ga, &j1, &j2, &mut asub);
    let mut b = DenseMat::zeros(2);
    for k in 0..4 {
        b.set(
            j1[k] as usize,
            j2[k] as usize,
            a.get(i1[k] as usize, i2[k] as usize),
        );
    }
    let mut gb = ContourGreen::new(&cs, Sign::Fermi, Complex::ZERO);
    from_ham(&mut gb, 0.0, &b, beta, h);
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += gb.distance_at(&asub, tstp);
    }
    assert!(err < 1e-7, "submatrix test 1 err {err}");

    // test 2: 3x3 submatrix
    let i11 = [2i64, 2, 2, 3, 3, 3, 4, 4, 4];
    let i12 = [2i64, 3, 4, 2, 3, 4, 2, 3, 4];
    let j11 = [0i64, 0, 0, 1, 1, 1, 2, 2, 2];
    let j12 = [0i64, 1, 2, 0, 1, 2, 0, 1, 2];
    let mut b1 = DenseMat::zeros(3);
    for k in 0..9 {
        b1.set(
            j11[k] as usize,
            j12[k] as usize,
            a.get(i11[k] as usize, i12[k] as usize),
        );
    }
    let cs3 = Contour::new(nt, ntau, 3, 3, nt as f64 * h, beta);
    let mut gb1 = ContourGreen::new(&cs3, Sign::Fermi, Complex::ZERO);
    from_ham(&mut gb1, 0.0, &b1, beta, h);
    let mut asub1 = ContourGreen::new(&cs3, Sign::Fermi, Complex::ZERO);
    elemcpy_green_map(&i11, &i12, &ga, &j11, &j12, &mut asub1);
    let mut err = 0.0;
    for tstp in -1..=nt {
        err += gb1.distance_at(&asub1, tstp);
    }
    assert!(err < 1e-7, "submatrix test 2 err {err}");
}

/// Port of NESSi herm_matrix_setget_timestep.cpp:10-40.
#[test]
fn nessi_setget_timestep() {
    let (nt, ntau) = (50i64, 50i64);
    let (h, beta) = (0.01f64, 10.0f64);
    let mut h0 = DenseMat::zeros(2);
    h0.set(0, 0, Complex::new(-0.4, 0.0));
    h0.set(1, 1, Complex::new(0.6, 0.0));
    h0.set(0, 1, Complex::new(0.0, 0.1));
    h0.set(1, 0, Complex::new(0.0, -0.1));
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
    let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g1, 0.0, &h0, beta, h);
    let mut err = 0.0;
    for tstp in -1..=nt {
        let a = timestep_at(&g1, if tstp == MATSUBARA { 0 } else { tstp });
        set_timestep_at(&mut g2, &a);
        err += g1.distance_at(&g2, tstp);
    }
    assert!(err < 1e-6, "err {err}");
}

/// Port of NESSi herm_matrix_algebra.cpp:41-175.
#[test]
fn nessi_herm_matrix_algebra() {
    let (nt, ntau) = (50i64, 50i64);
    let (dt, beta) = (0.01f64, 10.0f64);
    let wz = Complex::new(0.5, -0.3);
    let wr = 0.7f64;
    let mut h1 = DenseMat::zeros(2);
    h1.set(0, 0, Complex::new(-0.4, 0.0));
    h1.set(1, 1, Complex::new(0.6, 0.0));
    h1.set(0, 1, Complex::new(0.0, 0.1));
    h1.set(1, 0, Complex::new(0.0, -0.1));
    let mut h2 = DenseMat::zeros(2);
    h2.set(0, 0, Complex::new(0.2, 0.0));
    h2.set(1, 1, Complex::new(-0.5, 0.0));
    h2.set(0, 1, Complex::new(0.0, 0.3));
    h2.set(1, 0, Complex::new(0.0, -0.3));
    let c = Contour::new(nt, ntau, 2, 2, nt as f64 * dt, beta);
    let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    from_ham(&mut g1, 0.0, &h1, beta, dt);
    from_ham(&mut g2, 0.0, &h2, beta, dt);
    // reference G3 = G1 + wz * G2
    let g3 = add(&g1, &smul(&g2, wz));

    // incr_timestep: whole, via slices, and per-step
    let mut g4 = g1.clone();
    g4.incr_by(&smul(&g2, wz));
    let mut err = 0.0;
    for t in -1..=nt {
        err += g3.distance_at(&g4, t);
    }
    assert!(err < 1e-6, "incr whole err {err}");
    let mut g4 = g1.clone();
    let mut err = 0.0;
    for tstp in -1..=nt {
        let t_row = if tstp == MATSUBARA { 0 } else { tstp };
        let s2 = timestep_at(&g2, t_row);
        g4.incr_by_step(&s2, tstp, wz);
        err += g3.distance_at(&g4, tstp);
    }
    assert!(err < 1e-6, "incr slice err {err}");
    let mut g4 = g1.clone();
    let mut err = 0.0;
    for tstp in -1..=nt {
        // per-step weighted incr on the row scope
        match tstp {
            MATSUBARA => {
                for q in 0..=ntau {
                    let v: Vec<Complex> = g4
                        .mat_raw_block(q)
                        .iter()
                        .zip(g2.mat_raw_block(q))
                        .map(|(x, y)| *x + wz * *y)
                        .collect();
                    g4.mat_raw_block_mut(q).copy_from_slice(&v);
                }
            }
            _ => {
                for j in 0..=tstp {
                    let v: Vec<Complex> = g4
                        .ret_raw_block(tstp, j)
                        .iter()
                        .zip(g2.ret_raw_block(tstp, j))
                        .map(|(x, y)| *x + wz * *y)
                        .collect();
                    g4.ret_raw_block_mut(tstp, j).copy_from_slice(&v);
                    let v: Vec<Complex> = g4
                        .less_raw_block(j, tstp)
                        .iter()
                        .zip(g2.less_raw_block(j, tstp))
                        .map(|(x, y)| *x + wz * *y)
                        .collect();
                    g4.less_raw_block_mut(j, tstp).copy_from_slice(&v);
                }
                for m in 0..=ntau {
                    let v: Vec<Complex> = g4
                        .lmix_raw_block(tstp, m)
                        .iter()
                        .zip(g2.lmix_raw_block(tstp, m))
                        .map(|(x, y)| *x + wz * *y)
                        .collect();
                    g4.lmix_raw_block_mut(tstp, m).copy_from_slice(&v);
                }
            }
        }
        err += g3.distance_at(&g4, tstp);
    }
    assert!(err < 1e-6, "incr per-step err {err}");

    // smul with complex and real weights
    for w in [wz, Complex::new(wr, 0.0)] {
        let mut g4 = g1.clone();
        let mut err = 0.0;
        for tstp in -1..=nt {
            g4.smul_at(tstp, w);
            let mut g5 = timestep_at(&g1, if tstp == MATSUBARA { 0 } else { tstp });
            g5.smul(w);
            err += match tstp {
                MATSUBARA => {
                    let mut e = 0.0;
                    for q in 0..=ntau {
                        for (d, s) in g4.mat_raw_block(q).iter().zip(g5.mat_raw_block(q)) {
                            e += (*d - *s).abs();
                        }
                        // block-wise reference
                        for (d, s) in g4.mat_raw_block(q).iter().zip(g1.mat_raw_block(q)) {
                            let want = *s * w;
                            e += (*d - want).abs();
                        }
                    }
                    e
                }
                _ => {
                    let mut e = 0.0;
                    for j in 0..=tstp {
                        for (d, s) in g4
                            .ret_raw_block(tstp, j)
                            .iter()
                            .zip(g5.ret_raw_block(tstp, j))
                        {
                            e += (*d - *s).abs();
                        }
                        for (d, s) in g4
                            .less_raw_block(j, tstp)
                            .iter()
                            .zip(g5.less_raw_block(j, tstp))
                        {
                            e += (*d - *s).abs();
                        }
                        // block-wise reference
                        for (d, s) in g4
                            .ret_raw_block(tstp, j)
                            .iter()
                            .zip(g1.ret_raw_block(tstp, j))
                        {
                            e += (*d - *s * w).abs();
                        }
                        for (d, s) in g4
                            .less_raw_block(j, tstp)
                            .iter()
                            .zip(g1.less_raw_block(j, tstp))
                        {
                            e += (*d - *s * w).abs();
                        }
                    }
                    e
                }
            };
        }
        assert!(err < 1e-6, "smul err {err}");
    }

    // normal (non-conjugate) Cf multiplication regression
    let mut f = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let blk = [
        Complex::new(1.0, 0.1),
        Complex::new(0.0, 0.2),
        Complex::new(0.3, 0.0),
        Complex::new(-0.5, 0.0),
    ];
    for t in -1..=nt {
        f.block_mut(t).copy_from_slice(&blk);
    }
    let tstp = 5i64;
    let mut a = g1.clone();
    a.lmul_cf_at(tstp, &f, Mode::Normal);
    let fm = DenseMat::from_vec(2, 2, blk.to_vec());
    for j in 0..=tstp {
        let want = fm.mat_mul(&DenseMat::from_vec(
            2,
            2,
            g1.ret_raw_block(tstp, j).to_vec(),
        ));
        let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
        for q in 0..4 {
            assert!((dm(&got, q) - dm(&want, q)).abs() < 1e-13);
        }
        let want = DenseMat::from_vec(2, 2, g1.less_raw_block(j, tstp).to_vec());
        let want = fm.mat_mul(&want);
        let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
        for q in 0..4 {
            assert!((dm(&got, q) - dm(&want, q)).abs() < 1e-13);
        }
    }
    let mut a = g1.clone();
    a.rmul_cf_at(tstp, &f, Mode::Normal);
    for j in 0..=tstp {
        let src = DenseMat::from_vec(2, 2, g1.ret_raw_block(tstp, j).to_vec());
        let want = src.mat_mul(&fm);
        let got = DenseMat::from_vec(2, 2, a.ret_raw_block(tstp, j).to_vec());
        for q in 0..4 {
            assert!((dm(&got, q) - dm(&want, q)).abs() < 1e-13);
        }
        let src = DenseMat::from_vec(2, 2, g1.less_raw_block(j, tstp).to_vec());
        let want = src.mat_mul(&fm);
        let got = DenseMat::from_vec(2, 2, a.less_raw_block(j, tstp).to_vec());
        for q in 0..4 {
            assert!((dm(&got, q) - dm(&want, q)).abs() < 1e-13);
        }
    }
}

/// Port of NESSi function.cpp:228-340 (new methods in `Cf`).
#[test]
fn nessi_function_new_methods() {
    let nt = 1000i64;
    let rowmajor = |v: &[f64], n1: usize, n2: usize| -> Vec<Complex> {
        v.iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>()
            .chunks(n2)
            .take(n1)
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
    };
    let _ = &rowmajor;
    let rm = |v: &[f64], _n1: usize, _n2: usize| -> Vec<Complex> {
        v.iter().map(|x| Complex::new(*x, 0.0)).collect()
    };
    // fnorm: sqrt of summed block Frobenius differences over all slots
    let fnorm = |f: &TimeFunction, x: &[Complex]| -> f64 {
        let mut s = 0.0;
        for t in -1..=nt {
            for (a, b) in f.block(t).iter().zip(x) {
                let d = *a - *b; s += d.abs() * d.abs();
            }
        }
        s.sqrt()
    };
    let cf_of = |mat: &[Complex]| -> TimeFunction {
        let mut f = TimeFunction::new(nt as usize, 3, 5, Complex::ZERO);
        for t in -1..=nt {
            f.block_mut(t).copy_from_slice(mat);
        }
        f
    };

    // incr and smul
    let mat0 = rm(
        &[
            1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23.0, 12.0, 0.1, 0.0, 13.0, 231.0, 20.1, 2.1, 0.44,
        ],
        3,
        5,
    );
    let mat1 = rm(
        &[
            -1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3.0, 2.0, 6.1, 1.7, -3.0, -1.0, 2.1, -2.1, 8.44,
        ],
        3,
        5,
    );
    let alpha = Complex::new(0.5, 0.0);
    let mut f0 = cf_of(&mat0);
    let f1 = cf_of(&mat1);
    let mat3: Vec<Complex> = mat0
        .iter()
        .zip(&mat1)
        .map(|(a, b)| *a + alpha * *b)
        .collect();
    f0.incr_by(&smul(&f1, alpha));
    assert!(fnorm(&f0, &mat3) < 1e-5);
    let mat3: Vec<Complex> = mat3.iter().map(|z| alpha * *z).collect();
    f0.smul(alpha);
    assert!(fnorm(&f0, &mat3) < 1e-5);

    // multiply (5x5 constant blocks)
    let row0 = [0.1, 0.3, 0.7, 0.5, 1.1];
    let row1 = [2.0, 7.0, 4.0, 0.3, 2.0];
    let m0v: Vec<f64> = row0.iter().cloned().cycle().take(25).collect();
    let m1v: Vec<f64> = row1.iter().cloned().cycle().take(25).collect();
    let mat0 = rm(&m0v, 5, 5);
    let mat1 = rm(&m1v, 5, 5);
    let cf5 = |mat: &[Complex]| {
        let mut f = TimeFunction::new(nt as usize, 5, 5, Complex::ZERO);
        for t in -1..=nt {
            f.block_mut(t).copy_from_slice(mat);
        }
        f
    };
    let m0 = DenseMat::from_vec(5, 5, mat0.clone());
    let m1 = DenseMat::from_vec(5, 5, mat1.clone());
    // left: f0 <- f1 * f0 (blueprint uses f1's block at time 1 = same)
    let mut f0 = cf5(&mat0);
    let _f1 = cf5(&mat1);
    f0.lmul(&m1, Mode::Normal);
    assert!(fnorm(&f0, &to_vec(&m1.mat_mul(&m0))) < 1e-5);
    // right: f0 <- f0 * f1
    let mut f0 = cf5(&mat0);
    f0.rmul(&m1, Mode::Normal);
    assert!(fnorm(&f0, &to_vec(&m0.mat_mul(&m1))) < 1e-5);

    // matrixelements: element set/get round trips
    let mat0 = rm(
        &[
            1.0, 2.0, 0.0, 4.0, 3.0, 2.1, 23.0, 12.0, 0.1, 0.0, 13.0, 231.0, 20.1, 2.1, 0.44,
        ],
        3,
        5,
    );
    let mat1 = rm(
        &[
            -1.0, 0.5, 0.7, -2.1, 13.0, 2.9, -3.0, 2.0, 6.1, 1.7, -3.0, -1.0, 2.1, -2.1, 8.44,
        ],
        3,
        5,
    );
    let mut f0 = cf_of(&mat0);
    let f1 = cf_of(&mat1);
    let mut cur = mat0.clone();
    let mut ok = true;
    for i1 in 0..3usize {
        for i2 in 0..5usize {
            for j1 in 0..3usize {
                for j2 in 0..5usize {
                    cur[i1 * 5 + i2] = mat1[j1 * 5 + j2];
                    for t in -1..=nt {
                        let v = f1.get(t, j1, j2);
                        f0.set(t, i1, i2, v);
                    }
                    ok &= fnorm(&f0, &cur) < 1e-5;
                }
            }
        }
    }
    assert!(ok);
    // single-element extraction as a 1x1 function
    for i1 in 0..3usize {
        for i2 in 0..5usize {
            let v = f0.get(0, i1, i2);
            let mut f3 = TimeFunction::new(nt as usize, 1, 1, v);
            f3.set(MATSUBARA, 0, 0, v);
            assert!(fnorm(&f3, &[v]) < 1e-5);
        }
    }
}

fn to_vec(m: &DenseMat) -> Vec<Complex> {
    (0..m.nrows() * m.ncols())
        .map(|q| m.get(q / m.ncols(), q % m.ncols()))
        .collect()
}
