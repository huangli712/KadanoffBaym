// Project : KadanoffBaym
// File    : src/basics/algebra.rs
// Purpose : non-in-place algebra
//
// The in-place kernels live in `ops`; these are the owned counterparts
// (`add`, `sub`, `smul` returning new objects) for every type that
// implements `Algebra`.

use crate::complex::Complex;
use crate::types::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

/// In-place kernels required by the owned algebra.
pub trait Algebra: Clone {
    /// `self += other`.
    fn incr_by(&mut self, other: &Self);

    /// `self *= alpha`.
    fn smul_c(&mut self, alpha: Complex);
}

impl Algebra for ContourGreen {
    fn incr_by(&mut self, other: &Self) {
        ContourGreen::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        ContourGreen::smul(self, alpha);
    }
}

impl Algebra for GreenTimeStep {
    fn incr_by(&mut self, other: &Self) {
        GreenTimeStep::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        GreenTimeStep::smul(self, alpha);
    }
}

impl Algebra for TimeFunction {
    fn incr_by(&mut self, other: &Self) {
        TimeFunction::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        TimeFunction::smul(self, alpha);
    }
}

impl Algebra for MatComp {
    fn incr_by(&mut self, other: &Self) {
        MatComp::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        MatComp::smul(self, alpha);
    }
}

impl Algebra for RetComp {
    fn incr_by(&mut self, other: &Self) {
        RetComp::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        RetComp::smul(self, alpha);
    }
}

impl Algebra for LmixComp {
    fn incr_by(&mut self, other: &Self) {
        LmixComp::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        LmixComp::smul(self, alpha);
    }
}

impl Algebra for LessComp {
    fn incr_by(&mut self, other: &Self) {
        LessComp::incr_by(self, other);
    }

    fn smul_c(&mut self, alpha: Complex) {
        LessComp::smul(self, alpha);
    }
}

/// `a + b` (new object).
pub fn add<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = a.clone();
    out.incr_by(b);
    out
}

/// `a - b` (new object).
pub fn sub<T: Algebra>(a: &T, b: &T) -> T {
    let mut out = b.clone();
    out.smul_c(Complex::new(-1.0, 0.0));
    out.incr_by(a);
    out
}

/// `alpha * a` (new object), complex scalar.
pub fn smul<T: Algebra>(a: &T, alpha: Complex) -> T {
    let mut out = a.clone();
    out.smul_c(alpha);
    out
}

/// `alpha * a` (new object), real scalar.
pub fn smul_f64<T: Algebra>(a: &T, alpha: f64) -> T {
    smul(a, Complex::new(alpha, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::DenseMat;
    use crate::linalg::Mode;
    use crate::types::{Contour, Sign};

    use crate::types::MATSUBARA;

    use crate::basics::timestep_at;
    use crate::equilibrium::from_ham;

    fn dm(m: &DenseMat, q: usize) -> Complex {
        m.get(q / 2, q % 2)
    }

    /// Blueprint `algebra_block(seed)` as a row-major 2x2 block.
    fn algebra_block(seed: f64) -> [Complex; 4] {
        [
            Complex::new(seed, 1.0),
            Complex::new(-2.0, seed + 1.0),
            Complex::new(3.0, -(seed + 2.0)),
            Complex::new(seed / 2.0, -4.0),
        ]
    }

    fn fill_algebra_green(g: &mut ContourGreen, offset: f64) {
        let (nt, ntau) = (g.nt as i64, g.ntau as i64);
        for q in 0..=ntau {
            g.mat_raw_block_mut(q)
                .copy_from_slice(&algebra_block(offset + 11.0 + q as f64));
        }
        for i in 0..=nt {
            for j in 0..=i {
                g.ret_raw_block_mut(i, j)
                    .copy_from_slice(&algebra_block(offset + 111.0 + 10.0 * i as f64 + j as f64));
            }
            for q in 0..=ntau {
                g.lmix_raw_block_mut(i, q)
                    .copy_from_slice(&algebra_block(offset + 211.0 + 10.0 * i as f64 + q as f64));
            }
            for j in i..=nt {
                g.less_raw_block_mut(i, j)
                    .copy_from_slice(&algebra_block(offset + 311.0 + 10.0 * i as f64 + j as f64));
            }
        }
    }

    /// The basic algebra round-trip checks shared by every container type.
    fn basic_algebra_checks<T>(m1: &T, m2: &T, m3: &T)
    where
        T: Algebra + PartialEq,
    {
        assert!(add(m1, m2) == *m3);
        assert!(sub(m3, m1) == *m2);
        assert!(sub(m3, m2) == *m1);
        assert!(smul_f64(m1, 3.0) == *m3);
        assert!(smul_f64(m1, 2.0) == *m2);
        let combined = add(&smul_f64(m2, 2.0), &smul_f64(m1, -1.0));
        assert!(combined == *m3);
    }

    #[test]
    fn cf_basic_algebra() {
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let cf1 = TimeFunction::new(200, 2, 2, v1);
        let cf2 = TimeFunction::new(200, 2, 2, v2);
        let cf3 = TimeFunction::new(200, 2, 2, v3);
        basic_algebra_checks(&cf1, &cf2, &cf3);
    }

    #[test]
    fn mat_basic_algebra() {
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = MatComp::new(1000, 2, 2, v1);
        let m2 = MatComp::new(1000, 2, 2, v2);
        let m3 = MatComp::new(1000, 2, 2, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn ret_basic_algebra() {
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = RetComp::new(200, 2, 2, v1);
        let m2 = RetComp::new(200, 2, 2, v2);
        let m3 = RetComp::new(200, 2, 2, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn lmix_basic_algebra() {
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = LmixComp::new(200, 1000, 2, 2, v1);
        let m2 = LmixComp::new(200, 1000, 2, 2, v2);
        let m3 = LmixComp::new(200, 1000, 2, 2, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn less_basic_algebra() {
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = LessComp::new(200, 2, 2, v1);
        let m2 = LessComp::new(200, 2, 2, v2);
        let m3 = LessComp::new(200, 2, 2, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn green_timestep_basic_algebra() {
        // g* variants: whole GreenTimeStep algebra (covers the timestep
        // component objects)
        let c = Contour::new(100, 1000, 2, 2, 5.0, 4.0);
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = GreenTimeStep::new(&c, 100, Sign::Fermi, v1);
        let m2 = GreenTimeStep::new(&c, 100, Sign::Fermi, v2);
        let m3 = GreenTimeStep::new(&c, 100, Sign::Fermi, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn green_whole_basic_algebra() {
        let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
        let (v1, v2, v3) = (
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        );
        let m1 = ContourGreen::new(&c, Sign::Fermi, v1);
        let m2 = ContourGreen::new(&c, Sign::Fermi, v2);
        let m3 = ContourGreen::new(&c, Sign::Fermi, v3);
        basic_algebra_checks(&m1, &m2, &m3);
    }

    #[test]
    fn green_add_sub_scale() {
        let c = Contour::new(2, 2, 1, 1, 1.0, 1.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        a.set_mat(0, 0, 0, Complex::new(1.0, 0.0));
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        b.set_mat(0, 0, 0, Complex::new(0.0, 1.0));
        let s = add(&a, &b);
        assert_eq!(s.get_mat(0, 0, 0), Complex::new(1.0, 1.0));
        let d = sub(&a, &b);
        assert_eq!(d.get_mat(0, 0, 0), Complex::new(1.0, -1.0));
        let m = smul(&a, Complex::new(0.0, 2.0));
        assert_eq!(m.get_mat(0, 0, 0), Complex::new(0.0, 2.0));
        let m2 = smul_f64(&a, 3.0);
        assert_eq!(m2.get_mat(0, 0, 0), Complex::new(3.0, 0.0));
        // originals untouched
        assert_eq!(a.get_mat(0, 0, 0), Complex::new(1.0, 0.0));
        assert_eq!(b.get_mat(0, 0, 0), Complex::new(0.0, 1.0));
    }

    /// Non-in-place algebra contracts exercised through the whole-object
    /// API.
    #[test]
    fn full_component_non_in_place_algebra() {
        let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
        let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        fill_algebra_green(&mut a, 0.0);
        fill_algebra_green(&mut b, 500.0);
        // specific non-hermitian overrides
        let ar = [
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 1.0),
            Complex::new(4.0, -2.0),
            Complex::new(0.5, 3.0),
        ];
        let br = [
            Complex::new(-2.0, 1.0),
            Complex::new(5.0, -4.0),
            Complex::new(1.5, 2.0),
            Complex::new(3.0, -1.0),
        ];
        let al = [
            Complex::new(2.0, -3.0),
            Complex::new(1.0, 4.0),
            Complex::new(-5.0, 2.0),
            Complex::new(0.25, 1.0),
        ];
        let bl = [
            Complex::new(1.0, 1.0),
            Complex::new(-2.0, 0.5),
            Complex::new(3.0, -4.0),
            Complex::new(-1.0, 2.0),
        ];
        a.ret_raw_block_mut(2, 0).copy_from_slice(&ar);
        b.ret_raw_block_mut(2, 0).copy_from_slice(&br);
        a.less_raw_block_mut(0, 2).copy_from_slice(&al);
        b.less_raw_block_mut(0, 2).copy_from_slice(&bl);
        // non-hermitian sanity: ret(2,0) differs from its transpose and the
        // products do not commute
        let arm = DenseMat::from_vec(2, 2, ar.to_vec());
        let brm = DenseMat::from_vec(2, 2, br.to_vec());
        let t = arm.adjoint();
        assert_ne!(arm, t);
        let ab = arm.mat_mul(&brm);
        let ba = brm.mat_mul(&arm);
        assert_ne!(ab, ba);

        let a_original = a.clone();
        let b_original = b.clone();
        let alpha = Complex::new(0.75, -0.5);
        let real_weight = 0.25;

        let sum_result = add(&a, &b);
        let difference_result = sub(&a, &b);
        let complex_result = smul(&a, alpha);
        let real_result = smul_f64(&a, real_weight);

        // references built through the in-place ops (incr!/smul!)
        let mut sum_reference = a.clone();
        sum_reference.incr_by(&b);
        let mut difference_reference = a.clone();
        difference_reference.incr_by(&smul_f64(&b, -1.0));
        let mut complex_reference = a.clone();
        let mut real_reference = a.clone();
        for tstp in -1..=c.nt {
            complex_reference.smul_at(tstp, alpha);
            real_reference.smul_at(tstp, Complex::new(real_weight, 0.0));
        }

        let ntau = c.ntau;
        let nt = c.nt;
        for q in 0..=ntau {
            assert_eq!(
                sum_result.mat_raw_block(q)[0],
                a.mat_raw_block(q)[0] + b.mat_raw_block(q)[0]
            );
            assert_eq!(
                difference_result.mat_raw_block(q)[3],
                a.mat_raw_block(q)[3] - b.mat_raw_block(q)[3]
            );
            assert_eq!(
                complex_result.mat_raw_block(q)[1],
                alpha * a.mat_raw_block(q)[1]
            );
            assert_eq!(
                real_result.mat_raw_block(q)[2],
                Complex::new(real_weight, 0.0) * a.mat_raw_block(q)[2]
            );
        }
        for i in 0..=nt {
            for j in 0..=i {
                assert_eq!(
                    sum_result.ret_raw_block(i, j)[0],
                    a.ret_raw_block(i, j)[0] + b.ret_raw_block(i, j)[0]
                );
                assert_eq!(
                    difference_result.less_raw_block(j, i)[3],
                    a.less_raw_block(j, i)[3] - b.less_raw_block(j, i)[3]
                );
                assert_eq!(
                    complex_result.ret_raw_block(i, j)[1],
                    alpha * a.ret_raw_block(i, j)[1]
                );
                assert_eq!(
                    real_result.less_raw_block(j, i)[2],
                    Complex::new(real_weight, 0.0) * a.less_raw_block(j, i)[2]
                );
            }
            for q in 0..=ntau {
                assert_eq!(
                    sum_result.lmix_raw_block(i, q)[0],
                    a.lmix_raw_block(i, q)[0] + b.lmix_raw_block(i, q)[0]
                );
            }
        }

        // results agree with the in-place references component by component
        for (result, reference) in [
            (&sum_result, &sum_reference),
            (&difference_result, &difference_reference),
            (&complex_result, &complex_reference),
            (&real_result, &real_reference),
        ] {
            for q in 0..=ntau {
                assert_eq!(result.mat_raw_block(q), reference.mat_raw_block(q));
            }
            for i in 0..=nt {
                for j in 0..=i {
                    assert_eq!(result.ret_raw_block(i, j), reference.ret_raw_block(i, j));
                    assert_eq!(result.less_raw_block(j, i), reference.less_raw_block(j, i));
                }
                // hermiticity fallback of the results is preserved
                for j in 0..i {
                    let mut want = result.ret_block(i, j).adjoint();
                    want.scale(-1.0);
                    let mirror = result.ret_block(j, i);
                    assert_eq!(mirror, want);
                    let mut wantl = result.less_block(j, i).adjoint();
                    wantl.scale(-1.0);
                    let ml = result.less_block(i, j);
                    assert_eq!(ml, wantl);
                }
                for q in 0..=ntau {
                    assert_eq!(result.lmix_raw_block(i, q), reference.lmix_raw_block(i, q));
                }
            }
        }
        // scalar multiplication commutes
        let m1 = smul(&a.mat, alpha);
        let m2 = {
            let mut mm = a.mat.clone();
            mm.smul(alpha);
            mm
        };
        assert!(m1.equal(&m2));
        // originals untouched
        assert!(a.equal(&a_original));
        assert!(b.equal(&b_original));
    }

    #[test]
    fn time_function_algebra() {
        let mut a = TimeFunction::new(1, 1, 1, Complex::ZERO);
        a.set(0, 0, 0, Complex::new(2.0, 0.0));
        let b = add(&a, &a);
        assert_eq!(b.get(0, 0, 0), Complex::new(4.0, 0.0));
        let d = sub(&a, &a);
        assert_eq!(d.get(0, 0, 0), Complex::ZERO);
    }

    /// Element-wise algebra and scalar multiplication over the whole object.
    #[test]
    fn herm_matrix_algebra() {
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
}
