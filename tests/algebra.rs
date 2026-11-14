// Project : KadanoffBaym
// File    : tests/algebra.rs
// Purpose : port of julia/test/t_algebra.jl (basic algebra contracts)

mod common;

use common::prelude::*;
use kadanoffbaym::basics::{add, smul, smul_f64, sub, Algebra};
use kadanoffbaym::types::mat::MatComp;

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
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let cf1 = TimeFunction::new(200, 2, 2, v1);
    let cf2 = TimeFunction::new(200, 2, 2, v2);
    let cf3 = TimeFunction::new(200, 2, 2, v3);
    basic_algebra_checks(&cf1, &cf2, &cf3);
}

#[test]
fn mat_basic_algebra() {
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let m1 = MatComp::new(1000, 2, 2, v1);
    let m2 = MatComp::new(1000, 2, 2, v2);
    let m3 = MatComp::new(1000, 2, 2, v3);
    basic_algebra_checks(&m1, &m2, &m3);
}

#[test]
fn ret_basic_algebra() {
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let m1 = RetComp::new(200, 2, 2, v1);
    let m2 = RetComp::new(200, 2, 2, v2);
    let m3 = RetComp::new(200, 2, 2, v3);
    basic_algebra_checks(&m1, &m2, &m3);
}

#[test]
fn lmix_basic_algebra() {
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let m1 = LmixComp::new(200, 1000, 2, 2, v1);
    let m2 = LmixComp::new(200, 1000, 2, 2, v2);
    let m3 = LmixComp::new(200, 1000, 2, 2, v3);
    basic_algebra_checks(&m1, &m2, &m3);
}

#[test]
fn less_basic_algebra() {
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
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
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let m1 = GreenTimeStep::new(&c, 100, Sign::Fermi, v1);
    let m2 = GreenTimeStep::new(&c, 100, Sign::Fermi, v2);
    let m3 = GreenTimeStep::new(&c, 100, Sign::Fermi, v3);
    basic_algebra_checks(&m1, &m2, &m3);
}

#[test]
fn green_whole_basic_algebra() {
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let (v1, v2, v3) = (Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
    let m1 = ContourGreen::new(&c, Sign::Fermi, v1);
    let m2 = ContourGreen::new(&c, Sign::Fermi, v2);
    let m3 = ContourGreen::new(&c, Sign::Fermi, v3);
    basic_algebra_checks(&m1, &m2, &m3);
}

/// Port of NESSi herm_matrix_member.cpp:308-441 (non-in-place algebra
/// contracts exercised through the whole-object API).
#[test]
fn nessi_full_component_non_in_place_algebra() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    fill_algebra_green(&mut a, 0.0);
    fill_algebra_green(&mut b, 500.0);
    // specific non-hermitian overrides (Julia A.ret[3,1] / A.less[1,3])
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
    for tstp in -1..=(c.nt as i64) {
        complex_reference.smul_at(tstp, alpha);
        real_reference.smul_at(tstp, Complex::new(real_weight, 0.0));
    }

    let ntau = c.ntau as i64;
    let nt = c.nt as i64;
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
    // scalar multiplication commutes (blueprint `A.mat * α == α * A.mat`)
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
