mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::io::{
    green_to_string, pseudo_green_read_into, pseudo_green_to_string, read_pseudo_green_into,
    write_pseudo_green,
};
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::function::TimeFunction;
use kadanoffbaym::types::MATSUBARA;
use kadanoffbaym::basics::{pseudo_set_timestep, pseudo_timestep};
use kadanoffbaym::observables::{density_matrix_pseudo, density_matrix_pseudo_step, density_pseudo};
use kadanoffbaym::types::pseudo::PseudoGreen;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
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

fn rand_c64(state: &mut u64) -> Complex {
    let a = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let b = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(a * 4.0, b * 4.0)
}

fn contour() -> Contour {
    // the blueprint Cn(nt=6 points, ntau=7 points, 2, 2, 0.5, 2.0)
    Contour::new(5, 6, 2, 2, 0.5, 2.0)
}

/// Fill all the stored blocks with random numbers (blueprint `ps_fill!`).
fn ps_fill(g: &mut PseudoGreen, state: &mut u64) {
    let (nt, ntau) = (g.nt as i64, g.ntau as i64);
    let b = g.size1 * g.size2;
    for m in 0..=ntau {
        for q in 0..b {
            g.mat.block_mut(m as usize)[q] = rand_c64(state);
        }
    }
    for i in 0..=nt {
        for j in 0..=i {
            let v = rand_c64(state);
            g.ret.raw_block_at_mut(i, j).fill(v);
        }
        for m in 0..=ntau {
            let v = rand_c64(state);
            g.lmix.raw_block_at_mut(i, m).fill(v);
        }
        for j in i..=nt {
            let v = rand_c64(state);
            g.less.raw_block_at_mut(i, j).fill(v);
        }
    }
}

/// Elementwise equality of two pseudo-particle Green's functions.
fn eq_pseudo(a: &PseudoGreen, b: &PseudoGreen) -> bool {
    a.sign == b.sign
        && a.mat.raw() == b.mat.raw()
        && a.ret.raw() == b.ret.raw()
        && a.lmix.raw() == b.lmix.raw()
        && a.less.raw() == b.less.raw()
}

#[test]
fn pseudo_construction_and_equality() {
    let cc = contour();
    let g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    assert_eq!((g.size1, g.size2), (2, 2));
    assert_eq!(g.nt, 5); // 6 points
    assert_eq!(g.ntau, 6); // 7 points
    assert!(g.sign == Sign::Fermi);
    assert!(PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO).sign == Sign::Bose);
    let gv = PseudoGreen::new(&cc, Sign::Fermi, c(0.0, 0.5));
    assert!(gv.mat.block(2).iter().all(|z| *z == c(0.0, 0.5)));
    assert!(eq_pseudo(&gv, &PseudoGreen::new(&cc, Sign::Fermi, c(0.0, 0.5))));
    assert!(!eq_pseudo(&gv, &PseudoGreen::new(&cc, Sign::Fermi, c(0.0, 0.4))));
    // non-square matrices are rejected
    assert!(panics(|| {
        PseudoGreen::new(&Contour::new(5, 6, 2, 3, 0.5, 2.0), Sign::Fermi, Complex::ZERO)
    }));
    // the storage is the same as that of the standard Green
    let f = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    assert_eq!(g.ret.raw().len(), f.ret.raw().len());
    assert_eq!(g.less.raw().len(), f.less.raw().len());
    assert_eq!(g.lmix.raw().len(), f.lmix.raw().len());
    assert_eq!(g.mat.raw().len(), f.mat.raw().len());
}

#[test]
fn pseudo_greater_convention() {
    let cc = contour();
    let mut state = 42u64;
    let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    ps_fill(&mut g, &mut state);
    // the gtr property IS the stored ret component (structural in
    // Rust: there is no separate accessor, writing `gtr` is writing
    // `ret`); the other triangle is reconstructed by -adjoint
    let gc = g.to_contour();
    for i in 0..=5i64 {
        for j in (i + 1)..=5 {
            let a = gc.ret_block(i, j);
            let b = gc.ret_block(j, i);
            for q in 0..4 {
                let want = -(b.raw()[q].conj());
                assert_eq!(a.raw()[q], want);
            }
            let la = gc.less_block(i, j);
            let lb = gc.less_block(j, i);
            for q in 0..4 {
                assert_eq!(la.raw()[q], -(lb.raw()[q].conj()));
            }
        }
    }
    // the standard Green keeps gtr = less + ret
    let mut f = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    for i in 0..=5i64 {
        for j in 0..=i {
            let v = rand_c64(&mut state);
            f.ret.raw_block_at_mut(i, j).fill(v);
        }
        for j in i..=5 {
            let v = rand_c64(&mut state);
            f.less.raw_block_at_mut(i, j).fill(v);
        }
    }
    for i in 0..=5i64 {
        for j in 0..=i {
            let gt = f.gtr_block(i, j);
            let r = f.ret_block(i, j);
            let l = f.less_block(i, j);
            for q in 0..4 {
                assert_eq!(gt.raw()[q], r.raw()[q] + l.raw()[q]);
            }
        }
    }
    // the time step object shares the same convention
    let x = pseudo_timestep(&g, 2);
    assert_eq!(x.ret.raw_block_at(2, 2), gc.ret.raw_block_at(2, 2));
}

#[test]
fn pseudo_timestep_getset_and_distance() {
    let cc = contour();
    let mut state = 7u64;
    let mut g = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    ps_fill(&mut g, &mut state);
    let mut h = g.clone();
    for t in MATSUBARA..=5 {
        assert_eq!(g.distance_at(&h, t), 0.0);
    }
    for t in MATSUBARA..=5 {
        let x = pseudo_timestep(&g, t);
        if t != MATSUBARA {
            assert_eq!(x.tstp as i64, t);
        }
        assert_eq!(g.distance_at(&h, t), 0.0);
        h.zeros_step(t);
        assert!(t == MATSUBARA || g.distance_at(&h, t) > 0.0);
        pseudo_set_timestep(&mut h, t, &x);
        assert_eq!(g.distance_at(&h, t), 0.0);
    }
    assert!(eq_pseudo(&h, &g));
    // incompatible statistics
    assert!(!g.is_compatible(&PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO)));
    assert!(g.is_compatible(&PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO)));
}

#[test]
fn pseudo_traits() {
    let cc = contour();
    let mut state = 1u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    ps_fill(&mut a, &mut state);
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    ps_fill(&mut b, &mut state);
    // memcpy (the whole object)
    let x = a.clone();
    assert!(eq_pseudo(&x, &a));
    // memcpy at step 2
    let mut y = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    y.memcpy_step(&a, 2);
    assert_eq!(a.distance_at(&y, 2), 0.0);
    assert!(a.distance_at(&y, 1) > 0.0);
    // memset and zeros
    let mut z = a.clone();
    z.memset_step(1, c(0.0, 1.0));
    assert!(z.ret.raw_block_at(1, 0).iter().all(|q| *q == c(0.0, 1.0)));
    assert_eq!(z.distance_at(&a, 2), 0.0);
    assert_eq!(z.distance_at(&a, 0), 0.0);
    z.zeros_all();
    assert!(z.mat.raw().iter().all(|q| *q == Complex::ZERO));
    assert!(z.ret.raw().iter().all(|q| *q == Complex::ZERO));
    // incr: A + 2B, blockwise
    let mut s = a.clone();
    s.incr(&b, c(2.0, 0.0));
    for t in 0..=5i64 {
        for j in 0..=t {
            for q in 0..4 {
                assert_eq!(
                    s.ret.raw_block_at(t, j)[q],
                    a.ret.raw_block_at(t, j)[q] + c(2.0, 0.0) * b.ret.raw_block_at(t, j)[q]
                );
                assert_eq!(
                    s.less.raw_block_at(j, t)[q],
                    a.less.raw_block_at(j, t)[q] + c(2.0, 0.0) * b.less.raw_block_at(j, t)[q]
                );
            }
        }
        for m in 0..=6i64 {
            for q in 0..4 {
                assert_eq!(
                    s.lmix.raw_block_at(t, m)[q],
                    a.lmix.raw_block_at(t, m)[q] + c(2.0, 0.0) * b.lmix.raw_block_at(t, m)[q]
                );
            }
        }
    }
    for q in 0..4 {
        assert_eq!(
            s.mat.block(3)[q],
            a.mat.block(3)[q] + c(2.0, 0.0) * b.mat.block(3)[q]
        );
    }
    // smul
    let mut s2 = a.clone();
    s2.smul(c(0.0, 0.5));
    for q in 0..4 {
        assert_eq!(s2.ret.raw_block_at(4, 2)[q], c(0.0, 0.5) * a.ret.raw_block_at(4, 2)[q]);
        assert_eq!(s2.less.raw_block_at(1, 5)[q], c(0.0, 0.5) * a.less.raw_block_at(1, 5)[q]);
        assert_eq!(s2.lmix.raw_block_at(3, 6)[q], c(0.0, 0.5) * a.lmix.raw_block_at(3, 6)[q]);
        assert_eq!(s2.mat.block(1)[q], c(0.0, 0.5) * a.mat.block(1)[q]);
    }
}

#[test]
fn pseudo_multiplication_by_cf() {
    let cc = contour();
    let mut state = 3u64;
    let mut a = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    ps_fill(&mut a, &mut state);
    // f filled with random matrices (the Julia Cf(nt points))
    let mut f = TimeFunction::new(5, 2, 2, Complex::ZERO);
    for t in MATSUBARA..=5 {
        for q in 0..4 {
            f.block_mut(t)[q] = rand_c64(&mut state);
        }
    }
    // f must be non-commuting (2x2 with distinct values per step)
    // (each step is a constant multiple of one random value, so we
    // instead check the left/right products differ below)
    for t in MATSUBARA..=5 {
        let mut l = a.clone();
        let mut r = a.clone();
        l.lmul_cf(&f, t);
        r.rmul_cf(&f, t);
        if t == MATSUBARA {
            let f0 = f.block(MATSUBARA);
            for m in 0..=6i64 {
                let want = mul_scalar_mat(f0, &a.mat.block(m as usize));
                let want2 = mul_mat_scalar(&a.mat.block(m as usize), f0);
                for q in 0..4 {
                    assert_eq!(l.mat.block(m as usize)[q], want[q]);
                    assert_eq!(r.mat.block(m as usize)[q], want2[q]);
                }
            }
        } else {
            let f0 = f.block(MATSUBARA);
            let ft = f.block(t);
            for j in 0..=t {
                for q in 0..4 {
                    // L.ret[t,j] = f[t] * A.ret[t,j]
                    let want = mul_scalar_mat(ft, &a.ret.raw_block_at(t, j));
                    assert_eq!(l.ret.raw_block_at(t, j)[q], want[q]);
                    // R.ret[t,j] = A.ret[t,j] * f[j]
                    let want2 = mul_mat_scalar(&a.ret.raw_block_at(t, j), &f.block(j));
                    assert_eq!(r.ret.raw_block_at(t, j)[q], want2[q]);
                    // L.less[j,t] = f[j] * A.less[j,t]
                    let want3 = mul_scalar_mat(&f.block(j), &a.less.raw_block_at(j, t));
                    assert_eq!(l.less.raw_block_at(j, t)[q], want3[q]);
                    // R.less[j,t] = A.less[j,t] * f[t]
                    let want4 = mul_mat_scalar(&a.less.raw_block_at(j, t), ft);
                    assert_eq!(r.less.raw_block_at(j, t)[q], want4[q]);
                }
            }
            for m in 0..=6i64 {
                for q in 0..4 {
                    let want = mul_scalar_mat(ft, &a.lmix.raw_block_at(t, m));
                    assert_eq!(l.lmix.raw_block_at(t, m)[q], want[q]);
                    let want2 = mul_mat_scalar(&a.lmix.raw_block_at(t, m), f0);
                    assert_eq!(r.lmix.raw_block_at(t, m)[q], want2[q]);
                }
            }
            assert!(l.distance_at(&r, t) > 0.0);
        }
    }
}

/// f * M for a flat 2x2 block.
fn mul_scalar_mat(f: &[Complex], m: &[Complex]) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; 4];
    for i in 0..2 {
        for j in 0..2 {
            for l in 0..2 {
                out[i * 2 + j] += f[i * 2 + l] * m[l * 2 + j];
            }
        }
    }
    out
}

/// M * f for a flat 2x2 block.
fn mul_mat_scalar(m: &[Complex], f: &[Complex]) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; 4];
    for i in 0..2 {
        for j in 0..2 {
            for l in 0..2 {
                out[i * 2 + j] += m[i * 2 + l] * f[l * 2 + j];
            }
        }
    }
    out
}

#[test]
fn pseudo_density_matrix() {
    let cc = contour();
    for sign in [Sign::Fermi, Sign::Bose] {
        let mut state = 11u64;
        let mut g = PseudoGreen::new(&cc, sign, Complex::ZERO);
        ps_fill(&mut g, &mut state);
        // rho = -G^M(beta) on the Matsubara axis
        let rho0 = density_matrix_pseudo(&g, MATSUBARA);
        for q in 0..4 {
            assert_eq!(rho0[q], -(g.mat.block(6)[q]));
        }
        // rho = i xi G^<(t, t) at real time
        let xi = c(0.0, sign.as_i64() as f64);
        for n in 0..=5i64 {
            let rho = density_matrix_pseudo(&g, n);
            for q in 0..4 {
                assert_eq!(rho[q], xi * g.less.raw_block_at(n, n)[q]);
            }
            let x = pseudo_timestep(&g, n);
            let rho_step = density_matrix_pseudo_step(&x, n);
            for q in 0..4 {
                assert_eq!(rho_step[q], rho[q]);
            }
            assert_eq!(density_pseudo(&g, n), rho[0] + rho[3]);
        }
        // the step object at the Matsubara axis
        let x0 = pseudo_timestep(&g, MATSUBARA);
        let rho0s = density_matrix_pseudo_step(&x0, MATSUBARA);
        for q in 0..4 {
            assert_eq!(rho0s[q], rho0[q]);
        }
        // out-of-range and mismatched step object
        assert!(panics(|| density_matrix_pseudo(&g, 6)));
        assert!(panics(|| density_matrix_pseudo(&g, -2)));
        let x2 = pseudo_timestep(&g, 2);
        assert!(panics(|| density_matrix_pseudo_step(&x2, 3)));
    }
}

#[test]
fn pseudo_plain_text_io() {
    let cc = contour();
    let mut state = 13u64;
    let mut a = PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO);
    ps_fill(&mut a, &mut state);
    let text = pseudo_green_to_string(&a);
    assert!(text.lines().next().unwrap().contains("Pseudo"));
    let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    assert!(pseudo_green_read_into(&text, &mut b).is_ok());
    assert!(b.sign == Sign::Bose);
    for t in MATSUBARA..=5 {
        assert!(a.distance_at(&b, t) < 1e-10);
    }
    // a standard file can not be read into a pseudo-particle object
    let f = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    let std_text = green_to_string(&f);
    assert!(pseudo_green_read_into(&std_text, &mut b).is_err());
    // the sizes of the preallocated object are never changed
    let small = Contour::new(4, 6, 2, 2, 0.5, 2.0);
    let mut d = PseudoGreen::new(&small, Sign::Fermi, Complex::ZERO);
    assert!(pseudo_green_read_into(&text, &mut d).is_err());
    assert_eq!(d.nt, 4);
    // a missing file
    assert!(read_pseudo_green_into("/tmp/opencode/definitely-missing-pseudo.dat", &mut b).is_err());
    // the file wrappers round trip
    let path = "/tmp/opencode/pseudo_structs_io.dat";
    write_pseudo_green(path, &a).unwrap();
    let mut b2 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
    assert!(read_pseudo_green_into(path, &mut b2).is_ok());
    for t in MATSUBARA..=5 {
        assert!(b2.distance_at(&a, t) < 1e-10);
    }
}
