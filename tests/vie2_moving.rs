// Project : KadanoffBaym
// File    : tests/vie2_moving.rs
// Purpose : blueprint t_vie2_moving.jl: the truncated (moving) VIE2
//           G + F * G = Q
//
// NESSi has no moving VIE2; the truncated formulas are those of the
// full kernels without the mixed initial correlations. The random
// data use the deterministic splitmix64 generator (self-consistent).
// The alias check (Q === G) of the blueprint is subsumed by the Rust
// borrow checker (known deviation).

mod common;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::moving::window::init_from_g;
use kadanoffbaym::vie2::full::vie2;
use kadanoffbaym::dyson::mat::MatsubaraMethod;
use kadanoffbaym::vie2::moving::vie2_timestep_moving;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::ContourGreen;
use kadanoffbaym::types::moving::MovingGreen;
use kadanoffbaym::weights::Integrator;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

fn splitmix(state: &mut u64) -> u64 {
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_c64(state: &mut u64) -> Complex {
    let re = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let im = (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    Complex::new(re, im)
}

/// Full function with random real-time data, exponentially decaying in
/// t - t'. The Matsubara and mixing components are zero (blueprint
/// `vm_full`).
fn vm_full(cont: &Contour, state: &mut u64, scale: f64, les: bool) -> ContourGreen {
    let d = cont.size1 as usize;
    let ntime = cont.nt + 1; // KB point count
    let mut x = ContourGreen::new(cont, Sign::Fermi, Complex::ZERO);
    for n in 1..=ntime {
        for m in 1..=n {
            let tau = (n - m) as f64 * 0.05;
            let w = c(scale * (-tau).exp(), 0.0);
            let r: Vec<Complex> = (0..(d * d)).map(|_| rand_c64(state)).collect();
            for q in 0..(d * d) {
                x.ret.raw_block_at_mut(n - 1, m - 1)[q] = w * r[q];
            }
            let l: Vec<Complex> = if les {
                (0..(d * d)).map(|_| rand_c64(state)).collect()
            } else {
                vec![Complex::ZERO; d * d]
            };
            let ll = if m == n {
                let mut a = vec![Complex::ZERO; d * d];
                for i in 0..d {
                    for j in 0..d {
                        a[i * d + j] = (l[i * d + j] - l[j * d + i].conj()) / 2.0;
                    }
                }
                a
            } else {
                l
            };
            for q in 0..(d * d) {
                x.less.raw_block_at_mut(m - 1, n - 1)[q] = w * ll[q];
            }
        }
    }
    x
}

/// Kernel `F` and its hermitian conjugate partner (shared equal-time
/// lesser; blueprint `vm_kernel`).
fn vm_kernel(cont: &Contour, state: &mut u64) -> (ContourGreen, ContourGreen) {
    let d = cont.size1 as usize;
    let ntime = cont.nt + 1;
    let f = vm_full(cont, state, 0.3, true);
    let mut fcc = vm_full(cont, state, 0.3, true);
    for m in 0..=cont.nt {
        let src = f.less.raw_block_at(m, m).to_vec();
        let blk = fcc.less.raw_block_at_mut(m, m);
        for a in 0..d {
            for b in 0..d {
                blk[a * d + b] = -src[b * d + a].conj();
            }
        }
    }
    let _ = ntime;
    (f, fcc)
}

/// Moving copy of the full function `x` (partner `xcc`) at the 0-based
/// step `n0` (blueprint `vm_moving`).
fn vm_moving(x: &ContourGreen, xcc: &ContourGreen, tc: i64, n0: i64) -> MovingGreen {
    let mut m = MovingGreen::new(tc, x.ret.size1, x.sign);
    init_from_g(&mut m, x, xcc, n0);
    m
}

/// Leading row of the moving `gm` against the full `g` (blueprint
/// `vm_rowerr`).
fn vm_rowerr(gm: &MovingGreen, g: &ContourGreen, n: i64) -> (f64, f64) {
    let d = gm.ret.size1;
    let (mut er, mut el) = (0.0f64, 0.0f64);
    for j in 0..=gm.tc {
        if n - j < 0 {
            continue;
        }
        let gr = gm.ret_block(n, n - j);
        let fr = g.ret_block(n, n - j);
        for q in 0..(d * d) {
            er = er.max((gr[q] - fr.raw()[q]).abs());
        }
        let gl = gm.less_block(n - j, n);
        let fl = g.less_block(n - j, n);
        for q in 0..(d * d) {
            el = el.max((gl[q] - fl.raw()[q]).abs());
        }
    }
    (er, el)
}

#[test]
fn vie2_moving_zero_kernel_gives_q() {
    let (h, beta) = (0.05f64, 1.0);
    for &d in &[1usize, 2] {
        let (nt, tc, k) = (20i64, 9i64, 3i64);
        let mut state = 1u64;
        let cont = Contour::new(nt - 1, 4, d as i64, d as i64, (nt - 1) as f64 * h, beta);
        let q = vm_full(&cont, &mut state, 1.0, true);
        let z = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
        let n = nt - 1;
        let mut gm = vm_moving(&z, &z, tc, n);
        let fm = vm_moving(&z, &z, tc, n);
        let qm = vm_moving(&q, &q, tc, n);
        vie2_timestep_moving(n, &mut gm, &fm, &fm, &qm, h, k);
        let mut e = 0.0f64;
        for j in 0..=tc {
            for q in 0..(d * d) {
                e = e.max((gm.ret.mblock(0, j)[q] - qm.ret.mblock(0, j)[q]).abs());
                e = e.max((gm.less.mblock(0, j)[q] - qm.less.mblock(0, j)[q]).abs());
            }
        }
        assert!(e < 1e-13, "d = {d}: {e}");
    }
}

#[test]
fn vie2_moving_window_covers_history() {
    let (h, beta) = (0.05f64, 1.0);
    for k in 1..=5i64 {
        for &d in &[1usize, 2] {
            let nt = 2 * k + 6; // KB points
            let mut state = (10 * k + d as i64) as u64;
            // the Matsubara solver of order k needs enough points
            let cont = Contour::new(nt - 1, 12, d as i64, d as i64, (nt - 1) as f64 * h, beta);
            let (f, fcc) = vm_kernel(&cont, &mut state);
            let q = vm_full(&cont, &mut state, 1.0, true);
            let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
            let integ = Integrator::new(k);
            vie2(
                &mut g,
                &f,
                &fcc,
                &q,
                &integ,
                beta,
                h,
                MatsubaraMethod::Fixpoint,
            );
            // the full mixing components vanish for this input
            assert!(g.lmix.raw().iter().all(|z| *z == Complex::ZERO));
            for &n0 in &[(2 * k + 3), nt] {
                let n0_r = n0 - 1;
                let tc = n0_r;
                let mut gm = vm_moving(&g, &g, tc, n0_r);
                let fm = vm_moving(&f, &fcc, tc, n0_r);
                let fccm = vm_moving(&fcc, &f, tc, n0_r);
                let qm = vm_moving(&q, &q, tc, n0_r);
                vie2_timestep_moving(n0_r, &mut gm, &fm, &fccm, &qm, h, k);
                let (er, el) = vm_rowerr(&gm, &g, n0_r);
                assert!(er < 1e-10, "k = {k}, d = {d}: ret {er}");
                assert!(el < 1e-10, "k = {k}, d = {d}: less {el}");
            }
        }
    }
}

#[test]
fn vie2_moving_short_window_finite_history_unchanged() {
    let (h, beta) = (0.05f64, 1.0);
    let (d, nt, tc, k) = (2usize, 30i64, 9i64, 3i64);
    let mut state = 5u64;
    let cont = Contour::new(nt - 1, 4, d as i64, d as i64, (nt - 1) as f64 * h, beta);
    let (f, fcc) = vm_kernel(&cont, &mut state);
    let q = vm_full(&cont, &mut state, 1.0, true);
    let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    vie2(
        &mut g,
        &f,
        &fcc,
        &q,
        &Integrator::new(k),
        beta,
        h,
        MatsubaraMethod::Fixpoint,
    );
    let n = nt - 1;
    let mut gm = vm_moving(&g, &g, tc, n);
    let before = gm.ret.mblock(2, 4).to_vec();
    let fm = vm_moving(&f, &fcc, tc, n);
    let fccm = vm_moving(&fcc, &f, tc, n);
    let qm = vm_moving(&q, &q, tc, n);
    vie2_timestep_moving(n, &mut gm, &fm, &fccm, &qm, h, k);
    for j in 0..=tc {
        for z in gm.ret.mblock(0, j) {
            assert!(z.re.is_finite() && z.im.is_finite());
        }
        for z in gm.less.mblock(0, j) {
            assert!(z.re.is_finite() && z.im.is_finite());
        }
    }
    assert_eq!(gm.ret.mblock(2, 4), before);
    // the truncation changes the result
    let (er, el) = vm_rowerr(&gm, &g, n);
    assert!(er.max(el) > 1e-8, "er {er} el {el}");
}

#[test]
fn vie2_moving_nonzero_mixing_no_full_equivalence() {
    let (h, beta) = (0.05f64, 1.0);
    let (d, nt, k) = (2usize, 14i64, 2i64);
    let mut state = 7u64;
    let cont = Contour::new(nt - 1, 4, d as i64, d as i64, (nt - 1) as f64 * h, beta);
    let (mut f, mut fcc) = vm_kernel(&cont, &mut state);
    let mut q = vm_full(&cont, &mut state, 1.0, true);
    // nonzero mixing components of the kernel and the source
    for x in [&mut f, &mut fcc, &mut q] {
        for z in x.lmix.raw_mut().iter_mut() {
            *z = c(0.2, 0.0) * rand_c64(&mut state);
        }
    }
    let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
    vie2(
        &mut g,
        &f,
        &fcc,
        &q,
        &Integrator::new(k),
        beta,
        h,
        MatsubaraMethod::Fixpoint,
    );
    let lmax = g
        .lmix
        .raw()
        .iter()
        .map(|z| z.abs())
        .fold(0.0f64, f64::max);
    assert!(lmax > 0.01, "{lmax}");
    let n = nt - 1;
    let mut gm = vm_moving(&g, &g, n - 1, n);
    let fm = vm_moving(&f, &fcc, n - 1, n);
    let fccm = vm_moving(&fcc, &f, n - 1, n);
    let qm = vm_moving(&q, &q, n - 1, n);
    vie2_timestep_moving(n, &mut gm, &fm, &fccm, &qm, h, k);
    // the retarded component does not depend on the mixing components,
    // the lesser component misses the tv*vt term
    let (er, el) = vm_rowerr(&gm, &g, n);
    assert!(er < 1e-10, "ret {er}");
    assert!(el > 1e-6, "less {el}");
}

#[test]
fn vie2_moving_failure_paths() {
    let (h, beta) = (0.05f64, 1.0);
    let (d, nt) = (2usize, 20i64);
    let mut state = 9u64;
    let cont = Contour::new(nt - 1, 4, d as i64, d as i64, (nt - 1) as f64 * h, beta);
    let (f, fcc) = vm_kernel(&cont, &mut state);
    let q = vm_full(&cont, &mut state, 1.0, true);
    let n = nt - 1;
    let gm = vm_moving(&q, &q, 7, n);
    let fm = vm_moving(&f, &fcc, 7, n);
    let fccm = vm_moving(&fcc, &f, 7, n);
    let qm = vm_moving(&q, &q, 7, n);
    // tc >= 2k + 2: tc = 7 < 2*3 + 2
    let mut g1 = gm.clone();
    assert!(panics(|| {
        vie2_timestep_moving(n, &mut g1, &fm, &fccm, &qm, h, 3);
    }));
    // wrong leading time step
    let mut g2 = gm.clone();
    assert!(panics(|| {
        vie2_timestep_moving(n + 1, &mut g2, &fm, &fccm, &qm, h, 2);
    }));
    // (the alias check Q === G is subsumed by the borrow checker)
    // cut-off mismatch of Q
    let q8 = vm_moving(&q, &q, 8, n);
    let mut g3 = gm.clone();
    assert!(panics(|| {
        vie2_timestep_moving(n, &mut g3, &fm, &fccm, &q8, h, 2);
    }));
    // uninitialized G
    let mut gun = MovingGreen::new(7, d, Sign::Fermi);
    assert!(panics(|| {
        vie2_timestep_moving(n, &mut gun, &fm, &fccm, &qm, h, 2);
    }));
    // negative h
    let mut g4 = gm.clone();
    assert!(panics(|| {
        vie2_timestep_moving(n, &mut g4, &fm, &fccm, &qm, -h, 2);
    }));
}
