// Project : KadanoffBaym
// File    : tests/moving_bubble.rs
// Purpose : blueprint t_moving_bubble.jl: the moving Bubble1/Bubble2
//           diagrams against the full contour bubbles
//
// The full Green's functions have zero mixing components (the moving
// types do not store them), so the real-time components of the full
// and the moving bubbles agree. The random data use the deterministic
// splitmix64 generator (self-consistent).

mod common;

use common::*;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::bubble::{
    bubble1, bubble1_moving, bubble1_moving_herm, bubble2, bubble2_moving, bubble2_moving_herm,
};
use kadanoffbaym::moving::window::{forward, init_from_g, init_from_g_herm};
use kadanoffbaym::types::contour::{Contour, Sign};
use kadanoffbaym::types::green::ContourGreen;
use kadanoffbaym::types::moving::MovingGreen;

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

/// Full Green's function with fixed random real-time data, the
/// Matsubara and mixing components are zero (blueprint `mb_full`).
/// `nt_points` is the blueprint point count.
fn mb_full(nt_points: i64, d: usize, sign: Sign, seed: u64) -> ContourGreen {
    let mut state = seed;
    let cont = Contour::new(
        nt_points - 1,
        3,
        d as i64,
        d as i64,
        (nt_points - 1) as f64 * 0.1,
        1.0,
    );
    let mut f = ContourGreen::new(&cont, sign, Complex::ZERO);
    for n in 1..=nt_points {
        for m in 1..=n {
            let r: Vec<Complex> = (0..(d * d)).map(|_| rand_c64(&mut state)).collect();
            let rr = if m == n {
                let mut a = r.clone();
                for i in 0..d {
                    for j in 0..d {
                        a[i * d + j] = (r[i * d + j] - r[j * d + i].conj()) / 2.0;
                    }
                }
                a
            } else {
                r
            };
            f.ret.raw_block_at_mut(n - 1, m - 1).copy_from_slice(&rr);
            let l: Vec<Complex> = (0..(d * d)).map(|_| rand_c64(&mut state)).collect();
            let ll = if m == n {
                let mut a = l.clone();
                for i in 0..d {
                    for j in 0..d {
                        a[i * d + j] = (l[i * d + j] - l[j * d + i].conj()) / 2.0;
                    }
                }
                a
            } else {
                l
            };
            f.less.raw_block_at_mut(m - 1, n - 1).copy_from_slice(&ll);
        }
    }
    f
}

/// Moving copy of the full function `x` (hermitian partner `xcc`) with
/// the leading 0-based time step `n` (blueprint `mb_moving`).
fn mb_moving(x: &ContourGreen, xcc: &ContourGreen, tc: i64, n: i64) -> MovingGreen {
    let mut g = MovingGreen::new(tc, x.ret.size1, x.sign);
    init_from_g(&mut g, x, xcc, n);
    g
}

/// Fill the leading row of the moving function from the full
/// functions, the same rule as `init_from_g` (blueprint
/// `mb_fillrow!`).
fn mb_fillrow(xm: &mut MovingGreen, x: &ContourGreen, xcc: &ContourGreen) {
    let n = xm.tstp;
    let d = xm.ret.size1;
    for j in 0..=xm.tc.min(n) {
        let r = x.ret.raw_block_at(n, n - j);
        xm.ret.mblock_mut(0, j).copy_from_slice(r);
        let s = xcc.less.raw_block_at(n - j, n);
        let blk = xm.less.mblock_mut(0, j);
        for a in 0..d {
            for b in 0..d {
                blk[a * d + b] = -s[b * d + a].conj();
            }
        }
    }
}

/// Maximum difference between the leading row of the moving `cm` and
/// the full contour `cf` at the 0-based time step `n` (the lesser only
/// for hermitian inputs; blueprint `mb_rowdiff`).
fn mb_rowdiff(cm: &MovingGreen, cf: &ContourGreen, n: i64, less: bool) -> f64 {
    let d = cm.ret.size1;
    let mut e = 0.0f64;
    for j in 0..=cm.tc {
        if n - j < 0 {
            continue;
        }
        let gr = cm.ret_block(n, n - j);
        let fr = cf.ret_block(n, n - j);
        for q in 0..(d * d) {
            e = e.max((gr[q] - fr.raw()[q]).abs());
        }
        if !less {
            continue;
        }
        let gl = cm.less_block(n - j, n);
        let fl = cf.less_block(n - j, n);
        for q in 0..(d * d) {
            e = e.max((gl[q] - fl.raw()[q]).abs());
        }
    }
    e
}

fn sign_prod(sa: Sign, sb: Sign) -> Sign {
    // NESSi convention: Fermi = -1, Bose = +1; the product +1 is Bose
    if sa.as_i64() * sb.as_i64() == -1 {
        Sign::Fermi
    } else {
        Sign::Bose
    }
}

#[allow(clippy::too_many_arguments)]
fn step_bubble(
    name: i32,
    cm: &mut MovingGreen,
    t: i64,
    d: usize,
    imap: i32,
    am: &MovingGreen,
    accm: &MovingGreen,
    bm: &MovingGreen,
    bccm: &MovingGreen,
) {
    for c1 in 0..d {
        for c2 in 0..d {
            let idx: [usize; 4] = if imap == 0 {
                [d - 1 - c1, d - 1 - c2, c1, c2]
            } else {
                [d - 1 - c1, c2, c1, d - 1 - c2]
            };
            let (a1, a2, b1, b2) = (idx[0], idx[1], idx[2], idx[3]);
            if name == 0 {
                bubble1_moving(t, cm, c1, c2, am, accm, a1, a2, bm, bccm, b1, b2);
            } else {
                bubble2_moving(t, cm, c1, c2, am, accm, a1, a2, bm, bccm, b1, b2);
            }
        }
    }
}

#[test]
fn moving_bubble_full_equivalence() {
    for name in [0i32, 1] {
        for &d in &[1usize, 2] {
            for &(sa, sb) in &[
                (Sign::Fermi, Sign::Fermi),
                (Sign::Fermi, Sign::Bose),
                (Sign::Bose, Sign::Bose),
            ] {
                for &tc in &[3i64, 6] {
                    for &herm in &[true, false] {
                        for &imap in &[0i32, 1] {
                            let nt_points = 12i64;
                            let n = 7i64; // my step of the blueprint KB 8
                            let a_full = mb_full(nt_points, d, sa, 1);
                            let b_full = mb_full(nt_points, d, sb, 3);
                            let acc_full = if herm {
                                a_full.clone()
                            } else {
                                mb_full(nt_points, d, sa, 2)
                            };
                            let bcc_full = if herm {
                                b_full.clone()
                            } else {
                                mb_full(nt_points, d, sb, 4)
                            };
                            let sp = sign_prod(sa, sb);
                            let cont = Contour::new(
                                nt_points - 1,
                                3,
                                d as i64,
                                d as i64,
                                (nt_points - 1) as f64 * 0.1,
                                1.0,
                            );
                            let mut cf = ContourGreen::new(&cont, sp, Complex::ZERO);
                            let cmpless = herm && imap == 0;
                            for m in 0..=(nt_points - 1) {
                                for c1 in 0..d {
                                    for c2 in 0..d {
                                        let idx: [usize; 4] = if imap == 0 {
                                            [d - 1 - c1, d - 1 - c2, c1, c2]
                                        } else {
                                            [d - 1 - c1, c2, c1, d - 1 - c2]
                                        };
                                        let (a1, a2, b1, b2) =
                                            (idx[0], idx[1], idx[2], idx[3]);
                                        if name == 0 {
                                            bubble1(
                                                m,
                                                &mut cf,
                                                c1,
                                                c2,
                                                &a_full,
                                                &acc_full,
                                                a1,
                                                a2,
                                                &b_full,
                                                &bcc_full,
                                                b1,
                                                b2,
                                            );
                                        } else {
                                            bubble2(
                                                m,
                                                &mut cf,
                                                c1,
                                                c2,
                                                &a_full,
                                                &acc_full,
                                                a1,
                                                a2,
                                                &b_full,
                                                &bcc_full,
                                                b1,
                                                b2,
                                            );
                                        }
                                    }
                                }
                            }
                            let mut am = mb_moving(&a_full, &acc_full, tc, n);
                            let mut accm = mb_moving(&acc_full, &a_full, tc, n);
                            let mut bm = mb_moving(&b_full, &bcc_full, tc, n);
                            let mut bccm = mb_moving(&bcc_full, &b_full, tc, n);
                            let mut cm = MovingGreen::new(tc, d, sp);
                            let empty = ContourGreen::new(&cont, sp, Complex::ZERO);
                            init_from_g_herm(&mut cm, &empty, n);
                            step_bubble(name, &mut cm, n, d, imap, &am, &accm, &bm, &bccm);
                            let e0 = mb_rowdiff(&cm, &cf, n, cmpless);
                            assert!(e0 < 1e-12, "row {n}: {e0}");
                            // two forward! steps
                            for s in 1..=2i64 {
                                let pairs: Vec<(
                                    &mut MovingGreen,
                                    &ContourGreen,
                                    &ContourGreen,
                                )> = vec![
                                    (&mut am, &a_full, &acc_full),
                                    (&mut accm, &acc_full, &a_full),
                                    (&mut bm, &b_full, &bcc_full),
                                    (&mut bccm, &bcc_full, &b_full),
                                ];
                                for (xm, xf, xccf) in pairs {
                                    forward(xm);
                                    mb_fillrow(xm, xf, xccf);
                                }
                                forward(&mut cm);
                                step_bubble(
                                    name,
                                    &mut cm,
                                    n + s,
                                    d,
                                    imap,
                                    &am,
                                    &accm,
                                    &bm,
                                    &bccm,
                                );
                                let e = mb_rowdiff(&cm, &cf, n + s, cmpless);
                                assert!(e < 1e-12, "row {}: {e}", n + s);
                            }
                            let eh = mb_rowdiff(&cm, &cf, n, cmpless);
                            assert!(eh < 1e-12, "history {n}: {eh}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn moving_bubble_failure_paths() {
    let (d, tc, n) = (2usize, 3i64, 5i64);
    let ffull = mb_full(8, d, Sign::Fermi, 1);
    let bfull = mb_full(8, d, Sign::Bose, 2);
    let mut a = mb_moving(&ffull, &ffull, tc, n);
    let mut cb = mb_moving(&bfull, &bfull, tc, n);
    let mut cf = mb_moving(&ffull, &ffull, tc, n);
    let mut a_tc = mb_moving(&ffull, &ffull, tc + 1, n);
    // wrong leading time step
    assert!(panics(|| {
        bubble1_moving_herm(n + 1, &mut cb, 1, 1, &a, 0, 0, &a, 0, 0);
    }));
    // statistics mismatch
    assert!(panics(|| {
        let mut am = mb_moving(&ffull, &ffull, tc, n);
        bubble1_moving_herm(n, &mut am, 1, 1, &a, 0, 0, &a, 0, 0);
    }));
    assert!(panics(|| {
        bubble2_moving_herm(n, &mut cf, 1, 1, &a, 0, 0, &a, 0, 0);
    }));
    // orbital index out of bounds
    assert!(panics(|| {
        bubble2_moving(n, &mut cb, 3, 1, &a, &a, 1, 1, &a, &a, 1, 1);
    }));
    // cut-off mismatch
    assert!(panics(|| {
        bubble1_moving_herm(n, &mut cb, 1, 1, &a, 0, 0, &a_tc, 0, 0);
    }));
    // uninitialized partner
    assert!(panics(|| {
        let un = MovingGreen::new(tc, d, Sign::Fermi);
        bubble1_moving_herm(n, &mut cb, 1, 1, &a, 0, 0, &un, 0, 0);
    }));
}
