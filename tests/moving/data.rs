// Project : KadanoffBaym
// File    : tests/moving/data.rs
// Purpose : the moving (truncated-window) data layer
//
// Known deviations from the reference:
// * the stale-view checks (`timestep_view` + generation) are subsumed
// by the Rust borrow checker: a row borrow of a `MovingGreen`
// cannot outlive a mutating operation in safe code, so staleness is
// unrepresentable and the checks are replaced by data checks;
// * the storage gate is degraded to size/no-reallocation assertions
// (the `@timed`/GC machinery is reference-specific);
// * steady-state initialization is covered with `SteadyGreen`.

use std::f64::consts::PI;

use std::ptr::eq;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::{
    Contour, ContourGreen, MovingGreen, MovingTimeFunction, Sign, SteadyGreen, SteadyGrid,
    TimeFunction,
};

use kadanoffbaym::io::{moving_green_from_str, moving_time_function_from_str};
use kadanoffbaym::variants::{
    copy_timestep, forward, init_from_function, init_from_green, init_from_green_herm,
    init_from_steady,
};

use super::util::{c, panics};

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Full Green's function with a unique code in every independent
/// block . `nt_points` is the reference point
/// count.
fn full_green(nt_points: i64, d: usize, sign: Sign) -> ContourGreen {
    let cont = Contour::new(
        nt_points - 1,
        4,
        d as i64,
        d as i64,
        (nt_points - 1) as f64 * 0.1,
        1.0,
    );
    let mut f = ContourGreen::new(&cont, sign, Complex::ZERO);
    for n in 1..=nt_points {
        for m in 1..=n {
            for a in 1..=d {
                for b in 1..=d {
                    f.ret.raw_block_at_mut(n - 1, m - 1)[(a - 1) * d + (b - 1)] = c(
                        100.0 * n as f64 + m as f64 + 0.1 * a as f64,
                        0.01 * b as f64,
                    );
                }
            }
            let mut l = vec![Complex::ZERO; d * d];
            for a in 1..=d {
                for b in 1..=d {
                    l[(a - 1) * d + (b - 1)] =
                        c(0.3 * a as f64 - 0.2 * b as f64, 10.0 * n as f64 + m as f64);
                }
            }
            if m == n {
                let snap = l.clone();
                for a in 0..d {
                    for b in 0..d {
                        l[a * d + b] = (snap[a * d + b] - snap[b * d + a].conj()) / 2.0;
                    }
                }
            }
            f.less.raw_block_at_mut(m - 1, n - 1).copy_from_slice(&l);
        }
    }
    f
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn moving_data_exchange() {
    for &d in &[1usize, 2] {
        let (nt_points, tc, n0) = (12i64, 4i64, 7i64);
        let f = full_green(nt_points, d, Sign::Fermi);
        let mut g = MovingGreen::new(tc, d, Sign::Fermi);
        init_from_green_herm(&mut g, &f, n0);
        assert!(g.is_initialized() && g.tstp == n0);
        // every block of the window
        for n in (n0 - tc)..=n0 {
            for m in n.max(0)..=n {
                let gr = g.ret_block(n, m);
                let fr = f.ret_block(n, m);
                for (x, y) in gr.iter().zip(fr.raw().iter()) {
                    assert_eq!(*x, *y, "ret ({n}, {m})");
                }
                let gl = g.less_block(n, m);
                let fl = f.less_block(n, m);
                for (x, y) in gl.iter().zip(fl.raw().iter()) {
                    assert!((*x - *y).abs() < 1e-14, "less ({n}, {m})");
                }
                // the other time ordering is reconstructed
                let gr2 = g.ret_block(m, n);
                let fr2 = f.ret_block(m, n);
                for (x, y) in gr2.iter().zip(fr2.raw().iter()) {
                    assert_eq!(*x, *y);
                }
                let gl2 = g.less_block(m, n);
                let fl2 = f.less_block(m, n);
                for (x, y) in gl2.iter().zip(fl2.raw().iter()) {
                    assert!((*x - *y).abs() < 1e-13);
                }
                // gtr = ret + less
                for q in 0..(d * d) {
                    let gtr = gr[q] + gl[q];
                    let want = fr.raw()[q] + fl.raw()[q];
                    assert!((gtr - want).abs() < 1e-13);
                }
            }
        }
        // relative indices agree with reference
        let rel = g.ret.mblock(2, 1);
        let want = f.ret.raw_block_at(n0 - 2, n0 - 3);
        for q in 0..(d * d) {
            assert_eq!(rel[q], want[q]);
        }
        // outside the window
        assert!(panics(|| g.ret_block(n0 + 1, n0)));
        assert!(panics(|| g.ret_block(n0 - tc - 1, n0 - tc - 1)));
        assert!(panics(|| g.ret_block(n0, n0 - tc - 1)));
        // rows with n - 1 < tc are padded with zeros
        let mut g2 = MovingGreen::new(tc, d, Sign::Fermi);
        init_from_green_herm(&mut g2, &f, tc);
        assert!(g2.ret.mblock(tc, 1).iter().all(|z| *z == Complex::ZERO));
        let want = f.ret.raw_block_at(0, 0);
        for (x, y) in g2.ret.mblock(tc, 0).iter().zip(want.iter()) {
            assert_eq!(*x, *y);
        }
        // error paths
        let mut ge = MovingGreen::new(tc, d, Sign::Fermi);
        assert!(panics(|| init_from_green_herm(&mut ge, &f, tc - 1)));
        assert!(panics(|| init_from_green_herm(&mut ge, &f, 12)));
        assert!(panics(|| {
            init_from_green(&mut MovingGreen::new(tc, d, Sign::Bose), &f, &f, n0);
        }));
        assert!(panics(|| {
            init_from_green_herm(&mut MovingGreen::new(tc, d + 1, Sign::Fermi), &f, n0);
        }));
        // non-hermitian Gcc: the lesser is -Gcc^{<}(m, n)^dagger
        let mut fcc = full_green(nt_points, d, Sign::Fermi);
        for i in 0..=(nt_points - 1) {
            for m in 0..=i {
                for q in 0..(d * d) {
                    fcc.less.raw_block_at_mut(m, i)[q] = fcc.less.raw_block_at(m, i)[q] * 2.0;
                }
            }
        }
        let mut g3 = MovingGreen::new(tc, d, Sign::Fermi);
        init_from_green(&mut g3, &f, &fcc, n0);
        // relative block (1, 2): absolute (n0 - 1, n0 - 3)
        let got = g3.less.mblock(1, 2);
        let src = f.less.raw_block_at(n0 - 3, n0 - 1);
        for a in 0..d {
            for b in 0..d {
                let want = -2.0 * src[b * d + a].conj();
                assert!((got[a * d + b] - want).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn moving_forward_is_o1_and_keeps_the_history() {
    let (nt_points, tc, n0, d) = (12i64, 4i64, 7i64, 2usize);
    let f = full_green(nt_points, d, Sign::Fermi);
    let mut g = MovingGreen::new(tc, d, Sign::Fermi);
    init_from_green_herm(&mut g, &f, n0);
    let p_ret = g.ret.raw().as_ptr();
    let p_less = g.less.raw().as_ptr();
    for step in 1..=2i64 {
        forward(&mut g);
        assert_eq!(g.tstp, n0 + step);
        assert!(eq(g.ret.raw().as_ptr(), p_ret));
        assert!(eq(g.less.raw().as_ptr(), p_less));
        // the new leading row is zero
        for j in 0..=tc {
            assert!(g.ret.mblock(0, j).iter().all(|z| *z == Complex::ZERO));
            assert!(g.less.mblock(0, j).iter().all(|z| *z == Complex::ZERO));
        }
        // the history at absolute times is unchanged
        for n in (n0 + step - tc)..=n0 {
            for m in n.max(0)..=n {
                if n - m > tc || (n0 + step) - n > tc {
                    continue;
                }
                let gr = g.ret_block(n, m);
                let fr = f.ret_block(n, m);
                let gl = g.less_block(n, m);
                let fl = f.less_block(n, m);
                for q in 0..(d * d) {
                    assert!((gr[q] - fr.raw()[q]).abs() < 1e-14);
                    assert!((gl[q] - fl.raw()[q]).abs() < 1e-14);
                }
            }
        }
    }
    // ring wrap-around: tc + 1 steps return the head to its start
    let h = g.ret.head();
    for _ in 0..(tc + 1) {
        forward(&mut g);
    }
    assert_eq!(g.ret.head(), h);
    assert_eq!(g.less.head(), h);
}

#[test]
fn moving_copies_and_views() {
    let (tc, d) = (3i64, 2usize);
    let f = full_green(8, d, Sign::Fermi);
    let mut g = MovingGreen::new(tc, d, Sign::Fermi);
    init_from_green_herm(&mut g, &f, 5);
    // public accessor returns a copy
    let mut x = g.ret_block(5, 4);
    for z in x.iter_mut() {
        *z = Complex::ZERO;
    }
    let fr = f.ret_block(5, 4);
    for q in 0..(d * d) {
        assert_eq!(g.ret_block(5, 4)[q], fr.raw()[q]);
    }
    // a row access writes through to the parent
    g.ret.mblock_mut(1, 2).fill(c(1.0, 2.0));
    for q in 0..(d * d) {
        assert_eq!(g.ret_block(4, 2)[q], c(1.0, 2.0));
        assert_eq!(g.ret.mblock(1, 2)[q], c(1.0, 2.0));
    }
    // owning time step is decoupled
    let mut x = copy_timestep(&g, 0);
    assert_eq!(x.tstp, 5);
    let want = f.ret.raw_block_at(5, 4);
    for (xq, wq) in x.ret.mblock(1).iter().zip(want.iter()) {
        assert_eq!(*xq, *wq);
    }
    x.ret.mblock_mut(1).fill(Complex::ZERO);
    for (gq, wq) in g.ret_block(5, 4).iter().zip(want.iter()) {
        assert_eq!(*gq, *wq);
    }
    // bounds
    assert!(panics(|| g.ret.mblock(0, tc + 1)));
    // after forward! the data checks hold (the stale-view checks of
    // the reference are subsumed by the borrow checker)
    let before = g.ret.mblock(0, 0).to_vec();
    forward(&mut g);
    assert_eq!(g.ret.mblock(1, 0), before);
    // re-initialization
    init_from_green_herm(&mut g, &f, 5);
    let gl = g.less_block(5, 5);
    let fl = f.less_block(5, 5);
    for (x, y) in gl.iter().zip(fl.raw().iter()) {
        assert!((*x - *y).abs() < 1e-13);
    }
}

#[test]
fn moving_single_time_function() {
    let (tc, d, nt_points) = (3i64, 2usize, 9i64);
    let mut cf = TimeFunction::new((nt_points - 1) as usize, d, d, Complex::ZERO);
    for i in 0..=nt_points {
        for a in 1..=d {
            for b in 1..=d {
                cf.set(i - 1, a - 1, b - 1, c(i as f64, a as f64 + 10.0 * b as f64));
            }
        }
    }
    let mut f = MovingTimeFunction::new(tc, d);
    init_from_function(&mut f, &cf, 6);
    assert!(f.is_initialized() && f.tstp == 6);
    for delti in 0..=tc {
        let want = cf.block(6 - delti);
        assert_eq!(f.fslot(delti), want);
    }
    let p = f.raw().as_ptr();
    f.forward();
    assert_eq!(f.tstp, 7);
    assert!(eq(f.raw().as_ptr(), p));
    assert!(f.fslot(0).iter().all(|z| *z == Complex::ZERO));
    for delti in 1..=tc {
        let want = cf.block(7 - delti);
        assert_eq!(f.fslot(delti), want);
    }
    f.fslot_mut(0).copy_from_slice(cf.block(7));
    assert_eq!(f.fslot(0), cf.block(7));
    // bounds and error paths
    assert!(panics(|| f.fslot(tc + 1)));
    assert!(panics(|| {
        init_from_function(&mut MovingTimeFunction::new(tc, d), &cf, tc - 1);
    }));
    assert!(panics(|| {
        init_from_function(&mut MovingTimeFunction::new(tc, d + 1), &cf, 6);
    }));
}

#[test]
fn moving_text_io() {
    let (tc, d) = (3i64, 2usize);
    let f = full_green(9, d, Sign::Fermi);
    let mut g = MovingGreen::new(tc, d, Sign::Fermi);
    init_from_green_herm(&mut g, &f, 6);
    forward(&mut g);
    g.ret.mblock_mut(0, 1).fill(c(0.25, -1.5));
    let text = (g).to_string();
    let mut h = MovingGreen::new(tc, d, Sign::Fermi);
    moving_green_from_str(&text, &mut h).unwrap();
    assert!(h.is_initialized() && h.tstp == g.tstp);
    for i in 0..=tc {
        for j in 0..=tc {
            assert_eq!(h.ret.mblock(i, j), g.ret.mblock(i, j));
            assert_eq!(h.less.mblock(i, j), g.less.mblock(i, j));
        }
    }
    // bad inputs: the target is unchanged
    let snap = (
        h.ret.raw().to_vec(),
        h.less.raw().to_vec(),
        h.tstp,
        h.generation(),
    );
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let bad_cases: Vec<(&str, Vec<String>)> = vec![
        ("trunc", lines[..lines.len() - 3].to_vec()),
        ("version", {
            let mut v = lines.clone();
            v[0] = v[0].replace("KB_MOVING_V1", "KB_MOVING_V2");
            v
        }),
        ("tc", {
            let mut v = lines.clone();
            v[0] = v[0].replace(" F 3 ", " F 4 ");
            v
        }),
        ("index", {
            let mut v = lines.clone();
            v[1] = v[1].replacen("0 0", "0 7", 1);
            v
        }),
        ("nan", {
            let mut v = lines.clone();
            let parts: Vec<&str> = v[1].split_whitespace().collect();
            v[1] = format!(
                "{} {} {} {} NaN 0.0",
                parts[0], parts[1], parts[2], parts[3]
            );
            v
        }),
        ("state", {
            let mut v = lines.clone();
            let parts: Vec<&str> = v[0].split_whitespace().collect();
            v[0] = format!(
                "{} {} {} {} {} {} {}",
                parts[0], parts[1], parts[2], "0", parts[4], parts[5], parts[6]
            );
            v
        }),
        ("kind", {
            let mut v = lines.clone();
            v[0] = v[0].replace(" F ", " Cf ");
            v
        }),
    ];
    for (name, ls) in &bad_cases {
        let t = ls.join("\n");
        assert!(
            moving_green_from_str(&t, &mut h).is_err(),
            "case {name} must be rejected"
        );
    }
    assert_eq!(h.ret.raw(), snap.0.as_slice());
    assert_eq!(h.less.raw(), snap.1.as_slice());
    assert_eq!(h.tstp, snap.2);
    assert_eq!(h.generation(), snap.3);
    // an uninitialized object round trip
    let utext = (MovingGreen::new(tc, d, Sign::Fermi)).to_string();
    let mut u = MovingGreen::new(tc, d, Sign::Fermi);
    moving_green_from_str(&utext, &mut u).unwrap();
    assert!(!u.is_initialized());
    // single-time function
    let mut cf = TimeFunction::new(9, d, d, Complex::ZERO);
    for i in -1..=9i64 {
        for a in 0..d {
            for b in 0..d {
                cf.set(i, a, b, c((i + 1) as f64, -(i + 1) as f64));
            }
        }
    }
    let mut fm = MovingTimeFunction::new(tc, d);
    init_from_function(&mut fm, &cf, 5);
    fm.forward();
    let ftext = (fm).to_string();
    let mut f2 = MovingTimeFunction::new(tc, d);
    moving_time_function_from_str(&ftext, &mut f2).unwrap();
    assert_eq!(f2.tstp, 6);
    for i in 0..=tc {
        assert_eq!(f2.fslot(i), fm.fslot(i));
    }
    // truncated single-time text is rejected
    let flines: Vec<&str> = ftext.lines().collect();
    let ftrunc = flines[..flines.len() - 1].join("\n");
    assert!(moving_time_function_from_str(&ftrunc, &mut f2).is_err());
}

#[test]
fn moving_full_contour_alias_contract_is_unchanged() {
    let mut f = full_green(5, 2, Sign::Fermi);
    // the stored region writes through
    f.ret.raw_block_at_mut(3, 1)[0] = c(99.0, 0.0);
    assert_eq!(f.ret.raw_block_at(3, 1)[0], c(99.0, 0.0));
    // the derived region returns a copy: modifying it cannot affect
    // the parent (value semantics of the reconstruction)
    let y = f.ret_block(1, 3);
    let mut y2 = y.clone();
    y2.set(0, 0, c(-77.0, 0.0));
    assert_ne!(f.ret_block(1, 3).get(0, 0), c(-77.0, 0.0));
    let z = f.less_block(1, 3);
    let mut z2 = z.clone();
    z2.set(0, 0, c(5.0, 0.0));
    assert_ne!(f.less_block(1, 3).get(0, 0), c(5.0, 0.0));
    // the stored lesser region writes through
    f.less.raw_block_at_mut(1, 3)[0] = c(123.0, 0.0);
    assert_eq!(f.less.raw_block_at(1, 3)[0], c(123.0, 0.0));
}

#[test]
fn moving_storage_gate() {
    // degraded to size/no-reallocation assertions (the @timed/GC
    // machinery of the reference is not portable)
    let d = 2usize;
    for tc in [100i64, 500, 1000] {
        let g = MovingGreen::new(tc, d, Sign::Fermi);
        let payload = ((tc + 1) * (tc + 1)) as usize * d * d;
        assert_eq!(g.ret.raw().len(), payload);
        assert_eq!(g.less.raw().len(), payload);
    }
    // forward! does not reallocate the payload
    let f = full_green(12, d, Sign::Fermi);
    for tc in [4i64, 10] {
        let mut g = MovingGreen::new(tc, d, Sign::Fermi);
        init_from_green_herm(&mut g, &f, 10);
        let p = g.ret.raw().as_ptr();
        forward(&mut g);
        assert!(eq(g.ret.raw().as_ptr(), p));
    }
}

#[test]
fn moving_steady_initialization() {
    let (tc, d, nft, h) = (4i64, 2usize, 8i64, 0.1f64);
    let mut n = SteadyGreen::new(SteadyGrid::new_full(nft, h, 2.0 * PI / (nft as f64 * h)), d);
    for m in 1..=nft as usize {
        for a in 0..d {
            for b in 0..d {
                n.ret.time[(m - 1) * 4 + a * 2 + b] = c(m as f64, (a as i64 - b as i64) as f64);
                n.les.time[(m - 1) * 4 + a * 2 + b] = c((a + b + 2) as f64, -(m as f64));
            }
        }
    }
    let mut g = MovingGreen::new(tc, d, Sign::Fermi);
    let f = full_green(8, d, Sign::Fermi);
    init_from_green_herm(&mut g, &f, 5);
    init_from_steady(&mut g, &n);
    assert!(g.is_initialized() && g.tstp == 0);
    let dtmax = tc.min(nft / 2 - 1);
    for delti in 0..=tc {
        for dt in 0..=tc {
            if dt <= dtmax {
                assert_eq!(
                    g.ret.mblock(delti, dt),
                    &n.ret.time[dt as usize * 4..(dt as usize + 1) * 4]
                );
                assert_eq!(
                    g.less.mblock(delti, dt),
                    &n.les.time[dt as usize * 4..(dt as usize + 1) * 4]
                );
            } else {
                assert!(g.ret.mblock(delti, dt).iter().all(|z| *z == Complex::ZERO));
                assert!(g.less.mblock(delti, dt).iter().all(|z| *z == Complex::ZERO));
            }
        }
    }
    // absolute access below my step 0 is not allowed
    assert_eq!(g.ret_block(0, 0), n.ret.time[0..4].to_vec());
    assert!(panics(|| g.ret_block(0, -1)));
    forward(&mut g);
    assert_eq!(g.tstp, 1);
    assert_eq!(g.ret_block(0, 0), n.ret.time[0..4].to_vec());
    assert_eq!(g.ret.mblock(1, 1), &n.ret.time[4..2 * 4]);
    assert!(panics(|| {
        let mut gd = MovingGreen::new(tc, d + 1, Sign::Fermi);
        init_from_steady(&mut gd, &n);
    }));
}
