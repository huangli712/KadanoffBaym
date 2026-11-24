// Project : KadanoffBaym
// File    : tests/types/indexing.rs
// Purpose : index translation, construction, and element access of the
//           contour containers
//

use kadanoffbaym::storage::{set_timestep_at, timestep_at};

use super::util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const V1: Complex = Complex::new(0.2, -0.1);

const V2: Complex = Complex::new(1.0, 0.3);

const V3: Complex = Complex::new(0.3, 1.0);

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

fn block_equals(blk: &[Complex], want: &[Complex]) -> bool {
    blk.len() == want.len() && blk.iter().zip(want).all(|(a, b)| a == b)
}

/// `indexing_diagonal_block(seed)`.
fn diagonal_block(seed: f64, ndim: usize) -> Vec<Complex> {
    let v = seed;
    let full = [
        Complex::new(0.0, v),
        Complex::new(1.0, v + 1.0),
        Complex::new(-1.0, v + 1.0),
        Complex::new(0.0, -(v + 2.0)),
    ];
    if ndim == 1 {
        vec![full[0]]
    } else {
        full.to_vec()
    }
}

/// `fill_indexing_green!` with 0-based seeds (reference seed 10+q for
/// mat(q) becomes 11+n, etc.).
fn fill_indexing_green(g: &mut ContourGreen) {
    let ndim = g.size1.min(g.size2);
    for n in 0..=(g.ntau as i64) {
        let blk = index_block(11.0 + n as f64, ndim);
        g.mat_raw_block_mut(n).copy_from_slice(&blk);
    }
    for i in 0..=(g.nt as i64) {
        for j in 0..=i {
            let blk = if i == j {
                diagonal_block(101.0 + i as f64, ndim)
            } else {
                index_block(111.0 + 10.0 * i as f64 + j as f64, ndim)
            };
            g.ret_raw_block_mut(i, j).copy_from_slice(&blk);
        }
        for m in 0..=(g.ntau as i64) {
            let blk = index_block(211.0 + 10.0 * i as f64 + m as f64, ndim);
            g.lmix_raw_block_mut(i, m).copy_from_slice(&blk);
        }
        for j in i..=(g.nt as i64) {
            let blk = if i == j {
                diagonal_block(301.0 + i as f64, ndim)
            } else {
                index_block(311.0 + 10.0 * i as f64 + j as f64, ndim)
            };
            g.less_raw_block_mut(i, j).copy_from_slice(&blk);
        }
    }
}

fn filled_block(v: Complex, n: usize) -> Vec<Complex> {
    vec![v; n]
}

/// `indexing_block(seed)` (2x2, row-major; ndim = 1 takes the first
/// element like the reference's `[1:ndim,1:ndim]` slice).
fn index_block(seed: f64, ndim: usize) -> Vec<Complex> {
    let v = seed;
    let full = [
        Complex::new(v, v + 1.0),
        Complex::new(-0.5 * v, v + 2.0),
        Complex::new(v / 3.0, -(v + 3.0)),
        Complex::new(v + 4.0, -(v + 5.0)),
    ];
    if ndim == 1 {
        vec![full[0]]
    } else {
        full.to_vec()
    }
}

fn matrix_equals(m: &DenseMat, want: &[Complex]) -> bool {
    let (r, c) = (m.nrows(), m.ncols());
    r * c == want.len() && (0..want.len()).all(|k| m.get(k / c, k % c) == want[k])
}

/// `-want^dagger` for a square row-major block.
fn negative_adjoint(want: &[Complex], ndim: usize) -> Vec<Complex> {
    (0..ndim * ndim)
        .map(|k| {
            let a = k / ndim;
            let b = k % ndim;
            -want[b * ndim + a].conj()
        })
        .collect()
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn cf_getindex_setindex() {
    // reference ntime = 201 -> nt = 200; cf[0] is the Matsubara slot
    let mut cf = TimeFunction::new(200, 2, 2, V1);
    for t in -1..=200 {
        assert!(block_equals(cf.block(t), &filled_block(V1, 4)), "t = {t}");
    }
    for t in [MATSUBARA, 0, 200] {
        cf.block_mut(t).copy_from_slice(&filled_block(V2, 4));
    }
    for t in [MATSUBARA, 0, 200] {
        assert!(block_equals(cf.block(t), &filled_block(V2, 4)));
    }
    for t in [MATSUBARA, 0, 200] {
        cf.block_mut(t).copy_from_slice(&filled_block(V3, 4));
    }
    assert!(block_equals(cf.block(MATSUBARA), &filled_block(V3, 4)));
    assert!(block_equals(cf.block(0), &filled_block(V3, 4)));
    assert!(block_equals(cf.block(200), &filled_block(V3, 4)));
}

#[test]
fn gmat_getindex_setindex() {
    // reference ntau = 1001 -> ntau = 1000 intervals, 1001 blocks
    let mut m = MatComp::new(1000, 2, 2, V1);
    for k in 0..m.nblocks() {
        assert!(block_equals(m.block(k), &filled_block(V1, 4)));
    }
    for k in [0usize, 1, 1000] {
        m.block_mut(k).copy_from_slice(&filled_block(V2, 4));
    }
    for k in [0usize, 1, 1000] {
        assert!(block_equals(m.block(k), &filled_block(V2, 4)));
    }
    for k in [0usize, 1, 1000] {
        m.block_mut(k).copy_from_slice(&filled_block(V3, 4));
    }
    assert!(block_equals(m.block(1000), &filled_block(V3, 4)));
}

#[test]
fn gret_getindex_setindex() {
    // full-square read with the -x' fallback (via a full green)
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, V1);
    for i in 0..=200i64 {
        for j in 0..=200i64 {
            let m = g.ret_block(i, j);
            let want = if i >= j {
                filled_block(V1, 4)
            } else {
                negative_adjoint(&filled_block(V1, 4), 2)
            };
            assert!(matrix_equals(&m, &want), "ret({i},{j})");
        }
    }
    // set stored blocks
    g.ret_raw_block_mut(1, 0)
        .copy_from_slice(&filled_block(V2, 4));
    g.ret_raw_block_mut(200, 200)
        .copy_from_slice(&filled_block(V2, 4));
    assert!(block_equals(g.ret_raw_block(1, 0), &filled_block(V2, 4)));
    assert!(block_equals(
        g.ret_raw_block(200, 200),
        &filled_block(V2, 4)
    ));
    g.ret_raw_block_mut(1, 0)
        .copy_from_slice(&filled_block(V3, 4));
    g.ret_raw_block_mut(200, 200)
        .copy_from_slice(&filled_block(V3, 4));
    assert!(block_equals(g.ret_raw_block(1, 0), &filled_block(V3, 4)));
    assert!(block_equals(
        g.ret_raw_block(200, 200),
        &filled_block(V3, 4)
    ));
}

#[test]
fn glmix_getindex_setindex() {
    let mut l = LmixComp::new(200, 1000, 2, 2, V1);
    for k in 0..l.nblocks() {
        assert!(block_equals(l.block(k), &filled_block(V1, 4)));
    }
    // reference lmix[1,1], [2,3], [3,2], [ntime,ntau] -> (0,0),(1,2),(2,1),(200,1000)
    for (t, m) in [(0i64, 0i64), (1, 2), (2, 1), (200, 1000)] {
        let k = l.idx(t, m);
        l.block_mut(k).copy_from_slice(&filled_block(V2, 4));
    }
    for (t, m) in [(0i64, 0i64), (1, 2), (2, 1), (200, 1000)] {
        let k = l.idx(t, m);
        assert!(block_equals(l.block(k), &filled_block(V2, 4)), "({t},{m})");
    }
    for (t, m) in [(0i64, 0i64), (1, 2), (2, 1), (200, 1000)] {
        let k = l.idx(t, m);
        l.block_mut(k).copy_from_slice(&filled_block(V3, 4));
    }
    assert!(block_equals(
        l.block(l.idx(200, 1000)),
        &filled_block(V3, 4)
    ));
}

#[test]
fn gless_getindex_setindex() {
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, V1);
    for i in 0..=200i64 {
        for j in 0..=200i64 {
            let m = g.less_block(i, j);
            // less stores i <= j; i > j falls back to -x'
            let want = if i <= j {
                filled_block(V1, 4)
            } else {
                negative_adjoint(&filled_block(V1, 4), 2)
            };
            assert!(matrix_equals(&m, &want), "less({i},{j})");
        }
    }
    // reference less[1,2] and less[ntime,ntime] -> (0,1), (200,200)
    g.less_raw_block_mut(0, 1)
        .copy_from_slice(&filled_block(V2, 4));
    g.less_raw_block_mut(200, 200)
        .copy_from_slice(&filled_block(V2, 4));
    assert!(block_equals(g.less_raw_block(0, 1), &filled_block(V2, 4)));
    assert!(block_equals(
        g.less_raw_block(200, 200),
        &filled_block(V2, 4)
    ));
    g.less_raw_block_mut(0, 1)
        .copy_from_slice(&filled_block(V3, 4));
    g.less_raw_block_mut(200, 200)
        .copy_from_slice(&filled_block(V3, 4));
    assert!(block_equals(
        g.less_raw_block(200, 200),
        &filled_block(V3, 4)
    ));
}

#[test]
fn gmatm_getindex_setindex() {
    // matm[i] == mat[ntau - i + 1] * sign -> matminus(n) ==
    // mat(ntau_rust - n) * sign
    let c = Contour::new(5, 1000, 2, 2, 5.0, 4.0);
    let sign = Sign::Fermi;
    let xi = Complex::new(sign.as_i64() as f64, 0.0);
    for v in [V1, V2, V3] {
        let g = ContourGreen::new(&c, sign, v);
        for n in 0..=1000i64 {
            let want = g.get_mat(1000 - n, 0, 0) * xi;
            assert_eq!(g.get_matminus(n, 0, 0), want, "n = {n}");
        }
    }
}

#[test]
fn gadv_getindex_setindex() {
    let c = Contour::new(3, 5, 2, 2, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    for i in 0..=3i64 {
        for j in 0..=i {
            let blk = if i == j {
                diagonal_block(i as f64 + 1.0, 2)
            } else {
                index_block(10.0 * (i as f64 + 1.0) + (j as f64 + 1.0), 2)
            };
            g.ret_raw_block_mut(i, j).copy_from_slice(&blk);
        }
    }
    // adv[i,j] == ret[j,i]^dagger over the full square
    for i in 0..=3i64 {
        for j in 0..=3i64 {
            let adv = g.adv_block(i, j);
            let want = g.ret_block(j, i).adjoint();
            assert!(
                matrix_equals(&adv, &{
                    let mut w = Vec::new();
                    for a in 0..2 {
                        for b in 0..2 {
                            w.push(want.get(a, b));
                        }
                    }
                    w
                }),
                "adv({i},{j})"
            );
        }
    }
    // mutating the parent propagates to the view
    g.set_ret(3, 0, 0, 0, Complex::new(17.0, -3.0));
    assert_eq!(g.adv_block(0, 3).get(0, 0), Complex::new(17.0, 3.0));
}

#[test]
#[should_panic]
fn gadv_bounds_low() {
    let c = Contour::new(3, 5, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.adv_block(-1, 0);
}

#[test]
#[should_panic]
fn gadv_bounds_high() {
    let c = Contour::new(3, 5, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.adv_block(0, 4);
}

#[test]
fn grmix_getindex_setindex() {
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let sign = Sign::Fermi;
    let m_xi = Complex::new(-(sign.as_i64() as f64), 0.0);
    for v in [V1, V2, V3] {
        let g = ContourGreen::new(&c, sign, v);
        // rmix[i,j] == lmix[j, ntau - i + 1]' * (-sign) -> sample a grid
        for n in (0..=1000i64).step_by(97) {
            for t in (0..=200i64).step_by(31) {
                let rm = g.rmix_block(n, t);
                let src = g.lmix_block(t, 1000 - n);
                let want = src.adjoint();
                for a in 0..2 {
                    for b in 0..2 {
                        assert_eq!(rm.get(a, b), want.get(a, b) * m_xi, "({n},{t})");
                    }
                }
            }
        }
    }
}

#[test]
fn ggtr_getindex_setindex() {
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    for v in [V1, V2, V3] {
        let g = ContourGreen::new(&c, Sign::Fermi, v);
        for i in 0..=200i64 {
            for j in 0..=200i64 {
                let gtr = g.gtr_block(i, j);
                let r = g.ret_block(i, j);
                let l = g.less_block(i, j);
                for a in 0..2 {
                    for b in 0..2 {
                        assert_eq!(gtr.get(a, b), r.get(a, b) + l.get(a, b), "({i},{j})");
                    }
                }
            }
        }
    }
}

#[test]
fn gmat_timestep_getindex_setindex() {
    // the timestep mat component is the same container type
    let mut m = MatComp::new(1000, 2, 2, V1);
    for k in 0..m.nblocks() {
        assert!(block_equals(m.block(k), &filled_block(V1, 4)));
    }
    m.block_mut(1000).copy_from_slice(&filled_block(V2, 4));
    assert!(block_equals(m.block(1000), &filled_block(V2, 4)));
}

#[test]
fn gret_timestep_getindex_setindex() {
    // reference tstp = 101 -> Rust tstp = 100
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    let mut f = GreenTimeStep::new(&c, 100, Sign::Fermi, V1);
    // single-index ret[j] = current-row block (tstp, j); ret[i, tstp]
    // with i < tstp is the fallback -x'
    for j in 0..=100i64 {
        assert!(
            block_equals(f.ret_raw_block(100, j), &filled_block(V1, 4)),
            "j = {j}"
        );
        let m = f.ret_block(j, 100);
        let want = if j == 100 {
            filled_block(V1, 4)
        } else {
            negative_adjoint(&filled_block(V1, 4), 2)
        };
        assert!(matrix_equals(&m, &want), "ret({j},100)");
    }
    for j in [0i64, 1, 100] {
        f.ret_raw_block_mut(100, j)
            .copy_from_slice(&filled_block(V2, 4));
    }
    for j in [0i64, 1, 100] {
        assert!(block_equals(f.ret_raw_block(100, j), &filled_block(V2, 4)));
        let m = f.ret_block(j, 100);
        assert!(matrix_equals(&m, &negative_adjoint(&filled_block(V2, 4), 2)) || j == 100);
    }
}

#[test]
fn gless_timestep_getindex_setindex() {
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    let mut f = GreenTimeStep::new(&c, 100, Sign::Fermi, V1);
    // single-index less[i] = block (i, tstp); less[tstp, i] falls back
    for i in 0..=100i64 {
        assert!(
            block_equals(f.less_raw_block(i, 100), &filled_block(V1, 4)),
            "i = {i}"
        );
        let m = f.less_block(100, i);
        let want = if i == 100 {
            filled_block(V1, 4)
        } else {
            negative_adjoint(&filled_block(V1, 4), 2)
        };
        assert!(matrix_equals(&m, &want), "less(100,{i})");
    }
    for i in [0i64, 1, 100] {
        f.less_raw_block_mut(i, 100)
            .copy_from_slice(&filled_block(V2, 4));
    }
    assert!(block_equals(f.less_raw_block(0, 100), &filled_block(V2, 4)));
}

#[test]
fn gadv_timestep_getindex_setindex() {
    let c = Contour::new(4, 5, 2, 2, 1.0, 2.0);
    let mut f = GreenTimeStep::new(&c, 3, Sign::Fermi, Complex::ZERO);
    for j in 0..=3i64 {
        let blk = index_block(j as f64 + 1.0, 2);
        f.ret_raw_block_mut(3, j).copy_from_slice(&blk);
    }
    // adv[i] == ret[i]' (current row)
    for j in 0..=3i64 {
        let adv = f.adv_block(j, 3);
        let want = f.ret_block(3, j).adjoint();
        for a in 0..2 {
            for b in 0..2 {
                assert_eq!(adv.get(a, b), want.get(a, b), "j = {j}");
            }
        }
    }
    // write-through
    f.set_ret(3, 2, 0, 1, Complex::new(4.0, 9.0));
    assert_eq!(f.adv_block(2, 3).get(1, 0), Complex::new(4.0, -9.0));
}

#[test]
fn grmix_timestep_getindex_setindex() {
    // single-index: rmix[i] == lmix[ntau - i + 1]' * (-sign)
    let c = Contour::new(4, 1000, 2, 2, 1.0, 2.0);
    let sign = Sign::Fermi;
    let m_xi = Complex::new(-(sign.as_i64() as f64), 0.0);
    let f = GreenTimeStep::new(&c, 3, sign, V1);
    for n in (0..=1000i64).step_by(97) {
        let rm = f.rmix_block(n, 3);
        let src = f.lmix_block(3, 1000 - n);
        let want = src.adjoint();
        for a in 0..2 {
            for b in 0..2 {
                assert_eq!(rm.get(a, b), want.get(a, b) * m_xi, "n = {n}");
            }
        }
    }
}

#[test]
fn ggtr_timestep_getindex_setindex() {
    // gtr[i] == less[i] + ret[i, tstp]; gtr[tstp, j] == less[tstp, j] + ret[j]
    let c = Contour::new(200, 5, 2, 2, 5.0, 4.0);
    let f = GreenTimeStep::new(&c, 100, Sign::Fermi, V1);
    for j in 0..=100i64 {
        let gtr = f.gtr_block(j, 100);
        let l = f.less_block(j, 100);
        let r = f.ret_block(j, 100);
        for a in 0..2 {
            for b in 0..2 {
                assert_eq!(gtr.get(a, b), l.get(a, b) + r.get(a, b), "col j = {j}");
            }
        }
        let gtr2 = f.gtr_block(100, j);
        let l2 = f.less_block(100, j);
        let r2 = f.ret_block(100, j);
        for a in 0..2 {
            for b in 0..2 {
                assert_eq!(gtr2.get(a, b), l2.get(a, b) + r2.get(a, b), "row j = {j}");
            }
        }
    }
}

#[test]
fn f_getindex_setindex() {
    // reference: ntime = 201, tstp = 101 -> Rust nt = 200, tstp = 100
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    let sign = Sign::Fermi;
    let mut g = ContourGreen::new(&c, sign, V1);
    for v in [V2, V3] {
        let e0 = GreenTimeStep::new(&c, 0, sign, v);
        let e1 = GreenTimeStep::new(&c, 1, sign, v);
        let et = GreenTimeStep::new(&c, 100, sign, v);
        set_timestep_at(&mut g, &e0);
        set_timestep_at(&mut g, &e1);
        set_timestep_at(&mut g, &et);
        let s0 = timestep_at(&g, 0);
        let s1 = timestep_at(&g, 1);
        let st = timestep_at(&g, 100);
        assert!(s0.mat.equal(&e0.mat));
        assert!(s1.ret.equal(&e1.ret));
        assert!(s1.lmix.equal(&e1.lmix));
        assert!(s1.less.equal(&e1.less));
        assert!(st.ret.equal(&et.ret));
        assert!(st.lmix.equal(&et.lmix));
        assert!(st.less.equal(&et.less));
    }
}

#[test]
fn timestep_getindex_setindex() {
    // ports of the reference member-timestep contracts; Cn(4, 5) -> (3, 4)
    for ndim in [1usize, 2] {
        let c = Contour::new(3, 4, ndim as i64, ndim as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        fill_indexing_green(&mut g);

        // equilibrium slice shares the mat values
        let eq = timestep_at(&g, 0);
        for q in 0..=(g.ntau as i64) {
            assert!(
                block_equals(eq.mat_raw_block(q), g.mat_raw_block(q)),
                "q = {q}"
            );
        }
        assert_eq!(eq.sign, Sign::Fermi);
        assert_eq!(eq.tstp, 0);

        // timestep slice at tstp = 3
        let tstp = 3i64;
        let mut slice = timestep_at(&g, tstp);
        assert_eq!(slice.sign, Sign::Fermi);
        assert_eq!(slice.tstp as i64, tstp);
        for j in 1..=tstp {
            assert!(block_equals(
                slice.ret_raw_block(tstp, j),
                g.ret_raw_block(tstp, j)
            ));
            assert!(block_equals(
                slice.less_raw_block(j, tstp),
                g.less_raw_block(j, tstp)
            ));
            // gtr consistency inside the slice
            let gtr = slice.gtr_block(j, tstp);
            let want = g.gtr_block(j, tstp);
            for a in 0..ndim {
                for b in 0..ndim {
                    assert_eq!(gtr.get(a, b), want.get(a, b), "gtr({j})");
                }
            }
            let gtr2 = slice.gtr_block(tstp, j);
            let want2 = g.gtr_block(tstp, j);
            assert_eq!(gtr2.get(0, 0), want2.get(0, 0));
        }
        for q in 0..=(g.ntau as i64) {
            assert!(block_equals(
                slice.lmix_raw_block(tstp, q),
                g.lmix_raw_block(tstp, q)
            ));
        }

        // slices are independent copies; write-back moves the values
        let original = g.ret_raw_block(tstp, 0).to_vec();
        slice.set_ret(tstp, 0, 0, 0, Complex::new(41.0, 2.0));
        assert!(block_equals(g.ret_raw_block(tstp, 0), &original));
        let untouched = g.ret_raw_block(1, 0).to_vec();
        set_timestep_at(&mut g, &slice);
        assert_eq!(g.get_ret(tstp, 0, 0, 0), Complex::new(41.0, 2.0));
        assert!(block_equals(g.ret_raw_block(1, 0), &untouched));

        // equilibrium write-back replaces the mat component
        let replacement = GreenTimeStep::new(&c, 0, Sign::Fermi, Complex::new(7.0, 3.0));
        set_timestep_at(&mut g, &replacement);
        for q in 0..=(g.ntau as i64) {
            assert!(block_equals(
                g.mat_raw_block(q),
                &filled_block(Complex::new(7.0, 3.0), ndim * ndim)
            ));
        }
    }
}

#[test]
#[should_panic]
fn timestep_get_bounds_low() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = timestep_at(&g, -1);
}

#[test]
#[should_panic]
fn timestep_get_bounds_high() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = timestep_at(&g, 4);
}

#[test]
fn full_scalar_and_matrix_get_contracts() {
    for ndim in [1usize, 2] {
        let c = Contour::new(3, 4, ndim as i64, ndim as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        fill_indexing_green(&mut g);
        let ntau = 4i64; // max tau index

        for q in 0..=ntau {
            assert!(
                block_equals(g.mat_raw_block(q), &index_block(11.0 + q as f64, ndim)),
                "q = {q}"
            );
            // matm[q] == mat[ntau - q + 1] with sign = BOSE = +1
            assert_eq!(
                g.get_matminus(q, 0, 0),
                g.get_mat(ntau - q, 0, 0) * Complex::new(1.0, 0.0)
            );
        }
        for i in 0..=3i64 {
            for j in 0..=3i64 {
                let expected_ret = if i >= j {
                    if i == j {
                        diagonal_block(101.0 + i as f64, ndim)
                    } else {
                        index_block(111.0 + 10.0 * i as f64 + j as f64, ndim)
                    }
                } else {
                    negative_adjoint(&index_block(111.0 + 10.0 * j as f64 + i as f64, ndim), ndim)
                };
                let expected_less = if i <= j {
                    if i == j {
                        diagonal_block(301.0 + i as f64, ndim)
                    } else {
                        index_block(311.0 + 10.0 * i as f64 + j as f64, ndim)
                    }
                } else {
                    negative_adjoint(&index_block(311.0 + 10.0 * j as f64 + i as f64, ndim), ndim)
                };
                assert!(
                    matrix_equals(&g.ret_block(i, j), &expected_ret),
                    "ret({i},{j})"
                );
                assert!(
                    matrix_equals(&g.less_block(i, j), &expected_less),
                    "less({i},{j})"
                );
                // adv and gtr identities
                let adv = g.adv_block(i, j);
                let rji = g.ret_block(j, i);
                for a in 0..ndim {
                    for b in 0..ndim {
                        assert_eq!(adv.get(a, b), rji.get(b, a).conj());
                    }
                }
                let gtr = g.gtr_block(i, j);
                for a in 0..ndim {
                    for b in 0..ndim {
                        assert_eq!(
                            gtr.get(a, b),
                            expected_ret[a * ndim + b] + expected_less[a * ndim + b]
                        );
                    }
                }
            }
            for q in 0..=ntau {
                assert!(
                    block_equals(
                        g.lmix_raw_block(i, q),
                        &index_block(211.0 + 10.0 * i as f64 + q as f64, ndim)
                    ),
                    "lmix({i},{q})"
                );
                // rmix[q,i] == -lmix[i, ntau - q + 1]' (sign = BOSE)
                let rm = g.rmix_block(q, i);
                let src = g.lmix_block(i, ntau - q).adjoint();
                for a in 0..ndim {
                    for b in 0..ndim {
                        assert_eq!(rm.get(a, b), -(src.get(a, b)), "rmix({q},{i})");
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic]
fn get_bounds_mat() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.get_mat(-1, 0, 0);
}

#[test]
#[should_panic]
fn get_bounds_ret() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.ret_block(-1, 0);
}

#[test]
#[should_panic]
fn get_bounds_lmix() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.lmix_block(0, 5);
}

#[test]
#[should_panic]
fn get_bounds_less() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let _ = g.less_block(4, 0);
}

#[test]
fn derived_copy_independence() {
    // reference lines 749-758: derived blocks are copies, stored blocks
    // write through; setting the non-stored region errors
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    fill_indexing_green(&mut g);
    let stored = g.ret_raw_block(2, 0).to_vec();
    let mut derived = g.ret_block(0, 2);
    derived.scale(0.0);
    assert!(block_equals(g.ret_raw_block(2, 0), &stored));
    g.set_ret(2, 0, 0, 0, Complex::new(31.0, 6.0));
    assert_eq!(g.get_ret(2, 0, 0, 0), Complex::new(31.0, 6.0));
}

#[test]
#[should_panic]
fn set_nonstored_ret() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    g.set_ret(0, 2, 0, 0, Complex::ONE);
}

#[test]
#[should_panic]
fn set_nonstored_less() {
    let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    g.set_less(2, 0, 0, 0, Complex::ONE);
}

#[test]
fn nonhermitian_full_contracts() {
    for ndim in [1usize, 2] {
        let c = Contour::new(3, 4, ndim as i64, ndim as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        fill_indexing_green(&mut g);
        if ndim == 2 {
            // reference's explicit non-commuting blocks
            let ret_blk = [
                Complex::new(1.0, 1.0),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 2.0),
                Complex::new(-1.0, 1.0),
            ];
            let less_blk = [
                Complex::new(0.5, -1.0),
                Complex::new(-2.0, 3.0),
                Complex::new(1.0, 4.0),
                Complex::new(2.0, -2.0),
            ];
            g.ret_raw_block_mut(2, 0).copy_from_slice(&ret_blk);
            g.less_raw_block_mut(0, 2).copy_from_slice(&less_blk);
            // ret != transpose(ret) and the products do not commute
            let r = g.ret_block(2, 0);
            let mut rt = DenseMat::zeros(2);
            for a in 0..2 {
                for b in 0..2 {
                    rt.set(a, b, r.get(b, a));
                }
            }
            assert!(r.get(0, 1) != rt.get(0, 1));
            let l = g.less_block(0, 2);
            let rl = r.mat_mul(&l);
            let lr = l.mat_mul(&r);
            assert!(rl.get(0, 1) != lr.get(0, 1));
        }
        for i in 0..=3i64 {
            for j in 0..=3i64 {
                let gtr = g.gtr_block(i, j);
                let r = g.ret_block(i, j);
                let l = g.less_block(i, j);
                // gtr == ret + less and gtr antisymmetry
                let gtr_t = g.gtr_block(j, i);
                for a in 0..ndim {
                    for b in 0..ndim {
                        assert_eq!(gtr.get(a, b), r.get(a, b) + l.get(a, b));
                        assert_eq!(gtr.get(a, b), -gtr_t.get(b, a).conj());
                    }
                }
            }
        }
    }
}

#[test]
fn nonhermitian_timestep_and_view_contracts() {
    for ndim in [1usize, 2] {
        let c = Contour::new(3, 4, ndim as i64, ndim as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        fill_indexing_green(&mut g);
        let tstp = 3i64;
        let slice = timestep_at(&g, tstp);
        for i in 1..=tstp {
            assert!(block_equals(
                slice.ret_raw_block(tstp, i),
                g.ret_raw_block(tstp, i)
            ));
            assert!(block_equals(
                slice.less_raw_block(i, tstp),
                g.less_raw_block(i, tstp)
            ));
            // gtr within the slice and its antisymmetry
            let gtr = slice.gtr_block(i, tstp);
            let gtr_t = slice.gtr_block(tstp, i);
            for a in 0..ndim {
                for b in 0..ndim {
                    assert_eq!(gtr.get(a, b), -gtr_t.get(b, a).conj());
                }
            }
        }
        for q in 0..=(g.ntau as i64) {
            assert!(block_equals(
                slice.lmix_raw_block(tstp, q),
                g.lmix_raw_block(tstp, q)
            ));
            // slice.rmix[q] == G.rmix[q, tstp]
            let rm = slice.rmix_block(q, tstp);
            let want = g.rmix_block(q, tstp);
            for a in 0..ndim {
                for b in 0..ndim {
                    assert_eq!(rm.get(a, b), want.get(a, b), "q = {q}");
                }
            }
        }
    }
}

#[test]
fn member_set_get_and_matminus_contracts() {
    // rectangular 2x3 object; copy-in semantics
    let c = Contour::new(3, 4, 2, 3, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let block = vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(5.0, 4.0),
        Complex::new(-2.0, 3.0),
        Complex::new(7.0, 1.0),
        Complex::new(-4.0, -2.0),
    ];
    let expected = block.clone();
    g.mat_raw_block_mut(0).copy_from_slice(&block);
    g.ret_raw_block_mut(3, 0).copy_from_slice(&block);
    g.lmix_raw_block_mut(1, 4).copy_from_slice(&block);
    g.less_raw_block_mut(0, 3).copy_from_slice(&block);
    // mutating the source afterwards does not change the green (copies)
    let mut source = block.clone();
    source[0] = Complex::new(99.0, 99.0);
    assert!(block_equals(g.mat_raw_block(0), &expected));
    assert!(block_equals(g.ret_raw_block(3, 0), &expected));
    assert!(block_equals(g.lmix_raw_block(1, 4), &expected));
    assert!(block_equals(g.less_raw_block(0, 3), &expected));
    // scalar fills
    g.mat_raw_block_mut(4)
        .copy_from_slice(&filled_block(Complex::new(0.0, 2.0), 6));
    g.ret_raw_block_mut(3, 3)
        .copy_from_slice(&filled_block(Complex::new(0.0, 3.0), 6));
    g.lmix_raw_block_mut(3, 0)
        .copy_from_slice(&filled_block(Complex::new(0.0, 4.0), 6));
    g.less_raw_block_mut(3, 3)
        .copy_from_slice(&filled_block(Complex::new(0.0, 5.0), 6));
    assert!(block_equals(
        g.mat_raw_block(4),
        &filled_block(Complex::new(0.0, 2.0), 6)
    ));
    assert!(block_equals(
        g.ret_raw_block(3, 3),
        &filled_block(Complex::new(0.0, 3.0), 6)
    ));
    assert!(block_equals(
        g.lmix_raw_block(3, 0),
        &filled_block(Complex::new(0.0, 4.0), 6)
    ));
    assert!(block_equals(
        g.less_raw_block(3, 3),
        &filled_block(Complex::new(0.0, 5.0), 6)
    ));

    // matminus for both signs
    for sign in [Sign::Fermi, Sign::Bose] {
        let c = Contour::new(3, 4, 2, 2, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, sign, Complex::ZERO);
        fill_indexing_green(&mut g);
        let xi = Complex::new(sign.as_i64() as f64, 0.0);
        for q in 0..=(g.ntau as i64) {
            assert_eq!(
                g.get_matminus(q, 0, 0),
                g.get_mat(g.ntau as i64 - q, 0, 0) * xi,
                "q = {q}"
            );
        }
    }
}
