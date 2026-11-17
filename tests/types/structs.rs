// Project : KadanoffBaym
// File    : tests/structs.rs
// Purpose : port of t_structs (constructors and `==`
// semantics). Blueprint-only conveniences (the PCONTOUR global default
// constructors) are noted inline; everything else keeps the blueprint
// parameters. Index translation: the blueprint's 1-based grid sizes
// (ntime/ntau/tstp/i/j) map to the Rust 0-based conventions.

use super::util::prelude::*;
use kadanoffbaym::basics::timestep_at;

/// Port of the NESSi getset.cpp construction ledger.
#[test]
fn nessi_scalar_matrix_full_timestep_and_view_construction() {
    for (ndim, sign) in [(1usize, Sign::Fermi), (2, Sign::Bose)] {
        // blueprint Cn(4, 5, ndim, ndim, 1.0, 2.0) -> Rust (3, 4)
        let c = Contour::new(3, 4, ndim as i64, ndim as i64, 1.0, 2.0);
        let scalar_value = Complex::new(2.0, -3.0);
        let mut full_scalar = ContourGreen::new(&c, sign, scalar_value);
        let full_matrix = ContourGreen::new(&c, sign, scalar_value);
        let timestep_scalar = GreenTimeStep::new(&c, 3, sign, scalar_value);
        let timestep_matrix = GreenTimeStep::new(&c, 3, sign, scalar_value);
        let view = timestep_at(&full_matrix, 3);

        assert!(full_scalar.equal(&full_matrix));
        assert!(timestep_scalar.equal(&timestep_matrix));
        assert_eq!(full_matrix.sign, sign);
        assert_eq!(view.sign, sign);
        assert_eq!(view.tstp, 3);
        assert!(view.ret.equal(&timestep_matrix.ret));
        assert!(view.lmix.equal(&timestep_matrix.lmix));
        assert!(view.less.equal(&timestep_matrix.less));
        // every stored value equals the matrix value (view.ret[1] etc.)
        for j in 0..=3 {
            for a in 0..ndim {
                for b in 0..ndim {
                    assert_eq!(view.get_ret(3, j as i64, a, b), scalar_value);
                }
            }
        }
        // equilibrium mat component
        let eq_step = GreenTimeStep::new(&c, 0, sign, scalar_value);
        assert_eq!(eq_step.get_mat(0, 0, 0), scalar_value);
        // silence unused warning for the scalar variant
        full_scalar.set_mat(0, 0, 0, scalar_value);
    }
}

/// Port of "contiguous storage of F" (backing shapes, write-through,
/// derived-domain separation, auxiliary views, timestep copies).
#[test]
fn contiguous_storage_backing_and_aliasing() {
    // blueprint ntime = 11, ntau = 7 points -> Rust nt = 10, ntau = 6
    let (ntime, ntau) = (11usize, 7usize);
    let c = Contour::new(ntime as i64 - 1, ntau as i64 - 1, 2, 3, 1.0, 2.0);
    let mut g = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    let ntri = ntime * (ntime + 1) / 2;

    // backing shapes
    assert_eq!(g.mat.nblocks(), ntau);
    assert_eq!(g.ret.nblocks(), ntri);
    assert_eq!(g.lmix.nblocks(), ntime * ntau);
    assert_eq!(g.less.nblocks(), ntri);

    // write-through on the independent domain
    // blueprint: G.ret[5,2] = x (1-based) -> ret(4, 1)
    let x = vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 0.0),
    ];
    g.ret_raw_block_mut(4, 1).copy_from_slice(&x);
    assert_eq!(g.get_ret(4, 1, 0, 0), Complex::new(1.0, 0.0));
    assert_eq!(g.get_ret(4, 1, 1, 2), Complex::new(6.0, 0.0));
    // blueprint: G.lmix[3,4][2,3] = 7im -> lmix(2, 3) element (1, 2)
    g.set_lmix(2, 3, 1, 2, Complex::new(0.0, 7.0));
    assert_eq!(g.get_lmix(2, 3, 1, 2), Complex::new(0.0, 7.0));
    // blueprint: G.less[2,5][1,2] = -im -> less(1, 4) element (0, 1)
    g.set_less(1, 4, 0, 1, Complex::new(0.0, -1.0));
    assert_eq!(g.get_less(1, 4, 0, 1), Complex::new(0.0, -1.0));
    // blueprint: G.mat[6][2,1] = 0.5 -> mat(5) element (1, 0)
    g.set_mat(5, 1, 0, Complex::new(0.5, 0.0));
    assert_eq!(g.get_mat(5, 1, 0), Complex::new(0.5, 0.0));

    // separate storage slots: writing one block leaves its neighbor
    g.ret_raw_block_mut(3, 3).fill(Complex::ZERO);
    g.ret_raw_block_mut(3, 2).fill(Complex::ONE);
    assert!(g.ret_raw_block(3, 3).iter().all(|z| *z == Complex::ZERO));
    assert!(g.ret_raw_block(3, 2).iter().all(|z| *z == Complex::ONE));

    // derived (reconstructed) domain: values are copies, never aliased
    let c2 = Contour::new(ntime as i64 - 1, ntau as i64 - 1, 2, 2, 1.0, 2.0);
    let mut h = ContourGreen::new(&c2, Sign::Fermi, Complex::ZERO);
    // blueprint H.ret[4,1] = [1 2; 3 4] -> ret(3, 0)
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, Complex::new(1.0, 0.0));
    m.set(0, 1, Complex::new(2.0, 0.0));
    m.set(1, 0, Complex::new(3.0, 0.0));
    m.set(1, 1, Complex::new(4.0, 0.0));
    h.ret_raw_block_mut(3, 0).copy_from_slice(&[
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ]);
    let mut lm = DenseMat::zeros(2);
    lm.set(0, 0, Complex::new(5.0, 0.0));
    lm.set(0, 1, Complex::new(6.0, 0.0));
    lm.set(1, 0, Complex::new(7.0, 0.0));
    lm.set(1, 1, Complex::new(8.0, 0.0));
    h.less_raw_block_mut(0, 3).copy_from_slice(&[
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 0.0),
        Complex::new(7.0, 0.0),
        Complex::new(8.0, 0.0),
    ]);
    // reconstructed: blueprint r = H.ret[1,4] == -H.ret[4,1]'
    let r = h.ret_block(0, 3);
    assert_eq!(r.get(0, 0), m.get(0, 0).conj() * Complex::new(-1.0, 0.0));
    assert_eq!(r.get(1, 0), -(m.get(0, 1)));
    // blueprint l = H.less[4,1] == -H.less[1,4]'
    let l = h.less_block(3, 0);
    assert_eq!(l.get(0, 0), -(lm.get(0, 0)));
    assert_eq!(l.get(0, 1), -(lm.get(1, 0)));
    // mutating the copies leaves the parent untouched
    let mut r2 = r.clone();
    r2.scale(0.0);
    let mut l2 = l.clone();
    l2.scale(0.0);
    assert_eq!(h.get_ret(3, 0, 0, 0), Complex::new(1.0, 0.0));
    assert_eq!(h.get_less(0, 3, 0, 0), Complex::new(5.0, 0.0));

    // auxiliary views reference the parent values
    // blueprint: H.adv[1,4] == H.ret[4,1]' -> adv(0, 3) == ret(3, 0)†
    assert_eq!(
        h.adv_block(0, 3).get(0, 0),
        h.ret_block(3, 0).get(0, 0).conj()
    );
    assert_eq!(
        h.adv_block(0, 3).get(0, 1),
        h.ret_block(3, 0).get(1, 0).conj()
    );
    // blueprint: H.gtr[4,1] == H.ret[4,1] + H.less[4,1] -> gtr(3, 0)
    let gtr = h.gtr_block(3, 0);
    let rb = h.ret_block(3, 0);
    let lb = h.less_block(3, 0);
    assert_eq!(gtr.get(0, 0), rb.get(0, 0) + lb.get(0, 0));
    assert_eq!(gtr.get(1, 1), rb.get(1, 1) + lb.get(1, 1));
    // blueprint: H.rmix[2,3] == H.lmix[3,ntau-1]' * (-FERMI)
    // blueprint ntau = 7 -> lmix(2, 5); -FERMI = +1
    h.set_lmix(2, 5, 0, 0, Complex::new(0.0, 1.0));
    h.set_lmix(2, 5, 1, 0, Complex::new(2.0, 0.0));
    let rm = h.rmix_block(1, 2);
    assert_eq!(rm.get(0, 0), Complex::new(0.0, -1.0)); // conj(i) * (+1)
    assert_eq!(rm.get(0, 1), Complex::new(2.0, 0.0)); // adjoint: conj(block(1, 0))
    // blueprint: H.matm[2] == H.mat[ntau-1] * FERMI -> matminus(1) = mat(5)*sign
    h.set_mat(5, 0, 0, Complex::new(0.0, 2.0));
    assert_eq!(
        h.get_matminus(1, 0, 0),
        Complex::new(0.0, 2.0) * Complex::new(-1.0, 0.0)
    );

    // timestep slices are independent copies
    let mut v = timestep_at(&h, 3);
    v.set_ret(3, 0, 0, 0, Complex::new(42.0, 0.0));
    assert_eq!(h.get_ret(3, 0, 0, 0), Complex::new(1.0, 0.0));

    // payload accounting (blueprint kb_payload for 41 x 21, 2x2)
    let c41 = Contour::new(40, 20, 2, 2, 1.0, 2.0);
    let f41 = ContourGreen::new(&c41, Sign::Fermi, Complex::ZERO);
    // elements = 4 * (2*ntri + 41*21 + 21) = 4 * 2604; bytes = *16
    assert_eq!(f41.nwords() * 8, 166_656);
    // allocation footprint is four flat buffers regardless of the grid
    let c_small = Contour::new(10, 6, 2, 2, 1.0, 2.0);
    let c_big = Contour::new(100, 6, 2, 2, 1.0, 2.0);
    assert_eq!(
        std::mem::size_of::<ContourGreen>(),
        std::mem::size_of::<ContourGreen>()
    );
    let _ = (
        ContourGreen::new(&c_small, Sign::Fermi, Complex::ZERO),
        ContourGreen::new(&c_big, Sign::Fermi, Complex::ZERO),
    );
}
