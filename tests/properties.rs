// Project : KadanoffBaym
// File    : tests/properties.rs
// Purpose : port of julia/test/t_properties.jl

mod common;

use common::prelude::*;
use kadanoffbaym::core::indexing::{set_timestep_at, timestep_at};

#[test]
fn contour_properties() {
    // blueprint: ntime = 201 points, ntau = 1001 points
    let c = Contour::new(200, 1000, 2, 2, 5.0, 4.0);
    assert_eq!(c.nt + 1, 201);
    assert_eq!(c.ntau + 1, 1001);
    assert_eq!((c.size1, c.size2), (2, 2));
    assert_eq!(c.tmax, 5.0);
    assert_eq!(c.beta, 4.0);
    assert_eq!(c.dt, 5.0 / 200.0);
    assert_eq!(c.dtau, 4.0 / 1000.0);
    assert_eq!(c.size1, c.size2);
}

#[test]
fn cf_properties() {
    // blueprint: ntime = 101 points
    let c = Contour::new(100, 50, 2, 2, 5.0, 4.0);
    let cf1 = TimeFunction::new(100, 2, 2, C64::ZERO);
    let cf2 = TimeFunction::new(100, 2, 2, C64::ZERO);
    assert_eq!(cf1.raw().len(), 408); // (nt + 2) blocks x 4 elements
    assert_eq!((cf1.size1, cf1.size2), (2, 2));
    assert_eq!(cf1.size1, cf1.size2);
    assert!(cf1.is_compatible(&cf2));
    assert!(cf1.distance(&cf2) < 1e-7);
    let _ = &c;
}

#[test]
fn component_properties() {
    // G-mat: ntau = 51 points -> 50 intervals
    let c = Contour::new(100, 50, 2, 3, 5.0, 4.0);
    let m1 = MatComp::new(50, 2, 3, C64::ZERO);
    let m2 = MatComp::new(50, 2, 3, C64::ZERO);
    assert_eq!(m1.nblocks(), 51);
    assert_eq!((m1.size1, m1.size2), (2, 3));
    assert_ne!(m1.size1, m1.size2);
    assert!(m1.is_compatible(&m2));
    assert!(m1.distance(&m2) < 1e-7);

    // G-ret: ntime = 101 points -> nt = 100
    let r1 = RetComp::new(100, 2, 3, C64::ZERO);
    let r2 = RetComp::new(100, 2, 3, C64::ZERO);
    assert_eq!(r1.nblocks(), (101 * 102) / 2);
    assert_ne!(r1.size1, r1.size2);
    assert!(r1.is_compatible(&r2));

    // G-lmix
    let l1 = LmixComp::new(100, 50, 2, 3, C64::ZERO);
    let l2 = LmixComp::new(100, 50, 2, 3, C64::ZERO);
    assert_eq!(l1.nblocks(), 101 * 51);
    assert!(l1.is_compatible(&l2));

    // G-less
    let s1 = LessComp::new(100, 2, 3, C64::ZERO);
    let s2 = LessComp::new(100, 2, 3, C64::ZERO);
    assert_eq!(s1.nblocks(), (101 * 102) / 2);
    assert!(s1.is_compatible(&s2));
    let _ = &c;
}

#[test]
fn timestep_component_properties() {
    // g-mat / g-lmix share the component types; g-ret/g-less are
    // tstp-sized triangles (blueprint tstp = 101 -> Rust nt = 100)
    let m1 = MatComp::new(50, 2, 3, C64::ZERO);
    let m2 = MatComp::new(50, 2, 3, C64::ZERO);
    assert!(m1.is_compatible(&m2));
    let r1 = RetComp::new(100, 2, 3, C64::ZERO);
    let r2 = RetComp::new(100, 2, 3, C64::ZERO);
    assert_eq!(r1.nblocks(), (101 * 102) / 2);
    assert!(r1.distance(&r2) < 1e-7);
    let l1 = LmixComp::new(0, 50, 2, 3, C64::ZERO);
    assert_eq!(l1.nblocks(), 51);
    let s1 = LessComp::new(100, 2, 3, C64::ZERO);
    let s2 = LessComp::new(100, 2, 3, C64::ZERO);
    assert!(s1.distance(&s2) < 1e-7);
}

#[test]
fn green_properties() {
    // blueprint: ntime = 101 points, ntau = 51 points, dims 2x3
    let c = Contour::new(100, 50, 2, 3, 5.0, 4.0);
    let g1 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let g2 = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    assert_eq!(g1.sign, Sign::Fermi);
    assert_eq!(g1.nt + 1, 101);
    assert_eq!(g1.ntau + 1, 51);
    assert_eq!((g1.size1, g1.size2), (2, 3));
    assert_ne!(g1.size1, g1.size2);
    for tstp in -1..=(g1.nt as i64) {
        assert!(g1.distance_at(&g2, tstp) < 1e-7, "tstp {tstp}");
    }
    // derived-view contracts on a filled green
    let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    g.set_mat(3, 1, 2, C64::new(0.5, -0.25));
    assert_eq!(
        g.get_matminus(50 - 3, 1, 2),
        g.get_mat(3, 1, 2) * C64::new(-1.0, 0.0)
    );
    g.set_ret(4, 1, 0, 2, C64::new(1.0, 1.0));
    let adv = g.adv_block(1, 4);
    assert_eq!(adv.get(2, 0), C64::new(1.0, -1.0));
    g.set_lmix(2, 5, 0, 1, C64::new(0.0, 1.0));
    let rm = g.rmix_block(50 - 5, 2);
    assert_eq!(rm.get(1, 0), C64::new(0.0, 1.0) * C64::new(-1.0, 0.0));
    let gtr = g.gtr_block(4, 4);
    let rb = g.ret_block(4, 4);
    let lb = g.less_block(4, 4);
    assert_eq!(gtr.get(0, 0), rb.get(0, 0) + lb.get(0, 0));
}

#[test]
fn timestep_properties() {
    // blueprint tstp = 101 -> Rust tstp can be any valid row; use 100
    let c = Contour::new(100, 50, 2, 3, 5.0, 4.0);
    let f1 = GreenTimeStep::new(&c, 100, Sign::Fermi, C64::ZERO);
    let f2 = GreenTimeStep::new(&c, 100, Sign::Fermi, C64::ZERO);
    assert_eq!(f1.sign, Sign::Fermi);
    assert_eq!(f1.tstp, 100);
    assert_eq!(f1.ntau + 1, 51);
    assert_eq!((f1.size1, f1.size2), (2, 3));
    assert_ne!(f1.size1, f1.size2);
    assert!(f1.is_compatible(&f2));
    // timestep of a zero green agrees with a zero timestep object
    let g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
    let row = timestep_at(&g, 100);
    assert!(row.equal(&f1));
    // derived views of the timestep read its own storage
    let mut f = GreenTimeStep::new(&c, 100, Sign::Fermi, C64::ZERO);
    f.set_mat(3, 1, 2, C64::new(0.5, 0.0));
    assert_eq!(f.get_matminus(50 - 3, 1, 2), C64::new(-0.5, 0.0));
    f.set_ret(100, 7, 0, 2, C64::new(1.0, 2.0));
    let adv = f.adv_block(7, 100);
    assert_eq!(adv.get(2, 0), C64::new(1.0, -2.0));
}

/// Port of the NESSi property ledger: sizes/dims of the components of
/// the full object and its timestep slice, plus the fact that the
/// derived views read the parent storage.
#[test]
fn nessi_full_timestep_and_auxiliary_view_properties() {
    for &(nd1, nd2) in &[(1usize, 1usize), (2, 2), (2, 3)] {
        // blueprint Cn(4, 5, ...) -> Rust (3, 4)
        let c = Contour::new(3, 4, nd1 as i64, nd2 as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::new(1.0, 2.0));
        let slice = timestep_at(&g, 3);

        assert_eq!(g.nt + 1, 4);
        assert_eq!(g.ntau + 1, 5);
        assert_eq!((g.size1, g.size2), (nd1, nd2));
        assert_eq!(g.size1 == g.size2, nd1 == nd2);
        assert_eq!(g.mat.nblocks(), 5);
        assert_eq!(g.ret.nblocks(), (4 * 5) / 2);
        assert_eq!(g.lmix.nblocks(), 4 * 5);
        assert_eq!(g.less.nblocks(), (4 * 5) / 2);

        assert_eq!(slice.tstp, 3);
        assert_eq!(slice.ntau + 1, 5);
        assert_eq!((slice.size1, slice.size2), (nd1, nd2));
        assert_eq!(slice.mat.nblocks(), 5);
        assert_eq!(slice.ret.nblocks(), (4 * 5) / 2);
        assert_eq!(slice.lmix.nblocks(), 5);
        assert_eq!(slice.less.nblocks(), (4 * 5) / 2);

        // the derived views read the parent storage (blueprint checks
        // `matm.dataM === G.mat` etc. via object identity): mutating the
        // parent is reflected through every derived view
        g.set_mat(1, 0, 0, C64::new(3.0, 0.0));
        assert_eq!(g.get_matminus(4 - 1, 0, 0), C64::new(-3.0, 0.0));
        g.set_ret(3, 0, 0, 0, C64::new(5.0, 1.0));
        assert_eq!(g.adv_block(0, 3).get(0, 0), C64::new(5.0, -1.0));
        g.set_lmix(3, 2, 0, 0, C64::new(2.0, 2.0));
        assert_eq!(g.rmix_block(4 - 2, 3).get(0, 0), C64::new(2.0, -2.0));
        let gtr = g.gtr_block(3, 3);
        assert_eq!(
            gtr.get(0, 0),
            g.ret_block(3, 3).get(0, 0) + g.less_block(3, 3).get(0, 0)
        );
    }
}
