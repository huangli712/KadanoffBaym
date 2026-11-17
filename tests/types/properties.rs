// Project : KadanoffBaym
// File    : tests/properties.rs
// Purpose : port of t_properties

use super::util::prelude::*;
use kadanoffbaym::basics::timestep_at;

/// Port of the NESSi property ledger: sizes/dims of the components of
/// the full object and its timestep slice, plus the fact that the
/// derived views read the parent storage.
#[test]
fn nessi_full_timestep_and_auxiliary_view_properties() {
    for &(nd1, nd2) in &[(1usize, 1usize), (2, 2), (2, 3)] {
        // blueprint Cn(4, 5, ...) -> Rust (3, 4)
        let c = Contour::new(3, 4, nd1 as i64, nd2 as i64, 1.0, 2.0);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::new(1.0, 2.0));
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
        g.set_mat(1, 0, 0, Complex::new(3.0, 0.0));
        assert_eq!(g.get_matminus(4 - 1, 0, 0), Complex::new(-3.0, 0.0));
        g.set_ret(3, 0, 0, 0, Complex::new(5.0, 1.0));
        assert_eq!(g.adv_block(0, 3).get(0, 0), Complex::new(5.0, -1.0));
        g.set_lmix(3, 2, 0, 0, Complex::new(2.0, 2.0));
        assert_eq!(g.rmix_block(4 - 2, 3).get(0, 0), Complex::new(2.0, -2.0));
        let gtr = g.gtr_block(3, 3);
        assert_eq!(
            gtr.get(0, 0),
            g.ret_block(3, 3).get(0, 0) + g.less_block(3, 3).get(0, 0)
        );
    }
}
