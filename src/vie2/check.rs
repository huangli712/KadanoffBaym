// Project : KadanoffBaym
// File    : src/vie2/check.rs
// Purpose : argument checks of the VIE2 solvers (the standard and the
//           pseudo-particle family)

use crate::types::{ContourGreen, PseudoGreen, Sign};
use crate::weights::Integrator;

/// Check the compatibility of the arguments of the VIE2 solvers.
pub(super) fn check_vie2(g: &ContourGreen, f: &ContourGreen, fcc: &ContourGreen, q: &ContourGreen) {
    assert!(
        g.sign == Sign::Fermi || g.sign == Sign::Bose,
        "unknown statistics of G"
    );
    for (x, name) in [(f, "F"), (fcc, "Fcc"), (q, "Q")] {
        assert!(
            x.sign == g.sign,
            "G and {name} must have the same statistics"
        );
        assert!(
            x.size1 == g.size1 && x.size2 == g.size2,
            "G and {name} have different sizes"
        );
        assert!(
            x.ntau == g.ntau,
            "G and {name} have different imaginary time grids"
        );
        assert!(x.nt == g.nt, "G and {name} have different real time grids");
    }
    assert!(g.size1 == g.size2, "G must be square");
}

/// Check the order of the integrator.
pub(super) fn check_vie2_order(integ: &Integrator) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order of the integrator must be in 1..=5, got {}",
        integ.k
    );
}

/// Check the arguments of the pseudo-particle VIE2 solvers. `nmax < 0`
/// means that only the Matsubara axis is needed.
pub(super) fn check_pseudo_vie2(
    g: &PseudoGreen,
    f: &PseudoGreen,
    fcc: &PseudoGreen,
    q: &PseudoGreen,
    integ: &Integrator,
    nmax: i64,
) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order must be in 1..=5, got {}",
        integ.k
    );
    for x in [f, fcc, q] {
        assert!(
            x.sign == g.sign,
            "all the Green's functions must have the same statistics"
        );
        assert!(
            x.size1 == g.size1 && x.size2 == g.size2,
            "all the Green's functions must have the same size"
        );
        assert!(
            x.ntau == g.ntau,
            "all the Green's functions must have the same imaginary time grid"
        );
        assert!(
            x.nt == g.nt,
            "all the Green's functions must have the same real time grid"
        );
    }
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        g.ntau as i64 > integ.k,
        "ntau must be larger than solve_order + 1"
    );
    if nmax >= 0 {
        assert!(
            g.nt as i64 >= nmax,
            "G must have at least {} real time steps",
            nmax + 1
        );
    }
}
