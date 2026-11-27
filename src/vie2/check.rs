// Project : KadanoffBaym
// File    : src/vie2/check.rs
// Purpose : argument checks shared by the VIE2 solvers

use crate::types::{ContourGreen, Sign};
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
