// Project : KadanoffBaym
// File    : src/dyson/check.rs
// Purpose : argument checks shared by the Dyson solvers

use crate::types::{ContourGreen, Sign, TimeFunction};

use crate::equilibrium::check_hamiltonian;

/// Check the compatibility of the arguments of the Dyson solvers.
pub(super) fn check_dyson(
    g: &ContourGreen,
    sigma: &ContourGreen,
    h: &TimeFunction,
    solve_order: i64,
) {
    assert!(
        (1..=5).contains(&solve_order),
        "solve_order must be in 1..=5, got {solve_order}"
    );
    assert!(
        g.sign == Sign::Fermi || g.sign == Sign::Bose,
        "unknown statistics of G"
    );
    assert!(
        g.sign == sigma.sign,
        "G and Sigma must have the same statistics"
    );
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        sigma.size1 == g.size1 && sigma.size2 == g.size2,
        "G and Sigma have different sizes"
    );
    assert!(
        g.ntau == sigma.ntau,
        "G and Sigma have different imaginary time grids"
    );
    assert!(
        g.nt == sigma.nt,
        "G and Sigma have different real time grids"
    );
    assert!(
        h.size1 == g.size1 && h.size2 == g.size2,
        "H must be {} x {}, got {} x {}",
        g.size1,
        g.size2,
        h.size1,
        h.size2
    );
}

/// Check the arguments of the real time Dyson solvers; the time steps
/// `0, .., nmax` of `G`, `Sigma`, and `H` are needed.
pub(super) fn check_dyson_rt(
    g: &ContourGreen,
    sigma: &ContourGreen,
    h: &TimeFunction,
    k: i64,
    nmax: i64,
) {
    check_dyson(g, sigma, h, k);
    assert!(
        (g.nt as i64) >= nmax,
        "G must have at least {nmax} real time steps"
    );
    assert!(
        (h.nt as i64) >= nmax,
        "H must have at least {nmax} real time steps"
    );
    check_hamiltonian(h, 0..=nmax, g.size1);
}
