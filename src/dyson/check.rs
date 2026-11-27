// Project : KadanoffBaym
// File    : src/dyson/check.rs
// Purpose : argument checks of the Dyson solvers (the standard and the
//           pseudo-particle family)

use crate::types::{ContourGreen, MovingGreen, PseudoGreen};
use crate::types::{MovingTimeFunction, Sign, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

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

/// Check the arguments of the pseudo-particle Dyson solvers.
pub(super) fn check_pseudo_dyson(
    g: &PseudoGreen,
    sigma: &PseudoGreen,
    h: &TimeFunction,
    integ: &Integrator,
    nmax: i64,
) {
    check_pseudo_dyson_args(g, sigma, integ);
    if nmax >= 0 {
        assert!(
            g.nt as i64 >= nmax,
            "G must have at least {} real time steps",
            nmax + 1
        );
        assert!(
            h.nt as i64 >= nmax,
            "H must have at least {} real time steps",
            nmax + 1
        );
    }
    let d = g.size1;
    // Hamiltonian checks over the needed steps (the
    // Matsubara axis and 0..nmax), including the hermiticity check
    let tstps: Vec<i64> = if nmax >= 0 {
        (MATSUBARA..=nmax).collect()
    } else {
        vec![MATSUBARA]
    };
    check_hamiltonian(h, tstps.into_iter(), d);
}

/// Check the argument grids of the pseudo-particle Dyson solvers (no
/// Hamiltonian checks); shared by [`check_pseudo_dyson`] and the
/// Matsubara solver.
pub(super) fn check_pseudo_dyson_args(g: &PseudoGreen, sigma: &PseudoGreen, integ: &Integrator) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order must be in 1..=5, got {}",
        integ.k
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
        g.ntau as i64 > integ.k,
        "ntau must be larger than solve_order + 1"
    );
}

/// Check the moving Hamiltonian of the truncated Dyson solver:
/// initialized, `h.tc >= g.tc`, the leading time step, and the matrix size.
pub(super) fn check_moving_hamiltonian(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    assert!(h.is_initialized(), "the moving object is not initialized");
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}
