// Project : KadanoffBaym
// File    : util/fixtures.rs
// Purpose : the deterministic 2x2 test Hamiltonians
//

use kadanoffbaym::linalg::DenseMat;

use super::support::c;

/// Reference `` Hamiltonian `A`.
pub fn test_hamiltonian_a() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(-1.0, 0.0));
    m.set(0, 1, c(0.0, 0.3));
    m.set(1, 0, c(0.0, -0.3));
    m.set(1, 1, c(0.5, 0.0));
    m
}

/// Reference `` Hamiltonian `B`.
pub fn test_hamiltonian_b() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(1.0, 0.0));
    m.set(0, 1, c(0.2, 0.1));
    m.set(1, 0, c(0.2, -0.1));
    m.set(1, 1, c(1.5, 0.0));
    m
}

/// Reference `` interaction matrix `V`.
pub fn test_interaction() -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, c(0.4, 0.0));
    m.set(0, 1, c(0.0, 0.1));
    m.set(1, 0, c(-0.2, 0.0));
    m.set(1, 1, c(0.3, 0.0));
    m
}
