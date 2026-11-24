// Project : KadanoffBaym
// File    : util/fixtures.rs
// Purpose : the deterministic 2x2 test Hamiltonians of the t_dyson /
//           t_vie suites

use kadanoffbaym::complex::Complex;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

/// Blueprint `t_dyson` Hamiltonian `A`.
pub fn test_ha() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(-1.0, 0.0));
    m.set(0, 1, c(0.0, 0.3));
    m.set(1, 0, c(0.0, -0.3));
    m.set(1, 1, c(0.5, 0.0));
    m
}

/// Blueprint `t_dyson` Hamiltonian `B`.
pub fn test_hb() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(1.0, 0.0));
    m.set(0, 1, c(0.2, 0.1));
    m.set(1, 0, c(0.2, -0.1));
    m.set(1, 1, c(1.5, 0.0));
    m
}

/// Blueprint `t_dyson` interaction matrix `V`.
pub fn test_vv() -> kadanoffbaym::linalg::DenseMat {
    let mut m = kadanoffbaym::linalg::DenseMat::zeros(2);
    m.set(0, 0, c(0.4, 0.0));
    m.set(0, 1, c(0.0, 0.1));
    m.set(1, 0, c(-0.2, 0.0));
    m.set(1, 1, c(0.3, 0.0));
    m
}
