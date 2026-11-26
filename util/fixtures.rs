// Project : KadanoffBaym
// File    : util/fixtures.rs
// Purpose : the deterministic 2x2 test Hamiltonians and the helpers built
//           from them
//

use kadanoffbaym::complex::Complex;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::TimeFunction;

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

/// The pseudo-particle test Hamiltonian: the 2 x 2 form for `d = 2`, its
/// (1, 1) element otherwise.
pub fn pseudo_hamiltonian(d: usize) -> DenseMat {
    if d == 2 {
        DenseMat::from_vec(
            2,
            2,
            vec![c(0.3, 0.0), c(0.2, -0.1), c(0.2, 0.1), c(-0.4, 0.0)],
        )
    } else {
        DenseMat::from_vec(1, 1, vec![c(0.3, 0.0)])
    }
}

/// A time function equal to the matrix `m` at the Matsubara slot and at
/// every real time step up to `nt`.
pub fn constant_time_function(nt: i64, m: &DenseMat) -> TimeFunction {
    let d = m.nrows();
    let mut h = TimeFunction::new(nt as usize, d, d, Complex::ZERO);
    for t in -1..=nt {
        for i in 0..d {
            for j in 0..d {
                h.set(t, i, j, m.get(i, j));
            }
        }
    }
    h
}

/// The block matrix `[Ha V; V' Hb]` of an embedded model.
pub fn embedded_hamiltonian(ha: &DenseMat, hb: &DenseMat, v: &DenseMat) -> DenseMat {
    let (da, db) = (ha.nrows(), hb.nrows());
    let dfull = da + db;
    let mut hfull = DenseMat::zeros(dfull);
    for i in 0..dfull {
        for j in 0..dfull {
            let x = if i < da && j < da {
                ha.get(i, j)
            } else if i < da && j >= da {
                v.get(i, j - da)
            } else if i >= da && j < da {
                v.get(j, i - da).conj()
            } else {
                hb.get(i - da, j - da)
            };
            hfull.set(i, j, x);
        }
    }
    hfull
}
