// Project : KadanoffBaym
// File    : src/base/check.rs
// Purpose : solver preconditions shared across the equation families:
//           the moving (truncated) window checks and the Hamiltonian
//           sanity check

use crate::types::{MovingGreen, TimeFunction};

/// Check the moving objects of the truncated solvers: initialized, the
/// same cut-off, the leading time step `tstp`, the matrix size, and
/// `tc >= 2k + 2`.
pub(crate) fn check_moving_solver(tstp: i64, k: i64, objs: &[&MovingGreen]) {
    assert!((1..=5).contains(&k), "solve_order must be in 1:5, got {k}");
    for x in objs {
        assert!(x.initialized, "the moving object is not initialized");
        assert!(x.tc == objs[0].tc, "the cut-offs differ");
        assert!(
            x.tstp == tstp,
            "tstp = {tstp} is not the leading time step {}",
            x.tstp
        );
        assert!(
            x.ret.size1 == objs[0].ret.size1 && x.ret.size2 == objs[0].ret.size2,
            "the matrix sizes differ"
        );
    }
    assert!(
        objs[0].tc >= 2 * k + 2,
        "the truncated solver of order {k} requires tc >= {}, got {}",
        2 * k + 2,
        objs[0].tc
    );
}

/// Check that `h` is a finite and hermitian `d x d` matrix at all the
/// time steps in `tstps` (`MATSUBARA` for the Matsubara axis).
pub(crate) fn check_hamiltonian(h: &TimeFunction, tstps: impl Iterator<Item = i64>, d: usize) {
    assert!(
        h.size1 == d && h.size2 == d,
        "H must be {d} x {}, got {} x {}",
        d,
        h.size1,
        h.size2
    );
    for n in tstps {
        assert!(
            (-1..=(h.nt as i64)).contains(&n),
            "H is not defined at time step {n}"
        );
        let hn = h.block(n);
        for z in hn {
            assert!(
                z.re.is_finite() && z.im.is_finite(),
                "H is not finite at time step {n}"
            );
        }
        // ||H - H^dag|| <= tol max(1, ||H||)
        let mut r = 0.0f64;
        let mut norm = 0.0f64;
        for b in 0..d {
            for a in 0..d {
                let dz = hn[a * d + b] - hn[b * d + a].conj();
                r += dz.abs() * dz.abs();
                let e = hn[a * d + b];
                norm += e.abs() * e.abs();
            }
        }
        assert!(
            r.sqrt() <= 1e-12 * 1.0f64.max(norm.sqrt()),
            "H is not hermitian at time step {n}"
        );
    }
}
