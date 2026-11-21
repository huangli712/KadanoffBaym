// Project : KadanoffBaym
// File    : src/base/check.rs
// Purpose : solver preconditions of the moving (truncated) paths,
//           shared by the Dyson and VIE2 families

use crate::types::{MovingGreen, MovingTimeFunction};

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

/// Check the moving Hamiltonian of the truncated Dyson solver:
/// initialized, `h.tc >= g.tc`, the leading time step, and the matrix size.
pub(crate) fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    check_initialized_cfmoving(h);
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}

/// Throw if the moving single-time function is not initialized.
fn check_initialized_cfmoving(f: &MovingTimeFunction) {
    assert!(f.initialized, "the moving object is not initialized");
}
