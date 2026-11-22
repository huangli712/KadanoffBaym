// Project : KadanoffBaym
// File    : src/storage/moving.rs
// Purpose : moving-window preconditions shared by the truncated solvers
//           and the observables

use crate::types::MovingGreen;

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
