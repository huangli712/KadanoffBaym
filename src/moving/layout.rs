// Project : KadanoffBaym
// File    : src/moving/layout.rs
// Purpose : sanity checks of the moving (truncated) solvers (blueprint
//           julia/src/moving/layout.jl, serial parts)

use crate::types::moving::MovingGreen;
use crate::types::moving::CfMoving;

/// Throw if the moving object `g` is not initialized (blueprint
/// `_check_initialized` for `ℱᵐᵒᵛ`).
pub(crate) fn check_initialized_moving(g: &MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
}

/// Throw if the moving single-time function is not initialized.
pub(crate) fn check_initialized_cfmoving(f: &CfMoving) {
    assert!(f.initialized, "the moving object is not initialized");
}

/// Check the moving objects of the truncated solvers: initialized, the
/// same cut-off, the leading time step `tstp`, the matrix size, and
/// `tc >= 2k + 2` (blueprint `_check_moving_solver`).
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

/// Check the moving Hamiltonian of the truncated Dyson solver
/// (blueprint part of `dyson_timestep!`): initialized, `h.tc >= g.tc`,
/// the leading time step, and the matrix size.
pub(crate) fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &CfMoving) {
    check_initialized_cfmoving(h);
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}
