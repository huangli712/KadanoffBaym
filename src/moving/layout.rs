// Project : KadanoffBaym
// File    : src/moving/layout.rs
// Purpose : sanity checks of the moving (truncated) solvers (blueprint
//           julia/src/moving/layout.jl, serial parts)
//!
//! *Remarks* : *Moving (Truncated) Window*
//!
//! The moving types (`ℱᵐᵒᵛ`, `𝒻ᵐᵒᵛ`, `𝒻ᵛⁱᵉʷᵐᵒᵛ`, `Cfᵐᵒᵛ`) store a window
//! of `tc + 1` real time steps. The block `(delti, deltj)` of `ℱᵐᵒᵛ` is
//!
//! ```math
//! G^{R/<}(t_0 - \delta_i, t_0 - \delta_i - \delta_j),
//! ```
//!
//! where ``t_0`` is the KB time step `tstp` of the leading row. For an
//! initialized object the absolute KB time steps `(n, m)` with `n ≥ m` are
//! mapped to `delti = tstp - n` and `deltj = n - m`. The blocks with `n < m`
//! are reconstructed by the hermitian relations
//!
//! ```math
//! G^{R}(t_n, t_m) = -G^{R}(t_m, t_n)^{\dagger},\quad
//! G^{<}(t_n, t_m) = -G^{<}(t_m, t_n)^{\dagger}.
//! ```
//!
//! Relative accesses (on the components and the views) use the NESSi
//! indices `(delti, deltj)` directly. Only the boundaries to the full
//! contour (`init_from_G!`) and the steady state
//! (`init_from_ness!`) translate between the NESSi 0-based and the KB
//! 1-based time steps.
//!
//! Please see [`NESSi`] `cntr_herm_matrix_moving_impl.hpp`,
//! `cntr_herm_matrix_timestep_moving_view_impl.hpp`, and
//! `cntr_function_moving_impl.hpp`.
//!
//! *Remarks* : *Moving Time Steps*
//!
//! The traits of the moving types act on one time step (a row of the
//! window): an owning `𝒻ᵐᵒᵛ` or a view `𝒻ᵛⁱᵉʷᵐᵒᵛ` of a row of `ℱᵐᵒᵛ`. They
//! work on the contiguous row slices directly. A view writes through to its
//! parent and is checked against the generation of the parent first.
//!
//! Please see [`NESSi`] `cntr_herm_matrix_timestep_moving_view_impl.hpp`.

use crate::types::MovingGreen;
use crate::types::MovingTimeFunction;

#[allow(dead_code)]
/// Throw if the moving object `g` is not initialized (blueprint
/// `_check_initialized` for `ℱᵐᵒᵛ`).
pub(crate) fn check_initialized_moving(g: &MovingGreen) {
    assert!(g.initialized, "the moving object is not initialized");
}

/// Throw if the moving single-time function is not initialized.
pub(crate) fn check_initialized_cfmoving(f: &MovingTimeFunction) {
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
pub(crate) fn check_hamiltonian_moving(tstp: i64, g: &MovingGreen, h: &MovingTimeFunction) {
    check_initialized_cfmoving(h);
    assert!(
        h.tc >= g.tc && h.tstp == tstp && h.size1 == g.ret.size1 && h.size2 == g.ret.size2,
        "H does not fit to the moving Green's function"
    );
}
