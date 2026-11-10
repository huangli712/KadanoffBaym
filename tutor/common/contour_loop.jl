#
# Project : Lavender
# Source  : contour_loop.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Model independent stages of the Phase 5 examples: Matsubara iteration,
# bootstrapping, and real-time predictor/corrector propagation. The model
# equations are provided by callbacks in the order of the NESSi programs;
# no self-energy formula is held here.
#
# Events passed to the `log` callback are named tuples with the fields
# `stage` (:matsubara, :bootstrap, :predictor, :corrector), `kb_step`,
# `nessi_step`, `t`, `iter`, and `err` (NaN if not applicable).
#

"""
    ConvergenceError

A Matsubara or bootstrap iteration did not converge within the maximal
number of iterations.
"""
struct ConvergenceError <: Exception
    stage      :: Symbol
    iterations :: Int
    err        :: Float64
end

Base.showerror(io::IO, e::ConvergenceError) =
    print(io, "ConvergenceError: the $(e.stage) iteration did not converge after ",
          e.iterations, " iterations (error = ", e.err, ")")

_noop(_) = nothing

"""
    iterate_matsubara!(step!; max_iter, tol, converged = (iter, err) -> err < tol,
                       log = _noop)

Matsubara iteration. `step!(iter)` performs the iteration `iter = 0, 1,
⋯, max_iter` and returns its error. The loop ends after the first
iteration with `converged(iter, err)`; otherwise a `ConvergenceError` is
thrown. Returns the number of iterations and the last error.
"""
function iterate_matsubara!(step!::Function; max_iter::Integer, tol::Real,
                            converged::Function = (iter, err) -> err < tol,
                            log::Function = _noop)
    err = NaN
    for iter = 0:max_iter
        err = Float64(step!(iter))
        log((stage = :matsubara, kb_step = 0, nessi_step = -1, t = 0.0, iter = iter,
             err = err))
        converged(iter, err) && return (iterations = iter + 1, err = err)
    end
    throw(ConvergenceError(:matsubara, max_iter + 1, err))
end

"""
    iterate_bootstrap!(step!; max_iter, tol, min_iterations, log = _noop)

Bootstrapping. `step!(iter)` updates the start-up time steps and returns
the summed error of the KB time steps `1, ⋯, k + 1` (NESSi `0, ⋯, k`).
The loop ends after the first iteration with `err < tol` and `iter ≥
min_iterations` (NESSi: `iter > 2` for `min_iterations = 3`, the first
iteration for `min_iterations = 0`). Otherwise a `ConvergenceError` is
thrown.
"""
function iterate_bootstrap!(step!::Function; max_iter::Integer, tol::Real,
                            min_iterations::Integer, log::Function = _noop)
    min_iterations ≥ 0 || throw(ArgumentError("min_iterations must be ≥ 0"))
    err = NaN
    for iter = 0:max_iter
        err = Float64(step!(iter))
        log((stage = :bootstrap, kb_step = -1, nessi_step = -1, t = NaN, iter = iter,
             err = err))
        (err < tol && iter ≥ min_iterations) && return (iterations = iter + 1, err = err)
    end
    throw(ConvergenceError(:bootstrap, max_iter + 1, err))
end

"""
    propagate_timesteps!(predictor!, corrector!; first_step, last_step,
                         corrector_steps, h, log = _noop)

Real-time propagation of the KB time steps `first_step, ⋯, last_step`:
for every step `n`, `predictor!(n)` is called once and then
`corrector!(n, iter)` exactly `corrector_steps` times (`iter = 0, ⋯,
corrector_steps - 1`). The NESSi time step is `n - 1`, `t = (n - 1) h`.
"""
function propagate_timesteps!(predictor!::Function, corrector!::Function;
                              first_step::Integer, last_step::Integer,
                              corrector_steps::Integer, h::Real, log::Function = _noop)
    corrector_steps ≥ 1 || throw(ArgumentError("corrector_steps must be ≥ 1"))
    for n = first_step:last_step
        t = (n - 1) * h
        predictor!(n)
        log((stage = :predictor, kb_step = n, nessi_step = n - 1, t = t, iter = -1,
             err = NaN))
        for iter = 0:corrector_steps-1
            corrector!(n, iter)
            log((stage = :corrector, kb_step = n, nessi_step = n - 1, t = t, iter = iter,
                 err = NaN))
        end
    end
    return nothing
end
