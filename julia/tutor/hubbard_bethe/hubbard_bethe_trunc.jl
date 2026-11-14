#
# Project : Lavender
# Source  : hubbard_bethe_trunc.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/hubbard_bethe/hubbard_bethe_trunc.jl \
#         tutor/hubbard_bethe/hubbard_bethe_trunc.toml [--start-only] [--checkpoint]
#
# Interaction quench of the Hubbard model on the Bethe lattice, second
# order perturbation theory. A full Kadanoff-Baym start up to the KB time
# step nt + 1 (= tc + 1 is used for the handoff) is followed by the
# truncated (moving window, cut-off tc) propagation up to the KB time
# step tmax + 1. See bethe_trunc_model.jl.
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_density.dat : KB time step, t, local density (one spin),
#                          densities of the k points (one row per step);
#   * prefix_energy.dat  : KB time step, t, truncated kinetic integral,
#                          last corrector error.
#
# With --checkpoint the moving state after the handoff is written to
# `dir/prefix_checkpoint/` (plain text) and read back before the
# propagation. HDF5 is not supported.
#
# Port of NESSi examples/programs/trunc_bethe_start.cpp and trunc_bethe.cpp.
#

using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "hubbard_bethe_trunc_model.jl"))

"""
    read_params(cfgfile)

Read the TOML parameters as a named tuple.
"""
function read_params(cfgfile::String)
    p = inp_toml(cfgfile, true)
    c, m, s, o = p["contour"], p["model"], p["solver"], p["output"]
    q = (nt = c["nt"]::Int, ntau = c["ntau"]::Int, h = Float64(c["h"]),
         beta = Float64(c["beta"]), tc = c["tc"]::Int, tmax = c["tmax"]::Int,
         U = Float64(m["U"]), epsk = Float64.(m["epsk"]),
         SolveOrder = s["SolveOrder"]::Int, BootstrapMaxIter = s["BootstrapMaxIter"]::Int,
         BootstrapMaxErr = Float64(s["BootstrapMaxErr"]), CorrectorSteps = s["CorrectorSteps"]::Int,
         dir = String(o["dir"]), prefix = String(o["prefix"]))
    1 ≤ q.SolveOrder ≤ 5 || error("SolveOrder must be in 1:5")
    q.tc ≥ 2 * q.SolveOrder + 2 || error("tc must be ≥ 2 SolveOrder + 2")
    q.nt ≥ q.tc || error("nt must be ≥ tc")
    q.tmax ≥ q.tc || error("tmax must be ≥ tc")
    (q.h > 0.0 && q.beta > 0.0 && q.ntau ≥ 2) || error("invalid contour parameters")
    return q
end

"""
    main(cfgfile; start_only = false, checkpoint = false)

Run the full start and the moving propagation. Return a named tuple with
the start, the moving state, and the per-step results.
"""
function main(cfgfile::String; start_only::Bool = false, checkpoint::Bool = false)
    p = read_params(cfgfile)
    mkpath(p.dir)

    start = solve_bethe_start!(p)
    if start_only
        d = [real(density_matrix(start.G, n)[1, 1]) for n = 1:getntime(start.G)]
        all(isfinite, d) || error("non-finite density")
        @printf("full start: density(t_max) = %.10f\n", d[end])
        return (start = start, state = nothing, steps = NamedTuple[])
    end

    state = initialize_moving_state(start, p.tc)
    if checkpoint
        cdir = joinpath(p.dir, p.prefix * "_checkpoint")
        write_checkpoint(cdir, state, p.h)
        state = read_checkpoint(cdir, p.h)
    end

    # Densities of the window at the handoff, KB time steps 1 ⋯ tc + 1
    rows = [(tstp = gettstp(state.G) - j, density = moving_density(state.G, j),
             densk = [moving_density(g, j) for g in state.Gk], ekin = NaN, err = NaN)
            for j = p.tc:-1:0]
    steps = propagate_bethe_moving!(state, p)
    append!(rows, steps)

    open(joinpath(p.dir, p.prefix * "_density.dat"), "w") do io
        println(io, "# tstp  t  density  densk...")
        for r in rows
            @printf(io, "%d %.12e %.12e", r.tstp, (r.tstp - 1) * p.h, r.density)
            foreach(x -> @printf(io, " %.12e", x), r.densk)
            println(io)
        end
    end
    open(joinpath(p.dir, p.prefix * "_energy.dat"), "w") do io
        println(io, "# tstp  t  ekin_trunc  corrector_error")
        for r in steps
            @printf(io, "%d %.12e %.12e %.12e\n", r.tstp, (r.tstp - 1) * p.h, r.ekin, r.err)
        end
    end
    @printf("truncated propagation: density(t_max) = %.10f\n", rows[end].density)

    return (start = start, state = state, steps = steps, rows = rows)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    length(ARGS) ≥ 1 || error("usage: bethe_trunc.jl <config.toml> [--start-only] [--checkpoint] [--quick]")
    main(ARGS[1]; start_only = "--start-only" in ARGS, checkpoint = "--checkpoint" in ARGS)
end
