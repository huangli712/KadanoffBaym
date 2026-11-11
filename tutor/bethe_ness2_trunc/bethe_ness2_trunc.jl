#
# Project : Lavender
# Source  : bethe_ness2_trunc.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/bethe_ness2_trunc/bethe_ness2_trunc.jl \
#         tutor/bethe_ness2_trunc/bethe_ness2_trunc.toml
#
# Truncated (moving window) propagation of the Hubbard model on the Bethe
# lattice, started from the steady state of tutor/bethe_ness2 (second
# order perturbation theory, IPT or self-consistent PT). The steady-state
# G, Σ, and 𝒢 (Weiss field) are computed first with the parameters of the
# sections [grid], [model], [solver] (see bethe_ness2.toml), downsampled
# by the factor `downsampling`, and converted to moving windows with the
# cut-off tc. The relative NESS step 0 is the KB time step 1, the first
# forward leads to the KB time step 2.
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_trunc_density.dat : relative step, KB time step, density (one
#                                spin), last corrector error.
#
# Port of NESSi examples/programs/ness2_bethe_trunc.cpp. HDF5 is not
# supported, the steady state is handed over in memory.
#

using Printf
using KadanoffBaym

# The steady-state solver lives in its own module (it defines `main`)
module NessBethe
include(joinpath(@__DIR__, "..", "bethe_ness2", "bethe_ness2.jl"))
end
include(joinpath(@__DIR__, "..", "bethe_trunc", "bethe_trunc_model.jl"))

"""
    initialize_from_ness(Gness, tc, factor)

Moving window (cut-off `tc`) of the steady-state Green's function `Gness`
downsampled by `factor`. The window is initialized at the KB time step 1.
"""
function initialize_from_ness(Gness::NessHermMat, tc::Int, factor::Int)
    Gm = ℱᵐᵒᵛ(tc, 1, FERMI)
    init_from_ness!(Gm, downsample(Gness, factor))
    return Gm
end

"""
    main(cfgfile::String)

Compute the steady state, convert it to moving windows, and propagate.
Return a named tuple with the initial windows, the final state, and the
per-step results.
"""
function main(cfgfile::String)
    p = inp_toml(cfgfile, true)
    t = p["trunc"]
    tc = t["tc"]::Int
    tmax = t["tmax"]::Int
    factor = t["downsampling"]::Int
    ncorr = t["CorrectorSteps"]::Int
    k = t["SolveOrder"]::Int
    U = Float64(p["model"]["U"])
    μ = Float64(p["model"]["mu"])
    ipt = p["solver"]["ipt_flag"]::Int
    h = Float64(p["grid"]["h"]) * factor
    dir = String(p["output"]["dir"])
    prefix = String(p["output"]["prefix"])
    (1 ≤ k ≤ 5 && tc ≥ 2k + 2 && tmax ≥ 1 && factor ≥ 1 && ncorr ≥ 0) ||
        error("invalid [trunc] parameters")

    # Steady state (tutor/bethe_ness2)
    ness = NessBethe.main(cfgfile)

    G = initialize_from_ness(ness.G, tc, factor)
    Σ = initialize_from_ness(ness.S, tc, factor)
    Gw = initialize_from_ness(ness.Gweiss, tc, factor)
    K = initialize_from_ness(ness.G, tc, factor)
    for delti = 0:tc
        incr!(timestep_view(K, delti), timestep_view(Σ, delti), 1.0)
    end
    W = ℱᵐᵒᵛ(tc, 1, BOSE)
    init_from_ness!(W, NessHermMat(NessGrid(4 * (tc + 1), h), 1))
    Uf = Cfᵐᵒᵛ(tc, 1)
    eloc = Cfᵐᵒᵛ(tc, 1)
    cf = Cf(tc + 2, 1, 1, ComplexF64(U))
    init_from_function!(Uf, cf, tc + 1)
    init_from_function!(eloc, Cf(tc + 2, 1, 1, zero(ComplexF64)), tc + 1)
    initial = (G = deepcopy(G), Σ = deepcopy(Σ), Gweiss = deepcopy(Gw))
    # The functions are constant, their window is aligned with the NESS
    # windows (KB time step 1)
    for f in (Uf, eloc)
        f.tstp = 1
    end

    steps = NamedTuple[]
    for rel = 1:tmax
        for X in (Σ, K, G, Gw, W)
            forward!(X)
        end
        _forward_const!(Uf)
        _forward_const!(eloc)
        n = gettstp(G)
        extrapolate_timestep!(G, k)
        err = 0.0
        for _ = 0:ncorr
            gtmp = copy_timestep(G, 0)
            if ipt != 0
                dyson_timestep!(n, Gw, μ, eloc, G, h; solve_order = k)
            else
                memcpy!(timestep_view(G, 0), timestep_view(Gw, 0))
            end
            update_bethe_selfenergy!(Σ, Gw, W, Uf)
            update_bethe_kernel!(K, Σ, G)
            dyson_timestep!(n, G, μ, eloc, K, h; solve_order = k)
            err = distance(timestep_view(G, 0), gtmp)
        end
        push!(steps, (rel = rel, tstp = n, density = moving_density(G), err = err))
    end

    mkpath(dir)
    open(joinpath(dir, prefix * "_trunc_density.dat"), "w") do io
        println(io, "# rel  tstp  density  corrector_error")
        @printf(io, "%d %d %.12e %.12e\n", 0, 1, moving_density(initial.G), 0.0)
        for s in steps
            @printf(io, "%d %d %.12e %.12e\n", s.rel, s.tstp, s.density, s.err)
        end
    end
    @printf("density(NESS) = %.10f, density(t_max) = %.10f\n",
            moving_density(initial.G), steps[end].density)

    return (ness = ness, initial = initial, G = G, steps = steps)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    length(ARGS) ≥ 1 || error("usage: bethe_ness2_trunc.jl <config.toml> [--quick]")
    main(ARGS[1])
end
