#
# Project : Lavender
# Source  : trunc_bethe_model.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Model functions of the truncated (moving window) propagation of the
# Hubbard model on the Bethe lattice (bandwidth 4), second order
# perturbation theory, interaction quench U = 0 -> U at t = 0. The local
# Green's function solves the Dyson equation with the kernel K = Σ + G
# (hybridization Δ = G), the lattice Green's functions G_k solve the
# Dyson equation with ε_k and Σ. The self-energy is
#
#   W(t,t') = U(t) [i G(t,t') G(t',t)] U(t'),   Σ(t,t') = -i G(t,t') W(t',t).
#
# Port of NESSi examples/programs/trunc_bethe_start.cpp (full start) and
# trunc_bethe.cpp (moving propagation). HDF5 is not supported, the full
# start is handed over in memory (optionally through plain text
# checkpoint files). Time steps: NESSi tstp = n is the KB time step n + 1.
#

using Printf
using TOML
using KadanoffBaym

#=
### *Self-energy and Kernel*
=#

"""
    update_bethe_selfenergy!(tstp, Σ, G, W, U)

Second order self-energy at the time step `tstp` of the full contour
functions (`W` is a bosonic work function, `U` the interaction).
"""
function update_bethe_selfenergy!(tstp::Int, Σ::ℱ, G::ℱ, W::ℱ, U::Cf)
    bubble1!(tstp, W, G, G)
    smul!(U, W, tstp)
    smul!(W, U, tstp)
    bubble2!(tstp, Σ, G, W)
    smul!(Σ, tstp, -1.0)
    return Σ
end

"""
    update_bethe_selfenergy!(Σ, G, W, U)

Second order self-energy at the leading row of the moving functions.
"""
function update_bethe_selfenergy!(Σ::ℱᵐᵒᵛ, G::ℱᵐᵒᵛ, W::ℱᵐᵒᵛ, U::Cfᵐᵒᵛ)
    n = gettstp(G)
    bubble1!(n, W, G, G)
    w = timestep_view(W, 0)
    smul!(U, w)
    smul!(w, U)
    bubble2!(n, Σ, G, W)
    smul!(timestep_view(Σ, 0), -1.0)
    return Σ
end

"""
    update_bethe_kernel!(tstp, K, Σ, G)
    update_bethe_kernel!(K, Σ, G)

Kernel K = Σ + G at the time step `tstp` (full) or at the leading row
(moving).
"""
function update_bethe_kernel!(tstp::Int, K::ℱ, Σ::ℱ, G::ℱ)
    memcpy!(Σ, K, tstp)
    incr!(K, G, tstp, 1.0)
    return K
end
#
function update_bethe_kernel!(K::ℱᵐᵒᵛ, Σ::ℱᵐᵒᵛ, G::ℱᵐᵒᵛ)
    k = timestep_view(K, 0)
    memcpy!(timestep_view(Σ, 0), k)
    incr!(k, timestep_view(G, 0), 1.0)
    return K
end

# Density i ξ G^<(t,t) of the row `delti` (one spin, as in NESSi)
moving_density(G::ℱᵐᵒᵛ, delti::Int = 0) = real(im * getsign(G) * G.less[delti, 0][1, 1])

#=
### *Full Start*
=#

"""
    solve_bethe_start!(p)

Full Kadanoff-Baym start (NESSi trunc_bethe_start.cpp): noninteracting
Bethe Matsubara Green's function, bootstrapping of the first SolveOrder + 1
time steps, time stepping with CorrectorSteps corrector iterations up to
the KB time step `nt + 1`, and the full lattice Green's functions `G_k`.
Return a named tuple with `G`, `Σ`, `K`, `Gk`, `U`, `eloc`, `epsk`, and the
error log.
"""
function solve_bethe_start!(p)
    nt, ntau, h, β = p.nt, p.ntau, p.h, p.beta
    k, μ = p.SolveOrder, 0.0
    ntime = nt + 1
    C = Cn(ntime, ntau, 1, 1, nt * h, β)
    G = ℱ(C, FERMI)
    Σ = ℱ(C, FERMI)
    K = ℱ(C, FERMI)
    W = ℱ(C, BOSE)
    # U = 0 on the Matsubara branch (quench at t = 0)
    U = Cf(ntime, 1, 1, ComplexF64(p.U))
    U[0] = zeros(ComplexF64, 1, 1)
    eloc = Cf(ntime, 1, 1, zero(ComplexF64))
    errlog = NTuple{4,Any}[]

    green_equilibrium_mat_bethe!(G, β)
    memset!(Σ, 0, zero(ComplexF64))
    update_bethe_kernel!(0, K, Σ, G)

    # Bootstrapping, KB time steps 1 ⋯ k + 1
    nb = k + 1
    gtemp = 𝒻(nb, ntau, 1, 1, FERMI)
    set_t0_from_mat!(G)
    converged = false
    for iter = 0:p.BootstrapMaxIter
        memcpy!(G, gtemp, nb)
        for n = 1:nb
            update_bethe_selfenergy!(n, Σ, G, W, U)
            update_bethe_kernel!(n, K, Σ, G)
        end
        dyson_start!(G, μ, eloc, K, β, h; solve_order = k)
        err = distance(gtemp, G, nb)
        push!(errlog, (1, nb, iter, err))
        if err < p.BootstrapMaxErr && iter > 3
            converged = true
            break
        end
    end
    converged || error("bootstrapping not converged")

    # Time stepping
    for tstp = nb+1:ntime
        gtemp = 𝒻(tstp, ntau, 1, 1, FERMI)
        extrapolate_timestep!(tstp - 1, G, k)
        for iter = 0:p.CorrectorSteps
            memcpy!(G, gtemp, tstp)
            update_bethe_selfenergy!(tstp, Σ, G, W, U)
            update_bethe_kernel!(tstp, K, Σ, G)
            dyson_timestep!(tstp, G, μ, eloc, K, β, h; solve_order = k)
            push!(errlog, (2, tstp, iter, distance(gtemp, G, tstp)))
        end
    end

    # Lattice Green's functions
    Gk = ℱ[]
    epsk = Cf[]
    for ε in p.epsk
        e = Cf(ntime, 1, 1, ComplexF64(ε))
        g = ℱ(C, FERMI)
        dyson!(g, μ, e, Σ, β, h; solve_order = k)
        push!(Gk, g)
        push!(epsk, e)
    end

    return (G = G, Σ = Σ, K = K, Gk = Gk, U = U, eloc = eloc, epsk = epsk,
            errors = errlog)
end

#=
### *Full -> Moving Handoff*
=#

# Moving copy of the full function `X` at the KB time step n₀
function _moving_copy(X::ℱ, tc::Int, n₀::Int)
    M = ℱᵐᵒᵛ(tc, getdims(X)[1], getsign(X))
    set_from_G_backward!(M, X, n₀)
    return M
end
#
function _moving_copy(f::Cf, tc::Int, n₀::Int)
    M = Cfᵐᵒᵛ(tc, getdims(f)[1])
    set_from_function_backward!(M, f, n₀)
    return M
end

"""
    initialize_moving_state(start, tc)

Moving windows of the full start `start` at the KB time step `tc + 1`
(NESSi set_from_G_backward at NESSi tstp = tc). The kernel is set to
K = G + Σ on all rows. Return a named tuple.
"""
function initialize_moving_state(start, tc::Int)
    n₀ = tc + 1
    G = _moving_copy(start.G, tc, n₀)
    Σ = _moving_copy(start.Σ, tc, n₀)
    K = _moving_copy(start.G, tc, n₀)
    for delti = 0:tc
        incr!(timestep_view(K, delti), timestep_view(Σ, delti), 1.0)
    end
    # Bosonic work function of the bubbles
    C = Cn(getntime(start.G), getntau(start.G), 1, 1, 1.0, 1.0)
    W = _moving_copy(ℱ(C, BOSE), tc, n₀)
    Gk = [_moving_copy(g, tc, n₀) for g in start.Gk]
    U = _moving_copy(start.U, tc, n₀)
    eloc = _moving_copy(start.eloc, tc, n₀)
    epsk = [_moving_copy(e, tc, n₀) for e in start.epsk]
    return (G = G, Σ = Σ, K = K, W = W, Gk = Gk, U = U, eloc = eloc, epsk = epsk)
end

const _CHECKPOINT_OBJECTS = (:G, :Σ, :K, :W)

"""
    write_checkpoint(dir, state, h)

Write the moving state as plain text (`KB_MOVING_V1`) files into the
directory `dir`, together with `meta.toml` (tc, tstp, h, number of k
points).
"""
function write_checkpoint(dir::String, state, h::Float64)
    mkpath(dir)
    for s in _CHECKPOINT_OBJECTS
        write(joinpath(dir, "$s.dat"), getfield(state, s))
    end
    for (i, g) in enumerate(state.Gk)
        write(joinpath(dir, "Gk_$i.dat"), g)
    end
    for s in (:U, :eloc)
        write(joinpath(dir, "$s.dat"), getfield(state, s))
    end
    for (i, e) in enumerate(state.epsk)
        write(joinpath(dir, "epsk_$i.dat"), e)
    end
    meta = Dict("tc" => state.G.tc, "tstp" => gettstp(state.G), "h" => h,
                "nk" => length(state.Gk))
    open(joinpath(dir, "meta.toml"), "w") do io
        TOML.print(io, meta)
    end
    return dir
end

"""
    read_checkpoint(dir, h)

Read the moving state written by `write_checkpoint`. The metadata (tc,
tstp, h) and every file (header, sign, size, completeness) are validated.
"""
function read_checkpoint(dir::String, h::Float64)
    meta = TOML.parsefile(joinpath(dir, "meta.toml"))
    tc = meta["tc"]::Int
    tstp = meta["tstp"]::Int
    nk = meta["nk"]::Int
    Float64(meta["h"]) == h || throw(ArgumentError("checkpoint: h differs"))
    rd(fn, X) = (read!(joinpath(dir, fn), X); X)
    G = rd("G.dat", ℱᵐᵒᵛ(tc, 1, FERMI))
    Σ = rd("Σ.dat", ℱᵐᵒᵛ(tc, 1, FERMI))
    K = rd("K.dat", ℱᵐᵒᵛ(tc, 1, FERMI))
    W = rd("W.dat", ℱᵐᵒᵛ(tc, 1, BOSE))
    Gk = [rd("Gk_$i.dat", ℱᵐᵒᵛ(tc, 1, FERMI)) for i = 1:nk]
    U = rd("U.dat", Cfᵐᵒᵛ(tc, 1))
    eloc = rd("eloc.dat", Cfᵐᵒᵛ(tc, 1))
    epsk = [rd("epsk_$i.dat", Cfᵐᵒᵛ(tc, 1)) for i = 1:nk]
    for X in (G, Σ, K, W, Gk..., U, eloc, epsk...)
        (X.initialized && gettstp(X) == tstp) ||
            throw(ArgumentError("checkpoint: inconsistent state"))
    end
    return (G = G, Σ = Σ, K = K, W = W, Gk = Gk, U = U, eloc = eloc, epsk = epsk)
end

#=
### *Moving Propagation*
=#

# Forward the constant single-time function and set the leading slot
function _forward_const!(f::Cfᵐᵒᵛ)
    v = f[1]
    forward!(f)
    f[0] = v
    return f
end

"""
    propagate_bethe_moving!(state, p; io = nothing)

Truncated propagation (NESSi trunc_bethe.cpp) up to the KB time step
`tmax + 1`. Every step: forward of all windows, predictor
(extrapolation), CorrectorSteps + 1 corrector iterations (self-energy,
kernel, local moving Dyson), the lattice moving Dyson equations, and the
observables. Return a vector of named tuples (`tstp`, `density`,
`densk`, `ekin`, `err`), one per time step. The time steps are never
parallelized.
"""
function propagate_bethe_moving!(state, p)
    k, h, μ = p.SolveOrder, p.h, 0.0
    I = Integrator(k)
    G, Σ, K, W = state.G, state.Σ, state.K, state.W
    out = NamedTuple[]
    for n = gettstp(G)+1:p.tmax+1
        for X in (Σ, K, G, W)
            forward!(X)
        end
        _forward_const!(state.U)
        _forward_const!(state.eloc)
        extrapolate_timestep!(G, k)
        err = 0.0
        for _ = 0:p.CorrectorSteps
            gtmp = get_timestep(G, 0)
            update_bethe_selfenergy!(Σ, G, W, state.U)
            update_bethe_kernel!(K, Σ, G)
            dyson_timestep!(n, G, μ, state.eloc, K, h; solve_order = k)
            err = distance(timestep_view(G, 0), gtmp)
        end
        densk = Float64[]
        for (g, e) in zip(state.Gk, state.epsk)
            forward!(g)
            _forward_const!(e)
            dyson_timestep!(n, g, μ, e, Σ, h; solve_order = k)
            push!(densk, moving_density(g))
        end
        ekin = kinetic_energy_trunc(n, G, G, I, h)
        push!(out, (tstp = n, density = moving_density(G), densk = densk,
                    ekin = ekin, err = err))
    end
    return out
end
