#
# Project : Lavender
# Source  : ness2_bethe_prop.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/ness2_bethe_prop/ness2_bethe_prop.jl \
#         tutor/ness2_bethe_prop/ness2_bethe_prop.toml
#
# Real-time propagation of the DMFT equations of the Hubbard model on the
# Bethe lattice (bandwidth 4) in equilibrium. The local Green's function
# solves the Dyson equation with the kernel K = Σ + G (Δ = G), where the
# second order self-energy is
#
#   Σ(t,t') = -i 𝒢(t,t') W(t',t),  W(t,t') = U(t) [i 𝒢(t,t') 𝒢(t',t)] U(t'),
#
# with 𝒢 = G (self-consistent PT) or the Weiss field (IPT). The steps:
#
#   1. Matsubara iterations, starting from the noninteracting Bethe
#      Green's function (FIXPOINT or FOURIER Matsubara Dyson solver);
#   2. bootstrapping: the time steps 1 ⋯ SolveOrder + 1 are iterated;
#   3. time stepping with extrapolation (predictor) and CorrectorSteps
#      corrector iterations per time step.
#
# Finally the last time slice is converted to a steady-state Green's
# function (cntr2ness!) and back to a contour Green's function assuming
# time translation invariance (ness2cntr!). In equilibrium both densities
# agree, and the deviation of G^R from the time translation invariant
# form measures the quality of the propagation.
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_errors.dat  : stage (0 Matsubara, 1 bootstrapping, 2 time
#                          stepping), time step (KB), iteration, error;
#   * prefix_density.dat : time step (KB, 0 for Matsubara), t, density;
#   * prefix_ness.dat    : density at the last time step (contour and
#                          steady state) and the deviation of G^R from
#                          time translation invariance.
#
# Port of NESSi examples/programs/ness2_bethe_prop.cpp. HDF5 is not
# supported. The time steps are counted from 1 (KB), NESSi tstp = -1 is
# the KB time step 0 and NESSi tstp = n is the KB time step n + 1.
#

using Printf
using KadanoffBaym

"""
    sigma_timestep!(tstp, Σ, 𝒢, W, Uf)

Second order self-energy at the time step `tstp`: ``W = U [i 𝒢 𝒢] U``
(bosonic) and ``Σ = -i 𝒢 W``.
"""
function sigma_timestep!(tstp::Int, Σ::ℱ, 𝒢::ℱ, W::ℱ, Uf::Cf)
    bubble1!(tstp, W, 𝒢, 𝒢)
    smul!(Uf, W, tstp)
    smul!(W, Uf, tstp)
    bubble2!(tstp, Σ, 𝒢, W)
    smul!(Σ, tstp, -1.0)
end

"""
    main(cfgfile::String)

Read the parameters from `cfgfile` (TOML), run the propagation, and write
the results. Return the densities and the errors as a named tuple.
"""
function main(cfgfile::String)
    p = inp_toml(cfgfile, true)
    nt = p["contour"]["nt"]::Int
    ntau = p["contour"]["ntau"]::Int
    h = Float64(p["contour"]["h"])
    β = Float64(p["contour"]["beta"])
    U = Float64(p["model"]["U"])
    μ = Float64(p["model"]["mu"])
    k = p["solver"]["SolveOrder"]::Int
    ipt = p["solver"]["ipt_flag"]::Int
    method = Symbol(p["solver"]["mat_method"])
    maxiter = p["solver"]["BootstrapMaxIter"]::Int
    maxerr = Float64(p["solver"]["BootstrapMaxErr"])
    ncorr = p["solver"]["CorrectorSteps"]::Int
    dir = String(p["output"]["dir"])
    prefix = String(p["output"]["prefix"])
    @assert ntau ≥ 2 && 1 ≤ k ≤ 5 && nt ≥ k
    @assert h > 0.0 && β > 0.0 && maxiter ≥ 0 && ncorr ≥ 0
    @assert method in (:fixpoint, :fourier)
    mkpath(dir)

    ntime = nt + 1
    C = Cn(ntime, ntau, 1, 1, nt * h, β)
    G = ℱ(C, FERMI)
    Gweiss = ℱ(C, FERMI)
    Σ = ℱ(C, FERMI)
    K = ℱ(C, FERMI)
    W = ℱ(C, BOSE)
    Uf = Cf(ntime, 1, 1, ComplexF64(U))
    eloc = Cf(ntime, 1, 1, zero(ComplexF64))

    # K = Σ + G at the time step tstp
    function kernel!(tstp)
        memcpy!(Σ, K, tstp)
        incr!(K, G, tstp, 1.0)
    end

    ferr = open(joinpath(dir, prefix * "_errors.dat"), "w")
    println(ferr, "# stage  tstp  iteration  error")
    errors = Float64[]
    function record(stage, tstp, iter, err)
        push!(errors, err)
        @printf(ferr, "%d %d %d %.12e\n", stage, tstp, iter, err)
    end

    # Matsubara: noninteracting Bethe Green's function as initial guess
    green_equilibrium_mat_bethe!(G, β)
    memcpy!(G, Gweiss, 0)
    memset!(Σ, 0, zero(ComplexF64))
    kernel!(0)
    gtemp = 𝒻(0, ntau, 1, 1, FERMI)
    itermax = maxiter
    for iter = 0:maxiter
        memcpy!(G, gtemp, 0)
        if ipt != 0
            dyson_mat!(Gweiss, G, μ, eloc, β; solve_order = k, method = method)
        else
            memcpy!(G, Gweiss, 0)
        end
        sigma_timestep!(0, Σ, Gweiss, W, Uf)
        kernel!(0)
        dyson_mat!(G, K, μ, eloc, β; solve_order = k, method = method)
        err = distance(gtemp, G, 0)
        record(0, 0, iter, err)
        if err < maxerr && iter > 3
            itermax = iter
            break
        end
    end

    # Bootstrapping: time steps 1 ⋯ k + 1
    nb = k + 1
    gtemp = 𝒻(nb, ntau, 1, 1, FERMI)
    set_t0_from_mat!(G)
    for iter = 0:maxiter
        memcpy!(G, gtemp, nb)
        if ipt != 0
            dyson_start!(Gweiss, μ, eloc, G, β, h; solve_order = k)
        else
            for n = 1:nb
                memcpy!(G, Gweiss, n)
            end
        end
        for n = 1:nb
            sigma_timestep!(n, Σ, Gweiss, W, Uf)
            kernel!(n)
        end
        dyson_start!(G, μ, eloc, K, β, h; solve_order = k)
        err = distance(gtemp, G, nb)
        record(1, nb, iter, err)
        err < maxerr && iter > 3 && break
    end

    # Time stepping
    for tstp = nb+1:ntime
        gtemp = 𝒻(tstp, ntau, 1, 1, FERMI)
        extrapolate_timestep!(tstp - 1, G, k)
        for iter = 0:ncorr
            memcpy!(G, gtemp, tstp)
            if ipt != 0
                dyson_timestep!(tstp, Gweiss, μ, eloc, G, β, h; solve_order = k)
            else
                memcpy!(G, Gweiss, tstp)
            end
            sigma_timestep!(tstp, Σ, Gweiss, W, Uf)
            kernel!(tstp)
            dyson_timestep!(tstp, G, μ, eloc, K, β, h; solve_order = k)
            err = distance(gtemp, G, tstp)
            record(2, tstp, iter, err)
        end
    end
    close(ferr)

    # Densities, n = n↑ + n↓
    dens = [2.0 * real(density_matrix(G, tstp)[1, 1]) for tstp = 0:ntime]
    open(joinpath(dir, prefix * "_density.dat"), "w") do io
        println(io, "# tstp  t  density")
        @printf(io, "%d %.12e %.12e\n", 0, 0.0, dens[1])
        for tstp = 1:ntime
            @printf(io, "%d %.12e %.12e\n", tstp, (tstp - 1) * h, dens[tstp+1])
        end
    end

    # Steady-state view of the last time slice and back
    Gn = HermMatrixNESS(GridInfo(2 * ntime, h), 1)
    cntr2ness!(Gn, G, ntime)
    dens_ness = 2.0 * real(density_matrix(Gn, FERMI)[1, 1])
    Gtti = ℱ(C, FERMI)
    ness2cntr!(Gtti, Gn)
    tti = maximum(abs(G.ret[n, m][1, 1] - Gtti.ret[n, m][1, 1]) for n = 1:ntime for m = 1:n)
    open(joinpath(dir, prefix * "_ness.dat"), "w") do io
        println(io, "# density_contour  density_ness  max|G^R - G^R_tti|")
        @printf(io, "%.12e %.12e %.12e\n", dens[end], dens_ness, tti)
    end

    @printf("Matsubara iterations = %d, density(t_max) = %.8f, density(NESS) = %.8f, TTI deviation = %.3e\n",
            itermax, dens[end], dens_ness, tti)

    return (density = dens, density_ness = dens_ness, tti = tti,
            mat_iterations = itermax, errors = errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: ness2_bethe_prop.jl <config.toml>")
    main(ARGS[1])
end
