#
# Project : Lavender
# Source  : gw.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/gw/gw.jl tutor/gw/gw.toml \
#         [--parallel-axis=serial|k|kernel] [--current-mode=nessi|peierls_corrected]
#
# One-dimensional Hubbard model with nearest neighbour interaction in the
# GW approximation, driven by an electric field pulse (Peierls phase).
# Every k point carries G_k, the GW self-energy Σ_k, the polarization
# P_q, and the screened interaction W_q = V_q + V_q P_q W_q (VIE2 with
# singular kernel). The Hartree and Fock terms enter the single particle
# Hamiltonian. The steps (as in NESSi):
#
#   1. Matsubara iterations, starting from the noninteracting G;
#   2. bootstrapping of the time steps 1 ⋯ SolveOrder + 1;
#   3. time stepping with extrapolation (predictor) and corrector
#      iterations.
#
# Within one iteration of a time step n the stages are
#
#   density -> Hartree/Fock/GW self-energy -> polarization -> Dyson -> W,
#
# every stage reads the time step n of the other k points only before it
# is changed by a later stage, so the k points of one stage are
# independent (this reproduces the MPI snapshot of NESSi without copying
# the time step). With `parallel_axis = "k"` the k points of a stage are
# distributed over the threads; with `"kernel"` the k points are serial
# and the Dyson time step uses the threaded kernels. The results do not
# depend on the choice. The BLAS threads are set to 1 for "k"/"kernel"
# and restored afterwards.
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_observables.dat : KB time step, t, E, A, Ekin, Eint, Etot,
#     density (one spin), current_nessi [, current_peierls_corrected].
#     The KB time step 0 is the Matsubara branch.
#
# The default current (current_mode = "nessi") uses the velocity
# 2 t_h sin(k) of NESSi, which ignores the vector potential. With
# current_mode = "peierls_corrected" the gauge invariant current with the
# velocity 2 t_h sin(k - A) is written in an additional column.
#
# Port of NESSi examples/programs/gw.cpp (without MPI and HDF5).
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "gw_lattice.jl"))
include(joinpath(@__DIR__, "gw_selfenergy.jl"))
include(joinpath(@__DIR__, "gw_kpoints.jl"))

"""
    read_gw_params(cfgfile; parallel_axis = nothing, current_mode = nothing)

Read the TOML parameters as a named tuple. The keyword arguments override
the values of the file.
"""
function read_gw_params(cfgfile::String; parallel_axis = nothing, current_mode = nothing)
    p = inp_toml(cfgfile, true)
    m, c, s, e, o = p["model"], p["contour"], p["solver"], p["pulse"], p["output"]
    q = (Nk = m["Nk"]::Int, HoppingT = Float64(m["HoppingT"]), HubbardU = Float64(m["HubbardU"]),
         V = Float64(m["V"]), MuChem = Float64(m["MuChem"]), beta = Float64(m["beta"]),
         Nt = c["Nt"]::Int, Ntau = c["Ntau"]::Int, h = Float64(c["h"]),
         SolveOrder = s["SolveOrder"]::Int,
         MatsMaxIter = s["MatsMaxIter"]::Int, MatsMaxErr = Float64(s["MatsMaxErr"]),
         BootstrapMaxIter = s["BootstrapMaxIter"]::Int,
         BootstrapMaxErr = Float64(s["BootstrapMaxErr"]),
         TimeMaxErr = Float64(s["TimeMaxErr"]), CorrectorSteps = s["CorrectorSteps"]::Int,
         parallel_axis = Symbol(something(parallel_axis, s["parallel_axis"])),
         current_mode = String(something(current_mode, s["current_mode"])),
         E0 = Float64(e["E0"]), omega = Float64(e["omega"]), t0 = Float64(e["t0"]),
         width = Float64(e["width"]),
         dir = String(o["dir"]), prefix = String(o["prefix"]))
    q.Nk > 1 || error("Nk must be > 1")
    1 ≤ q.SolveOrder ≤ 5 || error("SolveOrder must be in 1:5")
    q.Nt ≥ q.SolveOrder || error("Nt must be ≥ SolveOrder")
    (q.h > 0.0 && q.beta > 0.0 && q.Ntau ≥ 2) || error("invalid contour parameters")
    q.parallel_axis in (:serial, :k, :kernel) ||
        error("parallel_axis must be serial, k, or kernel")
    q.current_mode in ("nessi", "peierls_corrected") ||
        error("current_mode must be nessi or peierls_corrected")
    return q
end

"""
    electric_field(p)

Electric field at the KB time steps, `E[n + 1]` (`E[1] = 0` on the
Matsubara branch), ``E(t) = E_0 e^{-((t - t_0)/w)^2} \\sin(\\omega (t - t_0))``.
"""
function electric_field(p)
    ntime = p.Nt + 1
    E = zeros(Float64, ntime + 1)
    for n = 1:ntime
        t = (n - 1) * p.h
        E[n+1] = p.E0 * exp(-((t - p.t0) / p.width)^2) * sin(p.omega * (t - p.t0))
    end
    return E
end

# Run f(k) for all k points, distributed over the threads if axis == :k
function _foreach_k(f, nk::Int, axis::Symbol)
    if axis == :k && Threads.nthreads() > 1
        Threads.@threads :static for k = 1:nk
            f(k)
        end
    else
        for k = 1:nk
            f(k)
        end
    end
    return nothing
end

"""
    gw_iteration!(n, iter, st)

One iteration at the KB time step `n`: density, Hartree/Fock/GW
self-energy, polarization, Dyson equation, and W. Return the summed
errors of G and W (summed over the k points in ascending order).

*References* : NESSi `gw.cpp`, the bodies of the iteration loops.
"""
function gw_iteration!(n::Int, iter::Int, st)
    p, lat, kps, dens = st.p, st.lat, st.kps, st.dens
    nk, k, axis = lat.nk, p.SolveOrder, p.parallel_axis
    # The density matrix is hermitian. During the bootstrapping G^<(t,t) is
    # anti-hermitian only up to the discretization error, the hermitian
    # part is used (NESSi gw.cpp uses ρ as it is), so that the effective
    # Hamiltonian h_k + Σ^H + Σ^F stays hermitian.
    _foreach_k(nk, axis) do j
        ρ = density_matrix(kps[j].G, n)
        dens[j][n+1] = (ρ + ρ') / 2
    end
    dn = [dens[j][n+1] for j = 1:nk]
    sh = sigma_hartree(lat, dn)
    _foreach_k(nk, axis) do j
        kps[j].SH[n] = sh
        kps[j].SF[n] = sigma_fock(j, lat, dn)
        sigma_gw!(n, j, kps, lat)
    end
    _foreach_k(nk, axis) do j
        polarization!(n, j, kps, lat)
    end
    ele = zeros(Float64, nk)
    bos = zeros(Float64, nk)
    _foreach_k(nk, axis) do j
        ele[j] = step_dyson_with_error!(kps[j], n, iter, lat, p.MuChem, p.beta, p.h, k;
                                        threaded = axis == :kernel)
    end
    _foreach_k(nk, axis) do j
        bos[j] = step_w_with_error!(kps[j], n, k, p.beta, p.h, st.Fsin, st.Q)
    end
    e₁ = 0.0
    e₂ = 0.0
    for j = 1:nk
        e₁ += ele[j]
        e₂ += bos[j]
    end
    return e₁, e₂
end

"""
    solve_gw!(p; io = stdout)

Matsubara iterations, bootstrapping, and time stepping. Return the state
(`p`, `lat`, `kps`, `dens`) and the error log.
"""
function solve_gw!(p; io = stdout)
    ntime = p.Nt + 1
    k = p.SolveOrder
    C = Cn(ntime, p.Ntau, 1, 1, p.Nt * p.h, p.beta)
    lat = GWLattice(p.Nk, p.HoppingT, p.HubbardU, p.V, electric_field(p), p.h, k)
    kps = [GWKPoint(C, lat, j) for j = 1:p.Nk]
    dens = [[zeros(ComplexF64, 1, 1) for _ = 0:ntime] for _ = 1:p.Nk]
    st = (p = p, lat = lat, kps = kps, dens = dens, Q = ℱ(C, BOSE),
          Fsin = Cf(ntime, 1, 1, zero(ComplexF64)))
    errlog = NTuple{5,Any}[]

    # Noninteracting start
    for kp in kps
        from_ham!(kp.G, p.MuChem, kp.hk[0], p.beta, p.h)
    end

    # Matsubara iterations
    converged = false
    for iter = 0:p.MatsMaxIter
        e₁, e₂ = gw_iteration!(0, iter, st)
        push!(errlog, (0, 0, iter, e₁, e₂))
        @printf(io, "tstp = 0 iter = %d  err ele = %.6e  err bos = %.6e\n", iter, e₁, e₂)
        if e₁ < p.MatsMaxErr && e₂ < p.MatsMaxErr
            converged = true
            break
        end
    end
    converged || error("Matsubara iteration not converged")

    # Bootstrapping, KB time steps 1 ⋯ k + 1
    for tb = 1:k+1
        for kp in kps
            if tb == 1
                from_mat!(kp.G)
                from_mat!(kp.W)
            else
                extrapolate_timestep!(tb - 1, kp.G, min(k, tb - 2))
                extrapolate_timestep!(tb - 1, kp.W, min(k, tb - 2))
            end
        end
        converged = false
        for iter = 0:p.BootstrapMaxIter
            e₁ = e₂ = 0.0
            for n = 1:tb
                e₁, e₂ = gw_iteration!(n, iter, st)
            end
            push!(errlog, (1, tb, iter, e₁, e₂))
            if e₁ < p.BootstrapMaxErr && e₂ < p.BootstrapMaxErr && iter > 2
                converged = true
                break
            end
        end
        converged || error("bootstrapping not converged at the time step $tb")
    end

    # Time stepping
    for tb = k+2:ntime
        for kp in kps
            extrapolate_timestep!(tb - 1, kp.G, k)
            extrapolate_timestep!(tb - 1, kp.W, k)
        end
        for iter = 0:p.CorrectorSteps-1
            e₁, e₂ = gw_iteration!(tb, iter, st)
            push!(errlog, (2, tb, iter, e₁, e₂))
            e₁ < p.TimeMaxErr && e₂ < p.TimeMaxErr && iter > 1 && break
        end
    end

    return st, errlog
end

"""
    gw_observables(st)

Observables at the KB time steps `0 ⋯ ntime` as a vector of named tuples.
"""
function gw_observables(st)
    p, lat, kps, dens = st.p, st.lat, st.kps, st.dens
    E = electric_field(p)
    rows = NamedTuple[]
    for n = 0:p.Nt+1
        dn = [dens[j][n+1] for j = 1:lat.nk]
        ekin = kinetic_energy_gw(n, lat, dn)
        eint = interaction_energy_gw(n, kps, lat, p.beta, p.h, p.SolveOrder)
        vn = [fill(ComplexF64(velocity_nessi(lat, kk)), 1, 1) for kk in lat.kpoints]
        vp = [fill(ComplexF64(velocity_peierls(lat, kk, n)), 1, 1) for kk in lat.kpoints]
        loc = sum(lat.kweights[j] * dn[j] for j = 1:lat.nk)
        push!(rows, (tstp = n, t = n == 0 ? 0.0 : (n - 1) * p.h, E = E[n+1],
                     A = lat.A[n+1], Ekin = ekin, Eint = eint, Etot = ekin + eint,
                     density = real(tr(loc)),
                     current_nessi = current(vn, dn, lat.kweights),
                     current_peierls_corrected = current(vp, dn, lat.kweights)))
    end
    return rows
end

"""
    with_gw_blas(f, axis)

Run `f()` with one BLAS thread for the threaded axes, restore the BLAS
thread count afterwards (also after an exception).
"""
function with_gw_blas(f, axis::Symbol)
    axis == :serial && return f()
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(nblas)
    end
end

"""
    main(cfgfile; parallel_axis = nothing, current_mode = nothing, io = stdout)

Run the GW example and write the observables. Return the state and the
observables.
"""
function main(cfgfile::String; parallel_axis = nothing, current_mode = nothing,
              io = stdout)
    p = read_gw_params(cfgfile; parallel_axis = parallel_axis, current_mode = current_mode)
    mkpath(p.dir)
    st, errlog = with_gw_blas(() -> solve_gw!(p; io = io), p.parallel_axis)
    rows = gw_observables(st)
    corrected = p.current_mode == "peierls_corrected"
    open(joinpath(p.dir, p.prefix * "_observables.dat"), "w") do f
        print(f, "# tstp  t  E  A  Ekin  Eint  Etot  density  current_nessi")
        println(f, corrected ? "  current_peierls_corrected" : "")
        for r in rows
            @printf(f, "%d %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e",
                    r.tstp, r.t, r.E, r.A, r.Ekin, r.Eint, r.Etot, r.density, r.current_nessi)
            corrected && @printf(f, " %.12e", r.current_peierls_corrected)
            println(f)
        end
    end
    @printf(io, "GW: Ekin(t_max) = %.10f, Eint(t_max) = %.10f, current(t_max) = %.10f\n",
            rows[end].Ekin, rows[end].Eint, rows[end].current_nessi)
    return (state = st, errors = errlog, observables = rows)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    length(ARGS) ≥ 1 || error("usage: gw.jl <config.toml> [--parallel-axis=...] [--current-mode=...] [--quick]")
    opt(name) = (i = findfirst(a -> startswith(a, "--$name="), ARGS);
                 i === nothing ? nothing : split(ARGS[i], "=")[2])
    main(ARGS[1]; parallel_axis = opt("parallel-axis"), current_mode = opt("current-mode"))
end
