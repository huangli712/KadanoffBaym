#
# Project : Lavender
# Source  : test_equilibrium.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#
# Usage:
#
#   julia --project=. tutor/test_equilibrium/test_equilibrium.jl \
#         tutor/test_equilibrium/test_equilibrium.toml out.dat
#
# The Matsubara Dyson equation of a single level `eps1` coupled to a second
# level `eps2` (hopping `i lambda`) is solved by embedding: the 1 × 1
# Green's function satisfies the Dyson equation with Σ = lambda² g2, where
# g2 is the free Green's function of `eps2`. The solution is compared with
# the (1,1) element of the exact 2 × 2 Green's function.
#
# Output: one line `err_fixpoint err_cg`, where err = distance / Ntau.
#
# Port of NESSi examples/programs/test_equilibrium.cpp. The FOURIER method
# of NESSi is deferred to Phase 7, so the columns differ from the original
# `err_fourier err_fixpoint`.
#

using Printf
using KadanoffBaym

"""
    main(cfgfile::String, outfile::String)

Read the parameters from `cfgfile` (TOML) and write the errors of the
FIXPOINT and CG methods into `outfile` (overwritten).
"""
function main(cfgfile::String, outfile::String)
    p = inp_toml(cfgfile, true)
    Ntau = p["Ntau"]::Int
    k = p["SolveOrder"]::Int
    β = Float64(p["beta"])
    ε₁ = Float64(p["eps1"])
    ε₂ = Float64(p["eps2"])
    λ = Float64(p["lambda"])
    μ = Float64(p["mu"])
    @assert iseven(Ntau) && Ntau ≥ 4
    @assert 1 ≤ k ≤ 5

    # Only the Matsubara component is needed, a minimal real time grid
    # is used. Ntau is the number of intervals.
    h = 1.0
    ntau = Ntau + 1
    C1 = Cn(2, ntau, 1, 1, h, β)
    C2 = Cn(2, ntau, 2, 2, h, β)

    # Exact Green's function of the 2 × 2 Hamiltonian
    G2 = ℱ(C2, FERMI)
    from_ham!(G2, μ, ComplexF64[ε₁ im*λ; -im*λ ε₂], β, h)
    Gx = ℱ(C1, FERMI)
    for m = 1:ntau
        Gx.mat[m] = G2.mat[m][1:1, 1:1]
    end

    # Embedding self-energy and Hamiltonian
    Σ = ℱ(C1, FERMI)
    from_ham!(Σ, μ, fill(ComplexF64(ε₂), 1, 1), β, h)
    smul!(Σ, 0, λ^2)
    H = Cf(2, 1, 1)
    for i = 0:2
        H[i] = fill(ComplexF64(ε₁), 1, 1)
    end

    err = Float64[]
    for method in (:fixpoint, :cg)
        G = ℱ(C1, FERMI)
        dyson_mat!(G, Σ, μ, H, β; solve_order = k, method = method)
        push!(err, distance(Gx, G, 0) / Ntau)
    end

    open(outfile, "w") do io
        println(io, "# err_fixpoint err_cg")
        @printf(io, "%.5e %.5e\n", err[1], err[2])
    end
    @printf("Ntau = %d, SolveOrder = %d: err_fixpoint = %.5e, err_cg = %.5e\n",
            Ntau, k, err[1], err[2])

    return err
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: test_equilibrium.jl <config.toml> <output.dat>")
    main(ARGS[1], ARGS[2])
end
