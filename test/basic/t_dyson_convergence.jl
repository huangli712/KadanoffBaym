#
# t_dyson_convergence.jl
#
# To test the convergence order of the real time Dyson solvers with
# respect to the time step, for solve_order = 1, ⋯, 5.
#
# Model: the 2 × 2 block of the free 4 × 4 Hamiltonian [Ha V; V' Hb]
# solves the Dyson equation for Ha with the embedding self-energy
# Σ = V gb V'. The blocks do not commute and the couplings are complex.
# The time interval [0, Tmax] is fixed, and the Matsubara component is
# taken from the exact solution, such that only the real time
# discretization is tested.
#
# The error of a component is the maximum over the time steps of the
# block Frobenius norm distance divided by the number of blocks, i.e.,
# a normalized error per element.
#
# The observed order of the method with solve_order = k is k + 1. The gate
# requires the order k + 1 - 0.5, which is stricter than the order k
# claimed by NESSi. A reduced-order mutant (data from solve_order = 2,
# validated as solve_order = 3) must be rejected by the same gate.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const DC_HA = ComplexF64[-1.0 0.3im; -0.3im 0.5]
const DC_HB = ComplexF64[1.0 0.2+0.1im; 0.2-0.1im 1.5]
const DC_V = ComplexF64[0.4 0.1im; -0.2 0.3]
const DC_TMAX = 4.0
const DC_BETA = 1.0
const DC_NTAU = 201
const DC_MU = 0.1
const DC_CAPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

#
# Errors of the ret, tv, and les components on the grid with Nt time
# intervals. The data are generated with the order `ks`.
#
function dc_errors(ks::Int, Nt::Int)
    h = DC_TMAX / Nt
    nt, ntau, β, μ = Nt + 1, DC_NTAU, DC_BETA, DC_MU
    C4 = Cn(nt, ntau, 4, 4, DC_TMAX, β)
    C2 = Cn(nt, ntau, 2, 2, DC_TMAX, β)
    #
    # Exact solution
    Gf = ℱ(C4, FERMI)
    from_ham!(Gf, μ, [DC_HA DC_V; DC_V' DC_HB], β, h)
    G_exact = ℱ(C2, FERMI)
    for m = 1:ntau
        G_exact.mat[m] = Gf.mat[m][1:2, 1:2]
    end
    for i = 1:nt, m = 1:ntau
        G_exact.lmix[i, m] = Gf.lmix[i, m][1:2, 1:2]
    end
    for i = 1:nt, j = 1:i
        G_exact.ret[i, j] = Gf.ret[i, j][1:2, 1:2]
        G_exact.less[j, i] = Gf.less[j, i][1:2, 1:2]
    end
    #
    # Embedding self-energy
    gb = ℱ(C2, FERMI)
    from_ham!(gb, μ, DC_HB, β, h)
    Σ = ℱ(C2, FERMI)
    for m = 1:ntau
        Σ.mat[m] = DC_V * gb.mat[m] * DC_V'
    end
    for i = 1:nt, m = 1:ntau
        Σ.lmix[i, m] = DC_V * gb.lmix[i, m] * DC_V'
    end
    for i = 1:nt, j = 1:i
        Σ.ret[i, j] = DC_V * gb.ret[i, j] * DC_V'
        Σ.less[j, i] = DC_V * gb.less[j, i] * DC_V'
    end
    H = Cf(nt, 2, 2)
    for i = 0:nt
        H[i] = DC_HA
    end
    #
    # Real time Dyson equation with the exact Matsubara component
    G = ℱ(C2, FERMI)
    G.mat.data .= G_exact.mat.data
    dyson_start!(G, μ, H, Σ, β, h; solve_order = ks)
    for nkb = ks+2:nt
        dyson_timestep!(nkb, G, μ, H, Σ, β, h; solve_order = ks)
    end
    #
    e_ret = maximum(distance(G, G_exact, n, Val(:ret)) / n for n = 1:nt)
    e_tv = maximum(distance(G, G_exact, n, Val(:tv)) / ntau for n = 1:nt)
    e_les = maximum(distance(G, G_exact, n, Val(:les)) / n for n = 1:nt)
    return (e_ret, e_tv, e_les)
end

#
# Order gates of the three components
#
function dc_gate(Ns, ks::Int)
    E = [dc_errors(ks, N) for N in Ns]
    for c = 1:3
        errs = [e[c] for e in E]
        @test all(isfinite, errs)
        test_order_gate(Ns, errs, ks + 1, DC_CAPS[ks]; order_tol = 0.5)
    end
end

#
# Reduced-order mutant: data from solve_order = 2, validated as 3
#
function dc_mutant(Ns)
    E = [dc_errors(2, N) for N in Ns]
    for c = 1:3
        errs = [e[c] for e in E]
        @test !check_order_gate(Ns, errs, 3 + 1, DC_CAPS[3]; order_tol = 0.5)[1]
    end
end

@testset "dyson convergence ($(kb_test_profile()))" begin
    if kb_release()
        @testset "release: full grids" begin
            @test kb_test_profile() == "release"
            Ns = [50, 100, 200, 400]
            for k = 1:5
                t = @elapsed dc_gate(Ns, k)
                println("    dyson convergence (release) solve_order = $k: $(round(t, digits = 1)) s")
            end
            dc_mutant(Ns)
        end
    else
        @testset "quick: reduced grids" begin
            @test kb_test_profile() == "quick"
            Ns = [12, 24, 48, 96]
            for k = 1:5
                t = @elapsed dc_gate(Ns, k)
                println("    dyson convergence (quick) solve_order = $k: $(round(t, digits = 1)) s")
            end
            dc_mutant(Ns)
        end
    end
end
