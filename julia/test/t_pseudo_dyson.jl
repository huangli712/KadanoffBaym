#
# t_pseudo_dyson.jl
#
# To test the pseudo-particle Dyson solvers: pseudo_dyson_mat!(),
# pseudo_dyson_start!(), pseudo_dyson_timestep!(), and pseudo_dyson!().
#
# The expected values are the closed forms of the atomic limit (Σ = 0)
# with a hermitian Hamiltonian H(t) = H0 + a(t), where the scalar a(t)
# commutes with H0:
#
#   Gᴹ(τ)       = -exp[(λ0 - H0) τ],
#   G^{>}(t,t') = -i exp[-i (H0 - λ0)(t - t') - i ∫_{t'}^{t} a],
#   G^⌉(t,τ)    = ξ G^{>}(t,0) exp[(λ0 - H0) (β - τ)],
#   G^<(t,t')   = -i ξ U(t) exp[-(H0 - λ0) β] U(t')†,
#
# where U(t) = exp[-i (H0 - λ0) t - i ∫_0^t a]. Note that the left-mixing
# component uses the Hamiltonian at the Matsubara axis H[0] = H0. There is
# no pseudo-particle Catch test in NESSi. The comparison with NESSi for a
# nonzero self-energy is in test/t_pseudo.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const PD_H0 = C64[0.3 0.2-0.1im; 0.2+0.1im -0.4]
const PD_λ0 = 0.5
pd_a(t) = 0.3 * sin(2.0 * t)
pd_A(t) = 0.15 * (1.0 - cos(2.0 * t)) # ∫_0^t a

# Hamiltonian H[0] = H0, H[n] = H0 + a(t)
function pd_ham(nt, h, H0)
    d = size(H0, 1)
    H = Cf(nt, d, d)
    H[0] = H0
    for n = 1:nt
        H[n] = H0 + pd_a((n - 1) * h) * Matrix{C64}(I, d, d)
    end
    return H
end

# Maximum errors of the four components with respect to the atomic limit
function pd_errors(G, H0, λ0, β, h)
    nt, ntau = getntime(G), getntau(G)
    d = size(H0, 1)
    ξ = getsign(G)
    δτ = β / (ntau - 1)
    A = H0 - λ0 * I
    U(t) = exp(-im * A * t - im * pd_A(t) * I)
    t(n) = (n - 1) * h
    errm = maximum(norm(G.mat[m] + exp(-A * (m - 1) * δτ)) for m = 1:ntau)
    errg = maximum(norm(G.ret[n,j] + im * U(t(n)) * U(t(j))') for n = 1:nt for j = 1:n)
    errt = maximum(norm(G.lmix[n,m] - ξ * G.ret[n,1] * exp(-A * (β - (m - 1) * δτ)))
                   for n = 1:nt for m = 1:ntau)
    errl = maximum(norm(G.less[j,n] - (-im * ξ) * U(t(j)) * exp(-A * β) * U(t(n))')
                   for n = 1:nt for j = 1:n)
    return errm, errg, errt, errl
end

@testset verbose = true "KadanoffBaym: pseudo-particle Dyson" begin

    β = 2.0

    @testset "atomic limit (sign = $sign, k = $k, size = $d)" for sign in (FERMI, BOSE),
                                                                  k in (1, 3, 5), d in (1, 2)
        nt, ntau, h = 12, 61, 0.01
        H0 = d == 1 ? fill(C64(0.3), 1, 1) : PD_H0
        C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
        G = ℱᵖˢ(C, sign)
        Σ = ℱᵖˢ(C, sign)
        pseudo_dyson!(G, PD_λ0, pd_ham(nt, h, H0), Σ, Integrator(k), β, h)
        errm, errg, errt, errl = pd_errors(G, H0, PD_λ0, β, h)
        tol = Dict(1 => 2e-2, 3 => 2e-5, 5 => 1e-7)[k]
        @test errm < tol
        @test errg < tol
        @test errt < tol
        @test errl < tol
        # G^{>}(t,t) = -i
        @test all(n -> G.ret[n,n] ≈ -im * Matrix{C64}(I, d, d), 1:nt)
        @test all(isfinite, G.mat.data) && all(isfinite, G.lmix.data)
    end

    @testset "grid convergence of the Matsubara component (k = $k)" for k in (2, 4)
        err = Float64[]
        for ntau in (21, 41, 81)
            C = Cn(6, ntau, 2, 2, 0.05, β)
            G = ℱᵖˢ(C); Σ = ℱᵖˢ(C)
            pseudo_dyson_mat!(G, PD_λ0, PD_H0, Σ, Integrator(k), β)
            δτ = β / (ntau - 1)
            push!(err, maximum(norm(G.mat[m] + exp((PD_λ0 * I - PD_H0) * (m - 1) * δτ))
                               for m = 1:ntau))
        end
        # The error decreases with an order not smaller than k
        @test err[2] < err[1] / 2^k && err[3] < err[2] / 2^k
    end

    @testset "start-up, time stepping, and the overall driver" begin
        nt, ntau, h, k = 14, 41, 0.02, 4
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        I4 = Integrator(k)
        H = pd_ham(nt, h, PD_H0)
        # A nonzero self-energy, built by a pseudo-particle Dyson solve
        Σ = ℱᵖˢ(C, FERMI)
        pseudo_dyson!(Σ, 0.2, pd_ham(nt, h, C64[0.1 0.05; 0.05 0.2]), ℱᵖˢ(C, FERMI),
                     I4, β, h)
        for t = 0:nt
            smul!(Σ, t, 0.3)
        end
        G = ℱᵖˢ(C, FERMI)
        pseudo_dyson!(G, PD_λ0, H, Σ, I4, β, h)
        X = ℱᵖˢ(C, FERMI)
        pseudo_dyson_mat!(X, PD_λ0, H, Σ, I4, β)
        pseudo_dyson_start!(X, PD_λ0, H, Σ, I4, β, h)
        for tstp = k+2:nt
            pseudo_dyson_timestep!(tstp, X, PD_λ0, H, Σ, I4, β, h)
        end
        @test all(t -> distance(G, X, t) == 0.0, 0:nt)
        # The self-energy changes the result
        Y = ℱᵖˢ(C, FERMI)
        pseudo_dyson!(Y, PD_λ0, H, ℱᵖˢ(C, FERMI), I4, β, h)
        @test distance(G, Y, 0) > 1e-3
        @test distance(G, Y, 2) > 1e-4
        # The start-up time steps are not changed by the time stepping
        @test_throws ArgumentError pseudo_dyson_timestep!(k + 1, X, PD_λ0, H, Σ, I4, β, h)
    end

    @testset "Hamiltonian adapter: Cf, matrices, and scalars" begin
        nt, ntau, h, k = 10, 31, 0.02, 3
        I3 = Integrator(k)
        # 2 × 2: Cf versus H0 + vector of matrices
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        Hcf = pd_ham(nt, h, PD_H0)
        Hv = [Hcf[n] for n = 1:nt]
        G1 = ℱᵖˢ(C); G2 = ℱᵖˢ(C); Σ = ℱᵖˢ(C)
        pseudo_dyson!(G1, PD_λ0, Hcf, Σ, I3, β, h)
        pseudo_dyson!(G2, PD_λ0, PD_H0, Hv, Σ, I3, β, h)
        @test all(t -> distance(G1, G2, t) < 1e-12, 0:nt)
        A = KadanoffBaym._assemble_pseudo_hamiltonian(G1, PD_H0, Hv)
        @test A[0] == PD_H0 && all(n -> A[n] == Hv[n], 1:nt)
        # 1 × 1: Cf, matrices, and scalars
        C1 = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        H1 = pd_ham(nt, h, fill(C64(0.3), 1, 1))
        Gs = [ℱᵖˢ(C1, BOSE) for _ = 1:3]
        Σ1 = ℱᵖˢ(C1, BOSE)
        pseudo_dyson!(Gs[1], PD_λ0, H1, Σ1, I3, β, h)
        pseudo_dyson!(Gs[2], PD_λ0, H1[0], [H1[n] for n = 1:nt], Σ1, I3, β, h)
        pseudo_dyson!(Gs[3], PD_λ0, H1[0][1], [H1[n][1] for n = 1:nt], Σ1, I3, β, h)
        @test all(t -> distance(Gs[1], Gs[2], t) < 1e-12, 0:nt)
        @test all(t -> distance(Gs[1], Gs[3], t) < 1e-12, 0:nt)
        # Scalars are not allowed for 2 × 2 Green's functions
        @test_throws DimensionMismatch pseudo_dyson!(G1, PD_λ0, 0.3, fill(0.3, nt), Σ, I3, β, h)
        @test_throws DimensionMismatch pseudo_dyson_mat!(G1, PD_λ0, 0.3, Σ, I3, β)
    end

    @testset "argument checks" begin
        C = Cn(10, 31, 2, 2, 0.2, β)
        G = ℱᵖˢ(C); Σ = ℱᵖˢ(C, BOSE)
        H = pd_ham(10, 0.02, PD_H0)
        @test_throws ArgumentError pseudo_dyson!(G, PD_λ0, H, Σ, Integrator(3), β, 0.02)
        @test_throws ArgumentError pseudo_dyson!(G, PD_λ0, H, ℱᵖˢ(C), Integrator(6), β, 0.02)
        Hn = pd_ham(10, 0.02, C64[0.3 0.2; 0.1 -0.4])
        @test_throws ArgumentError pseudo_dyson!(G, PD_λ0, Hn, ℱᵖˢ(C), Integrator(3), β, 0.02)
        # ℱ objects are rejected
        F = ℱ(C)
        @test_throws MethodError pseudo_dyson!(F, PD_λ0, H, F, Integrator(3), β, 0.02)
    end
end
