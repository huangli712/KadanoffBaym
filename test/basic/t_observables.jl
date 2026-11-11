#
# t_observables.jl
#
# To test density_matrix(), density(), kinetic_energy(), and
# correlation_energy().
#
# The expected correlation energies are evaluated by explicit Gregory
# sums written in this file, not by the convolution wrappers.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

#
# Gregory weight of the point m (0-based) on a grid with N intervals,
# N ≥ 2k + 1
#
function ob_greg(I::Integrator, m::Int, N::Int)
    k = I.k
    m ≤ k && return I.GIW.ω[m + 1]
    m ≥ N - k && return I.GIW.ω[N - m + 1]
    return 1.0
end

#
# 0.5 Re Tr[-i (Σ * G)^<(t,t)] for 1x1 functions, by explicit sums.
# Here tstp is the KB index (n = tstp - 1).
#
function ob_ecorr_rt(tstp, G, Σ, I, β, h)
    n = tstp - 1
    N = getntau(G) - 1
    δτ = β / N
    ξ = getsign(G)
    s = 0.0im
    # Σ^R(t,s) G^<(s,t) + Σ^<(t,s) G^A(s,t), s = t_m
    for m = 0:n
        w = h * I.GIW[n, m]
        ΣR = Σ.ret[n + 1, m + 1][1, 1]
        Gl = G.less[m + 1, n + 1][1, 1]
        Σl = -conj(Σ.less[m + 1, n + 1][1, 1])
        GA = conj(G.ret[n + 1, m + 1][1, 1])
        s += w * (ΣR * Gl + Σl * GA)
    end
    # -i Σ^⌉(t,τ) G^⌈(τ,t), G^⌈(τ,t) = -ξ G^⌉(t,β-τ)^*
    for m = 0:N
        Gv = -ξ * conj(G.lmix[n + 1, N - m + 1][1, 1])
        s += -im * δτ * ob_greg(I, m, N) * Σ.lmix[n + 1, m + 1][1, 1] * Gv
    end
    return 0.5 * real(-im * s)
end

#
# 0.5 Re Tr[-(Σ * G)^M(β)] = 0.5 Re Tr[-∫ Σ^M(β-τ) G^M(τ) dτ]
#
function ob_ecorr_mat(G, Σ, I, β)
    N = getntau(G) - 1
    δτ = β / N
    s = 0.0im
    for m = 0:N
        s += δτ * ob_greg(I, m, N) * Σ.mat[N - m + 1][1, 1] * G.mat[m + 1][1, 1]
    end
    return 0.5 * real(-s)
end

@testset "Observables ($(kb_test_profile()))" begin
    @testset "density matrix" begin
        # ntime ≠ ntau, so that the old index bug of density() is caught
        nt, ntau, β, h = 5, 9, 2.0, 0.1
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        for ξ in (FERMI, BOSE)
            G = ℱ(C, ξ)
            M = ComplexF64[0.3 0.1im; -0.1im 0.7]
            L = ComplexF64[0.2im 0.5; -0.5 0.4im]
            for m = 1:ntau
                G.mat[m] = (m + 1.0) * M
            end
            G.mat[ntau] = M
            G.less[3, 3] = L
            @test density_matrix(G, 0) == -M
            @test density_matrix(G, 3) == im * ξ * L
            @test density(G, 0) == -M
            @test density(G, 3) == im * ξ * L
            #
            g = 𝒻(3, ntau, 2, 2, ξ)
            g.less[3] = L
            @test density_matrix(g, 3) == im * ξ * L
            @test density(g, 3) == im * ξ * L
            g0 = 𝒻(0, ntau, 2, 2, ξ)
            g0.mat[ntau] = M
            @test density_matrix(g0, 0) == -M
            @test_throws BoundsError density_matrix(G, nt + 1)
            @test_throws ArgumentError density_matrix(g, 2)
        end
    end

    @testset "kinetic energy" begin
        H = ComplexF64[1.0 0.5-0.2im; 0.5+0.2im -0.3]
        ρ = ComplexF64[0.6 0.1+0.3im; 0.1-0.3im 0.25]
        Ek = 1.0 * 0.6 + 2 * real((0.5 - 0.2im) * (0.1 - 0.3im)) - 0.3 * 0.25
        @test abs(kinetic_energy(H, ρ) - Ek) < 1e-14
        #
        nt, ntau, β, h = 4, 11, 2.0, 0.1
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        G = ℱ(C, FERMI)
        from_ham!(G, 0.1, H, β, h)
        Hc = Cf(nt, 2, 2)
        for i = 0:nt
            Hc[i] = (1.0 + 0.1 * i) * H
        end
        for tstp = 0:nt
            @test abs(kinetic_energy(Hc, G, tstp) -
                      real(tr(Hc[tstp] * density_matrix(G, tstp)))) < 1e-14
        end
        @test_throws DimensionMismatch kinetic_energy(H, zeros(ComplexF64, 3, 3))
    end

    @testset "correlation energy" begin
        nt, ntau, β, h = 21, 41, 2.0, 0.05
        C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        for ξ in (FERMI, BOSE)
            G = ℱ(C, ξ)
            Σ = ℱ(C, ξ)
            μ = ξ == FERMI ? 0.1 : -1.5
            from_ham!(G, μ, fill(ComplexF64(0.4), 1, 1), β, h)
            # Σ = 0
            for tstp = 0:nt
                @test correlation_energy(tstp, G, Σ, Integrator(5), β, h) == 0.0
            end
            # Σ = V² g, with g a free Green's function
            from_ham!(Σ, μ, fill(ComplexF64(1.1), 1, 1), β, h)
            for t = 0:nt
                smul!(Σ, t, 0.25)
            end
            for k = 1:5
                I = Integrator(k)
                @test abs(correlation_energy(0, G, Σ, I, β, h) -
                          ob_ecorr_mat(G, Σ, I, β)) < 1e-11
                for tstp = 2k+3:nt
                    @test abs(correlation_energy(tstp, G, Σ, I, β, h) -
                              ob_ecorr_rt(tstp, G, Σ, I, β, h)) < 1e-11
                end
                @test correlation_energy(nt, G, Σ, β, h, k) ==
                      correlation_energy(nt, G, Σ, I, β, h)
            end
        end
        G = ℱ(C, FERMI)
        @test_throws ArgumentError correlation_energy(1, G, G, 2.0, 0.1, 6)
        @test_throws BoundsError correlation_energy(nt + 1, G, G, Integrator(2), 2.0, 0.1)
    end
end
