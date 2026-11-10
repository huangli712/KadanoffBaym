#
# t_matsubara_fourier.jl
#
# To test the Matsubara Fourier transform: get_tau(), get_omega(),
# set_first_order_tail!(), matsubara_dft!(), and matsubara_ft!().
#
# Models: the fermionic single pole G(τ) = -e^{-ετ}/(1 + e^{-βε}) and the
# bosonic single pole G(τ) = -e^{-ετ}/(1 - e^{-βε}) (ε > 0, no singular
# zero mode), both with G(iω) = 1/(iω - ε), and a 2 × 2 non-diagonal
# matrix built from them.
#
# NESSi: cntr_matsubara_impl.hpp.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

# Single pole at ε, N imaginary time intervals
function mf_pole(ε, N, β, sign)
    G = Gᵐᵃᵗ(N + 1, 1)
    z = sign == FERMI ? 1 + exp(-β * ε) : 1 - exp(-β * ε)
    for r = 0:N
        G[r+1] = fill(C64(-exp(-ε * r * β / N) / z), 1, 1)
    end
    return G
end

# 2 × 2 matrix U diag(g(ε₁), g(ε₂)) U'
const MF_U = [1 1im; 1im 1] / sqrt(2.0)
function mf_matrix(εs, N, β, sign)
    G = Gᵐᵃᵗ(N + 1, 2)
    g = [mf_pole(ε, N, β, sign) for ε in εs]
    for r = 1:N+1
        G[r] = MF_U * Diagonal([g[1][r][1], g[2][r][1]]) * MF_U'
    end
    return G
end

# Maximal relative error at selected frequencies
function mf_err(G, ε, β, sign, order; ms = (-7, -1, 0, 3, 11))
    N = getntau(G) - 1
    d = zeros(C64, N + 1, 1, 1)
    matsubara_dft!(d, G, sign)
    X = zeros(C64, 1, 1)
    return maximum(ms) do m
        matsubara_ft!(X, m, G, d, sign, β; order = order)
        ex = 1 / (im * get_omega(m, β, sign) - ε)
        abs(X[1] - ex) / abs(ex)
    end
end

@testset verbose = true "KadanoffBaym: Matsubara Fourier transform" begin

    β, ε = 2.0, 0.7

    @testset "grids" begin
        @test get_tau(8, 2.0, 17) == 1.0
        @test get_tau(0, 2.0, 17) == 0.0
        @test get_tau(16, 2.0, 17) == 2.0
        @test get_omega(0, 2.0, FERMI) == π / 2
        @test get_omega(0, 2.0, BOSE) == 0
        @test get_omega(-1, 2.0, FERMI) == -π / 2
        @test get_omega(-3, 2.0, BOSE) == -3π
        @test_throws ArgumentError get_tau(1, -1.0, 17)
        @test_throws ArgumentError get_tau(1, 1.0, 1)
        @test_throws ArgumentError get_omega(1, 0.0, FERMI)
        @test_throws ArgumentError get_omega(1, 1.0, 0)
    end

    @testset "first order tail (sign = $sign)" for sign in (FERMI, BOSE)
        N = 16
        d = zeros(C64, N + 1, 1, 1)
        c = reshape(C64[2.0], 1, 1)
        set_first_order_tail!(d, c, 4.0, sign)
        @test c == reshape(C64[2.0], 1, 1)
        if sign == FERMI
            @test all(d[:,1,1] .== -1)
        else
            @test d[1,1,1] == -1 && d[N+1,1,1] == 1
            @test d[N÷2+1,1,1] ≈ 0
        end
        # The tail removes the jump: G - c₀ is (anti)periodic
        G = mf_pole(ε, N, β, sign)
        a₀ = sign * G[N+1] - G[1]
        t = zeros(C64, N + 1, 1, 1)
        set_first_order_tail!(t, a₀, β, sign)
        @test abs((G[1][1] - t[1,1,1]) - sign * (G[N+1][1] - t[N+1,1,1])) ≤ 1e-12
        # The transform of the tail is a₀/(iω), exact for the cubic order
        T = Gᵐᵃᵗ(N + 1, 1)
        for r = 1:N+1
            T[r] = t[r,:,:]
        end
        raw = zeros(C64, N + 1, 1, 1)
        matsubara_dft!(raw, T, sign)
        X = zeros(C64, 1, 1)
        for m in (-3, 1, 5)
            matsubara_ft!(X, m, T, raw, sign, β; order = 3)
            @test abs(X[1] - a₀[1] / (im * get_omega(m, β, sign))) ≤ 1e-12
        end
        # Matrix coefficients and failure paths
        dm = zeros(C64, N + 1, 2, 2)
        cm = C64[1 2im; 3 4]
        set_first_order_tail!(dm, cm, β, sign)
        @test dm[1,:,:] ≈ -0.5 * cm
        @test_throws DimensionMismatch set_first_order_tail!(dm, c, β, sign)
        @test_throws ArgumentError set_first_order_tail!(dm, cm, -1.0, sign)
    end

    @testset "raw DFT" begin
        G = Gᵐᵃᵗ(17, 1)
        G[1] = 1.0 + 0im
        d = zeros(C64, 17, 1, 1)
        matsubara_dft!(d, G, BOSE)
        @test all(abs.(d[:,1,1] .- 1) .< 1e-14)
        # Direct sum for a 2 × 2 matrix function, m = 0:N
        N = 8
        G = mf_matrix((0.3, 1.1), N, β, FERMI)
        d = zeros(C64, N + 1, 2, 2)
        matsubara_dft!(d, G, FERMI)
        for m = 0:N
            S = sum(cispi((2m + 1) * r / N) * G[r+1] for r = 0:N)
            @test d[m+1,:,:] ≈ S atol = 1e-13
        end
        @test_throws DimensionMismatch matsubara_dft!(zeros(C64, N, 2, 2), G, FERMI)
        @test_throws ArgumentError matsubara_dft!(d, G, 2)
    end

    @testset "corrected transform (sign = $sign)" for sign in (FERMI, BOSE)
        # Cubic order on a moderate grid, negative and high modes
        @test mf_err(mf_pole(ε, 256, β, sign), ε, β, sign, 3) ≤ 1e-10
        @test mf_err(mf_pole(ε, 64, β, sign), ε, β, sign, 1) ≤ 1e-4
        @test mf_err(mf_pole(ε, 64, β, sign), ε, β, sign, 0) ≤ 1.0
        # Periodic mapping of the raw transform: m and m + N use the same
        # raw value, the difference comes from the correction only
        N = 32
        G = mf_pole(ε, N, β, sign)
        d = zeros(C64, N + 1, 1, 1)
        matsubara_dft!(d, G, sign)
        X₀ = zeros(C64, 1, 1)
        X₁ = zeros(C64, 1, 1)
        matsubara_ft!(X₀, 2, G, d, sign, β; order = 0)
        matsubara_ft!(X₁, 2 + N, G, d, sign, β; order = 0)
        @test X₀ == X₁
        # 2 × 2 non-diagonal matrix
        N = 256
        G = mf_matrix((0.3, 1.1), N, β, sign)
        d = zeros(C64, N + 1, 2, 2)
        matsubara_dft!(d, G, sign)
        X = zeros(C64, 2, 2)
        for m in (-5, 0, 4)
            matsubara_ft!(X, m, G, d, sign, β; order = 3)
            iω = im * get_omega(m, β, sign)
            ex = MF_U * Diagonal([1 / (iω - 0.3), 1 / (iω - 1.1)]) * MF_U'
            @test norm(X - ex) / norm(ex) ≤ 1e-10
        end
        # Failure paths
        @test_throws ArgumentError matsubara_ft!(X, 0, G, d, sign, β; order = 2)
        @test_throws ArgumentError matsubara_ft!(X, 0, G, d, sign, -β)
        @test_throws DimensionMismatch matsubara_ft!(zeros(C64, 1, 1), 0, G, d, sign, β)
        @test_throws DimensionMismatch matsubara_ft!(X, 0, G, d[1:N,:,:], sign, β)
    end

    if kb_release()
        @testset "release: order gates (sign = $sign)" for sign in (FERMI, BOSE)
            @test kb_test_profile() == "release"
            Ns = [32, 64, 128, 256]
            errs(order) = [mf_err(mf_pole(ε, N, β, sign), ε, β, sign, order) for N in Ns]
            test_order_gate(Ns, errs(3), 4, 1e-10)
            test_order_gate(Ns, errs(1), 2, 1e-5)
            test_order_gate(Ns, errs(0), 1, 0.5)
            # Mutants: the corrected transforms without the endpoint
            # corrections, and the raw transform with the wrong statistics
            function mutant(N, order)
                G = mf_pole(ε, N, β, sign)
                d = zeros(C64, N + 1, 1, 1)
                matsubara_dft!(d, G, sign)
                δτ = β / N
                return maximum((-7, -1, 0, 3, 11)) do m
                    one = sign == FERMI ? 1 : 0
                    θ = (2m + one) * π / N
                    W = order == 3 ? get_dftcorr_cubic(θ)[1] : get_dftcorr_linear(θ)[1]
                    X = δτ * W * d[mod(m, N)+1,1,1]
                    ex = 1 / (im * get_omega(m, β, sign) - ε)
                    abs(X - ex) / abs(ex)
                end
            end
            @test !check_order_gate(Ns, [mutant(N, 3) for N in Ns], 4, 1e-10)[1]
            @test !check_order_gate(Ns, [mutant(N, 1) for N in Ns], 2, 1e-5)[1]
            wrong = -sign
            e₀ = map(Ns) do N
                G = mf_pole(ε, N, β, sign)
                d = zeros(C64, N + 1, 1, 1)
                matsubara_dft!(d, G, wrong)
                X = zeros(C64, 1, 1)
                maximum((-1, 3)) do m
                    matsubara_ft!(X, m, G, d, wrong, β; order = 0)
                    ex = 1 / (im * get_omega(m, β, sign) - ε)
                    abs(X[1] - ex) / abs(ex)
                end
            end
            @test !check_order_gate(Ns, e₀, 1, 0.5)[1]
        end
    end
end
