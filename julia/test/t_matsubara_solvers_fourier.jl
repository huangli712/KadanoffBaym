#
# t_matsubara_solvers_fourier.jl
#
# To test the `:fourier` method of dyson_mat!() and vie2_mat!().
#
# NESSi: cntr_dyson_impl.hpp, dyson_mat_fourier_dispatch (pcf = 10), and
# cntr_vie2_impl.hpp, vie2_mat_fourier_dispatch (pcf = 20).
#
# The frequency sums are truncated at (2pcf + 1) N/2 modes and the
# remaining tail decays as 1/(iω)², so that the Fourier method converges
# only as N^{-3/2} towards the fixpoint (discrete integral equation)
# solution. The comparison with FIXPOINT therefore checks the convergence
# and a relative difference at the 1e-4 level, the point-wise comparison
# with NESSi (same algorithm, 1e-10) is in test/t_phase7_reference.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const MSF_H0 = C64[0.3 0.2-0.1im; 0.2+0.1im -0.4]
const MSF_HS = C64[0.5 0.1im; -0.1im -0.2]

# Hamiltonian, self-energy, and the VIE2 kernels on a 2 × 2 grid
function msf_setup(sign, ntau; β = 2.0, d = 2)
    C = Cn(2, ntau, d, d, 0.1, β)
    H = Cf(2, d, d)
    Hs = Cf(2, d, d)
    for i = 0:2
        H[i] = d == 2 ? MSF_H0 : fill(C64(0.3), 1, 1)
        Hs[i] = d == 2 ? MSF_HS : fill(C64(0.5), 1, 1)
    end
    μ = sign == FERMI ? 0.1 : -1.5
    Σ = ℱ(C, sign)
    dyson_mat!(Σ, ℱ(C, sign), sign == FERMI ? 0.0 : -1.0, Hs, β)
    Σ.mat.data .*= 0.3
    G0 = ℱ(C, sign)
    dyson_mat!(G0, ℱ(C, sign), μ, H, β)
    I5 = Integrator(5)
    F = ℱ(C, sign)
    Fcc = ℱ(C, sign)
    conv_mat(F.mat, G0.mat, Σ.mat, I5, β, sign)
    conv_mat(Fcc.mat, Σ.mat, G0.mat, I5, β, sign)
    F.mat.data .*= -1
    Fcc.mat.data .*= -1
    return C, H, μ, Σ, G0, F, Fcc, I5
end

reldiff(X, Y) = norm(X.mat.data - Y.mat.data) / norm(Y.mat.data)

@testset verbose = true "KadanoffBaym: Matsubara Fourier solvers" begin

    β = 2.0

    @testset "dyson_mat! :fourier, free particle (sign = $sign)" for sign in (FERMI, BOSE)
        # Analytic G(τ) = -e^{-ετ}/(1 - ξ e^{-βε}) for a single level
        ε = sign == FERMI ? 0.4 : 0.6
        errs = Float64[]
        for ntau in (33, 65, 129)
            C = Cn(2, ntau, 1, 1, 0.1, β)
            H = Cf(2, 1, 1)
            H[0] = fill(C64(ε), 1, 1)
            G = ℱ(C, sign)
            dyson_mat!(G, ℱ(C, sign), 0.0, H, β; method = :fourier)
            @test all(isfinite, G.mat.data)
            ex = [-exp(-ε * τ) / (1 - sign * exp(-β * ε)) for τ in range(0, β, length = ntau)]
            push!(errs, maximum(abs.(G.mat.data[1,1,:] .- ex)))
        end
        @test errs[1] < 2e-3
        @test errs[2] < errs[1] && errs[3] < errs[2]
    end

    @testset "dyson_mat! :fourier vs :fixpoint (sign = $sign, size = $d)" for sign in (FERMI, BOSE), d in (1, 2)
        diffs = Float64[]
        for ntau in (33, 65, 129)
            C, H, μ, Σ, G0, F, Fcc, I5 = msf_setup(sign, ntau; d = d)
            Gf = ℱ(C, sign)
            dyson_mat!(Gf, Σ, μ, H, β; method = :fixpoint)
            Go = ℱ(C, sign)
            dyson_mat!(Go, Σ, μ, H, β; method = :fourier, force_hermitian = false)
            push!(diffs, reldiff(Go, Gf))
            # Hermiticity of the matrix G(τ) without the symmetrization, the
            # bosonic frequency window is not symmetric (truncation level)
            @test maximum(r -> norm(Go.mat[r] - Go.mat[r]'), 1:ntau) <
                  (sign == FERMI ? 1e-10 : 1e-8)
            # KMS: the jump G(0) - ξ G(β) is -1 (the free tail)
            @test norm(Go.mat[1] - sign * Go.mat[ntau] + I) < 1e-10
            # The public method uses pcf = 10 and the cubic correction
            Gp = ℱ(C, sign)
            KadanoffBaym._dyson_mat_fourier!(Gp, Σ, μ, H, β; pcf = 10, order = 3)
            @test Gp.mat.data == Go.mat.data
        end
        @test diffs[1] < 5e-4
        @test diffs[2] < diffs[1] && diffs[3] < diffs[2]
    end

    @testset "vie2_mat! :fourier vs :fixpoint (sign = $sign)" for sign in (FERMI, BOSE)
        diffs = Float64[]
        for ntau in (33, 65, 129)
            C, H, μ, Σ, G0, F, Fcc, I5 = msf_setup(sign, ntau)
            Xf = ℱ(C, sign)
            vie2_mat!(Xf, F, Fcc, G0, β, I5; method = :fixpoint)
            Xo = ℱ(C, sign)
            vie2_mat!(Xo, F, Fcc, G0, β, I5; method = :fourier)
            push!(diffs, reldiff(Xo, Xf))
            # F = 0 gives G = Q up to the truncated tail
            Z = ℱ(C, sign)
            X₀ = ℱ(C, sign)
            vie2_mat!(X₀, Z, Z, G0, β, I5; method = :fourier)
            @test reldiff(X₀, G0) < 1e-3
            # Fcc is not used by the frequency-domain solver
            Xc = ℱ(C, sign)
            vie2_mat!(Xc, F, Z, G0, β, I5; method = :fourier)
            @test Xc.mat.data == Xo.mat.data
            # pcf = 20, not the Dyson window pcf = 10
            X₂₀ = ℱ(C, sign)
            X₁₀ = ℱ(C, sign)
            KadanoffBaym._vie2_mat_fourier!(X₂₀, F, G0, β; pcf = 20, order = 3)
            KadanoffBaym._vie2_mat_fourier!(X₁₀, F, G0, β; pcf = 10, order = 3)
            @test X₂₀.mat.data == Xo.mat.data
            @test reldiff(X₁₀, X₂₀) > 1e-7
        end
        @test diffs[1] < 1e-4
        @test diffs[2] < diffs[1] && diffs[3] < diffs[2]
    end

    @testset "failure paths" begin
        # Even number of points = odd number of intervals
        C = Cn(2, 34, 1, 1, 0.1, β)
        H = Cf(2, 1, 1)
        H[0] = fill(C64(0.3), 1, 1)
        G = ℱ(C, FERMI)
        @test_throws ArgumentError dyson_mat!(G, ℱ(C, FERMI), 0.0, H, β; method = :fourier)
        @test_throws ArgumentError vie2_mat!(G, G, G, G, β, Integrator(5); method = :fourier)
        # Singular bosonic zero mode, H - μ + Σ(0) = 0
        C = Cn(2, 33, 1, 1, 0.1, β)
        H = Cf(2, 1, 1)
        H[0] = zeros(C64, 1, 1)
        @test_throws Exception dyson_mat!(ℱ(C, BOSE), ℱ(C, BOSE), 0.0, H, β; method = :fourier)
        # Unknown method
        @test_throws ArgumentError dyson_mat!(ℱ(C, BOSE), ℱ(C, BOSE), 0.0, H, β; method = :fft)
    end
end
