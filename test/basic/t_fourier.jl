#
# t_fourier.jl
#
# To test the corrected Fourier integrals: get_dftcorr_linear(),
# get_dftcorr_cubic(), complex_dftcor_cubic(), dft_cplx(), and
# adft_func().
#
# NESSi: fourier.cpp and fourier.hpp. The point-wise comparison with
# NESSi is in test/nessi/t_phase7_reference.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

# Linearly corrected DFT, built from the public correction coefficients
function fo_linear(ω, a, b, v)
    N = length(v) - 1
    δ = (b - a) / N
    W, α = get_dftcorr_linear(ω * δ)
    s = sum(cis(j * ω * δ) * v[j+1] for j = 0:N)
    return δ * cis(ω * a) * (W * s + α[1] * v[1] + cis(ω * (b - a)) * conj(α[1]) * v[N+1])
end

# Cubically corrected DFT without the endpoint correction (mutant)
function fo_mutant(ω, a, b, v)
    N = length(v) - 1
    δ = (b - a) / N
    W, _ = get_dftcorr_cubic(ω * δ)
    s = sum(cis(j * ω * δ) * v[j+1] for j = 0:N)
    return δ * cis(ω * a) * W * s
end

# Analytic integrals ∫ₐᵇ e^{iωt} f(t) dt
fo_exact_exp(ω, a, b) = (exp((im * ω - 1) * b) - exp((im * ω - 1) * a)) / (im * ω - 1)
fo_exact_cexp(ω, ν, a, b) = ω + ν == 0 ? complex(b - a) :
    (cis((ω + ν) * b) - cis((ω + ν) * a)) / (im * (ω + ν))

@testset verbose = true "KadanoffBaym: corrected Fourier integrals" begin

    @testset "correction coefficients" begin
        for t in (0.0, 1e-8, -1e-8, 0.049999, 0.050001, -0.049999, -0.050001, 0.7, -0.7, 3.0)
            W, α = get_dftcorr_cubic(t)
            @test isfinite(W) && all(isfinite, α) && length(α) == 4
            W₁, α₁ = get_dftcorr_linear(t)
            @test isfinite(W₁) && all(isfinite, α₁) && length(α₁) == 2
            @test α₁[2] == 0
        end
        # Continuity at |θ| = 0.05
        for f in (get_dftcorr_cubic, get_dftcorr_linear), s in (1, -1)
            W₋, α₋ = f(s * (0.05 - 1e-12))
            W₊, α₊ = f(s * (0.05 + 1e-12))
            @test abs(W₋ - W₊) < 1e-9
            @test maximum(abs.(collect(α₋) .- collect(α₊))) < 1e-9
        end
        # θ = 0: W = 1, the trapezoidal rule for the linear correction
        @test get_dftcorr_cubic(0.0)[1] == 1.0
        @test get_dftcorr_linear(0.0) == (1.0, (-0.5 + 0.0im, 0.0im))
        @test_throws ArgumentError get_dftcorr_cubic(NaN)
        @test_throws ArgumentError get_dftcorr_linear(Inf)
    end

    @testset "complex_dftcor_cubic" begin
        e = ComplexF64.(1:8)
        c, w = complex_dftcor_cubic(0.0, 0.1, 0.0, 0.7, e)
        @test isfinite(c) && isfinite(w)
        # At ω = 0: α = (-2/3, 7/24, -1/6, 1/24), both ends real
        α = (-2 / 3, 7 / 24, -1 / 6, 1 / 24)
        @test c ≈ sum(α[j+1] * e[j+1] + α[j+1] * e[8-j] for j = 0:3)
        @test_throws ArgumentError complex_dftcor_cubic(0.0, 0.1, 0.0, 0.7, e[1:7])
        @test_throws ArgumentError complex_dftcor_cubic(0.0, 0.0, 0.0, 0.7, e)
        @test_throws ArgumentError complex_dftcor_cubic(0.0, 0.1, 0.7, 0.7, e)
    end

    @testset "dft_cplx: analytic integrals" begin
        a, b = 0.0, 4.0
        # Exponential
        for ω in (0.0, 0.7, -2.3, 10.0)
            v = [exp(-t) for t in range(a, b, length = 65)]
            r, e = dft_cplx(ω, a, b, v)
            ex = fo_exact_exp(ω, a, b)
            @test abs(r - ex) < 1e-6
            # The error estimate is conservative
            @test abs(r - ex) ≤ abs(e)
        end
        # Constant and complex exponential are exact up to rounding
        v = ones(65)
        @test abs(dft_cplx(0.0, a, b, v)[1] - 4.0) < 1e-13
        @test abs(dft_cplx(1.3, a, b, v)[1] - fo_exact_cexp(1.3, 0.0, a, b)) < 1e-6
        # Two frequencies
        v = [cis(2.0 * t) + 0.5 * cis(-3.0 * t) for t in range(a, b, length = 129)]
        ex = fo_exact_cexp(0.4, 2.0, a, b) + 0.5 * fo_exact_cexp(0.4, -3.0, a, b)
        @test abs(dft_cplx(0.4, a, b, v)[1] - ex) < 1e-5
        # Failure paths
        @test_throws ArgumentError dft_cplx(0.0, a, b, ones(17))
        @test_throws ArgumentError dft_cplx(0.0, a, b, ones(18))
        @test_throws ArgumentError dft_cplx(0.0, b, a, ones(19))
        @test_throws ArgumentError dft_cplx(NaN, a, b, ones(19))
        @test dft_cplx(0.0, a, b, ones(19))[1] ≈ 4.0
    end

    @testset "cubic is more accurate than linear" begin
        a, b, ω = 0.0, 1.0, 0.7
        ex = fo_exact_exp(ω, a, b)
        for N in (32, 64)
            v = [exp(-t) for t in range(a, b, length = N + 1)]
            @test abs(dft_cplx(ω, a, b, v)[1] - ex) < 1e-3 * abs(fo_linear(ω, a, b, v) - ex)
        end
    end

    @testset "comparison with the plain DFT (Wave 1 FFT)" begin
        # At the grid frequencies ω = 2πk/(Nδ) the plain sum is a DFT
        N, a, b = 32, 0.0, 2.0
        δ = (b - a) / N
        v = [exp(-t) * cos(3t) + im * t for t in range(a, b, length = N + 1)]
        y = fft(v[1:N], FFT_BACKWARD)
        for k in (0, 3, 17)
            ω = 2π * k / (N * δ)
            r, _ = dft_cplx(ω, a, b, v)
            endpoints = [v[1:4]; v[N-2:N+1]]
            c, W = complex_dftcor_cubic(ω, δ, a, b, endpoints)
            plain = (r / (δ * cis(ω * a)) - c) / W
            @test abs(plain - (y[k+1] + v[N+1])) < 1e-11 * N
        end
    end

    @testset "adaptive Fourier integral" begin
        ex = sqrt(π) * exp(-0.7^2 / 4)
        r, e = adft_func(x -> exp(-x^2), 0.7, -5, 5; n = 32, limit = 64)
        @test abs(r - ex) < 1e-10
        @test isfinite(e)
        # More panels are more accurate
        r₁, _ = adft_func(x -> exp(-x^2), 0.7, -5, 5; n = 18, limit = 3)
        r₂, _ = adft_func(x -> exp(-x^2), 0.7, -5, 5; n = 18, limit = 40)
        @test abs(r₂ - ex) < abs(r₁ - ex)
        # A single panel is dft_cplx on the whole interval
        t = range(-5, 5, length = 19)
        @test abs(adft_func(x -> exp(-x^2), 0.7, -5, 5; n = 18, limit = 1)[1] -
                  dft_cplx(0.7, -5, 5, exp.(-t .^ 2))[1]) < 1e-14
        # Failure paths
        f = x -> exp(-x^2)
        @test_throws ArgumentError adft_func(f, 0.7, 5, -5)
        @test_throws ArgumentError adft_func(f, 0.7, -5, 5; n = 14)
        @test_throws ArgumentError adft_func(f, 0.7, -5, 5; n = 19)
        @test_throws ArgumentError adft_func(f, 0.7, -5, 5; n = 20, limit = 0)
        # n = 16 passes the sampler but the corrected DFT needs n > 16
        @test_throws ArgumentError adft_func(f, 0.7, -5, 5; n = 16, limit = 4)
        # The default arguments are executable: Gaussian, ∫ e^{iωx - x²}
        # = √π e^{-ω²/4} (the tails beyond |x| = 5 are below 1e-11)
        r, _ = adft_func(f, 0.7, -5, 5)
        @test abs(r - sqrt(π) * exp(-0.7^2 / 4)) < 1e-10
        @test adft_func(f, 0.7, -5, 5) == adft_func(f, 0.7, -5, 5; n = 18, limit = 100)
    end

    if kb_release()
        @testset "release: convergence order gates" begin
            @test kb_test_profile() == "release"
            a, b = 0.0, 1.0
            Ns = [32, 64, 128, 256]
            for ω in (0.7, 5.0)
                ex = fo_exact_exp(ω, a, b)
                grid(N) = [exp(-t) for t in range(a, b, length = N + 1)]
                ec = [abs(dft_cplx(ω, a, b, grid(N))[1] - ex) for N in Ns]
                el = [abs(fo_linear(ω, a, b, grid(N)) - ex) for N in Ns]
                em = [abs(fo_mutant(ω, a, b, grid(N)) - ex) for N in Ns]
                test_order_gate(Ns, ec, 4, 1e-10)
                test_order_gate(Ns, el, 2, 1e-5)
                # The mutant without the endpoint correction is rejected
                @test !check_order_gate(Ns, em, 4, 1e-10)[1]
                # The error estimate decreases with the grid
                ee = [abs(dft_cplx(ω, a, b, grid(N))[2]) for N in Ns]
                @test all(diff(ee) .< 0)
            end
        end
    end
end
