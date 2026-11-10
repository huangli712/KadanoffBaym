#
# t_spectral.jl
#
# To test the Wigner slices and the windowed spectral functions
# (NESSi python3/SpectrumCNTRhdf5.py).
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

# Contour Green's function with ntime real time points
sp_contour(ntime, d, h) = Cn(ntime, 5, d, d, (ntime - 1) * h, 1.0)

# Damped single poles G^R_ii(t) = -i e^{-i ε_i t - η t} for t ≥ 0
function sp_pole_gf(ntime, h, ε::Vector{Float64}, η)
    d = length(ε)
    G = ℱ(sp_contour(ntime, d, h), FERMI)
    for t₁ = 1:ntime, t₂ = 1:t₁
        Δ = (t₁ - t₂) * h
        G.ret[t₁, t₂] = Matrix(Diagonal(C64[-im * exp(-(im * e + η) * Δ) for e in ε]))
        # Lesser at t₂ ≤ t₁ (stored triangle), G^<(t₂,t₁) = 0.3i e^{iεΔ}
        G.less[t₂, t₁] = Matrix(Diagonal(C64[0.3im * exp(im * e * Δ) for e in ε]))
    end
    return G
end

lorentz(ω, ε, η) = η / (π * ((ω - ε)^2 + η^2))

@testset verbose = true "KadanoffBaym: spectral functions" begin

    @testset "Wigner slices" begin
        ntime, h = 21, 0.1
        G = ℱ(sp_contour(ntime, 2, h), FERMI)
        for t₁ = 1:ntime, t₂ = 1:t₁
            G.ret[t₁, t₂] = fill(C64(t₁ - t₂), 2, 2)
            G.less[t₂, t₁] = fill(C64(im * (t₁ + t₂)), 2, 2)
        end
        for tav in (1, 5, 11, 15, 21)
            t_rel, ret, les = read_gf_ret_les_tavtrel(G, tav, h)
            nj = min(tav - 1, ntime - tav) + 1
            @test t_rel == [2.0 * j * h for j = 0:nj-1]
            @test size(ret) == (nj, 2, 2) && size(les) == (nj, 2, 2)
            @test all(j -> ret[j+1, 1, 2] == 2j, 0:nj-1)
            # G^<(t₁,t₂) = -G^<(t₂,t₁)^† for t₁ ≥ t₂
            @test all(j -> les[j+1, 2, 1] == -conj(im * 2tav), 1:nj-1)
            @test les[1, 1, 1] == im * 2tav
        end
        @test_throws ArgumentError read_gf_ret_les_tavtrel(G, 0, h)
        @test_throws ArgumentError read_gf_ret_les_tavtrel(G, ntime + 1, h)
        @test_throws ArgumentError read_gf_ret_les_tavtrel(G, 3, 0.0)
        Gr = ℱ(Cn(ntime, 5, 2, 3, (ntime - 1) * h, 1.0), FERMI)
        @test_throws ArgumentError read_gf_ret_les_tavtrel(Gr, 3, h)
    end

    @testset "damped poles, identity window" begin
        # η T ≈ 20, the truncation of the time window is negligible
        ε, η, h = [0.5, -1.0], 0.5, 0.02
        ntime = 2001
        G = sp_pole_gf(ntime, h, ε, η)
        tav = 1001
        t_rel, ret, les = read_gf_ret_les_tavtrel(G, tav, h)
        @test length(t_rel) == 1001
        Δω = 0.01
        ω = collect(-2.0:Δω:2.0)
        for method in (:cubic, :linear)
            A = evaluate_spectrum_windowed(ret, t_rel, ω, method, t -> ones(length(t)))
            for i = 1:2
                k = argmax(A[:, i])
                @test abs(ω[k] - ε[i]) ≤ Δω
                kc = argmin(abs.(ω .- ε[i]))
                @test abs(A[kc, i] - lorentz(ω[kc], ε[i], η)) / lorentz(ω[kc], ε[i], η) < 1e-2
                @test maximum(abs.(A[:, i] .- lorentz.(ω, ε[i], η))) * π * η < 1e-2
            end
        end
    end

    @testset "windows converge with the window length" begin
        ε, η, h = [0.3], 0.3, 0.02
        ntime = 4001
        G = sp_pole_gf(ntime, h, ε, η)
        t_rel, ret, _ = read_gf_ret_les_tavtrel(G, 2001, h)
        T = t_rel[end]
        ω = collect(-0.5:0.01:1.1)
        hann(L) = t -> [x ≤ L ? 0.5 * (1 + cos(π * x / L)) : 0.0 for x in t]
        gauss(σ) = t -> exp.(-(t ./ σ) .^ 2)
        for (name, win) in (("Hann", hann), ("Gaussian", gauss))
            errs = Float64[]
            for L in (T / 8, T / 4, T / 2)
                A = evaluate_spectrum_windowed(ret, t_rel, ω, :linear, win(L))
                k = argmax(A[:, 1])
                @test abs(ω[k] - ε[1]) ≤ 0.01
                kc = argmin(abs.(ω .- ε[1]))
                push!(errs, abs(A[kc, 1] - lorentz(ω[kc], ε[1], η)))
            end
            @test errs[3] < errs[2] < errs[1]
        end
        # Complex window: a phase e^{iΔt} shifts the peak by -Δ
        A = evaluate_spectrum_windowed(ret, t_rel, ω, :cubic, t -> cis.(0.2 .* t))
        @test abs(ω[argmax(A[:, 1])] - (ε[1] - 0.2)) ≤ 0.01
    end

    @testset "boundary slices and the cubic interval rule" begin
        ε, η, h = [0.4], 0.5, 0.05
        ntime = 41
        G = sp_pole_gf(ntime, h, ε, η)
        ω = [-1.0, 0.0, 0.4, 1.0]
        one(t) = ones(length(t))
        # The slice closest to the boundary with two points
        t_rel, ret, _ = read_gf_ret_les_tavtrel(G, 2, h)
        @test length(t_rel) == 2
        A = evaluate_spectrum_windowed(ret, t_rel, ω, :linear, one)
        @test all(isfinite, A)
        @test_throws ArgumentError evaluate_spectrum_windowed(ret, t_rel, ω, :cubic, one)
        # ω = 0 uses the trapezoidal rule
        A₀ = evaluate_spectrum_windowed(ret, t_rel, [0.0], :linear, one)
        v = ret[:, 1, 1]
        @test A₀[1, 1] ≈ -imag(0.5 * (v[1] + v[2]) * 2h) / π
        # A single point is not enough
        t₁, r₁, _ = read_gf_ret_les_tavtrel(G, 1, h)
        @test_throws ArgumentError evaluate_spectrum_windowed(r₁, t₁, ω, :linear, one)
        # N = 16 is rejected, N = 18 (19 points) is the smallest cubic grid
        t₁₇, r₁₇, _ = read_gf_ret_les_tavtrel(G, 17, h)
        @test length(t₁₇) == 17
        @test_throws ArgumentError evaluate_spectrum_windowed(r₁₇, t₁₇, ω, :cubic, one)
        t₁₉, r₁₉, _ = read_gf_ret_les_tavtrel(G, 19, h)
        @test length(t₁₉) == 19
        A = evaluate_spectrum_windowed(r₁₉, t₁₉, ω, :cubic, one)
        @test all(isfinite, A)
        # Odd number of intervals
        @test_throws ArgumentError evaluate_spectrum_windowed(r₁₉[1:18, :, :], t₁₉[1:18], ω, :cubic, one)
        # Cubic and linear agree on a smooth integrand
        Al = evaluate_spectrum_windowed(r₁₉, t₁₉, ω, :linear, one)
        @test maximum(abs.(A .- Al)) < 1e-2
    end

    @testset "invalid input" begin
        t = collect(0.0:0.1:2.0)
        g = zeros(C64, 21, 2, 2)
        ω = [0.0, 1.0]
        one(x) = ones(length(x))
        @test_throws ArgumentError evaluate_spectrum_windowed(g, t, ω, :spline, one)
        @test_throws ArgumentError evaluate_spectrum_windowed(zeros(C64, 21, 2, 3), t, ω, :linear, one)
        @test_throws DimensionMismatch evaluate_spectrum_windowed(g, t[1:20], ω, :linear, one)
        @test_throws DimensionMismatch evaluate_spectrum_windowed(g, t, ω, :linear, x -> ones(3))
        @test_throws DimensionMismatch evaluate_spectrum_windowed(g, t, ω, :linear, x -> 1.0)
        @test_throws ArgumentError evaluate_spectrum_windowed(g, reverse(t), ω, :linear, one)
        tb = copy(t)
        tb[5] += 0.03
        @test_throws ArgumentError evaluate_spectrum_windowed(g, tb, ω, :linear, one)
        @test_throws ArgumentError evaluate_spectrum_windowed(g, t, [NaN], :linear, one)
        A = evaluate_spectrum_windowed(g, t, ω, :cubic, one)
        @test size(A) == (2, 2) && all(iszero, A)
    end
end
