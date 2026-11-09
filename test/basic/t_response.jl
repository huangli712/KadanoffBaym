#
# t_response.jl
#
# To test the response convolution c(t) = ∫ W(t,t') f(t'). There is no
# Catch test for it in NESSi 2.0.0, so it is verified with analytic
# integrals.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using LinearAlgebra

#
# ∫₀ˣ ∑ₚ c[p+1] sᵖ ds
#
rs_polyint(c, x) = sum(c[p+1] * x^(p + 1) / (p + 1) for p = 0:length(c)-1)

#
# Product of two polynomials given by their coefficients.
#
function rs_polymul(a, b)
    c = zeros(promote_type(eltype(a), eltype(b)), length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b)
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

#
# Hermitian 2x2 matrix
#
function rs_herm()
    X = rand(ComplexF64, 2, 2)
    return (X + X') / 2
end

#
# Build a 2x2 response function with
#   Wᴿ(t,s) = i [M₀ + M₁ (t + s)],   Wᴹ(τ) = ∑ₚ Mₚ τᵖ,   W^⌉(t,τ) = ∑ₚ Lₚ τᵖ.
# The retarded component satisfies Wᴿ(t,s) = -[Wᴿ(s,t)]†.
#
function rs_make(ntime, ntau, h, beta, k)
    C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
    W = ℱ(C)
    M₀ = rs_herm()
    M₁ = rs_herm()
    Mτ = [rand(ComplexF64, 2, 2) for _ = 0:k]
    Lτ = [rand(ComplexF64, 2, 2) for _ = 0:k]
    hv = beta / (ntau - 1)
    for i = 1:ntime, j = 1:i
        W.ret[i,j] = im .* (M₀ .+ M₁ .* ((i - 1) * h + (j - 1) * h))
    end
    for m = 1:ntau
        τ = (m - 1) * hv
        W.mat[m] = sum(Mτ[p+1] .* τ^p for p = 0:k)
        for i = 1:ntime
            W.lmix[i,m] = sum(Lτ[p+1] .* τ^p for p = 0:k)
        end
    end
    return C, W, M₀, M₁, Mτ, Lτ
end

#
# Exact response for the matrix elements W[a1,a2] and f[b1,b2], where
# f(t) = ∑ₚ Fₚ tᵖ (p ≤ k - 1) and f(0⁻) = F₀⁻.
#
function rs_exact(tstp, h, beta, k, M₀, M₁, Mτ, Lτ, F, F0, a1, a2, b1, b2)
    fc = [F[p+1][b1,b2] for p = 0:k-1]
    f₀ = F0[b1,b2]
    if tstp == 0
        return f₀ * rs_polyint([Mτ[p+1][a1,a2] for p = 0:k], beta)
    end
    t = (tstp - 1) * h
    # Wᴿ(t,s) = i [M₀ + M₁ t] + i M₁ s
    wc = [im * (M₀[a1,a2] + M₁[a1,a2] * t), im * M₁[a1,a2]]
    c = rs_polyint(rs_polymul(wc, fc), t)
    c += -im * f₀ * rs_polyint([Lτ[p+1][a1,a2] for p = 0:k], beta)
    return c
end

#
# Analytic response c(T) = ∫₀ᵀ -i e^{-iε(T-s)} cos(s) ds of a free 1x1
# Green's function, used in the convergence test.
#
function rs_free_error(k, ord, N)
    ε = 0.7
    T = 1.0
    exact = -im * exp(-im * ε * T) *
            ((exp(im * (ε + 1) * T) - 1) / (2im * (ε + 1)) +
             (exp(im * (ε - 1) * T) - 1) / (2im * (ε - 1)))
    h = T / N
    C = Cn(N + 1, 2 * k + 3, 1, 1, T, 1.0)
    W = ℱ(C)
    init_green!(W, fill(C64(ε), 1, 1), 0.0, 1.0, h)
    f = Cf(N + 1, 1)
    for i = 1:N+1
        f[i] = fill(C64(cos((i - 1) * h)), 1, 1)
    end
    f[0] = zeros(C64, 1, 1)
    c = response_convolution(N + 1, W, 1, 1, f, 1, 1, Integrator(ord), 1.0, h)
    return abs(c - exact)
end

@testset verbose = true "KadanoffBaym: response.jl" begin

    @testset "polynomial exactness" begin
        ntime = 14
        h = 0.1
        beta = 1.5
        for k = 1:5
            ntau = 2 * k + 5
            C, W, M₀, M₁, Mτ, Lτ = rs_make(ntime, ntau, h, beta, k)
            F = [rand(ComplexF64, 2, 2) for _ = 0:k-1]
            F0 = rand(ComplexF64, 2, 2)
            f = Cf(ntime, 2)
            for i = 1:ntime
                f[i] = sum(F[p+1] .* ((i - 1) * h)^p for p = 0:k-1)
            end
            f[0] = F0
            I = Integrator(k)
            for a1 = 1:2, a2 = 1:2, b1 = 1:2, b2 = 1:2
                for tstp = 0:ntime
                    c = response_convolution(tstp, W, a1, a2, f, b1, b2, I, beta, h)
                    e = rs_exact(tstp, h, beta, k, M₀, M₁, Mτ, Lτ, F, F0, a1, a2, b1, b2)
                    @test isapprox(c, e; atol = 1e-12, rtol = 1e-12)
                end
            end
        end
    end

    @testset "ℱ and 𝒻 versions agree" begin
        ntime = 12
        h = 0.1
        beta = 1.5
        k = 3
        ntau = 11
        C, W, _, _, _, _ = rs_make(ntime, ntau, h, beta, k)
        f = Cf(ntime, 2)
        for i = 0:ntime
            f[i] = rand(ComplexF64, 2, 2)
        end
        I = Integrator(k)
        for tstp in [0; k+2:ntime]
            c₁ = response_convolution(tstp, W, 1, 2, f, 2, 1, I, beta, h)
            c₂ = response_convolution(tstp, W[tstp], 1, 2, f, 2, 1, I, beta, h)
            @test c₁ == c₂
            @test response_convolution(tstp, W, 1, 2, f, 2, 1, k, beta, h) == c₁
            @test response_convolution(tstp, W[tstp], 1, 2, f, 2, 1, k, beta, h) == c₂
        end
        # 𝒻 is not allowed in the start-up region
        for tstp = 1:k
            @test_throws AssertionError response_convolution(tstp, W[tstp], 1, 1, f, 1, 1, I, beta, h)
        end
        # tstp - 1 = k is allowed
        @test response_convolution(k + 1, W[k+1], 1, 2, f, 2, 1, I, beta, h) ==
              response_convolution(k + 1, W, 1, 2, f, 2, 1, I, beta, h)
        # Mismatched time step
        @test_throws AssertionError response_convolution(5, W[6], 1, 1, f, 1, 1, I, beta, h)
        # Invalid orbital indices and order
        @test_throws AssertionError response_convolution(5, W, 3, 1, f, 1, 1, I, beta, h)
        @test_throws AssertionError response_convolution(5, W, 1, 1, f, 1, 0, I, beta, h)
        @test_throws AssertionError response_convolution(5, W, 1, 1, f, 1, 1, 6, beta, h)
        # Start-up region requires the time steps 1, ⋯, k + 1
        Cs = Cn(3, ntau, 2, 2, 0.2, beta)
        Ws = ℱ(Cs)
        @test_throws AssertionError response_convolution(2, Ws, 1, 1, Cf(Cs), 1, 1, I, beta, h)
    end

    @testset "free response function" begin
        # A coarse check at quick profile: high order, moderate grid
        @test rs_free_error(5, 5, 32) < 1e-10
        @test rs_free_error(3, 3, 64) < 1e-10
    end

    #
    # Release only: convergence order gate and reduced-order mutants. The
    # observed order of the formal method is at least k + 1. For smooth
    # integrands the Gregory rule of order k is observed to converge with
    # order k + 2, so the rule of order k - 1 still reaches order k + 1.
    # Thus the mutant uses order k - 2 (k ≥ 3), which must be rejected by
    # the gate.
    #
    if kb_release()
        @testset "release: response convolution order gate" begin
            @test kb_test_profile() == "release"
            grids = Dict(1 => [64, 128, 256, 512], 2 => [16, 32, 64, 128],
                         3 => [8, 16, 32, 64], 4 => [4, 8, 16, 32],
                         5 => [6, 8, 12, 16, 24])
            for k = 1:5
                Ns = grids[k]
                errs = [rs_free_error(k, k, N) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                if k ≥ 3
                    errm = [rs_free_error(k, k - 2, N) for N in Ns]
                    @test !check_order_gate(Ns, errm, k + 1, 1e-8)[1]
                end
            end
        end
    end
end
