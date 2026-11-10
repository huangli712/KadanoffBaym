#
# t_fft.jl
#
# To test the pure Julia FFT: plan_fft(), allocate_fft_scratch(),
# clear_fft_plan_cache!(), fft(), fft!(), and fft_strided!().
#
# The oracle is the naive O(N²) DFT. The round-trip semantics follows
# NESSi ness2_fft_array_basic.cpp. The production gate (release only)
# covers N = 4093 (prime, Bluestein), 4096, and 16384 (NESSi
# ness2_herm_matrix_ness_integral_transform.cpp), the 2 × 2 batched
# transform along dim = 3, the allocation of the hot calls, and the
# relative scaling of the run time.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using Random
using LinearAlgebra
using KadanoffBaym

# Naive DFT with the exponent sign s
function ft_naive(x::AbstractVector, s::Integer)
    N = length(x)
    return ComplexF64[sum(x[n+1] * cispi(2.0 * s * mod(n * k, N) / N) for n = 0:N-1)
                      for k = 0:N-1]
end

# Relative error, the absolute error is used for near-zero references
function ft_err(y, r)
    nr = norm(r)
    return nr > 1e-12 ? norm(y - r) / nr : norm(y - r)
end

@testset verbose = true "KadanoffBaym: FFT" begin

    Random.seed!(20261110)
    Ns = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 25, 27, 32, 49, 60, 64, 97, 101, 127, 256)

    @testset "accuracy against the naive DFT (N = $N)" for N in Ns
        data = (zeros(ComplexF64, N),
                [i == 1 ? 1.0 + 0.0im : 0.0im for i = 1:N],
                rand(ComplexF64, N) .- (0.5 + 0.5im),
                ComplexF64[exp(10.0 * randn()) * cis(2π * rand()) for _ = 1:N])
        for x in data, d in (FFT_FORWARD, FFT_BACKWARD)
            r = ft_naive(x, Int(d))
            x0 = copy(x)
            # Out of place, the input is not changed
            y = fft(x, d)
            @test x == x0
            @test ft_err(y, r) ≤ 1e-12
            # In place with an explicit plan and scratch
            p = plan_fft(N, d)
            s = allocate_fft_scratch(p)
            z = copy(x)
            @test fft!(z, d; plan = p, scratch = s) === z
            @test ft_err(z, r) ≤ 1e-12
        end
        # Real input
        xr = rand(N)
        @test ft_err(fft(xr), ft_naive(xr, -1)) ≤ 1e-12
        # Round trip
        x = rand(ComplexF64, N)
        @test norm(fft(fft(x), FFT_BACKWARD) / N - x) / norm(x) ≤ 1e-12
    end

    @testset "sign and normalization" begin
        # Single frequency: x_n = exp(+2πi n k₀/N) is a delta for FFT_FORWARD
        for N in (12, 97), k₀ in (1, 5)
            x = ComplexF64[cispi(2.0 * mod(n * k₀, N) / N) for n = 0:N-1]
            y = fft(x, FFT_FORWARD)
            @test abs(y[k₀+1] - N) < 1e-10
            @test norm(y[setdiff(1:N, k₀ + 1)]) < 1e-10
            z = fft(x, FFT_BACKWARD)
            @test abs(z[mod(-k₀, N)+1] - N) < 1e-10
        end
        # Unnormalized
        @test fft(ones(ComplexF64, 7))[1] ≈ 7.0
        @test fft(ones(ComplexF64, 8), FFT_BACKWARD)[1] ≈ 8.0
    end

    @testset "plans, cache, and scratch" begin
        clear_fft_plan_cache!()
        p = plan_fft(4093, FFT_FORWARD)
        @test p.n == 4093
        @test p.bluestein_n == 8192
        @test p.scratch_len == 2 * p.bluestein_n
        @test length(allocate_fft_scratch(p)) == p.scratch_len
        @test plan_fft(4093, FFT_FORWARD) === p
        @test plan_fft(4093, FFT_BACKWARD) !== p
        clear_fft_plan_cache!()
        @test plan_fft(4093, FFT_FORWARD) !== p
        #
        q = plan_fft(60)
        @test q.bluestein_n == 0
        @test q.scratch_len == 60
        @test prod(q.factors) == 60
        @test plan_fft(1).scratch_len == 1
        # Mixed products with a large prime use Bluestein
        @test plan_fft(34).bluestein_n == 128
        @test plan_fft(26).bluestein_n == 0
    end

    @testset "failure paths" begin
        @test_throws ArgumentError plan_fft(0)
        @test_throws ArgumentError fft(ComplexF64[])
        p = plan_fft(8)
        @test_throws DimensionMismatch fft!(zeros(ComplexF64, 9); plan = p)
        @test_throws DimensionMismatch fft!(zeros(ComplexF64, 8); plan = p,
                                            scratch = zeros(ComplexF64, 7))
        @test_throws ArgumentError fft!(zeros(ComplexF64, 8), FFT_BACKWARD; plan = p)
        pb = plan_fft(97)
        @test_throws DimensionMismatch fft!(zeros(ComplexF64, 97); plan = pb,
                                            scratch = zeros(ComplexF64, pb.scratch_len - 1))
        A = zeros(ComplexF64, 2, 2, 8)
        @test_throws ArgumentError fft_strided!(A, A, 4, FFT_FORWARD)
        @test_throws DimensionMismatch fft_strided!(zeros(ComplexF64, 2, 2, 7), A, 3, FFT_FORWARD)
        @test_throws DimensionMismatch fft_strided!(A, A, 3, FFT_FORWARD; plan = plan_fft(7))
    end

    @testset "strided and batched transforms" begin
        for N in (30, 97)
            A = rand(ComplexF64, 2, 2, N)
            B = similar(A)
            p = plan_fft(N, FFT_FORWARD)
            s = allocate_fft_scratch(p)
            fft_strided!(B, A, 3, FFT_FORWARD; plan = p, scratch = s)
            for i = 1:2, j = 1:2
                @test ft_err(B[i,j,:], ft_naive(A[i,j,:], -1)) ≤ 1e-12
            end
            C = similar(A)
            pb = plan_fft(N, FFT_BACKWARD)
            fft_strided!(C, B, 3, FFT_BACKWARD; plan = pb, scratch = allocate_fft_scratch(pb))
            @test norm(C / N - A) / norm(A) < 1e-12
            # In place, dest === src
            D = copy(A)
            fft_strided!(D, D, 3, FFT_FORWARD)
            @test D == B
            # Other dimensions
            E = similar(A)
            fft_strided!(E, A, 1, FFT_FORWARD)
            @test ft_err(E[:,2,5], ft_naive(A[:,2,5], -1)) ≤ 1e-12
        end
        A = rand(ComplexF64, 2, 2, 30)
        @test strides(A) == (1, 2, 4)
        @test size(@view A[:,:,1]) == (2, 2)
    end

    if kb_release()
        @testset "release: production gate" begin
            @test kb_test_profile() == "release"
            times = Dict{Int,Float64}()
            for N in (4093, 4096, 16384)
                # Pulse and single frequency analytic results
                x = zeros(ComplexF64, N)
                x[1] = 1.0
                @test all(y -> abs(y - 1.0) < 1e-12, fft(x))
                k₀ = 37
                x = ComplexF64[cispi(2.0 * mod(n * k₀, N) / N) for n = 0:N-1]
                y = fft(x)
                @test abs(y[k₀+1] - N) / N < 1e-12
                @test norm(y[setdiff(1:N, k₀ + 1)]) / N < 1e-11
                # Round trip
                x = rand(ComplexF64, N)
                p = plan_fft(N, FFT_FORWARD)
                pb = plan_fft(N, FFT_BACKWARD)
                s = allocate_fft_scratch(p)
                z = copy(x)
                fft!(z, FFT_FORWARD; plan = p, scratch = s)
                fft!(z, FFT_BACKWARD; plan = pb, scratch = s)
                @test norm(z / N - x) / norm(x) < 1e-12
                # 2 × 2 batched along dim = 3
                A = rand(ComplexF64, 2, 2, N)
                B = similar(A)
                fft_strided!(B, A, 3, FFT_FORWARD; plan = p, scratch = s)
                fft_strided!(B, B, 3, FFT_BACKWARD; plan = pb, scratch = s)
                @test norm(B / N - A) / norm(A) < 1e-12
                # Hot calls, allocations
                fft!(z, FFT_FORWARD; plan = p, scratch = s)
                a₁ = @allocated fft!(z, FFT_FORWARD; plan = p, scratch = s)
                a₂ = @allocated fft_strided!(B, A, 3, FFT_FORWARD; plan = p, scratch = s)
                @test a₁ ≤ 4096
                @test a₂ ≤ 4096
                # Minimum of the wall times
                times[N] = minimum(@elapsed(fft!(z, FFT_FORWARD; plan = p, scratch = s))
                                   for _ = 1:5)
            end
            # Relative scaling, rejects O(N²) paths
            @test times[16384] / times[4096] < 10
            @test times[4093] / times[4096] < 8
        end
    end
end
