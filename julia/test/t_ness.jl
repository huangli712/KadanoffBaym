#
# t_ness.jl
#
# To test the NESS2 data layer: NessGrid, NessArray, NessHermMat, the
# element access, the algebra, the bare transforms, the physical integral
# transforms, and the plain text I/O.
#
# Ported NESSi Catch tests (non-HDF5): ness2_grid_info.cpp,
# ness2_fft_array_basic.cpp:6-103, ness2_fft_array_elements.cpp,
# ness2_fft_array_ops.cpp, ness2_herm_matrix_ness_elements.cpp,
# ness2_herm_matrix_ness_ops.cpp, and
# ness2_herm_matrix_ness_integral_transform.cpp (Nft = 16384 in the
# release profile).
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const KB = KadanoffBaym

# The Catch test of the integral transforms, ness2_herm_matrix_ness_
# integral_transform.cpp. Returns the four round-trip errors / Nft.
function ness_catch_roundtrip(Nft, method)
    h = 0.01
    nt = Nft ÷ 2 - 1
    f(x) = complex(sin(h * x), cos(h * x)) * exp(-h * h * x * x)
    errs = Float64[]
    for comp in (:ret, :les)
        set! = comp == :ret ? set_ret! : set_les!
        A(G) = comp == :ret ? G.ret : G.les
        # Initialized in time
        S = NessHermMat(NessGrid(Nft, h), 1)
        for t = 0:nt
            v = f(t)
            t == 0 && (v = v - conj(v))
            set!(S, t + 1, v, FFT_TIME)
        end
        integral_transform_to_freq!(S, method)
        S₁ = deepcopy(S)
        integral_transform_to_time!(S₁, method)
        push!(errs, distance(A(S), A(S₁), FFT_TIME) / Nft)
        # Initialized in frequency
        W = NessHermMat(NessGrid(Nft, h), 1)
        for w = 0:Nft-1
            set!(W, w + 1, f(w), FFT_FREQ)
        end
        integral_transform_to_time!(W, method)
        W₁ = deepcopy(W)
        integral_transform_to_freq!(W₁, method)
        push!(errs, distance(A(W), A(W₁), FFT_FREQ) / Nft)
    end
    return errs
end

@testset verbose = true "KadanoffBaym: NESS2 data layer" begin

    @testset "grid info (ness2_grid_info.cpp)" begin
        Nt, h₁, h₂, dw₂ = 1000, 0.1, 0.01, 0.1
        g = NessGrid(Nt, h₁)
        @test g.h == h₁ && g.Nft == Nt
        @test abs(g.dw - π / ((Nt ÷ 2) * h₁)) < 1e-10
        expect(g, n, s) = (n - 1 < Nt ÷ 2 ? n - 1 : n - 1 - Nt) * s
        for g in (g, set_h(g, h₂), set_dw(g, dw₂))
            @test sum(abs(time_at(g, n) - expect(g, n, g.h)) for n = 1:Nt) / Nt < 1e-10
            @test sum(abs(freq_at(g, n) - expect(g, n, g.dw)) for n = 1:Nt) / Nt < 1e-10
        end
        @test set_h(g, h₂).h == h₂
        @test set_dw(g, dw₂).dw == dw₂
        @test abs(set_dw(g, dw₂).h - π / ((Nt ÷ 2) * dw₂)) < 1e-14
        # 1-based storage index
        g = NessGrid(16, 0.25)
        @test g.dw ≈ π / 2
        @test time_at(g, 1) == 0.0
        @test time_at(g, 8) == 7 * 0.25
        @test time_at(g, 9) == -8 * 0.25
        @test time_at(g, 16) == -0.25
        @test freq_at(g, 9) == -8 * g.dw
        @test_throws BoundsError freq_at(g, 0)
        @test_throws BoundsError time_at(g, 17)
        @test_throws ArgumentError NessGrid(15, 0.1)
        @test_throws ArgumentError NessGrid(2, 0.1)
        @test_throws ArgumentError NessGrid(16, 0.0)
        @test_throws ArgumentError set_h(g, -1.0)
        @test_throws ArgumentError set_dw(g, 0.0)
    end

    @testset "NessArray basic (ness2_fft_array_basic.cpp)" begin
        A = NessArray(16, 2)
        @test getnft(A) == 16 && getdims(A) == (2, 2)
        @test size(A.time) == (2, 2, 16) && size(A.freq) == (2, 2, 16)
        @test strides(A.time) == (1, 2, 4)
        @test length(A.scratch) == KB.plan_fft(16, FFT_BACKWARD).scratch_len
        @test !(A isa CnAbstractType)
        # Round trip of the bare transforms, unnormalized
        Nft = 8
        A = NessArray(Nft, 2)
        for t = 1:Nft
            A[t, FFT_TIME] = C64[(t - 1) + (i - 1) * 0.1 + im * (j - 1) * 0.2 for i = 1:2, j = 1:2]
        end
        orig = copy(A.time)
        fft_to_freq!(A)
        @test A.time == orig
        fft_to_time!(A)
        @test norm(A.time / Nft - orig) < 1e-10
        # Signs: a pulse at t = h gives exp(+iω h) in frequency
        B = NessArray(Nft, 1)
        B[2, FFT_TIME] = 1.0 + 0im
        fft_to_freq!(B)
        @test all(k -> B.freq[1,1,k] ≈ cispi(2.0 * (k - 1) / Nft), 1:Nft)
    end

    @testset "NessArray elements (ness2_fft_array_elements.cpp)" begin
        A = NessArray(4, 2)
        M = fill(C64(1.0), 2, 2)
        A[1, FFT_TIME] = M
        @test A[1, FFT_TIME] == M
        # Copy on get and set, no aliasing
        X = A[1, FFT_TIME]
        X[1, 1] = 5.0
        @test A[1, FFT_TIME] == M
        M[2, 2] = 7.0
        @test A[1, FFT_TIME][2, 2] == 1.0
        # Scalar
        S = NessArray(4, 1)
        S[1, FFT_TIME] = 1.23 + 4.56im
        @test S[1, FFT_TIME][1] == 1.23 + 4.56im
        @test_throws ArgumentError (A[1, FFT_TIME] = 1.0 + 0im)
        @test_throws DimensionMismatch (A[1, FFT_TIME] = ones(C64, 3, 3))
        @test_throws BoundsError A[5, FFT_TIME]
        # set_matrixelement!
        B = NessArray(4, 2)
        for t = 1:4
            B[t, FFT_TIME] = C64[0 1+1im; 0 0]
        end
        A = NessArray(4, 2)
        set_matrixelement!(A, 2, 1, B, 1, 2, FFT_TIME)
        @test all(t -> A[t, FFT_TIME][2, 1] == 1.0 + 1.0im, 1:4)
        @test_throws DimensionMismatch set_matrixelement!(A, 2, 1, NessArray(8, 2), 1, 2, FFT_TIME)
    end

    @testset "NessArray ops (ness2_fft_array_ops.cpp)" begin
        A = NessArray(4, 2)
        B = NessArray(4, 2)
        for t = 1:4
            B[t, FFT_TIME] = C64[(t - 1) + (i - 1) + (j - 1) + (i - 1) * (j - 1) for i = 1:2, j = 1:2]
        end
        zeros!(A, FFT_TIME)
        incr!(A, B, 2.0, FFT_TIME)
        @test norm(A.time - 2 * B.time) < 1e-12
        smul!(A, 0.5, FFT_TIME)
        @test norm(A.time - B.time) < 1e-12
        # The frequency data is not touched
        @test all(iszero, A.freq)
        # Left and right multiplication by the swap matrix, and a
        # non-commuting check
        for t = 1:4
            A[t, FFT_TIME] = C64[1+(i-1)+(j-1)+im*(t-1) for i = 1:2, j = 1:2]
        end
        Mswap = C64[0 1; 1 0]
        C = deepcopy(A)
        left_multiply!(C, Mswap, FFT_TIME)
        left_multiply!(C, Mswap, FFT_TIME)
        @test norm(C.time - A.time) < 1e-12
        right_multiply!(C, Mswap, FFT_TIME)
        right_multiply!(C, Mswap, FFT_TIME)
        @test norm(C.time - A.time) < 1e-12
        P = C64[1 2im; 0 3]
        left_multiply!(C, P, FFT_TIME)
        @test C[3, FFT_TIME] ≈ P * A[3, FFT_TIME]
        C = deepcopy(A)
        right_multiply!(C, P, FFT_TIME; hermconj = true)
        @test C[2, FFT_TIME] ≈ A[2, FFT_TIME] * P'
        C = deepcopy(A)
        left_multiply!(C, P, FFT_TIME; hermconj = true)
        @test C[2, FFT_TIME] ≈ P' * A[2, FFT_TIME]
        @test_throws DimensionMismatch left_multiply!(C, ones(C64, 3, 3), FFT_TIME)
        # memset!, zeros!, memcpy!
        memset!(A, FFT_FREQ, 2.0)
        @test all(==(2.0 + 0im), A.freq)
        zeros!(A, FFT_TIME)
        zeros!(A, FFT_FREQ)
        @test all(iszero, A.time) && all(iszero, A.freq)
        D = NessArray(4, 2)
        memcpy!(B, D)
        @test D == B
        @test_throws DimensionMismatch incr!(A, NessArray(8, 2), 1.0, FFT_TIME)
        @test_throws DimensionMismatch memcpy!(A, NessArray(4, 3))
        # distance
        @test distance(A, A, FFT_TIME) == 0.0
        E = NessArray(4, 2)
        E.time[1, 2, 3] = 3.0 + 4.0im
        @test distance(E, NessArray(4, 2), FFT_TIME) == 5.0
    end

    @testset "NessHermMat elements and ops" begin
        G = NessHermMat(NessGrid(16, 0.1), 2)
        @test G isa CnAbstractStationaryFunction{C64}
        @test !(G isa CnAbstractFunction)
        @test !(G.ret isa CnAbstractType)
        @test size(G.ret.time) == (2, 2, 16)
        @test getnft(G) == 16 && getdims(G) == (2, 2)
        set_les!(G, 16, ones(C64, 2, 2), FFT_TIME)
        @test get_les(G, 16, FFT_TIME) == ones(2, 2)
        M = C64[1 2; 3 4]
        set_ret!(G, 1, M, FFT_FREQ)
        @test get_ret(G, 1, FFT_FREQ) == M
        # Scalar
        S = NessHermMat(NessGrid(4, 0.1), 1)
        set_ret!(S, 1, 1.23 + 4.56im, FFT_TIME)
        set_les!(S, 1, 1.23 + 4.56im, FFT_TIME)
        @test get_ret(S, 1, FFT_TIME)[1] == 1.23 + 4.56im
        @test get_les(S, 1, FFT_TIME)[1] == 1.23 + 4.56im
        # Construction against NessArray (ness2_herm_matrix_ness_elements.cpp)
        G = NessHermMat(NessGrid(4, 0.1), 2)
        R = NessArray(4, 2)
        for t = 1:4
            X = C64[(t - 1) + (i - 1) + im * (-2 * (t - 1) + (j - 1)) for i = 1:2, j = 1:2]
            R[t, FFT_TIME] = X
            set_ret!(G, t, X, FFT_TIME)
            set_les!(G, t, X, FFT_TIME)
        end
        @test distance(G.ret, R, FFT_TIME) < 1e-12
        @test distance(G.les, R, FFT_TIME) < 1e-12
        # Ops (ness2_herm_matrix_ness_ops.cpp)
        A = NessHermMat(NessGrid(4, 0.1), 2)
        incr!(A, G, 2.0, FFT_TIME)
        @test norm(A.ret.time - 2 * G.ret.time) < 1e-12
        @test norm(A.les.time - 2 * G.les.time) < 1e-12
        smul!(A, 0.5, FFT_TIME)
        @test distance(A, G, FFT_TIME) < 1e-12
        Mswap = C64[0 1; 1 0]
        B = deepcopy(G)
        left_multiply!(B, Mswap, FFT_TIME)
        left_multiply!(B, Mswap, FFT_TIME)
        right_multiply!(B, Mswap, FFT_TIME)
        right_multiply!(B, Mswap, FFT_TIME)
        @test distance(B, G, FFT_TIME) < 1e-12
        memset!(B, FFT_FREQ, 1.0)
        zeros!(B, FFT_TIME)
        zeros!(B, FFT_FREQ)
        @test all(iszero, B.ret.time) && all(iszero, B.les.freq)
        C = NessHermMat(NessGrid(4, 0.1), 2)
        memcpy!(G, C)
        @test C == G
        set_matrixelement!(C, 1, 1, G, 2, 2, FFT_TIME)
        @test all(t -> get_les(C, t, FFT_TIME)[1, 1] == get_les(G, t, FFT_TIME)[2, 2], 1:4)
        @test iscompatible(C, G)
        @test !iscompatible(C, NessHermMat(NessGrid(8, 0.1), 2))
    end

    @testset "integral transforms: analytic retarded function (method = $m)" for m in (FFT_TRAPEZ, FFT_CUBIC)
        # G^R(t) = -i e^{-(η + iε) t}, G^R(ω) = 1/(ω - ε + iη)
        ε, η = 0.3, 1.0
        errs = Float64[]
        for h in (0.1, 0.05, 0.025)
            Nft = round(Int, 102.4 / h)
            G = NessHermMat(NessGrid(Nft, h), 1)
            for n = 1:Nft÷2
                t = (n - 1) * h
                set_ret!(G, n, -im * exp(-(η + im * ε) * t), FFT_TIME)
            end
            integral_transform_to_freq!(G, m)
            # Nyquist point
            @test G.ret.freq[1, 1, Nft÷2+1] == 0
            g = G.grid
            e = maximum(n -> abs(G.ret.freq[1,1,n] - 1 / (freq_at(g, n) - ε + im * η)),
                        [1, 2, 5, 11, Nft])
            push!(errs, e)
        end
        if m == FFT_TRAPEZ
            # Second order
            @test errs[1] / errs[2] > 3.5 && errs[2] / errs[3] > 3.5
        else
            # Fourth order, more accurate than the trapezoidal rule
            @test errs[1] / errs[2] > 12 && errs[2] / errs[3] > 12
            @test errs[3] < 1e-8
        end
    end

    @testset "integral transforms: physical round trip (method = $m)" for m in (FFT_TRAPEZ, FFT_CUBIC)
        # Band-limited and decaying data, G^R(0) and G^<(0) anti-hermitian
        h, Nft = 0.1, 512
        G = NessHermMat(NessGrid(Nft, h), 2)
        U = C64[1 1im; 1im 1] / sqrt(2.0)
        for n = 1:Nft
            t = time_at(G.grid, n)
            D = Diagonal([exp(-t^2 / 2), 0.5 * exp(-t^2 / 3)])
            L = im .* (U * D * U')
            set_les!(G, n, L, FFT_TIME)
            t ≥ 0 && n ≤ Nft ÷ 2 && set_ret!(G, n, -im .* (U * D * U') .* cis(-0.3 * t), FFT_TIME)
        end
        G₀ = deepcopy(G)
        integral_transform_to_freq!(G, m)
        integral_transform_to_time!(G, m)
        # The trapezoidal transforms are inverse to each other on this
        # subspace. The cubic correction factors W(ωh) and W(-tΔω) of the
        # two directions are not, the round trip is accurate to O(h⁴).
        tol = m == FFT_TRAPEZ ? 1e-10 : 1e-6
        @test distance(G.ret, G₀.ret, FFT_TIME) / Nft < tol
        @test distance(G.les, G₀.les, FFT_TIME) / Nft < tol
        # Lesser hermiticity G^<(-t) = -G^<(t)†
        @test all(n -> get_les(G, Nft - n + 2, FFT_TIME) ≈ -get_les(G, n, FFT_TIME)', 2:Nft÷2)
        @test all(iszero, G.les.time[:, :, Nft÷2+1])
        # Retarded at negative times is zero
        @test all(iszero, G.ret.time[:, :, Nft÷2+1:end])
        @test_throws ArgumentError integral_transform_to_freq!(G, FFT_ADAPTIVE)
        @test_throws ArgumentError integral_transform_to_time!(G, FFT_ADAPTIVE)
    end

    @testset "integral transforms: Catch round trip (Nft = 1024)" begin
        # Same data as the NESSi Catch test on a 16 times smaller grid, the
        # errors / Nft scale as 1/Nft, so the Catch tolerance 1e-3 at
        # Nft = 16384 corresponds to 1.6e-2 here
        for m in (FFT_TRAPEZ, FFT_CUBIC)
            errs = ness_catch_roundtrip(1024, m)
            @test all(e -> isfinite(e) && e < 1.6e-2, errs)
        end
    end

    @testset "plain text I/O" begin
        G = NessHermMat(NessGrid(8, 0.1), 2)
        for X in (G.ret.time, G.ret.freq, G.les.time, G.les.freq)
            X .= rand(C64, size(X))
        end
        path = tempname()
        write_ness(path, G)
        H = read_ness(path)
        @test H == G
        A = NessArray(4, 2)
        A.time .= rand(C64, size(A.time))
        A.freq .= rand(C64, size(A.freq))
        write_fftarray(path, A)
        @test read_fftarray(path) == A
        # Corrupted files
        lines = readlines(path)
        bad(ls) = (write(path, join(ls, "\n") * "\n"); path)
        @test_throws ArgumentError read_fftarray(bad(lines[1:end-1]))
        @test_throws ArgumentError read_fftarray(bad([lines[1:end-1]; lines[2]]))
        @test_throws ArgumentError read_fftarray(bad(["# FFTArray 4"; lines[2:end]]))
        @test_throws ArgumentError read_fftarray(bad(["# HermMatrixNESS 4 2 0.1 1.0"; lines[2:end]]))
        @test_throws ArgumentError read_fftarray(bad([lines[1]; replace(lines[2], "time" => "tyme"); lines[3:end]]))
        @test_throws ArgumentError read_fftarray(bad([lines[1]; "time data 9 1 1 0.0 0.0"; lines[3:end]]))
        @test_throws ArgumentError read_fftarray(bad([lines[1]; "time data 1 1 1 NaN 0.0"; lines[3:end]]))
        @test_throws ArgumentError read_fftarray(bad([lines[1]; "time data 1 1 1 0.0"; lines[3:end]]))
        write_ness(path, G)
        lines = readlines(path)
        @test_throws ArgumentError read_ness(bad([lines[1]; replace(lines[2], "ret" => "adv"); lines[3:end]]))
        @test_throws ArgumentError read_ness(bad(["# HermMatrixNESS 7 2 0.1 1.0"; lines[2:end]]))
        rm(path)
    end

    if kb_release()
        @testset "release: Catch integral transform round trip (Nft = 16384)" begin
            @test kb_test_profile() == "release"
            for m in (FFT_TRAPEZ, FFT_CUBIC)
                errs = ness_catch_roundtrip(16384, m)
                @test all(e -> e < 1e-3, errs)
            end
        end
    end
end
