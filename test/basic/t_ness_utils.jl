#
# t_ness_utils.jl
#
# To test the NESS2 algorithms: sigma_bath(), dyson_ness!(), the DOS
# types, from_dos_ness!(), force_equilibrium!(), the density
# matrices, the bubbles, the resampling, and the conversions between the
# NESS2 and the contour Green's functions.
#
# Ported NESSi Catch tests (non-HDF5): ness2_herm_matrix_ness_basic.cpp
# (7-269), ness2_density_matrix.cpp, and ness2_resample.cpp.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

# Contour Green's function with ntime real time points
nu_contour(ntime, ntau, d, h, β) = Cn(ntime, ntau, d, d, (ntime - 1) * h, β)

@testset verbose = true "KadanoffBaym: NESS2 algorithms" begin

    @testset "sigma_bath" begin
        r, l = sigma_bath(0.0, 0.1, BATH_CONST, 0.2, 10.0, 0.0)
        @test r ≈ -0.2im && l ≈ 0.2im
        @test sigma_bath(0.3, 0.1, BATH_NONE, 0.2, 10.0, 0.0) == (0.0im, 0.0im)
        # GAUSS: exp(-ω²) as in NESSi, independent of h
        r₁, l₁ = sigma_bath(0.7, 0.1, BATH_GAUSS, 0.2, 10.0, 0.1)
        r₂, l₂ = sigma_bath(0.7, 0.01, BATH_GAUSS, 0.2, 10.0, 0.1)
        @test r₁ == r₂ && l₁ == l₂
        @test r₁ ≈ -0.2im * exp(-0.49)
        @test l₁ ≈ 0.4im * exp(-0.49) / (1 + exp(10.0 * 0.6))
        # OHMIC: finite at ω = 0, the limit -2iη/β
        r, l = sigma_bath(0.0, 0.1, BATH_OHMIC, 0.2, 10.0, 0.0)
        @test r == 0 && l ≈ -0.04im
        r, l = sigma_bath(1e-9, 0.1, BATH_OHMIC, 0.2, 10.0, 0.0)
        @test isfinite(l) && l ≈ -0.04im
        r, l = sigma_bath(0.5, 0.1, BATH_OHMIC, 0.2, 10.0, 0.0)
        @test l ≈ -0.4im * 0.5 * exp(-0.25) / (exp(5.0) - 1)
        # Cut-offs
        @test sigma_bath(20.0, 0.1, BATH_CONST, 0.2, 10.0, 0.0)[2] == 0
        @test sigma_bath(-20.0, 0.1, BATH_CONST, 0.2, 10.0, 0.0)[2] ≈ 0.4im
        @test_throws ArgumentError sigma_bath(0.0, 0.1, BATH_CONST, -0.2, 10.0, 0.0)
        @test_throws ArgumentError sigma_bath(0.0, 0.1, BATH_CONST, 0.2, 0.0, 0.0)
        @test_throws ArgumentError sigma_bath(NaN, 0.1, BATH_CONST, 0.2, 1.0, 0.0)
    end

    @testset "DOS types" begin
        d = BetheDOS(-2, 2)
        @test d(3) == 0 && d(0) > 0
        @test d(0) ≈ 1 / π
        @test_throws ArgumentError BetheDOS(2, -2)
        g = GaussianDOS(0.0, 2.0)
        @test g.lo == -12 && g.hi == 12
        @test g(0.0) == 1 / (2.0 * 2.506623299999997)
        @test g(20.0) > 0
        @test_throws ArgumentError GaussianDOS(0.0, 0.0)
        o = OhmicSymDOS(1.0)
        @test o(-0.5) == -o(0.5) && o(0.0) == 0
        @test o.lo == -20 && o.hi == 20
        @test_throws ArgumentError OhmicSymDOS(-1.0)
    end

    @testset "dyson_ness!: single level with a constant bath" begin
        # G^R(t) = -i e^{-i(ε-μ)t - ηt}, up to the band truncation
        Nft, h = 8192, 0.01
        ε, η, β, μ = 0.3, 0.5, 5.0, 0.1
        G = HermMatrixNESS(GridInfo(Nft, h), 1)
        S = HermMatrixNESS(GridInfo(Nft, h), 1)
        dyson_ness!(G, μ, fill(C64(ε), 1, 1), S; bath_type = BATH_CONST,
                    bath_eta = η, bath_beta = β, bath_mu = μ)
        e = maximum(n -> abs(G.ret.time[1,1,n] + im * exp(-(im * (ε - μ) + η) * (n - 1) * h)), 2:400)
        @test e < 5e-3
        @test all(iszero, G.ret.time[:, :, Nft÷2+1:end])
        @test_throws ArgumentError dyson_ness!(G, μ, fill(C64(ε), 1, 1), S; method = FFT_CUBIC)
        @test_throws DimensionMismatch dyson_ness!(G, μ, zeros(C64, 2, 2), S)
        @test_throws ArgumentError dyson_ness!(G, μ, fill(C64(ε), 1, 1), G)
    end

    @testset "dyson_ness!: sentinels, double call, Sigma unchanged" begin
        Nft, h = 64, 0.1
        g = GridInfo(Nft, h)
        H = C64[0.2 0.1im; -0.1im -0.3]
        function sigma(scale)
            S = HermMatrixNESS(g, 2)
            for n = 1:Nft
                t = time_at(g, n)
                D = scale * exp(-t^2) * C64[1 0.2; 0.2 0.5]
                set_les!(S, n, im .* D, FFT_TIME)
                0 ≤ t && n ≤ Nft ÷ 2 && set_ret!(S, n, -im .* D, FFT_TIME)
            end
            return S
        end
        S₁ = sigma(0.3)
        S₀ = deepcopy(S₁)
        G = HermMatrixNESS(g, 2)
        G.ret.freq .= reshape(1:4Nft, 2, 2, Nft) .+ 0.5im
        G.les.freq .= reshape(4Nft:-1:1, 2, 2, Nft) .- 0.25im
        dyson_ness!(G, 0.1, H, S₁; bath_type = BATH_CONST, bath_eta = 0.1, bath_beta = 2.0)
        @test S₁ == S₀
        nfreq = (Nft ÷ 2 - 1) ÷ 3
        band = Set(mod(w, Nft) + 1 for w = -nfreq+1:nfreq-1)
        @test all(n -> n in band || (iszero(G.ret.freq[:, :, n]) && iszero(G.les.freq[:, :, n])), 1:Nft)
        # Second call with other sentinels and another self-energy
        S₂ = sigma(0.7)
        G.ret.freq .= 7.0
        G.les.freq .= -3.0im
        dyson_ness!(G, 0.1, H, S₂; bath_type = BATH_CONST, bath_eta = 0.1, bath_beta = 2.0)
        F = HermMatrixNESS(g, 2)
        dyson_ness!(F, 0.1, H, S₂; bath_type = BATH_CONST, bath_eta = 0.1, bath_beta = 2.0)
        @test G == F
        # Lesser hermiticity and Nyquist
        @test all(n -> get_les(G, Nft - n + 2, FFT_TIME) ≈ -get_les(G, n, FFT_TIME)', 2:Nft÷2)
        @test iszero(G.les.time[:, :, Nft÷2+1])
    end

    @testset "from_dos_ness!: TRAPEZ and ADAPTIVE (sign = $sign)" for sign in (FERMI, BOSE)
        dos = sign == FERMI ? GaussianDOS(0.0, 1.0) : OhmicSymDOS(1.0)
        μ = sign == FERMI ? 0.3 : 0.0
        A = HermMatrixNESS(GridInfo(1024, 0.1), 2)
        B = HermMatrixNESS(GridInfo(1024, 0.1), 2)
        from_dos_ness!(A, sign, dos, 2.0, μ; method = FFT_TRAPEZ)
        from_dos_ness!(B, sign, dos, 2.0, μ; method = FFT_ADAPTIVE)
        @test maximum(abs.(A.ret.time[:, :, 1:200] - B.ret.time[:, :, 1:200])) < 1e-5
        @test maximum(abs.(A.les.time[:, :, 1:200] - B.les.time[:, :, 1:200])) < 1e-3
        # Spectral normalization and diagonal structure. GaussianDOS uses
        # the approximate constant 2.506623299999997 of NESSi, so its
        # norm is √(2π)/2.506623299999997; the ohmic DOS is odd.
        if sign == FERMI
            @test A.ret.time[1, 1, 1] ≈ -im * sqrt(2π) / 2.506623299999997 atol = 1e-10
        else
            @test abs(A.ret.time[1, 1, 1]) < 1e-10
        end
        @test A.ret.time[1, 2, 5] == 0
        @test all(iszero, A.ret.time[:, :, 513:end])
        if sign == FERMI
            # Occupation n = ∫ A f, G^<(0) = i n
            n = sum(x -> dos(x) / (1 + exp(2.0 * (x - μ))), range(-6, 6, length = 24001)) * 12 / 24000
            @test A.les.time[1, 1, 1] ≈ im * n atol = 1e-8
        end
        # A plain function needs the window
        @test_throws ArgumentError from_dos_ness!(B, sign, x -> dos(x), 2.0, μ;
                                                         method = FFT_ADAPTIVE)
        from_dos_ness!(B, sign, x -> dos(x), 2.0, μ; method = FFT_ADAPTIVE,
                                lo = dos.lo, hi = dos.hi)
        @test isfinite(B.les.time[1, 1, 1])
        @test_throws ArgumentError from_dos_ness!(B, 0, dos, 2.0, μ)
        @test_throws ArgumentError from_dos_ness!(B, sign, dos, 2.0, μ; method = FFT_CUBIC)
    end

    @testset "from_dos_ness! vs contour from_dos! (Catch)" begin
        Nft, β, μ, h = 64, 10.0, 0.0, 0.1
        dos = GaussianDOS(0.0, 2.0)
        C = nu_contour(Nft ÷ 2 + 1, 11, 1, h, β)
        Gc = ℱ(C, FERMI)
        from_dos!(Gc, dos, dos.lo, dos.hi, β, h; μ = μ, limit = 100, nn = 20)
        Gf = HermMatrixNESS(GridInfo(Nft, h), 1)
        cntr2ness!(Gf, Gc, Nft ÷ 2)
        Gd = HermMatrixNESS(GridInfo(Nft, h), 1)
        from_dos_ness!(Gd, FERMI, dos, β, μ; method = FFT_ADAPTIVE)
        @test maximum(abs.(Gf.ret.time .- Gd.ret.time)) < 1e-12
        @test maximum(abs.(Gf.les.time .- Gd.les.time)) < 1e-12
    end

    @testset "force_equilibrium!" begin
        # Catch case: the retarded data is put into the frequency domain,
        # which is overwritten, both results are zero
        Nft, β, μ, h = 64, 10.0, 0.0, 0.1
        dos = GaussianDOS(0.0, 2.0)
        Gr = HermMatrixNESS(GridInfo(Nft, h), 1)
        from_dos_ness!(Gr, FERMI, dos, β, μ; method = FFT_ADAPTIVE)
        Ge = HermMatrixNESS(GridInfo(Nft, h), 1)
        Ge.ret.freq .= Gr.ret.time
        force_equilibrium!(Ge, FERMI, β, μ)
        @test distance(Ge, Gr, FFT_FREQ) / Nft < 1e-10
        # Physical case: agrees with from_dos_ness! (TRAPEZ). For
        # bosons the band lies above μ and μ is not a frequency mode. A
        # DOS with square-root edges (Bethe) is not used here: the
        # trapezoidal sum of its norm has an O(dw^1.5) error, which shows
        # up as a constant spectral weight outside the band.
        for (sign, d, m) in ((FERMI, GaussianDOS(0.0, 1.0), 0.3), (BOSE, GaussianDOS(3.0, 0.5), -1.0))
            G = HermMatrixNESS(GridInfo(1024, 0.1), 2)
            from_dos_ness!(G, sign, d, 2.0, m)
            F = deepcopy(G)
            F.les.time .= 0
            T = copy(F.ret.time)
            force_equilibrium!(F, sign, 2.0, m)
            @test F.ret.time == T
            @test distance(F.les, G.les, FFT_TIME) / 1024 < 1e-10
            @test iszero(F.les.time[:, :, 513])
        end
        # Bosonic zero mode ω = μ = 0, linear zero crossing
        lin(x) = x * exp(-x^2)
        G = HermMatrixNESS(GridInfo(1024, 0.1), 1)
        from_dos_ness!(G, BOSE, lin, 2.0, 0.0)
        F = deepcopy(G)
        force_equilibrium!(F, BOSE, 2.0, 0.0)
        @test all(isfinite, F.les.time)
        R = F.ret.freq
        spec(n) = -(R[1, 1, n] - conj(R[1, 1, n])) / (2π * im)
        slope = 0.5 * ((spec(2) - spec(1)) + (spec(1) - spec(1024))) / F.grid.dw
        Gfreq = deepcopy(F)
        integral_transform_to_freq!(Gfreq, FFT_TRAPEZ)
        @test abs(F.les.freq[1, 1, 1] - (-2π * im / 2.0) * slope) < 1e-12 * abs(slope)
        @test distance(F.les, G.les, FFT_TIME) / 1024 < 1e-4
        # Non-zero and non-linear zero modes
        for bad in (x -> exp(-x^2), x -> x^2 * exp(-x^2))
            G = HermMatrixNESS(GridInfo(1024, 0.1), 1)
            from_dos_ness!(G, FERMI, bad, 2.0, 0.0)
            @test_throws DomainError force_equilibrium!(G, BOSE, 2.0, 0.0)
        end
        @test_throws ArgumentError force_equilibrium!(G, 0, 2.0, 0.0)
    end

    @testset "density matrices (Catch)" begin
        # The Catch grid. A smaller NESS grid (Nt = 256, time window 1.28)
        # does not resolve the Green's functions, the tolerances of the
        # Catch test then fail.
        h, Nt, ntau, β, μ = 0.01, 1024, 100, 10.0, 0.0
        Gn = HermMatrixNESS(GridInfo(Nt, h), 1)
        dos = GaussianDOS(0.0, 2.0)
        from_dos_ness!(Gn, FERMI, dos, β, μ)
        C = nu_contour(Nt + 1, ntau + 1, 1, h, β)
        Gc = ℱ(C, FERMI)
        from_dos!(Gc, dos, dos.lo, dos.hi, β, h; μ = μ)
        @test abs(real(density_matrix(Gc, 1)[1]) - real(density_matrix(Gn, FERMI)[1])) < 1e-6
        @test_throws ArgumentError density_matrix(Gn, 0)
        # Convolution density matrix
        dosb = GaussianDOS(0.0, 4.0)
        An = deepcopy(Gn)
        Bn = HermMatrixNESS(GridInfo(Nt, h), 1)
        from_dos_ness!(Bn, FERMI, dosb, β, μ)
        Bc = ℱ(C, FERMI)
        from_dos!(Bc, dosb, dosb.lo, dosb.hi, β, h; μ = μ)
        A₀ = deepcopy(An)
        B₀ = deepcopy(Bn)
        ρn = convolution_density_matrix(An, Bn, FERMI)
        @test An == A₀ && Bn == B₀
        ρc = convolution_density_matrix(1, Gc, Bc, Integrator(5), β, h)
        @test abs(real(ρc[1]) - real(ρn[1])) < 1e-3
        @test_throws DimensionMismatch convolution_density_matrix(An, HermMatrixNESS(GridInfo(Nt, h), 2), FERMI)
        @test_throws DimensionMismatch convolution_density_matrix(An, HermMatrixNESS(GridInfo(2Nt, h), 1), FERMI)
    end

    @testset "Bubble1/2 vs the contour bubbles (Catch)" begin
        Nft, ntau, β, h = 32, 20, 10.0, 0.1
        nt = Nft ÷ 2 - 1
        CA = nu_contour(nt + 1, ntau + 1, 2, h, β)
        CB = nu_contour(nt + 1, ntau + 1, 1, h, β)
        A = ℱ(CA, FERMI)
        B = ℱ(CB, FERMI)
        for t = 0:nt, t₁ = 0:t
            Mr = C64[0.1t + 0.01t₁^2 + 0.01i - 0.02im * j for i = 0:1, j = 0:1]
            Ml = C64[0.05t₁ + 0.02i + 0.01t^2 + im * (0.03j + 0.005t * t₁) for i = 0:1, j = 0:1]
            br = fill(complex(0.2t + 0.05t₁, -0.1t₁ + 0.03), 1, 1)
            bl = fill(complex(0.1t₁ + 0.03t, 0.1t + 0.02), 1, 1)
            if t == t₁
                Ml = Ml - Ml'
                Mr = Mr - Mr'
                br = br - br'
                bl = bl - bl'
            end
            A.ret[t+1, t₁+1] = Mr
            A.less[t₁+1, t+1] = Ml
            B.ret[t+1, t₁+1] = br
            B.less[t₁+1, t+1] = bl
        end
        An = HermMatrixNESS(GridInfo(Nft, h), 2)
        Bn = HermMatrixNESS(GridInfo(Nft, h), 1)
        cntr2ness!(An, A, nt + 1)
        cntr2ness!(Bn, B, nt + 1)
        for (bc, bn) in ((bubble1!, bubble1_ness!), (bubble2!, bubble2_ness!))
            # C_ij = A_ij * B_11 and C_ij = B_11 * A_ij
            for order in 1:2
                # Fermion * fermion bubble is bosonic
                C = ℱ(CA, BOSE)
                Cn₂ = HermMatrixNESS(GridInfo(Nft, h), 2)
                for i = 1:2, j = 1:2
                    if order == 1
                        bc(nt + 1, C, i, j, A, i, j, B, 1, 1)
                        bn(Cn₂, i, j, An, i, j, Bn, 1, 1)
                    else
                        bc(nt + 1, C, i, j, B, 1, 1, A, i, j)
                        bn(Cn₂, i, j, Bn, 1, 1, An, i, j)
                    end
                end
                Cn₁ = HermMatrixNESS(GridInfo(Nft, h), 2)
                cntr2ness!(Cn₁, C, nt + 1)
                @test distance(Cn₁, Cn₂, FFT_TIME) < 1e-10
            end
        end
        # Scalar version resets the output
        Cs = HermMatrixNESS(GridInfo(Nft, h), 1)
        Cs.ret.time .= 9.0
        Cs.les.time .= 9.0
        bubble1_ness!(Cs, Bn, Bn)
        @test all(iszero, Cs.ret.time[:, :, Nft÷2+1:end])
        @test Cs.les.time[1, 1, Nft÷2+1] == 0
        @test_throws BoundsError bubble2_ness!(Cs, 1, 2, Bn, 1, 1, Bn, 1, 1)
        @test_throws DimensionMismatch bubble2_ness!(Cs, Bn, HermMatrixNESS(GridInfo(2Nft, h), 1))
    end

    @testset "NESS <-> contour round trip (Catch)" begin
        Nft, sz, ntau = 32, 2, 20
        nt = Nft ÷ 2 - 1
        h = 0.1
        Gn = HermMatrixNESS(GridInfo(Nft, h), sz)
        for t = 0:nt
            Mr = C64[0.01t + 0.001i + im * (-0.002j + 0.1t) for i = 0:sz-1, j = 0:sz-1]
            Ml = C64[0.002t + 0.001i + im * (0.001j + 0.1t^2) for i = 0:sz-1, j = 0:sz-1]
            if t == 0
                Mr = Mr - (Mr - Mr')
                Ml = Ml - Ml'
            end
            set_ret!(Gn, t + 1, Mr, FFT_TIME)
            set_les!(Gn, t + 1, Ml, FFT_TIME)
            t > 0 && set_les!(Gn, Nft - t + 1, -Ml', FFT_TIME)
        end
        C = nu_contour(nt + 1, ntau + 1, sz, h, 1.0)
        G = ℱ(C, FERMI)
        # Sentinels in the Matsubara and mixing components
        G.mat[3] = fill(C64(7.0), sz, sz)
        G.lmix[2, 4] = fill(C64(-5.0), sz, sz)
        ness2cntr!(G, Gn)
        @test G.mat[3] == fill(C64(7.0), sz, sz)
        @test G.lmix[2, 4] == fill(C64(-5.0), sz, sz)
        Gb = HermMatrixNESS(GridInfo(Nft, h), sz)
        cntr2ness!(Gb, G, nt + 1)
        @test distance(Gn, Gb, FFT_TIME) < 1e-12
        # Translation invariance of the contour function
        @test G.ret[nt+1, 3] == G.ret[nt-1, 1]
        # Beyond the half grid the values are zero
        C₂ = nu_contour(Nft, ntau + 1, sz, h, 1.0)
        G₂ = ℱ(C₂, FERMI)
        G₂.ret[Nft, 1] = ones(C64, sz, sz)
        ness2cntr!(G₂, Gn)
        @test iszero(G₂.ret[Nft, 1]) && iszero(G₂.less[1, Nft])
        @test G₂.ret[Nft, Nft÷2+1] == Gn.ret.time[:, :, Nft÷2]
        # tstp smaller than the half grid: the tail is zero
        Gt = HermMatrixNESS(GridInfo(Nft, h), sz)
        cntr2ness!(Gt, G, 4)
        @test all(iszero, Gt.ret.time[:, :, 5:end])
        @test Gt.ret.time[:, :, 4] == G.ret[4, 1]
        @test_throws ArgumentError cntr2ness!(Gt, G, 0)
        @test_throws ArgumentError cntr2ness!(Gt, G, nt + 2)
        @test_throws DimensionMismatch cntr2ness!(HermMatrixNESS(GridInfo(Nft, h), 1), G, 2)
        @test_throws DimensionMismatch ness2cntr!(G, HermMatrixNESS(GridInfo(Nft, h), 1))
    end

    @testset "resampling (Catch)" begin
        h₁, h₂, N₁, N₂ = 0.01, 0.1, 1000, 100
        β, μ = 10.0, 0.0
        dos = GaussianDOS(0.0, 2.0)
        G₁ = HermMatrixNESS(GridInfo(N₁, h₁), 1)
        G₂ = HermMatrixNESS(GridInfo(N₂, h₂), 1)
        from_dos_ness!(G₁, FERMI, dos, β, μ)
        from_dos_ness!(G₂, FERMI, dos, β, μ)
        G₁₀ = deepcopy(G₁)
        Gdo = downsample(G₁, 10)
        @test Gdo.grid.h ≈ h₂ && Gdo.grid.dw ≈ G₁.grid.dw
        @test distance(Gdo, G₂, FFT_TIME) / N₂ < 1e-6
        Gup = upsample(G₂, 10)
        @test Gup.grid.h ≈ h₁ && getnft(Gup) == N₁
        @test distance(Gup, G₁, FFT_TIME) / N₁ < 1e-6
        @test distance(resample(G₁, 10), G₂, FFT_TIME) / N₂ < 1e-6
        @test distance(resample(G₂, -10), G₁, FFT_TIME) / N₁ < 1e-6
        @test G₁ == G₁₀
        @test_throws ArgumentError resample(G₁, 0)
        @test_throws ArgumentError downsample(G₁, 3)
        @test_throws ArgumentError upsample(G₁, 0)
        # FFTArray: every factor-th time point
        A = FFTArray(8, 1)
        A.time[1, 1, :] .= 1:8
        A.freq[1, 1, :] .= 11:18
        B = downsample(A, 2)
        @test B.time[1, 1, :] == [1, 3, 5, 7]
        @test B.freq[1, 1, :] == [11, 12, 17, 18]
    end
end
