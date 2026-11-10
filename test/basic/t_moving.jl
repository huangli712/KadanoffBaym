#
# t_moving.jl
#
# To test the moving (truncated window) data layer: type tree, explicit
# initialization, absolute/relative indexing, forward!, the backfill from a
# full contour function, the NESS initialization, views, the moving
# single-time function, plain text I/O, and the storage gate.
#
# NESSi: libcntr/test/moving_data_exchange.cpp and
# moving_green_textio.cpp (the HDF5 test is not ported).
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const KB = KadanoffBaym

# Full Green's function with a unique code in every independent block
function mv_full(nt::Int, d::Int, sign::Int = FERMI)
    C = Cn(nt, 5, d, d, (nt - 1) * 0.1, 1.0)
    F = ℱ(C, sign)
    for n = 1:nt, m = 1:n
        F.ret[n, m] = [complex(100n + m + 0.1a, 0.01b) for a = 1:d, b = 1:d]
        L = [complex(0.3a - 0.2b, 10n + m) for a = 1:d, b = 1:d]
        # G^<(t,t) is anti-hermitian
        F.less[m, n] = m == n ? (L - L') / 2 : L
    end
    return F
end

# Construction measured in a compiled function (no global-scope overhead)
mv_timed(tc::Int, d::Int) = @timed ℱᵐᵒᵛ(tc, d)

@testset verbose = true "KadanoffBaym: moving data layer" begin

    @testset "type tree and construction" begin
        G = ℱᵐᵒᵛ(4, 2)
        @test G isa KB.CnAbstractMovingFunction && G isa KB.CnAbstractFunction
        @test G isa KB.CnAbstractType
        @test G.ret isa KB.CnAbstractMovingMatrix && G.ret isa KB.CnAbstractMatrix
        @test G.less isa KB.CnAbstractMovingMatrix
        x = 𝒻ᵐᵒᵛ(4, 2, 7)
        @test x isa KB.CnAbstractMovingFunction
        @test x.ret isa KB.CnAbstractMovingVector && x.ret isa KB.CnAbstractVector
        f = Cfᵐᵒᵛ(4, 2)
        @test f isa KB.CnAbstractMovingFunction
        @test size(G.ret.data) == (2, 2, 5, 5) && size(G.less.data) == (2, 2, 5, 5)
        @test size(f.data) == (2, 2, 5)
        @test !G.initialized && G.tstp == 0 && !f.initialized
        @test getsize(G) == 4 && getdims(G) == (2, 2) && getsign(G) == FERMI
        @test getsign(ℱᵐᵒᵛ(4, 1, BOSE)) == BOSE
        @test_throws ArgumentError ℱᵐᵒᵛ(-1, 2)
        @test_throws ArgumentError ℱᵐᵒᵛ(3, 0)
        @test_throws ArgumentError ℱᵐᵒᵛ(3, 2, 0)
        @test_throws OverflowError ℱᵐᵒᵛ(typemax(Int) ÷ 2, 2)
    end

    @testset "uninitialized objects are rejected" begin
        G = ℱᵐᵒᵛ(3, 2)
        @test_throws ArgumentError gettstp(G)
        @test_throws ArgumentError G[:ret, 1, 1]
        @test_throws ArgumentError timestep_view(G)
        @test_throws ArgumentError get_timestep(G)
        @test_throws ArgumentError forward!(G)
        f = Cfᵐᵒᵛ(3, 2)
        @test_throws ArgumentError gettstp(f)
        @test_throws ArgumentError forward!(f)
    end

    @testset "moving data exchange (scalar and matrix)" for d in (1, 2)
        nt, tc, n₀ = 12, 4, 8
        F = mv_full(nt, d)
        G = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G, F, n₀)
        @test G.initialized && gettstp(G) == n₀
        # Every block of the window
        for n = n₀-tc:n₀, m = max(1, n - tc):n
            @test G[:ret, n, m] == F.ret[n, m]
            @test maximum(abs.(G[:less, n, m] - F.less[n, m])) < 1e-14
            # The other time ordering is reconstructed
            @test G[:ret, m, n] == F.ret[m, n]
            @test G[:less, m, n] == F.less[m, n]
            @test G[:gtr, n, m] ≈ F.ret[n, m] + F.less[n, m]
        end
        # Relative indices agree with NESSi (t₀ - delti, t₀ - delti - deltj)
        @test G.ret[2, 1] == F.ret[n₀ - 2, n₀ - 3]
        # Outside the window
        @test_throws BoundsError G[:ret, n₀ + 1, n₀]
        @test_throws BoundsError G[:ret, n₀ - tc - 1, n₀ - tc - 1]
        @test_throws BoundsError G[:ret, n₀, n₀ - tc - 1]
        # Rows with n - 1 < tc are padded with zeros
        G2 = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G2, F, tc + 1)
        @test G2.ret[tc, 1] == zeros(d, d)
        @test G2.ret[tc, 0] == F.ret[1, 1]
        @test_throws ArgumentError set_from_G_backward!(ℱᵐᵒᵛ(tc, d), F, tc)
        @test_throws ArgumentError set_from_G_backward!(ℱᵐᵒᵛ(tc, d), F, nt + 1)
        @test_throws ArgumentError set_from_G_backward!(ℱᵐᵒᵛ(tc, d, BOSE), F, n₀)
        @test_throws DimensionMismatch set_from_G_backward!(ℱᵐᵒᵛ(tc, d + 1), F, n₀)
        # Non-hermitian Gcc: the lesser is -Gcc^<(m,n)^†
        Fcc = mv_full(nt, d)
        for n = 1:nt, m = 1:n
            Fcc.less[m, n] = 2 * Matrix(F.less[m, n])
        end
        G3 = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G3, F, Fcc, n₀)
        @test G3.less[1, 2] ≈ -2 * Matrix(F.less[n₀ - 3, n₀ - 1]')
    end

    @testset "forward! is O(1) and keeps the history" begin
        nt, tc, n₀, d = 12, 4, 8, 2
        F = mv_full(nt, d)
        G = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G, F, n₀)
        id_ret = objectid(G.ret.data)
        id_less = objectid(G.less.data)
        for step = 1:2
            forward!(G)
            @test gettstp(G) == n₀ + step
            @test objectid(G.ret.data) == id_ret && objectid(G.less.data) == id_less
            # The new leading row is zero
            @test all(iszero, G.ret[0, j] for j = 0:tc)
            @test all(iszero, G.less[0, j] for j = 0:tc)
            # The history at absolute times is unchanged
            for n = n₀+step-tc:n₀, m = max(1, n - tc):n
                n - m ≤ tc || continue
                (n₀ + step) - n ≤ tc || continue
                @test maximum(abs.(G[:ret, n, m] - F.ret[n, m])) < 1e-14
                @test maximum(abs.(G[:less, n, m] - F.less[n, m])) < 1e-14
            end
        end
        # Ring wrap-around: tc + 1 steps return the head to its start
        h = G.ret.head
        for _ = 1:tc+1
            forward!(G)
        end
        @test G.ret.head == h && G.less.head == h
    end

    @testset "copies and views" begin
        tc, d = 3, 2
        F = mv_full(8, d)
        G = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G, F, 6)
        # Public getindex returns a copy
        X = G[:ret, 6, 5]
        X .= 0
        @test G[:ret, 6, 5] == F.ret[6, 5]
        Y = G.ret[0, 1]
        Y .= 0
        @test G.ret[0, 1] == F.ret[6, 5]
        # A view writes through to the parent
        v = timestep_view(G, 1)
        @test gettstp(v) == 5
        @test v[:ret, 0] == F.ret[5, 5]
        v[:ret, 2] = fill(1.0 + 2im, d, d)
        @test G[:ret, 5, 3] == fill(1.0 + 2im, d, d)
        @test G.ret[1, 2] == fill(1.0 + 2im, d, d)
        # Owning time step is decoupled
        x = get_timestep(G, 0)
        @test gettstp(x) == 6 && x[:ret, 1] == F.ret[6, 5]
        x[:ret, 1] = zeros(C64, d, d)
        @test G[:ret, 6, 5] == F.ret[6, 5]
        @test_throws BoundsError v[:ret, tc + 1]
        @test_throws ArgumentError v[:mat, 0]
        @test_throws DimensionMismatch (v[:less, 0] = zeros(C64, d + 1, d))
        # Stale views: forward!
        before = G.ret[0, 0]
        forward!(G)
        @test_throws ArgumentError v[:ret, 0]
        @test_throws ArgumentError (v[:ret, 0] = ones(C64, d, d))
        @test_throws ArgumentError gettstp(v)
        @test G.ret[1, 0] == before
        # Stale views: set_from_G_backward!
        v = timestep_view(G, 0)
        set_from_G_backward!(G, F, 6)
        @test_throws ArgumentError (v[:less, 0] = ones(C64, d, d))
        @test G[:less, 6, 6] ≈ F.less[6, 6]
    end

    @testset "NESS initialization: relative step 0 is KB step 1" begin
        tc, d, Nft, h = 4, 2, 8, 0.1
        N = HermMatrixNESS(GridInfo(Nft, h, 2π / (Nft * h)), d)
        for n = 1:Nft
            N.ret.time[:, :, n] .= [complex(n, a - b) for a = 1:d, b = 1:d]
            N.les.time[:, :, n] .= [complex(a + b, -n) for a = 1:d, b = 1:d]
        end
        G = ℱᵐᵒᵛ(tc, d)
        v = 0
        set_from_G_backward!(G, mv_full(8, d), 6)
        v = timestep_view(G, 0)
        initialize_from_ness!(G, N)
        @test_throws ArgumentError v[:ret, 0]
        @test G.initialized && gettstp(G) == 1
        dtmax = min(tc, Nft ÷ 2 - 1)
        for delti = 0:tc, dt = 0:tc
            if dt ≤ dtmax
                @test G.ret[delti, dt] == N.ret.time[:, :, dt+1]
                @test G.less[delti, dt] == N.les.time[:, :, dt+1]
            else
                @test iszero(G.ret[delti, dt]) && iszero(G.less[delti, dt])
            end
        end
        # Absolute access below KB step 1 is not allowed
        @test G[:ret, 1, 1] == N.ret.time[:, :, 1]
        @test_throws BoundsError G[:ret, 1, 0]
        forward!(G)
        @test gettstp(G) == 2
        @test G[:ret, 1, 1] == N.ret.time[:, :, 1]
        @test G.ret[1, 1] == N.ret.time[:, :, 2]
        @test_throws DimensionMismatch initialize_from_ness!(ℱᵐᵒᵛ(tc, d + 1), N)
    end

    @testset "moving single-time function" begin
        tc, d, nt = 3, 2, 9
        cf = Cf(nt, d, d)
        for i = 0:nt
            cf[i] = [complex(F64(i), a + 10.0b) for a = 1:d, b = 1:d]
        end
        f = Cfᵐᵒᵛ(tc, d)
        set_from_function_backward!(f, cf, 7)
        @test f.initialized && gettstp(f) == 7
        @test all(f[delti] == cf[7 - delti] for delti = 0:tc)
        id = objectid(f.data)
        forward!(f)
        @test gettstp(f) == 8 && objectid(f.data) == id
        @test iszero(f[0])
        @test all(f[delti] == cf[8 - delti] for delti = 1:tc)
        f[0] = cf[8]
        @test f[0] == cf[8]
        @test_throws BoundsError f[tc + 1]
        @test_throws ArgumentError set_from_function_backward!(Cfᵐᵒᵛ(tc, d), cf, tc)
        @test_throws DimensionMismatch set_from_function_backward!(Cfᵐᵒᵛ(tc, d + 1), cf, 7)
    end

    @testset "moving text io" begin
        tc, d = 3, 2
        F = mv_full(9, d)
        G = ℱᵐᵒᵛ(tc, d)
        set_from_G_backward!(G, F, 7)
        forward!(G)
        G.ret[0, 1] = fill(0.25 - 1.5im, d, d)
        mktempdir() do dir
            fn = joinpath(dir, "g.dat")
            write(fn, G)
            H = ℱᵐᵒᵛ(tc, d)
            v = nothing
            read!(fn, H)
            @test H.initialized && gettstp(H) == gettstp(G)
            @test all(H.ret[i, j] == G.ret[i, j] && H.less[i, j] == G.less[i, j]
                      for i = 0:tc, j = 0:tc)
            # Stale view after a successful read!
            v = timestep_view(H, 0)
            read!(fn, H)
            @test_throws ArgumentError (v[:ret, 0] = zeros(C64, d, d))
            @test H.ret[0, 1] == G.ret[0, 1]
            # Bad inputs: the target is unchanged
            snap = (copy(H.ret.data), copy(H.less.data), H.tstp, H.generation)
            lines = readlines(fn)
            function bad(name, ls, target = H)
                p = joinpath(dir, name)
                write(p, join(ls, "\n") * "\n")
                @test_throws ArgumentError read!(p, target)
            end
            bad("trunc.dat", lines[1:end-3])
            bad("version.dat", [replace(lines[1], "KB_MOVING_V1" => "KB_MOVING_V2"); lines[2:end]])
            bad("tc.dat", [replace(lines[1], " F $tc " => " F $(tc + 1) "); lines[2:end]])
            bad("index.dat", [lines[1]; replace(lines[2], r"^0 0" => "0 7"); lines[3:end]])
            bad("nan.dat", [lines[1]; replace(lines[2], r" \S+ \S+$" => " NaN 0.0"); lines[3:end]])
            bad("state.dat", [replace(lines[1], r" 1 (\d+) " => s" 0 \1 "); lines[2:end]])
            bad("kind.dat", [replace(lines[1], " F " => " Cf "); lines[2:end]])
            @test (H.ret.data, H.less.data, H.tstp, H.generation) == snap
            # An uninitialized object round trip
            fu = joinpath(dir, "u.dat")
            write(fu, ℱᵐᵒᵛ(tc, d))
            U = ℱᵐᵒᵛ(tc, d)
            read!(fu, U)
            @test !U.initialized
            # Single-time function
            cf = Cf(9, d, d)
            for i = 0:9
                cf[i] = fill(complex(F64(i), -F64(i)), d, d)
            end
            f = Cfᵐᵒᵛ(tc, d)
            set_from_function_backward!(f, cf, 6)
            forward!(f)
            ff = joinpath(dir, "f.dat")
            write(ff, f)
            f2 = Cfᵐᵒᵛ(tc, d)
            read!(ff, f2)
            @test gettstp(f2) == 7 && all(f2[i] == f[i] for i = 0:tc)
            bad("ftrunc.dat", readlines(ff)[1:end-1], f2)
        end
    end

    @testset "full contour alias contract is unchanged" begin
        F = mv_full(5, 2)
        X = F.ret[4, 2]
        X[1, 1] = 99.0
        @test F.ret[4, 2][1, 1] == 99.0
        Y = F.ret[2, 4]
        Y[1, 1] = -77.0
        @test F.ret[2, 4][1, 1] != -77.0
        Z = F.less[2, 4]
        Z[1, 1] = 5.0
        @test F.less[2, 4][1, 1] == 5.0
        W = F.less[4, 2]
        W[1, 1] = 123.0
        @test F.less[4, 2][1, 1] != 123.0
    end

    @testset "storage gate" begin
        d = 2
        # Warm up the constructor
        ℱᵐᵒᵛ(3, d)
        counts = Int[]
        for tc in (100, 500, 1000)
            payload = 2 * (tc + 1)^2 * d^2 * sizeof(C64)
            # The first call of each size is a warm-up
            mv_timed(tc, d)
            GC.gc()
            st = mv_timed(tc, d)
            G = st.value
            # One backing array per component, no per-block matrices
            @test fieldtype(typeof(G.ret), :data) == Array{C64,4}
            @test fieldtype(typeof(G.less), :data) == Array{C64,4}
            @test !any(T -> T <: AbstractMatrix, fieldtypes(typeof(G.ret)))
            @test !any(T -> T <: AbstractMatrix, fieldtypes(typeof(G)))
            sz = Base.summarysize((G.ret, G.less))
            @test payload ≤ sz ≤ payload + 65536
            @test payload ≤ st.bytes ≤ payload + 65536
            push!(counts, Base.gc_alloc_count(st.gcstats))
            G = nothing
            GC.gc()
        end
        @test maximum(counts) - minimum(counts) ≤ 16
        # forward! does not allocate in proportion to the window
        F = mv_full(12, d)
        for tc in (4, 10)
            G = ℱᵐᵒᵛ(tc, d)
            set_from_G_backward!(G, F, 11)
            forward!(G)
            @test (@allocated forward!(G)) ≤ 4096
        end
    end
end
