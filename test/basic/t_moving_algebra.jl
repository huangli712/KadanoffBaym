#
# t_moving_algebra.jl
#
# To test the traits of the moving time steps (copy, incr, smul,
# multiplication by a moving single-time function), the moving
# extrapolation and the moving distance.
#
# NESSi: libcntr/test/moving_green_simple_algebra.cpp (matrixelement, incr,
# smul, four function multiplication sections), plus alias, stale view,
# and ring wrap-around regressions.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using Random
using KadanoffBaym

const KB = KadanoffBaym

# Moving Green's function filled with fixed random data at the leading
# time step tstp
function ma_random(tc, d, tstp; seed = 1)
    rng = MersenneTwister(seed)
    nt = tstp + 2
    C = Cn(nt, 3, d, d, (nt - 1) * 0.1, 1.0)
    F = ℱ(C, FERMI)
    for n = 1:nt, m = 1:n
        F.ret[n, m] = randn(rng, C64, d, d)
        L = randn(rng, C64, d, d)
        F.less[m, n] = m == n ? (L - L') / 2 : L
    end
    G = ℱᵐᵒᵛ(tc, d)
    set_from_G_backward!(G, F, tstp)
    return G
end

function ma_func(tc, d, tstp; seed = 7)
    rng = MersenneTwister(seed)
    cf = Cf(tstp + 2, d, d)
    for i = 0:tstp+2
        cf[i] = randn(rng, C64, d, d)
    end
    f = Cfᵐᵒᵛ(tc, d)
    set_from_function_backward!(f, cf, tstp)
    return f
end

rowcopy(G, c, delti) = [ (c == :ret ? G.ret : G.less)[delti, j] for j = 0:G.tc ]

@testset verbose = true "KadanoffBaym: moving algebra" begin

    @testset "matrix elements ($d x $d)" for d in (1, 3)
        tc, tstp = 5, 9
        G = ma_random(tc, d, tstp)
        v = timestep_view(G, 2)
        x = get_timestep(G, 2)
        for j = 0:tc
            @test v[:ret, j] == G.ret[2, j] == x[:ret, j]
            @test v[:less, j] == G.less[2, j] == x[:less, j]
        end
        # Single matrix elements written through a view
        B = v[:ret, 3]
        B[1, d] = 4.0 - 2.0im
        v[:ret, 3] = B
        @test G.ret[2, 3][1, d] == 4.0 - 2.0im
        @test G[:ret, tstp - 2, tstp - 5][1, d] == 4.0 - 2.0im
    end

    @testset "incr! and smul!" for d in (1, 2)
        tc, tstp = 4, 8
        G₁ = ma_random(tc, d, tstp; seed = 2)
        G₂ = ma_random(tc, d, tstp; seed = 3)
        α = 0.7 - 1.3im
        r1 = rowcopy(G₁, :ret, 1); l1 = rowcopy(G₁, :less, 1)
        r2 = rowcopy(G₂, :ret, 0); l2 = rowcopy(G₂, :less, 0)
        incr!(timestep_view(G₁, 1), timestep_view(G₂, 0), α)
        @test maximum(maximum(abs.(G₁.ret[1, j] - (r1[j+1] + α * r2[j+1]))) for j = 0:tc) < 1e-14
        @test maximum(maximum(abs.(G₁.less[1, j] - (l1[j+1] + α * l2[j+1]))) for j = 0:tc) < 1e-14
        # Other rows are not touched
        @test rowcopy(G₁, :ret, 0) == rowcopy(ma_random(tc, d, tstp; seed = 2), :ret, 0)
        # Owning time step
        x = get_timestep(G₂, 0)
        incr!(x, timestep_view(G₂, 0), 1.0)
        @test all(x[:ret, j] == 2 .* r2[j+1] for j = 0:tc)
        # smul!
        smul!(timestep_view(G₂, 0), α)
        @test maximum(maximum(abs.(G₂.ret[0, j] - α * r2[j+1])) for j = 0:tc) < 1e-14
        @test maximum(maximum(abs.(G₂.less[0, j] - α * l2[j+1])) for j = 0:tc) < 1e-14
        # Alias: dst === src
        G₃ = ma_random(tc, d, tstp; seed = 4)
        r3 = rowcopy(G₃, :ret, 0)
        v = timestep_view(G₃, 0)
        incr!(v, v, 2.0)
        @test all(G₃.ret[0, j] == 3 .* r3[j+1] for j = 0:tc)
        # memcpy! between rows and zeros!
        G₄ = ma_random(tc, d, tstp; seed = 5)
        memcpy!(timestep_view(G₄, 3), timestep_view(G₄, 0))
        @test rowcopy(G₄, :ret, 0) == rowcopy(G₄, :ret, 3)
        @test rowcopy(G₄, :less, 0) == rowcopy(G₄, :less, 3)
        zeros!(timestep_view(G₄, 1))
        @test all(iszero, rowcopy(G₄, :ret, 1)) && all(iszero, rowcopy(G₄, :less, 1))
        # Whole object copy
        G₅ = ℱᵐᵒᵛ(tc, d)
        memcpy!(G₄, G₅)
        @test gettstp(G₅) == tstp
        @test all(G₅[:ret, n, m] == G₄[:ret, n, m] for n = tstp-tc:tstp for m = max(1, n - tc):n)
        # Incompatible rows
        @test_throws ArgumentError incr!(timestep_view(G₁, 0), timestep_view(ma_random(tc + 1, d, tstp), 0), 1.0)
    end

    @testset "function multiplication ($(adj ? "adjoint" : "plain"), d = $d)" for adj in (false, true), d in (1, 2)
        tc, tstp = 4, 8
        G = ma_random(tc, d, tstp; seed = 11)
        f = ma_func(tc, d, tstp)
        r = rowcopy(G, :ret, 0); l = rowcopy(G, :less, 0)
        op(X) = adj ? Matrix(X') : X
        # Left: f(t) G(t, t - j)
        v = timestep_view(G, 0)
        adj ? smul!(f, v, Val(:adjoint)) : smul!(f, v)
        F0 = op(f[0])
        @test maximum(maximum(abs.(G.ret[0, j] - F0 * r[j+1])) for j = 0:tc) < 1e-13
        @test maximum(maximum(abs.(G.less[0, j] - F0 * l[j+1])) for j = 0:tc) < 1e-13
        # Right: G(t, t - j) f(t - j)
        G = ma_random(tc, d, tstp; seed = 11)
        v = timestep_view(G, 0)
        adj ? smul!(v, f, Val(:adjoint)) : smul!(v, f)
        @test maximum(maximum(abs.(G.ret[0, j] - r[j+1] * op(f[j]))) for j = 0:tc) < 1e-13
        @test maximum(maximum(abs.(G.less[0, j] - l[j+1] * op(f[j]))) for j = 0:tc) < 1e-13
        # The time step must be the leading step of f
        @test_throws ArgumentError smul!(f, timestep_view(G, 1))
        @test_throws ArgumentError smul!(Cfᵐᵒᵛ(tc, d), timestep_view(G, 0))
    end

    @testset "stale views and ring wrap-around" begin
        tc, d, tstp = 3, 2, 6
        G = ma_random(tc, d, tstp; seed = 21)
        v = timestep_view(G, 0)
        forward!(G)
        @test_throws ArgumentError incr!(v, timestep_view(G, 1), 1.0)
        @test_throws ArgumentError smul!(v, 2.0)
        @test_throws ArgumentError zeros!(v)
        @test_throws ArgumentError distance(v, timestep_view(G, 1))
        # Fill the leading row after every forward! and wrap the ring twice
        F = ℱᵐᵒᵛ(tc, d)
        memcpy!(G, F)
        vals = Dict{Int,Matrix{C64}}()
        for s = 1:2(tc + 1)
            forward!(F)
            n = gettstp(F)
            vals[n] = fill(complex(n, -n), d, d)
            w = timestep_view(F, 0)
            for j = 0:tc
                w[:ret, j] = vals[n]
            end
        end
        n₀ = gettstp(F)
        @test all(F[:ret, n, n - 1] == vals[n] for n = n₀-tc:n₀)
    end

    @testset "extrapolation (orders 0..5)" begin
        d, h = 2, 0.1
        for k = 0:5
            tc = 2k + 3
            n₀ = tc + 2
            nt = n₀ + 1
            # Polynomial of total order k in t and t' (a constant for k = 0,
            # since order 0 copies the previous row)
            P(t, s) = k == 0 ? [complex(F64(a), F64(b)) for a = 1:d, b = 1:d] :
                [complex((1 + 0.3a) * t^k - 0.5b * s * t^(k - 1) + a,
                         0.2 * s^k + b * t^k) for a = 1:d, b = 1:d]
            C = Cn(nt, 3, d, d, (nt - 1) * h, 1.0)
            F = ℱ(C, FERMI)
            for n = 1:nt, m = 1:n
                t = (n - 1) * h
                s = (m - 1) * h
                F.ret[n, m] = P(t, s)
                L = P(s, t)
                F.less[m, n] = m == n ? (L - L') / 2 : L
            end
            G = ℱᵐᵒᵛ(tc, d)
            set_from_G_backward!(G, F, n₀)
            forward!(G)
            extrapolate_timestep!(G, k)
            n = n₀ + 1
            e = maximum(maximum(abs.(G[:ret, n, n - j] - F.ret[n, n - j])) for j = 0:tc)
            @test e < 1e-12
            # The history is unchanged
            @test G[:ret, n₀, n₀ - 2] == F.ret[n₀, n₀ - 2]
        end
        # Insufficient window
        G = ma_random(4, 2, 7)
        @test_throws ArgumentError extrapolate_timestep!(G, 2)
    end

    @testset "distance: sum of block Frobenius norms" begin
        tc, d, tstp = 2, 2, 5
        G₁ = ma_random(tc, d, tstp; seed = 31)
        G₂ = ℱᵐᵒᵛ(tc, d)
        memcpy!(G₁, G₂)
        @test distance(G₁, G₂, tstp) == 0.0
        # Different block norms in two blocks
        B₁ = [1.0 2.0; 0.0 -2.0] .+ 0im
        B₂ = [0.5im 0.0; 0.0 0.0]
        v = timestep_view(G₂, 0)
        v[:ret, 0] = v[:ret, 0] + B₁
        v[:less, 2] = v[:less, 2] + B₂
        expect = norm(B₁) + norm(B₂)
        @test abs(distance(G₁, G₂, tstp) - expect) < 1e-14
        @test abs(distance(G₂, G₁, tstp) - expect) < 1e-14
        @test abs(distance(timestep_view(G₁, 0), get_timestep(G₂, 0)) - expect) < 1e-14
        @test abs(expect - sqrt(norm(B₁)^2 + norm(B₂)^2)) > 0.1
        @test abs(expect - (norm(B₁)^2 + norm(B₂)^2)) > 0.1
        # Older rows are unchanged
        @test distance(G₁, G₂, tstp - 1) == 0.0
        # The same definition as the full contour distance
        nt = 6
        C = Cn(nt, 3, d, d, 0.5, 1.0)
        A = ℱ(C, FERMI)
        B = ℱ(C, FERMI)
        rng = MersenneTwister(5)
        for n = 1:nt, m = 1:n
            A.ret[n, m] = randn(rng, C64, d, d)
            B.ret[n, m] = randn(rng, C64, d, d)
            A.less[m, n] = randn(rng, C64, d, d)
            B.less[m, n] = randn(rng, C64, d, d)
        end
        GA = ℱᵐᵒᵛ(nt - 1, d)
        GB = ℱᵐᵒᵛ(nt - 1, d)
        set_from_G_backward!(GA, A, nt)
        set_from_G_backward!(GB, B, nt)
        # moving lesser of row 0: -A^<(t-j,t)^†, the norm is unchanged
        @test abs(distance(GA, GB, nt) -
                  (distance(A, B, nt, Val(:ret)) + distance(A, B, nt, Val(:les)))) < 1e-12
    end
end
