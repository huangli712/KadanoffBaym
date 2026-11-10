#
# t_vie2_moving.jl
#
# To test the truncated (moving) VIE2 G + F * G = Q. NESSi has no moving
# VIE2; the truncated formulas are those of the full kernels without the
# mixed initial correlations (see src/vie.jl, Remarks: Truncated VIE2).
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using Random
using KadanoffBaym

# Full function with random real-time data, exponentially decaying in
# t - t'. The Matsubara and mixing components are zero.
function vm_full(C, rng, scale; les = true)
    d = C.ndim1
    X = ℱ(C, FERMI)
    for n = 1:C.ntime, m = 1:n
        τ = (n - m) * 0.05
        X.ret[n, m] = scale * exp(-τ) .* randn(rng, C64, d, d)
        L = les ? scale * exp(-τ) .* randn(rng, C64, d, d) : zeros(C64, d, d)
        X.less[m, n] = m == n ? (L - L') / 2 : L
    end
    return X
end

# Kernel F and its hermitian conjugate partner (shared equal-time lesser)
function vm_kernel(C, rng)
    F = vm_full(C, rng, 0.3)
    Fcc = vm_full(C, rng, 0.3)
    for m = 1:C.ntime
        Fcc.less[m, m] = -Matrix(F.less[m, m]')
    end
    return F, Fcc
end

function vm_moving(X, Xcc, tc, n₀)
    M = ℱᵐᵒᵛ(tc, getdims(X)[1], getsign(X))
    init_from_G!(M, X, Xcc, n₀)
    return M
end

function vm_rowerr(Gm, G, n)
    er = 0.0
    el = 0.0
    for j = 0:Gm.tc
        n - j ≥ 1 || continue
        er = max(er, maximum(abs.(Gm[:ret, n, n - j] - G.ret[n, n - j])))
        el = max(el, maximum(abs.(Gm[:less, n - j, n] - G.less[n - j, n])))
    end
    return er, el
end

@testset verbose = true "KadanoffBaym: truncated VIE2" begin
    h, β = 0.05, 1.0

    @testset "F = 0 gives G = Q (d = $d)" for d in (1, 2)
        nt, tc, k = 20, 9, 3
        rng = MersenneTwister(1)
        C = Cn(nt, 5, d, d, (nt - 1) * h, β)
        Q = vm_full(C, rng, 1.0)
        Z = ℱ(C, FERMI)
        Gm = vm_moving(Z, Z, tc, nt)
        Fm = vm_moving(Z, Z, tc, nt)
        Qm = vm_moving(Q, Q, tc, nt)
        vie2_timestep!(nt, Gm, Fm, Fm, Qm, Integrator(k), h)
        e = maximum(maximum(abs.(Gm.ret[0, j] - Qm.ret[0, j])) for j = 0:tc)
        e = max(e, maximum(maximum(abs.(Gm.less[0, j] - Qm.less[0, j])) for j = 0:tc))
        @test e < 1e-13
    end

    @testset "zero mixing components, window covering the history (k = $k, d = $d)" for k = 1:5, d in (1, 2)
        nt = 2k + 6
        rng = MersenneTwister(10k + d)
        # The Matsubara solver of order k needs enough imaginary time points
        C = Cn(nt, 13, d, d, (nt - 1) * h, β)
        F, Fcc = vm_kernel(C, rng)
        Q = vm_full(C, rng, 1.0)
        G = ℱ(C, FERMI)
        I = Integrator(k)
        vie2!(G, F, Fcc, Q, I, β, h)
        # The full mixing components vanish for this input
        @test maximum(abs, G.lmix.data) == 0.0
        for n₀ in (2k + 3, nt)
            tc = n₀ - 1
            Gm = vm_moving(G, G, tc, n₀)
            vie2_timestep!(n₀, Gm, vm_moving(F, Fcc, tc, n₀), vm_moving(Fcc, F, tc, n₀),
                           vm_moving(Q, Q, tc, n₀), I, h)
            er, el = vm_rowerr(Gm, G, n₀)
            @test er < 1e-10
            @test el < 1e-10
        end
    end

    @testset "short window: finite, history unchanged" begin
        d, nt, tc, k = 2, 30, 9, 3
        rng = MersenneTwister(5)
        C = Cn(nt, 5, d, d, (nt - 1) * h, β)
        F, Fcc = vm_kernel(C, rng)
        Q = vm_full(C, rng, 1.0)
        G = ℱ(C, FERMI)
        vie2!(G, F, Fcc, Q, Integrator(k), β, h)
        Gm = vm_moving(G, G, tc, nt)
        before = Gm.ret[2, 4]
        vie2_timestep!(nt, Gm, vm_moving(F, Fcc, tc, nt), vm_moving(Fcc, F, tc, nt),
                       vm_moving(Q, Q, tc, nt), Integrator(k), h)
        @test all(j -> all(isfinite, Gm.ret[0, j]) && all(isfinite, Gm.less[0, j]), 0:tc)
        @test Gm.ret[2, 4] == before
        # The truncation changes the result
        er, el = vm_rowerr(Gm, G, nt)
        @test max(er, el) > 1e-8
    end

    @testset "nonzero mixing components: no full equivalence" begin
        d, nt, k = 2, 14, 2
        rng = MersenneTwister(7)
        C = Cn(nt, 5, d, d, (nt - 1) * h, β)
        F, Fcc = vm_kernel(C, rng)
        Q = vm_full(C, rng, 1.0)
        # Nonzero mixing components of the kernel and the source
        for X in (F, Fcc, Q)
            X.lmix.data .= 0.2 .* randn(rng, C64, size(X.lmix.data))
        end
        G = ℱ(C, FERMI)
        vie2!(G, F, Fcc, Q, Integrator(k), β, h)
        @test maximum(abs, G.lmix.data) > 0.01
        Gm = vm_moving(G, G, nt - 1, nt)
        vie2_timestep!(nt, Gm, vm_moving(F, Fcc, nt - 1, nt), vm_moving(Fcc, F, nt - 1, nt),
                       vm_moving(Q, Q, nt - 1, nt), Integrator(k), h)
        # The retarded component does not depend on the mixing components,
        # the lesser component misses the tv*vt term
        er, el = vm_rowerr(Gm, G, nt)
        @test er < 1e-10
        @test el > 1e-6
    end

    @testset "failure paths" begin
        d, nt = 2, 20
        rng = MersenneTwister(9)
        C = Cn(nt, 5, d, d, (nt - 1) * h, β)
        F, Fcc = vm_kernel(C, rng)
        Q = vm_full(C, rng, 1.0)
        mk(X, Xcc, tc) = vm_moving(X, Xcc, tc, nt)
        Gm, Fm, Fccm, Qm = mk(Q, Q, 7), mk(F, Fcc, 7), mk(Fcc, F, 7), mk(Q, Q, 7)
        @test_throws ArgumentError vie2_timestep!(nt, Gm, Fm, Fccm, Qm, Integrator(3), h)
        @test_throws ArgumentError vie2_timestep!(nt + 1, Gm, Fm, Fccm, Qm, Integrator(2), h)
        @test_throws ArgumentError vie2_timestep!(nt, Gm, Fm, Fccm, Gm, Integrator(2), h)
        @test_throws ArgumentError vie2_timestep!(nt, Gm, Fm, Fccm, mk(Q, Q, 8), Integrator(2), h)
        @test_throws ArgumentError vie2_timestep!(nt, ℱᵐᵒᵛ(7, d), Fm, Fccm, Qm, Integrator(2), h)
        @test_throws ArgumentError vie2_timestep!(nt, Gm, Fm, Fccm, Qm, Integrator(2), -h)
    end
end
