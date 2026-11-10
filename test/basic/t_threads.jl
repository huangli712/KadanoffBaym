#
# t_threads.jl
#
# To test the threaded paths (`threaded = true`) of the convolution, the
# full Dyson and VIE2 time steps, and the moving (truncated) solvers. The
# threaded results must agree with the serial ones; since every output
# block is accumulated in the same order, they agree bitwise. With
# `Threads.nthreads() == 1` the threaded path falls back to the serial
# code. No wall time is measured here.
#
# The FFT plan cache contract follows NESSi libcntr/test/ness2_omp_manager.cpp:
# the plans are cached per (length, direction), the thread count is not a
# part of the key, and every worker uses its own scratch vector.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using Random
using KadanoffBaym

const KB = KadanoffBaym

# Random full Green's function with all components
function th_random(C, rng)
    X = ℱ(C, FERMI)
    d = C.ndim1
    for m = 1:C.ntau
        X.mat[m] = randn(rng, C64, d, d)
    end
    for n = 1:C.ntime
        for m = 1:n
            X.ret[n, m] = randn(rng, C64, d, d)
            L = randn(rng, C64, d, d)
            X.less[m, n] = m == n ? (L - L') / 2 : L
        end
        for m = 1:C.ntau
            X.lmix[n, m] = randn(rng, C64, d, d)
        end
    end
    return X
end

th_same(X, Y) = X.mat.data == Y.mat.data && X.ret.data == Y.ret.data &&
                X.lmix.data == Y.lmix.data && X.less.data == Y.less.data

# A short range self-energy and the full Dyson solution
function th_dyson_setup(d, nt, ntau, h, β, μ, k)
    C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
    H₀ = d == 1 ? fill(C64(-0.4), 1, 1) : ComplexF64[-1.0 0.2im; -0.2im 1.0]
    Σ = ℱ(C, FERMI)
    green_from_H!(Σ, μ, H₀, β, h)
    for n = 1:nt, m = 1:n
        f = 0.3 * exp(-((n - m) * h)^2) * cos(0.2 * (n - 1) * h)
        Σ.ret[n, m] = f .* Σ.ret[n, m]
        Σ.less[m, n] = f .* Σ.less[m, n]
    end
    Σ.mat.data .*= 0.3
    Σ.lmix.data .*= 0.3
    Hf = Cf(nt, d, d)
    for i = 0:nt
        Hf[i] = H₀
    end
    return C, Σ, Hf
end

@testset verbose = true "KadanoffBaym: threads" begin
    nrep = kb_test_profile() == :release ? 100 : 10
    @test Threads.nthreads() ≥ 1

    @testset "convolution (d = $d, k = $k)" for d in (1, 2), k in (1, 3, 5)
        rng = MersenneTwister(10d + k)
        C = Cn(14, 13, d, d, 1.3, 1.0)
        A, Acc, B, Bcc = (th_random(C, rng) for _ = 1:4)
        f = Cf(14, d, d)
        for i = 0:14
            f[i] = randn(rng, C64, d, d)
        end
        I = Integrator(k)
        X₁ = ℱ(C, FERMI)
        convolution(X₁, A, Acc, B, Bcc, I, 1.0, 0.1)
        Y₁ = ℱ(C, FERMI)
        convolution(Y₁, A, Acc, f, B, Bcc, I, 1.0, 0.1)
        same = true
        for _ = 1:nrep
            X₂ = ℱ(C, FERMI)
            convolution(X₂, A, Acc, B, Bcc, I, 1.0, 0.1; threaded = true)
            Y₂ = ℱ(C, FERMI)
            convolution(Y₂, A, Acc, f, B, Bcc, I, 1.0, 0.1; threaded = true)
            same &= th_same(X₁, X₂) && th_same(Y₁, Y₂)
        end
        @test same
        # A single time step
        Z₁ = ℱ(C, FERMI)
        Z₂ = ℱ(C, FERMI)
        convolution_time_step(9, Z₁, A, Acc, B, Bcc, I, 1.0, 0.1)
        convolution_time_step(9, Z₂, A, Acc, B, Bcc, I, 1.0, 0.1; threaded = true)
        @test th_same(Z₁, Z₂)
    end

    @testset "full Dyson and VIE2 time steps (d = $d, k = $k)" for d in (1, 2), k in (2, 5)
        nt, ntau, h, β, μ = 16, 21, 0.05, 1.0, 0.1
        C, Σ, Hf = th_dyson_setup(d, nt, ntau, h, β, μ, k)
        G₁ = ℱ(C, FERMI)
        G₂ = ℱ(C, FERMI)
        for G in (G₁, G₂)
            dyson_mat!(G, Σ, μ, Hf, β; solve_order = k)
            dyson_start!(G, μ, Hf, Σ, β, h; solve_order = k)
        end
        for n = k+2:nt
            dyson_timestep!(n, G₁, μ, Hf, Σ, β, h; solve_order = k)
            dyson_timestep!(n, G₂, μ, Hf, Σ, β, h; solve_order = k, threaded = true)
        end
        @test th_same(G₁, G₂)
        @test density(G₁, nt) == density(G₂, nt)

        I = Integrator(k)
        Qcopy = deepcopy(G₁)
        X₁ = ℱ(C, FERMI)
        X₂ = ℱ(C, FERMI)
        for n = 0:nt
            vie2_timestep!(n, X₁, Σ, Σ, G₁, I, β, h)
            vie2_timestep!(n, X₂, Σ, Σ, G₁, I, β, h; threaded = true)
        end
        @test th_same(X₁, X₂)
        # The source term is not changed
        @test th_same(G₁, Qcopy)
    end

    @testset "moving solvers (d = $d, k = $k)" for d in (1, 2), k in (1, 3, 5)
        nt, ntau, h, β, μ, tc = 40, 13, 0.02, 1.0, -0.1, 15
        C, Σ, Hf = th_dyson_setup(d, nt, ntau, h, β, μ, 3)
        fill!(Σ.mat.data, zero(C64))
        fill!(Σ.lmix.data, zero(C64))
        G = ℱ(C, FERMI)
        dyson!(G, μ, Hf, Σ, β, h; solve_order = 3)
        mk(X) = (M = ℱᵐᵒᵛ(tc, d); set_from_G_backward!(M, X, nt); M)
        Hm = Cfᵐᵒᵛ(tc, d)
        set_from_function_backward!(Hm, Hf, nt)
        S = mk(Σ)
        G₁ = mk(G)
        dyson_timestep!(nt, G₁, μ, Hm, S, h; solve_order = k)
        X₁ = mk(G)
        vie2_timestep!(nt, X₁, S, S, mk(G), Integrator(k), h)
        same = true
        for _ = 1:nrep
            G₂ = mk(G)
            dyson_timestep!(nt, G₂, μ, Hm, S, h; solve_order = k, threaded = true)
            X₂ = mk(G)
            vie2_timestep!(nt, X₂, S, S, mk(G), Integrator(k), h; threaded = true)
            same &= G₁.ret.data == G₂.ret.data && G₁.less.data == G₂.less.data &&
                    X₁.ret.data == X₂.ret.data && X₁.less.data == X₂.less.data
        end
        @test same
        # The output and the self-energy share no storage
        @test_throws ArgumentError dyson_timestep!(nt, S, μ, Hm, S, h; solve_order = k,
                                                   threaded = true)
    end

    @testset "FFT plan cache thread contract" begin
        N = 1024
        clear_fft_plan_cache!()
        nw = max(4, Threads.nthreads())
        plans = Vector{Any}(undef, nw)
        scratches = Vector{Vector{C64}}(undef, nw)
        rng = MersenneTwister(3)
        xs = [randn(rng, C64, N) for _ = 1:nw]
        ys = [copy(x) for x in xs]
        Threads.@threads :static for w = 1:nw
            p = plan_fft(N, KB.FFT_FORWARD)
            s = allocate_fft_scratch(p)
            plans[w] = p
            scratches[w] = s
            fft!(ys[w], KB.FFT_FORWARD; plan = p, scratch = s)
        end
        # The same key returns the same immutable plan
        @test all(p -> p === plans[1], plans)
        @test !ismutable(plans[1])
        # Every worker owns its scratch
        @test length(unique(objectid.(scratches))) == nw
        # Threaded results agree with the serial transform
        @test all(w -> isapprox(ys[w], fft(xs[w], KB.FFT_FORWARD); atol = 1e-10, rtol = 1e-10),
                  1:nw)
        # The old plan stays usable after clearing the cache, a new request
        # returns a new plan
        old = plans[1]
        clear_fft_plan_cache!()
        y = copy(xs[1])
        fft!(y, KB.FFT_FORWARD; plan = old, scratch = scratches[1])
        @test y == ys[1]
        @test plan_fft(N, KB.FFT_FORWARD) !== old
    end
end
