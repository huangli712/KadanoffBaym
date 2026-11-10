#
# Project : Lavender
# Source  : KadanoffBaym.jl
# File    : benchmark/phase8_threads.jl
#
# Benchmark of the threaded kernels (Phase 8, Wave 4).
#
# Usage:
#
#   KB_TEST_PROFILE=quick   JULIA_NUM_THREADS=4 julia --project=. benchmark/phase8_threads.jl --quick
#   KB_TEST_PROFILE=release JULIA_NUM_THREADS=4 julia --project=. benchmark/phase8_threads.jl --release
#
# --quick   : checks that the threaded paths are used, the serial/threaded
#             errors, and the BLAS thread restoration. No wall time gate.
# --release : frozen parameters, median of 5 wall times for 1, 2, and 4
#             threads (the thread count of the kernels is limited by
#             chunking the outputs), writes benchmark/phase8_threads.dat
#             and checks the timing gates t4/t1 ≤ 1.10 for every kernel
#             and the geometric-mean speedup ≥ 1.20.
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "test", "common", "profile.jl"))

#
# Frozen parameters, see .omo/plans/08-phase8-moving-parallel.md P8-W4-T7
#
# ntau is the number of KB imaginary time points (NESSi ntau + 1)
const FROZEN_RELEASE = (ntime = 128, ntau = 257, d = 2, k = 5, tc = 256)
const FROZEN_QUICK = (ntime = 24, ntau = 33, d = 2, k = 3, tc = 16)

"""
    with_blas_threads(f, axis::Symbol)

Run `f()` with one BLAS thread if `axis` is `:k` or `:kernel`, the BLAS
thread count is restored afterwards, also if `f` throws.
"""
function with_blas_threads(f, axis::Symbol)
    axis in (:k, :kernel, :serial) || throw(ArgumentError("unknown parallel axis $axis"))
    axis == :serial && return f()
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(nblas)
    end
end

# Short range self-energy and Hamiltonian on the contour
function bench_setup(p)
    nt, ntau, d = p.ntime, p.ntau, p.d
    h, β, μ = 0.02, 1.0, -0.1
    C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
    H₀ = ComplexF64[-1.0 0.2im; -0.2im 1.0]
    Σ = ℱ(C, FERMI)
    green_from_H!(Σ, μ, H₀, β, h)
    for n = 1:nt, m = 1:n
        f = 0.3 * exp(-((n - m) * h)^2)
        Σ.ret[n, m] = f .* Σ.ret[n, m]
        Σ.less[m, n] = f .* Σ.less[m, n]
    end
    Σ.mat.data .*= 0.3
    Σ.lmix.data .*= 0.3
    Hf = Cf(nt, d, d)
    for i = 0:nt
        Hf[i] = H₀
    end
    G = ℱ(C, FERMI)
    dyson!(G, μ, Hf, Σ, β, h; solve_order = p.k)
    return (C = C, Σ = Σ, Hf = Hf, G = G, h = h, β = β, μ = μ)
end

maxdiff(X, Y) = maximum(maximum(abs.(a .- b)) for (a, b) in
                        ((X.ret.data, Y.ret.data), (X.less.data, Y.less.data)))

"""
    benchmark_kernel(name, p, s, threaded)

Run one kernel. Returns the output object.
"""
function benchmark_kernel(name::Symbol, p, s, threaded::Bool)
    k, h, β, μ = p.k, s.h, s.β, s.μ
    if name == :convolution
        X = ℱ(s.C, FERMI)
        convolution(X, s.Σ, s.Σ, s.G, s.G, Integrator(k), β, h; threaded = threaded)
        return X
    elseif name == :dyson
        X = deepcopy(s.G)
        dyson_timestep!(p.ntime, X, μ, s.Hf, s.Σ, β, h; solve_order = k, threaded = threaded)
        return X
    elseif name == :vie2
        X = deepcopy(s.G)
        vie2_timestep!(p.ntime, X, s.Σ, s.Σ, s.G, Integrator(k), β, h; threaded = threaded)
        return X
    elseif name == :moving
        tc = min(p.tc, p.ntime - 1)
        mk(Y) = (M = ℱᵐᵒᵛ(tc, p.d); set_from_G_backward!(M, Y, p.ntime); M)
        Hm = Cfᵐᵒᵛ(tc, p.d)
        set_from_function_backward!(Hm, s.Hf, p.ntime)
        X = mk(s.G)
        dyson_timestep!(p.ntime, X, μ, Hm, mk(s.Σ), h; solve_order = k, threaded = threaded)
        return X
    end
    throw(ArgumentError("unknown kernel $name"))
end

const KERNELS = (:convolution, :dyson, :vie2, :moving)

# Median wall time of `nrep` runs after a warm-up (no Statistics dependency)
function median_time(f, nrep)
    f()
    ts = sort([@elapsed(f()) for _ = 1:nrep])
    m = length(ts)
    return isodd(m) ? ts[(m + 1) ÷ 2] : (ts[m ÷ 2] + ts[m ÷ 2 + 1]) / 2
end

function main(args)
    mode = "--quick" in args ? :quick : ("--release" in args ? :release : :none)
    mode == :none && error("usage: phase8_threads.jl --quick | --release")
    prof = Symbol(kb_test_profile())
    prof == mode || error("$mode benchmark requires KB_TEST_PROFILE=$mode, got $prof")

    p = mode == :release ? FROZEN_RELEASE : FROZEN_QUICK
    println("phase8_threads: mode = $mode, Threads.nthreads() = $(Threads.nthreads())")
    println("frozen parameters: $p")
    Threads.nthreads() > 1 || error("the benchmark needs JULIA_NUM_THREADS > 1")

    nblas = BLAS.get_num_threads()
    rows = String[]
    gm = Float64[]
    ok = true
    with_blas_threads(:kernel) do
        s = bench_setup(p)
        for name in KERNELS
            X₀ = benchmark_kernel(name, p, s, false)
            X₁ = benchmark_kernel(name, p, s, true)
            err = maxdiff(X₀, X₁)
            bytes = @allocated benchmark_kernel(name, p, s, true)
            ok &= err ≤ 1e-11
            if mode == :release
                t₁ = median_time(() -> benchmark_kernel(name, p, s, false), 5)
                t₄ = median_time(() -> benchmark_kernel(name, p, s, true), 5)
                push!(gm, t₁ / t₄)
                ok &= t₄ / t₁ ≤ 1.10
                push!(rows, @sprintf("%-12s %12.6f %12.6f %8.3f %12d %10.3e",
                                     name, t₁, t₄, t₁ / t₄, bytes, err))
            else
                push!(rows, @sprintf("%-12s %12d %10.3e", name, bytes, err))
            end
            println(rows[end])
        end
    end
    # BLAS threads restored, also after an exception
    ok &= BLAS.get_num_threads() == nblas
    try
        with_blas_threads(:k) do
            error("injected")
        end
    catch e
        e isa ErrorException || rethrow()
    end
    ok &= BLAS.get_num_threads() == nblas

    if mode == :release
        g = exp(sum(log.(gm)) / length(gm))
        @printf("geometric-mean speedup = %.3f\n", g)
        ok &= g ≥ 1.20
        open(joinpath(@__DIR__, "phase8_threads.dat"), "w") do io
            println(io, "# kernel t1 t4 speedup bytes max_error, threads = $(Threads.nthreads())")
            foreach(r -> println(io, r), rows)
            @printf(io, "# geometric-mean speedup %.3f\n", g)
        end
    end

    ok || error("phase8_threads: gate failed")
    println("phase8_threads: all gates passed")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
