#
# Project : Lavender
# Source  : integration.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#
# Usage:
#
#   julia --project=. tutor/integration/integration.jl \
#         tutor/integration/integration.toml out.dat
#
# The integral ∫₀ˣ exp(i t) dt = [exp(i x) - 1] / i is evaluated by the
# Gregory quadrature of order k on the grid x = n h, n = 0, ⋯, npts.
#
# Output: two columns `x abs(error)`.
#
# In addition, the convergence order is checked with the shared gate
# tutor/common/order_gate.jl for k = 1, ⋯, 5: the error at x = X (fixed)
# must decrease as h^(k+2). The rectangle rule (first order) is used as
# a reduced-order mutant, which must fail the same gate.
#
# Port of NESSi examples/programs/integration.cpp.
#

using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

"""
    gregory_integral(n::Int, h::Float64, I::Integrator)

Evaluate ``\\int^{nh}_0 e^{it} dt`` by the Gregory quadrature.
"""
function gregory_integral(n::Int, h::Float64, I::Integrator)
    k = I.k
    f(j) = cis(j * h)
    s = zero(ComplexF64)
    if n ≤ 2 * k + 2
        for j = 0:max(n, k)
            s += I.GIW[n, j] * f(j)
        end
    else
        for j = 0:k
            s += I.GIW.ω[j + 1] * f(j)
        end
        for j = k+1:n-k-1
            s += f(j)
        end
        for j = n-k:n
            s += I.GIW.ω[n - j + 1] * f(j)
        end
    end
    return h * s
end

"""
    rectangle_integral(n::Int, h::Float64)

Evaluate ``\\int^{nh}_0 e^{it} dt`` by the (first order) rectangle rule.
"""
rectangle_integral(n::Int, h::Float64) = h * sum(cis(j * h) for j = 0:n-1; init = 0.0im)

exact_integral(x::Float64) = (cis(x) - 1.0) / im

"""
    main(cfgfile::String, outfile::String)

Read the parameters from `cfgfile` (TOML), write the errors into
`outfile` (overwritten), and run the convergence-order gate.
"""
function main(cfgfile::String, outfile::String)
    p = inp_toml(cfgfile, true)
    npts = p["npts"]::Int
    h = Float64(p["h"])
    k = p["k"]::Int
    @assert 1 ≤ k ≤ 5 && npts ≥ 0 && h > 0.0

    I = Integrator(k)
    open(outfile, "w") do io
        println(io, "# x abs(error)")
        for n = 0:npts
            err = abs(gregory_integral(n, h, I) - exact_integral(n * h))
            @printf(io, "%.10e %.10e\n", n * h, err)
        end
    end

    # Convergence order at x = X with N = X / h intervals
    X = 2.5
    Ns = [16, 32, 64, 128]
    caps = (1e-6, 1e-8, 1e-10, 1e-11, 1e-12)
    @testset "integration: convergence order" begin
        for kk = 1:5
            Ik = Integrator(kk)
            errs = [abs(gregory_integral(N, X / N, Ik) - exact_integral(X)) for N in Ns]
            ord = test_order_gate(Ns, errs, kk + 2, caps[kk]; order_tol = 0.35)
            @printf("    k = %d: observed order %.2f, finest error %.2e\n", kk, ord, errs[end])
            # Reduced-order mutant: the rectangle rule
            errm = [abs(rectangle_integral(N, X / N) - exact_integral(X)) for N in Ns]
            @test !check_order_gate(Ns, errm, kk + 2, caps[kk]; order_tol = 0.35)[1]
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: integration.jl <config.toml> <output.dat>")
    main(ARGS[1], ARGS[2])
end
