#
# order_gate.jl
#
# Shared convergence-order gate for all order tests (P2, P3, P4, P7).
# Later phases must include this file and call test_order_gate(); copying
# or weakening the criteria is not allowed.
#
# Criteria:
#   1. at least 4 grids, strictly increasing grid sizes N;
#   2. all errors are positive and finite;
#   3. at least 3 errors above the plateau threshold (round-off floor);
#   4. the least-squares slope of -log(err) versus log(N) over the
#      non-plateau points is ≥ expected - order_tol;
#   5. the error on the finest grid is ≤ cap.
#

using Test

"""
    check_order_gate(Ns, errs, expected, cap;
                     plateau = 1e-13, order_tol = 0.3)

Evaluate the convergence-order gate. Returns `(ok, observed, msg)`, where
`observed` is the fitted order (NaN if it could not be fitted) and `msg`
explains the first violated criterion.
"""
function check_order_gate(Ns::AbstractVector{<:Integer},
                          errs::AbstractVector{<:Real},
                          expected::Real, cap::Real;
                          plateau::Real = 1e-13, order_tol::Real = 0.3)
    length(Ns) == length(errs) ||
        return (false, NaN, "length(Ns) != length(errs)")
    length(Ns) ≥ 4 ||
        return (false, NaN, "at least 4 grids are required, got $(length(Ns))")
    all(diff(Ns) .> 0) ||
        return (false, NaN, "grid sizes must be strictly increasing")
    all(e -> isfinite(e) && e > 0, errs) ||
        return (false, NaN, "errors must be positive and finite")
    #
    sel = findall(e -> e > plateau, errs)
    length(sel) ≥ 3 ||
        return (false, NaN, "at least 3 non-plateau errors are required, got $(length(sel))")
    #
    x = log.(Float64.(Ns[sel]))
    y = log.(Float64.(errs[sel]))
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    observed = -sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    observed ≥ expected - order_tol ||
        return (false, observed, "observed order $observed < expected $expected - $order_tol")
    errs[end] ≤ cap ||
        return (false, observed, "finest-grid error $(errs[end]) > cap $cap")
    return (true, observed, "ok")
end

"""
    test_order_gate(Ns, errs, expected, cap; kwargs...)

Record the convergence-order gate as a test. Returns the observed order.
"""
function test_order_gate(Ns, errs, expected, cap; kwargs...)
    ok, observed, msg = check_order_gate(Ns, errs, expected, cap; kwargs...)
    ok || @error "order gate failed" Ns errs expected cap observed msg
    @test ok
    return observed
end

@testset "order gate: self test" begin
    Ns = [10, 20, 40, 80]
    # A clean fourth-order sequence passes
    errs4 = [3.0 * N^-4.0 for N in Ns]
    @test check_order_gate(Ns, errs4, 4, 1e-6)[1]
    @test isapprox(check_order_gate(Ns, errs4, 4, 1e-6)[2], 4.0; atol = 1e-10)
    test_order_gate(Ns, errs4, 4, 1e-6)
    #
    # Plateau at the finest grid is tolerated if 3 points remain
    errsp = [errs4[1:3]; 1e-15]
    @test check_order_gate(Ns, errsp, 4, 1e-6)[1]
    #
    # Fewer than 4 grids
    @test !check_order_gate(Ns[1:3], errs4[1:3], 4, 1e-6)[1]
    # Fewer than 3 non-plateau points
    @test !check_order_gate(Ns, [errs4[1:2]; 1e-15; 1e-16], 4, 1e-6)[1]
    # Finest-grid cap exceeded
    @test !check_order_gate(Ns, errs4, 4, errs4[end] / 2)[1]
    # Non-increasing grids, bad errors
    @test !check_order_gate([10, 20, 20, 40], errs4, 4, 1e-6)[1]
    @test !check_order_gate(Ns, [errs4[1:3]; NaN], 4, 1e-6)[1]
    @test !check_order_gate(Ns, [errs4[1:3]; 0.0], 4, 1e-6)[1]
    #
    # Independent reduced-order mutant: a third-order method must not pass
    # a fourth-order gate, and a first-order one must not pass second order.
    errs3 = [0.5 * N^-3.0 for N in Ns]
    @test !check_order_gate(Ns, errs3, 4, 1.0)[1]
    errs1 = [0.2 / N + 1e-3 / N^2 for N in [8, 16, 32, 64, 128]]
    @test !check_order_gate([8, 16, 32, 64, 128], errs1, 2, 1.0)[1]
end
