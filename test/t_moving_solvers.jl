#
# t_moving_solvers.jl
#
# Pointwise comparison of the Phase 8 moving (truncated) solver functions
# with the NESSi reference data (generators in .nessi/kbref, see
# .omo/plans/08-phase8-moving-parallel.md, Wave 7): the moving window
# algebra (p8_moving_case*.dat) and the truncated solvers on closed-form
# inputs (p8_moving_solver.dat). The example-level moving comparisons are
# in tutor/{hubbard_bethe,gw}/validate.jl.
#
# The NESSi time step `step` of a series record is the KB time step
# `step + 1`. The primary condition is isapprox(actual, expected;
# atol = 1e-10, rtol = 1e-10) for every record.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))
isdefined(Main, :read_kbref) || include(joinpath(@__DIR__, "common", "common.jl"))

using Test
using KadanoffBaym

const MOVREF_DATA = joinpath(@__DIR__, "data")

movref_ok(a, b) = isapprox(a, b; atol = 1e-10, rtol = 1e-10)

# Inputs of p8_moving_solver_ref (see .nessi/kbref/p8_moving_solver_ref.cpp)
function movref_solver_case(size::Int, k::Int, md)
    nt = parse(Int, md["nt"])
    ntau = parse(Int, md["ntau"]) + 1
    tc = parse(Int, md["tc"])
    β, h, μ = 1.0, 0.02, -0.1
    ntime = nt + 1
    C = Cn(ntime, ntau, size, size, nt * h, β)
    H0 = size == 1 ? fill(ComplexF64(-0.3), 1, 1) :
         ComplexF64[-0.3 0.2im; -0.2im 0.4]
    Σ = ℱ(C, FERMI)
    from_ham!(Σ, μ, H0, β, h)
    for n = 1:ntime, m = 1:n
        f = 0.3 * exp(-((n - m) * h)^2)
        Σ.ret[n, m] = f .* Σ.ret[n, m]
        Σ.less[m, n] = f .* Σ.less[m, n]
    end
    fill!(Σ.lmix.data, zero(ComplexF64))
    fill!(Σ.mat.data, zero(ComplexF64))
    Hf = Cf(ntime, size, size)
    for i = 0:ntime
        Hf[i] = H0
    end
    G = ℱ(C, FERMI)
    dyson!(G, μ, Hf, Σ, β, h; solve_order = k)
    mk(X) = (M = ℱᵐᵒᵛ(tc, size, getsign(X)); init_from_G!(M, X, ntime); M)
    Hm = Cfᵐᵒᵛ(tc, size)
    init_from_function!(Hm, Hf, ntime)
    return (Gm = mk(G), Sm = mk(Σ), Wm = mk(ℱ(C, BOSE)), Hm = Hm, ntime = ntime,
            tc = tc, μ = μ, h = h)
end

@testset verbose = true "KadanoffBaym: Phase 8 moving solvers" begin
    @testset "moving data and algebra (p8_moving_case*.dat)" begin
        d₀ = read_kbref(joinpath(MOVREF_DATA, "p8_moving_case0.dat"))
        x = movref_solver_case(2, 3, merge(d₀.metadata, Dict("nt" => "30", "ntau" => "40")))
        G, S, n = x.Gm, x.Sm, x.ntime
        F(m) = ComplexF64[1+0.1*m*x.h 0.2im; -0.1 0.8-0.05*m*x.h]
        # Window of the full G after the four NESSi operations
        states = Any[deepcopy(G)]
        forward!(G)
        extrapolate_timestep!(G, 3)
        push!(states, deepcopy(G))
        cf = Cf(n + 1, 2, 2)
        for m = 1:n+1
            cf[m] = F(m - 1)
        end
        Fm = Cfᵐᵒᵛ(x.tc, 2)
        init_from_function!(Fm, cf, n + 1)
        v = timestep_view(G, 0)
        smul!(Fm, v)
        smul!(v, Fm)
        push!(states, deepcopy(G))
        forward!(S)
        # The leading row of S is set to the row 1 (see p8_moving_ref.cpp)
        memcpy!(timestep_view(S, 1), timestep_view(S, 0))
        v = timestep_view(G, 0)
        smul!(v, 0.7)
        incr!(v, timestep_view(S, 0), 0.3)
        push!(states, deepcopy(G))
        ok = true
        used = 0
        for c = 0:3
            d = read_kbref(joinpath(MOVREF_DATA, "p8_moving_case$c.dat"))
            X = states[c+1]
            # NESSi time step tstp is the KB time step tstp + 1
            @test gettstp(X) == parse(Int, d.metadata["tstp"]) + 1
            for r in d.rows
                A = r.component == "ret" ? X.ret : X.less
                ok &= movref_ok(A[r.i, r.j][r.a+1, r.b+1], r.value)
                used += 1
            end
        end
        @test ok
        @test used == 4 * 2 * (x.tc + 1)^2 * 4
    end

    @testset "moving solvers (p8_moving_solver.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_moving_solver.dat"))
        cases = ((1, 1), (1, 3), (1, 5), (2, 1), (2, 3), (2, 5))
        used = 0
        ok = true
        for (c, (size, k)) in enumerate(cases)
            x = movref_solver_case(size, k, d.metadata)
            n = x.ntime
            rows = filter(r -> r.i == c - 1, d.rows)
            # Bubbles of the orbitals a (G) and b (Σ), before the Dyson step
            B1 = Dict{Tuple{Int,Int},Any}()
            B2 = Dict{Tuple{Int,Int},Any}()
            for a = 1:size, b = 1:size
                W1 = deepcopy(x.Wm)
                W2 = deepcopy(x.Wm)
                bubble1!(n, W1, 1, 1, x.Gm, x.Gm, a, a, x.Sm, x.Sm, b, b)
                bubble2!(n, W2, 1, 1, x.Gm, x.Gm, a, a, x.Sm, x.Sm, b, b)
                B1[(a, b)] = W1
                B2[(a, b)] = W2
            end
            kin = kinetic_energy_trunc(n, x.Gm, x.Sm, Integrator(k), x.h)
            # Truncated VIE2 G + Σ * G = G (Q = the window of G), before the
            # Dyson step changes the leading row of x.Gm
            Gv = deepcopy(x.Gm)
            vie2_timestep!(n, Gv, x.Sm, x.Sm, x.Gm, Integrator(k), x.h)
            dyson_timestep!(n, x.Gm, x.μ, x.Hm, x.Sm, x.h; solve_order = k)
            for r in rows
                a, b = r.a + 1, r.b + 1
                if r.component == "kinetic"
                    v = kin
                elseif startswith(r.component, "dyson")
                    X = r.component == "dyson_ret" ? x.Gm.ret : x.Gm.less
                    v = X[0, r.j][a, b]
                elseif startswith(r.component, "vie2")
                    X = r.component == "vie2_ret" ? Gv.ret : Gv.less
                    v = X[0, r.j][a, b]
                else
                    W = startswith(r.component, "bubble1") ? B1[(a, b)] : B2[(a, b)]
                    X = endswith(r.component, "_ret") ? W.ret : W.less
                    v = X[0, r.j][1, 1]
                end
                ok &= movref_ok(v, r.value)
                used += 1
            end
        end
        @test ok
        @test used == length(d.rows)
    end
end
