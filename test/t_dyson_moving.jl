#
# t_dyson_moving.jl
#
# To test the truncated (moving) Dyson equation and the truncated kinetic
# energy integral.
#
# NESSi: libcntr/test/dyson_moving.cpp (the full Dyson equation is solved
# with Sigma.tv = 0, then the moving solver is started from the full
# history and both are compared). The moving Dyson equation of NESSi uses
# a different discretization (equation of motion in the first time for
# the retarded component), so the full and the moving solutions agree up
# to the time propagation error only; the agreement is checked by its
# convergence with h and for a fine grid. The pointwise NESSi reference is
# test/t_moving_solvers.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const KB = KadanoffBaym

#
# Self-energy of dyson_moving.cpp: a free 2x2 Green's function with a
# short range in t - t', time translation symmetry broken by cosines. The
# Matsubara and mixing components are zero unless `tv = true`.
#
function dm_setup(nt, ntau, h, β, μ, k; trange, tv = false)
    d = 2
    H₀ = ComplexF64[-1.0 0.2im; -0.2im 1.0]
    C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
    Σ = ℱ(C, FERMI)
    from_ham!(Σ, μ, H₀, β, h)
    for n = 1:nt, m = 1:n
        t = (n - 1) * h
        s = (m - 1) * h
        f = exp(-((t - s) / trange)^2) * cos(0.2 * t) * cos(0.2 * s)
        Σ.ret[n, m] = f .* Σ.ret[n, m]
        Σ.less[m, n] = f .* Σ.less[m, n]
    end
    fill!(Σ.mat.data, zero(C64))
    tv ? (Σ.lmix.data .*= 0.5) : fill!(Σ.lmix.data, zero(C64))
    Hf = Cf(nt, d, d)
    Hf[0] = H₀
    for n = 1:nt
        Hf[n] = H₀ .* cos(0.1 * (n - 1) * h)
    end
    G = ℱ(C, FERMI)
    dyson!(G, μ, Hf, Σ, β, h; solve_order = k)
    return G, Σ, Hf
end

# Moving copies of the full functions at the KB time step n₀
function dm_moving(G, Σ, Hf, tc, n₀)
    d = getdims(G)[1]
    Gm = ℱᵐᵒᵛ(tc, d)
    Sm = ℱᵐᵒᵛ(tc, d)
    Hm = Cfᵐᵒᵛ(tc, d)
    init_from_G!(Gm, G, n₀)
    init_from_G!(Sm, Σ, n₀)
    init_from_function!(Hm, Hf, n₀)
    return Gm, Sm, Hm
end

# Leading row of the moving G against the full G at the time step n
function dm_rowerr(Gm, G, n)
    er = 0.0
    el = 0.0
    for j = 0:Gm.tc
        n - j ≥ 1 || continue
        er = max(er, maximum(abs.(Gm[:ret, n, n - j] - G.ret[n, n - j])))
        el = max(el, maximum(abs.(Gm[:less, n - j, n] - G.less[n - j, n])))
    end
    return er, el
end

# Owning moving time step of the full G at the time step n (NESSi
# herm_matrix_timestep_moving::set_timestep)
function dm_fullrow(G, tc, n)
    d = getdims(G)[1]
    x = 𝒻ᵐᵒᵛ(tc, d, n, getsign(G))
    for j = 0:min(tc, n - 1)
        x[:ret, j] = G.ret[n, n - j]
        x[:less, j] = -Matrix(G.less[n - j, n]')
    end
    return x
end

# One step of the moving propagation, Σ and H are taken from the full ones
function dm_step!(Gm, Sm, Hm, Σ, Hf, μ, h, k)
    forward!(Sm)
    forward!(Hm)
    forward!(Gm)
    n = gettstp(Gm)
    v = timestep_view(Sm, 0)
    for j = 0:min(Sm.tc, n - 1)
        v[:ret, j] = Σ.ret[n, n - j]
        v[:less, j] = -Matrix(Σ.less[n - j, n]')
    end
    Hm[0] = Hf[n]
    dyson_timestep!(n, Gm, μ, Hm, Sm, h; solve_order = k)
    return n
end

@testset verbose = true "KadanoffBaym: truncated Dyson equation" begin
    β, μ = 1.0, -0.1

    @testset "free propagation (Σ = 0) vs from_ham" begin
        d, h, k, tc = 2, 0.01, 5, 14
        H₀ = ComplexF64[-0.7 0.3; 0.3 0.4]
        nt = 40
        C = Cn(nt, 11, d, d, (nt - 1) * h, β)
        G = ℱ(C, FERMI)
        from_ham!(G, μ, H₀, β, h)
        Σ = ℱ(C, FERMI)
        Hf = Cf(nt, d, d)
        for n = 0:nt
            Hf[n] = H₀
        end
        Gm, Sm, Hm = dm_moving(G, Σ, Hf, tc, tc + 1)
        for _ = tc+2:nt
            dm_step!(Gm, Sm, Hm, Σ, Hf, μ, h, k)
        end
        er, el = dm_rowerr(Gm, G, nt)
        @test er < 1e-10
        @test el < 1e-10
        # G^<(t,t) is anti-hermitian
        @test norm(Gm.less[0, 0] + Gm.less[0, 0]') < 1e-12
    end

    @testset "zero mixing components, window covering the history" begin
        # The discretizations of the full and the moving solvers differ,
        # the difference converges with h
        T = 0.8
        errs = Float64[]
        ahs = Float64[]
        for h in (0.02, 0.01)
            nt = round(Int, T / h) + 1
            G, Σ, Hf = dm_setup(nt, 21, h, β, μ, 5; trange = 1.0)
            Gm, Sm, Hm = dm_moving(G, Σ, Hf, nt - 1, nt)
            dyson_timestep!(nt, Gm, μ, Hm, Sm, h; solve_order = 5)
            er, el = dm_rowerr(Gm, G, nt)
            push!(errs, max(er, el))
            # G^<(t,t) is anti-hermitian up to the discretization error
            push!(ahs, norm(Gm.less[0, 0] + Gm.less[0, 0]'))
        end
        @test errs[2] < 1e-10
        @test errs[1] / errs[2] > 2^5
        @test ahs[2] < 1e-10
        @test ahs[1] / ahs[2] > 2^5
    end

    @testset "NESSi dyson_moving.cpp: truncated propagation" begin
        if kb_test_profile() == :release
            nt, ntau, h, tc = 200, 100, 0.02, 100
        else
            nt, ntau, h, tc = 60, 21, 0.02, 20
        end
        k = 5
        G, Σ, Hf = dm_setup(nt, ntau, h, β, μ, k; trange = tc * h / 5)
        Gm, Sm, Hm = dm_moving(G, Σ, Hf, tc, tc + 1)
        err = 0.0
        for _ = tc+2:nt
            n = dm_step!(Gm, Sm, Hm, Σ, Hf, μ, h, k)
            err += distance(timestep_view(Gm, 0), dm_fullrow(G, tc, n))
        end
        @test err / nt < 1e-3
    end

    @testset "nonzero mixing components: no full equivalence" begin
        h, nt = 0.01, 41
        G₀, Σ₀, Hf₀ = dm_setup(nt, 21, h, β, μ, 5; trange = 1.0)
        G₁, Σ₁, Hf₁ = dm_setup(nt, 21, h, β, μ, 5; trange = 1.0, tv = true)
        @test maximum(abs, Σ₁.lmix.data) > 0.01
        Gm, Sm, Hm = dm_moving(G₁, Σ₁, Hf₁, nt - 1, nt)
        dyson_timestep!(nt, Gm, μ, Hm, Sm, h; solve_order = 5)
        er, el = dm_rowerr(Gm, G₁, nt)
        # The initial correlations are missing in the moving solver
        @test max(er, el) > 1e-6
    end

    @testset "orders 1..5 converge to the full solution" for k = 1:5
        T = 0.4
        e = Float64[]
        for h in (0.02, 0.01)
            nt = round(Int, T / h) + 1
            G, Σ, Hf = dm_setup(nt, 11, h, β, μ, k; trange = 1.0)
            Gm, Sm, Hm = dm_moving(G, Σ, Hf, nt - 1, nt)
            dyson_timestep!(nt, Gm, μ, Hm, Sm, h; solve_order = k)
            push!(e, maximum(dm_rowerr(Gm, G, nt)))
        end
        @test e[1] / e[2] > 2^(k - 0.5)
    end

    @testset "truncated kinetic energy" begin
        h, nt, k, tc = 0.02, 30, 3, 12
        G, Σ, Hf = dm_setup(nt, 11, h, β, μ, k; trange = 1.0)
        Gm, Sm, _ = dm_moving(G, Σ, Hf, tc, nt)
        I = Integrator(k)
        # Direct sum over the full blocks
        s = zeros(C64, 2, 2)
        for l = 0:tc
            w = l ≤ k ? I.GIW.ω[l+1] : (l ≥ tc - k ? I.GIW.ω[tc-l+1] : 1.0)
            GR = G.ret[nt, nt - l]
            GL = -Matrix(G.less[nt - l, nt]')
            DR = Σ.ret[nt, nt - l]
            DL = -Matrix(Σ.less[nt - l, nt]')
            s .+= h * w .* (GR' * DL .- GL' * DR)
        end
        @test abs(kinetic_energy_trunc(nt, Gm, Sm, I, h) - imag(tr(s))) < 1e-12
        # tc ≥ 2k + 2 is required
        Gs, Ss, _ = dm_moving(G, Σ, Hf, 11, nt)
        @test_throws ArgumentError kinetic_energy_trunc(nt, Gs, Ss, Integrator(5), h)
    end

    @testset "failure paths" begin
        h, nt = 0.02, 30
        G, Σ, Hf = dm_setup(nt, 11, h, β, μ, 3; trange = 1.0)
        Gm, Sm, Hm = dm_moving(G, Σ, Hf, 9, 20)
        # tc ≥ 2k + 2
        @test_throws ArgumentError dyson_timestep!(20, Gm, μ, Hm, Sm, h; solve_order = 4)
        @test_throws ArgumentError dyson_timestep!(20, Gm, μ, Hm, Sm, h; solve_order = 6)
        @test_throws ArgumentError dyson_timestep!(21, Gm, μ, Hm, Sm, h; solve_order = 3)
        @test_throws ArgumentError dyson_timestep!(20, ℱᵐᵒᵛ(9, 2), μ, Hm, Sm, h; solve_order = 3)
        @test_throws ArgumentError dyson_timestep!(20, Gm, μ, Cfᵐᵒᵛ(9, 2), Sm, h; solve_order = 3)
        @test_throws ArgumentError dyson_timestep!(20, Gm, μ, Hm, Gm, h; solve_order = 3)
        # The history rows are not changed
        before = Gm.ret[1, 3]
        dyson_timestep!(20, Gm, μ, Hm, Sm, h; solve_order = 3)
        @test Gm.ret[1, 3] == before
    end
end
