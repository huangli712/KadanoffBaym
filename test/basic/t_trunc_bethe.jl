#
# t_trunc_bethe.jl
#
# Regression of the truncated Bethe examples tutor/bethe_trunc and
# tutor/bethe_ness2_trunc (NESSi examples/programs/trunc_bethe_start.cpp,
# trunc_bethe.cpp, ness2_bethe_trunc.cpp). The pointwise comparison with
# the NESSi reference p8_trunc_bethe.dat / p8_ness2_bethe_trunc.dat is
# done in test/nessi/t_moving_reference.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const TB_DIR = joinpath(@__DIR__, "..", "..", "tutor")
include(joinpath(TB_DIR, "bethe_trunc", "bethe_trunc_model.jl"))
module TBNessTrunc
include(joinpath(@__DIR__, "..", "..", "tutor", "bethe_ness2_trunc", "bethe_ness2_trunc.jl"))
end
module TBMain
include(joinpath(@__DIR__, "..", "..", "tutor", "bethe_trunc", "bethe_trunc.jl"))
end

const TB_PARAMS = (nt = 15, ntau = 51, h = 0.02, beta = 5.0, tc = 15, tmax = 25, U = 1.0,
                   epsk = [-1.0, 1.0], SolveOrder = 5, BootstrapMaxIter = 40,
                   BootstrapMaxErr = 1e-10, CorrectorSteps = 3)

# Leading row of the moving G against the full G at the KB time step n
function tb_rowerr(Gm, G, n)
    e = 0.0
    for j = 0:Gm.tc
        n - j ≥ 1 || continue
        e = max(e, maximum(abs.(Gm[:ret, n, n - j] - G.ret[n, n - j])))
        e = max(e, maximum(abs.(Gm[:less, n - j, n] - G.less[n - j, n])))
    end
    return e
end

@testset verbose = true "KadanoffBaym: truncated Bethe examples" begin
    p = TB_PARAMS
    start = solve_bethe_start!(p)
    tc = p.tc
    n₀ = tc + 1

    @testset "full start" begin
        d = [real(density_matrix(start.G, n)[1, 1]) for n = 1:n₀]
        @test all(isfinite, d)
        # Half filling is conserved
        @test maximum(abs.(d .- 0.5)) < 1e-8
        # The bootstrapping converged
        @test any(e -> e[1] == 1 && e[4] < p.BootstrapMaxErr, start.errors)
    end

    @testset "moving self-energy = full self-energy" begin
        st = initialize_moving_state(start, tc)
        # The stored full Σ at n₀ belongs to the last corrector iteration,
        # so it is recomputed from the final full G first
        Σf = deepcopy(start.Σ)
        Wf = ℱ(Cn(getntime(start.G), getntau(start.G), 1, 1, 1.0, 1.0), BOSE)
        update_bethe_selfenergy!(n₀, Σf, start.G, Wf, start.U)
        # Recompute the leading row from the moving G
        update_bethe_selfenergy!(st.Σ, st.G, st.W, st.U)
        # The moving window stores G^<(t,t-j) = -G^<(t-j,t)†, the full
        # bubble reads G^<(t-j,t). They differ only by the anti-hermiticity
        # error of the diagonal blocks G^<(t,t) (discretization error).
        ah = maximum(norm(start.G.less[n, n] + start.G.less[n, n]') for n = 1:n₀)
        @test ah < 1e-8
        @test tb_rowerr(st.Σ, Σf, n₀) < 10 * ah + 1e-11
        # With an exactly anti-hermitian diagonal both agree to round-off
        Gh = deepcopy(start.G)
        for n = 1:n₀
            L = Gh.less[n, n]
            Gh.less[n, n] = (L - L') / 2
        end
        Σh = deepcopy(start.Σ)
        update_bethe_selfenergy!(n₀, Σh, Gh, Wf, start.U)
        sh = initialize_moving_state(merge(start, (G = Gh,)), tc)
        update_bethe_selfenergy!(sh.Σ, sh.G, sh.W, sh.U)
        @test tb_rowerr(sh.Σ, Σh, n₀) < 1e-11
        update_bethe_kernel!(st.K, st.Σ, st.G)
        e = maximum(maximum(abs.(st.K[:ret, n₀, n₀ - j] -
                                 (Σf.ret[n₀, n₀ - j] + start.G.ret[n₀, n₀ - j]))) for j = 0:tc)
        @test e < 1e-11
    end

    @testset "lattice G_k, zero mixing self-energy" begin
        # Σ^⌉ = 0 (U = 0 on the Matsubara branch), the window covers the
        # whole history: the moving and full Dyson equations agree up to
        # the discretization error (see t_dyson_moving.jl)
        @test maximum(abs, start.Σ.lmix.data) == 0.0
        st = initialize_moving_state(start, tc)
        for (g, e, gf) in zip(st.Gk, st.epsk, start.Gk)
            dyson_timestep!(n₀, g, 0.0, e, st.Σ, p.h; solve_order = p.SolveOrder)
            @test tb_rowerr(g, gf, n₀) < 1e-8
        end
    end

    @testset "in-memory handoff vs checkpoint" begin
        dir = mktempdir()
        s₁ = initialize_moving_state(start, tc)
        write_checkpoint(dir, s₁, p.h)
        s₂ = read_checkpoint(dir, p.h)
        # The reloaded state agrees exactly with the in-memory state
        same(X, Y) = gettstp(X) == gettstp(Y) &&
            all(X.ret[i, j] == Y.ret[i, j] && X.less[i, j] == Y.less[i, j]
                for i = 0:X.tc for j = 0:X.tc)
        @test all(same(getfield(s₁, s), getfield(s₂, s)) for s in (:G, :Σ, :K, :W))
        @test all(same(a, b) for (a, b) in zip(s₁.Gk, s₂.Gk))
        r₁ = propagate_bethe_moving!(s₁, p)
        r₂ = propagate_bethe_moving!(s₂, p)
        @test length(r₁) == p.tmax - tc
        @test all(i -> r₁[i].density == r₂[i].density && r₁[i].densk == r₂[i].densk &&
                       r₁[i].ekin == r₂[i].ekin, eachindex(r₁))
        @test all(r -> isfinite(r.density) && isfinite(r.ekin) && all(isfinite, r.densk), r₁)
        # Corrupted checkpoint files are rejected
        lines = readlines(joinpath(dir, "G.dat"))
        write(joinpath(dir, "G.dat"), join(lines[1:end-1], "\n") * "\n")
        @test_throws ArgumentError read_checkpoint(dir, p.h)
        write(joinpath(dir, "G.dat"), join(lines, "\n") * "\n")
        @test_throws ArgumentError read_checkpoint(dir, 2 * p.h)
    end

    @testset "uninitialized windows" begin
        G = ℱᵐᵒᵛ(tc, 1)
        @test_throws ArgumentError forward!(G)
        @test_throws ArgumentError dyson_timestep!(1, G, 0.0, Cfᵐᵒᵛ(tc, 1), ℱᵐᵒᵛ(tc, 1), p.h;
                                                   solve_order = 5)
    end

    @testset "tutor entry point" begin
        dir = mktempdir()
        cfg = joinpath(dir, "tb.toml")
        txt = read(joinpath(TB_DIR, "bethe_trunc", "bethe_trunc.toml"), String)
        write(cfg, replace(txt, r"dir = \".*\"" => "dir = \"$(escape_string(dir))\""))
        res = TBMain.main(cfg)
        q = TBMain.read_params(cfg)
        dens = filter(l -> !startswith(l, "#"), readlines(joinpath(dir, "bethe_trunc_density.dat")))
        ener = filter(l -> !startswith(l, "#"), readlines(joinpath(dir, "bethe_trunc_energy.dat")))
        @test length(dens) == q.tmax + 1
        @test length(ener) == q.tmax - q.tc
        @test all(l -> all(isfinite, parse.(Float64, split(l))), vcat(dens, ener))
    end

    @testset "NESS start" begin
        dir = mktempdir()
        cfg = joinpath(dir, "nbt.toml")
        txt = read(joinpath(TB_DIR, "bethe_ness2_trunc", "bethe_ness2_trunc.toml"), String)
        txt = replace(txt, r"dir = \".*\"" => "dir = \"$(escape_string(dir))\"",
                      r"Nft = \d+" => "Nft = 256", r"Niter = \d+" => "Niter = 20",
                      r"tmax = \d+" => "tmax = 4")
        write(cfg, txt)
        res = TBNessTrunc.main(cfg)
        G₀ = res.initial.G
        # NESS relative step 0 is the KB time step 1
        @test gettstp(G₀) == 1
        @test res.steps[1].tstp == 2
        # Initial window = downsampled steady state
        Gd = downsample(res.ness.G, 2)
        e = maximum(maximum(abs.(G₀.ret[i, j] - Gd.ret.time[:, :, j + 1])) +
                    maximum(abs.(G₀.less[i, j] - Gd.les.time[:, :, j + 1]))
                    for i = 0:G₀.tc for j = 0:G₀.tc)
        @test e < 1e-10
        @test all(s -> isfinite(s.density), res.steps)
        # The absolute indices of the uninitialized history are rejected
        @test_throws BoundsError G₀[:ret, 1, 0]
    end
end
