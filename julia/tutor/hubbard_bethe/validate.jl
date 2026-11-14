#
# validate.jl
#
# Validation of the Hubbard Bethe examples:
#
#   * hubbard_bethe_ness2 / hubbard_bethe_ness2_prop (steady-state DMFT
#     and the propagation to the steady state) against the NESSi
#     reference data tutor/data/phase7_examples_bethe{,_prop}.dat
#     (generators .nessi/kbref/phase7_examples_bethe{,_prop}_ref.cpp);
#   * hubbard_bethe_trunc and hubbard_bethe_ness2_trunc (truncated
#     propagation, NESSi examples/programs/trunc_bethe_start.cpp,
#     trunc_bethe.cpp, ness2_bethe_trunc.cpp): regression tests and the
#     pointwise comparison with tutor/data/p8_trunc_bethe.dat and
#     p8_ness2_bethe_trunc.dat (generators .nessi/kbref/).
#
# Usage: julia --project=. tutor/hubbard_bethe/validate.jl
#

using Test
using LinearAlgebra
using KadanoffBaym

isdefined(Main, :read_rows) ||
    include(joinpath(@__DIR__, "..", "common", "example_tools.jl"))
isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

@testset verbose = true "NESS2 tutorial: Hubbard Bethe" begin

    @testset "hubbard_bethe_ness2" begin
        m = load_tutorial("hubbard_bethe", "hubbard_bethe_ness2")
        function cfg_bethe(dir, Niter, errmax, ipt, out_every; mixing = 0.6)
            cfg = joinpath(dir, "bethe.toml")
            write(cfg, """
            [grid]
            Nft = 512
            h = 0.05
            [model]
            beta = 5.0
            U = 1.5
            mu = 0.2
            [solver]
            Niter = $Niter
            errmax = $errmax
            mixing = $mixing
            ipt_flag = $ipt
            out_every = $out_every
            [output]
            dir = "$(escape_string(dir))"
            prefix = "bethe"
            """)
            return cfg
        end
        # Niter is reached
        for ipt in (1, 0)
            mktempdir() do dir
                r = m.main(cfg_bethe(dir, 6, 1e-30, ipt, 3))
                @test r.iterations == 6 && length(r.errors) == 6
                @test all(isfinite, r.errors) && all(isfinite, (r.density, r.Eint))
                rows = read_rows(joinpath(dir, "bethe_iterations.dat"))
                @test length(rows) == 6
                @test all(x -> length(x) == 3 && all(isfinite, x), rows)
                @test [Int(x[1]) for x in rows] == 1:6
                obs = read_rows(joinpath(dir, "bethe_observables.dat"))
                @test length(obs[1]) == 3 && Int(obs[1][3]) == 6
                for f in ("bethe_G_it3.dat", "bethe_G_it6.dat", "bethe_S_it6.dat",
                          "bethe_G_time.dat", "bethe_G_freq.dat",
                          "bethe_S_time.dat", "bethe_S_freq.dat")
                    rows = read_rows(joinpath(dir, f))
                    @test length(rows) == 512
                    @test all(x -> length(x) == 5 && all(isfinite, x), rows)
                end
                @test !isfile(joinpath(dir, "bethe_G_it4.dat"))
                @test no_hdf5(dir)
            end
        end
        # Convergence after at least four iterations
        mktempdir() do dir
            r = m.main(cfg_bethe(dir, 100, 1e-3, 1, 0))
            @test 4 ≤ r.iterations < 100
            @test r.errors[end] < 1e-3
            @test length(read_rows(joinpath(dir, "bethe_iterations.dat"))) == r.iterations
            @test !isfile(joinpath(dir, "bethe_G_it1.dat"))
        end
        mktempdir() do dir
            @test_throws AssertionError m.main(cfg_bethe(dir, 5, 1e-3, 1, 0; mixing = 0.0))
        end
    end

    @testset "hubbard_bethe_ness2: NESSi reference" begin
        m = load_tutorial("hubbard_bethe", "hubbard_bethe_ness2")
        ref = read_kbref(joinpath(NESS_EXAMPLE_DATA, "phase7_examples_bethe.dat"))
        @test ref.metadata["case"] == "bethe"
        @test Set(r.component for r in ref.rows) ==
              Set(("gf_ret_time", "gf_les_time", "gf_ret_freq", "gf_les_freq",
                   "density", "eint", "iteration_error"))
        mktempdir() do dir
            cfg = joinpath(dir, "bethe.toml")
            write(cfg, """
            [grid]
            Nft = 512
            h = 0.05
            [model]
            beta = 5.0
            U = 1.5
            mu = 0.2
            [solver]
            Niter = 6
            errmax = 1e-30
            mixing = 0.6
            ipt_flag = 1
            out_every = 0
            [output]
            dir = "$(escape_string(dir))"
            prefix = "bethe"
            """)
            res = m.main(cfg)
            # The iteration keys agree exactly
            @test sort([r.j for r in ref.rows if r.component == "iteration_error"]) ==
                  collect(1:length(res.errors))
            function kbval(r)
                startswith(r.component, "gf_") && return example_gf(res.G, r)
                r.component == "density" && return complex(res.density)
                r.component == "eint" && return complex(res.Eint)
                r.component == "iteration_error" &&
                    return r.j ≤ length(res.errors) ? complex(res.errors[r.j]) : nothing
                return nothing
            end
            @test isempty(example_compare(ref, kbval))
        end
    end

    @testset "hubbard_bethe_ness2_prop: NESSi reference" begin
        m = load_tutorial("hubbard_bethe", "hubbard_bethe_ness2_prop")
        ref = read_kbref(joinpath(NESS_EXAMPLE_DATA, "phase7_examples_bethe_prop.dat"))
        @test ref.metadata["case"] == "bethe_prop"
        @test Set(r.component for r in ref.rows) ==
              Set(("gf_ret_time", "gf_les_time", "density", "iteration_error"))
        mktempdir() do dir
            cfg = joinpath(dir, "prop.toml")
            write(cfg, """
            [contour]
            nt = 8
            ntau = 41
            h = 0.05
            beta = 5.0
            [model]
            U = 1.5
            mu = 0.0
            [solver]
            SolveOrder = 3
            ipt_flag = 1
            mat_method = "fixpoint"
            BootstrapMaxIter = 20
            BootstrapMaxErr = 1e-8
            CorrectorSteps = 2
            [output]
            dir = "$(escape_string(dir))"
            prefix = "prop"
            """)
            res = m.main(cfg)
            # Errors keyed by (stage, time step, iteration)
            errs = Dict((s, t, it) => e for (s, t, it, e) in res.error_log)
            @test length(errs) == length(res.error_log)
            rkeys = Set((r.a, r.b, r.j) for r in ref.rows if r.component == "iteration_error")
            @test rkeys == Set(keys(errs))
            Nft = getnft(res.Gness)
            function kbval(r)
                # G^<(t_nt - t, t_nt) is stored at the negative time -t
                r.component == "gf_ret_time" && return res.Gness.ret.time[1, 1, r.j+1]
                r.component == "gf_les_time" && return res.Gness.les.time[1, 1, mod(-r.j, Nft)+1]
                r.component == "density" && return complex(res.density[r.j+1])
                if r.component == "iteration_error"
                    e = get(errs, (r.a, r.b, r.j), nothing)
                    return e === nothing ? nothing : complex(e)
                end
                return nothing
            end
            @test isempty(example_compare(ref, kbval))
        end
    end
end

#
# Regression of the truncated Bethe examples hubbard_bethe_trunc and
# hubbard_bethe_ness2_trunc (NESSi examples/programs/trunc_bethe_start.cpp,
# trunc_bethe.cpp, ness2_bethe_trunc.cpp).
#
include(joinpath(@__DIR__, "hubbard_bethe_trunc_model.jl"))
module TBNessTrunc
include(joinpath(@__DIR__, "hubbard_bethe_ness2_trunc.jl"))
end
module TBMain
include(joinpath(@__DIR__, "hubbard_bethe_trunc.jl"))
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
        # the discretization error (see the package test suite)
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
        txt = read(joinpath(@__DIR__, "hubbard_bethe_trunc.toml"), String)
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
        txt = read(joinpath(@__DIR__, "hubbard_bethe_ness2_trunc.toml"), String)
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

#
# Pointwise comparison of the truncated Bethe examples with the NESSi
# reference data (generators .nessi/kbref/p8_trunc_bethe_ref.cpp and
# p8_ness2_bethe_trunc_ref.cpp). The NESSi time step `step` of a series
# record is the KB time step `step + 1`.
#
module MovRefTB
include(joinpath(@__DIR__, "hubbard_bethe_trunc_model.jl"))
end
module MovRefNBT
include(joinpath(@__DIR__, "hubbard_bethe_ness2_trunc.jl"))
end

const MOVREF_DATA = joinpath(@__DIR__, "..", "..", "..", "data", "tutor")

movref_ok(a, b) = isapprox(a, b; atol = 1e-10, rtol = 1e-10)

@testset verbose = true "Phase 8 NESSi reference: truncated Bethe" begin
    @testset "trunc Bethe (p8_trunc_bethe.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_trunc_bethe.dat"))
        md = d.metadata
        tc = parse(Int, md["tc"])
        # KB ntau counts the points (NESSi ntau + 1)
        p = (nt = parse(Int, md["nt"]), ntau = parse(Int, md["ntau"]) + 1, h = 0.02,
             beta = 5.0, tc = tc, tmax = parse(Int, md["nsteps"]) - 1, U = 1.0,
             epsk = Float64[], SolveOrder = 5, BootstrapMaxIter = 40,
             BootstrapMaxErr = 1e-10, CorrectorSteps = 3)
        start = MovRefTB.solve_bethe_start!(p)
        st = MovRefTB.initialize_moving_state(start, tc)
        # Densities of the window rows at the handoff (NESSi steps 0..tc)
        dens = Dict(tc - j => MovRefTB.moving_density(st.G, j) for j = 0:tc)
        out = MovRefTB.propagate_bethe_moving!(st, p)
        # Records of the propagation, keyed by the NESSi step
        for r in out
            dens[r.tstp - 1] = r.density
        end
        last = p.tmax
        used = 0
        ok = true
        for r in d.rows
            s = r.i
            if r.component == "density"
                v = dens[s]
            elseif r.component in ("error", "ekin")
                o = out[s - tc]
                @assert o.tstp == s + 1
                v = r.component == "error" ? o.err : o.ekin
            else
                # Two-time leading rows, only the last step is kept in memory
                s == last || continue
                X = startswith(r.component, "g_") ? st.G : st.Σ
                v = endswith(r.component, "_ret") ? X.ret[0, r.j][1] : X.less[0, r.j][1]
            end
            ok &= movref_ok(v, r.value)
            used += 1
        end
        @test ok
        # every scalar record and the leading rows of the last step
        @test used == count(r -> r.component in ("density", "error", "ekin") || r.i == last,
                            d.rows)
    end

    @testset "NESS2 trunc Bethe (p8_ness2_bethe_trunc.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_ness2_bethe_trunc.dat"))
        dir = mktempdir()
        cfg = joinpath(dir, "nbt.toml")
        txt = read(joinpath(@__DIR__, "hubbard_bethe_ness2_trunc.toml"), String)
        txt = replace(txt, r"dir = \".*\"" => "dir = \"$(escape_string(dir))\"",
                      r"Nft = \d+" => "Nft = 256", r"Niter = \d+" => "Niter = 20",
                      r"tmax = \d+" => "tmax = $(parse(Int, d.metadata["nsteps"]) - 1)")
        write(cfg, txt)
        res = redirect_stdout(devnull) do
            MovRefNBT.main(cfg)
        end
        last = length(res.steps)
        dens = Dict(0 => MovRefNBT.moving_density(res.initial.G))
        for s in res.steps
            dens[s.rel] = s.density
        end
        ok = true
        used = 0
        for r in d.rows
            if r.component == "density"
                v = dens[r.i]
            elseif r.i == 0 || r.i == last
                X = r.i == 0 ? res.initial.G : res.G
                v = r.component == "g_ret" ? X.ret[0, r.j][1] : X.less[0, r.j][1]
            else
                continue
            end
            ok &= movref_ok(v, r.value)
            used += 1
        end
        @test ok
        @test used == count(r -> r.component == "density" || r.i in (0, last), d.rows)
        # NESS relative step 0 is the KB time step 1
        @test gettstp(res.initial.G) == 1
    end
end
