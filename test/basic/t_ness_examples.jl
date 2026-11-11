#
# t_ness_examples.jl
#
# Smoke tests of the NESS2 tutorials: tutor/ness2_siam, tutor/ness2_bethe,
# and tutor/ness2_bethe_prop. Small temporary configurations are used, the
# output files are checked for the columns, finite values, and the number
# of rows; the production accuracy is not tested.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using KadanoffBaym

isdefined(Main, :read_kbref) || include(joinpath(@__DIR__, "..", "nessi", "common.jl"))

const NESS_TUTOR = joinpath(@__DIR__, "..", "..", "tutor")
const NESS_EXAMPLE_DATA = joinpath(@__DIR__, "..", "nessi", "data")

# Load a tutorial script into a fresh module
function load_tutorial(name::String)
    m = Module(Symbol(name))
    Base.include(m, joinpath(NESS_TUTOR, name, name * ".jl"))
    return m
end

# Numeric rows of a text file (comment lines are skipped)
function read_rows(fname::String)
    rows = Vector{Vector{Float64}}()
    for line in eachline(fname)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        push!(rows, parse.(Float64, split(line)))
    end
    return rows
end

no_hdf5(dir) = !any(f -> endswith(f, ".h5") || endswith(f, ".hdf5"), readdir(dir))

# Compare every reference record with the KB value `kbval(row)` (atol =
# rtol = 1e-10). Every record must be consumed: `kbval` returns `nothing`
# for a record without a KB counterpart, which is a failure.
function example_compare(ref, kbval)
    fails = String[]
    for r in ref.rows
        x = kbval(r)
        if x === nothing || !isapprox(x, r.value; atol = 1e-10, rtol = 1e-10)
            push!(fails, "$(r.component) ($(r.i), $(r.j), $(r.a), $(r.b)): kb = $x, nessi = $(r.value)")
        end
    end
    isempty(fails) || println("example reference failures:\n  ", join(first(fails, 10), "\n  "))
    return fails
end

# G^R/G^< of a steady-state Green's function, component gf_{ret,les}_{time,freq}
function example_gf(G::NessHermMat, r)
    A = occursin("ret", r.component) ? G.ret : G.les
    D = endswith(r.component, "time") ? A.time : A.freq
    return D[1, 1, r.j+1]
end

@testset verbose = true "KadanoffBaym: NESS2 tutorials" begin

    @testset "ness2_siam" begin
        m = load_tutorial("ness2_siam")
        mktempdir() do dir
            cfg = joinpath(dir, "siam.toml")
            write(cfg, """
            [model]
            U = 1.5
            beta = 10.0
            V = 0.8
            epsd = 0.3
            eta = 0.0
            [grid]
            Nft = 512
            h = 0.05
            [output]
            dir = "$(escape_string(dir))"
            prefix = "siam"
            """)
            r = m.main(cfg)
            @test all(isfinite, (r.density, r.Eint, r.current, r.current_GSL))
            @test abs(r.current_GSL - r.current) < 1e-8
            obs = read_rows(joinpath(dir, "siam_observables.dat"))
            @test length(obs) == 1 && length(obs[1]) == 4 && all(isfinite, obs[1])
            @test obs[1][1] ≈ r.density && abs(obs[1][4]) < 1e-8
            for f in ("siam_G_time.dat", "siam_G_freq.dat")
                rows = read_rows(joinpath(dir, f))
                @test length(rows) == 512
                @test all(x -> length(x) == 5 && all(isfinite, x), rows)
            end
            @test no_hdf5(dir)
        end
        # Even number of the grid points is required
        mktempdir() do dir
            cfg = joinpath(dir, "bad.toml")
            write(cfg, """
            [model]
            U = 1.0
            beta = 10.0
            V = 0.0
            epsd = 0.0
            eta = 0.0
            [grid]
            Nft = 511
            h = 0.05
            [output]
            dir = "$(escape_string(dir))"
            prefix = "bad"
            """)
            @test_throws AssertionError m.main(cfg)
        end
    end

    @testset "ness2_bethe" begin
        m = load_tutorial("ness2_bethe")
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

    # NESSi reference: the parameters are those of
    # .nessi/kbref/phase7_examples_siam_ref.cpp
    @testset "ness2_siam: NESSi reference" begin
        m = load_tutorial("ness2_siam")
        ref = read_kbref(joinpath(NESS_EXAMPLE_DATA, "phase7_examples_siam.dat"))
        @test ref.metadata["case"] == "siam"
        @test Set(r.component for r in ref.rows) ==
              Set(("gf_ret_time", "gf_les_time", "gf_ret_freq", "gf_les_freq",
                   "density", "current", "eint"))
        mktempdir() do dir
            cfg = joinpath(dir, "siam.toml")
            write(cfg, """
            [model]
            U = 1.5
            beta = 10.0
            V = 0.8
            epsd = 0.3
            eta = 0.1
            [grid]
            Nft = 512
            h = 0.05
            [output]
            dir = "$(escape_string(dir))"
            prefix = "siam"
            """)
            res = m.main(cfg)
            function kbval(r)
                startswith(r.component, "gf_") && return example_gf(res.G, r)
                r.component == "density" && return complex(res.density)
                r.component == "eint" && return complex(res.Eint)
                r.component == "current" && return complex(r.j == 0 ? res.current : res.current_GSL)
                return nothing
            end
            @test isempty(example_compare(ref, kbval))
        end
    end

    @testset "ness2_bethe: NESSi reference" begin
        m = load_tutorial("ness2_bethe")
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

    @testset "ness2_bethe_prop: NESSi reference" begin
        m = load_tutorial("ness2_bethe_prop")
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

    @testset "ness2_bethe_prop ($method, ipt = $ipt)" for (method, ipt) in (("fixpoint", 1), ("fourier", 0))
        m = load_tutorial("ness2_bethe_prop")
        mktempdir() do dir
            nt, ntau, k, ncorr = 8, 41, 3, 2
            cfg = joinpath(dir, "prop.toml")
            write(cfg, """
            [contour]
            nt = $nt
            ntau = $ntau
            h = 0.05
            beta = 5.0
            [model]
            U = 1.5
            mu = 0.0
            [solver]
            SolveOrder = $k
            ipt_flag = $ipt
            mat_method = "$method"
            BootstrapMaxIter = 20
            BootstrapMaxErr = 1e-8
            CorrectorSteps = $ncorr
            [output]
            dir = "$(escape_string(dir))"
            prefix = "prop"
            """)
            r = m.main(cfg)
            ntime = nt + 1
            @test length(r.density) == ntime + 1 && all(isfinite, r.density)
            @test all(isfinite, r.errors)
            dens = read_rows(joinpath(dir, "prop_density.dat"))
            @test length(dens) == ntime + 1
            @test all(x -> length(x) == 3 && all(isfinite, x), dens)
            errs = read_rows(joinpath(dir, "prop_errors.dat"))
            @test all(x -> length(x) == 4 && all(isfinite, x), errs)
            @test count(x -> x[1] == 0, errs) == r.mat_iterations + 1
            @test count(x -> x[1] == 1, errs) ≥ 1
            # Time stepping: CorrectorSteps + 1 iterations per time step
            @test count(x -> x[1] == 2, errs) == (ntime - k - 1) * (ncorr + 1)
            # Bridge to the steady state
            ness = read_rows(joinpath(dir, "prop_ness.dat"))
            @test length(ness[1]) == 3 && all(isfinite, ness[1])
            @test r.density_ness ≈ r.density[end] rtol = 1e-12
            @test r.tti < 1e-4
            @test no_hdf5(dir)
        end
    end
end
