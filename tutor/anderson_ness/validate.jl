#
# validate.jl
#
# Validation of the steady state of the single impurity Anderson model
# (anderson_ness.jl): smoke test of the run and its output files, and the
# pointwise comparison with the NESSi reference data
# tutor/data/phase7_examples_siam.dat (generator
# .nessi/kbref/phase7_examples_siam_ref.cpp).
#
# Usage: julia --project=. tutor/anderson_ness/validate.jl
#

using Test
using KadanoffBaym

isdefined(Main, :read_rows) ||
    include(joinpath(@__DIR__, "..", "common", "example_tools.jl"))

@testset verbose = true "NESS2 tutorial: Anderson impurity" begin

    @testset "anderson_ness" begin
        m = load_tutorial("anderson_ness")
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

    # NESSi reference: the parameters are those of
    # .nessi/kbref/phase7_examples_siam_ref.cpp
    @testset "anderson_ness: NESSi reference" begin
        m = load_tutorial("anderson_ness")
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
end
