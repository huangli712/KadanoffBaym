#
# t_p5_examples.jl
#
# Helpers of the Phase 5 examples (tutor/common): TOML schema and contour
# mapping, time dependent parameters, deterministic text output, and the
# three-stage loop. No physical model is run here.
#

using Test
using KadanoffBaym

include(joinpath(@__DIR__, "..", "..", "tutor", "common", "example_config.jl"))
include(joinpath(@__DIR__, "..", "..", "tutor", "common", "text_output.jl"))
include(joinpath(@__DIR__, "..", "..", "tutor", "common", "contour_loop.jl"))

const P5EX_BASE = """
[contour]
nt = 6
ntau = 20
h = 0.05
beta = 4.0
solve_order = 3

[solver]
mats_max_iter = 10
mats_max_error = 1e-8
bootstrap_max_iter = 10
bootstrap_max_error = 1e-8
corrector_steps = 2

[model]
U = 1.0
Nsites = 2

[output]
dir = "."
prefix = "ex"

[profile.quick]
contour = { nt = 4 }

[profile.release]
"""

function p5ex_write(dir, txt; name = "ex.toml")
    p = joinpath(dir, name)
    write(p, txt)
    return p
end

p5ex_load(p; kw...) = load_example_config(p; model_required = ("U", "Nsites"), kw...)

@testset verbose = true "Phase 5 example helpers" begin
    mktempdir() do dir
        @testset "TOML schema and contour mapping" begin
            p = p5ex_write(dir, P5EX_BASE)
            cq = p5ex_load(p)
            cr = p5ex_load(p; profile = "release")
            c = contour_params(cq)
            @test cq.profile == "quick"
            @test c.nt == 4 && c.ntime == 5 && c.ntau == 20 && c.ntau_points == 21
            @test c.k == 3 && c.I.k == 3 && c.h == 0.05 && c.beta == 4.0
            @test contour_params(cr).nt == 6
            s = solver_params(cq)
            @test s.output_every == 1 && s.corrector_steps == 2
            @test cq.output["save_green_components"] == false
            bad(txt) = p5ex_load(p5ex_write(dir, txt; name = "bad.toml"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "solve_order = 3" => "solve_order = 6"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "solve_order = 3" => "solve_order = 0"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "U = 1.0" => "U = 1.0\nV = 2.0"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "U = 1.0\n" => ""))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "U = 1.0" => "U = nan"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "h = 0.05" => "h = inf"))
            @test_throws ArgumentError bad(P5EX_BASE * "\n[extra]\nx = 1\n")
            @test_throws ArgumentError bad(replace(P5EX_BASE, "[profile.release]\n" => ""))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "contour = { nt = 4 }" =>
                                                              "contour = { nq = 4 }"))
            # nt < solve_order (the quick profile sets nt)
            @test_throws ArgumentError bad(replace(P5EX_BASE, "contour = { nt = 4 }" =>
                                                              "contour = { nt = 2 }"))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "prefix = \"ex\"" =>
                                                              "prefix = \"ex.h5\""))
            @test_throws ArgumentError bad(replace(P5EX_BASE, "ntau = 20" => "ntau = 20.5"))
            @test_throws ArgumentError p5ex_load(p; profile = "fast")
        end

        @testset "time dependent parameters" begin
            txt = P5EX_BASE * """
            [drive.J]
            mode = "constant"
            value = 0.5

            [drive.g]
            mode = "file"
            path = "g.dat"
            """
            p = p5ex_write(dir, txt; name = "drive.toml")
            # nt = 4 (quick): nt + 2 = 6 values, the first is NESSi tstp = -1
            write(joinpath(dir, "g.dat"), "# comment\n1.0 2.0\n3.0\n4.0 5.0 6.0\n")
            cfg = load_example_config(p; model_required = ("U", "Nsites"), drives = ("J", "g"))
            J = Cf(5, 1, 1, zero(ComplexF64))
            load_tvector!(J, cfg, "J")
            @test all(i -> J[i][1, 1] == 0.5, 0:5)
            g = Cf(5, 1, 1, zero(ComplexF64))
            load_tvector!(g, cfg, "g")
            @test g[0][1, 1] == 1.0
            @test [real(g[i][1, 1]) for i = 1:5] == [2.0, 3.0, 4.0, 5.0, 6.0]
            # One value short
            write(joinpath(dir, "g.dat"), "1 2 3 4 5\n")
            @test_throws ArgumentError load_tvector!(g, cfg, "g")
            # Relative paths are resolved from the TOML directory
            sub = joinpath(dir, "sub")
            mkpath(sub)
            write(joinpath(sub, "g.dat"), "1 2 3 4 5 6\n")
            p2 = p5ex_write(sub, txt; name = "drive.toml")
            cfg2 = load_example_config(p2; model_required = ("U", "Nsites"), drives = ("J", "g"))
            load_tvector!(g, cfg2, "g")
            @test real(g[5][1, 1]) == 6.0
            # Unknown drive and bad mode
            @test_throws ArgumentError load_example_config(p; model_required = ("U", "Nsites"),
                                                           drives = ("J",))
            @test_throws ArgumentError load_example_config(
                p5ex_write(dir, replace(txt, "mode = \"constant\"" => "mode = \"ramp\"");
                           name = "bm.toml"); model_required = ("U", "Nsites"),
                drives = ("J", "g"))
        end

        @testset "deterministic text output" begin
            a = joinpath(dir, "a.dat")
            b = joinpath(dir, "b.dat")
            rows = [[0.0, 0.1, 1 / 3], [0.05, -2.5e-17, π]]
            write_table(a, ["t", "x", "y"], rows)
            write_table(b, ["t", "x", "y"], rows)
            @test read(a) == read(b)
            lines = readlines(a)
            @test lines[1] == "# 1:t 2:x 3:y"
            @test parse.(Float64, split(lines[3])) == rows[2]
            # A failing row leaves no partial file
            c = joinpath(dir, "c.dat")
            failing = (i == 2 ? error("boom") : [1.0, 2.0] for i = 1:3)
            @test_throws ErrorException write_table(c, ["x", "y"], failing)
            @test !isfile(c)
            @test_throws DomainError write_table(c, ["x"], [[NaN]])
            @test !isfile(c)
            @test length(readdir(dir)) == length(unique(readdir(dir)))
            @test !any(f -> startswith(f, "jl_"), readdir(dir))
            # Green's function components
            g = joinpath(dir, "g_green.dat")
            write_green_components(g, [("mat", 0, -1, 0, 0, 0.5 - 0.25im),
                                       ("les", 1, 3, 0, 0, 1e-20 + 2im)])
            gl = readlines(g)
            @test gl[1] == "# component i j a b re im"
            @test split(gl[2])[1:5] == ["mat", "0", "-1", "0", "0"]
            @test parse(Float64, split(gl[3])[7]) == 2.0
        end

        @testset "three-stage loop" begin
            ev = Any[]
            logf(e) = push!(ev, e)
            # Matsubara: converged after the error drops below tol
            errs = [1.0, 0.1, 1e-9, 1e-12]
            r = iterate_matsubara!(i -> errs[i+1]; max_iter = 3, tol = 1e-8, log = logf)
            @test r.iterations == 3
            @test_throws ConvergenceError iterate_matsubara!(i -> 1.0; max_iter = 2, tol = 1e-8)
            # min_iterations = 0 exits after the first iteration
            n = Ref(0)
            r = iterate_bootstrap!(i -> (n[] += 1; 0.0); max_iter = 5, tol = 1e-8,
                                   min_iterations = 0)
            @test n[] == 1 && r.iterations == 1
            # min_iterations = 3 does not exit earlier (NESSi iter > 2)
            n[] = 0
            r = iterate_bootstrap!(i -> (n[] += 1; 0.0); max_iter = 5, tol = 1e-8,
                                   min_iterations = 3)
            @test n[] == 4 && r.iterations == 4
            @test_throws ConvergenceError iterate_bootstrap!(i -> 1.0; max_iter = 2, tol = 1e-8,
                                                             min_iterations = 0)
            @test_throws ArgumentError iterate_bootstrap!(i -> 0.0; max_iter = 2, tol = 1e-8,
                                                          min_iterations = -1)
            # Propagation: predictor once, then the correctors
            empty!(ev)
            seq = Any[]
            propagate_timesteps!(n -> push!(seq, (:p, n)), (n, i) -> push!(seq, (:c, n, i));
                                 first_step = 5, last_step = 6, corrector_steps = 2, h = 0.1,
                                 log = logf)
            @test seq == [(:p, 5), (:c, 5, 0), (:c, 5, 1), (:p, 6), (:c, 6, 0), (:c, 6, 1)]
            @test [e.stage for e in ev] ==
                  [:predictor, :corrector, :corrector, :predictor, :corrector, :corrector]
            @test all(e -> e.nessi_step == e.kb_step - 1, ev)
            @test all(e -> e.t ≈ (e.kb_step - 1) * 0.1, ev)
            # Command line
            a = parse_example_args(["x.toml", "--profile", "release", "--validate-only"])
            @test a.profile == "release" && a.validate_only
            @test parse_example_args(["x.toml"]).profile == "quick"
            @test_throws ArgumentError parse_example_args(["x.toml", "--bogus"])
        end
    end
end
