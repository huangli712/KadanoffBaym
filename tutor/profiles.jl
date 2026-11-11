#
# profiles.jl
#
# Frozen-parameter and performance checks of the ten example entries
# (hubbard 2B/GW/TPP, FKM, Holstein Bethe/uMig/Nambu, Holstein impurity).
# Every entry runs its model `--profile quick` parameters in a temporary
# copy of its tutor directory (the entries are compiled here).
#
# Included from tutor/validate.jl, or directly:
#
#   KB_TEST_PROFILE=quick   julia --project=. tutor/profiles.jl --report <file>
#   KB_TEST_PROFILE=release julia --project=. tutor/profiles.jl --report <file>
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using TOML
using KadanoffBaym

module P5RunHubbard
    include(joinpath(@__DIR__, "hubbard_chain", "hubbard_chain_impl.jl"))
end

module P5RunFKM
    include(joinpath(@__DIR__, "fkm_bethe", "fkm_bethe.jl"))
end

module P5RunHolsteinBethe
    include(joinpath(@__DIR__, "holstein_bethe", "holstein_bethe_impl.jl"))
end

module P5RunHolsteinImpurity
    include(joinpath(@__DIR__, "holstein_impurity", "holstein_impurity_impl.jl"))
end

const P5RUN_TUTOR = @__DIR__

# Copy the TOML file and the drive files of a tutor directory into a new
# temporary directory; returns the arguments of the entry
function p5run_args(sub::AbstractString, toml::AbstractString)
    dir = mktempdir()
    src = joinpath(P5RUN_TUTOR, sub)
    for f in readdir(src)
        (endswith(f, ".dat") || f == toml) && cp(joinpath(src, f), joinpath(dir, f))
    end
    return [joinpath(dir, toml), "--profile", "quick"]
end

"""
    P5RUN_ENTRIES

`(name, run)` of the ten entries; `run()` executes the entry once.
"""
const P5RUN_ENTRIES = [
    "hubbard_2b" => () -> P5RunHubbard.hubbard_main(p5run_args("hubbard_chain", "hubbard_chain.toml"), "2B"; io = devnull),
    "hubbard_gw" => () -> P5RunHubbard.hubbard_main(p5run_args("hubbard_chain", "hubbard_chain.toml"), "GW"; io = devnull),
    "hubbard_tpp" => () -> P5RunHubbard.hubbard_main(p5run_args("hubbard_chain", "hubbard_chain.toml"), "TPP"; io = devnull),
    "fkm_bethe_quench" => () -> P5RunFKM.main(p5run_args("fkm_bethe", "fkm_bethe.toml"); io = devnull),
    "holstein_bethe_migdal" => () -> P5RunHolsteinBethe.holstein_bethe_main(
        p5run_args("holstein_bethe", "holstein_bethe_migdal.toml"), "Migdal"; io = devnull),
    "holstein_bethe_umig" => () -> P5RunHolsteinBethe.holstein_bethe_main(
        p5run_args("holstein_bethe", "holstein_bethe_umig.toml"), "uMig"; io = devnull),
    "holstein_nambu_migdal" => () -> P5RunHolsteinBethe.holstein_nambu_main(
        p5run_args("holstein_bethe", "holstein_bethe_nambu_migdal.toml"), "Migdal"; io = devnull),
    "holstein_nambu_umig" => () -> P5RunHolsteinBethe.holstein_nambu_main(
        p5run_args("holstein_bethe", "holstein_bethe_nambu_umig.toml"), "uMig"; io = devnull),
    "holstein_impurity_migdal" => () -> P5RunHolsteinImpurity.holstein_impurity_main(
        p5run_args("holstein_impurity", "holstein_impurity_singlebath_migdal.toml"), "Migdal"; io = devnull),
    "holstein_impurity_umig" => () -> P5RunHolsteinImpurity.holstein_impurity_main(
        p5run_args("holstein_impurity", "holstein_impurity_singlebath_umig.toml"), "uMig"; io = devnull),
]

"""
    p5run_measure(name, run)

Run an entry once and return its measurement: wall time, allocation
count, allocated bytes, GC time, and exit status (`"ok"` or the error).
"""
function p5run_measure(name::AbstractString, run::Function)
    status = "ok"
    st = @timed try
        run()
    catch e
        status = "error: " * sprint(showerror, e)
        nothing
    end
    return Dict{String,Any}("name" => name, "wall_s" => st.time,
                            "alloc_count" => Int(Base.gc_alloc_count(st.gcstats)),
                            "alloc_bytes" => Int(st.bytes), "gc_s" => st.gctime,
                            "status" => status)
end

isdefined(Main, :load_example_config) ||
    include(joinpath(@__DIR__, "common", "example_config.jl"))

# file => (nt, ntau, k, mats_max_iter, bootstrap_max_iter, corrector_steps)
const P5Q_FROZEN = [
    "hubbard_chain/hubbard_chain.toml" => (10, 40, 3, 100, 50, 3),
    "fkm_bethe/fkm_bethe.toml" => (10, 50, 3, 100, 100, 3),
    "holstein_bethe/holstein_bethe_migdal.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_bethe/holstein_bethe_umig.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_bethe/holstein_bethe_nambu_migdal.toml" => (6, 32, 3, 300, 60, 3),
    "holstein_bethe/holstein_bethe_nambu_umig.toml" => (6, 32, 3, 300, 60, 3),
    "holstein_impurity/holstein_impurity_singlebath_migdal.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_impurity/holstein_impurity_singlebath_umig.toml" => (8, 40, 3, 100, 50, 3),
]

# P5 quick sub-budget (s) of the measured second runs
const P5Q_QUICK_BUDGET = 60.0

# Release suite: per-entry (wall s, allocation GiB) of plan 3.6 and totals
const P5Q_RELEASE_BUDGET = Dict(
    "hubbard_2b" => (45.0, 2.0), "hubbard_gw" => (90.0, 4.0), "hubbard_tpp" => (90.0, 4.0),
    "fkm_bethe_quench" => (45.0, 2.0),
    "holstein_bethe_migdal" => (180.0, 6.0), "holstein_bethe_umig" => (90.0, 4.0),
    "holstein_nambu_migdal" => (300.0, 12.0), "holstein_nambu_umig" => (180.0, 8.0),
    "holstein_impurity_migdal" => (180.0, 6.0), "holstein_impurity_umig" => (90.0, 4.0))
const P5Q_RELEASE_TOTAL = (25 * 60.0, 60.0)

"""
    run_quick_profile_suite()

Warm up and run every entry a second time, return the measurements of
the second runs. Requires `KB_TEST_PROFILE=quick`.
"""
function run_quick_profile_suite()
    kb_test_profile() == "quick" || error("run_quick_profile_suite requires KB_TEST_PROFILE=quick")
    return p5q_second_runs()
end

# Warm up (precompilation) and measure the second run of every entry
function p5q_second_runs()
    out = Dict{String,Any}[]
    for (name, run) in P5RUN_ENTRIES
        p5run_measure(name, run)
        push!(out, p5run_measure(name, run))
    end
    return out
end

"""
    run_release_model_suite()

Run the ten entries with their model `--profile quick` parameters (the
second run after the precompilation is measured, plan 3.6), return the
measurements. Requires `KB_TEST_PROFILE=release` (the suite is not part
of the default quick profile).
"""
function run_release_model_suite()
    kb_test_profile() == "release" || error("run_release_model_suite requires KB_TEST_PROFILE=release")
    return p5q_second_runs()
end

"""
    write_profile_report(path, profile, entries)

Write the deterministic TOML report: Julia version, threads, CPU, the
per-entry measurements (in the order of the entries), and the totals.
"""
function write_profile_report(path::AbstractString, profile::AbstractString, entries)
    rep = Dict{String,Any}(
        "profile" => profile, "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(), "cpu" => strip(Sys.cpu_info()[1].model),
        "entries" => entries,
        "total_wall_s" => sum(e["wall_s"] for e in entries),
        "total_alloc_bytes" => sum(e["alloc_bytes"] for e in entries))
    open(path, "w") do io
        TOML.print(io, rep; sorted = true)
    end
    return rep
end

# Checks of the measurements
function p5q_check(profile, entries)
    @test all(e["status"] == "ok" for e in entries)
    if profile == "quick"
        @test length(entries) == 10
        @test sum(e["wall_s"] for e in entries) ≤ P5Q_QUICK_BUDGET
    else
        @test length(entries) == 10
        @test Set(e["name"] for e in entries) == Set(keys(P5Q_RELEASE_BUDGET))
        for e in entries
            w, a = P5Q_RELEASE_BUDGET[e["name"]]
            @test e["wall_s"] ≤ w
            @test e["alloc_bytes"] ≤ a * 2.0^30
        end
        @test sum(e["wall_s"] for e in entries) ≤ P5Q_RELEASE_TOTAL[1]
        @test sum(e["alloc_bytes"] for e in entries) ≤ P5Q_RELEASE_TOTAL[2] * 2.0^30
    end
end

@testset verbose = true "Phase 5: frozen quick profiles" begin
    for (f, v) in P5Q_FROZEN
        @testset "$f" begin
            path = joinpath(P5RUN_TUTOR, f)
            raw = TOML.parsefile(path)
            cfg = load_example_config(path; profile = "quick",
                                      model_required = Tuple(keys(raw["model"])),
                                      drives = Tuple(keys(get(raw, "drive", Dict()))))
            c, s = contour_params(cfg), solver_params(cfg)
            @test (c.ntime - 1, c.ntau_points - 1, c.k) == v[1:3]
            @test (s.mats_max_iter, s.bootstrap_max_iter, s.corrector_steps) == v[4:6]
            # Matsubara, bootstrap, and at least one propagation step
            @test c.ntime - 1 ≥ c.k + 2
        end
    end
    # The release suite is not available in the quick profile
    if kb_test_profile() == "quick" && abspath(PROGRAM_FILE) != @__FILE__
        @test_throws ErrorException run_release_model_suite()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    i = findfirst(==("--report"), ARGS)
    (i === nothing || i == length(ARGS)) && error("usage: profiles.jl --report <file>")
    profile = kb_test_profile()
    entries = profile == "quick" ? run_quick_profile_suite() : run_release_model_suite()
    write_profile_report(ARGS[i+1], profile, entries)
    @testset "Phase 5: $profile profile report" begin
        p5q_check(profile, entries)
    end
elseif kb_test_profile() == "release"
    # Release suite of the validation runner (no report file)
    @testset "Phase 5: release model suite" begin
        p5q_check("release", run_release_model_suite())
    end
end
