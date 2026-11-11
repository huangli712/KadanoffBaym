#
# p5_profile_runner.jl
#
# The ten Phase 5 example entries for the profile suites of
# t_p5_quick_profiles.jl. Every entry runs its model `--profile quick`
# parameters in a temporary copy of its tutor directory. Loaded on demand
# only (the entries are compiled here).
#

module P5RunHubbard
    include(joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain_impl.jl"))
end

module P5RunFKM
    include(joinpath(@__DIR__, "..", "tutor", "fkm_bethe", "fkm_bethe.jl"))
end

module P5RunHolsteinBethe
    include(joinpath(@__DIR__, "..", "tutor", "holstein_bethe", "holstein_bethe_impl.jl"))
end

module P5RunHolsteinImpurity
    include(joinpath(@__DIR__, "..", "tutor", "holstein_impurity", "holstein_impurity_impl.jl"))
end

const P5RUN_TUTOR = joinpath(@__DIR__, "..", "tutor")

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
