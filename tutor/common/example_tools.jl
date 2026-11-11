#
# example_tools.jl
#
# Shared helpers of the tutor validation scripts that run whole example
# programs (tutor/anderson_ness/validate.jl and
# tutor/hubbard_bethe/validate.jl): loading a tutorial script into a fresh
# module, reading the text output, and comparing with kbref reference
# records.
#

using Test
using KadanoffBaym

isdefined(Main, :read_kbref) ||
    include(joinpath(@__DIR__, "kbref.jl"))

const NESS_TUTOR = joinpath(@__DIR__, "..")
const NESS_EXAMPLE_DATA = joinpath(@__DIR__, "..", "data")

# Load a tutorial script into a fresh module
function load_tutorial(dir::String, name::String)
    m = Module(Symbol(name))
    Base.include(m, joinpath(NESS_TUTOR, dir, name * ".jl"))
    return m
end

load_tutorial(name::String) = load_tutorial(name, name)

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
