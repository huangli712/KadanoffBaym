#
# holstein_common.jl
#
# Shared helpers of tutor/holstein_bethe/validate.jl and
# tutor/holstein_impurity/validate.jl: run a Holstein entry of the quick
# profile in a temporary directory, and compare the observables with a
# kbref `observable` file (atol = rtol = 1e-10). The reference files are
# read from tutor/data or from the directory given by the environment
# variable KB_P5_REFDIR.
#

using Test
using LinearAlgebra
using KadanoffBaym

isdefined(Main, :read_kbref) ||
    include(joinpath(@__DIR__, "kbref.jl"))

const HOL_REFDIR = get(ENV, "KB_P5_REFDIR", joinpath(@__DIR__, "..", "..", "..", "data", "tutor"))
const HOL_TUTOR = joinpath(@__DIR__, "..")

# Copy the TOML file and the drive files of a tutor directory into a
# temporary directory; returns the path of the TOML file
function hol_setup(sub::AbstractString, toml::AbstractString)
    dir = mktempdir()
    src = joinpath(HOL_TUTOR, sub)
    for f in readdir(src)
        endswith(f, ".dat") && cp(joinpath(src, f), joinpath(dir, f))
    end
    cp(joinpath(src, toml), joinpath(dir, toml))
    return joinpath(dir, toml)
end

# Value of a kbref row in a result: `scalars` maps the component to the
# vector indexed by the KB time step + 1, `rho(t, a, b)` gives the density
# matrix element at the NESSi time step t
function hol_value(row, scalars, rho)
    row.component == "rho" && return rho(row.i, row.a, row.b)
    return real(scalars[row.component][row.i+2])
end

# Pointwise comparison; returns (pass, maximal error, component, time step)
function hol_compare(d, scalars, rho)
    ok = true
    worst = (0.0, "", -2)
    for row in d.rows
        v = hol_value(row, scalars, rho)
        e = abs(v - row.value)
        ok &= e ≤ 1e-10 + 1e-10 * abs(row.value)
        e > worst[1] && (worst = (e, row.component, row.i))
    end
    return ok, worst
end

# The same data with the rows of the component c at the NESSi time step 1
# changed by δ (for `rho` only the off-diagonal, i.e. anomalous, elements)
function hol_tamper(d, c, δ)
    hit(x) = x.component == c && x.i == 1 && (c != "rho" || x.a != x.b)
    return KbrefData(d.path, d.metadata, [hit(x) ? KbrefRow(x.component, x.i, x.j, x.a, x.b,
                                                     x.value + δ) : x for x in d.rows])
end
