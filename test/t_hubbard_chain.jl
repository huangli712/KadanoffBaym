#
# t_hubbard_chain.jl
#
# Hubbard chain examples (tutor/hubbard_chain, 2B/GW/TPP) against the
# NESSi reference driver .nessi/kbref/hubbard_chain_text.cpp (same
# algorithm, atol = rtol = 1e-10) and against the exact solution
# .nessi/nessi-2.0.0/examples/data/hubbardchain_exact_n2_U1_w5.dat
# (independent physical baseline, matched by time, approximation
# thresholds calibrated per approximation).
#
# The reference files are read from test/data or from the
# directory given by the environment variable KB_P5_REFDIR.
#

isdefined(Main, :read_kbref) || include(joinpath(@__DIR__, "common", "common.jl"))

# The example is loaded into its own module (the tutor/common files are
# included by several examples and tests)
module HubbardChainNessi

using Test
using KadanoffBaym
import Main: read_kbref

include(joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain_impl.jl"))

const HUB_REFDIR = get(ENV, "KB_P5_REFDIR", joinpath(@__DIR__, "data"))
const HUB_EXACT = joinpath(@__DIR__, "..", ".nessi", "nessi-2.0.0", "examples", "data",
                           "hubbardchain_exact_n2_U1_w5.dat")
const HUB_TOML = joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain.toml")

# Run an approximation of the quick profile, the output goes to a
# temporary directory
function hub_run(approx)
    dir = mktempdir()
    p = joinpath(dir, "hubbard_chain.toml")
    write(p, replace(read(HUB_TOML, String), "dir = \".\"" => "dir = \"$(escape_string(dir))\""))
    r = hubbard_main([p, "--profile", "quick"], approx; io = devnull)
    return r, dir
end

# Maximum relative-absolute error of the kbref observables; returns
# (pass, maximal error, component, time step)
function compare_hubbard_nessi(r, d)
    occ, en = r.occupation, r.energy
    worst = (0.0, "", -2)
    ok = true
    for row in d.rows
        n = row.i + 1
        kb = row.component == "n1" ? occ[n][2] :
             en[n][1+findfirst(==(row.component), ("Ekin", "Epot", "Etot"))]
        ref = real(row.value)
        e = abs(kb - ref)
        ok &= e ≤ 1e-10 + 1e-10 * abs(ref)
        e > worst[1] && (worst = (e, row.component, row.i))
    end
    return ok, worst
end

function compare_hubbard_green(st, d)
    kb = Dict((c, i, j) => v for (c, i, j, a, b, v) in hubbard_selected_green(st))
    worst = 0.0
    ok = true
    for row in d.rows
        v = kb[(row.component, row.i, row.j)]
        e = abs(v - row.value)
        ok &= e ≤ 1e-10 + 1e-10 * abs(row.value)
        worst = max(worst, e)
    end
    return ok, worst
end

# Exact solution matched by time: maximal deviation of n1 and Ekin on the
# common time points
function compare_hubbard_exact(r)
    ex = [parse.(Float64, split(l)) for l in eachline(HUB_EXACT) if !isempty(strip(l))]
    tmap = Dict(round(x[1]; digits = 8) => x for x in ex)
    dn = 0.0
    de = 0.0
    m = 0
    for (o, e) in zip(r.occupation, r.energy)
        key = round(o[1]; digits = 8)
        haskey(tmap, key) || continue
        x = tmap[key]
        dn = max(dn, abs(o[2] - x[2]))
        de = max(de, abs(e[2] - x[3]))
        m += 1
    end
    return m, dn, de
end

# Thresholds of the deviations from the exact solution on the quick grid
# (t ≤ 0.5), recorded from the 2B/GW/TPP runs; the approximations are not
# exact, these are trend checks only
const HUB_EXACT_TOL = Dict("2B" => (3e-3, 3e-2), "GW" => (5e-3, 3e-2), "TPP" => (5e-3, 3e-2))

@testset verbose = true "Phase 5: Hubbard chain" begin
    for approx in ("2B", "GW", "TPP")
        @testset "$approx" begin
            r, dir = hub_run(approx)
            lc = lowercase(approx)
            # Output files
            @test isfile(joinpath(dir, "hubbardchain_$(lc)_occupation.dat"))
            @test isfile(joinpath(dir, "hubbardchain_$(lc)_energy.dat"))
            # NESSi (same algorithm)
            d = read_kbref(joinpath(HUB_REFDIR, "hubbard_$(lc)_quick.dat"))
            @test d.metadata["data_kind"] == "observable"
            @test d.metadata["approximation"] == approx
            @test length(d.rows) == 4 * length(r.occupation)
            ok, worst = compare_hubbard_nessi(r, d)
            ok || @info "Hubbard $approx vs NESSi" worst
            @test ok
            g = read_kbref(joinpath(HUB_REFDIR, "hubbard_$(lc)_quick_green.dat"))
            okg, wg = compare_hubbard_green(r.state, g)
            okg || @info "Hubbard $approx Green vs NESSi" wg
            @test okg
            # Exact solution (trend)
            m, dn, de = compare_hubbard_exact(r)
            @test m == length(r.occupation)
            @test dn < HUB_EXACT_TOL[approx][1]
            @test de < HUB_EXACT_TOL[approx][2]
        end
    end
end

end # module HubbardChainNessi
