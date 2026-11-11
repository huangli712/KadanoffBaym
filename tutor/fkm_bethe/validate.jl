#
# validate.jl
#
# Falicov-Kimball quench example against the NESSi reference driver
# .nessi/kbref/fkm_bethe_quench_text.cpp (port of the Catch test
# libcntr/test/fkm_bethe_quench.cpp): den0 and den1 pointwise with atol =
# rtol = 1e-10, and the Catch aggregate check (sum of the squared errors
# < 1e-7) for both columns (the Catch test checks err0 twice, :139/:142).
#
# The reference files are read from tutor/data or from the directory
# given by the environment variable KB_P5_REFDIR.
#
# Usage: julia --project=. tutor/fkm_bethe/validate.jl
#

using Test
using KadanoffBaym

isdefined(Main, :read_kbref) ||
    include(joinpath(@__DIR__, "..", "common", "kbref.jl"))

module FKMExample
    include(joinpath(@__DIR__, "fkm_bethe.jl"))
end

const FKM_REFDIR = get(ENV, "KB_P5_REFDIR", joinpath(@__DIR__, "..", "data"))
const FKM_TOML = joinpath(@__DIR__, "fkm_bethe.toml")

# Pointwise comparison; returns (pass, maximal error, err0, err1)
function compare_fkm(dens, d)
    kb = Dict(n - 1 => (d0, d1) for (n, d0, d1) in dens)
    ok = true
    worst = 0.0
    err = Dict("den0" => 0.0, "den1" => 0.0)
    for r in d.rows
        r.component == "total" && continue
        v = kb[r.i][r.component == "den0" ? 1 : 2]
        e = abs(v - real(r.value))
        ok &= e ≤ 1e-10 + 1e-10 * abs(real(r.value))
        worst = max(worst, e)
        err[r.component] += e^2
    end
    return ok, worst, err["den0"], err["den1"]
end

@testset verbose = true "Phase 5: FKM Bethe quench" begin
    mktempdir() do dir
        p = joinpath(dir, "fkm.toml")
        write(p, replace(read(FKM_TOML, String), "dir = \".\"" => "dir = \"$(escape_string(dir))\""))
        r = FKMExample.main([p]; io = devnull)
        d = read_kbref(joinpath(FKM_REFDIR, "fkm_bethe_quench_quick.dat"))
        @test d.metadata["model"] == "fkm"
        @test length(d.rows) == 3 * length(r.densities)
        ok, worst, e0, e1 = compare_fkm(r.densities, d)
        ok || @info "FKM vs NESSi" worst
        @test ok
        @test e0 < 1e-7
        @test e1 < 1e-7
        # The output has nt + 1 real time rows; den0 + den1 = 1 up to the
        # self-consistency threshold
        lines = filter(l -> !startswith(l, "#"), readlines(joinpath(dir, "fkm_bethe_quench_density.dat")))
        @test length(lines) == r.config.contour["nt"] + 1
        @test maximum(abs(d0 + d1 - 1) for (n, d0, d1) in r.densities) < 1e-6
        # A tampered den1 is detected
        bad = KbrefData(d.path, d.metadata, [x.component == "den1" && x.i == 5 ?
                                     KbrefRow(x.component, x.i, x.j, x.a, x.b, x.value + 1e-8) : x
                                     for x in d.rows])
        @test !compare_fkm(r.densities, bad)[1]
    end
end
