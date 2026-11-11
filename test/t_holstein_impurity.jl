#
# t_holstein_impurity.jl
#
# Holstein impurity with a single bath site (tutor/holstein_impurity),
# Migdal and uMig, against the NESSi reference driver
# .nessi/kbref/holstein_impurity_text.cpp (same algorithm, atol = rtol =
# 1e-10), quick profile. The density uses `rho` with a = b = site (0
# impurity, 1 bath site).
#

isdefined(Main, :hol_compare) || include(joinpath(@__DIR__, "t_holstein_common.jl"))

module HolsteinImpurity
    include(joinpath(@__DIR__, "..", "tutor", "holstein_impurity", "holstein_impurity_impl.jl"))
end

@testset verbose = true "Phase 5: Holstein impurity" begin
    for approx in ("Migdal", "uMig")
        @testset "$approx" begin
            p = hol_setup("holstein_impurity", "holstein_impurity_singlebath_$(lowercase(approx)).toml")
            r = HolsteinImpurity.holstein_impurity_main([p], approx; io = devnull)
            out = joinpath(dirname(p), "holstein_impurity_singlebath_$(lowercase(approx))_obs.dat")
            @test isfile(out)
            # 14 columns, no NaN
            lines = filter(l -> !startswith(l, "#"), readlines(out))
            @test all(length(split(l)) == 14 for l in lines)
            @test all(isfinite, parse.(Float64, reduce(vcat, split.(lines))))
            d = read_kbref(joinpath(HOL_REFDIR, "holstein_impurity_$(lowercase(approx))_quick.dat"))
            sc = Dict("Xph" => r.Xph, "Pph" => r.Pph, "XX" => r.XX, "Ekin" => r.Ekin,
                      "Enx_MF" => r.Enx_MF, "Enx_corr" => r.Enx_corr, "Eph_cl" => r.Eph_cl,
                      "Eph_corr" => r.Eph_corr, "Etot" => r.Etot)
            ρ(t, a, b) = a == 0 ? r.rho00[t+2] : r.rho11[t+2]
            ok, worst = hol_compare(d, sc, ρ)
            ok || @info "Holstein impurity $approx vs NESSi" worst
            @test ok
            # A tampered Eph_corr is detected
            @test !hol_compare(hol_tamper(d, "Eph_corr", 1e-8), sc, ρ)[1]
            # Occupations in [0, 2] (two spins)
            @test all(0 ≤ 2real(x) ≤ 2 for x in r.rho00)
            @test all(0 ≤ 2real(x) ≤ 2 for x in r.rho11)
            # Etot identity
            @test maximum(abs.(r.Etot .- (r.Ekin .+ r.Enx_MF .+ r.Enx_corr .+ r.Eph_cl .+
                                          r.Eph_corr))) < 1e-12
        end
    end
end
