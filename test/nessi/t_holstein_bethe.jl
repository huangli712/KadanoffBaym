#
# t_holstein_bethe.jl
#
# Holstein examples on the Bethe lattice (tutor/holstein_bethe), normal
# state and Nambu form, Migdal and uMig, against the NESSi reference
# drivers .nessi/kbref/holstein_{bethe,nambu}_text.cpp (same algorithm,
# atol = rtol = 1e-10), quick profile.
#

isdefined(Main, :hol_compare) || include(joinpath(@__DIR__, "t_holstein_common.jl"))

module HolsteinBethe
    include(joinpath(@__DIR__, "..", "..", "tutor", "holstein_bethe", "holstein_bethe_impl.jl"))
end

@testset verbose = true "Phase 5: Holstein Bethe" begin
    for approx in ("Migdal", "uMig")
        @testset "normal $approx" begin
            p = hol_setup("holstein_bethe", "Holstein_bethe_$approx.toml")
            r = HolsteinBethe.holstein_bethe_main([p], approx; io = devnull)
            @test isfile(joinpath(dirname(p), "Holstein_bethe_$(approx)_obs.dat"))
            d = read_kbref(joinpath(HOL_REFDIR, "holstein_bethe_$(lowercase(approx))_quick.dat"))
            @test d.metadata["approximation"] == approx
            sc = Dict("Xph" => r.Xph, "Pph" => r.Pph, "XX" => r.XX, "Ekin" => r.Ekin,
                      "Enx_MF" => r.Enx_MF, "Enx_corr" => r.Enx_corr, "Eph_cl" => r.Eph_cl,
                      "Eph_corr" => r.Eph_corr, "Etot" => r.Etot)
            ok, worst = hol_compare(d, sc, (t, a, b) -> r.rho[t+2][a+1,b+1])
            ok || @info "Holstein Bethe $approx vs NESSi" worst
            @test ok
            @test !hol_compare(hol_tamper(d, "Etot", 1e-8), sc, (t, a, b) -> r.rho[t+2][a+1,b+1])[1]
            # Etot identity
            @test maximum(abs.(r.Etot .- (r.Ekin .+ r.Enx_MF .+ r.Enx_corr .+ r.Eph_cl .+
                                          r.Eph_corr))) < 1e-12
        end
    end
    for approx in ("Migdal", "uMig")
        @testset "Nambu $approx" begin
            p = hol_setup("holstein_bethe", "Holstein_bethe_Nambu_$approx.toml")
            r = HolsteinBethe.holstein_nambu_main([p], approx; io = devnull)
            d = read_kbref(joinpath(HOL_REFDIR, "holstein_nambu_$(lowercase(approx))_quick.dat"))
            sc = Dict("XX" => r.XX, "Ekin" => r.Ekin, "Enx_MF" => r.Enx_MF,
                      "Enx_corr" => r.Enx_corr, "Eph" => r.Eph, "Etot" => r.Etot)
            ρ(t, a, b) = r.rho[t+2][a+1,b+1]
            ok, worst = hol_compare(d, sc, ρ)
            ok || @info "Holstein Nambu $approx vs NESSi" worst
            @test ok
            # A tampered anomalous density is detected
            @test !hol_compare(hol_tamper(d, "rho", 1e-8), sc, ρ)[1]
            # Nambu symmetry of the Matsubara density matrix
            @test abs(r.rho[1][1,2] - conj(r.rho[1][2,1])) < 1e-10
            if approx == "Migdal"
                # Superconducting quick parameters
                @test abs(r.rho[end][1,2]) > 1e-6
            else
                # uMig stays normal at the same coupling
                @test abs(r.rho[end][1,2]) < 1e-6
            end
        end
    end
end
