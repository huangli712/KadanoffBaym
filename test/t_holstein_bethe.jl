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
    include(joinpath(@__DIR__, "..", "tutor", "holstein_bethe", "holstein_bethe_impl.jl"))
end

@testset verbose = true "Phase 5: Holstein Bethe" begin
    for approx in ("Migdal", "uMig")
        @testset "normal $approx" begin
            p = hol_setup("holstein_bethe", "holstein_bethe_$(lowercase(approx)).toml")
            r = HolsteinBethe.holstein_bethe_main([p], approx; io = devnull)
            @test isfile(joinpath(dirname(p), "holstein_bethe_$(lowercase(approx))_obs.dat"))
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
            if approx == "uMig"
                # XX of the bare phonon: 2 D_XX of the single pole (analytic)
                c = r.config.contour
                D = ℱ(Cn(c["nt"] + 1, c["ntau"] + 1, 1, 1, c["nt"] * c["h"], c["beta"]), BOSE)
                green_single_pole!(D, Float64(r.config.model["Phfreq_w0"]), Float64(c["beta"]),
                                      Float64(c["h"]))
                @test maximum(abs(r.XX[n+1] - 2im * D.less[n, n][1,1])
                              for n = 1:c["nt"]+1) < 1e-10
            end
        end
    end
    @testset "normal Migdal without drive: energy conservation" begin
        # The quick drive covers the whole quick time window, so this fixture
        # switches it off; ntau = 200 (the drift on the quick ntau = 40 grid
        # is 7.5e-5, dominated by the imaginary-time discretization of the
        # initial correlations)
        p = hol_setup("holstein_bethe", "holstein_bethe_migdal.toml")
        write(p, replace(read(p, String),
                         "[drive.dHopping]\nmode = \"sin2\"\namplitude = 1.0\nend = 0.6" =>
                         "[drive.dHopping]\nmode = \"constant\"\nvalue = 0.0",
                         "[profile.quick]\n" => "[profile.quick]\ncontour = { ntau = 200 }\n"))
        r = HolsteinBethe.holstein_bethe_main([p], "Migdal"; io = devnull)
        Et = real.(r.Etot[2:end])
        @test maximum(Et) - minimum(Et) < 1e-5
    end
    @testset "Nambu weak coupling (normal state)" begin
        res = Dict{String,Any}()
        for approx in ("Migdal", "uMig")
            p = hol_setup("holstein_bethe", "holstein_bethe_nambu_$(lowercase(approx)).toml")
            write(p, replace(read(p, String), "model = { El_Ph_g = 1.0 }" => "model = { El_Ph_g = 0.02 }"))
            res[approx] = HolsteinBethe.holstein_nambu_main([p], approx; io = devnull)
        end
        m, u = res["Migdal"], res["uMig"]
        # Normal state: no anomalous density
        @test maximum(abs(ρ[1,2]) for ρ in m.rho) < 1e-8
        # Nambu symmetry rho_12 = conj(rho_21)
        @test maximum(abs(ρ[1,2] - conj(ρ[2,1])) for ρ in m.rho) < 1e-10
        # The vertex corrections of the phonon are of higher order in g
        @test maximum(norm(m.rho[n] - u.rho[n]) for n in eachindex(m.rho)) < 1e-4
        @test maximum(abs.(m.Etot .- u.Etot)) < 1e-4
    end
    for approx in ("Migdal", "uMig")
        @testset "Nambu $approx" begin
            p = hol_setup("holstein_bethe", "holstein_bethe_nambu_$(lowercase(approx)).toml")
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
