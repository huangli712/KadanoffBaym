#
# validate.jl
#
# Holstein example on the Bethe lattice, normal state and Nambu form,
# Migdal and uMig, against the NESSi reference drivers
# .nessi/kbref/holstein_{bethe,nambu}_text.cpp (same algorithm, atol =
# rtol = 1e-10), quick profile. Additional unit tests of the Holstein
# helpers (holstein_nambu_impl.jl, holstein_energy.jl): zero coupling,
# D = D0 for Pi = 0, the phonon displacement, the bare phonon energy, and
# the normal limit of the Nambu self-energies.
#
# Usage: julia --project=. tutor/holstein_bethe/validate.jl
#

isdefined(Main, :hol_compare) ||
    include(joinpath(@__DIR__, "..", "common", "holstein_common.jl"))

module HolsteinBethe
    include(joinpath(@__DIR__, "holstein_bethe_impl.jl"))
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

#
# Unit tests of the Holstein helpers (holstein_nambu_impl.jl,
# holstein_energy.jl): zero coupling, D = D0 for Pi = 0, the phonon
# displacement, the bare phonon energy, and the normal limit of the Nambu
# self-energies.
#
module P5Holstein
    include(joinpath(@__DIR__, "holstein_nambu_impl.jl"))
    include(joinpath(@__DIR__, "holstein_energy.jl"))
end

using .P5Holstein: holstein_phonon, holstein_scalar_cf, sigma_mig!, sigma_umig!,
                   sigma_mig_start!, sigma_umig_start!, sigma_mig_sc!, sigma_umig_sc!,
                   sigma_mig_sc_start!, sigma_umig_sc_start!, step_D!, start_D!,
                   gregory_integrate, phonon_displacement!, phonon_energy_quantum

const P5H = (ntime = 8, ntau = 21, k = 3, β = 4.0, h = 0.05, w0 = 1.0)

p5h_cn(size) = Cn(P5H.ntime, P5H.ntau, size, size, (P5H.ntime - 1) * P5H.h, P5H.β)

# Free Green's function of a level ε (size 1) or of diag(ε, ε) (size 2)
function p5h_green(ε, size)
    G = ℱ(p5h_cn(size), FERMI)
    from_ham!(G, 0.0, Matrix{ComplexF64}(ε * I, size, size), P5H.β, P5H.h)
    return G
end

p5h_phonon(size) = holstein_phonon(P5H.ntime, P5H.ntau, size, P5H.w0, P5H.β, P5H.h, P5H.k)

# Maximal difference of two contour functions over the time steps ns
p5h_dist(A, B, ns) = maximum(distance(A, B, n) for n in ns)

@testset verbose = true "Phase 5: Holstein helpers" begin
    @testset "zero coupling gives zero self-energy" begin
        G = p5h_green(0.3, 1)
        g = holstein_scalar_cf(P5H.ntime, zeros(P5H.ntime + 1))
        for f! in (sigma_mig!, sigma_umig!)
            Σ = ℱ(p5h_cn(1), FERMI)
            ph = p5h_phonon(1)
            f!(0, G, Σ, g, ph)
            (f! === sigma_mig! ? sigma_mig_start! : sigma_umig_start!)(G, Σ, g, ph)
            @test p5h_dist(Σ, ℱ(p5h_cn(1), FERMI), 0:P5H.k+1) == 0.0
        end
    end

    @testset "D = D0 for Pi = 0" begin
        ph = p5h_phonon(1)
        step_D!(0, ph)
        start_D!(ph)
        @test p5h_dist(ph.D, ph.D0, 0:P5H.k+1) < 1e-10
        step_D!(P5H.k + 2, ph)
        @test distance(ph.D, ph.D0, P5H.k + 2) < 1e-10
    end

    @testset "phonon Dyson equation with a nonzero kernel" begin
        # The real-time VIE2 (start-up collocation, retarded time stepping)
        # and the convolution weights agree up to the discretization error,
        # O(h^(k+1)); with h = 0.01 the residual is below 1e-10 (8e-10 for
        # h = 0.05)
        h = 0.01
        ph = holstein_phonon(P5H.ntime, P5H.ntau, 1, P5H.w0, P5H.β, h, P5H.k)
        # Small bosonic Pi: 0.05 times a single-pole XX propagator
        green_single_pole!(ph.Pi, 1.5, P5H.β, h)
        foreach(n -> smul!(ph.Pi, n, 0.05), 0:P5H.ntime)
        step_D!(0, ph)
        start_D!(ph)
        foreach(n -> step_D!(n, ph), P5H.k+2:P5H.ntime)
        # Residual of D - D0 * Pi * D = D0 (D0_Pi = -D0 * Pi, Pi_D0 = -Pi * D0)
        C = ℱ(Cn(P5H.ntime, P5H.ntau, 1, 1, (P5H.ntime - 1) * h, P5H.β), BOSE)
        convolution(C, ph.D0_Pi, ph.Pi_D0, ph.D, ph.D, ph.I, P5H.β, h)
        rm = maximum(abs(ph.D.mat[m][1,1] + C.mat[m][1,1] - ph.D0.mat[m][1,1]) for m = 1:P5H.ntau)
        rt = 0.0
        for n = 1:P5H.ntime, j = 1:n
            rt = max(rt, abs(ph.D.ret[n, j][1,1] + C.ret[n, j][1,1] - ph.D0.ret[n, j][1,1]))
            rt = max(rt, abs(ph.D.less[j, n][1,1] + C.less[j, n][1,1] - ph.D0.less[j, n][1,1]))
        end
        @test rm < 1e-10
        @test rt < 1e-10
        # D differs from D0
        @test p5h_dist(ph.D, ph.D0, 0:P5H.ntime) > 1e-3
    end

    @testset "Gregory integration and phonon displacement" begin
        ph = p5h_phonon(1)
        # Polynomials of degree ≤ k are integrated exactly
        for n = 1:7
            f = [ComplexF64((j * P5H.h)^3 - 2(j * P5H.h)) for j = 0:max(n, P5H.k)]
            @test abs(gregory_integrate(ph.I, f, n) * P5H.h -
                      ((n * P5H.h)^4 / 4 - (n * P5H.h)^2)) < 1e-12
        end
        # Constant density: constant displacement X0 = -2 g n / w0
        nt = P5H.ntime
        ntot = fill(ComplexF64(0.8), nt + 1)
        gv = fill(ComplexF64(0.5), nt + 1)
        X = zeros(ComplexF64, nt + 1)
        # The direct formula is used at n = 0 and n > k (NESSi tstp ≥ k),
        # the start-up version at n = 1, ⋯, k + 1
        steps = [0; P5H.k+1:nt]
        for n in steps
            phonon_displacement!(n, X, ntot, gv, ph)
        end
        P5Holstein.phonon_displacement_start!(X, ntot, gv, ph)
        @test maximum(abs.(X .+ 2 * 0.5 * 0.8 / P5H.w0)) < 1e-14
        # Linear density: direct quadrature of D0^R(t,t') [g n(t') - g n(0)]
        ntot = ComplexF64[0.8 + 0.1 * max(n - 1, 0) * P5H.h for n = 0:nt]
        for n in steps
            phonon_displacement!(n, X, ntot, gv, ph)
        end
        for n = P5H.k+1:nt
            f = [ph.D0.ret[n, j+1][1,1] * (gv[j+2] * ntot[j+2] - gv[1] * ntot[1]) for j = 0:n-1]
            s = sum(ph.I.GIW[n-1, j] * f[j+1] for j = 0:n-1)
            @test abs(X[n+1] - (s * P5H.h - 2 * 0.5 * 0.8 / P5H.w0)) < 1e-12
        end
    end

    @testset "bare phonon energy" begin
        ph = p5h_phonon(1)
        Pi = ℱ(p5h_cn(1), BOSE)
        E = phonon_energy_quantum(ph.D0, Pi, ph)
        @test maximum(abs.(E .- P5H.w0 * (bose(P5H.β, P5H.w0) + 0.5))) < 1e-10
    end

    @testset "Nambu self-energies in the normal limit" begin
        G1 = p5h_green(0.3, 1)
        G2 = p5h_green(0.3, 2)
        g1 = holstein_scalar_cf(P5H.ntime, fill(0.4, P5H.ntime + 1))
        g2 = Cf(P5H.ntime, 2, 2, zero(ComplexF64))
        for n = 0:P5H.ntime
            g2[n] = ComplexF64[0.4 0; 0 -0.4]
        end
        for (fn!, fs!, fns!, fss!) in ((sigma_umig!, sigma_umig_sc!, sigma_umig_start!, sigma_umig_sc_start!),
                                        (sigma_mig!, sigma_mig_sc!, sigma_mig_start!, sigma_mig_sc_start!))
            Σ1 = ℱ(p5h_cn(1), FERMI)
            Σ2 = ℱ(p5h_cn(2), FERMI)
            ph1, ph2 = p5h_phonon(1), p5h_phonon(2)
            fn!(0, G1, Σ1, g1, ph1)
            fs!(0, G2, Σ2, g2, ph2)
            fns!(G1, Σ1, g1, ph1)
            fss!(G2, Σ2, g2, ph2)
            err = 0.0
            for n = 0:P5H.k+1, a = 1:2
                if n == 0
                    for m = 1:P5H.ntau
                        err = max(err, abs(Σ2.mat[m][a,a] - Σ1.mat[m][1,1]))
                    end
                else
                    for j = 1:n
                        err = max(err, abs(Σ2.ret[n, j][a,a] - Σ1.ret[n, j][1,1]))
                        err = max(err, abs(Σ2.less[j, n][a,a] - Σ1.less[j, n][1,1]))
                    end
                    for m = 1:P5H.ntau
                        err = max(err, abs(Σ2.lmix[n, m][a,a] - Σ1.lmix[n, m][1,1]))
                    end
                end
            end
            @test err < 1e-12
            # No anomalous self-energy without an anomalous Green's function
            @test maximum(abs(Σ2.mat[m][1,2]) for m = 1:P5H.ntau) < 1e-14
        end
    end
end
