#
# t_gw.jl
#
# Regression of the GW example tutor/gw (NESSi examples/programs/gw.cpp):
# lattice helpers, Hartree/Fock self-energies, the end-to-end run with the
# quick parameters (energy and particle number conservation), and the
# independence of the results from the parallel axis.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

module GWExample
include(joinpath(@__DIR__, "..", "tutor", "gw", "gw.jl"))
end

# Write a small TOML file of the GW example and return its path
function gw_toml(dir::String; Nk = 4, Nt = 7, Ntau = 31, axis = "serial",
                 mode = "peierls_corrected")
    file = joinpath(dir, "gw_$(axis)_$(mode).toml")
    open(file, "w") do f
        print(f, """
        [model]
        Nk = $Nk
        HoppingT = 1.0
        HubbardU = 1.0
        V = 0.25
        MuChem = 0.0
        beta = 5.0

        [contour]
        Nt = $Nt
        Ntau = $Ntau
        h = 0.02

        [solver]
        SolveOrder = 5
        MatsMaxIter = 60
        MatsMaxErr = 1e-9
        BootstrapMaxIter = 40
        BootstrapMaxErr = 1e-10
        TimeMaxErr = 1e-10
        CorrectorSteps = 6
        parallel_axis = "$axis"
        current_mode = "$mode"

        [pulse]
        E0 = 1.0
        omega = 2.0
        t0 = 0.05
        width = 0.1

        [output]
        dir = "$(escape_string(dir))"
        prefix = "gw_$(axis)"
        """)
    end
    return file
end

@testset verbose = true "KadanoffBaym: GW example" begin
    G = GWExample

    @testset "lattice" begin
        nk = 6
        gk, kp, kw = G.build_kgrid(nk)
        @test gk == 3
        @test kp[gk+1] == 0.0
        @test sum(kw) ≈ 1.0
        lat = G.GWLattice(nk, 1.0, 1.0, 0.25, zeros(8), 0.02, 3)
        # k1 - k2 folded back to the Brillouin zone
        for k1 = 1:nk, k2 = 1:nk
            j = G.add_kpoints(lat, k1, 1, k2, -1)
            dk = kp[k1] - kp[k2] - kp[j]
            @test abs(sin(dk / 2)) < 1e-12
        end
        # A(t) = -∫E for a constant field E = 1 is -t (exact for every order)
        E = ones(10)
        for k = 1:5
            A = G.efield_to_afield(E, 0.1, k)
            @test A[1] == 0.0
            @test maximum(abs(A[n+1] + (n - 1) * 0.1) for n = 1:9) < 1e-12
        end
        @test G.interaction_vertex(lat, 0.0) ≈ 1.5
        @test G.dispersion(lat, 0.0, 0) ≈ -2.0
    end

    @testset "Hartree and Fock" begin
        nk = 4
        lat = G.GWLattice(nk, 1.0, 1.0, 0.25, zeros(8), 0.02, 3)
        dens = [fill(ComplexF64(0.1j), 1, 1) for j = 1:nk]
        n̄ = sum(lat.kweights[j] * 0.1j for j = 1:nk)
        # Σ^H = U n̄ + 2 (V_0 - U) n̄
        @test G.sigma_hartree(lat, dens)[1, 1] ≈ lat.U * n̄ + 2 * 2lat.V * n̄
        for k = 1:nk
            ref = -sum(lat.kweights[q] * (G.interaction_vertex(lat, lat.kpoints[q]) - lat.U) *
                       0.1 * G.add_kpoints(lat, k, 1, q, -1) for q = 1:nk)
            @test G.sigma_fock(k, lat, dens)[1, 1] ≈ ref
        end
    end

    mktempdir() do dir
        res = G.main(gw_toml(dir); io = devnull)
        rows = res.observables

        @testset "end to end" begin
            @test all(r -> isfinite(r.Etot) && isfinite(r.density), rows)
            # The Matsubara iterations and the bootstrapping converged
            @test any(e -> e[1] == 0 && e[4] < 1e-9, res.errors)
            @test any(e -> e[1] == 1 && e[4] < 1e-10, res.errors)
            # Particle number and (for a short pulse) total energy
            dens = [r.density for r in rows]
            @test maximum(abs.(dens .- dens[1])) < 1e-7
            # The default current column ignores A, the corrected one not
            @test maximum(abs(r.current_nessi) for r in rows) < 1e-5
            @test maximum(abs(r.current_peierls_corrected) for r in rows) > 1e-4
            @test isfile(joinpath(dir, "gw_serial_observables.dat"))
        end

        @testset "parallel axis does not change the results" begin
            # The threaded axes run with one BLAS thread, the serial one with
            # the default BLAS threads, so they agree up to rounding
            for axis in ("k", "kernel")
                r2 = G.main(gw_toml(dir; axis = axis); io = devnull).observables
                @test all(i -> abs(rows[i].Etot - r2[i].Etot) < 1e-12 &&
                               abs(rows[i].density - r2[i].density) < 1e-12 &&
                               abs(rows[i].current_peierls_corrected -
                                   r2[i].current_peierls_corrected) < 1e-12, eachindex(rows))
            end
        end
    end
end
