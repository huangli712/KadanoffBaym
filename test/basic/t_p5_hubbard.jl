#
# t_p5_hubbard.jl
#
# Unit tests of the Hubbard chain examples (tutor/hubbard_chain): the
# Hartree Hamiltonian, the VIE2 residuals of the GW and TPP ladders, the
# energies, the command line, and the configuration errors. The NESSi and
# exact comparisons are in test/nessi/t_hubbard_chain.jl.
#

using Test
using LinearAlgebra
using KadanoffBaym

isdefined(Main, :hubbard_main) ||
    include(joinpath(@__DIR__, "..", "..", "tutor", "hubbard_chain", "hubbard_chain_impl.jl"))

const P5H_TOML = joinpath(@__DIR__, "..", "..", "tutor", "hubbard_chain", "hubbard_chain.toml")

function p5h_config(dir; edit = identity)
    p = joinpath(dir, "hubbard_chain.toml")
    write(p, edit(replace(read(P5H_TOML, String),
                          "dir = \".\"" => "dir = \"$(escape_string(dir))\"")))
    return p
end

# Residual of the Matsubara VIE2 Chi + XU * Chi = X (the real-time
# ladders are checked pointwise against NESSi in t_hubbard_chain.jl)
function p5h_vie2_residual(st)
    C = deepcopy(st.S)
    convolution(C, st.XU, st.UX, st.S, st.S, st.I, st.β, st.h)
    return maximum(norm(st.S.mat[m] + C.mat[m] - st.X.mat[m]) for m = 1:getntau(st.G))
end

@testset verbose = true "Phase 5: Hubbard chain units" begin
    mktempdir() do dir
        p = p5h_config(dir)
        @testset "Hartree Hamiltonian and energies" begin
            cfg = load_hubbard_config(p, "quick")
            st = hubbard_chain("2B", cfg)
            # μ is shifted by U n / Nsites
            ρ = density_matrix(st.G, 0)
            @test st.μ ≈ cfg.model["MuChem"] + cfg.model["HubbardU"] * real(tr(ρ)) / 2 atol = 1e-14
            hartree_hamiltonian!(1, st)
            H = st.eps_mf[1]
            @test H == H'
            ρ = density_matrix(st.G, 1)
            @test H ≈ st.eps0 + Diagonal(cfg.model["HubbardU"] .* real.(diag(ρ))) atol = 1e-14
            ekin, epot, etot = hubbard_energies(1, st)
            @test ekin ≈ real(tr(st.eps0 * ρ)) atol = 1e-14
            @test etot ≈ ekin + epot atol = 1e-14
        end

        @testset "VIE2 residuals ($a)" for a in ("GW", "TPP")
            r = hubbard_main([p], a; io = devnull)
            @test p5h_vie2_residual(r.state) < 1e-10
            # Observables are finite; the particle number of the chain is
            # conserved up to the discretization error of the coarse quick
            # grid (1.6e-3, identical in the NESSi reference driver)
            n = [sum(o[2:end]) for o in r.occupation]
            @test all(isfinite, n)
            @test maximum(abs.(n .- n[1])) < 5e-3
        end

        @testset "command line and configuration" begin
            # --validate-only does not propagate and writes nothing
            out = joinpath(dir, "v")
            mkpath(out)
            pv = p5h_config(out)
            r = hubbard_main([pv, "--validate-only"], "2B"; io = devnull)
            @test !haskey(r, :occupation)
            @test !any(endswith(".dat"), readdir(out))
            @test_throws ArgumentError hubbard_main([pv, "--profile", "fast"], "2B"; io = devnull)
            @test_throws ArgumentError hubbard_main([pv], "RPA"; io = devnull)
            bad = p5h_config(joinpath(mkpath(joinpath(dir, "b"))); edit = s -> replace(s, "RampSite = 1" => "RampSite = 3"))
            @test_throws ArgumentError hubbard_main([bad], "2B"; io = devnull)
            bad = p5h_config(joinpath(mkpath(joinpath(dir, "c"))); edit = s -> replace(s, "HubbardU = 1.0\n" => ""))
            @test_throws ArgumentError hubbard_main([bad], "2B"; io = devnull)
            # A Matsubara iteration that cannot converge
            bad = p5h_config(joinpath(mkpath(joinpath(dir, "d")));
                             edit = s -> replace(s, "mats_max_iter = 100, bootstrap" => "mats_max_iter = 1, bootstrap"))
            @test_throws ConvergenceError hubbard_main([bad], "2B"; io = devnull)
            # The entry scripts exit with a nonzero code on errors
            jl = Base.julia_cmd()
            script = joinpath(@__DIR__, "..", "..", "tutor", "hubbard_chain", "hubbard_chain_2b.jl")
            proj = dirname(Base.active_project())
            cmd = `$jl --startup-file=no --project=$proj $script $bad`
            @test !success(pipeline(cmd; stdout = devnull, stderr = devnull))
        end
    end
end
