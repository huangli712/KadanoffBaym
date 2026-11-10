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

# Full lesser and greater matrices G^<(t_a,t_b), G^>(t_a,t_b) at the KB
# time steps a, b ≥ 1 (G^> from the stored retarded and lesser components)
p5h_les(G, a, b) = Matrix(G.less[a, b])
p5h_gtr(G, a, b) = a ≥ b ? Matrix(G.ret[a, b]) + p5h_les(G, a, b) : -Matrix(p5h_gtr(G, b, a))'

# Maximal deviation of the retarded and lesser components of X (all time
# steps) from the explicit lesser/greater functions les(a,b), gtr(a,b)
function p5h_dev(X, les, gtr)
    e = 0.0
    for n = 1:getntime(X), j = 1:n
        e = max(e, norm(Matrix(X.less[j, n]) - les(j, n)))
        e = max(e, norm(Matrix(X.ret[n, j]) - (gtr(n, j) - les(n, j))))
    end
    return e
end

# Elementwise products of the explicit bubbles: B1 = i A(t,t') B(t',t)^T,
# B2 = i A(t,t') B(t,t') (NESSi Bubble1/Bubble2 with c = a = b indices)
p5h_b1(A, B) = im .* A .* transpose(B)
p5h_b2(A, B) = im .* A .* B

# Residual of the VIE (1 + s X U) * Y = X on the whole contour (Matsubara
# and all time steps), with an explicit sign s
function p5h_vie_residual(Y, X, Ut, s, st)
    ntime, ntau = getntime(X), getntau(X)
    N = getdims(X)[1]
    C₀ = Cn(ntime, ntau, N, N, (ntime - 1) * st.h, st.β)
    F, Fcc, C = ℱ(C₀, BOSE), ℱ(C₀, BOSE), ℱ(C₀, BOSE)
    for n = 0:ntime
        memcpy!(X, F, n)
        memcpy!(X, Fcc, n)
        smul!(F, Ut, n)
        smul!(Ut, Fcc, n)
        smul!(F, n, s)
        smul!(Fcc, n, s)
    end
    convolution(C, F, Fcc, Y, Y, st.I, st.β, st.h)
    e = maximum(norm(Y.mat[m] + C.mat[m] - X.mat[m]) for m = 1:ntau)
    for n = 1:ntime, j = 1:n
        e = max(e, norm(Matrix(Y.ret[n, j]) + Matrix(C.ret[n, j]) - Matrix(X.ret[n, j])))
        e = max(e, norm(Matrix(Y.less[j, n]) + Matrix(C.less[j, n]) - Matrix(X.less[j, n])))
    end
    return e
end

# State with a non-real Green's function (complex hermitian Hamiltonian)
function p5h_state(p, approx)
    st = hubbard_chain(approx, load_hubbard_config(p, "quick"))
    Hc = ComplexF64[0.2 0.3+0.4im; 0.3-0.4im -0.1]
    green_from_H!(st.G, 0.1, Hc, st.β, st.h)
    return st
end

# Run the self-energy of the approximation on the whole contour
function p5h_sigma!(st)
    update_sigma!(0, st)
    update_sigma_start!(st)
    for n = st.k+2:getntime(st.G)
        update_sigma!(n, st)
    end
    return st
end

@testset verbose = true "Phase 5: Hubbard chain units" begin
    mktempdir() do dir
        p = p5h_config(dir)
        @testset "self-energy components (non-real contour)" begin
            U = 1.0
            tol = 1e-12
            # P = -i G(t,t') G(t',t)
            st = p5h_state(p, "2B")
            G = st.G
            @test maximum(abs(imag(x)) for x in p5h_les(G, 2, 4)) > 1e-3
            for n = 1:getntime(G)
                polarization!(n, G, st.P)
            end
            les = (a, b) -> -p5h_b1(p5h_les(G, a, b), p5h_gtr(G, b, a))
            gtr = (a, b) -> -p5h_b1(p5h_gtr(G, a, b), p5h_les(G, b, a))
            @test p5h_dev(st.P, les, gtr) < tol
            # Fixtures: an extra factor i or a flipped sign of P is detected
            Pi = deepcopy(st.P)
            foreach(n -> smul!(Pi, n, im), 1:getntime(G))
            @test p5h_dev(Pi, les, gtr) > 1e-3
            foreach(n -> smul!(Pi, n, im), 1:getntime(G))
            @test p5h_dev(Pi, les, gtr) > 1e-3
            # Σ_2B = i G(t,t') W(t,t'), W = U P U
            p5h_sigma!(st)
            Wl = (a, b) -> U^2 .* les(a, b)
            Wg = (a, b) -> U^2 .* gtr(a, b)
            @test p5h_dev(st.Σ, (a, b) -> p5h_b2(p5h_les(G, a, b), Wl(a, b)),
                          (a, b) -> p5h_b2(p5h_gtr(G, a, b), Wg(a, b))) < tol
            # GW: Σ = i G(t,t') [U χ U](t,t'), (1 - P U) * χ = P
            st = p5h_state(p, "GW")
            p5h_sigma!(st)
            G, χ = st.G, st.S
            @test p5h_vie_residual(χ, st.X, st.Ut, -1.0, st) < 1e-10
            @test p5h_vie_residual(χ, st.X, st.Ut, +1.0, st) > 1e-4
            @test p5h_dev(st.Σ, (a, b) -> p5h_b2(p5h_les(G, a, b), U^2 .* p5h_les(χ, a, b)),
                          (a, b) -> p5h_b2(p5h_gtr(G, a, b), U^2 .* p5h_gtr(χ, a, b))) < tol
            # TPP: Φ = -i G(t,t') G(t,t'), (1 + Φ U) * T = Φ,
            # Σ = i [U T U](t,t') G(t',t)
            st = p5h_state(p, "TPP")
            p5h_sigma!(st)
            G, T, Φ = st.G, st.S, st.X
            Φl = (a, b) -> -p5h_b2(p5h_les(G, a, b), p5h_les(G, a, b))
            Φg = (a, b) -> -p5h_b2(p5h_gtr(G, a, b), p5h_gtr(G, a, b))
            @test p5h_dev(Φ, Φl, Φg) < tol
            Φf = deepcopy(Φ)
            foreach(n -> smul!(Φf, n, -1.0), 1:getntime(G))
            @test p5h_dev(Φf, Φl, Φg) > 1e-3
            @test p5h_vie_residual(T, Φ, st.Ut, +1.0, st) < 1e-10
            @test p5h_vie_residual(T, Φ, st.Ut, -1.0, st) > 1e-4
            @test p5h_dev(st.Σ, (a, b) -> p5h_b1(U^2 .* p5h_les(T, a, b), p5h_gtr(G, b, a)),
                          (a, b) -> p5h_b1(U^2 .* p5h_gtr(T, a, b), p5h_les(G, b, a))) < tol
        end

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
