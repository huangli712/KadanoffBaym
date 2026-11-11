#
# t_p5_hubbard.jl
#
# Unit tests of the Hubbard chain examples (tutor/hubbard_chain): the
# Hartree Hamiltonian, the VIE2 residuals of the GW and TPP ladders, the
# energies, the command line, and the configuration errors. The NESSi and
# exact comparisons are in test/t_hubbard_chain.jl.
#

# The example is loaded into its own module (the tutor/common files are
# included by several examples and tests)
module P5HubbardUnits

using Test
using LinearAlgebra
using KadanoffBaym

include(joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain_impl.jl"))

const P5H_TOML = joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain.toml")

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

p5h_ret(G, a, b) = Matrix(G.ret[a, b])

# Maximal deviation of the retarded and lesser components of X (all time
# steps) from the explicit functions les(a,b) (a ≤ b) and ret(a,b) (a ≥ b)
function p5h_dev(X, les, ret)
    e = 0.0
    for n = 1:getntime(X), j = 1:n
        e = max(e, norm(Matrix(X.less[j, n]) - les(j, n)))
        e = max(e, norm(Matrix(X.ret[n, j]) - ret(n, j)))
    end
    return e
end

# Explicit Langreth rules of the bubbles with c = a = b orbital indices
# (NESSi Bubble1/Bubble2), elementwise in the orbital indices, with
# C^R = C^> - C^< and the NESSi hermitian partner convention: the lesser
# component at t ≥ t' (diagonal included) is -A^<(t',t)†, the greater
# component at t ≤ t' is -A^R(t',t)† + A^<(t,t').
#   Bubble1  C(t,t') = i A(t,t') B(t',t)
#   Bubble2  C(t,t') = i A(t,t') B(t,t')
low_les(A, a, b) = -Matrix(A.less[b, a])'                      # a ≥ b
low_gtr(A, a, b) = p5h_ret(A, a, b) + low_les(A, a, b)         # a ≥ b
up_gtr(A, b, a) = -p5h_ret(A, a, b)' + p5h_les(A, b, a)        # b ≤ a
b1_les(A, B, j, n) = im .* p5h_les(A, j, n) .* transpose(low_gtr(B, n, j))
b1_ret(A, B, n, j) = im .* (low_gtr(A, n, j) .* transpose(p5h_les(B, j, n)) .-
                            low_les(A, n, j) .* transpose(up_gtr(B, j, n)))
b2_les(A, B, j, n) = im .* p5h_les(A, j, n) .* p5h_les(B, j, n)
b2_ret(A, B, n, j) = im .* (low_gtr(A, n, j) .* low_gtr(B, n, j) .-
                            low_les(A, n, j) .* low_les(B, n, j))

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
    from_ham!(st.G, 0.1, Hc, st.β, st.h)
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
            Pl = (a, b) -> -b1_les(G, G, a, b)
            Pr = (a, b) -> -b1_ret(G, G, a, b)
            @test p5h_dev(st.P, Pl, Pr) < tol
            # Fixtures: an extra factor i or a flipped sign of P is detected
            Pf = deepcopy(st.P)
            foreach(n -> smul!(Pf, n, im), 1:getntime(G))
            @test p5h_dev(Pf, Pl, Pr) > 1e-3
            foreach(n -> smul!(Pf, n, im), 1:getntime(G))
            @test p5h_dev(Pf, Pl, Pr) > 1e-3
            # Σ_2B = i G(t,t') W(t,t'), W = U P U (U = 1)
            p5h_sigma!(st)
            @test p5h_dev(st.Σ, (a, b) -> U^2 .* b2_les(G, st.W, a, b),
                          (a, b) -> U^2 .* b2_ret(G, st.W, a, b)) < tol
            @test p5h_dev(st.W, (a, b) -> U^2 .* Pl(a, b), (a, b) -> U^2 .* Pr(a, b)) < tol
            # The real-time VIE2 (start-up collocation, retarded time stepping)
            # and the convolution weights differ at the discretization level
            # (< 1e-7 on the quick grid); the solution itself is compared with
            # NESSi pointwise in test/t_hubbard_chain.jl. The wrong sign
            # of the kernel gives a residual of order one.
            tol_vie = 1e-7
            # GW: Σ = i G(t,t') [U χ U](t,t'), (1 - P U) * χ = P
            st = p5h_state(p, "GW")
            p5h_sigma!(st)
            G, χ = st.G, st.S
            @test p5h_dev(st.X, (a, b) -> -b1_les(G, G, a, b), (a, b) -> -b1_ret(G, G, a, b)) < tol
            @test p5h_vie_residual(χ, st.X, st.Ut, -1.0, st) < tol_vie
            @test p5h_vie_residual(χ, st.X, st.Ut, +1.0, st) > 1e-4
            @test p5h_dev(st.Σ, (a, b) -> U^2 .* b2_les(G, χ, a, b),
                          (a, b) -> U^2 .* b2_ret(G, χ, a, b)) < tol
            # TPP: Φ = -i G(t,t') G(t,t'), (1 + Φ U) * T = Φ,
            # Σ = i [U T U](t,t') G(t',t)
            st = p5h_state(p, "TPP")
            p5h_sigma!(st)
            G, T, Φ = st.G, st.S, st.X
            Φl = (a, b) -> -b2_les(G, G, a, b)
            Φr = (a, b) -> -b2_ret(G, G, a, b)
            @test p5h_dev(Φ, Φl, Φr) < tol
            Φf = deepcopy(Φ)
            foreach(n -> smul!(Φf, n, -1.0), 1:getntime(G))
            @test p5h_dev(Φf, Φl, Φr) > 1e-3
            @test p5h_vie_residual(T, Φ, st.Ut, +1.0, st) < tol_vie
            @test p5h_vie_residual(T, Φ, st.Ut, -1.0, st) > 1e-4
            @test p5h_dev(st.Σ, (a, b) -> U^2 .* b1_les(T, G, a, b),
                          (a, b) -> U^2 .* b1_ret(T, G, a, b)) < tol
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
            script = joinpath(@__DIR__, "..", "tutor", "hubbard_chain", "hubbard_chain_2b.jl")
            proj = dirname(Base.active_project())
            cmd = `$jl --startup-file=no --project=$proj $script $bad`
            @test !success(pipeline(cmd; stdout = devnull, stderr = devnull))
        end
    end
end

end # module P5HubbardUnits
