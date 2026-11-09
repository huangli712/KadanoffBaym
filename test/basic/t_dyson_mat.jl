#
# t_dyson_mat.jl
#
# To test the Matsubara Dyson solver dyson_mat! and the private
# corrected-DFT core in src/matsubara_fourier_core.jl.
#
# NESSi Catch tests ported here:
#   libcntr/test/matsubara.cpp:34-104, sections Fixpoint and Conjugate
#   gradient (the section Dyson uses CNTR_MAT_FOURIER, which belongs to
#   Phase 7)
#

using Test
using LinearAlgebra
using KadanoffBaym

const KB = KadanoffBaym

#
# Matsubara Green's function of the free hermitian Hamiltonian H
#
function dm_free_mat(H::Matrix{ComplexF64}, μ, β, ntau, sign)
    d = size(H, 1)
    Hf = Cf(1, d, d)
    Hf[0] = H
    G = KB.Gᵐᵃᵗ(ntau, d, d)
    KB._free_matsubara_from_H!(G, μ, Hf, β, sign)
    return G
end

#
# Embedding model: the (a,a) block of the free Green's function of
#     Hfull = [Ha V; V' Hb]
# solves the Dyson equation for Ha with Σ = V gb V'.
#
function dm_embedding(Ha, Hb, V, μ, β, ntau, sign)
    da, db = size(Ha, 1), size(Hb, 1)
    Hfull = [Ha V; V' Hb]
    Gfull = dm_free_mat(Hfull, μ, β, ntau, sign)
    gb = dm_free_mat(Hb, μ, β, ntau, sign)
    #
    C = Cn(2, ntau, da, da, 1.0, β)
    Σ = ℱ(C, sign)
    G_exact = ℱ(C, sign)
    for m = 1:ntau
        Σ.mat[m] = V * gb.data[:, :, m] * V'
        G_exact.mat[m] = Gfull.data[1:da, 1:da, m]
    end
    H = Cf(2, da, da)
    H[0] = Ha
    return C, Σ, G_exact, H
end

@testset "dyson_mat: corrected DFT core" begin
    #
    # Boundary terms at θ = 0
    corfac, endcor = KB._dftcorr_cubic(0.0)
    @test corfac == 1.0
    @test endcor[1] == complex(-2.0 / 3.0, 0.0)
    @test endcor[2] == complex(7.0 / 24.0, 0.0)
    @test endcor[3] == complex(-1.0 / 6.0, 0.0)
    @test endcor[4] == complex(1.0 / 24.0, 0.0)
    #
    # Continuity of the series and trigonometric branches at |θ| = 0.05
    for s in (1.0, -1.0)
        c₁, e₁ = KB._dftcorr_cubic(s * (0.05 - 1e-12))
        c₂, e₂ = KB._dftcorr_cubic(s * (0.05 + 1e-12))
        @test abs(c₁ - c₂) < 1e-9
        @test maximum(abs.(collect(e₁) .- collect(e₂))) < 1e-9
    end
    #
    # Fourier transform of A(τ) = exp(-τK), K is a non-hermitian and
    # non-normal 2 × 2 matrix:
    #     ∫₀ᵝ dτ exp(iωτ) A(τ) = (iω - K)⁻¹ (ξ exp(-βK) - 1).
    K = ComplexF64[0.7 0.3+0.2im; -0.1+0.4im -0.5]
    β = 4.0
    for sign in (FERMI, BOSE), N in (64, 128)
        ntau = N + 1
        A = KB.Gᵐᵃᵗ(ntau, 2, 2)
        for m = 1:ntau
            A[m] = exp(-(m - 1) * (β / N) * K)
        end
        mdft = KB._matsubara_dft(A, sign)
        one = sign == FERMI ? 1 : 0
        err = 0.0
        for n in (-3N - 1, -N, -2, 0, 1, 7, N ÷ 2, 2N + 3)
            ω = (2n + one) * π / β
            exact = (im * ω * I - K) \ (sign * exp(-β * K) - I)
            num = KB._matsubara_ft_direct(n, A, mdft, sign, β, 3)
            err = max(err, norm(num - exact))
        end
        # Fourth order corrected DFT
        @test err < (N == 64 ? 2e-6 : 2e-7)
        #
        # Without the boundary correction the error is much larger
        num0 = KB._matsubara_ft_direct(1, A, mdft, sign, β, 0)
        ω₁ = (2 + one) * π / β
        exact₁ = (im * ω₁ * I - K) \ (sign * exp(-β * K) - I)
        @test norm(num0 - exact₁) > 1e-3
        @test_throws ArgumentError KB._matsubara_ft_direct(1, A, mdft, sign, β, 2)
    end
end

@testset "dyson_mat: matsubara seed" begin
    β = 5.0
    ntau = 201
    for sign in (FERMI, BOSE)
        μ = sign == FERMI ? 0.0 : -2.0
        Ha = ComplexF64[-1.0 0.3im; -0.3im 0.5]
        Hb = ComplexF64[1.0 0.2; 0.2 1.5]
        V = ComplexF64[0.4 0.1im; -0.2 0.3]
        C, Σ, G_exact, H = dm_embedding(Ha, Hb, V, μ, β, ntau, sign)
        I5 = Integrator(5)
        G0, F = KB._dyson_mat_kernel(G_exact, Σ, μ, H, β, I5)
        #
        # (1 + F) X = G0 is the Matsubara Dyson equation. The seed is an
        # approximation limited by the truncated frequency sum.
        X = KB.Gᵐᵃᵗ(ntau, 2, 2)
        KB._matsubara_seed!(X, F, G0, β, sign; pcf = 5)
        @test maximum(abs.(X.data .- G_exact.mat.data)) < 2e-3
        #
        # The ℱ interface gives the same result
        Xf = ℱ(C, sign)
        Ff = ℱ(C, sign)
        Bf = ℱ(C, sign)
        Ff.mat.data .= F.data
        Bf.mat.data .= G0.data
        KB._matsubara_seed!(Xf, Ff, Bf, β, I5; pcf = 5)
        @test Xf.mat.data == X.data
        #
        # F = 0: the seed reproduces B
        Z = KB.Gᵐᵃᵗ(ntau, 2, 2)
        KB._matsubara_seed!(X, Z, G0, β, sign; pcf = 5)
        @test maximum(abs.(X.data .- G0.data)) < 1e-3
        #
        # Error paths
        Xodd = KB.Gᵐᵃᵗ(ntau + 1, 2, 2)
        @test_throws ArgumentError KB._matsubara_seed!(Xodd, KB.Gᵐᵃᵗ(ntau + 1, 2, 2),
                                                       KB.Gᵐᵃᵗ(ntau + 1, 2, 2), β, sign; pcf = 5)
        @test_throws DimensionMismatch KB._matsubara_seed!(Xodd, F, G0, β, sign; pcf = 5)
        @test_throws ArgumentError KB._matsubara_seed!(X, F, G0, β, 0; pcf = 5)
        @test_throws DomainError KB._matsubara_seed!(X, F, KB.Gᵐᵃᵗ(ntau, 2, 2, NaN + 0.0im),
                                                     β, sign; pcf = 5)
    end
end

@testset "dyson_mat: NESSi Catch (matsubara.cpp)" begin
    eps1, eps2, lam = -1.0, 1.0, 0.2
    μ, β, ntau = 0.0, 20.0, 401
    h2x2 = ComplexF64[eps1 im*lam; -im*lam eps2]
    G2x2 = dm_free_mat(h2x2, μ, β, ntau, FERMI)
    g22 = dm_free_mat(fill(ComplexF64(eps2), 1, 1), μ, β, ntau, FERMI)
    C = Cn(2, ntau, 1, 1, 1.0, β)
    G_exact = ℱ(C, FERMI)
    Σ = ℱ(C, FERMI)
    for m = 1:ntau
        G_exact.mat[m] = G2x2.data[1:1, 1:1, m]
        Σ.mat[m] = lam^2 .* g22.data[:, :, m]
    end
    H = Cf(2, 1, 1)
    H[0] = fill(ComplexF64(eps1), 1, 1)
    #
    for method in (:fixpoint, :cg)
        G = ℱ(C, FERMI)
        dyson_mat!(G, Σ, μ, H, β; solve_order = 5, method = method)
        @test distance(G_exact, G, 0) < 1e-7
    end
end

@testset "dyson_mat: embedding model (sign = $sign)" for sign in (FERMI, BOSE)
    β = 5.0
    ntau = 161
    μ = sign == FERMI ? 0.1 : -2.0
    # Non-commuting 2 × 2 blocks, complex couplings
    Ha = ComplexF64[-1.0 0.3im; -0.3im 0.5]
    Hb = ComplexF64[1.0 0.2+0.1im; 0.2-0.1im 1.5]
    V = ComplexF64[0.4 0.1im; -0.2 0.3]
    C, Σ, G_exact, H = dm_embedding(Ha, Hb, V, μ, β, ntau, sign)
    #
    Gs = Dict{Symbol,Any}()
    for method in (:fixpoint, :cg)
        G = ℱ(C, sign)
        dyson_mat!(G, Σ, μ, H, β; solve_order = 5, method = method)
        @test distance(G, G_exact, 0) / (ntau - 1) < 1e-7
        # Hermitian Matsubara component
        for m = 1:ntau
            @test norm(G.mat[m] - G.mat[m]') < 1e-14
        end
        Gs[method] = G
    end
    @test maximum(abs.(Gs[:fixpoint].mat.data .- Gs[:cg].mat.data)) < 1e-9
    #
    # The accuracy depends on the integration order
    G1 = ℱ(C, sign)
    dyson_mat!(G1, Σ, μ, H, β; solve_order = 1)
    @test distance(G1, G_exact, 0) > 100 * distance(Gs[:fixpoint], G_exact, 0)
    #
    # Without the symmetrization the result is still hermitian up to the
    # discretization error
    Gn = ℱ(C, sign)
    dyson_mat!(Gn, Σ, μ, H, β; force_hermitian = false)
    @test maximum(abs.(Gn.mat.data .- Gs[:fixpoint].mat.data)) < 1e-8
end

@testset "dyson_mat: iterations" begin
    β, ntau, μ = 5.0, 81, 0.0
    Ha = ComplexF64[-1.0 0.3im; -0.3im 0.5]
    Hb = ComplexF64[1.0 0.2; 0.2 1.5]
    V = ComplexF64[0.4 0.1im; -0.2 0.3]
    C, Σ, G_exact, H = dm_embedding(Ha, Hb, V, μ, β, ntau, FERMI)
    I5 = Integrator(5)
    #
    # Fixpoint: maxiter = 0 is the seed, the corrections converge to the
    # discrete solution
    Gref = ℱ(C, FERMI)
    KB._dyson_mat_fixpoint!(Gref, Σ, μ, H, β, I5; maxiter = 30)
    errs = Float64[]
    for n = 0:6
        G = ℱ(C, FERMI)
        KB._dyson_mat_fixpoint!(G, Σ, μ, H, β, I5; maxiter = n)
        push!(errs, distance(G, Gref, 0))
    end
    @test errs[2] < errs[1]
    @test errs[end] < 1e-10
    #
    # CG: no iteration
    G = ℱ(C, FERMI)
    @test KB._dyson_mat_cg!(G, Σ, μ, H, β, I5; maxiter = 0) == 0
    G = ℱ(C, FERMI)
    @test KB._dyson_mat_cg!(G, Σ, μ, H, β, I5; tol = 1e3) == 0
    # CG: the iteration limit is exhausted
    G = ℱ(C, FERMI)
    @test KB._dyson_mat_cg!(G, Σ, μ, H, β, I5; maxiter = 2, tol = 0.0) == 2
    # CG: stop inside the loop
    G = ℱ(C, FERMI)
    niter = KB._dyson_mat_cg!(G, Σ, μ, H, β, I5; tol = 1e-8)
    @test 1 ≤ niter < 40
    @test distance(G, Gref, 0) < 1e-6
    # CG: the default parameters. CG solves the hermitized equation, its
    # discrete solution differs from the fixpoint one at the level of the
    # discretization error.
    G = ℱ(C, FERMI)
    KB._dyson_mat_cg!(G, Σ, μ, H, β, I5)
    @test distance(G, Gref, 0) < 1e-7
    #
    # The inner product sums the real parts of all elements of X'Y
    X = KB.Gᵐᵃᵗ(ntau, 2, 2)
    Y = KB.Gᵐᵃᵗ(ntau, 2, 2)
    X[40] = ComplexF64[1.0 2.0im; 0.5 -1.0]
    Y[40] = ComplexF64[0.3 1.0; -2.0im 0.7]
    @test KB._mat_inner(X, Y, I5) ≈ sum(real, X[40]' * Y[40])
    @test !(KB._mat_inner(X, Y, I5) ≈ real(tr(X[40]' * Y[40])))
    # Gregory end weights
    X[1] = ComplexF64[1.0 0.0; 0.0 0.0]
    Y[1] = ComplexF64[1.0 0.0; 0.0 0.0]
    X[ntau-1] = ComplexF64[0.0 0.0; 0.0 1.0]
    Y[ntau-1] = ComplexF64[0.0 0.0; 0.0 1.0]
    @test KB._mat_inner(X, Y, I5) ≈ sum(real, X[40]' * Y[40]) + I5.GIW.ω[1] + I5.GIW.ω[2]
end

@testset "dyson_mat: error paths" begin
    β, ntau, μ = 5.0, 41, 0.0
    Ha = ComplexF64[-1.0 0.3im; -0.3im 0.5]
    Hb = ComplexF64[1.0 0.2; 0.2 1.5]
    V = ComplexF64[0.4 0.1im; -0.2 0.3]
    C, Σ, G_exact, H = dm_embedding(Ha, Hb, V, μ, β, ntau, FERMI)
    G = ℱ(C, FERMI)
    #
    # Methods: the final set is {:fixpoint, :cg, :fourier}
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, H, β; method = :fourier)
    @test_throws "Phase 7" dyson_mat!(G, Σ, μ, H, β; method = :fourier)
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, H, β; method = :steep)
    @test_throws "unknown method" dyson_mat!(G, Σ, μ, H, β; method = :steep)
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, H, β; method = :foo)
    #
    # Solve order
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, H, β; solve_order = 0)
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, H, β; solve_order = 6)
    #
    # Statistics, shapes, and grids
    @test_throws ArgumentError dyson_mat!(ℱ(C, BOSE), Σ, μ, H, β)
    C1 = Cn(2, ntau, 1, 1, 1.0, β)
    @test_throws DimensionMismatch dyson_mat!(ℱ(C1, FERMI), Σ, μ, H, β)
    C2 = Cn(2, ntau + 2, 2, 2, 1.0, β)
    @test_throws DimensionMismatch dyson_mat!(ℱ(C2, FERMI), Σ, μ, H, β)
    C3 = Cn(2, ntau + 1, 2, 2, 1.0, β)
    Σ3 = ℱ(C3, FERMI)
    @test_throws ArgumentError dyson_mat!(ℱ(C3, FERMI), Σ3, μ, H, β)
    H3 = Cf(2, 3, 3)
    @test_throws DimensionMismatch dyson_mat!(G, Σ, μ, H3, β)
    #
    # Non-hermitian or non-finite Hamiltonian
    Hn = Cf(2, 2, 2)
    Hn[0] = ComplexF64[-1.0 0.3; 0.0 0.5]
    @test_throws ArgumentError dyson_mat!(G, Σ, μ, Hn, β)
    Hn[0] = ComplexF64[NaN 0.0; 0.0 0.5]
    @test_throws DomainError dyson_mat!(G, Σ, μ, Hn, β)
    #
    # Non-finite self-energy
    Σn = ℱ(C, FERMI)
    Σn.mat[3] = fill(NaN + 0.0im, 2, 2)
    @test_throws DomainError dyson_mat!(G, Σn, μ, H, β)
    #
    # Self-energy with non-hermitian Matsubara blocks is accepted
    Σh = ℱ(C, FERMI)
    for m = 1:ntau
        Σh.mat[m] = Σ.mat[m] + ComplexF64[0.0 0.01; 0.0 0.0]
    end
    @test dyson_mat!(ℱ(C, FERMI), Σh, μ, H, β) isa ℱ
end
