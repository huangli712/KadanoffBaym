#
# t_base.jl
#
# To test the initialization of contour-ordered Green's functions, i.e.,
# the from_ham!() function, against the closed-form free solutions.
#
# NESSi reference: libcntr/cntr/cntr_equilibrium_impl.hpp (green_from_H).
#

using LinearAlgebra

@testset verbose = true "KadanoffBaym: base.jl" begin
    ntime = 101
    ntau = 101
    tmax = 1.0
    beta = 5.0
    h = tmax / (ntime - 1)
    dtau = beta / (ntau - 1)
    mu = 0.3
    tol = 1e-12

    @testset "from_ham!: 1x1 fermion and boson" begin
        # sign, ε (band energy)
        cases = ((FERMI, 1.0), (BOSE, 1.0))
        for (sign, ε) in cases
            C = Cn(ntime, ntau, 1, 1, tmax, beta)
            G = ℱ(C, sign)
            H = fill(ε + 0.0im, 1, 1)
            from_ham!(G, mu, H, beta, h)
            #
            ξ = sign
            ϵ = ε - mu
            # Distribution function: f(ϵ) for fermions and b(ϵ) for bosons
            n = (sign == FERMI) ? 1.0 / (exp(beta * ϵ) + 1.0) :
                                  1.0 / (exp(beta * ϵ) - 1.0)
            #
            # Matsubara: G(τ) = -e^{-ϵτ} (1 + ξ n)
            for m = 1:ntau
                τ = (m - 1) * dtau
                x = -exp(-ϵ * τ) * (1.0 + ξ * n)
                @test isapprox(G.mat[m][1,1], x; atol = tol, rtol = tol)
            end
            #
            # Retarded: G(t,t') = -i e^{-iϵ(t-t')}
            for i = 1:ntime, j = 1:i
                x = -im * exp(-im * ϵ * (i - j) * h)
                @test isapprox(G.ret[i,j][1,1], x; atol = tol, rtol = tol)
            end
            #
            # Left-mixing: G(t,τ) = -iξ n e^{ϵτ} e^{-iϵt}
            for i = 1:ntime, m = 1:ntau
                τ = (m - 1) * dtau
                x = -im * ξ * n * exp(ϵ * τ) * exp(-im * ϵ * (i - 1) * h)
                @test isapprox(G.lmix[i,m][1,1], x; atol = tol, rtol = tol)
            end
            #
            # Lesser: G(t,t') = -iξ n e^{-iϵ(t-t')}
            for j = 1:ntime, i = 1:j
                x = -im * ξ * n * exp(-im * ϵ * (i - j) * h)
                @test isapprox(G.less[i,j][1,1], x; atol = tol, rtol = tol)
            end
            #
            # KMS boundary: G(0⁺) - ξ G(β⁻) = -1
            @test isapprox(G.mat[1][1,1] - ξ * G.mat[ntau][1,1], -1.0;
                           atol = tol, rtol = tol)
            # Boundary: G⌉(0,τ) = iξ Gᴹ(β-τ)
            for m = 1:ntau
                x = im * ξ * G.mat[ntau-m+1][1,1]
                @test isapprox(G.lmix[1,m][1,1], x; atol = tol, rtol = tol)
            end
            # Equal-time: G<(0,0) = G⌉(0,0)
            @test isapprox(G.less[1,1][1,1], G.lmix[1,1][1,1];
                           atol = tol, rtol = tol)
        end
    end

    @testset "from_ham!: 2x2 hermitian fermion" begin
        C = Cn(ntime, ntau, 2, 2, tmax, beta)
        G = ℱ(C, FERMI)
        H = zeros(C64, 2, 2)
        H[1,1] = sqrt(2.0)
        H[1,2] = sqrt(2.0) * im
        H[2,1] = -sqrt(2.0) * im
        H[2,2] = -sqrt(2.0)
        from_ham!(G, mu, H, beta, h)
        #
        𝕀 = Matrix{C64}(I, 2, 2)
        Hμ = H - mu * 𝕀
        vals, vecs = eigen(Hermitian(Hμ))
        # Occupation matrix ρ = f(H - μ)
        ρ = vecs * diagm(1.0 ./ (exp.(beta .* vals) .+ 1.0)) * vecs'
        #
        for i = 1:ntime
            # G^R(t,t) = -i𝕀
            @test isapprox(G.ret[i,i], -im * 𝕀; atol = tol, rtol = tol)
            # G^<(t,t) = iρ, anti-hermitian and time independent
            @test isapprox(G.less[i,i], im * ρ; atol = tol, rtol = tol)
            @test isapprox(G.less[i,i], -G.less[i,i]'; atol = tol, rtol = tol)
        end
        #
        # Retarded: G(t,t') = -i exp(-i(H-μ)(t-t'))
        for i = 1:ntime, j = 1:i
            x = -im * exp(-im * Hμ * (i - j) * h)
            @test isapprox(G.ret[i,j], x; atol = tol, rtol = tol)
        end
        #
        # Lesser: G(t,t') = i exp(-i(H-μ)t) ρ exp(i(H-μ)t')
        for j = 1:ntime, i = 1:j
            x = im * exp(-im * Hμ * (i - 1) * h) * ρ * exp(im * Hμ * (j - 1) * h)
            @test isapprox(G.less[i,j], x; atol = tol, rtol = tol)
        end
        #
        # Matsubara: hermitian, and G(β⁻) = -ρ, G(0⁺) = ρ - 𝕀
        for m = 1:ntau
            @test isapprox(G.mat[m], G.mat[m]'; atol = tol, rtol = tol)
        end
        @test isapprox(G.mat[ntau], -ρ; atol = tol, rtol = tol)
        @test isapprox(G.mat[1], ρ - 𝕀; atol = tol, rtol = tol)
        #
        # Boundary: G⌉(0,τ) = -i Gᴹ(β-τ)
        for m = 1:ntau
            x = -im * G.mat[ntau-m+1]
            @test isapprox(G.lmix[1,m], x; atol = tol, rtol = tol)
        end
    end
end

println("All tests pass!\n")
