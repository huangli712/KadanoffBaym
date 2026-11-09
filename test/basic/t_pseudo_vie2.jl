#
# t_pseudo_vie2.jl
#
# To test the pseudo-particle VIE2 solvers: pseudo_vie2_mat!(),
# pseudo_vie2_start!(), pseudo_vie2_timestep!(), and pseudo_vie2!().
#
# The tests are:
#
# (1) F = 0: G = Q exactly.
# (2) Discrete residuals. Following NESSi, the Matsubara, greater, and the
#     left-mixing (tstp ≥ k + 2), and the start-up lesser components solve
#     G + F ⋆ G = Q, while the left-mixing (start-up, reversed imaginary
#     time) and the lesser (tstp ≥ k + 2) components solve the conjugate
#     form G + G ⋆ Fcc = Q. Each component
#     must satisfy its own discrete equation up to rounding errors. The
#     start-up and the time stepping left-mixing paths are distinguished
#     in this way.
# (3) Consistency with the pseudo-particle Dyson equation, i.e.,
#     (1 - G0 ⋆ Σ) ⋆ G = G0, where the difference is the discretization
#     error which decreases with the grid.
# (4) Q is not changed, and the overall driver equals the stepwise calls.
#
# There is no pseudo-particle Catch test in NESSi. The comparison with
# NESSi is in test/nessi/t_pseudo.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const PV_H0 = C64[0.3 0.2-0.1im; 0.2+0.1im -0.4]
const PV_λ0 = 0.5

# Build G0 (free), Σ, F = -G0 ⋆ Σ, and Fcc = -Σ ⋆ G0
function pv_setup(sign, k, nt, ntau, h, β; d = 2)
    C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
    Ik = Integrator(k)
    H0 = d == 2 ? PV_H0 : fill(C64(0.3), 1, 1)
    H = Cf(nt, d, d)
    for i = 0:nt
        H[i] = H0 + (i == 0 ? 0.0 : 0.3 * sin(2 * (i - 1) * h)) * I
    end
    G0 = PseudoGF(C, sign)
    pseudodyson!(G0, PV_λ0, H, PseudoGF(C, sign), Ik, β, h)
    Σ = PseudoGF(C, sign)
    pseudodyson!(Σ, 0.2, H, PseudoGF(C, sign), Ik, β, h)
    for t = 0:nt
        smul!(Σ, t, 0.3)
    end
    F = PseudoGF(C, sign); Fcc = PseudoGF(C, sign)
    pseudo_convolution!(F, G0, G0, Σ, Σ, Ik, β, h)
    pseudo_convolution!(Fcc, Σ, Σ, G0, G0, Ik, β, h)
    for t = 0:nt
        smul!(F, t, -1.0)
        smul!(Fcc, t, -1.0)
    end
    return C, Ik, H, G0, Σ, F, Fcc
end

@testset verbose = true "KadanoffBaym: pseudo-particle VIE2" begin

    β = 2.0

    @testset "F = 0 (sign = $sign, k = $k)" for sign in (FERMI, BOSE), k in (1, 3, 5)
        nt, ntau, h = 12, 31, 0.01
        C, Ik, H, G0, _, _, _ = pv_setup(sign, k, nt, ntau, h, β)
        Z = PseudoGF(C, sign)
        X = PseudoGF(C, sign)
        pseudo_vie2!(X, Z, Z, G0, Ik, β, h)
        @test maximum(t -> distance(X, G0, t), 0:nt) < 1e-13
    end

    @testset "discrete residuals (k = $k, size = $d)" for k in (1, 3, 5), d in (1, 2)
        nt, ntau, h = 14, 41, 0.01
        C, Ik, H, G0, Σ, F, Fcc = pv_setup(FERMI, k, nt, ntau, h, β; d = d)
        G = PseudoGF(C, FERMI)
        pseudo_vie2!(G, F, Fcc, G0, Ik, β, h)
        # R = G + F ⋆ G - Q
        R = PseudoGF(C, FERMI)
        pseudo_convolution!(R, F, Fcc, G, G, Ik, β, h)
        incr!(R, G, 1.0); incr!(R, G0, -1.0)
        # Rc = G + G ⋆ Fcc - Q
        Rc = PseudoGF(C, FERMI)
        pseudo_convolution!(Rc, G, G, Fcc, F, Ik, β, h)
        incr!(Rc, G, 1.0); incr!(Rc, G0, -1.0)
        #
        @test maximum(m -> norm(R.mat[m]), 1:ntau) < 1e-10
        @test maximum(norm(R.ret[n,j]) for n = 1:nt for j = 1:n) < 1e-10
        # Left-mixing: start-up and time stepping use different paths
        @test maximum(norm(Rc.lmix[n,m]) for n = 1:k+1 for m = 1:ntau) < 1e-10
        @test maximum(norm(R.lmix[n,m]) for n = k+2:nt for m = 1:ntau) < 1e-10
        # Lesser: the start-up steps (standard VIE2 kernel) solve the F form,
        # the time stepping solves the conjugate form
        @test maximum(norm(R.less[j,n]) for n = 1:k+1 for j = 1:n) < 1e-10
        @test maximum(norm(Rc.less[j,n]) for n = k+2:nt for j = 1:n) < 1e-10
        if k == 1
            # The two forms differ by the discretization error, so the
            # residual test above distinguishes the two left-mixing paths
            @test maximum(norm(R.lmix[k+1,m]) for m = 1:ntau) > 1e-10
            @test maximum(norm(Rc.lmix[nt,m]) for m = 1:ntau) > 1e-10
        end
    end

    @testset "consistency with pseudodyson (k = $k)" for k in (2, 3)
        err = Float64[]
        for (nt, ntau, h) in ((14, 41, 0.01), (27, 81, 0.005), (53, 161, 0.0025))
            C, Ik, H, G0, Σ, F, Fcc = pv_setup(FERMI, k, nt, ntau, h, β)
            Y = PseudoGF(C, FERMI)
            pseudo_vie2!(Y, F, Fcc, G0, Ik, β, h)
            G = PseudoGF(C, FERMI)
            pseudodyson!(G, PV_λ0, H, Σ, Ik, β, h)
            # The same physical time t = 0.13
            e = max(maximum(m -> norm(Y.mat[m] - G.mat[m]), 1:ntau),
                    maximum(j -> norm(Y.ret[nt,j] - G.ret[nt,j]), 1:nt),
                    maximum(m -> norm(Y.lmix[nt,m] - G.lmix[nt,m]), 1:ntau),
                    maximum(j -> norm(Y.less[j,nt] - G.less[j,nt]), 1:nt))
            push!(err, e)
        end
        @test err[1] < 2e-3
        @test err[2] < err[1] / 2^k && err[3] < err[2] / 2^k
    end

    @testset "Q unchanged, stepwise and overall drivers" begin
        nt, ntau, h, k = 14, 31, 0.01, 4
        C, Ik, H, G0, Σ, F, Fcc = pv_setup(BOSE, k, nt, ntau, h, β)
        Q = deepcopy(G0)
        G = PseudoGF(C, BOSE)
        pseudo_vie2!(G, F, Fcc, Q, Ik, β, h)
        @test Q == G0
        X = PseudoGF(C, BOSE)
        pseudo_vie2_mat!(X, F, Fcc, Q, β, Ik)
        pseudo_vie2_start!(X, F, Fcc, Q, Ik, β, h)
        # The start-up results at tstp = k + 1 are kept
        Xs = X[k+1]
        for tstp = k+2:nt
            pseudo_vie2_timestep!(tstp, X, F, Fcc, Q, Ik, β, h)
        end
        @test Q == G0
        @test distance(X, Xs, k + 1) == 0.0
        @test all(t -> distance(G, X, t) == 0.0, 0:nt)
        # Repeated time stepping gives the same result
        pseudo_vie2_timestep!(nt, X, F, Fcc, Q, Ik, β, h)
        @test distance(G, X, nt) == 0.0
        @test_throws ArgumentError pseudo_vie2_timestep!(k + 1, X, F, Fcc, Q, Ik, β, h)
    end

    @testset "argument checks" begin
        C = Cn(10, 31, 2, 2, 0.1, β)
        G = PseudoGF(C)
        @test_throws ArgumentError pseudo_vie2!(G, PseudoGF(C, BOSE), G, G, Integrator(3), β, 0.01)
        @test_throws ArgumentError pseudo_vie2!(G, G, G, G, Integrator(6), β, 0.01)
        @test_throws DimensionMismatch pseudo_vie2!(G, PseudoGF(Cn(10, 21, 2, 2, 0.1, β)),
                                                    G, G, Integrator(3), β, 0.01)
        F = ℱ(C)
        @test_throws MethodError pseudo_vie2!(F, F, F, F, Integrator(3), β, 0.01)
    end
end
