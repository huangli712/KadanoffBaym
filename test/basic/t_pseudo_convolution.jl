#
# t_pseudo_convolution.jl
#
# To test the pseudo-particle convolution: the restricted Matsubara and
# left-mixing integrals, the masks, the contour function f, the start-up
# and late time steps, the consistency of the overall and stepwise
# drivers, and the equal-time density matrix.
#
# The expected values are evaluated from closed forms (constant and linear
# kernels, which are integrated exactly by the Gregory weights), not from
# the implementation under test. The time-diagonal comparison follows
# NESSi libcntr/test/convolution.cpp:848-885, but the maximum error over
# all the time steps is checked (the Catch test checks the last one only).
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

#
# Fill a pseudo-particle Green's function with constant blocks:
# Gᴹ(τ) = M, G^{>}(t,t') = R, G^⌉(t,τ) = X, G^<(t,t') = L. R and L must be
# anti-hermitian, so that the values outside the stored triangles (given
# by -adjoint) are the same constants.
#
function pc_const!(G, M, R, X, L)
    nt, ntau = getntime(G), getntau(G)
    for m = 1:ntau
        G.mat[m] = M
    end
    for i = 1:nt
        for j = 1:i
            G.ret[i,j] = R
        end
        for m = 1:ntau
            G.lmix[i,m] = X
        end
        for j = i:nt
            G.less[i,j] = L
        end
    end
    return G
end

# A random anti-hermitian matrix
pc_ah(d) = (X = rand(ComplexF64, d, d); X - X')

# Copy the storage of a PseudoGF object to a ℱ object
function pc_to_std(G::PseudoGF, C::Cn)
    F = ℱ(C, getsign(G))
    memcpy!(G.mat, F.mat)
    memcpy!(getfield(G, :ret), F.ret)
    memcpy!(G.lmix, F.lmix)
    memcpy!(G.less, F.less)
    return F
end

# Random data in the stored triangles
function pc_rand!(G)
    nt, ntau = getntime(G), getntau(G)
    d = getdims(G)
    for m = 1:ntau
        G.mat[m] = rand(ComplexF64, d...)
    end
    for i = 1:nt
        for j = 1:i
            G.ret[i,j] = rand(ComplexF64, d...)
        end
        for m = 1:ntau
            G.lmix[i,m] = rand(ComplexF64, d...)
        end
        for j = i:nt
            G.less[i,j] = rand(ComplexF64, d...)
        end
    end
    return G
end

@testset verbose = true "KadanoffBaym: pseudo-particle convolution" begin

    nt, ntau, β, h = 12, 21, 2.0, 0.05
    δτ = β / (ntau - 1)
    τ(m) = (m - 1) * δτ
    t(n) = (n - 1) * h

    @testset "constant kernels, 1x1 and 2x2 (k = $k)" for k = 1:5, d in (1, 2)
        I = Integrator(k)
        C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
        Ma = rand(ComplexF64, d, d); Mb = rand(ComplexF64, d, d)
        Ra = pc_ah(d); Rb = pc_ah(d)
        Xa = rand(ComplexF64, d, d); Xb = rand(ComplexF64, d, d)
        La = pc_ah(d); Lb = pc_ah(d)
        F0 = rand(ComplexF64, d, d); F1 = rand(ComplexF64, d, d)
        A = pc_const!(PseudoGF(C), Ma, Ra, Xa, La)
        B = pc_const!(PseudoGF(C), Mb, Rb, Xb, Lb)
        f = Cf(nt, d, d)
        f[0] = F0
        for i = 1:nt
            f[i] = F1
        end
        X = PseudoGF(C)
        pseudo_convolution!(X, A, A, f, B, B, I, β, h)
        # Restricted Matsubara integral: τ Aᴹ f₀ Bᴹ. The tolerance is
        # relative to the magnitude of the result (rounding errors).
        scale = 1 + β * norm(Ma * F0 * Mb) + β * norm(Xa * F0 * Mb) + t(nt) * norm(Ra * F1 * Xb)
        @test maximum(m -> norm(X.mat[m] - τ(m) * Ma * F0 * Mb), 1:ntau) < 1e-12 * scale
        # Left-mixing: t A^{>} f B^⌉ + (β - τ) A^⌉ f₀ Bᴹ, start-up and late steps
        err = 0.0
        for n = 1:nt, m = 1:ntau
            ref = t(n) * Ra * F1 * Xb + (β - τ(m)) * Xa * F0 * Mb
            err = max(err, norm(X.lmix[n,m] - ref))
        end
        @test err < 1e-12 * scale
        # The end point τ = β of the imaginary time integral vanishes
        @test norm(X.lmix[1,ntau]) < 1e-14
        if d == 2
            @test norm(Ma * F0 * Mb - Mb * F0 * Ma) > 1e-3
        end
    end

    @testset "linear kernels (k = $k)" for k = 2:5
        I = Integrator(k)
        C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        A = PseudoGF(C); B = PseudoGF(C)
        for m = 1:ntau
            A.mat[m] = fill(1.0 + 0.0im, 1, 1)
            B.mat[m] = fill(complex(τ(m)), 1, 1)
        end
        for n = 1:nt, m = 1:ntau
            A.lmix[n,m] = fill(complex(τ(m)), 1, 1)
        end
        X = PseudoGF(C)
        pseudo_convolution_matsubara!(X, A, B, I, β)
        # ∫₀^τ x dx = τ²/2
        @test maximum(m -> abs(X.mat[m][1] - τ(m)^2 / 2), 1:ntau) < 1e-12
        for n = 1:nt
            pseudo_convolution_timestep!(n, X, A, B, I, β, h)
        end
        # ∫_τ^β x (x - τ) dx = (β - τ)² (2β + τ) / 6
        err = maximum(abs(X.lmix[n,m][1] - (β - τ(m))^2 * (2β + τ(m)) / 6)
                      for n = 1:nt, m = 1:ntau)
        @test err < 1e-12
    end

    @testset "greater and lesser components are the standard ones" begin
        I = Integrator(4)
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        A = pc_rand!(PseudoGF(C)); Acc = pc_rand!(PseudoGF(C))
        B = pc_rand!(PseudoGF(C)); Bcc = pc_rand!(PseudoGF(C))
        f = Cf(nt, 2, 2)
        for i = 0:nt
            f[i] = rand(ComplexF64, 2, 2)
        end
        X = PseudoGF(C)
        Y = ℱ(C)
        Fa, Fc, Fb, Fd = (pc_to_std(G, C) for G in (A, Acc, B, Bcc))
        err = 0.0
        for n = 1:nt
            incr_pseudo_convolution!(n, X, A, Acc, f, B, Bcc, I, β, h, 0.7)
            incr_convolution_ret!(n, Y, Fa, Fc, f, Fb, Fd, I, h, 0.7)
            incr_convolution_les!(n, Y, Fa, Fc, f, Fb, Fd, I, β, h, 0.7)
            for j = 1:n
                err = max(err, norm(X.ret[n,j] - Y.ret[n,j]), norm(X.less[j,n] - Y.less[j,n]))
            end
        end
        @test err == 0.0
        # The left-mixing component differs from the standard one
        incr_convolution_lmix!(3, Y, Fa, Fc, f, Fb, Fd, I, β, h, 0.7)
        @test maximum(m -> norm(X.lmix[3,m] - Y.lmix[3,m]), 1:ntau) > 1e-3
    end

    @testset "masks" begin
        I = Integrator(3)
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        A = pc_rand!(PseudoGF(C)); B = pc_rand!(PseudoGF(C))
        X0 = pc_rand!(PseudoGF(C))
        mask = [isodd(m) for m = 1:ntau]
        # Matsubara
        X = deepcopy(X0); Y = deepcopy(X0)
        incr_pseudo_convolution_mat!(X, A, nothing, B, I, β, 1.5; mask = mask)
        incr_pseudo_convolution_mat!(Y, A, nothing, B, I, β, 1.5)
        for m = 1:ntau
            @test X.mat[m] == (mask[m] ? Y.mat[m] : X0.mat[m])
        end
        # Left-mixing
        X = deepcopy(X0); Y = deepcopy(X0)
        incr_pseudo_convolution_lmix!(5, X, A, A, nothing, B, B, I, β, h, 1.5; mask = mask)
        incr_pseudo_convolution_lmix!(5, Y, A, A, nothing, B, B, I, β, h, 1.5)
        for m = 1:ntau
            @test X.lmix[5,m] == (mask[m] ? Y.lmix[5,m] : X0.lmix[5,m])
        end
        @test distance(X, X0, 4) == 0.0
        @test_throws AssertionError incr_pseudo_convolution_mat!(X, A, nothing, B, I, β, 1.0;
                                                                 mask = trues(3))
    end

    @testset "overall, stepwise, and time slice targets" begin
        I = Integrator(5)
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        A = pc_rand!(PseudoGF(C)); B = pc_rand!(PseudoGF(C))
        f = Cf(nt, 2, 2)
        for i = 0:nt
            f[i] = rand(ComplexF64, 2, 2)
        end
        X = PseudoGF(C)
        pseudo_convolution!(X, A, f, B, I, β, h)
        Y = pc_rand!(PseudoGF(C))
        for n = 0:nt
            # The old values are discarded, repeated calls give the same result
            pseudo_convolution_timestep!(n, Y, A, f, B, I, β, h)
            pseudo_convolution_timestep!(n, Y, A, f, B, I, β, h)
            @test distance(X, Y, n) == 0.0
        end
        # A time slice target gives the same increment
        for n = 0:nt
            g = PseudoGFStep(C, n)
            incr_pseudo_convolution!(n, g, A, A, f, B, B, I, β, h, 1.0)
            @test distance(X, g, n) == 0.0
        end
    end

    @testset "time-diagonal density matrix (k = $k)" for k in (1, 3, 5)
        I = Integrator(k)
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        A = pc_rand!(PseudoGF(C)); Acc = pc_rand!(PseudoGF(C))
        B = pc_rand!(PseudoGF(C)); Bcc = pc_rand!(PseudoGF(C))
        f = Cf(nt, 2, 2)
        for i = 0:nt
            f[i] = rand(ComplexF64, 2, 2)
        end
        X = PseudoGF(C)
        pseudo_convolution!(X, A, Acc, f, B, Bcc, I, β, h)
        err = norm(convolution_density_matrix(0, A, Acc, f, B, Bcc, I, β, h) + X.mat[ntau])
        for n = 1:nt
            ρ = convolution_density_matrix(n, A, Acc, f, B, Bcc, I, β, h)
            # Following NESSi, the incremental lesser convolution uses
            # -Acc^<(t,t)† instead of A^<(t,t) in A^< ∗ f ∗ Bᴬ for the
            # early time steps (n - 1 ≤ 2k + 2), while the density matrix
            # always uses A^<(t,t). The difference is the diagonal term.
            L = X.less[n,n]
            if n - 1 ≤ 2k + 2
                Δ = -adjoint(Acc.less[n,n]) - A.less[n,n]
                L = L - h * I.GIW[n-1,n-1] * Δ * f[n] * adjoint(Bcc.gtr[n,n])
            end
            err = max(err, norm(ρ - (-im) * L))
        end
        @test err < 1e-10
        # The correction is not trivial for these (non-partner) inputs
        @test norm(adjoint(Acc.less[1,1]) + A.less[1,1]) > 1e-3
    end

    @testset "dispatch boundaries" begin
        I = Integrator(2)
        C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        P = PseudoGF(C); F = ℱ(C)
        # The standard Matsubara and left-mixing convolutions reject PseudoGF
        @test_throws MethodError incr_convolution_mat!(P, P, nothing, P, I, β, 1.0)
        @test_throws MethodError incr_convolution_lmix!(2, P, P, P, nothing, P, P, I, β, h, 1.0)
        @test_throws MethodError incr_convolution!(2, P, P, P, nothing, P, P, I, β, h, 1.0)
        # The pseudo-particle convolutions reject ℱ
        @test_throws MethodError pseudo_convolution!(F, F, F, I, β, h)
        @test_throws MethodError incr_pseudo_convolution!(2, F, F, F, nothing, F, F, I, β, h, 1.0)
    end
end
