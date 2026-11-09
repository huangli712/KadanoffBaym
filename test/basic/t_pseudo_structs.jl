#
# t_pseudo_structs.jl
#
# To test the pseudo-particle Green's functions, PseudoGF and PseudoGFStep:
# construction, equality, the greater (gtr) convention, time step get/set,
# traits (memcpy!, memset!, zeros!, incr!, smul!), the left and right
# multiplications by a Cf object, the plain text I/O, and the density
# matrix (herm_pseudo::density_matrix).
#
# NESSi reference: libcntr/cntr/cntr_herm_pseudo_impl.hpp:38-63,287-465,
# 707-958. There is no pseudo-particle Catch test in NESSi.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const PS_KB = KadanoffBaym

# Fill all the stored blocks with random numbers
function ps_fill!(G, nt, ntau)
    for m = 1:ntau
        G.mat[m] = rand(ComplexF64, getdims(G)...)
    end
    for i = 1:nt
        for j = 1:i
            G.ret[i,j] = rand(ComplexF64, getdims(G)...)
        end
        for m = 1:ntau
            G.lmix[i,m] = rand(ComplexF64, getdims(G)...)
        end
        for j = i:nt
            G.less[i,j] = rand(ComplexF64, getdims(G)...)
        end
    end
    return G
end

@testset verbose = true "KadanoffBaym: pseudo-particle structs" begin

    nt, ntau = 6, 7
    C = Cn(nt, ntau, 2, 2, 0.5, 2.0)

    @testset "construction and equality" begin
        G = PseudoGF(C)
        @test getdims(G) == (2, 2)
        @test getntime(G) == nt
        @test getntau(G) == ntau
        @test getsign(G) == FERMI
        @test PseudoGF(C, BOSE).sign == BOSE
        @test PseudoGF(C, 0.5im, FERMI).mat[3] == fill(0.5im, 2, 2)
        @test PseudoGF(C, 0.5im, FERMI) == PseudoGF(C, 0.5im, FERMI)
        @test PseudoGF(C, 0.5im, FERMI) != PseudoGF(C, 0.4im, FERMI)
        @test_throws AssertionError PseudoGF(C, 0)
        # Non-square matrices are rejected
        @test_throws AssertionError PseudoGF(Cn(nt, ntau, 2, 3, 0.5, 2.0))
        # The storage is the same as that of ℱ
        F = ℱ(C)
        @test size(G.ret.data) == size(F.ret.data)
        @test size(G.less.data) == size(F.less.data)
        @test size(G.lmix.data) == size(F.lmix.data)
        @test size(G.mat.data) == size(F.mat.data)
        @test PS_KB.gf_convention(PseudoGF) == Val(:pseudo)
        @test PS_KB.gf_convention(ℱ) == Val(:standard)
        @test PS_KB.gf_convention(G) == Val(:pseudo)
        @test !(G isa ℱ)
        @test G isa PS_KB.CnAbstractFunction{ComplexF64}
    end

    @testset "greater convention" begin
        G = ps_fill!(PseudoGF(C), nt, ntau)
        # The gtr property is the stored ret component itself
        @test G.gtr === G.ret
        X = rand(ComplexF64, 2, 2)
        G.gtr[4,2] = X
        @test G.ret[4,2] == X
        # The other triangle is given by -adjoint
        for i = 1:nt, j = i+1:nt
            @test G.ret[i,j] == -adjoint(G.ret[j,i])
            @test G.less[j,i] == -adjoint(G.less[i,j])
        end
        @test_throws ErrorException G.adv
        @test G.matm isa PS_KB.Gᵐᵃᵗᵐ
        @test G.rmix isa PS_KB.Gʳᵐⁱˣ
        # The standard ℱ keeps gtr = less + ret
        F = ℱ(C)
        for i = 1:nt
            for j = 1:i
                F.ret[i,j] = rand(ComplexF64, 2, 2)
            end
            for j = i:nt
                F.less[i,j] = rand(ComplexF64, 2, 2)
            end
        end
        for i = 1:nt, j = 1:i
            @test F.gtr[i,j] == F.less[i,j] + F.ret[i,j]
        end
        # The time step object
        g = G[3]
        @test g isa PseudoGFStep
        @test g.gtr === g.ret
        @test_throws ErrorException g.adv
    end

    @testset "time step get/set and distance" begin
        G = ps_fill!(PseudoGF(C), nt, ntau)
        H = deepcopy(G)
        @test all(t -> distance(G, H, t) == 0.0, 0:nt)
        for t = 0:nt
            g = G[t]
            @test gettstp(g) == t
            @test distance(G, g, t) == 0.0
            zeros!(H, t)
            @test t == 0 || distance(G, H, t) > 0.0
            H[t] = g
            @test distance(G, H, t) == 0.0
        end
        @test H == G
        # Incompatible signs
        @test !iscompatible(G, PseudoGF(C, BOSE))
        @test iscompatible(G, PseudoGF(C, FERMI))
    end

    @testset "traits" begin
        A = ps_fill!(PseudoGF(C), nt, ntau)
        B = ps_fill!(PseudoGF(C), nt, ntau)
        # memcpy!
        X = PseudoGF(C)
        memcpy!(A, X)
        @test X == A
        @test X.gtr === X.ret
        Y = PseudoGF(C)
        memcpy!(A, Y, 3)
        @test distance(A, Y, 3) == 0.0
        @test distance(A, Y, 2) > 0.0
        # memset! and zeros!
        Z = deepcopy(A)
        memset!(Z, 2, 1.0im)
        @test Z.ret[2,1] == fill(1.0im, 2, 2)
        @test distance(Z, A, 3) == 0.0
        @test distance(Z, A, 1) == 0.0
        zeros!(Z)
        @test all(iszero, Z.mat.data) && all(iszero, Z.ret.data)
        # incr!: A + 2B, blockwise
        S = deepcopy(A)
        incr!(S, B, 2.0)
        for t = 1:nt
            for j = 1:t
                @test S.ret[t,j] ≈ A.ret[t,j] + 2 * B.ret[t,j]
                @test S.less[j,t] ≈ A.less[j,t] + 2 * B.less[j,t]
            end
            for m = 1:ntau
                @test S.lmix[t,m] ≈ A.lmix[t,m] + 2 * B.lmix[t,m]
            end
        end
        @test S.mat[4] ≈ A.mat[4] + 2 * B.mat[4]
        # smul!
        S = deepcopy(A)
        smul!(S, 0.5im)
        @test S.ret[5,3] ≈ 0.5im * A.ret[5,3]
        @test S.less[2,6] ≈ 0.5im * A.less[2,6]
        @test S.lmix[4,7] ≈ 0.5im * A.lmix[4,7]
        @test S.mat[2] ≈ 0.5im * A.mat[2]
    end

    @testset "multiplication by Cf (2x2 non-commuting)" begin
        A = ps_fill!(PseudoGF(C), nt, ntau)
        f = Cf(nt, 2, 2)
        for i = 0:nt
            f[i] = rand(ComplexF64, 2, 2)
        end
        @test f[2] * f[3] != f[3] * f[2]
        for t = 0:nt
            L = deepcopy(A)
            R = deepcopy(A)
            smul!(f, L, t)
            smul!(R, f, t)
            if t == 0
                for m = 1:ntau
                    @test L.mat[m] ≈ f[0] * A.mat[m]
                    @test R.mat[m] ≈ A.mat[m] * f[0]
                end
            else
                for j = 1:t
                    @test L.ret[t,j] ≈ f[t] * A.ret[t,j]
                    @test R.ret[t,j] ≈ A.ret[t,j] * f[j]
                    @test L.less[j,t] ≈ f[j] * A.less[j,t]
                    @test R.less[j,t] ≈ A.less[j,t] * f[t]
                end
                for m = 1:ntau
                    @test L.lmix[t,m] ≈ f[t] * A.lmix[t,m]
                    @test R.lmix[t,m] ≈ A.lmix[t,m] * f[0]
                end
                @test distance(L, R, t) > 0.0
            end
        end
    end

    @testset "density matrix (sign = $sign)" for sign in (FERMI, BOSE)
        G = ps_fill!(PseudoGF(C, sign), nt, ntau)
        # ρ = -Gᴹ(β) on the Matsubara axis, ρ = i ξ G^<(t,t) at real time
        @test density_matrix(G, 0) == -G.mat[ntau]
        for n = 1:nt
            @test density_matrix(G, n) == im * sign * G.less[n,n]
            @test density_matrix(G[n], n) == density_matrix(G, n)
            @test density(G, n) == density_matrix(G, n)
            @test density(G[n], n) == density_matrix(G, n)
        end
        @test density_matrix(G[0], 0) == density_matrix(G, 0)
        @test_throws BoundsError density_matrix(G, nt + 1)
        @test_throws BoundsError density_matrix(G, -1)
        @test_throws ArgumentError density_matrix(G[2], 3)
    end

    @testset "plain text I/O" begin
        A = ps_fill!(PseudoGF(C, BOSE), nt, ntau)
        mktempdir() do dir
            fn = joinpath(dir, "pseudo.data")
            write(fn, A)
            @test occursin("Pseudo", readline(fn))
            B = PseudoGF(C, FERMI)
            read!(fn, B)
            @test B.sign == BOSE
            @test all(t -> distance(A, B, t) < 1e-10, 0:nt)
            @test B.gtr === B.ret
            # A standard file can not be read into a PseudoGF object
            fs = joinpath(dir, "standard.data")
            write(fs, ℱ(C))
            @test_throws ErrorException read!(fs, PseudoGF(C))
            # The sizes of the preallocated object are never changed
            D = PseudoGF(Cn(nt - 1, ntau, 2, 2, 0.5, 2.0))
            @test_throws DimensionMismatch read!(fn, D)
            @test getntime(D) == nt - 1
            @test_throws ErrorException read!(joinpath(dir, "none"), B)
        end
    end
end
