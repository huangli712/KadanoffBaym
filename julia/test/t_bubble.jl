#
# t_bubble.jl
#
# To test the bubble diagrams, bubble1!() and bubble2!().
#
# The expected values are evaluated from the closed forms of the free
# Green's functions (see NESSi libcntr/test/bubble.cpp), not from the
# implementation under test. All the combinations FF/FB/BF/BB, the 1x1
# and 2x2 cases, all the orbital indices, and an explicit non-hermitian
# pair (A, Acc) are covered.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "common", "profile.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

#
# Closed forms of the free Green's function of the hermitian matrix H.
# Let H - μ = R diag(ε) R†, n = 1 / (exp(βε) - ξ).
#
struct BBFreeGF
    R::Matrix{ComplexF64}
    ε::Vector{Float64}
    n::Vector{Float64}
    ξ::Int
end

function BBFreeGF(H::Matrix{ComplexF64}, μ::Float64, β::Float64, ξ::Int)
    ε, R = eigen(Hermitian(H - μ * I))
    n = 1.0 ./ (exp.(β .* ε) .- ξ)
    return BBFreeGF(R, ε, n, ξ)
end

bb_mk(g, d) = g.R * Diagonal(d) * g.R'
bb_les(g, t, t′) = -im * g.ξ * bb_mk(g, g.n .* cis.(-g.ε .* (t - t′)))
bb_gtr(g, t, t′) = -im * bb_mk(g, (1 .+ g.ξ .* g.n) .* cis.(-g.ε .* (t - t′)))
bb_mat(g, τ) = -bb_mk(g, (1 .+ g.ξ .* g.n) .* exp.(-g.ε .* τ))
bb_tv(g, t, τ) = -im * g.ξ * bb_mk(g, g.n .* cis.(-g.ε .* t) .* exp.(g.ε .* τ))
bb_vt(g, τ, t) = -im * bb_mk(g, (1 .+ g.ξ .* g.n) .* cis.(g.ε .* t) .* exp.(-g.ε .* τ))

#
# Contour function from the closed forms, multiplied by the constant c
#
function bb_gf(C::Cn, g::BBFreeGF, c::ComplexF64, h::Float64, β::Float64)
    nt, ntau = C.ntime, C.ntau
    G = ℱ(C, g.ξ)
    δτ = β / (ntau - 1)
    for m = 1:ntau
        G.mat[m] = c * bb_mat(g, (m - 1) * δτ)
    end
    for i = 1:nt, m = 1:ntau
        G.lmix[i, m] = c * bb_tv(g, (i - 1) * h, (m - 1) * δτ)
    end
    for i = 1:nt, j = 1:i
        t, t′ = (i - 1) * h, (j - 1) * h
        G.ret[i, j] = c * (bb_gtr(g, t, t′) - bb_les(g, t, t′))
        G.less[j, i] = c * bb_les(g, t′, t)
    end
    return G
end

#
# Maximum deviation of the element (c1,c2) of C from the expected values
#
function bb_check(C, c1, c2, nt, ntau, h, β, emat, eret, eles, etv)
    δτ = β / (ntau - 1)
    err = 0.0
    for m = 1:ntau
        err = max(err, abs(C.mat[m][c1,c2] - emat((m - 1) * δτ)))
    end
    for i = 1:nt
        t = (i - 1) * h
        for j = 1:i
            t′ = (j - 1) * h
            err = max(err, abs(C.ret[i, j][c1,c2] - eret(t, t′)))
            err = max(err, abs(C.less[j, i][c1,c2] - eles(t′, t)))
        end
        for m = 1:ntau
            err = max(err, abs(C.lmix[i, m][c1,c2] - etv(t, (m - 1) * δτ)))
        end
    end
    return err
end

function bb_run(nt, ntau, β, h, d, ξa, ξb; cA = 1.0 + 0.0im, cB = 1.0 + 0.0im)
    Ha = d == 1 ? fill(ComplexF64(0.4), 1, 1) : ComplexF64[0.4 0.3-0.2im; 0.3+0.2im -0.6]
    Hb = d == 1 ? fill(ComplexF64(1.2), 1, 1) : ComplexF64[1.2 0.1im; -0.1im 1.9]
    ga = BBFreeGF(Ha, ξa == FERMI ? 0.1 : -0.5, β, ξa)
    gb = BBFreeGF(Hb, ξb == FERMI ? 0.2 : -0.3, β, ξb)
    C = Cn(nt, ntau, d, d, (nt - 1) * h, β)
    A = bb_gf(C, ga, cA, h, β)
    Acc = bb_gf(C, ga, conj(cA), h, β)
    B = bb_gf(C, gb, cB, h, β)
    Bcc = bb_gf(C, gb, conj(cB), h, β)
    c = cA * cB
    errmax = 0.0
    for a1 = 1:d, a2 = 1:d, b1 = 1:d, b2 = 1:d
        # Bubble1: C(t,t') = i A_{a1a2}(t,t') B_{b2b1}(t',t)
        C1 = ℱ(C, ξa * ξb)
        for tstp = 0:nt
            if a1 == a2 == b1 == b2 == 1 && d == 1
                bubble1!(tstp, C1, A, Acc, B, Bcc)
            else
                bubble1!(tstp, C1, 1, d, A, Acc, a1, a2, B, Bcc, b1, b2)
            end
        end
        c1, c2 = d == 1 ? (1, 1) : (1, d)
        e = bb_check(C1, c1, c2, nt, ntau, h, β,
            τ -> -c * ξb * bb_mat(ga, τ)[a1,a2] * bb_mat(gb, β - τ)[b2,b1],
            (t, t′) -> im * c * (bb_gtr(ga, t, t′)[a1,a2] * bb_les(gb, t′, t)[b2,b1] -
                                 bb_les(ga, t, t′)[a1,a2] * bb_gtr(gb, t′, t)[b2,b1]),
            (t′, t) -> im * c * bb_les(ga, t′, t)[a1,a2] * bb_gtr(gb, t, t′)[b2,b1],
            (t, τ) -> im * c * bb_tv(ga, t, τ)[a1,a2] * bb_vt(gb, τ, t)[b2,b1])
        errmax = max(errmax, e)
        # Bubble2: C(t,t') = i A_{a1a2}(t,t') B_{b1b2}(t,t')
        C2 = ℱ(C, ξa * ξb)
        for tstp = 0:nt
            if a1 == a2 == b1 == b2 == 1 && d == 1
                bubble2!(tstp, C2, A, Acc, B, Bcc)
            else
                bubble2!(tstp, C2, 1, d, A, Acc, a1, a2, B, Bcc, b1, b2)
            end
        end
        e = bb_check(C2, c1, c2, nt, ntau, h, β,
            τ -> -c * bb_mat(ga, τ)[a1,a2] * bb_mat(gb, τ)[b1,b2],
            (t, t′) -> im * c * (bb_gtr(ga, t, t′)[a1,a2] * bb_gtr(gb, t, t′)[b1,b2] -
                                 bb_les(ga, t, t′)[a1,a2] * bb_les(gb, t, t′)[b1,b2]),
            (t′, t) -> im * c * bb_les(ga, t′, t)[a1,a2] * bb_les(gb, t′, t)[b1,b2],
            (t, τ) -> im * c * bb_tv(ga, t, τ)[a1,a2] * bb_tv(gb, t, τ)[b1,b2])
        errmax = max(errmax, e)
    end
    return errmax
end

@testset "Bubble ($(kb_test_profile()))" begin
    nt, ntau, β, h = kb_release() ? (101, 201, 5.0, 0.02) : (11, 21, 2.0, 0.1)

    @testset "closed forms vs from_ham!" begin
        H = ComplexF64[0.4 0.3-0.2im; 0.3+0.2im -0.6]
        for ξ in (FERMI, BOSE)
            μ = ξ == FERMI ? 0.1 : -1.5
            C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
            G = ℱ(C, ξ)
            from_ham!(G, μ, H, β, h)
            Gx = bb_gf(C, BBFreeGF(H, μ, β, ξ), 1.0 + 0.0im, h, β)
            @test maximum(distance(G, Gx, n) for n = 0:nt) < 1e-10
        end
    end

    @testset "bubble1!/bubble2! $(d)x$(d) $(ξa == FERMI ? 'F' : 'B')$(ξb == FERMI ? 'F' : 'B')" for
            d in (1, 2), ξa in (FERMI, BOSE), ξb in (FERMI, BOSE)
        @test bb_run(nt, ntau, β, h, d, ξa, ξb) < 1e-10
    end

    @testset "explicit Acc != A and Bcc != B" begin
        for d in (1, 2), ξa in (FERMI, BOSE), ξb in (FERMI, BOSE)
            @test bb_run(nt, ntau, β, h, d, ξa, ξb;
                         cA = 0.6 + 0.8im, cB = -0.3 + 1.1im) < 1e-10
        end
    end

    @testset "overload equivalence and untouched elements" begin
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        H = ComplexF64[0.4 0.3-0.2im; 0.3+0.2im -0.6]
        A = bb_gf(C, BBFreeGF(H, 0.1, β, FERMI), 1.0 + 0.0im, h, β)
        for f in (bubble1!, bubble2!)
            X = ℱ(C, BOSE)
            Y = ℱ(C, BOSE)
            # A sentinel in the other orbital elements
            for m = 1:ntau
                X.mat[m] = fill(7.0 + 0.0im, 2, 2)
            end
            for tstp = 0:nt
                f(tstp, X, 2, 1, A, 1, 2, A, 2, 2)
                f(tstp, Y, 2, 1, A, A, 1, 2, A, A, 2, 2)
            end
            @test all(n -> distance(X, Y, n) == 0.0, 1:nt)
            @test all(m -> X.mat[m][1,1] == 7.0 && X.mat[m][1,2] == 7.0 && X.mat[m][2,2] == 7.0, 1:ntau)
            @test all(m -> X.mat[m][2,1] == Y.mat[m][2,1], 1:ntau)
            # Every method needs the time step as the first argument
            @test all(m -> m.sig.parameters[2] == Int64, methods(f))
        end
    end

    @testset "error paths" begin
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        A = ℱ(C, FERMI)
        X = ℱ(C, BOSE)
        @test_throws ArgumentError bubble1!(1, A, A, A)
        @test_throws ArgumentError bubble2!(1, ℱ(C, FERMI), A, A)
        @test_throws ArgumentError bubble1!(1, ℱ(C, BOSE), 1, 1, A, ℱ(C, BOSE), 1, 1, A, A, 1, 1)
        @test_throws BoundsError bubble1!(1, X, 3, 1, A, 1, 1, A, 1, 1)
        @test_throws BoundsError bubble2!(1, X, 1, 1, A, 0, 1, A, 1, 1)
        @test_throws BoundsError bubble2!(nt + 1, X, A, A)
        @test_throws BoundsError bubble1!(-1, X, A, A)
    end
end
