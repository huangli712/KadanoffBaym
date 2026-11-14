#
# t_green_single_pole.jl
#
# To test the bosonic single pole Green's function of X = (b + b†)/√2.
#
# NESSi Catch tests ported here:
#   libcntr/test/green_single_pole.cpp:13-134
#

using Test
using LinearAlgebra
using KadanoffBaym

#
# Reference formulas ported from libcntr/test/green_single_pole.cpp
#
sp_fb(w0, β) = 1.0 / (exp(β * w0) - 1.0)
sp_mat(w0, β, τ) = -(1.0 + sp_fb(w0, β)) * exp(-w0 * τ) - sp_fb(w0, β) * exp(w0 * τ)
sp_ret(w0, t1, t2) = complex(-2.0 * sin(w0 * (t1 - t2)) * (t1 ≥ t2 ? 1.0 : 0.0))
sp_less(w0, β, t1, t2) = -im * sp_fb(w0, β) * exp(-im * w0 * (t1 - t2)) -
                         im * (1.0 + sp_fb(w0, β)) * exp(im * w0 * (t1 - t2))
sp_lmix(w0, β, t1, τ2) = -im * sp_fb(w0, β) * exp(τ2 * w0) * exp(-im * w0 * t1) -
                         im * (1.0 + sp_fb(w0, β)) * exp(-τ2 * w0) * exp(im * w0 * t1)

function sp_reference(C::Cn, w0, β, dt)
    nt, ntau = getntime(C), getntau(C)
    δτ = β / (ntau - 1)
    D = ℱ(C, BOSE)
    for i = 1:ntau
        D.mat[i] = fill(complex(sp_mat(w0, β, (i - 1) * δτ) / 2.0), 1, 1)
    end
    for i2 = 1:ntau, i1 = 1:nt
        D.lmix[i1, i2] = fill(sp_lmix(w0, β, (i1 - 1) * dt, (i2 - 1) * δτ) / 2.0, 1, 1)
    end
    for i2 = 1:nt, i1 = 1:i2
        D.less[i1, i2] = fill(sp_less(w0, β, (i1 - 1) * dt, (i2 - 1) * dt) / 2.0, 1, 1)
        D.ret[i2, i1] = fill(sp_ret(w0, (i2 - 1) * dt, (i1 - 1) * dt) / 2.0, 1, 1)
    end
    return D
end

@testset "green_single_pole: NESSi Catch" begin
    nt, ntau, w0, β, dt = 100, 100, 1.4, 20.0, 0.01
    C = Cn(nt + 1, ntau + 1, 1, 1, nt * dt, β)
    D0 = ℱ(C, BOSE)
    green_single_pole!(D0, w0, β, dt)
    D_ref = sp_reference(C, w0, β, dt)
    #
    err = sum(distance(D_ref, D0, tstp) for tstp = 0:nt+1)
    @test err < 1e-7
    #
    # Point-wise error of every component
    for s in (:mat, :ret, :lmix, :less)
        @test maximum(abs.(getfield(D0, s).data .- getfield(D_ref, s).data)) < 1e-12
    end
    for tstp = 0:nt+1
        @test distance(D_ref, D0, tstp) < 1e-12
    end
end

@testset "green_single_pole: matsubara, timestep, and full" begin
    nt, ntau, w0, β, dt = 21, 31, 0.8, 3.0, 0.05
    C = Cn(nt, ntau, 2, 2, (nt - 1) * dt, β)
    Df = ℱ(C, BOSE)
    green_single_pole!(Df, w0, β, dt)
    #
    Ds = ℱ(C, BOSE)
    green_single_pole_mat!(Ds, w0, β)
    for nkb = 1:nt
        green_single_pole_timestep!(nkb, Ds, w0, β, dt)
    end
    @test Ds == Df
    #
    # Only the (1,1) element is written
    for s in (:mat, :ret, :lmix, :less)
        X = getfield(Df, s).data
        @test all(iszero, X[2, 2, ntuple(_ -> Colon(), ndims(X) - 2)...])
        @test all(iszero, X[1, 2, ntuple(_ -> Colon(), ndims(X) - 2)...])
        @test all(iszero, X[2, 1, ntuple(_ -> Colon(), ndims(X) - 2)...])
    end
    #
    # The (1,1) element agrees with the 1 × 1 reference
    C1 = Cn(nt, ntau, 1, 1, (nt - 1) * dt, β)
    D_ref = sp_reference(C1, w0, β, dt)
    for s in (:mat, :ret, :lmix, :less)
        X = getfield(Df, s).data
        R = getfield(D_ref, s).data
        @test maximum(abs.(X[1:1, 1:1, ntuple(_ -> Colon(), ndims(X) - 2)...] .- R)) < 1e-12
    end
end

@testset "green_single_pole: error paths" begin
    C = Cn(5, 11, 1, 1, 0.4, 2.0)
    @test_throws ArgumentError green_single_pole!(ℱ(C, FERMI), 1.0, 2.0, 0.1)
    @test_throws ArgumentError green_single_pole_mat!(ℱ(C, FERMI), 1.0, 2.0)
    @test_throws ArgumentError green_single_pole!(ℱ(C, BOSE), -1.0, 2.0, 0.1)
    @test_throws ArgumentError green_single_pole!(ℱ(C, BOSE), 0.0, 2.0, 0.1)
    @test_throws ArgumentError green_single_pole_timestep!(0, ℱ(C, BOSE), 1.0, 2.0, 0.1)
    @test_throws ArgumentError green_single_pole_timestep!(6, ℱ(C, BOSE), 1.0, 2.0, 0.1)
end
