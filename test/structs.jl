haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true failfast = true "KadanoffBaym: structs.jl" begin
    @testset "Cn Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        #
        C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        C₂ = Cn(ndim1, ndim2, tmax, beta)
        C₃ = Cn(ndim1, tmax, beta)
        C₄ = Cn(tmax, beta)
        #
        @test C₁ == C₂
        @test C₁ == C₃
        @test C₁ == C₄
    end
    #
    @testset "Cf Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 0.0 + 0.0im
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        cf₁ = Cf(ntime, ndim1, ndim2, v)
        cf₂ = Cf(ntime, ndim1, ndim2)
        cf₃ = Cf(ntime, ndim1)
        cf₄ = Cf(ntime, x)
        cf₅ = Cf(C, x)
        cf₆ = Cf(C, v)
        cf₇ = Cf(C)
        #
        @test cf₁ == cf₂
        @test cf₁ == cf₃
    end
end
