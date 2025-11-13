haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true "KadanoffBaym: structs.jl" begin
    @testset "Cn    Struct: Properties" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        @test getsize(C) == (ntime, ntau)
        @test getntime(C) == ntime
        @test getntau(C) == ntau
        @test getdims(C) == (ndim1, ndim2)
        @test gettmax(C) == tmax
        @test getbeta(C) == beta
        @test getdt(C) ==  tmax / (ntime - 1)
        @test getdtau(C) == beta / (ntau - 1)
        @test equaldims(C) == (ndim1 == ndim2)
    end
    #
    @testset "Cf    Struct: Properties" begin
        ntime = 101
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        cf = Cf(ntime, ndim1, ndim2, v)
        #
        @test getsize(cf) == ntime
        @test getntime(cf) == ntime
        @test getdims(cf) == (ndim1, ndim2)
        @test equaldims(cf) == (ndim1 == ndim2)
    end
    #
    @testset "Gᵐᵃᵗ  Struct: Properties" begin
    end
    #
    @testset "Gʳᵉᵗ  Struct: Properties" begin
    end
    #
    @testset "Gˡᵐⁱˣ Struct: Properties" begin
    end
    #
    @testset "gᵐᵃᵗ  Struct: Properties" begin
    end
    #
    @testset "gʳᵉᵗ  Struct: Properties" begin
    end
    #
    @testset "gˡᵐⁱˣ Struct: Properties" begin
    end
    #
    @testset "gˡᵉˢˢ Struct: Properties" begin
    end
end
