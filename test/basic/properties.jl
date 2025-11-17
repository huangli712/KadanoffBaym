#
# properties.jl
#
# To test the `getxxx()` functions.
#

@testset verbose = true "KadanoffBaym: structs.jl" begin
    @testset "Cn    Struct: Properties  " begin
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
    @testset "Cf    Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        cf1 = Cf(ntime, ndim1, ndim2, v)
        cf2 = Cf(C, v)
        #
        @test getsize(cf1) == ntime
        @test getntime(cf1) == ntime
        @test getdims(cf1) == (ndim1, ndim2)
        @test equaldims(cf1) == (ndim1 == ndim2)
        @test iscompatible(cf1, cf2)
        @test iscompatible(cf1, C)
        @test iscompatible(C, cf2)
        @test distance(cf1, cf2) < ϵ
    end
    #
    @testset "Gᵐᵃᵗ  Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        mat1 = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat2 = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat1) == ntau
        @test getntau(mat1) == ntau
        @test getdims(mat1) == (ndim1, ndim2)
        @test equaldims(mat1) == (ndim1 == ndim2)
        @test iscompatible(mat1, mat2)
        @test iscompatible(C, mat1)
        @test iscompatible(mat2, C)
        @test distance(mat1, mat2) < ϵ
    end
    #
    @testset "Gʳᵉᵗ  Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        ret1 = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret2 = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret1) == ntime
        @test getntime(ret1) == ntime
        @test getdims(ret1) == (ndim1, ndim2)
        @test equaldims(ret1) == (ndim1 == ndim2)
        @test iscompatible(ret1, ret2)
        @test iscompatible(C, ret1)
        @test iscompatible(ret2, C)
        for tstp = 1:getntime(ret1)
            @test distance(ret1, ret2, tstp) < ϵ
        end
    end
    #
    @testset "Gˡᵐⁱˣ Struct: Properties  " begin
    end
    #
    @testset "gᵐᵃᵗ  Struct: Properties  " begin
    end
    #
    @testset "gʳᵉᵗ  Struct: Properties  " begin
    end
    #
    @testset "gˡᵐⁱˣ Struct: Properties  " begin
    end
    #
    @testset "gˡᵉˢˢ Struct: Properties  " begin
    end
end

println("All tests pass!\n")
