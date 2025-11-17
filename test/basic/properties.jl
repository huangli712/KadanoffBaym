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
        lmix1 = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix2 = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix1) == (ntime, ntau)
        @test getntime(lmix1) == ntime
        @test getntau(lmix1) == ntau
        @test getdims(lmix1) == (ndim1, ndim2)
        @test equaldims(lmix1) == (ndim1 == ndim2)
        @test iscompatible(lmix1, lmix2)
        @test iscompatible(C, lmix1)
        @test iscompatible(lmix2, C)
        for tstp = 1:getntime(lmix1)
            @test distance(lmix1, lmix2, tstp) < ϵ
        end
    end
    #
    @testset "Gˡᵉˢˢ Struct: Properties  " begin
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
        less1 = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less2 = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less1) == ntime
        @test getntime(less1) == ntime
        @test getdims(less1) == (ndim1, ndim2)
        @test equaldims(less1) == (ndim1 == ndim2)
        @test iscompatible(less1, less2)
        @test iscompatible(C, less1)
        @test iscompatible(less2, C)
        for tstp = 1:getntime(less1)
            @test distance(less1, less2, tstp) < ϵ
        end
    end
    #
    @testset "gᵐᵃᵗ  Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        tstp = 101
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        mat1 = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat2 = gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat3 = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat1) == ntau
        @test getntau(mat1) == ntau
        @test getdims(mat1) == (ndim1, ndim2)
        @test equaldims(mat1) == (ndim1 == ndim2)
        @test iscompatible(mat1, mat2)
        @test iscompatible(mat1, mat3)
        @test iscompatible(mat3, mat2)
        @test iscompatible(C, mat1)
        @test iscompatible(mat2, C)
        @test distance(mat1, mat2) < ϵ
        @test distance(mat1, mat3) < ϵ
        @test distance(mat3, mat2) < ϵ
    end
    #
    @testset "gʳᵉᵗ  Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        tstp = 101
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        ret1 = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret2 = gʳᵉᵗ(tstp, ndim1, ndim2)
        ret3 = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret1) == tstp
        @test gettstp(ret1) == tstp
        @test getdims(ret1) == (ndim1, ndim2)
        @test equaldims(ret1) == (ndim1 == ndim2)
        @test iscompatible(ret1, ret2)
        @test iscompatible(ret1, ret3)
        @test iscompatible(ret3, ret2)
        @test iscompatible(C, ret1)
        @test iscompatible(ret2, C)
        @test distance(ret1, ret2) < ϵ
        @test distance(ret1, ret3, tstp) < ϵ
        @test distance(ret3, ret2, tstp) < ϵ
    end
    #
    @testset "gˡᵐⁱˣ Struct: Properties  " begin
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
        lmix1 = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix2 = gˡᵐⁱˣ(ntau, ndim1, ndim2)
        lmix3 = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix1) == ntau
        @test getntau(lmix1) == ntau
        @test getdims(lmix1) == (ndim1, ndim2)
        @test equaldims(lmix1) == (ndim1 == ndim2)
        @test iscompatible(lmix1, lmix2)
        @test iscompatible(lmix1, lmix3)
        @test iscompatible(lmix3, lmix2)
        @test iscompatible(C, lmix1)
        @test iscompatible(lmix2, C)
        @test distance(lmix1, lmix2) < ϵ
        for tstp = 1:getntime(lmix3)
            @test distance(lmix1, lmix3, tstp) < ϵ
            @test distance(lmix3, lmix2, tstp) < ϵ
        end
    end
    #
    @testset "gˡᵉˢˢ Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        tstp = 101
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        less1 = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less2 = gˡᵉˢˢ(tstp, ndim1, ndim2)
        less3 = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less1) == tstp
        @test gettstp(less1) == tstp
        @test getdims(less1) == (ndim1, ndim2)
        @test equaldims(less1) == (ndim1 == ndim2)
        @test iscompatible(less1, less2)
        @test iscompatible(less1, less3)
        @test iscompatible(less3, less2)
        @test iscompatible(C, less1)
        @test iscompatible(less2, C)
        @test distance(less1, less2) < ϵ
        @test distance(less1, less3, tstp) < ϵ
        @test distance(less3, less2, tstp) < ϵ
    end
    #
    @testset "ℱ     Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        cfm1 = ℱ(C, v, sign)
        cfm2 = ℱ(C, sign)
        #
        @test getsign(cfm1) == sign
        @test getntime(cfm1) == ntime
        @test getntau(cfm1) == ntau
        @test getdims(cfm1) == (ndim1, ndim2)
        @test equaldims(cfm1) == (ndim1 == ndim2)
        for tstp = 0:getntime(cfm1)
            @test distance(cfm1, cfm2, tstp) < ϵ
        end
    end
    #
    @testset "𝒻     Struct: Properties  " begin
    end
end

println("All tests pass!\n")
