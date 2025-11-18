#
# properties.jl
#
# To test the `getxxx()` functions for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: properties.jl" begin
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
        cf₁ = Cf(ntime, ndim1, ndim2, v)
        cf₂ = Cf(C, v)
        #
        @test getsize(cf₁) == ntime
        @test getntime(cf₁) == ntime
        @test getdims(cf₁) == (ndim1, ndim2)
        @test equaldims(cf₁) == (ndim1 == ndim2)
        @test iscompatible(cf₁, cf₂)
        @test iscompatible(cf₁, C)
        @test iscompatible(C, cf₂)
        @test distance(cf₁, cf₂) < ϵ
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
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat2 = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat₁) == ntau
        @test getntau(mat₁) == ntau
        @test getdims(mat₁) == (ndim1, ndim2)
        @test equaldims(mat₁) == (ndim1 == ndim2)
        @test iscompatible(mat₁, mat2)
        @test iscompatible(C, mat₁)
        @test iscompatible(mat2, C)
        @test distance(mat₁, mat2) < ϵ
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
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret2 = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret₁) == ntime
        @test getntime(ret₁) == ntime
        @test getdims(ret₁) == (ndim1, ndim2)
        @test equaldims(ret₁) == (ndim1 == ndim2)
        @test iscompatible(ret₁, ret2)
        @test iscompatible(C, ret₁)
        @test iscompatible(ret2, C)
        for tstp = 1:getntime(ret₁)
            @test distance(ret₁, ret2, tstp) < ϵ
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
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix2 = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix₁) == (ntime, ntau)
        @test getntime(lmix₁) == ntime
        @test getntau(lmix₁) == ntau
        @test getdims(lmix₁) == (ndim1, ndim2)
        @test equaldims(lmix₁) == (ndim1 == ndim2)
        @test iscompatible(lmix₁, lmix2)
        @test iscompatible(C, lmix₁)
        @test iscompatible(lmix2, C)
        for tstp = 1:getntime(lmix₁)
            @test distance(lmix₁, lmix2, tstp) < ϵ
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
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less2 = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less₁) == ntime
        @test getntime(less₁) == ntime
        @test getdims(less₁) == (ndim1, ndim2)
        @test equaldims(less₁) == (ndim1 == ndim2)
        @test iscompatible(less₁, less2)
        @test iscompatible(C, less₁)
        @test iscompatible(less2, C)
        for tstp = 1:getntime(less₁)
            @test distance(less₁, less2, tstp) < ϵ
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
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat2 = gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat3 = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat₁) == ntau
        @test getntau(mat₁) == ntau
        @test getdims(mat₁) == (ndim1, ndim2)
        @test equaldims(mat₁) == (ndim1 == ndim2)
        @test iscompatible(mat₁, mat2)
        @test iscompatible(mat₁, mat3)
        @test iscompatible(mat3, mat2)
        @test iscompatible(C, mat₁)
        @test iscompatible(mat2, C)
        @test distance(mat₁, mat2) < ϵ
        @test distance(mat₁, mat3) < ϵ
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
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret2 = gʳᵉᵗ(tstp, ndim1, ndim2)
        ret3 = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret₁) == tstp
        @test gettstp(ret₁) == tstp
        @test getdims(ret₁) == (ndim1, ndim2)
        @test equaldims(ret₁) == (ndim1 == ndim2)
        @test iscompatible(ret₁, ret2)
        @test iscompatible(ret₁, ret3)
        @test iscompatible(ret3, ret2)
        @test iscompatible(C, ret₁)
        @test iscompatible(ret2, C)
        @test distance(ret₁, ret2) < ϵ
        @test distance(ret₁, ret3, tstp) < ϵ
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
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix2 = gˡᵐⁱˣ(ntau, ndim1, ndim2)
        lmix3 = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix₁) == ntau
        @test getntau(lmix₁) == ntau
        @test getdims(lmix₁) == (ndim1, ndim2)
        @test equaldims(lmix₁) == (ndim1 == ndim2)
        @test iscompatible(lmix₁, lmix2)
        @test iscompatible(lmix₁, lmix3)
        @test iscompatible(lmix3, lmix2)
        @test iscompatible(C, lmix₁)
        @test iscompatible(lmix2, C)
        @test distance(lmix₁, lmix2) < ϵ
        for tstp = 1:getntime(lmix3)
            @test distance(lmix₁, lmix3, tstp) < ϵ
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
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less2 = gˡᵉˢˢ(tstp, ndim1, ndim2)
        less3 = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less₁) == tstp
        @test gettstp(less₁) == tstp
        @test getdims(less₁) == (ndim1, ndim2)
        @test equaldims(less₁) == (ndim1 == ndim2)
        @test iscompatible(less₁, less2)
        @test iscompatible(less₁, less3)
        @test iscompatible(less3, less2)
        @test iscompatible(C, less₁)
        @test iscompatible(less2, C)
        @test distance(less₁, less2) < ϵ
        @test distance(less₁, less3, tstp) < ϵ
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
        @test cfm1.matm == Gᵐᵃᵗᵐ(sign, cfm1.mat)
        @test cfm1.adv == Gᵃᵈᵛ(cfm1.ret)
        @test cfm1.rmix == Gʳᵐⁱˣ(sign, cfm1.lmix)
        @test cfm1.gtr == Gᵍᵗʳ(cfm1.less, cfm1.ret)
    end
    #
    @testset "𝒻     Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        tstp = 101
        sign = FERMI
        ϵ = 1.0e-7
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        #
        cfv₁ = 𝒻(C, tstp, v, sign)
        cfv₂ = 𝒻(C, tstp, sign)
        cfv₃ = 𝒻(C, 0, v, sign) # tstp = 0
        cfv₄ = 𝒻(C, 0, sign) # tstp = 0
        cfm₁ = ℱ(C, v, sign)
        cfm₂ = ℱ(C, sign)
        #
        @test getsign(cfv₁) == sign
        @test gettstp(cfv₁) == tstp
        @test getntau(cfv₁) == ntau
        @test getdims(cfv₁) == (ndim1, ndim2)
        @test equaldims(cfv₁) == (ndim1 == ndim2)
        @test distance(cfv₁, cfv₂, tstp) < ϵ
        @test distance(cfv₁, cfm₁, tstp) < ϵ
        @test distance(cfm₂, cfv₁, tstp) < ϵ
        @test distance(cfv₃, cfv₄, 0) < ϵ
        @test distance(cfv₃, cfm₁, 0) < ϵ
        @test distance(cfm₂, cfv₃, 0) < ϵ
        @test cfv₁.matm == gᵐᵃᵗᵐ(sign, cfv₁.mat)
        @test cfv₁.adv == gᵃᵈᵛ(cfv₁.ret)
        @test cfv₁.rmix == gʳᵐⁱˣ(sign, cfv₁.lmix)
        @test cfv₁.gtr == gᵍᵗʳ(cfv₁.less, cfv₁.ret)
    end
end

println("All tests pass!\n")
