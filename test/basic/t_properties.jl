#
# t_properties.jl
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat₁) == ntau
        @test getntau(mat₁) == ntau
        @test getdims(mat₁) == (ndim1, ndim2)
        @test equaldims(mat₁) == (ndim1 == ndim2)
        @test iscompatible(mat₁, mat₂)
        @test iscompatible(C, mat₁)
        @test iscompatible(mat₂, C)
        @test distance(mat₁, mat₂) < ϵ
    end
    #
    @testset "Gʳᵉᵗ  Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret₁) == ntime
        @test getntime(ret₁) == ntime
        @test getdims(ret₁) == (ndim1, ndim2)
        @test equaldims(ret₁) == (ndim1 == ndim2)
        @test iscompatible(ret₁, ret₂)
        @test iscompatible(C, ret₁)
        @test iscompatible(ret₂, C)
        for tstp = 1:getntime(ret₁)
            @test distance(ret₁, ret₂, tstp) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix₂ = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix₁) == (ntime, ntau)
        @test getntime(lmix₁) == ntime
        @test getntau(lmix₁) == ntau
        @test getdims(lmix₁) == (ndim1, ndim2)
        @test equaldims(lmix₁) == (ndim1 == ndim2)
        @test iscompatible(lmix₁, lmix₂)
        @test iscompatible(C, lmix₁)
        @test iscompatible(lmix₂, C)
        for tstp = 1:getntime(lmix₁)
            @test distance(lmix₁, lmix₂, tstp) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less₂ = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less₁) == ntime
        @test getntime(less₁) == ntime
        @test getdims(less₁) == (ndim1, ndim2)
        @test equaldims(less₁) == (ndim1 == ndim2)
        @test iscompatible(less₁, less₂)
        @test iscompatible(C, less₁)
        @test iscompatible(less₂, C)
        for tstp = 1:getntime(less₁)
            @test distance(less₁, less₂, tstp) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat₃ = Gᵐᵃᵗ(C, v)
        #
        @test getsize(mat₁) == ntau
        @test getntau(mat₁) == ntau
        @test getdims(mat₁) == (ndim1, ndim2)
        @test equaldims(mat₁) == (ndim1 == ndim2)
        @test iscompatible(mat₁, mat₂)
        @test iscompatible(mat₁, mat₃)
        @test iscompatible(mat₃, mat₂)
        @test iscompatible(C, mat₁)
        @test iscompatible(mat₂, C)
        @test distance(mat₁, mat₂) < ϵ
        @test distance(mat₁, mat₃) < ϵ
        @test distance(mat₃, mat₂) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2)
        ret₃ = Gʳᵉᵗ(C, v)
        #
        @test getsize(ret₁) == tstp
        @test gettstp(ret₁) == tstp
        @test getdims(ret₁) == (ndim1, ndim2)
        @test equaldims(ret₁) == (ndim1 == ndim2)
        @test iscompatible(ret₁, ret₂)
        @test iscompatible(ret₁, ret₃)
        @test iscompatible(ret₃, ret₂)
        @test iscompatible(C, ret₁)
        @test iscompatible(ret₂, C)
        @test distance(ret₁, ret₂) < ϵ
        @test distance(ret₁, ret₃, tstp) < ϵ
        @test distance(ret₃, ret₂, tstp) < ϵ
    end
    #
    @testset "gˡᵐⁱˣ Struct: Properties  " begin
        ntime = 101
        ntau = 51
        ndim1 = 2
        ndim2 = 3
        tmax = 5.0
        beta = 4.0
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2)
        lmix₃ = Gˡᵐⁱˣ(C, v)
        #
        @test getsize(lmix₁) == ntau
        @test getntau(lmix₁) == ntau
        @test getdims(lmix₁) == (ndim1, ndim2)
        @test equaldims(lmix₁) == (ndim1 == ndim2)
        @test iscompatible(lmix₁, lmix₂)
        @test iscompatible(lmix₁, lmix₃)
        @test iscompatible(lmix₃, lmix₂)
        @test iscompatible(C, lmix₁)
        @test iscompatible(lmix₂, C)
        @test distance(lmix₁, lmix₂) < ϵ
        for tstp = 1:getntime(lmix₃)
            @test distance(lmix₁, lmix₃, tstp) < ϵ
            @test distance(lmix₃, lmix₂, tstp) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2)
        less₃ = Gˡᵉˢˢ(C, v)
        #
        @test getsize(less₁) == tstp
        @test gettstp(less₁) == tstp
        @test getdims(less₁) == (ndim1, ndim2)
        @test equaldims(less₁) == (ndim1 == ndim2)
        @test iscompatible(less₁, less₂)
        @test iscompatible(less₁, less₃)
        @test iscompatible(less₃, less₂)
        @test iscompatible(C, less₁)
        @test iscompatible(less₂, C)
        @test distance(less₁, less₂) < ϵ
        @test distance(less₁, less₃, tstp) < ϵ
        @test distance(less₃, less₂, tstp) < ϵ
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
        #
        cfm₁ = ℱ(C, v, sign)
        cfm₂ = ℱ(C, sign)
        #
        @test getsign(cfm₁) == sign
        @test getntime(cfm₁) == ntime
        @test getntau(cfm₁) == ntau
        @test getdims(cfm₁) == (ndim1, ndim2)
        @test equaldims(cfm₁) == (ndim1 == ndim2)
        for tstp = 0:getntime(cfm₁)
            @test distance(cfm₁, cfm₂, tstp) < ϵ
        end
        @test cfm₁.matm == Gᵐᵃᵗᵐ(sign, cfm₁.mat)
        @test cfm₁.adv == Gᵃᵈᵛ(cfm₁.ret)
        @test cfm₁.rmix == Gʳᵐⁱˣ(sign, cfm₁.lmix)
        @test cfm₁.gtr == Gᵍᵗʳ(cfm₁.less, cfm₁.ret)
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
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v = zero(C64)
        ϵ = 1.0e-7
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
