#
# structs.jl
#
# To test constructors and `==` operators for contour-ordered Green's
# functions.
#

# We setup PCONTOUR, such that `Cn()` can work correctly.
rev_dict_c(_PCONTOUR)

@testset verbose = true "KadanoffBaym: structs.jl" begin
    @testset "Cn    Struct: Constructors" begin
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
        C₅ = Cn()
        #
        @test C₁ == C₂
        @test C₁ == C₃
        @test C₁ == C₄
        @test C₁ != C₅ # Their dimensional sizes don't match.
    end
    #
    @testset "Cf    Struct: Constructors" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
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
        cf₈ = Cf()
        #
        @test cf₁ == cf₂
        @test cf₁ == cf₃
        @test cf₁ == cf₄
        @test cf₁ == cf₅
        @test cf₁ == cf₆
        @test cf₁ == cf₇
        @test cf₁ != cf₈ # Their dimensional sizes don't match.
    end
    #
    @testset "Gᵐᵃᵗ  Struct: Constructors" begin
        type = "mat"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = Gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat₃ = Gᵐᵃᵗ(ntau, ndim1)
        mat₄ = Gᵐᵃᵗ(ntau, x)
        mat₅ = Gᵐᵃᵗ(C, x)
        mat₆ = Gᵐᵃᵗ(C, v)
        mat₇ = Gᵐᵃᵗ(C)
        mat₈ = Gᵐᵃᵗ()
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
        @test mat₁ == mat₄
        @test mat₁ == mat₅
        @test mat₁ == mat₆
        @test mat₁ == mat₇
        @test mat₁ != mat₈ # Their dimensional sizes don't match.
    end
    #
    @testset "Gʳᵉᵗ  Struct: Constructors" begin
        type = "ret"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(ntime, ndim1, ndim2)
        ret₃ = Gʳᵉᵗ(ntime, ndim1)
        ret₄ = Gʳᵉᵗ(ntime, x)
        ret₅ = Gʳᵉᵗ(C, x)
        ret₆ = Gʳᵉᵗ(C, v)
        ret₇ = Gʳᵉᵗ(C)
        ret₈ = Gʳᵉᵗ()
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
        @test ret₁ == ret₄
        @test ret₁ == ret₅
        @test ret₁ == ret₆
        @test ret₁ == ret₇
        @test ret₁ != ret₈ # Their dimensional sizes don't match.
    end
    #
    @testset "Gˡᵐⁱˣ Struct: Constructors" begin
        type = "lmix"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix₂ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2)
        lmix₃ = Gˡᵐⁱˣ(ntime, ntau, ndim1)
        lmix₄ = Gˡᵐⁱˣ(ntime, ntau, x)
        lmix₅ = Gˡᵐⁱˣ(C, x)
        lmix₆ = Gˡᵐⁱˣ(C, v)
        lmix₇ = Gˡᵐⁱˣ(C)
        lmix₈ = Gˡᵐⁱˣ()
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
        @test lmix₁ == lmix₄
        @test lmix₁ == lmix₅
        @test lmix₁ == lmix₆
        @test lmix₁ == lmix₇
        @test lmix₁ != lmix₈ # Their dimensional sizes don't match.
    end
    #
    @testset "Gˡᵉˢˢ Struct: Constructors" begin
        type = "less"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less₂ = Gˡᵉˢˢ(ntime, ndim1, ndim2)
        less₃ = Gˡᵉˢˢ(ntime, ndim1)
        less₄ = Gˡᵉˢˢ(ntime, x)
        less₅ = Gˡᵉˢˢ(C, x)
        less₆ = Gˡᵉˢˢ(C, v)
        less₇ = Gˡᵉˢˢ(C)
        less₈ = Gˡᵉˢˢ()
        #
        @test less₁ == less₂
        @test less₁ == less₃
        @test less₁ == less₄
        @test less₁ == less₅
        @test less₁ == less₆
        @test less₁ == less₇
        @test less₁ != less₈ # Their dimensional sizes don't match.
    end
    #
    @testset "Gᵐᵃᵗᵐ Struct: Constructors" begin
        type = "matm"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = Gᵐᵃᵗ(C)
        mat₃ = Gᵐᵃᵗ()
        matm₁ = Gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = Gᵐᵃᵗᵐ(sign, mat₂)
        matm₃ = Gᵐᵃᵗᵐ(sign, mat₃)
        #
        @test mat₁ == mat₂
        @test mat₁ != mat₃ # Their dimensional sizes don't match.
        @test matm₁ == matm₂
        @test matm₁ != matm₃ # Their dimensional sizes don't match.
    end
    #
    @testset "Gᵃᵈᵛ  Struct: Constructors" begin
        type = "adv"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(C)
        ret₃ = Gʳᵉᵗ()
        adv₁ = Gᵃᵈᵛ(ret₁)
        adv₂ = Gᵃᵈᵛ(ret₂)
        adv₃ = Gᵃᵈᵛ(ret₃)
        #
        @test ret₁ == ret₂
        @test ret₁ != ret₃ # Their dimensional sizes don't match.
        @test adv₁ == adv₂
        @test adv₁ != adv₃ # Their dimensional sizes don't match.
    end
    #
    @testset "Gʳᵐⁱˣ Struct: Constructors" begin
        type = "rmix"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v)
        lmix₂ = Gˡᵐⁱˣ(C)
        lmix₃ = Gˡᵐⁱˣ()
        rmix₁ = Gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = Gʳᵐⁱˣ(sign, lmix₂)
        rmix₃ = Gʳᵐⁱˣ(sign, lmix₃)
        #
        @test lmix₁ == lmix₂
        @test lmix₁ != lmix₃
        @test rmix₁ == rmix₂
        @test rmix₁ != rmix₃ # Their dimensional sizes don't match.
    end
    #
    @testset "Gᵍᵗʳ  Struct: Constructors" begin
        type = "gtr"
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = zero(C64)
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        x = zeros(C64, ndim1, ndim2)
        #
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v)
        less₂ = Gˡᵉˢˢ(C)
        less₃ = Gˡᵉˢˢ()
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v)
        ret₂ = Gʳᵉᵗ(C)
        ret₃ = Gʳᵉᵗ()
        gtr₁ = Gᵍᵗʳ(less₁, ret₁)
        gtr₂ = Gᵍᵗʳ(less₂, ret₂)
        gtr₃ = Gᵍᵗʳ(less₃, ret₃)
        #
        @test less₁ == less₂
        @test less₁ != less₃ # Their dimensional sizes don't match.
        @test ret₁ == ret₂
        @test ret₁ != ret₃ # Their dimensional sizes don't match.
        @test gtr₁ == gtr₂
        @test gtr₁ != gtr₃ # Their dimensional sizes don't match.
    end
    #
    @testset "gᵐᵃᵗ  Struct: Constructors" begin
        type = "mat"
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2)
        mat₃ = gᵐᵃᵗ(ntau, ndim1)
        mat₄ = gᵐᵃᵗ(ntau, x)
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
        @test mat₁ == mat₄
    end
    #
    @testset "gʳᵉᵗ  Struct: Constructors" begin
        type = "ret"
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2)
        ret₃ = gʳᵉᵗ(tstp, ndim1)
        ret₄ = gʳᵉᵗ(tstp, x)
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
        @test ret₁ == ret₄
    end
    #
    @testset "gˡᵐⁱˣ Struct: Constructors" begin
        type = "lmix"
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1)
        lmix₄ = gˡᵐⁱˣ(ntau, x)
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
        @test lmix₁ == lmix₄
    end
    #
    @testset "gˡᵉˢˢ Struct: Constructors" begin
        type = "less"
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2)
        less₃ = gˡᵉˢˢ(tstp, ndim1)
        less₄ = gˡᵉˢˢ(tstp, x)
        #
        @test less₁ == less₂
        @test less₁ == less₃
        @test less₁ == less₄
    end
    #
    @testset "gᵐᵃᵗᵐ Struct: Constructors" begin
        type = "matm"
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, x)
        matm₁ = gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = gᵐᵃᵗᵐ(sign, mat₂)
        #
        @test mat₁ == mat₂
        @test matm₁ == matm₂
    end
    #
    @testset "gᵃᵈᵛ  Struct: Constructors" begin
        type = "adv"
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, x)
        adv₁ = gᵃᵈᵛ(ret₁)
        adv₂ = gᵃᵈᵛ(ret₂)
        #
        @test ret₁ == ret₂
        @test adv₁ == adv₂
    end
    #
    @testset "gʳᵐⁱˣ Struct: Constructors" begin
        type = "rmix"
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, x)
        rmix₁ = gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = gʳᵐⁱˣ(sign, lmix₂)
        #
        @test lmix₁ == lmix₂
        @test rmix₁ == rmix₂
    end
    #
    @testset "gᵍᵗʳ  Struct: Constructors" begin
        type = "gtr"
        tstp = 201
        ndim1 = 2
        ndim2 = 2
        v = zero(C64)
        #
        x = zeros(C64, ndim1, ndim2)
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v)
        less₂ = gˡᵉˢˢ(tstp, x)
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, x)
        gtr₁ = gᵍᵗʳ(less₁, ret₁)
        gtr₂ = gᵍᵗʳ(less₂, ret₂)
        #
        @test less₁ == less₂
        @test ret₁ == ret₂
        @test gtr₁ == gtr₂
    end
end

println("All tests pass!\n")
