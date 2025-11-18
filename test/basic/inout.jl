#
# inout.jl
#
# To test read and write operations for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: inout.jl" begin
    @testset "Cn    Struct: read/write" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        fn = "Cn.data"
        #
        C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        C₂ = Cn(2.0, 2.0)
        C₃ = Cn(3.0, 3.0)
        #
        @test C₁ != C₂
        @test C₁ != C₃
        #
        write(fn, C₁)
        read!(fn, C₂)
        open(fn, "r") do fin
            read!(fin, C₃)
        end
        #
        @test C₁ == C₂
        @test C₁ == C₃
    end
    #
    @testset "Cf    Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "Cf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cf₁ = Cf(C, v₁)
        cf₂ = Cf(C, v₂)
        cf₃ = Cf(C, v₃)
        #
        @test cf₁ != cf₂
        @test cf₁ != cf₃
        #
        write(fn, cf₁)
        read!(fn, cf₂)
        open(fn, "r") do fin
            read!(fin, cf₃)
        end
        #
        @test cf₁ == cf₂
        @test cf₁ == cf₃
    end
    #
    @testset "Cf    Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "Cf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cf₁ = Cf(C, v₁)
        cf₂ = Cf(C, v₂)
        cf₃ = Cf(C, v₃)
        #
        @test cf₁ != cf₂
        @test cf₁ != cf₃
        #
        write(fn, cf₁)
        read!(fn, cf₂)
        open(fn, "r") do fin
            read!(fin, cf₃)
        end
        #
        @test cf₁ == cf₂
        @test cf₁ == cf₃
    end
    #
    @testset "Gᵐᵃᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "mat.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        mat₁ = Gᵐᵃᵗ(C, v₁)
        mat₂ = Gᵐᵃᵗ(C, v₂)
        mat₃ = Gᵐᵃᵗ(C, v₃)
        #
        @test mat₁ != mat₂
        @test mat₁ != mat₃
        #
        write(fn, mat₁)
        read!(fn, mat₂)
        open(fn, "r") do fin
            read!(fin, mat₃)
        end
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
    end
    #
    @testset "Gᵐᵃᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "mat.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        mat₁ = Gᵐᵃᵗ(C, v₁)
        mat₂ = Gᵐᵃᵗ(C, v₂)
        mat₃ = Gᵐᵃᵗ(C, v₃)
        #
        @test mat₁ != mat₂
        @test mat₁!= mat₃
        #
        write(fn, mat₁)
        read!(fn, mat₂)
        open(fn, "r") do fin
            read!(fin, mat₃)
        end
        #
        @test mat₁ == mat₂
        @test mat₁ == mat₃
    end
    #
    @testset "Gʳᵉᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret = Gʳᵉᵗ(C, v)
        ret₂ = Gʳᵉᵗ(C, v₂)
        ret₃ = Gʳᵉᵗ(C, v₃)
        #
        @test ret != ret₂
        @test ret != ret₃
        #
        write(fn, ret)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret == ret₂
        @test ret == ret₃
    end
    #
    @testset "Gʳᵉᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret = Gʳᵉᵗ(C, v)
        ret₂ = Gʳᵉᵗ(C, v₂)
        ret₃ = Gʳᵉᵗ(C, v₃)
        #
        @test ret != ret₂
        @test ret != ret₃
        #
        write(fn, ret)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret == ret₂
        @test ret == ret₃
    end
    #
    @testset "Gˡᵐⁱˣ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix = Gˡᵐⁱˣ(C, v)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        lmix₃ = Gˡᵐⁱˣ(C, v₃)
        #
        @test lmix != lmix₂
        @test lmix != lmix₃
        #
        write(fn, lmix)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix == lmix₂
        @test lmix == lmix₃
    end
    #
    @testset "Gˡᵐⁱˣ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix = Gˡᵐⁱˣ(C, v)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        lmix₃ = Gˡᵐⁱˣ(C, v₃)
        #
        @test lmix != lmix₂
        @test lmix != lmix₃
        #
        write(fn, lmix)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix == lmix₂
        @test lmix == lmix₃
    end
    #
    @testset "Gˡᵉˢˢ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less = Gˡᵉˢˢ(C, v)
        less₂ = Gˡᵉˢˢ(C, v₂)
        less₃ = Gˡᵉˢˢ(C, v₃)
        #
        @test less != less₂
        @test less != less₃
        #
        write(fn, less)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less == less₂
        @test less == less₃
    end
    #
    @testset "Gˡᵉˢˢ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less = Gˡᵉˢˢ(C, v)
        less₂ = Gˡᵉˢˢ(C, v₂)
        less₃ = Gˡᵉˢˢ(C, v₃)
        #
        @test less != less₂
        @test less != less₃
        #
        write(fn, less)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less == less₂
        @test less == less₃
    end
    #
    @testset "gᵐᵃᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "mat.data"
        #
        mat = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
        #
        @test mat != mat₂
        @test mat != mat₃
        #
        write(fn, mat)
        read!(fn, mat₂)
        open(fn, "r") do fin
            read!(fin, mat₃)
        end
        #
        @test mat == mat₂
        @test mat == mat₃
    end
    #
    @testset "gᵐᵃᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "mat.data"
        #
        mat = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
        #
        @test mat != mat₂
        @test mat != mat₃
        #
        write(fn, mat)
        read!(fn, mat₂)
        open(fn, "r") do fin
            read!(fin, mat₃)
        end
        #
        @test mat == mat₂
        @test mat == mat₃
    end
    #
    @testset "gʳᵉᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "ret.data"
        #
        ret = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
        #
        @test ret != ret₂
        @test ret != ret₃
        #
        write(fn, ret)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret == ret₂
        @test ret == ret₃
    end
    #
    @testset "gʳᵉᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "ret.data"
        #
        ret = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
        #
        @test ret != ret₂
        @test ret != ret₃
        #
        write(fn, ret)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret == ret₂
        @test ret == ret₃
    end
    #
    @testset "gˡᵐⁱˣ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "lmix.data"
        #
        lmix = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        #
        @test lmix != lmix₂
        @test lmix != lmix₃
        #
        write(fn, lmix)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix == lmix₂
        @test lmix == lmix₃
    end
    #
    @testset "gˡᵐⁱˣ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "lmix.data"
        #
        lmix = gˡᵐⁱˣ(ntau, ndim1, ndim2, v)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        #
        @test lmix != lmix₂
        @test lmix != lmix₃
        #
        write(fn, lmix)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix == lmix₂
        @test lmix == lmix₃
    end
    #
    @testset "gˡᵉˢˢ Struct: read/write (complx)" begin
    end
end
