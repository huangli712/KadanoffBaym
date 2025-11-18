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
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        C₁ = Cn(2.0, 2.0)
        C₂ = Cn(3.0, 3.0)
        #
        @test C != C₁
        @test C != C₂
        #
        write(fn, C)
        read!(fn, C₁)
        open(fn, "r") do fin
            read!(fin, C₂)
        end
        #
        @test C == C₁
        @test C == C₂
    end
    #
    @testset "Cf    Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "Cf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cf = Cf(C, v)
        cf₁ = Cf(C, v₁)
        cf₂ = Cf(C, v₂)
        #
        @test cf != cf₁
        @test cf != cf₂
        #
        write(fn, cf)
        read!(fn, cf₁)
        open(fn, "r") do fin
            read!(fin, cf₂)
        end
        #
        @test cf == cf₁
        @test cf == cf₂
    end
    #
    @testset "Cf    Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₁ = 0.3
        v₂ = 1.2
        fn = "Cf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cf = Cf(C, v)
        cf₁ = Cf(C, v₁)
        cf₂ = Cf(C, v₂)
        #
        @test cf != cf₁
        @test cf != cf₂
        #
        write(fn, cf)
        read!(fn, cf₁)
        open(fn, "r") do fin
            read!(fin, cf₂)
        end
        #
        @test cf == cf₁
        @test cf == cf₂
    end
    #
    @testset "Gᵐᵃᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "mat.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        mat = Gᵐᵃᵗ(C, v)
        mat₁ = Gᵐᵃᵗ(C, v₁)
        mat₂ = Gᵐᵃᵗ(C, v₂)
        #
        @test mat != mat₁
        @test mat != mat₂
        #
        write(fn, mat)
        read!(fn, mat₁)
        open(fn, "r") do fin
            read!(fin, mat₂)
        end
        #
        @test mat == mat₁
        @test mat == mat₂
    end
    #
    @testset "Gᵐᵃᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₁ = 0.3
        v₂ = 1.2
        fn = "mat.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        mat = Gᵐᵃᵗ(C, v)
        mat₁ = Gᵐᵃᵗ(C, v₁)
        mat₂ = Gᵐᵃᵗ(C, v₂)
        #
        @test mat != mat₁
        @test mat != mat₂
        #
        write(fn, mat)
        read!(fn, mat₁)
        open(fn, "r") do fin
            read!(fin, mat₂)
        end
        #
        @test mat == mat₁
        @test mat == mat₂
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
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret = Gʳᵉᵗ(C, v)
        ret₁ = Gʳᵉᵗ(C, v₁)
        ret₂ = Gʳᵉᵗ(C, v₂)
        #
        @test ret != ret₁
        @test ret != ret₂
        #
        write(fn, ret)
        read!(fn, ret₁)
        open(fn, "r") do fin
            read!(fin, ret₂)
        end
        #
        @test ret == ret₁
        @test ret == ret₂
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
        v₁ = 0.3
        v₂ = 1.2
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret = Gʳᵉᵗ(C, v)
        ret₁ = Gʳᵉᵗ(C, v₁)
        ret₂ = Gʳᵉᵗ(C, v₂)
        #
        @test ret != ret₁
        @test ret != ret₂
        #
        write(fn, ret)
        read!(fn, ret₁)
        open(fn, "r") do fin
            read!(fin, ret₂)
        end
        #
        @test ret == ret₁
        @test ret == ret₂
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
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix = Gˡᵐⁱˣ(C, v)
        lmix₁ = Gˡᵐⁱˣ(C, v₁)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        #
        @test lmix != lmix₁
        @test lmix != lmix₂
        #
        write(fn, lmix)
        read!(fn, lmix₁)
        open(fn, "r") do fin
            read!(fin, lmix₂)
        end
        #
        @test lmix == lmix₁
        @test lmix == lmix₂
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
        v₁ = 0.3
        v₂ = 1.2
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix = Gˡᵐⁱˣ(C, v)
        lmix₁ = Gˡᵐⁱˣ(C, v₁)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        #
        @test lmix != lmix₁
        @test lmix != lmix₂
        #
        write(fn, lmix)
        read!(fn, lmix₁)
        open(fn, "r") do fin
            read!(fin, lmix₂)
        end
        #
        @test lmix == lmix₁
        @test lmix == lmix₂
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
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less = Gˡᵉˢˢ(C, v)
        less₁ = Gˡᵉˢˢ(C, v₁)
        less₂ = Gˡᵉˢˢ(C, v₂)
        #
        @test less != less₁
        @test less != less₂
        #
        write(fn, less)
        read!(fn, less₁)
        open(fn, "r") do fin
            read!(fin, less₂)
        end
        #
        @test less == less₁
        @test less == less₂
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
        v₁ = 0.3
        v₂ = 1.2
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less = Gˡᵉˢˢ(C, v)
        less₁ = Gˡᵉˢˢ(C, v₁)
        less₂ = Gˡᵉˢˢ(C, v₂)
        #
        @test less != less₁
        @test less != less₂
        #
        write(fn, less)
        read!(fn, less₁)
        open(fn, "r") do fin
            read!(fin, less₂)
        end
        #
        @test less == less₁
        @test less == less₂
    end
    #
    @testset "gᵐᵃᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0 - 0.3im
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "mat.data"
        #
        mat = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        #
        @test mat != mat₁
        @test mat != mat₂
        #
        write(fn, mat)
        read!(fn, mat₁)
        open(fn, "r") do fin
            read!(fin, mat₂)
        end
        #
        @test mat == mat₁
        @test mat == mat₂
    end
    #
    @testset "gᵐᵃᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v = 1.0
        v₁ = 0.3
        v₂ = 1.2
        fn = "mat.data"
        #
        mat = gᵐᵃᵗ(ntau, ndim1, ndim2, v)
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        #
        @test mat != mat₁
        @test mat != mat₂
        #
        write(fn, mat)
        read!(fn, mat₁)
        open(fn, "r") do fin
            read!(fin, mat₂)
        end
        #
        @test mat == mat₁
        @test mat == mat₂
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
        v₁ = 0.3 + 0.3im
        v₂ = 1.2 - 0.3im
        fn = "ret.data"
        #
        ret = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        #
        @test ret != ret₁
        @test ret != ret₂
        #
        write(fn, ret)
        read!(fn, ret₁)
        open(fn, "r") do fin
            read!(fin, ret₂)
        end
        #
        @test ret == ret₁
        @test ret == ret₂
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
        v₁ = 0.3
        v₂ = 1.2
        fn = "ret.data"
        #
        ret = gʳᵉᵗ(tstp, ndim1, ndim2, v)
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        #
        @test ret != ret₁
        @test ret != ret₂
        #
        write(fn, ret)
        read!(fn, ret₁)
        open(fn, "r") do fin
            read!(fin, ret₂)
        end
        #
        @test ret == ret₁
        @test ret == ret₂
    end
    #
    @testset "gˡᵐⁱˣ Struct: read/write (complx)" begin
    end
    #
    @testset "gˡᵉˢˢ Struct: read/write (complx)" begin
    end
end
