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
        #
        fn = "Cn.data"
        wfn = "Cn.wrong"
        #
        C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        C₂ = Cn(2.0, 2.0) # Try different tmax and beta
        C₃ = Cn(3.0, 3.0) # Try different tmax and beta
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
        #
        try
            read!(wfn, C₂)
        catch ex
            catch_error()
        end
    end
    #
    @testset "Cf    Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        #
        fn = "Cf.data"
        wfn = "Cf.wrong"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        #
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
        #
        try
            read!(wfn, cf₂)
        catch ex
            catch_error()
        end
    end
    #
    @testset "Cf    Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 1
        ndim2 = 1
        tmax = 5.0
        beta = 4.0
        #
        fn = "Cf.data"
        wfn = "Cf.wrong"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        #
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
        #
        try
            read!(wfn, cf₂)
        catch ex
            catch_error()
        end
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
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret₁ = Gʳᵉᵗ(C, v₁)
        ret₂ = Gʳᵉᵗ(C, v₂)
        ret₃ = Gʳᵉᵗ(C, v₃)
        #
        @test ret₁ != ret₂
        @test ret₁ != ret₃
        #
        write(fn, ret₁)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
    end
    #
    @testset "Gʳᵉᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "ret.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        ret₁ = Gʳᵉᵗ(C, v₁)
        ret₂ = Gʳᵉᵗ(C, v₂)
        ret₃ = Gʳᵉᵗ(C, v₃)
        #
        @test ret₁ != ret₂
        @test ret₁ != ret₃
        #
        write(fn, ret₁)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
    end
    #
    @testset "Gˡᵐⁱˣ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix₁ = Gˡᵐⁱˣ(C, v₁)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        lmix₃ = Gˡᵐⁱˣ(C, v₃)
        #
        @test lmix₁ != lmix₂
        @test lmix₁ != lmix₃
        #
        write(fn, lmix₁)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
    end
    #
    @testset "Gˡᵐⁱˣ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "lmix.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        lmix₁ = Gˡᵐⁱˣ(C, v₁)
        lmix₂ = Gˡᵐⁱˣ(C, v₂)
        lmix₃ = Gˡᵐⁱˣ(C, v₃)
        #
        @test lmix₁ != lmix₂
        @test lmix₁ != lmix₃
        #
        write(fn, lmix₁)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
    end
    #
    @testset "Gˡᵉˢˢ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less₁ = Gˡᵉˢˢ(C, v₁)
        less₂ = Gˡᵉˢˢ(C, v₂)
        less₃ = Gˡᵉˢˢ(C, v₃)
        #
        @test less₁ != less₂
        @test less₁ != less₃
        #
        write(fn, less₁)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less₁ == less₂
        @test less₁ == less₃
    end
    #
    @testset "Gˡᵉˢˢ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 3
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "less.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        less₁ = Gˡᵉˢˢ(C, v₁)
        less₂ = Gˡᵉˢˢ(C, v₂)
        less₃ = Gˡᵉˢˢ(C, v₃)
        #
        @test less₁ != less₂
        @test less₁ != less₃
        #
        write(fn, less₁)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less₁ == less₂
        @test less₁ == less₃
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
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "mat.data"
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
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
    @testset "gᵐᵃᵗ  Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "mat.data"
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
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
    @testset "gʳᵉᵗ  Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "ret.data"
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
        #
        @test ret₁ != ret₂
        @test ret₁ != ret₃
        #
        write(fn, ret₁)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
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
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "ret.data"
        #
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
        #
        @test ret₁ != ret₂
        @test ret₁ != ret₃
        #
        write(fn, ret₁)
        read!(fn, ret₂)
        open(fn, "r") do fin
            read!(fin, ret₃)
        end
        #
        @test ret₁ == ret₂
        @test ret₁ == ret₃
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
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "lmix.data"
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        #
        @test lmix₁ != lmix₂
        @test lmix₁ != lmix₃
        #
        write(fn, lmix₁)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
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
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "lmix.data"
        #
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        #
        @test lmix₁ != lmix₂
        @test lmix₁ != lmix₃
        #
        write(fn, lmix₁)
        read!(fn, lmix₂)
        open(fn, "r") do fin
            read!(fin, lmix₃)
        end
        #
        @test lmix₁ == lmix₂
        @test lmix₁ == lmix₃
    end
    #
    @testset "gˡᵉˢˢ Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "less.data"
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
        less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
        #
        @test less₁ != less₂
        @test less₁ != less₃
        #
        write(fn, less₁)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less₁ == less₂
        @test less₁ == less₃
    end
    #
    @testset "gˡᵉˢˢ Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "less.data"
        #
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
        less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
        #
        @test less₁ != less₂
        @test less₁ != less₃
        #
        write(fn, less₁)
        read!(fn, less₂)
        open(fn, "r") do fin
            read!(fin, less₃)
        end
        #
        @test less₁ == less₂
        @test less₁ == less₃
    end
    #
    @testset "ℱ     Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "gf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cfm₁ = ℱ(C, v₁, sign)
        cfm₂ = ℱ(C, v₂, sign)
        cfm₃ = ℱ(C, v₃, sign)
        #
        @test cfm₁ != cfm₂
        @test cfm₁ != cfm₃
        #
        write(fn, cfm₁)
        read!(fn, cfm₂)
        open(fn, "r") do fin
            read!(fin, cfm₃)
        end
        #
        @test cfm₁ == cfm₂
        @test cfm₁ == cfm₃
    end
    #
    @testset "ℱ     Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        sign = FERMI
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "gf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cfm₁ = ℱ(C, v₁, sign)
        cfm₂ = ℱ(C, v₂, sign)
        cfm₃ = ℱ(C, v₃, sign)
        #
        @test cfm₁ != cfm₂
        @test cfm₁ != cfm₃
        #
        write(fn, cfm₁)
        read!(fn, cfm₂)
        open(fn, "r") do fin
            read!(fin, cfm₃)
        end
        #
        @test cfm₁ == cfm₂
        @test cfm₁ == cfm₃
    end
    #
    @testset "𝒻     Struct: read/write (complx)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        sign = FERMI
        v₁ = 1.0 - 0.3im
        v₂ = 0.3 + 0.3im
        v₃ = 1.2 - 0.3im
        fn = "gf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cfv₁ = 𝒻(C, tstp, v₁, sign)
        cfv₂ = 𝒻(C, tstp, v₂, sign)
        cfv₃ = 𝒻(C, tstp, v₃, sign)
        #
        @test cfv₁ != cfv₂
        @test cfv₁ != cfv₃
        #
        write(fn, cfv₁)
        read!(fn, cfv₂)
        open(fn, "r") do fin
            read!(fin, cfv₃)
        end
        #
        @test cfv₁ == cfv₂
        @test cfv₁ == cfv₃
    end
    #
    @testset "𝒻     Struct: read/write (real)" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = ntime
        sign = FERMI
        v₁ = 1.0
        v₂ = 0.3
        v₃ = 1.2
        fn = "gf.data"
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        cfv₁ = 𝒻(C, tstp, v₁, sign)
        cfv₂ = 𝒻(C, tstp, v₂, sign)
        cfv₃ = 𝒻(C, tstp, v₃, sign)
        #
        @test cfv₁ != cfv₂
        @test cfv₁ != cfv₃
        #
        write(fn, cfv₁)
        read!(fn, cfv₂)
        open(fn, "r") do fin
            read!(fin, cfv₃)
        end
        #
        @test cfv₁ == cfv₂
        @test cfv₁ == cfv₃
    end
end

println("All tests pass!\n")
