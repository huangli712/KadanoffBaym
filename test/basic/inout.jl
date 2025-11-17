#
# inout.jl
#
# To test the read and write operations.
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
    @testset "Cf    Struct: read/write (cmplx)" begin
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
end
