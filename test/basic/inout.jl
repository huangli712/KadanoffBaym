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
            read!(fn, C₂)
        end
        #
        @test C == C₁
        @test C == C₂
    end
end
