#
# indexing.jl
#
# To test getindex() and setindex() for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: indexing.jl" begin
    @testset "Cf    Struct: getindex/setindex" begin
        ntime = 1001
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = zero(C64)
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        cf = Cf(ntime, ndim1, ndim2, v₁)
        @test cf[0] == x₁
        @test cf[1] == x₁
        @test cf[ntime] == x₁
        #
        cf[0] = x₂
        cf[1] = x₂
        cf[ntime] = x₂
        @test cf[0] == x₂
        @test cf[1] == x₂
        @test cf[ntime] == x₂
        #
        cf[0] = v₃
        cf[1] = v₃
        cf[ntime] = v₃
        @test cf[0] == x₃
        @test cf[1] == x₃
        @test cf[ntime] == x₃
    end
    #
    @testset "Gᵐᵃᵗ  Struct: getindex/setindex" begin
    end
    #
    @testset "Gʳᵉᵗ  Struct: getindex/setindex" begin
    end
    #
    @testset "Gʳᵉᵗ  Struct: getindex/setindex" begin
    end
    #
    @testset "Gʳᵉᵗ  Struct: getindex/setindex" begin
    end
    #
    @testset "Gʳᵉᵗ  Struct: getindex/setindex" begin
    end
end