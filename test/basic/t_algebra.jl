#
# algebra.jl
#
# To test basic algebra for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: indexing.jl" begin
    @testset "Cf    Struct: basic algebra" begin
        ntime = 201
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 1.0 + 0.0im
        v₂ = 2.0 + 0.0im
        v₃ = 3.0 + 0.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        cf₁ = Cf(ntime, ndim1, ndim2, v₁)
        cf₂ = Cf(ntime, ndim1, ndim2, v₂)
        cf₃ = Cf(ntime, ndim1, ndim2, v₃)
        #
        @test cf₃ == cf₁ + cf₂
        @test cf₂ == cf₃ - cf₁
        @test cf₁ == cf₃ - cf₂
        @test cf₃ == 3.0 * cf₁
        @test cf₃ == cf₁ * 3.0
        @test cf₂ == 2.0 * cf₁
        @test cf₂ == cf₁ * 2.0
    end
end

println("All tests pass!\n")
