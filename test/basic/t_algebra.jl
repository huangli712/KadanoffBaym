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
        @test cf₃ == 2.0 * cf₂ - 1.0 * cf₁
    end
    #
    @testset "Gᵐᵃᵗ  Struct: basic algebra" begin
        ntau = 1001
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
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
        #
        @test mat₃ == mat₁ + mat₂
        @test mat₂ == mat₃ - mat₁
        @test mat₁ == mat₃ - mat₂
        @test mat₃ == 3.0 * mat₁
        @test mat₃ == mat₁ * 3.0
        @test mat₂ == 2.0 * mat₁
        @test mat₂ == mat₁ * 2.0
        @test mat₃ == 2.0 * mat₂ - 1.0 * mat₁
    end
    #
    @testset "Gʳᵉᵗ  Struct: basic algebra" begin
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
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₁)
        ret₂ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₂)
        ret₃ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₃)
        #
        @test ret₃ == ret₁ + ret₂
        @test ret₂ == ret₃ - ret₁
        @test ret₁ == ret₃ - ret₂
        @test ret₃ == 3.0 * ret₁
        @test ret₃ == ret₁ * 3.0
        @test ret₂ == 2.0 * ret₁
        @test ret₂ == ret₁ * 2.0
        @test ret₃ == 2.0 * ret₂ - 1.0 * ret₁
    end
end

println("All tests pass!\n")
