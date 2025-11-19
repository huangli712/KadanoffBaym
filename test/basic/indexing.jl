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
        v₁ = 0.2 - 0.1im
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
        ntau = 201
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        mat = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        @test mat[1] == x₁
        @test mat[2] == x₁
        @test mat[ntau] == x₁
        #
        #
        mat[1] = x₂
        mat[2] = x₂
        mat[ntau] = x₂
        @test mat[1] == x₂
        @test mat[2] == x₂
        @test mat[ntau] == x₂
        #
        mat[1] = v₃
        mat[2] = v₃
        mat[ntau] = v₃
        @test mat[1] == x₃
        @test mat[2] == x₃
        @test mat[ntau] == x₃
    end
    #
    @testset "Gʳᵉᵗ  Struct: getindex/setindex" begin
        ntime = 1001
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        ret = Gʳᵉᵗ(ntime, ndim1, ndim2, v₁)
        @test ret[1,2] == -x₁'
        @test ret[2,1] == x₁
        @test ret[ntime,ntime] == x₁
        #
        ret[2,1] = x₂
        ret[ntime,ntime] = x₂
        @test ret[2,1] == x₂
        @test ret[ntime,ntime] == x₂
        #
        ret[2,1] = v₃
        ret[ntime,ntime] = v₃
        @test ret[2,1] == x₃
        @test ret[ntime,ntime] == x₃
    end
    #
    @testset "Gˡᵐⁱˣ Struct: getindex/setindex" begin
        ntime = 1001
        ntau = 201
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        lmix = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₁)
        @test lmix[1,1] == x₁
        @test lmix[2,3] == x₁
        @test lmix[3,2] == x₁
        @test lmix[ntime,ntau] == x₁
        #
        lmix[1,1] = x₂
        lmix[2,3] = x₂
        lmix[3,2] = x₂
        lmix[ntime,ntau] = x₂
        @test lmix[1,1] == x₂
        @test lmix[2,3] == x₂
        @test lmix[3,2] == x₂
        @test lmix[ntime,ntau] == x₂
        #
        lmix[1,1] = v₃
        lmix[2,3] = v₃
        lmix[3,2] = v₃
        lmix[ntime,ntau] = v₃
        @test lmix[1,1] == x₃
        @test lmix[2,3] == x₃
        @test lmix[3,2] == x₃
        @test lmix[ntime,ntau] == x₃
    end
    #
    @testset "Gˡᵉˢˢ Struct: getindex/setindex" begin
        ntime = 1001
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        less = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₁)
        @test less[1,2] == x₁
        @test less[2,1] == -x₁'
        @test less[ntime,ntime] == x₁
        #
        less[1,2] = x₂
        less[ntime,ntime] = x₂
        @test less[1,2] == x₂
        @test less[ntime,ntime] == x₂
        #
        less[1,2] = v₃
        less[ntime,ntime] = v₃
        @test less[1,2] == x₃
        @test less[ntime,ntime] == x₃
    end
    #
    @testset "gᵐᵃᵗ  Struct: getindex/setindex" begin
        ntau = 201
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        mat = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        @test mat[1] == x₁
        @test mat[2] == x₁
        @test mat[ntau] == x₁
        #
        mat[1] = x₂
        mat[2] = x₂
        mat[ntau] = x₂
        @test mat[1] == x₂
        @test mat[2] == x₂
        @test mat[ntau] == x₂
        #
        mat[1] = v₃
        mat[2] = v₃
        mat[ntau] = v₃
        @test mat[1] == x₃
        @test mat[2] == x₃
        @test mat[ntau] == x₃
    end
    #
    @testset "gʳᵉᵗ  Struct: getindex/setindex" begin
        tstp = 101
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        ret = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        @test ret[1] == x₁
        @test ret[2] == x₁
        @test ret[tstp] == x₁
        @test ret[1,tstp] == -ret[1]'
        @test ret[2,tstp] == -ret[2]'
        @test ret[tstp,tstp] == -ret[tstp]'
        #
        ret[1] = x₂
        ret[2] = x₂
        ret[tstp] = x₂
        @test ret[1] == x₂
        @test ret[2] == x₂
        @test ret[tstp] == x₂
        @test ret[1,tstp] == -ret[1]'
        @test ret[2,tstp] == -ret[2]'
        @test ret[tstp,tstp] == -ret[tstp]'
        #
        ret[1] = v₃
        ret[2] = v₃
        ret[tstp] = v₃
        @test ret[1] == x₃
        @test ret[2] == x₃
        @test ret[tstp] == x₃
        @test ret[1,tstp] == -ret[1]'
        @test ret[2,tstp] == -ret[2]'
        @test ret[tstp,tstp] == -ret[tstp]'
    end
    #
    @testset "gˡᵐⁱˣ Struct: getindex/setindex" begin
        ntau = 201
        ndim1 = 2
        ndim2 = 2
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        lmix = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
        @test lmix[1] == x₁
        @test lmix[2] == x₁
        @test lmix[ntau] == x₁
        #
        lmix[1] = x₂
        lmix[2] = x₂
        lmix[ntau] = x₂
        @test lmix[1] == x₂
        @test lmix[2] == x₂
        @test lmix[ntau] == x₂
        #
        lmix[1] = v₃
        lmix[2] = v₃
        lmix[ntau] = v₃
        @test lmix[1] == x₃
        @test lmix[2] == x₃
        @test lmix[ntau] == x₃
    end
    #
    @testset "gˡᵉˢˢ Struct: getindex/setindex" begin
    end
end