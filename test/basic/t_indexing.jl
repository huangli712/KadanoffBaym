#
# t_indexing.jl
#
# To test getindex() and setindex() for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: indexing.jl" begin
    @testset "Cf    Struct: getindex/setindex" begin
        ntime = 201
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
        for i = 0:ntime
            @test cf[i] == x₁
        end
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
        ntau = 1001
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
        for i = 1:ntau
            @test mat[i] == x₁
        end
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
        ntime = 201
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
        for i = 1:ntime
            for j = 1:ntime
                if i ≥ j
                    @test ret[i,j] == x₁
                else
                    @test ret[i,j] == -x₁'
                end
            end
        end
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
        ntime = 201
        ntau = 1001
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
        for i = 1:ntime
            for j = 1:ntau
                @test lmix[i,j] == x₁
            end
        end
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
        ntime = 201
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
        for i = 1:ntime
            for j = 1:ntime
                if i > j
                    @test less[i,j] == -x₁'
                else
                    @test less[i,j] == x₁
                end
            end
        end
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
    @testset "Gᵐᵃᵗᵐ Struct: getindex/setindex" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        mat₁ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = Gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
        matm₁ = Gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = Gᵐᵃᵗᵐ(sign, mat₂)
        matm₃ = Gᵐᵃᵗᵐ(sign, mat₃)
        for i = 1:ntau
            @test matm₁[i] == mat₁[ntau - i + 1] * sign
            @test matm₂[i] == mat₂[ntau - i + 1] * sign
            @test matm₃[i] == mat₃[ntau - i + 1] * sign
        end
    end
    #
    @testset "Gᵃᵈᵛ  Struct: getindex/setindex" begin

    end
    #
    @testset "Gʳᵐⁱˣ Struct: getindex/setindex" begin
        
    end
    #
    @testset "Gᵍᵗʳ  Struct: getindex/setindex" begin
        ntime = 201
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
        ret₁ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₁)
        ret₂ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₂)
        ret₃ = Gʳᵉᵗ(ntime, ndim1, ndim2, v₃)
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₁)
        less₂ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₂)
        less₃ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₃)
        gtr₁ = Gᵍᵗʳ(less₁, ret₁)
        gtr₂ = Gᵍᵗʳ(less₁, ret₂)
        gtr₃ = Gᵍᵗʳ(less₁, ret₃)
        #
        for i = 1:ntime
            for j = 1:ntime
                @test gtr₁[i,j] == less₁[i,j] + ret₁[i,j]
            end
        end
    end
    #
    @testset "gᵐᵃᵗ  Struct: getindex/setindex" begin
        ntau = 1001
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
        for i = 1:ntau
            @test mat[i] == x₁
        end
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
        for i = 1:tstp
            @test ret[i] == x₁
            @test ret[i,tstp] == -x₁'
        end
        #
        ret[1] = x₂
        ret[2] = x₂
        ret[tstp] = x₂
        @test ret[1] == x₂
        @test ret[2] == x₂
        @test ret[tstp] == x₂
        @test ret[1,tstp] == -x₂'
        @test ret[2,tstp] == -x₂'
        @test ret[tstp,tstp] == -x₂'
        #
        ret[1] = v₃
        ret[2] = v₃
        ret[tstp] = v₃
        @test ret[1] == x₃
        @test ret[2] == x₃
        @test ret[tstp] == x₃
        @test ret[1,tstp] == -x₃'
        @test ret[2,tstp] == -x₃'
        @test ret[tstp,tstp] == -x₃'
    end
    #
    @testset "gˡᵐⁱˣ Struct: getindex/setindex" begin
        ntau = 1001
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
        for i = 1:ntau
            @test lmix[i] == x₁
        end
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
        less = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
        for i = 1:tstp
            @test less[i] == x₁
            @test less[tstp,i] == -x₁'
        end
        #
        less[1] = x₂
        less[2] = x₂
        less[tstp] = x₂
        @test less[1] == x₂
        @test less[2] == x₂
        @test less[tstp] == x₂
        @test less[tstp,1] == -x₂'
        @test less[tstp,2] == -x₂'
        @test less[tstp,tstp] == -x₂'
        #
        less[1] = v₃
        less[2] = v₃
        less[tstp] = v₃
        @test less[1] == x₃
        @test less[2] == x₃
        @test less[tstp] == x₃
        @test less[tstp,1] == -x₃'
        @test less[tstp,2] == -x₃'
        @test less[tstp,tstp] == -x₃'
    end
    #
    @testset "gᵐᵃᵗᵐ Struct: getindex/setindex" begin
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        sign = FERMI
        #
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
        matm₁ = gᵐᵃᵗᵐ(sign, mat₁)
        matm₂ = gᵐᵃᵗᵐ(sign, mat₂)
        matm₃ = gᵐᵃᵗᵐ(sign, mat₃)
        for i = 1:ntau
            @test matm₁[i] == mat₁[ntau - i + 1] * sign
            @test matm₂[i] == mat₂[ntau - i + 1] * sign
            @test matm₃[i] == mat₃[ntau - i + 1] * sign
        end
    end
    #
    @testset "gᵃᵈᵛ  Struct: getindex/setindex" begin

    end
    #
    @testset "gʳᵐⁱˣ Struct: getindex/setindex" begin

    end
    #
    @testset "gᵍᵗʳ  Struct: getindex/setindex" begin

    end
    #
    @testset "ℱ     Struct: getindex/setindex" begin
        ntime = 201
        ntau = 1001
        ndim1 = 2
        ndim2 = 2
        tmax = 5.0
        beta = 4.0
        tstp = 101
        sign = FERMI
        #
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        v₁ = 0.2 - 0.1im
        v₂ = 1.0 + 0.3im
        v₃ = 0.3 + 1.0im
        x₁ = fill(v₁, (ndim1, ndim2))
        x₂ = fill(v₂, (ndim1, ndim2))
        x₃ = fill(v₃, (ndim1, ndim2))
        #
        cfm = ℱ(C, v₁, sign)
        #
        cfv₁ = 𝒻(C, 0, v₁, sign)
        cfv₂ = 𝒻(C, 1, v₁, sign)
        cfv₃ = 𝒻(C, tstp, v₁, sign)
        @test cfm[0].mat == cfv₁.mat
        @test cfm[1].ret == cfv₂.ret
        @test cfm[1].lmix == cfv₂.lmix
        @test cfm[1].less == cfv₂.less
        @test cfm[tstp].ret == cfv₃.ret
        @test cfm[tstp].lmix == cfv₃.lmix
        @test cfm[tstp].less == cfv₃.less
        #
        cfv₁ = 𝒻(C, 0, v₂, sign)
        cfv₂ = 𝒻(C, 1, v₂, sign)
        cfv₃ = 𝒻(C, tstp, v₂, sign)
        cfm[0] = cfv₁
        cfm[1] = cfv₂
        cfm[tstp] = cfv₃
        @test cfm[0].mat == cfv₁.mat
        @test cfm[1].ret == cfv₂.ret
        @test cfm[1].lmix == cfv₂.lmix
        @test cfm[1].less == cfv₂.less
        @test cfm[tstp].ret == cfv₃.ret
        @test cfm[tstp].lmix == cfv₃.lmix
        @test cfm[tstp].less == cfv₃.less
        #
        cfv₁ = 𝒻(C, 0, v₃, sign)
        cfv₂ = 𝒻(C, 1, v₃, sign)
        cfv₃ = 𝒻(C, tstp, v₃, sign)
        cfm[0] = cfv₁
        cfm[1] = cfv₂
        cfm[tstp] = cfv₃
        @test cfm[0].mat == cfv₁.mat
        @test cfm[1].ret == cfv₂.ret
        @test cfm[1].lmix == cfv₂.lmix
        @test cfm[1].less == cfv₂.less
        @test cfm[tstp].ret == cfv₃.ret
        @test cfm[tstp].lmix == cfv₃.lmix
        @test cfm[tstp].less == cfv₃.less
    end
    #
    @testset "𝒻     Struct: getindex/setindex" begin
    end
end

println("All tests pass!\n")
