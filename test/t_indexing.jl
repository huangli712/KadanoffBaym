#
# t_indexing.jl
#
# To test getindex() and setindex() for contour-ordered Green's functions.
#

function indexing_block(seed::Real)
    value = Float64(seed)
    return C64[
        value + (value + 1.0) * im      -0.5 * value + (value + 2.0) * im;
        value / 3.0 - (value + 3.0) * im value + 4.0 - (value + 5.0) * im
    ]
end

function indexing_diagonal_block(seed::Real)
    value = Float64(seed)
    return C64[
        value * im             1.0 + (value + 1.0) * im;
        -1.0 + (value + 1.0) * im -(value + 2.0) * im
    ]
end

function fill_indexing_green!(G::ℱ{C64})
    ndim1, ndim2 = getdims(G)
    @assert ndim1 ≤ 2
    @assert ndim2 ≤ 2

    for q = 1:getntau(G)
        G.mat[q] = indexing_block(10 + q)[1:ndim1,1:ndim2]
    end
    for i = 1:getntime(G)
        for j = 1:i
            block = i == j ? indexing_diagonal_block(100 + i) : indexing_block(100 + 10 * i + j)
            G.ret[i,j] = block[1:ndim1,1:ndim2]
        end
        for q = 1:getntau(G)
            G.lmix[i,q] = indexing_block(200 + 10 * i + q)[1:ndim1,1:ndim2]
        end
        for j = i:getntime(G)
            block = i == j ? indexing_diagonal_block(300 + i) : indexing_block(300 + 10 * i + j)
            G.less[i,j] = block[1:ndim1,1:ndim2]
        end
    end
    return G
end

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
        #
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
        #
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
        #
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
        #
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
        #
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
        #
        for i = 1:ntau
            @test matm₁[i] == mat₁[ntau - i + 1] * sign
            @test matm₂[i] == mat₂[ntau - i + 1] * sign
            @test matm₃[i] == mat₃[ntau - i + 1] * sign
        end
    end
    #
    @testset "Gᵃᵈᵛ  Struct: getindex/setindex" begin
        # Port of NESSi getset.cpp:35-115 (advanced reconstruction contract).
        ntime = 4
        ndim = 2
        ret = Gʳᵉᵗ(ntime, ndim, ndim)
        for i = 1:ntime
            for j = 1:i
                ret[i,j] = i == j ? indexing_diagonal_block(i) : indexing_block(10 * i + j)
            end
        end
        adv = Gᵃᵈᵛ(ret)

        for i = 1:ntime
            for j = 1:ntime
                @test adv[i,j] == ret[j,i]'
            end
        end

        parent = copy(ret[ntime,1])
        derived = adv[1,ntime]
        fill!(derived, zero(C64))
        @test ret[ntime,1] == parent
        ret[ntime,1][1,1] = 17.0 - 3.0im
        @test adv[1,ntime][1,1] == 17.0 + 3.0im
        @test_throws AssertionError adv[0,1]
        @test_throws AssertionError adv[1,ntime + 1]
    end
    #
    @testset "Gʳᵐⁱˣ Struct: getindex/setindex" begin
        ntime = 201
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
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₁)
        lmix₂ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₂)
        lmix₃ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₃)
        rmix₁ = Gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = Gʳᵐⁱˣ(sign, lmix₂)
        rmix₃ = Gʳᵐⁱˣ(sign, lmix₃)
        #
        for i = 1:ntau
            for j = 1:ntime
                @test rmix₁[i,j] == lmix₁[j,ntau - i + 1]' * (-sign)
                @test rmix₂[i,j] == lmix₂[j,ntau - i + 1]' * (-sign)
                @test rmix₃[i,j] == lmix₃[j,ntau - i + 1]' * (-sign)
            end
        end
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
        gtr₂ = Gᵍᵗʳ(less₂, ret₂)
        gtr₃ = Gᵍᵗʳ(less₃, ret₃)
        #
        for i = 1:ntime
            for j = 1:ntime
                @test gtr₁[i,j] == less₁[i,j] + ret₁[i,j]
                @test gtr₂[i,j] == less₂[i,j] + ret₂[i,j]
                @test gtr₃[i,j] == less₃[i,j] + ret₃[i,j]
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
        #
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
        #
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
        #
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
        #
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
        #
        for i = 1:ntau
            @test matm₁[i] == mat₁[ntau - i + 1] * sign
            @test matm₂[i] == mat₂[ntau - i + 1] * sign
            @test matm₃[i] == mat₃[ntau - i + 1] * sign
        end
    end
    #
    @testset "gᵃᵈᵛ  Struct: getindex/setindex" begin
        # Port of NESSi getset.cpp:117-194 (timestep advanced reconstruction).
        tstp = 4
        ret = gʳᵉᵗ(tstp, 2, 2)
        for i = 1:tstp
            ret[i] = indexing_block(i)
        end
        adv = gᵃᵈᵛ(ret)

        for i = 1:tstp
            @test adv[i] == ret[i]'
        end

        parent = copy(ret[tstp])
        derived = adv[tstp]
        fill!(derived, zero(C64))
        @test ret[tstp] == parent
        ret[tstp][1,2] = 4.0 + 9.0im
        @test adv[tstp][2,1] == 4.0 - 9.0im
        @test_throws AssertionError adv[0]
        @test_throws AssertionError adv[tstp + 1]
    end
    #
    @testset "gʳᵐⁱˣ Struct: getindex/setindex" begin
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
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        rmix₁ = gʳᵐⁱˣ(sign, lmix₁)
        rmix₂ = gʳᵐⁱˣ(sign, lmix₂)
        rmix₃ = gʳᵐⁱˣ(sign, lmix₃)
        #
        for i = 1:ntau
            @test rmix₁[i] == lmix₁[ntau - i + 1]' * (-sign)
            @test rmix₂[i] == lmix₂[ntau - i + 1]' * (-sign)
            @test rmix₃[i] == lmix₃[ntau - i + 1]' * (-sign)
        end
    end
    #
    @testset "gᵍᵗʳ  Struct: getindex/setindex" begin
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
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
        less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
        gtr₁ = gᵍᵗʳ(less₁, ret₁)
        gtr₂ = gᵍᵗʳ(less₂, ret₂)
        gtr₃ = gᵍᵗʳ(less₃, ret₃)
        #
        for i = 1:tstp
            @test gtr₁[i] == less₁[i] + ret₁[i,tstp]
            @test gtr₂[i] == less₂[i] + ret₂[i,tstp]
            @test gtr₃[i] == less₃[i] + ret₃[i,tstp]
        end
        #
        for j = 1:tstp
            @test gtr₁[tstp,j] == less₁[tstp,j] + ret₁[j]
            @test gtr₂[tstp,j] == less₂[tstp,j] + ret₂[j]
            @test gtr₃[tstp,j] == less₃[tstp,j] + ret₃[j]
        end
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
        cfv₁ = 𝒻(C, 0, v₁, sign)
        cfv₂ = 𝒻(C, 1, v₁, sign)
        cfv₃ = 𝒻(C, tstp, v₁, sign)
        #
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
        # Ports of NESSi getset.cpp:117-194,196-315,
        # herm_matrix_member.cpp:447-466,
        # herm_member_timestep.cpp:216-237, and
        # herm_member_timestep_view.cpp:42-59.
        for ndim in (1, 2)
            C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
            G = fill_indexing_green!(ℱ(C, FERMI))

            equilibrium = G[0]
            for q = 1:getntau(G)
                @test equilibrium.mat[q] == G.mat[q]
            end
            @test getsign(equilibrium) == FERMI
            @test gettstp(equilibrium) == 0

            tstp = 3
            slice = G[tstp]
            @test getsign(slice) == FERMI
            @test gettstp(slice) == tstp
            for j = 1:tstp
                @test slice.ret[j] == G.ret[tstp,j]
                @test slice.less[j] == G.less[j,tstp]
                @test slice.gtr[j] == G.gtr[j,tstp]
                @test slice.gtr[tstp,j] == G.gtr[tstp,j]
            end
            for q = 1:getntau(G)
                @test slice.lmix[q] == G.lmix[tstp,q]
            end

            original = copy(G.ret[tstp,1])
            slice.ret[1][1,1] = 41.0 + 2.0im
            @test G.ret[tstp,1] == original
            untouched = copy(G.ret[2,1])
            G[tstp] = slice
            @test G.ret[tstp,1][1,1] == 41.0 + 2.0im
            @test G.ret[2,1] == untouched

            replacement = 𝒻(C, 0, fill(7.0 + 3.0im, ndim, ndim), FERMI)
            G[0] = replacement
            for q = 1:getntau(G)
                @test G.mat[q] == replacement.mat[q]
            end

            @test_throws AssertionError G[-1]
            @test_throws AssertionError G[getntime(G) + 1]
            wrong_tstp = 𝒻(C, 2, FERMI)
            @test_throws AssertionError (G[1] = wrong_tstp)
        end
    end
    #
    # Ports of NESSi getset.cpp:35-74 and 76-115.
    @testset "NESSi full scalar and matrix get contracts" begin
        for ndim in (1, 2)
            C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
            G = fill_indexing_green!(ℱ(C, BOSE))

            for q = 1:getntau(G)
                expected = indexing_block(10 + q)[1:ndim,1:ndim]
                @test G.mat[q] == expected
                @test G.matm[q] == G.mat[getntau(G) - q + 1]
            end
            for i = 1:getntime(G)
                for j = 1:getntime(G)
                    ret_block = i ≥ j ?
                        (i == j ? indexing_diagonal_block(100 + i) : indexing_block(100 + 10 * i + j)) :
                        -(indexing_block(100 + 10 * j + i)')
                    less_block = i ≤ j ?
                        (i == j ? indexing_diagonal_block(300 + i) : indexing_block(300 + 10 * i + j)) :
                        -(indexing_block(300 + 10 * j + i)')
                    expected_ret = ret_block[1:ndim,1:ndim]
                    expected_less = less_block[1:ndim,1:ndim]
                    @test G.ret[i,j] == expected_ret
                    @test G.adv[i,j] == G.ret[j,i]'
                    @test G.less[i,j] == expected_less
                    @test G.gtr[i,j] == expected_ret + expected_less
                end
                for q = 1:getntau(G)
                    expected = indexing_block(200 + 10 * i + q)[1:ndim,1:ndim]
                    @test G.lmix[i,q] == expected
                    @test G.rmix[q,i] == -G.lmix[i,getntau(G) - q + 1]'
                end
            end

            @test_throws AssertionError G.mat[0]
            @test_throws AssertionError G.ret[0,1]
            @test_throws AssertionError G.lmix[1,getntau(G) + 1]
            @test_throws AssertionError G.less[getntime(G) + 1,1]
        end

        C = Cn(4, 5, 2, 2, 1.0, 2.0)
        G = fill_indexing_green!(ℱ(C, FERMI))
        stored = copy(G.ret[3,1])
        derived = G.ret[1,3]
        fill!(derived, zero(C64))
        @test G.ret[3,1] == stored
        G.ret[3,1][1,1] = 31.0 + 6.0im
        @test G.ret[3,1][1,1] == 31.0 + 6.0im
        @test_throws AssertionError (G.ret[1,3] = indexing_block(1))
        @test_throws AssertionError (G.less[3,1] = indexing_block(1))
    end
    #
    # Ports of NESSi getset_nonherm.cpp:58-175.
    @testset "NESSi non-Hermitian full scalar and matrix get contracts" begin
        for ndim in (1, 2)
            C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
            G = fill_indexing_green!(ℱ(C, FERMI))
            if ndim == 2
                G.ret[3,1] = C64[1.0 + 1.0im 2.0 - 1.0im; 3.0 + 2.0im -1.0 + 1.0im]
                G.less[1,3] = C64[0.5 - 1.0im -2.0 + 3.0im; 1.0 + 4.0im 2.0 - 2.0im]
                @test G.ret[3,1] != transpose(G.ret[3,1])
                @test G.ret[3,1] * G.less[1,3] != G.less[1,3] * G.ret[3,1]
            end
            for i = 1:getntime(G)
                for j = 1:getntime(G)
                    @test G.gtr[i,j] == G.ret[i,j] + G.less[i,j]
                    @test G.gtr[i,j] == -G.gtr[j,i]'
                end
            end
        end
    end
    #
    # Ports of NESSi getset_nonherm.cpp:178-328 and 333-454.
    @testset "NESSi non-Hermitian timestep and view get contracts" begin
        for ndim in (1, 2)
            C = Cn(4, 5, ndim, ndim, 1.0, 2.0)
            G = fill_indexing_green!(ℱ(C, FERMI))
            tstp = 3
            slice = G[tstp]
            for i = 1:tstp
                @test slice.ret[i] == G.ret[tstp,i]
                @test slice.less[i] == G.less[i,tstp]
                @test slice.gtr[i] == G.gtr[i,tstp]
                @test slice.gtr[tstp,i] == G.gtr[tstp,i]
                @test slice.gtr[i] == -slice.gtr[tstp,i]'
            end
            for q = 1:getntau(G)
                @test slice.lmix[q] == G.lmix[tstp,q]
                @test slice.rmix[q] == G.rmix[q,tstp]
            end
        end
    end
    #
    # Ports of NESSi herm_matrix_member.cpp:246-251 and 590-611.
    @testset "NESSi member set/get and matminus contracts" begin
        Crect = Cn(4, 5, 2, 3, 1.0, 2.0)
        Grect = ℱ(Crect, FERMI)
        block = C64[1.0 + 2.0im 3.0 - 1.0im 5.0 + 4.0im;
                     -2.0 + 3.0im 7.0 + 1.0im -4.0 - 2.0im]
        expected = copy(block)
        Grect.mat[1] = block
        Grect.ret[4,1] = block
        Grect.lmix[2,5] = block
        Grect.less[1,4] = block
        block[1,1] = 99.0 + 99.0im
        @test Grect.mat[1] == expected
        @test Grect.ret[4,1] == expected
        @test Grect.lmix[2,5] == expected
        @test Grect.less[1,4] == expected
        Grect.mat[5] = 2.0im
        Grect.ret[4,4] = 3.0im
        Grect.lmix[4,1] = 4.0im
        Grect.less[4,4] = 5.0im
        @test Grect.mat[5] == fill(2.0im, 2, 3)
        @test Grect.ret[4,4] == fill(3.0im, 2, 3)
        @test Grect.lmix[4,1] == fill(4.0im, 2, 3)
        @test Grect.less[4,4] == fill(5.0im, 2, 3)
        @test_throws AssertionError (Grect.mat[1] = zeros(C64, 3, 2))

        for sign in (FERMI, BOSE)
            C = Cn(4, 5, 2, 2, 1.0, 2.0)
            G = fill_indexing_green!(ℱ(C, sign))
            for q = 1:getntau(G)
                @test G.matm[q] == sign * G.mat[getntau(G) - q + 1]
            end
        end
    end
end

println("All tests pass!\n")
