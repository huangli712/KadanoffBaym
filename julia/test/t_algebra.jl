#
# algebra.jl
#
# To test basic algebra for contour-ordered Green's functions.
#

function algebra_block(seed::Real)
    value = Float64(seed)
    return C64[
        value + im             -2.0 + (value + 1.0) * im;
        3.0 - (value + 2.0) * im value / 2.0 - 4.0im
    ]
end

function fill_algebra_green!(G::ℱ{C64}, offset::Int)
    for q = 1:getntau(G)
        G.mat[q] = algebra_block(offset + 10 + q)
    end
    for i = 1:getntime(G)
        for j = 1:i
            G.ret[i,j] = algebra_block(offset + 100 + 10 * i + j)
        end
        for q = 1:getntau(G)
            G.lmix[i,q] = algebra_block(offset + 200 + 10 * i + q)
        end
        for j = i:getntime(G)
            G.less[i,j] = algebra_block(offset + 300 + 10 * i + j)
        end
    end
    return G
end

@testset verbose = true "KadanoffBaym: algebra.jl" begin
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
    #
    @testset "Gˡᵐⁱˣ Struct: basic algebra" begin
        ntime = 201
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
        lmix₁ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₁)
        lmix₂ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₂)
        lmix₃ = Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, v₃)
        #
        @test lmix₃ == lmix₁ + lmix₂
        @test lmix₂ == lmix₃ - lmix₁
        @test lmix₁ == lmix₃ - lmix₂
        @test lmix₃ == 3.0 * lmix₁
        @test lmix₃ == lmix₁ * 3.0
        @test lmix₂ == 2.0 * lmix₁
        @test lmix₂ == lmix₁ * 2.0
        @test lmix₃ == 2.0 * lmix₂ - 1.0 * lmix₁
    end
    #
    @testset "Gˡᵉˢˢ Struct: basic algebra" begin
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
        less₁ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₁)
        less₂ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₂)
        less₃ = Gˡᵉˢˢ(ntime, ndim1, ndim2, v₃)
        #
        @test less₃ == less₁ + less₂
        @test less₂ == less₃ - less₁
        @test less₁ == less₃ - less₂
        @test less₃ == 3.0 * less₁
        @test less₃ == less₁ * 3.0
        @test less₂ == 2.0 * less₁
        @test less₂ == less₁ * 2.0
        @test less₃ == 2.0 * less₂ - 1.0 * less₁
    end
    #
    @testset "gᵐᵃᵗ  Struct: basic algebra" begin
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
        mat₁ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₁)
        mat₂ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₂)
        mat₃ = gᵐᵃᵗ(ntau, ndim1, ndim2, v₃)
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
    @testset "gʳᵉᵗ  Struct: basic algebra" begin
        tstp = 101
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
        ret₁ = gʳᵉᵗ(tstp, ndim1, ndim2, v₁)
        ret₂ = gʳᵉᵗ(tstp, ndim1, ndim2, v₂)
        ret₃ = gʳᵉᵗ(tstp, ndim1, ndim2, v₃)
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
    #
    @testset "gˡᵐⁱˣ Struct: basic algebra" begin
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
        lmix₁ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₁)
        lmix₂ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₂)
        lmix₃ = gˡᵐⁱˣ(ntau, ndim1, ndim2, v₃)
        #
        @test lmix₃ == lmix₁ + lmix₂
        @test lmix₂ == lmix₃ - lmix₁
        @test lmix₁ == lmix₃ - lmix₂
        @test lmix₃ == 3.0 * lmix₁
        @test lmix₃ == lmix₁ * 3.0
        @test lmix₂ == 2.0 * lmix₁
        @test lmix₂ == lmix₁ * 2.0
        @test lmix₃ == 2.0 * lmix₂ - 1.0 * lmix₁
    end
    #
    @testset "gˡᵉˢˢ Struct: basic algebra" begin
        tstp = 201
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
        less₁ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₁)
        less₂ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₂)
        less₃ = gˡᵉˢˢ(tstp, ndim1, ndim2, v₃)
        #
        @test less₃ == less₁ + less₂
        @test less₂ == less₃ - less₁
        @test less₁ == less₃ - less₂
        @test less₃ == 3.0 * less₁
        @test less₃ == less₁ * 3.0
        @test less₂ == 2.0 * less₁
        @test less₂ == less₁ * 2.0
        @test less₃ == 2.0 * less₂ - 1.0 * less₁
    end
    #
    # Port of NESSi herm_matrix_member.cpp:308-441. KB has no full ℱ
    # non-in-place algebra API, so this ports the contract through its four
    # existing full component APIs.
    @testset "NESSi full component non-in-place algebra" begin
        C = Cn(4, 5, 2, 2, 1.0, 2.0)
        A = fill_algebra_green!(ℱ(C, FERMI), 0)
        B = fill_algebra_green!(ℱ(C, FERMI), 500)
        A.ret[3,1] = C64[1.0 + 2.0im -3.0 + 1.0im; 4.0 - 2.0im 0.5 + 3.0im]
        B.ret[3,1] = C64[-2.0 + 1.0im 5.0 - 4.0im; 1.5 + 2.0im 3.0 - 1.0im]
        A.less[1,3] = C64[2.0 - 3.0im 1.0 + 4.0im; -5.0 + 2.0im 0.25 + 1.0im]
        B.less[1,3] = C64[1.0 + 1.0im -2.0 + 0.5im; 3.0 - 4.0im -1.0 + 2.0im]
        @test A.ret[3,1] != transpose(A.ret[3,1])
        @test A.ret[3,1] * B.ret[3,1] != B.ret[3,1] * A.ret[3,1]

        Aoriginal = deepcopy(A)
        Boriginal = deepcopy(B)
        α = 0.75 - 0.5im
        real_weight = 0.25
        sum_result = (
            mat = A.mat + B.mat,
            ret = A.ret + B.ret,
            lmix = A.lmix + B.lmix,
            less = A.less + B.less,
        )
        difference_result = (
            mat = A.mat - B.mat,
            ret = A.ret - B.ret,
            lmix = A.lmix - B.lmix,
            less = A.less - B.less,
        )
        complex_result = (
            mat = α * A.mat,
            ret = α * A.ret,
            lmix = α * A.lmix,
            less = α * A.less,
        )
        real_result = (
            mat = A.mat * real_weight,
            ret = A.ret * real_weight,
            lmix = A.lmix * real_weight,
            less = A.less * real_weight,
        )

        sum_reference = deepcopy(A)
        difference_reference = deepcopy(A)
        complex_reference = deepcopy(A)
        real_reference = deepcopy(A)
        incr!(sum_reference, B, 1.0)
        incr!(difference_reference, B, -1.0)
        for tstp = 0:getntime(A)
            smul!(complex_reference, tstp, α)
            smul!(real_reference, tstp, real_weight)
        end

        for q = 1:getntau(A)
            @test sum_result.mat[q] == A.mat[q] + B.mat[q]
            @test difference_result.mat[q] == A.mat[q] - B.mat[q]
            @test complex_result.mat[q] == α * A.mat[q]
            @test real_result.mat[q] == real_weight * A.mat[q]
        end
        for i = 1:getntime(A)
            for j = 1:i
                @test sum_result.ret[i,j] == A.ret[i,j] + B.ret[i,j]
                @test difference_result.ret[i,j] == A.ret[i,j] - B.ret[i,j]
                @test complex_result.ret[i,j] == α * A.ret[i,j]
                @test real_result.ret[i,j] == real_weight * A.ret[i,j]
                @test sum_result.less[j,i] == A.less[j,i] + B.less[j,i]
                @test difference_result.less[j,i] == A.less[j,i] - B.less[j,i]
                @test complex_result.less[j,i] == α * A.less[j,i]
                @test real_result.less[j,i] == real_weight * A.less[j,i]
            end
            for q = 1:getntau(A)
                @test sum_result.lmix[i,q] == A.lmix[i,q] + B.lmix[i,q]
                @test difference_result.lmix[i,q] == A.lmix[i,q] - B.lmix[i,q]
                @test complex_result.lmix[i,q] == α * A.lmix[i,q]
                @test real_result.lmix[i,q] == real_weight * A.lmix[i,q]
            end
        end

        for (result, reference) in (
            (sum_result, sum_reference),
            (difference_result, difference_reference),
            (complex_result, complex_reference),
            (real_result, real_reference),
        )
            for q = 1:getntau(A)
                @test result.mat[q] == reference.mat[q]
            end
            for i = 1:getntime(A)
                for j = 1:i
                    @test result.ret[i,j] == reference.ret[i,j]
                    @test result.less[j,i] == reference.less[j,i]
                end
                for j = 1:i - 1
                    @test result.ret[j,i] == -result.ret[i,j]'
                    @test result.less[i,j] == -result.less[j,i]'
                end
                for q = 1:getntau(A)
                    @test result.lmix[i,q] == reference.lmix[i,q]
                end
            end
        end
        @test A.mat * α == α * A.mat
        @test A.ret * α == α * A.ret
        @test A.lmix * α == α * A.lmix
        @test A.less * α == α * A.less
        @test A == Aoriginal
        @test B == Boriginal
    end
end

println("All tests pass!\n")
