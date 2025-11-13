haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true "KadanoffBaym: traits.jl" begin
    ntime = 101
    ntau = 51
    ndim1 = 2
    ndim2 = 2
    tmax = 1.0
    beta = 10.0
    dt = 0.01
    mu = 0.0
    ϵ = 1e-6; ϵ₁ = -0.4; ϵ₂ = 0.6; ϵ₃ = 0.435; ϵ₄ = 0.5676
    λ₁ = 0.1; λ₂ = 0.1566
    wr = 0.3
    wz = 1.0 - 0.3im
    #
    C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
    G1 = ℱ(C, FERMI)
    G2 = ℱ(C, FERMI)
    G3 = ℱ(C, FERMI)
    G4 = ℱ(C, FERMI)
    #
    H1 = fill(zero(C64), ndim1, ndim1)
    H2 = fill(zero(C64), ndim1, ndim1)
    H1[1,1] = ϵ₁
    H1[2,2] = ϵ₂
    H1[1,2] = im * λ₁
    H1[2,1] = -im * λ₁
    H2[1,1] = ϵ₃
    H2[2,2] = ϵ₄
    H2[1,2] = im * λ₂
    H2[2,1] = -im * λ₂
    #
    init_green!(G1, H1, mu, beta, dt)
    init_green!(G2, H2, mu, beta, dt)
    #
    @testset "incr! and memcpy!" begin
        mat1 = fill(zero(C64), ndim1, ndim1)
        mat2 = fill(zero(C64), ndim1, ndim1)
        mat3 = fill(zero(C64), ndim1, ndim1)
        #
        ret1 = fill(zero(C64), ndim1, ndim1)
        ret2 = fill(zero(C64), ndim1, ndim1)
        ret3 = fill(zero(C64), ndim1, ndim1)
        #
        lmix1 = fill(zero(C64), ndim1, ndim1)
        lmix2 = fill(zero(C64), ndim1, ndim1)
        lmix3 = fill(zero(C64), ndim1, ndim1)
        #
        less1 = fill(zero(C64), ndim1, ndim1)
        less2 = fill(zero(C64), ndim1, ndim1)
        less3 = fill(zero(C64), ndim1, ndim1)

        # For mat component
        for q = 1:ntau
            @. mat1 = G1.mat[q]
            @. mat2 = G2.mat[q]
            @. mat3 = mat1 + wz * mat2
            G3.mat[q] = mat3
        end

        for i = 1:ntime
            # For ret and less components
            for j = 1:i
                @. ret1 = G1.ret[i,j]
                @. ret2 = G2.ret[i,j]
                @. ret3 = ret1 + wz * ret2
                G3.ret[i,j] = ret3

                @. less1 = G1.less[j,i]
                @. less2 = G2.less[j,i]
                @. less3 = less1 + wz * less2
                G3.less[j,i] = less3
            end

            # For lmix component
            for q = 1:ntau
                @. lmix1 = G1.lmix[i,q]
                @. lmix2 = G2.lmix[i,q]
                @. lmix3 = lmix1 + wz * lmix2
                G3.lmix[i,q] = lmix3
            end
        end

        err = 0.0
        init_green!(G4, H1, mu, beta, dt)
        incr!(G4, G2, wz)
        for tstp = 0:ntime
            err = err + distance(G3, G4, tstp)
        end
        @test err < ϵ

        err = 0.0
        init_green!(G4, H1, mu, beta, dt)
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G2, A, tstp)
            incr!(G4, A, tstp, wz)
            err = err + distance(G3, G4, tstp)
        end
        @test err < ϵ

        err = 0.0
        init_green!(G4, H1, mu, beta, dt)
        for tstp = 0:ntime
            incr!(G4, G2, tstp, wz)
            err = err + distance(G3, G4, tstp)
        end
        @test err < ϵ
    end
    #
    @testset "smul! (complex weight)" begin
        mat1 = fill(zero(C64), ndim1, ndim1)
        mat3 = fill(zero(C64), ndim1, ndim1)
        #
        ret1 = fill(zero(C64), ndim1, ndim1)
        ret3 = fill(zero(C64), ndim1, ndim1)
        #
        lmix1 = fill(zero(C64), ndim1, ndim1)
        lmix3 = fill(zero(C64), ndim1, ndim1)
        #
        less1 = fill(zero(C64), ndim1, ndim1)
        less3 = fill(zero(C64), ndim1, ndim1)

        # For mat component
        for q=1:ntau
            @. mat1 = G1.mat[q]
            @. mat3 = mat1 * wz
            G3.mat[q] = mat3
        end

        for i=1:ntime
            # For ret and less components
            for j=1:i
                @. ret1 = G1.ret[i,j]
                @. ret3 = ret1 * wz
                G3.ret[i,j] = ret3

                @. less1 = G1.less[j,i]
                @. less3 = less1 * wz
                G3.less[j,i] = less3
            end

            # For lmix component
            for q=1:ntau
                @. lmix1 = G1.lmix[i,q]
                @. lmix3 = lmix1 * wz
                G3.lmix[i,q] = lmix3
            end
        end

        err = 0.0
        init_green!(G4, H1, mu, beta, dt)
        for tstp = 0:ntime
            smul!(G4, tstp, wz)
            err = err + distance(G3, G4, tstp)
        end
        @test err < ϵ
    end
    #
    @testset "smul! (real weight)" begin
        mat1 = fill(zero(C64), ndim1, ndim1)
        mat3 = fill(zero(C64), ndim1, ndim1)
        #
        ret1 = fill(zero(C64), ndim1, ndim1)
        ret3 = fill(zero(C64), ndim1, ndim1)
        #
        lmix1 = fill(zero(C64), ndim1, ndim1)
        lmix3 = fill(zero(C64), ndim1, ndim1)
        #
        less1 = fill(zero(C64), ndim1, ndim1)
        less3 = fill(zero(C64), ndim1, ndim1)

        # For mat component
        for q=1:ntau
            @. mat1 = G1.mat[q]
            @. mat3 = mat1 * wr
            G3.mat[q] = mat3
        end

        for i=1:ntime
            # For ret and less components
            for j=1:i
                @. ret1 = G1.ret[i,j]
                @. ret3 = ret1 * wr
                G3.ret[i,j] = ret3

                @. less1 = G1.less[j,i]
                @. less3 = less1 * wr
                G3.less[j,i] = less3
            end

            # For lmix component
            for q=1:ntau
                @. lmix1 = G1.lmix[i,q]
                @. lmix3 = lmix1 * wr
                G3.lmix[i,q] = lmix3
            end
        end

        err = 0.0
        init_green!(G4, H1, mu, beta, dt)
        for tstp = 0:ntime
            smul!(G4, tstp, wr)
            err = err + distance(G3, G4, tstp)
        end
        @test err < ϵ 
    end
end
