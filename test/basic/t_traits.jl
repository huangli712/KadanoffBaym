#
# t_traits.jl
#
# To test the basic traits for contour-ordered Green's functions.
#

function exact_leftmultiply_tstp(β::F64, tmax::F64, G::ℱ{T}) where {T}
    ntime = getntime(G)
    ntau = getntau(G)
    ndim1, ndim2 = getdims(G)
    @assert ndim1 == 2
    @assert ndim2 == 2

    δt = tmax / (ntime - 1)
    δτ = β / (ntau - 1)

    ϵ₁ = -1.7071067811865475+0.17677669529663675im
    ϵ₂ = -0.2928932188134524-0.17677669529663687im
    ϵ₃ = -0.07322330470336319-0.7071067811865475im
    ϵ₄ = -0.4267766952966368+0.7071067811865475im
    ϵ₅ = -0.4267766952966371+1.0606601717798207im
    ϵ₆ = -0.0732233047033631-1.0606601717798212im
    ϵ₇ = -0.43933982822017864-0.17677669529663687im
    ϵ₈ = -2.560660171779821+0.17677669529663687im

    # mat and lmix
    mat = fill(zero(C64), ndim1, ndim2)
    lmix = fill(zero(C64), ndim1, ndim2)
    for m = 1:ntau
        τ = (m - 1) * δτ
		mat[1,1] = ϵ₁ * exp(2.0*(β-τ)) + ϵ₂ * exp(2.0*τ)
		mat[1,2] = ϵ₃ * exp(2.0*(β-τ)) + ϵ₄ * exp(2.0*τ)
		mat[2,1] = ϵ₅ * exp(2.0*(β-τ)) + ϵ₆ * exp(2.0*τ)
		mat[2,2] = ϵ₇ * exp(2.0*(β-τ)) + ϵ₈ * exp(2.0*τ)
        @. G.mat[m] = mat / ( 1.0 + exp(2.0*β) )

        for n = 1:ntime
            t1 = (n - 1) * δt
			lmix[1,1] = ϵ₂ * exp(2.0*(β-τ+im*t1)) + ϵ₁ * exp(2.0*(τ-im*t1))
			lmix[1,2] = ϵ₄ * exp(2.0*(β-τ+im*t1)) + ϵ₃ * exp(2.0*(τ-im*t1))
			lmix[2,1] = ϵ₆ * exp(2.0*(β-τ+im*t1)) + ϵ₅ * exp(2.0*(τ-im*t1))
			lmix[2,2] = ϵ₈ * exp(2.0*(β-τ+im*t1)) + ϵ₇ * exp(2.0*(τ-im*t1))
			@. G.lmix[n,m] = -im * cos(t1) * lmix / ( 1.0 + exp(2.0*β) )
        end
    end

	# Les + ret
	ret = fill(zero(C64), ndim1, ndim2)
    less = fill(zero(C64), ndim1, ndim2)

    for m = 1:ntime
        for n = 1:m
			t1 = (m - 1) * δt
			t2 = (n - 1) * δt

			# ret
			ret[1,1] = exp(4.0*im*t1)*(0.17677669529663687-0.2928932188134524im)-exp(4.0*im*t2)*(0.1767766952966371+1.707106781186548im)
			ret[1,2] = exp(4.0*im*t1)*(-0.7071067811865475-0.4267766952966369im)+exp(4.0*im*t2)*(0.7071067811865475-0.07322330470336313im)
			ret[2,1] = exp(4.0*im*t1)*(1.0606601717798212-0.07322330470336319im)-exp(4.0*im*t2)*(1.0606601717798216+0.426776695296637im)
			ret[2,2] = exp(4.0*im*t1)*(-0.17677669529663675-2.5606601717798214im)+exp(4.0*im*t2)*(0.1767766952966369-0.4393398282201787im)
			@. G.ret[m,n] = exp(-2.0*im*(t2+t1)) * cos(t1) * ret

			# less
			t1 = (n - 1) * δt
			t2 = (m - 1) * δt
			less[1,1] = (0.1767766952966371+1.707106781186548im)*exp(2.0*im*(t2-t1))-(0.17677669529663687-0.2928932188134524im)*exp(2.0*im*(t1-t2)+2.0*β)
			less[1,2] = (-0.7071067811865475+0.07322330470336313im)*exp(2.0*im*(t2-t1))+(0.7071067811865475+0.4267766952966369im)*exp(2.0*im*(t1-t2)+2.0*β)
			less[2,1] = (1.0606601717798216+0.426776695296637im)*exp(2.0*im*(t2-t1))+(-1.0606601717798212+0.07322330470336319im)*exp(2.0*im*(t1-t2)+2.0*β)
			less[2,2] = (-0.1767766952966369+0.4393398282201787im)*exp(2.0*im*(t2-t1))+(0.17677669529663675+2.5606601717798214im)*exp(2.0*im*(t1-t2)+2.0*β)
			G.less[n,m] = cos(t1) * less / ( 1.0 + exp(2.0*β) )
		end
	end
end

function exact_rightmultiply_tstp(β::F64, tmax::F64, G::ℱ{T}) where {T}
    ntime = getntime(G)
    ntau = getntau(G)
    ndim1, ndim2 = getdims(G)
    @assert ndim1 == 2
    @assert ndim2 == 2

    δt = tmax / (ntime - 1)
    δτ = β / (ntau - 1)

    ϵ₁ = -1.7071067811865475-0.17677669529663675im
    ϵ₂ = -0.2928932188134524+0.17677669529663687im
    ϵ₃ = -0.4267766952966371-1.0606601717798207im
    ϵ₄ = -0.0732233047033631+1.0606601717798212im
    ϵ₅ = -0.07322330470336319+0.7071067811865475im
    ϵ₆ = -0.4267766952966368-0.7071067811865475im
    ϵ₇ = -0.43933982822017864+0.17677669529663687im
    ϵ₈ = -2.560660171779821-0.17677669529663687im

    # For mat and lmix components
    mat = fill(zero(C64), ndim1, ndim2)
    lmix = fill(zero(C64), ndim1, ndim2)
    for m = 1:ntau
        τ = (m - 1) * δτ
		mat[1,1] = ϵ₁ * exp(2.0*(β-τ)) + ϵ₂ * exp(2.0*τ)
		mat[1,2] = ϵ₃ * exp(2.0*(β-τ)) + ϵ₄ * exp(2.0*τ)
		mat[2,1] = ϵ₅ * exp(2.0*(β-τ)) + ϵ₆ * exp(2.0*τ)
		mat[2,2] = ϵ₇ * exp(2.0*(β-τ)) + ϵ₈ * exp(2.0*τ)
        @. G.mat[m] = mat / ( 1.0 + exp(2.0*β) )

        for n = 1:ntime
			t1 = (n - 1) * δt
			lmix[1,1] = ϵ₂ * exp(2.0*(β-τ+im*t1)) + ϵ₁ * exp(2.0*(τ-im*t1))
			lmix[1,2] = ϵ₄ * exp(2.0*(β-τ+im*t1)) + ϵ₃ * exp(2.0*(τ-im*t1))
			lmix[2,1] = ϵ₆ * exp(2.0*(β-τ+im*t1)) + ϵ₅ * exp(2.0*(τ-im*t1))
			lmix[2,2] = ϵ₈ * exp(2.0*(β-τ+im*t1)) + ϵ₇ * exp(2.0*(τ-im*t1))
            @. G.lmix[n,m] = -im * lmix / ( 1.0 + exp(2.0*β) )
        end
    end

	# For ret and less components
	ret = fill(zero(C64), ndim1, ndim2)
    less = fill(zero(C64), ndim1, ndim2)
    for m = 1:ntime
        for n = 1:m
			t1 = (m - 1) * δt
			t2 = (n - 1) * δt

			# ret
			ret[1,1] = exp(4.0*im*t1) * ϵ₂ + exp(4.0*im*t2) * ϵ₁
			ret[1,2] = exp(4.0*im*t1) * ϵ₄ + exp(4.0*im*t2) * ϵ₃
			ret[2,1] = exp(4.0*im*t1) * ϵ₆ + exp(4.0*im*t2) * ϵ₅
			ret[2,2] = exp(4.0*im*t1) * ϵ₈ + exp(4.0*im*t2) * ϵ₇
			@. G.ret[m,n] = exp(-2.0*im*(t2+t1)) * cos(t2) * im * ret

			# less
			t1 = (n - 1)*δt
			t2 = (m - 1)*δt
			less[1,1] = ϵ₁ * exp(2.0*im*(t2-t1)) + ϵ₂ * exp(2.0*im*(t1-t2)+2.0*β)
			less[1,2] = ϵ₃ * exp(2.0*im*(t2-t1)) + ϵ₄ * exp(2.0*im*(t1-t2)+2.0*β)
			less[2,1] = ϵ₅ * exp(2.0*im*(t2-t1)) + ϵ₆ * exp(2.0*im*(t1-t2)+2.0*β)
			less[2,2] = ϵ₇ * exp(2.0*im*(t2-t1)) + ϵ₈ * exp(2.0*im*(t1-t2)+2.0*β)
			@. G.less[n,m] = -im * cos(t2) * less / ( 1.0 + exp(2.0*β) )
		end
	end
end

@testset verbose = true "KadanoffBaym: traits.jl" begin
    ntime = 201
    ntau = 1001
    ndim1 = 2
    ndim2 = 2
    tmax = 5.0
    beta = 4.0
    sign = FERMI
    #
    δt = 0.025
    μ = 0.0
    ϵ = 1e-6
    ϵ₁ = -0.4; ϵ₂ = 0.6; ϵ₃ = 0.435; ϵ₄ = 0.5676
    λ₁ = 0.1; λ₂ = 0.1566
    wr = 0.3
    wz = 1.0 - 0.3im
    #
    C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    G₁ = ℱ(C, sign)
    G₂ = ℱ(C, sign)
    G₃ = ℱ(C, sign)
    G₄ = ℱ(C, sign)
    #
    H₁ = fill(zero(C64), ndim1, ndim2)
    H₁[1,1] = ϵ₁
    H₁[2,2] = ϵ₂
    H₁[1,2] = im * λ₁
    H₁[2,1] = -im * λ₁
    #
    H₂ = fill(zero(C64), ndim1, ndim2)
    H₂[1,1] = ϵ₃
    H₂[2,2] = ϵ₄
    H₂[1,2] = im * λ₂
    H₂[2,1] = -im * λ₂
    #
    H₃ = fill(zero(C64), ndim1, ndim2)
    H₃[1,1] = sqrt(2.0)
    H₃[2,2] = -sqrt(2.0)
    H₃[1,2] = im * sqrt(2.0)
    H₃[2,1] = -im * sqrt(2.0)
    #
    init_green!(G₁, H₁, μ, beta, δt)
    init_green!(G₂, H₂, μ, beta, δt)
    #
    # For Cn struct
    @testset "comprehensive test 1" begin
        Cnew = deepcopy(C)
        @test Cnew == C
        #
        Cnew.tmax = 4.0
        Cnew.beta = 8.0
        refresh!(Cnew)
        @test Cnew != C
        #
        Cnew.tmax = C.tmax
        Cnew.beta = C.beta
        @test Cnew == C
    end
    #
    # For Cf struct
    @testset "comprehensive test 2" begin
        v₁ = 1.0 + 0.0im
        v₂ = 2.0 + 0.0im
        v₃ = 3.0 + 0.0im
        #
        𝕃 = C64[-1.2+0.3im  -1.2+0.3im; 1.8-0.3im  1.8-0.3im]
        ℝ = C64[-1.2-0.3im  1.8+0.3im; -1.2-0.3im  1.8+0.3im]
        #
        cf₁ = Cf(C, v₁)
        cf₂ = Cf(C, v₂)
        cf₃ = Cf(C, v₃)
        cf₄ = Cf(C)
        #
        @test cf₁ != cf₄
        #
        zeros!(cf₁)
        @test cf₁ == cf₄
        #
        memset!(cf₁, v₁)
        @test cf₁ != cf₄
        #
        memcpy!(cf₁, cf₄)
        @test cf₁ == cf₄
        #
        incr!(cf₁, cf₂, 1.0 + 0.0im)
        @test cf₁ == cf₃
        #
        memset!(cf₁, v₁)
        smul!(cf₁, 3.0 + 0.0im)
        @test cf₁ == cf₃
        #
        memset!(cf₄, v₃)
        smul!(H₁, cf₄)
        for i = 0:cf₄.ntime
            @test cf₄[i] ≈ 𝕃
        end
        #
        memset!(cf₄, v₃)
        smul!(cf₄, H₁)
        for i = 0:cf₄.ntime
            @test cf₄[i] ≈ ℝ
        end
    end
    #
    # For memcpy!()
    @testset "comprehensive test 3" begin
        err = 0.0
        memcpy!(G₁, G₄)
        for tstp = 0:ntime
            err = err + distance(G₁, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For memcpy!()
    @testset "comprehensive test 4" begin
        err = 0.0
        for tstp = 0:ntime
            memcpy!(G₁, G₄, tstp)
            err = err + distance(G₁, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For memcpy!()
    @testset "comprehensive test 5" begin
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            B = 𝒻(C, tstp)
            memcpy!(G₁, A, tstp)
            memcpy!(A, B, tstp)
            err = err + distance(A, B, tstp)
        end
        @test err < ϵ
    end
    #
    # For memcpy!()
    @testset "comprehensive test 6" begin
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G₁, A, tstp)
            memcpy!(A, G₄, tstp)
            err = err + distance(G₁, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For zeros!()
    @testset "comprehensive test 7" begin
        @test G₃ != G₄
        zeros!(G₄)
        @test G₃ == G₄
    end
    #
    # For zeros!()
    @testset "comprehensive test 8" begin
        memcpy!(G₁, G₄)
        @test G₃ != G₄
        zeros!(G₄.mat)
        zeros!(G₄.ret)
        zeros!(G₄.lmix)
        zeros!(G₄.less)
        @test G₃ == G₄
    end
    #
    # For zeros!()
    @testset "comprehensive test 9" begin
        memcpy!(G₁, G₃)
        memcpy!(G₁, G₄)
        err = 0.0
        for tstp = 0:ntime
            zeros!(G₃, tstp)
            if tstp > 0
                zeros!(G₄.ret, tstp)
                zeros!(G₄.lmix, tstp)
                zeros!(G₄.less, tstp)
            else
                zeros!(G₄.mat)
            end
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For zeros!()
    @testset "comprehensive test 10" begin
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            B = 𝒻(C, tstp)
            memcpy!(G₁, A, tstp)
            @test A != B
            zeros!(A)
            @test A == B 
        end
    end
    #
    # For zeros!()
    @testset "comprehensive test 11" begin
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            B = 𝒻(C, tstp)
            #
            memcpy!(G₁, A, tstp)
            memcpy!(G₂, B, tstp)
            #
            zeros!(A, tstp)
            if tstp > 0
                zeros!(B.ret)
                zeros!(B.lmix)
                zeros!(B.less)
            else
                zeros!(B.mat)
            end
            #
            err = err + distance(A, B, tstp)
        end
        @test err < ϵ
    end
    #
    # For memset!()
    @testset "comprehensive test 12" begin
        v₁ = 1.0 + 1.0im
        err = 0.0
        memset!(G₃, v₁)
        for tstp = 0:ntime
            memset!(G₄, tstp, v₁)
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For smul!()
    @testset "comprehensive test 13" begin
        α = 2.0 + 0.0im
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            B = 𝒻(C, tstp)
            #
            memcpy!(G₁, A, tstp)
            memcpy!(G₁, B, tstp)
            #
            if tstp > 0
                smul!(A.ret, α)
                smul!(A.lmix, α)
                smul!(A.less, α)
            else
                smul!(A.mat, α)
            end
            smul!(B, tstp, α)
            #
            err = err + distance(A, B, tstp)
        end
        @test err < ϵ
    end
    #
    # For smul!()
    @testset "comprehensive test 14" begin
        mat₁ = fill(zero(C64), ndim1, ndim1)
        mat₃ = fill(zero(C64), ndim1, ndim1)
        #
        ret₁ = fill(zero(C64), ndim1, ndim1)
        ret₃ = fill(zero(C64), ndim1, ndim1)
        #
        lmix₁ = fill(zero(C64), ndim1, ndim1)
        lmix₃ = fill(zero(C64), ndim1, ndim1)
        #
        less₁ = fill(zero(C64), ndim1, ndim1)
        less₃ = fill(zero(C64), ndim1, ndim1)

        # G₃ = G₁ * wz

        # For mat component
        for q=1:ntau
            @. mat₁ = G₁.mat[q]
            @. mat₃ = mat₁ * wz
            G₃.mat[q] = mat₃
        end

        for i=1:ntime
            # For ret and less components
            for j=1:i
                @. ret₁ = G₁.ret[i,j]
                @. ret₃ = ret₁ * wz
                G₃.ret[i,j] = ret₃

                @. less₁ = G₁.less[j,i]
                @. less₃ = less₁ * wz
                G₃.less[j,i] = less₃
            end

            # For lmix component
            for q=1:ntau
                @. lmix₁ = G₁.lmix[i,q]
                @. lmix₃ = lmix₁ * wz
                G₃.lmix[i,q] = lmix₃
            end
        end

        # Actually, G₄ = G₁ * wz = G₃
        err = 0.0
        init_green!(G₄, H₁, μ, beta, δt)
        for tstp = 0:ntime
            smul!(G₄, tstp, wz)
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For smul!()
    @testset "comprehensive test 15" begin
        mat₁ = fill(zero(C64), ndim1, ndim1)
        mat₃ = fill(zero(C64), ndim1, ndim1)
        #
        ret₁ = fill(zero(C64), ndim1, ndim1)
        ret₃ = fill(zero(C64), ndim1, ndim1)
        #
        lmix₁ = fill(zero(C64), ndim1, ndim1)
        lmix₃ = fill(zero(C64), ndim1, ndim1)
        #
        less₁ = fill(zero(C64), ndim1, ndim1)
        less₃ = fill(zero(C64), ndim1, ndim1)

        # G₃ = G₁ * wr

        # For mat component
        for q = 1:ntau
            @. mat₁ = G₁.mat[q]
            @. mat₃ = mat₁ * wr
            G₃.mat[q] = mat₃
        end

        for i = 1:ntime
            # For ret and less components
            for j = 1:i
                @. ret₁ = G₁.ret[i,j]
                @. ret₃ = ret₁ * wr
                G₃.ret[i,j] = ret₃

                @. less₁ = G₁.less[j,i]
                @. less₃ = less₁ * wr
                G₃.less[j,i] = less₃
            end

            # For lmix component
            for q = 1:ntau
                @. lmix₁ = G₁.lmix[i,q]
                @. lmix₃ = lmix₁ * wr
                G₃.lmix[i,q] = lmix₃
            end
        end

        # Actually, G₄ = G₁ * wr = G₃
        err = 0.0
        init_green!(G₄, H₁, μ, beta, δt)
        for tstp = 0:ntime
            smul!(G₄, tstp, wr)
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For incr!()
    @testset "comprehensive test 16" begin
        mat₁ = fill(zero(C64), ndim1, ndim1)
        mat₂ = fill(zero(C64), ndim1, ndim1)
        mat₃ = fill(zero(C64), ndim1, ndim1)
        #
        ret₁ = fill(zero(C64), ndim1, ndim1)
        ret₂ = fill(zero(C64), ndim1, ndim1)
        ret₃ = fill(zero(C64), ndim1, ndim1)
        #
        lmix₁ = fill(zero(C64), ndim1, ndim1)
        lmix₂ = fill(zero(C64), ndim1, ndim1)
        lmix₃ = fill(zero(C64), ndim1, ndim1)
        #
        less₁ = fill(zero(C64), ndim1, ndim1)
        less₂ = fill(zero(C64), ndim1, ndim1)
        less₃ = fill(zero(C64), ndim1, ndim1)

        # G₃ = G₁ + wz * G₂

        # For mat component
        for q = 1:ntau
            @. mat₁ = G₁.mat[q]
            @. mat₂ = G₂.mat[q]
            @. mat₃ = mat₁ + wz * mat₂
            G₃.mat[q] = mat₃
        end

        for i = 1:ntime
            # For ret and less components
            for j = 1:i
                @. ret₁ = G₁.ret[i,j]
                @. ret₂ = G₂.ret[i,j]
                @. ret₃ = ret₁ + wz * ret₂
                G₃.ret[i,j] = ret₃

                @. less₁ = G₁.less[j,i]
                @. less₂ = G₂.less[j,i]
                @. less₃ = less₁ + wz * less₂
                G₃.less[j,i] = less₃
            end

            # For lmix component
            for q = 1:ntau
                @. lmix₁ = G₁.lmix[i,q]
                @. lmix₂ = G₂.lmix[i,q]
                @. lmix₃ = lmix₁ + wz * lmix₂
                G₃.lmix[i,q] = lmix₃
            end
        end

        # Actually, G₄ = G₁ + wz * G₂ = G₃
        err = 0.0
        init_green!(G₄, H₁, μ, beta, δt)
        incr!(G₄, G₂, wz)
        for tstp = 0:ntime
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ

        err = 0.0
        init_green!(G₄, H₁, μ, beta, δt)
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G₂, A, tstp)
            incr!(G₄, A, tstp, wz)
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ

        err = 0.0
        init_green!(G₄, H₁, μ, beta, δt)
        for tstp = 0:ntime
            incr!(G₄, G₂, tstp, wz)
            err = err + distance(G₃, G₄, tstp)
        end
        @test err < ϵ
    end
    #
    # For smul!()
    @testset "comprehensive test 17" begin
        cf = Cf(C)
        for tstp = 0:ntime
            if tstp == 0
                t = 0
            else
                t = (tstp - 1) * δt
            end
            #
            cf[tstp] = C64[2.0*cos(t) 0.5*cos(t); 0.5*cos(t) 3.0*cos(t)]
        end
        #
        init_green!(G₃, H₃, μ, beta, δt)
        init_green!(G₄, H₃, μ, beta, δt)

        for tstp = 0:ntime
            smul!(cf, G₃, tstp)
            smul!(G₄, cf, tstp)
        end
        #
        err = 0.0
        𝕃 = ℱ(C, sign)
        exact_leftmultiply_tstp(beta, tmax, 𝕃)
        for tstp = 0:ntime
            err = err + distance(G₃, 𝕃, tstp)
        end
        @test err < ϵ
        #
        err = 0.0
        ℝ = ℱ(C, sign)
        exact_rightmultiply_tstp(beta, tmax, ℝ)
        for tstp = 0:ntime
            err = err + distance(G₄, ℝ, tstp)
        end
        @test err < ϵ
    end
    #
    # For smul!()
    @testset "comprehensive test 18" begin
        cf = Cf(C)
        unity = Cf(C)
        #
        for tstp = 0:ntime
            if tstp == 0
                t = 0
            else
                t = (tstp - 1) * δt
            end
            #
            cf[tstp] = C64[2.0*cos(t) 0.5*cos(t); 0.5*cos(t) 3.0*cos(t)]
            unity[tstp] = C64[1.0 0.0; 0.0 1.0]
        end
        #
        init_green!(G₃, H₃, μ, beta, δt)

        err = 0.0
        𝕃 = ℱ(C, sign)
        exact_leftmultiply_tstp(beta, tmax, 𝕃)
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G₃, A, tstp)
            smul!(cf, A, tstp)
            err = err + distance(𝕃, A, tstp)
        end
        @test err < ϵ

        err = 0.0
        ℝ = ℱ(C, sign)
        exact_rightmultiply_tstp(beta, tmax, ℝ)
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G₃, A, tstp)
            smul!(A, cf, tstp)
            err = err + distance(A, ℝ, tstp)
        end
        @test err < ϵ

        G₄ = deepcopy(G₃)
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            smul!(G₄, unity * 4.0, tstp)
            memcpy!(G₃, A, tstp)
            incr!(A, G₃, tstp, 3.0)
            err = err + distance(G₄, A, tstp)
        end
        @test err < ϵ

        G₄ = deepcopy(G₃)
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            smul!(G₄, unity * 4.0, tstp)
            memcpy!(G₃, A, tstp)
            incr!(A, A, tstp, 3.0)
            err = err + distance(G₄, A, tstp)
        end
        @test err < ϵ
    end
end

println("All tests pass!\n")
