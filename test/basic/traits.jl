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

@testset verbose = true "KadanoffBaym: traits.jl" begin
    ntime = 101
    ntau = 51
    ndim1 = 2
    ndim2 = 2
    tmax = 1.0
    beta = 10.0
    dt = 0.01
    mu = 0.0 
    ϵ = 1e-6; ϵ₁ = -0.4; ϵ₂ = 0.6
    λ = 0.1
    #
    C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
    G1 = ℱ(C, FERMI)
    G2 = ℱ(C, FERMI)
    #
    H0 = fill(zero(C64), ndim1, ndim1)
    H0[1,1] = ϵ₁
    H0[2,2] = ϵ₂
    H0[1,2] = im * λ
    H0[2,1] = -im * λ
    #
    init_green!(G1, H0, mu, beta, dt)
    #
    @testset "memcpy!" begin
        err = 0.0
        for tstp = 0:ntime
            A = 𝒻(C, tstp)
            memcpy!(G1, A, tstp)
            memcpy!(A, G2, tstp)
            err = err + distance(G1, G2, tstp)
        end
        @test err < ϵ
    end
end

function exact_rightmultiply_tstp(beta::F64, dt::F64, G::ℱ{T}) where {T}
    ntau = getntau(G)
    ntime = getntime(G)
    ndim1, _ = getdims(G)
    @assert ndim1 == 2

    dtau = beta / (ntau - 1)

    # For mat and lmix components
    mat = fill(zero(C64), ndim1, ndim1)
    lmix = fill(zero(C64), ndim1, ndim1)
    for m = 1:ntau
        tau = (m - 1) * dtau
		mat[1,1]=(-1.7071067811865475-0.17677669529663675im)*exp(-2*tau)*exp(beta*2.0) + (-0.2928932188134524+0.17677669529663687im)*exp(2.0*tau)
		mat[1,2]=(-0.4267766952966371-1.0606601717798207im)*exp(-2*tau)*exp(beta*2.0) + (-0.0732233047033631+1.0606601717798212im)*exp(2.0*tau)
		mat[2,1]=(-0.07322330470336319+0.7071067811865475im)*exp(-2*tau)*exp(beta*2.0) + (-0.4267766952966368-0.7071067811865475im)*exp(2.0*tau)
		mat[2,2]=(-0.43933982822017864+0.17677669529663687im)*exp(-2*tau)*exp(beta*2.0) + (-2.560660171779821-0.17677669529663687im)*exp(2.0*tau)
		mat = mat / (1.0+exp(2.0*beta))
        G.mat[m] = mat

        for n = 1:ntime
			t1 = (n - 1) * dt
			lmix[1,1]=(0.17677669529663687+0.2928932188134524im)*exp(2.0*im*t1 + 2.0*(beta-tau)) - (0.1767766952966371-1.707106781186548im)*exp(-2.0*im*t1 + tau*2.0)
			lmix[1,2]=(1.0606601717798212+0.07322330470336319im)*exp(2.0*im*t1 + 2.0*(beta-tau)) - (1.0606601717798216-0.426776695296637im)*exp(-2.0*im*t1 + tau*2.0)
			lmix[2,1]=-(0.7071067811865475-0.4267766952966369im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (0.7071067811865475+0.07322330470336313im)*exp(-2.0*im*t1 + tau*2.0)
			lmix[2,2]=-(0.17677669529663675-2.5606601717798214im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (0.1767766952966369+0.4393398282201787im)*exp(-2.0*im*t1 + tau*2.0)
			lmix = lmix / (1.0+exp(2.0*beta))
            G.lmix[n,m] = lmix
        end
    end

	# For ret and less components
	ret = fill(zero(C64), ndim1, ndim1)
    less = fill(zero(C64), ndim1, ndim1)
    for m = 1:ntime
        for n = 1:m
			t1 = (m - 1)*dt
			t2 = (n - 1)*dt

			# ret
			ret[1,1]=exp(-2.0*im*(t2+t1))*cos(t2)*(exp(4.0*im*t1)*(-0.17677669529663687-0.2928932188134524im)+exp(4.0*im*t2)*(0.1767766952966371-1.707106781186548im))
			ret[1,2]=exp(-2.0*im*(t2+t1))*cos(t2)*(exp(4.0*im*t1)*(-1.0606601717798212-0.07322330470336319im)+exp(4.0*im*t2)*(1.0606601717798216-0.426776695296637im))
			ret[2,1]=exp(-2.0*im*(t2+t1))*cos(t2)*(exp(4.0*im*t1)*(0.7071067811865475-0.4267766952966369im)-exp(4.0*im*t2)*(0.7071067811865475+0.07322330470336313im))
			ret[2,2]=exp(-2.0*im*(t2+t1))*cos(t2)*(exp(4.0*im*t1)*(0.17677669529663675-2.5606601717798214im)-exp(4.0*im*t2)*(0.1767766952966369+0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=exp(-2.0*im*(t2+t1))*cos(t2)*((-0.1767766952966371+1.707106781186548im)*exp(4.0*im*t2)+(0.17677669529663687+0.2928932188134524im)*exp(4.0*im*t1+2.0*beta))
			less[1,2]=exp(-2.0*im*(t2+t1))*cos(t2)*((-1.0606601717798216+0.426776695296637im)*exp(4.0*im*t2)+(1.0606601717798212+0.07322330470336319im)*exp(4.0*im*t1+2.0*beta))
			less[2,1]=cos(t2)*((0.7071067811865475+0.07322330470336313im)*exp(2.0*im*(t2-t1))+(-0.7071067811865475+0.4267766952966369im)*exp(2.0*im*(t1-t2)+2.0*beta))
			less[2,2]=cos(t2)*((0.1767766952966369+0.4393398282201787im)*exp(2.0*im*(t2-t1))+(-0.17677669529663675+2.5606601717798214im)*exp(2.0*im*(t1-t2)+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end

function exact_leftmultiply_tstp(beta::F64, dt::F64, G::ℱ{T}) where {T}
    ntau = getntau(G)
    ntime = getntime(G)
    ndim1, _ = getdims(G)
    @assert ndim1 == 2

    dtau = beta / (ntau - 1)

    # mat and lmix
    mat = fill(zero(C64), ndim1, ndim1)
    lmix = fill(zero(C64), ndim1, ndim1)
    for m = 1:ntau
        tau = (m - 1) * dtau
		mat[1,1]=(-1.7071067811865475+0.17677669529663675im)*exp(-2*tau)*exp(beta*2.0) + (-0.2928932188134524-0.17677669529663687im)*exp(2.0*tau)
		mat[1,2]=(-0.07322330470336319-0.7071067811865475im)*exp(-2*tau)*exp(beta*2.0) + (-0.4267766952966368+0.7071067811865475im)*exp(2.0*tau)
		mat[2,1]=(-0.4267766952966371+1.0606601717798207im)*exp(-2*tau)*exp(beta*2.0) + (-0.0732233047033631-1.0606601717798212im)*exp(2.0*tau)
		mat[2,2]=(-0.43933982822017864-0.17677669529663687im)*exp(-2*tau)*exp(beta*2.0) + (-2.560660171779821+0.17677669529663687im)*exp(2.0*tau)
		mat = mat / (1.0+exp(2.0*beta))
        G.mat[m] = mat

        for n = 1:ntime
            t1 = (n - 1) * dt
			lmix[1,1]=cos(t1) * ((-0.17677669529663687+0.2928932188134524im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (0.1767766952966371+1.707106781186548im)*exp(-2.0*im*t1 + tau*2.0))
			lmix[1,2]=cos(t1) * ((0.7071067811865475+0.4267766952966369im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (-0.7071067811865475+0.07322330470336313im)*exp(-2.0*im*t1 + tau*2.0))
			lmix[2,1]=cos(t1) * ((-1.0606601717798212+0.07322330470336319im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (1.0606601717798216+0.426776695296637im)*exp(-2.0*im*t1 + tau*2.0))
			lmix[2,2]=cos(t1) * ((0.17677669529663675+2.5606601717798214im)*exp(2.0*im*t1 + 2.0*(beta-tau)) + (-0.1767766952966369+0.4393398282201787im)*exp(-2.0*im*t1 + tau*2.0))
            lmix = lmix / (1.0+exp(2.0*beta))
			G.lmix[n,m] = lmix
        end
    end

	# Les + ret
	ret = fill(zero(C64), ndim1, ndim1)
    less = fill(zero(C64), ndim1, ndim1)

    for m = 1:ntime
        for n = 1:m
			t1 = (m - 1)*dt
			t2 = (n - 1)*dt

			# ret
			ret[1,1]=exp(-2.0*im*(t2+t1))*cos(t1)*(exp(4.0*im*t1)*(0.17677669529663687-0.2928932188134524im)-exp(4.0*im*t2)*(0.1767766952966371+1.707106781186548im))
			ret[1,2]=exp(-2.0*im*(t2+t1))*cos(t1)*(exp(4.0*im*t1)*(-0.7071067811865475-0.4267766952966369im)+exp(4.0*im*t2)*(0.7071067811865475-0.07322330470336313im))
			ret[2,1]=exp(-2.0*im*(t2+t1))*cos(t1)*(exp(4.0*im*t1)*(1.0606601717798212-0.07322330470336319im)-exp(4.0*im*t2)*(1.0606601717798216+0.426776695296637im))
			ret[2,2]=exp(-2.0*im*(t2+t1))*cos(t1)*(exp(4.0*im*t1)*(-0.17677669529663675-2.5606601717798214im)+exp(4.0*im*t2)*(0.1767766952966369-0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=cos(t1) *((0.1767766952966371+1.707106781186548im)*exp(2.0*im*(t2-t1))-(0.17677669529663687-0.2928932188134524im)*exp(2.0*im*(t1-t2)+2.0*beta))
			less[1,2]=cos(t1) * exp(-2.0*im*(t2+t1))*((-0.7071067811865475+0.07322330470336313im)*exp(4.0*im*t2)+(0.7071067811865475+0.4267766952966369im)*exp(4.0*im*t1+2.0*beta))
			less[2,1]=cos(t1) *((1.0606601717798216+0.426776695296637im)*exp(2.0*im*(t2-t1))+(-1.0606601717798212+0.07322330470336319im)*exp(2.0*im*(t1-t2)+2.0*beta))
			less[2,2]=cos(t1) * exp(-2.0*im*(t2+t1)) *((-0.1767766952966369+0.4393398282201787im)*exp(4.0*im*t2)+(0.17677669529663675+2.5606601717798214im)*exp(4.0*im*t1+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end

function setget(cfv::𝒻{T}, a::Element{T}) where {T}
    toterr = 0.0

    # For Matsubara component
    for m = 1:getntau(cfv)
        tmp = similar(a)
        @. cfv.mat[m] = a
        @. tmp = cfv.mat[m]
        toterr = toterr + abs(sum(a - tmp))
    end

    # For left-mixing component
    for m = 1:getntau(cfv)
        tmp = similar(a)
        @. cfv.lmix[m] = a
        @. tmp = cfv.lmix[m]
        toterr = toterr + abs(sum(a - tmp))
    end

    # For retarded and lesser components
    for m = 1:gettstp(cfv)
        ret = similar(a)
        less = similar(a)
        @. cfv.ret[m] = a
        @. ret = cfv.ret[m]
        toterr = toterr + abs(sum(a - ret))
        @. cfv.less[m] = a
        @. less = cfv.less[m]
        toterr = toterr + abs(sum(a - less))
    end

    return toterr
end

@testset verbose = true "KadanoffBaym: traits.jl" begin
    ntime = 51
    ntau = 501
    ndim1 = 2
    ndim2 = 5
    tmax = 0.5
    beta = 5.0
    dt = 0.01
    mu = 0.0
    ϵ = 1.0e-7
    #
    C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
    G = ℱ(C, FERMI)
    A = ℱ(C, FERMI)
    B = ℱ(C, FERMI)
    #
    H = fill(zero(C64), ndim1, ndim1)
    H[1,1] = sqrt(2.0)
    H[1,2] = sqrt(2.0) * im
    H[2,1] = sqrt(2.0) * (-im)
    H[2,2] = -sqrt(2.0)
    #
    init_green!(A, H, 0.0, beta, dt)
    #
    funcC = Cf(C)
    unity = Cf(C)
    c = fill(zero(C64), ndim1, ndim1)
    one = fill(zero(C64), ndim1, ndim1)
    #
    for tstp = 0:ntime
        if tstp == 0
            t = 0
        else
            t = (tstp - 1) * dt
        end
        #
        c[1,1] = 2.0 * cos(t)
        c[1,2] = 0.5 * cos(t)
        c[2,1] = 0.5 * cos(t)
        c[2,2] = 3.0 * cos(t)
        #
        one[1,1] = 1.0
        one[1,2] = 0.0
        one[2,1] = 0.0
        one[2,2] = 1.0
        #
        funcC[tstp] = c
        unity[tstp] = one
    end
    #
    exactR = ℱ(C, FERMI)
    exactL = ℱ(C, FERMI)
    exact_rightmultiply_tstp(beta, dt, exactR)
    exact_leftmultiply_tstp(beta, dt, exactL)
    #
    @testset "memcpy!" begin
        @test getntime(G) == ntime

        err = 0.0
        for tstp = 0:ntime
            Atstp = 𝒻(C, tstp)
            err = err + setget(Atstp, H)
        end
        @test err < ϵ

        err = 0.0
        for tstp = 0:ntime
            Atstp = 𝒻(C, tstp)
            memcpy!(A, Atstp, tstp)
            memcpy!(Atstp, B, tstp)
            err = err + distance(A, B, tstp)
        end
        @test err < ϵ

        err = 0.0
        for tstp = 0:ntime
            Atstp = 𝒻(C, tstp)
            memcpy!(A, Atstp, tstp)
            smul!(Atstp, funcC, tstp)
            err = err + distance(Atstp, exactR, tstp)
        end
        @test err < ϵ
    end
end