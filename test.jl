include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

#
# See NESSi/libcntr/test/herm_matrix_algebra.cpp
#

println("Test CnFunM and related structs")

# Parameters
ndim1 = 2; ndim2 = 2
ntime = 101
ntau = 51
eps = 1e-6
dt = 0.01; mu = 0.0; beta = 10.0
tmax = 1.0
eps1 = -0.4; eps2 = 0.6; lam1 = 0.1
eps3 = 0.435; eps4 = 0.5676; lam2 = 0.1566
wr = 0.3
wz = 1.0 - 0.3im

# Contour and Green
C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
G1 = CnFunM(C, FERMI)
G2 = CnFunM(C, FERMI)
G3 = CnFunM(C, FERMI)
G4 = CnFunM(C, FERMI)

# Matrix
h1 = fill(zero(C64), ndim1, ndim1)
h2 = fill(zero(C64), ndim1, ndim1)

# Setup Matrix
h1[1,1] = eps1
h1[2,2] = eps2
h1[1,2] = CZI * lam1
h1[2,1] = -CZI * lam1

h2[1,1] = eps3
h2[2,2] = eps4
h2[1,2] = CZI * lam2
h2[2,1] = -CZI * lam2

# Setup Green
init_green!(G1, h1, mu, beta, dt)
init_green!(G2, h2, mu, beta, dt)

# Test Set/Get/incr!/memcpy!
# TEST 1 & TEST 2 & TEST 3
begin
    mat1 = fill(zero(C64), ndim1, ndim1)
    mat2 = fill(zero(C64), ndim1, ndim1)
    mat3 = fill(zero(C64), ndim1, ndim1)

    ret1 = fill(zero(C64), ndim1, ndim1)
    ret2 = fill(zero(C64), ndim1, ndim1)
    ret3 = fill(zero(C64), ndim1, ndim1)

    lmix1 = fill(zero(C64), ndim1, ndim1)
    lmix2 = fill(zero(C64), ndim1, ndim1)
    lmix3 = fill(zero(C64), ndim1, ndim1)

    less1 = fill(zero(C64), ndim1, ndim1)
    less2 = fill(zero(C64), ndim1, ndim1)
    less3 = fill(zero(C64), ndim1, ndim1)

    # For mat component
    for q=1:ntau
        @. mat1 = G1.mat[q]
        @. mat2 = G2.mat[q]
        @. mat3 = mat1 + wz * mat2
        G3.mat[q] = mat3
    end

    for i=1:ntime
        # For ret and less components
        for j=1:i
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
        for q=1:ntau
            @. lmix1 = G1.lmix[i,q]
            @. lmix2 = G2.lmix[i,q]
            @. lmix3 = lmix1 + wz * lmix2
            G3.lmix[i,q] = lmix3
        end
    end

    err = 0.0
    init_green!(G4, h1, mu, beta, dt)
    incr!(G4, G2, wz)
    for tstp = 0:ntime
        global err = err + distance(G3, G4, tstp)
    end
    @test err < eps

    err = 0.0
    init_green!(G4, h1, mu, beta, dt)
    for tstp = 0:ntime
        A = CnFunV(C, tstp)
        memcpy!(G2, A, tstp)
        incr!(G4, A, tstp, wz)
        global err = err + distance(G3, G4, tstp)
    end
    @test err < eps

    err = 0.0
    init_green!(G4, h1, mu, beta, dt)
    for tstp = 0:ntime
        incr!(G4, G2, tstp, wz)
        global err = err + distance(G3, G4, tstp)
    end
    @test err < eps
end

# Test smul
# TEST 4
begin
    mat1 = fill(zero(C64), ndim1, ndim1)
    mat3 = fill(zero(C64), ndim1, ndim1)

    ret1 = fill(zero(C64), ndim1, ndim1)
    ret3 = fill(zero(C64), ndim1, ndim1)

    lmix1 = fill(zero(C64), ndim1, ndim1)
    lmix3 = fill(zero(C64), ndim1, ndim1)

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
    init_green!(G4, h1, mu, beta, dt)
    for tstp = 0:ntime
        smul!(G4, tstp, wz)
        global err = err + distance(G3, G4, tstp)
    end
    @test err < eps
end

# Test smul
# TEST 5
begin
    mat1 = fill(zero(C64), ndim1, ndim1)
    mat3 = fill(zero(C64), ndim1, ndim1)

    ret1 = fill(zero(C64), ndim1, ndim1)
    ret3 = fill(zero(C64), ndim1, ndim1)

    lmix1 = fill(zero(C64), ndim1, ndim1)
    lmix3 = fill(zero(C64), ndim1, ndim1)

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
    init_green!(G4, h1, mu, beta, dt)
    for tstp = 0:ntime
        smul!(G4, tstp, wr)
        global err = err + distance(G3, G4, tstp)
    end
    @test err < eps 
end

include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

#
# See NESSi/libcntr/test/herm_member_timestep.cpp
#

function setget(cfv::CnFunV{T}, a::Element{T}) where {T}
    toterr = 0.0

    # For Matsubara component
    for m = 1:getntau(cfv)
        tmp = similar(a)
        cfv.mat[m] = a
        tmp = cfv.mat[m]
        toterr = toterr + abs(sum(a - tmp))
    end

    # For left-mixing component
    for m = 1:getntau(cfv)
        tmp = similar(a)
        cfv.lmix[m] = a
        tmp = cfv.lmix[m]
        toterr = toterr + abs(sum(a - tmp))
    end

    # For retarded and lesser components
    for m = 1:gettstp(cfv)
        ret = similar(a)
        less = similar(a)
        cfv.ret[m] = a
        ret = cfv.ret[m]
        toterr = toterr + abs(sum(a - ret))
        cfv.less[m] = a
        less = cfv.less[m]
        toterr = toterr + abs(sum(a - less))
    end

    return toterr
end

function exact_rightmultiply_tstp(beta::F64, dt::F64, G::CnFunM{T}) where {T}
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
		mat[1,1]=(-1.7071067811865475-0.17677669529663675im)*exp(-2*tau)*exp(beta*2.0) + (-0.2928932188134524+0.17677669529663687im)*exp(2.0*tau)
		mat[1,2]=(-0.4267766952966371-1.0606601717798207im)*exp(-2*tau)*exp(beta*2.0) + (-0.0732233047033631+1.0606601717798212im)*exp(2.0*tau)
		mat[2,1]=(-0.07322330470336319+0.7071067811865475im)*exp(-2*tau)*exp(beta*2.0) + (-0.4267766952966368-0.7071067811865475im)*exp(2.0*tau)
		mat[2,2]=(-0.43933982822017864+0.17677669529663687im)*exp(-2*tau)*exp(beta*2.0) + (-2.560660171779821-0.17677669529663687im)*exp(2.0*tau)
		mat = mat / (1.0+exp(2.0*beta))
        G.mat[m] = mat

        for n = 1:ntime
			t1 = (n - 1) * dt
			lmix[1,1]=(0.17677669529663687+0.2928932188134524im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) - (0.1767766952966371-1.707106781186548im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[1,2]=(1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) - (1.0606601717798216-0.426776695296637im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[2,1]=-(0.7071067811865475-0.4267766952966369im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.7071067811865475+0.07322330470336313im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[2,2]=-(0.17677669529663675-2.5606601717798214im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.1767766952966369+0.4393398282201787im)*exp(-2.0*CZI*t1 + tau*2.0)
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
			ret[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(-0.17677669529663687-0.2928932188134524im)+exp(4.0*CZI*t2)*(0.1767766952966371-1.707106781186548im))
			ret[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(-1.0606601717798212-0.07322330470336319im)+exp(4.0*CZI*t2)*(1.0606601717798216-0.426776695296637im))
			ret[2,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(0.7071067811865475-0.4267766952966369im)-exp(4.0*CZI*t2)*(0.7071067811865475+0.07322330470336313im))
			ret[2,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(0.17677669529663675-2.5606601717798214im)-exp(4.0*CZI*t2)*(0.1767766952966369+0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*((-0.1767766952966371+1.707106781186548im)*exp(4.0*CZI*t2)+(0.17677669529663687+0.2928932188134524im)*exp(4.0*CZI*t1+2.0*beta))
			less[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*((-1.0606601717798216+0.426776695296637im)*exp(4.0*CZI*t2)+(1.0606601717798212+0.07322330470336319im)*exp(4.0*CZI*t1+2.0*beta))
			less[2,1]=cos(t2)*((0.7071067811865475+0.07322330470336313im)*exp(2.0*CZI*(t2-t1))+(-0.7071067811865475+0.4267766952966369im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[2,2]=cos(t2)*((0.1767766952966369+0.4393398282201787im)*exp(2.0*CZI*(t2-t1))+(-0.17677669529663675+2.5606601717798214im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end

function exact_leftmultiply_tstp(beta::F64, dt::F64, G::CnFunM{T}) where {T}
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
			lmix[1,1]=cos(t1) * ((-0.17677669529663687+0.2928932188134524im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.1767766952966371+1.707106781186548im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[1,2]=cos(t1) * ((0.7071067811865475+0.4267766952966369im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (-0.7071067811865475+0.07322330470336313im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[2,1]=cos(t1) * ((-1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (1.0606601717798216+0.426776695296637im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[2,2]=cos(t1) * ((0.17677669529663675+2.5606601717798214im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (-0.1767766952966369+0.4393398282201787im)*exp(-2.0*CZI*t1 + tau*2.0))
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
			ret[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(0.17677669529663687-0.2928932188134524im)-exp(4.0*CZI*t2)*(0.1767766952966371+1.707106781186548im))
			ret[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(-0.7071067811865475-0.4267766952966369im)+exp(4.0*CZI*t2)*(0.7071067811865475-0.07322330470336313im))
			ret[2,1]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(1.0606601717798212-0.07322330470336319im)-exp(4.0*CZI*t2)*(1.0606601717798216+0.426776695296637im))
			ret[2,2]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(-0.17677669529663675-2.5606601717798214im)+exp(4.0*CZI*t2)*(0.1767766952966369-0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=cos(t1) *((0.1767766952966371+1.707106781186548im)*exp(2.0*CZI*(t2-t1))-(0.17677669529663687-0.2928932188134524im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[1,2]=cos(t1) * exp(-2.0*CZI*(t2+t1))*((-0.7071067811865475+0.07322330470336313im)*exp(4.0*CZI*t2)+(0.7071067811865475+0.4267766952966369im)*exp(4.0*CZI*t1+2.0*beta))
			less[2,1]=cos(t1) *((1.0606601717798216+0.426776695296637im)*exp(2.0*CZI*(t2-t1))+(-1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[2,2]=cos(t1) * exp(-2.0*CZI*(t2+t1)) *((-0.1767766952966369+0.4393398282201787im)*exp(4.0*CZI*t2)+(0.17677669529663675+2.5606601717798214im)*exp(4.0*CZI*t1+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end


println("Test CnFunV and related structs")

# Parameters
ntime = 51
ntau = 501
ndim1 = 2; ndim2 = 5
eps = 1.0e-7
beta = 5.0
tmax = 0.5
h = 0.01
mu = 0.0

# Contour and Green
C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
g = CnFunM(C, FERMI)
A = CnFunM(C, FERMI)

# Setup H₀
a = fill(zero(C64), ndim1, ndim1)
a[1,1] = sqrt(2.0)
a[1,2] = sqrt(2.0) * CZI
a[2,1] = sqrt(2.0) * (-CZI)
a[2,2] = -sqrt(2.0)

# Init Green A
init_green!(A, a, 0.0, beta, h)

# Setup functions
funcC = CnFunF(C)
unity = CnFunF(C)
c = fill(zero(C64), ndim1, ndim1)
one = fill(zero(C64), ndim1, ndim1)
for tstp = 0:ntime
    if tstp == 0
        t = 0
    else
        t = (tstp - 1) * h
    end
    c[1,1] = 2.0 * cos(t)
    c[1,2] = 0.5 * cos(t)
    c[2,1] = 0.5 * cos(t)
    c[2,2] = 3.0 * cos(t)

    one[1,1] = 1.0
    one[1,2] = 0.0
    one[2,1] = 0.0
    one[2,2] = 1.0
    funcC[tstp] = c
    unity[tstp] = one
end

# Generate exact solution
exactR = CnFunM(C, FERMI)
exactL = CnFunM(C, FERMI)
exact_rightmultiply_tstp(beta, h, exactR)
exact_leftmultiply_tstp(beta, h, exactL)

# TEST 1
@test getntime(g) == ntime

# Set/Get from CnFunV
# TEST 2
begin
    err = 0.0
    errB = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        global err = err + setget(Atstp, a)
    end
    @test err < eps
end

# TEST 3
begin
    B = CnFunM(C, FERMI)
    err = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        memcpy!(A, Atstp, tstp)
        memcpy!(Atstp, B, tstp)
        global err = err + distance(A, B, tstp)
    end
    @test err < eps
end

# Right multiply
# TEST 4
begin
    err = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        memcpy!(A, Atstp, tstp)
        smul!(Atstp, funcC, tstp)
        global err = err + distance(Atstp, exactR, tstp)
    end
    @test err < eps
end

# Left multiply
# TEST 5
begin
    err = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        memcpy!(A, Atstp, tstp)
        smul!(funcC, Atstp, tstp)
        global err = err + distance(exactL, Atstp, tstp)
    end
    @test err < eps
end

# Right multiply, incr, Add
# TEST 6
begin
    A2 = deepcopy(A)
    err = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        smul!(A2, unity * 4.0, tstp)
        memcpy!(A, Atstp, tstp)
        incr!(Atstp, A, tstp, 3.0) 
        global err = err + distance(A2, Atstp, tstp)
    end
    @test err < eps
end

# TEST 7
begin
    A2 = deepcopy(A)
    err = 0.0
    for tstp = 0:ntime
        Atstp = CnFunV(C, tstp)
        smul!(A2, unity * 4.222, tstp)
        memcpy!(A, Atstp, tstp)
        incr!(Atstp, Atstp, tstp, 3.222) 
        global err = err + distance(A2, Atstp, tstp)
    end
    @test err < eps
end

include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

#
# See NESSi/libcntr/test/herm_matrix_setget_timestep.cpp
#

println("Test CnFunM and related structs")

# Parameters
ndim1 = 2; ndim2 = 2
ntime = 101
ntau = 51
eps = 1e-6
h = 0.01; mu = 0.0; beta = 10.0
tmax = 1.0
eps1 = -0.4; eps2 = 0.6; lam = 0.1

# Contour and Green
C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
G1 = CnFunM(C, FERMI)
G2 = CnFunM(C, FERMI)

# Matrix
h0 = fill(zero(C64), ndim1, ndim1)

# Setup Matrix
h0[1,1] = eps1
h0[2,2] = eps2
h0[1,2] = CZI * lam
h0[2,1] = -CZI * lam

# Setup Green
init_green!(G1, h0, mu, beta, h)

# Test memcpy!
# TEST 1
begin
    err = 0.0
    for tstp = 0:ntime
        A = CnFunV(C, tstp)
        memcpy!(G1, A, tstp)
        memcpy!(A, G2, tstp)
        global err = err + distance(G1, G2, tstp)
    end

    @test err < eps
end

include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

function exact_rightmultiply_tstp(beta::F64, dt::F64, G::CnFunM{T}) where {T}
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
		mat[1,1]=(-1.7071067811865475-0.17677669529663675im)*exp(-2*tau)*exp(beta*2.0) + (-0.2928932188134524+0.17677669529663687im)*exp(2.0*tau)
		mat[1,2]=(-0.4267766952966371-1.0606601717798207im)*exp(-2*tau)*exp(beta*2.0) + (-0.0732233047033631+1.0606601717798212im)*exp(2.0*tau)
		mat[2,1]=(-0.07322330470336319+0.7071067811865475im)*exp(-2*tau)*exp(beta*2.0) + (-0.4267766952966368-0.7071067811865475im)*exp(2.0*tau)
		mat[2,2]=(-0.43933982822017864+0.17677669529663687im)*exp(-2*tau)*exp(beta*2.0) + (-2.560660171779821-0.17677669529663687im)*exp(2.0*tau)
		mat = mat / (1.0+exp(2.0*beta))
        G.mat[m] = mat

        for n = 1:ntime
			t1 = (n - 1) * dt
			lmix[1,1]=(0.17677669529663687+0.2928932188134524im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) - (0.1767766952966371-1.707106781186548im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[1,2]=(1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) - (1.0606601717798216-0.426776695296637im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[2,1]=-(0.7071067811865475-0.4267766952966369im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.7071067811865475+0.07322330470336313im)*exp(-2.0*CZI*t1 + tau*2.0)
			lmix[2,2]=-(0.17677669529663675-2.5606601717798214im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.1767766952966369+0.4393398282201787im)*exp(-2.0*CZI*t1 + tau*2.0)
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
			ret[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(-0.17677669529663687-0.2928932188134524im)+exp(4.0*CZI*t2)*(0.1767766952966371-1.707106781186548im))
			ret[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(-1.0606601717798212-0.07322330470336319im)+exp(4.0*CZI*t2)*(1.0606601717798216-0.426776695296637im))
			ret[2,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(0.7071067811865475-0.4267766952966369im)-exp(4.0*CZI*t2)*(0.7071067811865475+0.07322330470336313im))
			ret[2,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*(exp(4.0*CZI*t1)*(0.17677669529663675-2.5606601717798214im)-exp(4.0*CZI*t2)*(0.1767766952966369+0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t2)*((-0.1767766952966371+1.707106781186548im)*exp(4.0*CZI*t2)+(0.17677669529663687+0.2928932188134524im)*exp(4.0*CZI*t1+2.0*beta))
			less[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t2)*((-1.0606601717798216+0.426776695296637im)*exp(4.0*CZI*t2)+(1.0606601717798212+0.07322330470336319im)*exp(4.0*CZI*t1+2.0*beta))
			less[2,1]=cos(t2)*((0.7071067811865475+0.07322330470336313im)*exp(2.0*CZI*(t2-t1))+(-0.7071067811865475+0.4267766952966369im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[2,2]=cos(t2)*((0.1767766952966369+0.4393398282201787im)*exp(2.0*CZI*(t2-t1))+(-0.17677669529663675+2.5606601717798214im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end

function exact_leftmultiply_tstp(beta::F64, dt::F64, G::CnFunM{T}) where {T}
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
			lmix[1,1]=cos(t1) * ((-0.17677669529663687+0.2928932188134524im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (0.1767766952966371+1.707106781186548im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[1,2]=cos(t1) * ((0.7071067811865475+0.4267766952966369im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (-0.7071067811865475+0.07322330470336313im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[2,1]=cos(t1) * ((-1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (1.0606601717798216+0.426776695296637im)*exp(-2.0*CZI*t1 + tau*2.0))
			lmix[2,2]=cos(t1) * ((0.17677669529663675+2.5606601717798214im)*exp(2.0*CZI*t1 + 2.0*(beta-tau)) + (-0.1767766952966369+0.4393398282201787im)*exp(-2.0*CZI*t1 + tau*2.0))
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
			ret[1,1]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(0.17677669529663687-0.2928932188134524im)-exp(4.0*CZI*t2)*(0.1767766952966371+1.707106781186548im))
			ret[1,2]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(-0.7071067811865475-0.4267766952966369im)+exp(4.0*CZI*t2)*(0.7071067811865475-0.07322330470336313im))
			ret[2,1]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(1.0606601717798212-0.07322330470336319im)-exp(4.0*CZI*t2)*(1.0606601717798216+0.426776695296637im))
			ret[2,2]=exp(-2.0*CZI*(t2+t1))*cos(t1)*(exp(4.0*CZI*t1)*(-0.17677669529663675-2.5606601717798214im)+exp(4.0*CZI*t2)*(0.1767766952966369-0.4393398282201787im))
			G.ret[m,n] = ret

			# less
			t1 = (n - 1)*dt
			t2 = (m - 1)*dt
			less[1,1]=cos(t1) *((0.1767766952966371+1.707106781186548im)*exp(2.0*CZI*(t2-t1))-(0.17677669529663687-0.2928932188134524im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[1,2]=cos(t1) * exp(-2.0*CZI*(t2+t1))*((-0.7071067811865475+0.07322330470336313im)*exp(4.0*CZI*t2)+(0.7071067811865475+0.4267766952966369im)*exp(4.0*CZI*t1+2.0*beta))
			less[2,1]=cos(t1) *((1.0606601717798216+0.426776695296637im)*exp(2.0*CZI*(t2-t1))+(-1.0606601717798212+0.07322330470336319im)*exp(2.0*CZI*(t1-t2)+2.0*beta))
			less[2,2]=cos(t1) * exp(-2.0*CZI*(t2+t1)) *((-0.1767766952966369+0.4393398282201787im)*exp(4.0*CZI*t2)+(0.17677669529663675+2.5606601717798214im)*exp(4.0*CZI*t1+2.0*beta))
			less = less / (1.0+exp(2.0*beta))
			G.less[n,m] = less
		end
	end
end

#
# See NESSi/libcntr/test/herm_matrix_member.cpp
#

println("Test CnFunM and related structs")

# Parameters
ntime = 101
ntau = 51
ndim1 = 2; ndim2 = 5
eps = 1.0e-7
beta = 5.0
tmax = 0.5
h = 0.01
mu = 0.0

# Contour and Green
C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
A = CnFunM(C, FERMI)

# Setup H₀
a = fill(zero(C64), ndim1, ndim1)
a[1,1] = sqrt(2.0)
a[1,2] = sqrt(2.0) * CZI
a[2,1] = sqrt(2.0) * (-CZI)
a[2,2] = -sqrt(2.0)

# Init Green A
init_green!(A, a, 0.0, beta, h)

# Setup functions
funcC = CnFunF(C)
c = fill(zero(C64), ndim1, ndim1)
for tstp = 0:ntime
    if tstp == 0
        t = 0
    else
        t = (tstp - 1) * h
    end
    c[1,1] = 2.0 * cos(t)
    c[1,2] = 0.5 * cos(t)
    c[2,1] = 0.5 * cos(t)
    c[2,2] = 3.0 * cos(t)
    funcC[tstp] = c
end

# Test Left and right multiply
# TEST 1 and TEST 2
begin
    Ar = deepcopy(A)
    Al = deepcopy(A)
    for tstp = 0:ntime
        smul!(Ar, funcC, tstp)
        smul!(funcC, Al, tstp)
    end

    # Generate exact solution
    exactR = CnFunM(C, FERMI)
    exactL = CnFunM(C, FERMI)
    exact_rightmultiply_tstp(beta, h, exactR)
    exact_leftmultiply_tstp(beta, h, exactL)

    err = 0.0
    for tstp = 0:ntime
        global err = err + distance(Ar, exactR, tstp)
    end
    @test err < eps

    err = 0.0
    for tstp = 0:ntime
        global err = err + distance(Al, exactL, tstp)
    end
    @test err < eps
end

# Test Get matsubara
# TEST 3
begin
    err = 0.0
    for t=1:ntau
        ma = A.matm[t]
        mb = A.mat[ntau-t+1]
        global err = err + sum(abs, mb + ma)
    end
    @test err < eps
end

include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

#
# See NESSi/libcntr/test/integration.cpp
#

println("Test integration weights")

# Parameters
k = 5
h = 0.1
eps = 1.0e-4

# Test PolynomialInterpolationWeights
# TEST 1
begin
    PIW = PolynomialInterpolationWeights(k)
    err = 0.0
    for i = 0:PIW.k-1
        t = (i + 0.5) * h
        finter = 0.0

        for l=0:PIW.k
            t1 = 1.0
            weight = PIW[0,l]
            for n=1:PIW.k
                t1 = t1*(i+0.5)
                weight = weight + t1 * PIW[n,l]
            end
            finter = finter + cos(h*l) * weight
        end
        global err = err + abs(cos(t) - finter)
        @test err < eps
    end
end

# Test PolynomialDifferentiationWeights
# TEST 2
begin
    PDW = PolynomialDifferentiationWeights(k)
    err = 0.0
    for i = 0:PDW.k
        df_exact = -sin(h*i)
        df_approx = 0.0
        for l = 0:PDW.k
            df_approx = df_approx + (1.0/h) * PDW[i,l] * cos(h*l)
        end
        global err = err + abs(df_exact - df_approx)
        @test err < eps
    end
end

# Test PolynomialIntegrationWeights
# TEST 3
begin
    PIW = PolynomialIntegrationWeights(k)
    err = 0.0
    for i=1:PIW.k
        for j=0:i-1
            I_approx = 0.0
            I_exact = sin(h*j) - sin(h*i)
            for l=0:PIW.k
                I_approx = I_approx + h * PIW[i,j,l] * cos(h*l)
            end
            global err = err + abs(I_exact - I_approx)
        end
    end
    @test err < eps
end

# Test BackwardDifferentiationWeights
# TEST 4
begin
    BDW = BackwardDifferentiationWeights(k)
    t1 = 0.5
    df_approx = 0.0
    for l=0:k + 1 # Here k ≠ BDW.k ≡ k + 1 
        global df_approx = df_approx + BDW[l] * cos(t1 - l*h) / h
    end
    df_exact = -sin(t1)
    err = abs(df_exact - df_approx)
    @test err < eps
end

# Test BoundaryConvolutionWeights
# TEST 5
begin
    BCW = BoundaryConvolutionWeights(k)
    ntau = 101
    beta = 5.0
    eps1 = -0.1; eps2 = 0.2
    dtau = beta / (ntau - 1)

    A = zeros(F64, ntau)
    B = zeros(F64, ntau)
    R_exact = zeros(F64, k)

    R_exact[1] = 0.0137659
    R_exact[2] = 0.0274638
    R_exact[3] = 0.0410948
    R_exact[4] = 0.0546598

    for m = 1:ntau
        A[m] = -fermi(beta, (m-1)*dtau, -eps1)
        B[m] = -fermi(beta, (m-1)*dtau, -eps2)
    end

    err = 0.0
    for m = 1:k-1
        R_approx = 0.0
        for j = 0:k
            for l = 0:k
                R_approx = R_approx + dtau * BCW[m-1,j,l] * A[j + 1] * B[j + 1]
            end
        end
        global err = err + abs(R_approx - R_exact[m])
    end
    @test err < eps
end

# Test GregoryIntegrationWeights
# TEST 6
begin
    tmax = 2.5*pi
    nt = 100
    h = tmax / nt

    fn = [cos(h*i) for i = 0:nt]
    #fn = [exp(h*i*im) for i = 0:nt]

    GIW = GregoryIntegrationWeights(k)
    err = 0.0
    for i = k+1:nt
        exact = sin(h*i)
        #exact = -im*( exp(h*i*im) - 1.0 )
        approx = 0.0
        for j = 0:i
            approx = approx + GIW[i,j] * fn[j+1]
        end
        global err = err + abs(exact - approx*h)
    end
    @test err < eps
end

include("../src/KadanoffBaym.jl")
using .KadanoffBaym
using Test

# Parameters
ntime = 8
ntau = 101
beta = 5.0
h = 0.01
wa = 1.123
wb = 0.345
k = 5
mu = 0.0
tmax = 0.08
eps = 1.0e-6
ndim1 = 1; ndim2 = 1

# Contour and Green's functions
C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
A = CnFunM(C, BOSE)
B = CnFunM(C, BOSE)
AB = CnFunM(C, BOSE)

# H₀
eps_a = fill(zero(C64), ndim1, ndim1)
eps_b = fill(zero(C64), ndim1, ndim1)
eps_a[1,1] = wa
eps_b[1,1] = wb

# Generate A and B
init_green!(A, eps_a, mu, beta, h)
init_green!(B, eps_b, mu, beta, h)

#for t = 1:ntau
#    @show t, A.mat[t], B.mat[t]
#end

#m = 290
I = Integrator(k)
#c_mat_mat_2(m, AB.mat, A.mat, B.mat, I, A.sign)

for m = 1:ntau
    c_mat_mat_2(m, AB.mat, A.mat, B.mat, I, A.sign)
    @show m, AB.mat[m]
end