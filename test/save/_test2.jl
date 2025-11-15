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
