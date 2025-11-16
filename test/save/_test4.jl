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
