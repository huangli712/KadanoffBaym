
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

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


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
