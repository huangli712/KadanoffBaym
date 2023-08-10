include("../src/KadanoffBaym.jl")

using .KadanoffBaym

#println("hello world!")
#C = Cn(2, 5.0, 10.0)
#G = CnFun(C)

#G.mat[2] = [0+0.1im 1.0-0.2im; 0.0 + 0.0im 0.0+0.3im]
#@show G.mat[2]
#G.mat = 3 * G.mat # (3.0 + 0.0im)
#@show G.mat[2]
#zeros!(G.mat)
#@show G.mat[2]
#reset!(G.mat, 1.0)
#@show G.mat[2]

#M = copy(G.mat)
#@show M[2]
#@show M.ntau
#M.ntau = 10
#@show M.ntau
#@show G.mat.ntau

#k = 5
#Wi = calc_poly_interpolation(k)
#
#for i=0:k
#    for j=0:k
#        println("i: $i  j: $j  W: ", Wi[i+1,j+1])
#    end
#end
#

#Wd = calc_poly_differentiation(k, Wi)
#
#for i=0:k
#    for j=0:k
#        println("i: $i  j: $j  W: ", Wd[i+1,j+1])
#    end
#end
#

#Wt = calc_poly_integration(k, Wi)
#
#for i=0:k
#    for j=0:k
#        for l=0:k
#            println("i: $i  j: $j  l: $l  W: ", Wt[i+1,j+1,l+1])
#        end
#    end
#end
#

#Wb = calc_backward_differentiation(k, Wi)
#
#for i=0:k
#    println("i: $i  W: ", Wb[i+1])
#end
#

#Wstart = calc_gregory_start(5, Wt)

#Wc = calc_boundary_convolution(k, Wi)
#
#for m=1:k-1
#    for i=0:k
#        for j=0:k
#            println("m: $m i: $i j: $j W: ", Wc[m,i+1,j+1])
#        end
#    end
#end
#

#
#k = 1
#
#PIW = PolynomialInterpolationWeights(k)
#for i=0:PIW.k
#    for j=0:PIW.k
#        println("i: $i j: $j W: ", PIW[i,j])
#    end
#end


#
#PDW = PolynomialDifferentiationWeights(k)
#for i=0:PDW.k
#    for j=0:PDW.k
#        println("i: $i j: $j W: ", PDW[i,j])
#    end
#end
#

#
#PIW = PolynomialIntegrationWeights(k)
#for i=0:PIW.k
#    for j=0:PIW.k
#        for l=0:PIW.k
#            println("i: $i j: $j l: $l W: ", PIW[i,j,l])
#        end
#    end
#end
#

#
#BDW = BackwardDifferentiationWeights(k)
#for i=0:BDW.k
#    println("i: $i W: ", BDW[i])
#end
#

#
#BCW = BoundaryConvolutionWeights(k)
#for m=0:BCW.k-2
#    for i=0:BCW.k
#        for j=0:BCW.k
#            println("m: $m i: $i j: $j W: ", BCW[m,i,j])
#        end
#    end
#end
#

#
#Wi = calc_poly_interpolation(k)
#Wt = calc_poly_integration(k, Wi)
#Ws = calc_gregory_start(k, Wt)
#BDW = BackwardDifferentiationWeights(k)
#for i=0:k
#    for j=0:k
#        println("i: $i j: $j W: ", Ws[i+1,j+1])
#    end
#end
#
#for i=0:BDW.k
#    println("i: $i W: ", BDW[i])
#end
#

#for k = 0:16
#    @show k, laplace(k), float(laplace(k))
#end

#for k = 0:6
#    B = γⱼ(k)
#    @show k, B
#end

#
#k = 5
#W = GregoryIntegrationWeights(k)
#@show W.σ
#@show W.Σ
#@show W.ω
#

#
#k = 1
#GIW = GregoryIntegrationWeights(k)
##@show GIW.σ
#@show GIW.Σ
#@show GIW.ω
#
#for n = 0:3*k + 4
#    for j = 0:n
#        @show n, j, GIW[n,j]
#    end
#end
#

#tmax = 2.5*pi
#nt = 100
#h = tmax / nt
#
##fn = [cos(h*i) for i = 0:nt]
#fn = [exp(h*i*im) for i = 0:nt]
#
#k = 5
#GIW = GregoryIntegrationWeights(k)
#for i = k+1:nt
#    #exact = sin(h*i)
#    exact = -im*( exp(h*i*im) - 1.0 )
#    approx = 0.0
#    for j = 0:i
#        approx = approx + GIW[i,j] * fn[j+1]
#    end
#    @show i*h/pi, abs(exact - approx*h)
#end
#

println("Hello World!")

ntime = 101
ntau = 101
ndim = 2
beta = 5.0
tmax = 5.0
h = 0.01
mu = 0.0
#
C₁ = Cn(ntime, ntau, 1, 1, tmax, beta)
cfm₁ = CnFunM(C₁)
H₁ = fill(CONE, C₁.ndim1, C₁.ndim2)
init_green!(cfm₁, H₁, mu, beta, h)

C₂ = Cn(ntime, ntau, ndim, ndim, tmax, beta)
cfm₂ = CnFunM(C₂, BOSE)
H₂ = zeros(C64, ndim, ndim)
H₂[1,1] = sqrt(2.0)
H₂[1,2] = sqrt(2.0) * CZI
H₂[2,1] = sqrt(2.0) * (-CZI)
H₂[2,2] = -sqrt(2.0)
init_green!(cfm₂, H₂, mu, beta, h)

#for i = 1:ntau
#    @show i, cfm₁.mat[i]
#end

#for i = 1:ntime
#    for j = 1:ntime
#        @show i, j, cfm₁.less[i,j]
#        @show i, j, cfm₁.ret[i,j]
#        @show i, j, cfm₁.gtr[i,j], cfm₁.less[i,j] + cfm₁.ret[i,j]
#        @show i, j, cfm₁.gtr[i,j]
#    end
#end
#@show cfm₁.gtr[1,2]
#@show cfm₁.gtr.dataL[][1,2], cfm₁.gtr.dataR[][1,2]
#@show cfm₁.gtr.dataL[][1,2] + cfm₁.gtr.dataR[][1,2]
#@show cfm₁.less[1,2], cfm₁.ret[1,2]
#@show cfm₁.less[1,2] + cfm₁.ret[1,2]

#for i = 1:ntime
#    for m = 1:ntau
#        #@show i, m, cfm₁.lmix[i,m]
#        @show m, i, cfm₁.rmix[m,i]
#    end
#end

#@show cfm₁.sign, cfm₁.matm.sign, cfm₁.rmix.sign
#@show cfm₂.sign, cfm₂.matm.sign, cfm₂.rmix.sign

#for i = 1:ntau
#    @show i, cfm₂.mat[i]
#end

#for i = 1:ntime
#    for j = 1:ntime
#        @show i, j, cfm₂.less[i,j]
#        @show i, j, cfm₂.ret[i,j]
#        @show i, j, cfm₂.gtr[i,j]
#    end
#end

#for i = 1:ntime
#    for m = 1:ntau
#        #@show i, m, cfm₂.lmix[i,m]
#        #@show m, i, cfm₂.rmix[m,i]
#    end
#end


include("../src/KadanoffBaym.jl")

using .KadanoffBaym

C = Cn(5.0, 10.0)
x = zeros(C64, C.ndim1, C.ndim2)
tstp = 20

#
#cff = CnFunF(C)
#cff =CnFunF(C.ntime, x)
#cff = CnFunF(C, x)
#println(cff[1] == cff[2])
#println(cff[1] === cff[2])
#println(cff[1])
#println(cff[2])
#cff[1] = 2.0im
#println(cff[1])
#println(cff[2])
#

#-------------------------------
# For CnMatM and CnMatV
#-------------------------------

#
#mat = CnMatM(C)
#mat = CnMatM(C.ntau, x)
#mat = CnMatM(C, x)
#println(mat[1] == mat[2])
#println(mat[1] === mat[2])
#println(mat[1])
#println(mat[2])
#mat[1] = 2.0im
#println(mat[1])
#println(mat[2])
#

#
#mat = CnMatV(C.ntau, C.ndim1)
#mat = CnMatV(C.ntau, x)
#println(mat[1] == mat[2])
#println(mat[1] === mat[2])
#println(mat[1])
#println(mat[2])
#mat[1] = 2.0im
#println(mat[1])
#println(mat[2])
#

#
#mat1 = CnMatM(C)
#mat2 = CnMatM(C.ntau, x)
#mat1[1] = 2.0im
#println(mat1.data == mat2.data)
#println(mat1.data === mat2.data)
#println(mat1[1], mat1[2])
#println(mat2[1], mat2[2])
#memcpy!(mat1, mat2)
#println(mat1.data == mat2.data)
#println(mat1.data === mat2.data)
#println(mat1[1] === mat2[1])
#println(mat1[2] === mat2[2])
#println(mat1[1], mat1[2])
#println(mat2[1], mat2[2])
#

#
#mat1 = CnMatM(C)
#mat2 = CnMatV(C.ntau, x)
#mat1[1] = 2.0im
#println(mat1.data == mat2.data)
#println(mat1.data === mat2.data)
#println(mat1[1], mat1[2])
#println(mat2[1], mat2[2])
#memcpy!(mat2, mat1)
#println(mat1.data == mat2.data)
#println(mat1.data === mat2.data)
#println(mat1[1] === mat2[1])
#println(mat1[2] === mat2[2])
#println(mat1[1], mat1[2])
#println(mat2[1], mat2[2])
#

#-------------------------------
# For CnRetM and CnRetV
#-------------------------------

#
#ret = CnRetM(C)
#ret = CnRetM(C.ntime, x)
#ret = CnRetM(C, x)
#println(ret[10,1] == ret[10,2])
#println(ret[10,1] === ret[10,2])
#println(ret[10,1])
#println(ret[10,2])
#ret[10,1] = 2.0im
#println(ret[10,1])
#println(ret[10,2])
#

#
#ret = CnRetV(tstp, C.ndim1)
#ret = CnRetV(tstp, x)
#println(ret[10] == ret[8])
#println(ret[10] === ret[8])
#println(ret[10])
#println(ret[8])
#ret[10] = 2.0im
#println(ret[10])
#println(ret[8])
#

#
#ret1 = CnRetM(C)
#ret2 = CnRetM(C.ntime, x)
#ret1[10,1] = 2.0im
#println(ret1.data == ret2.data)
#println(ret1.data === ret2.data)
#println(ret1[10,1],ret1[10,2])
#println(ret2[10,1],ret2[10,2])
#memcpy!(ret1, ret2)
#println(ret1.data == ret2.data)
#println(ret1.data === ret2.data)
#println(ret1[10,1] === ret2[10,1])
#println(ret1[10,2] === ret2[10,2])
#println(ret1[10,1],ret1[10,2])
#println(ret2[10,1],ret2[10,2])
#

#
#ret1 = CnRetM(C)
#ret2 = CnRetV(tstp, x)
#ret1[tstp,1] = 2.0im
#ret1[tstp,2] = -0.36-0.12im
#println(ret1.data == ret2.data)
#println(ret1.data === ret2.data)
#println(ret1[tstp,1],ret1[tstp,2])
#println(ret2[1],ret2[2])
#memcpy!(ret2, ret1)
#println(ret1.data == ret2.data)
#println(ret1.data === ret2.data)
#println(ret1[tstp,1] === ret2[1])
#println(ret1[tstp,2] === ret2[2])
#println(ret1[tstp,1],ret1[tstp,2])
#println(ret2[1],ret2[2])
#

#-------------------------------
# For CnLmixM and CnLmixV
#-------------------------------

#
#lmix = CnLmixM(C)
#lmix = CnLmixM(C.ntime, C.ntau, x)
#lmix = CnLmixM(C, x)
#println(lmix[10,1] == lmix[10,2])
#println(lmix[10,1] === lmix[10,2])
#println(lmix[10,1])
#println(lmix[10,2])
#lmix[10,1] = 2.0im
#println(lmix[10,1])
#println(lmix[10,2])
#

#
#lmix = CnLmixV(C.ntau, C.ndim1)
#lmix = CnLmixV(C.ntau, x)
#println(lmix[10] == lmix[12])
#println(lmix[10] === lmix[12])
#println(lmix[10])
#println(lmix[12])
#lmix[10] = 2.0im
#println(lmix[10])
#println(lmix[12])
#

#
#lmix1 = CnLmixM(C)
#lmix2 = CnLmixM(C.ntime, C.ntau, x)
#lmix1[10,1] = 2.0im
#lmix1[10,2] = 2.13 - 3.4im
#println(lmix1.data == lmix2.data)
#println(lmix1.data === lmix2.data)
#println(lmix1[10,1],lmix1[10,2])
#println(lmix2[10,1],lmix2[10,2])
#memcpy!(lmix1, lmix2)
#println(lmix1.data == lmix2.data)
#println(lmix1.data === lmix2.data)
#println(lmix1[10,1] === lmix2[10,1])
#println(lmix1[10,2] === lmix2[10,2])
#println(lmix1[10,1],lmix1[10,2])
#println(lmix2[10,1],lmix2[10,2])
#

#
#lmix1 = CnLmixM(C)
#lmix2 = CnLmixV(C.ntau, x)
#lmix1[tstp,1] = 2.0im
#lmix1[tstp,2] = 2.13 - 3.4im
#println(lmix1.data == lmix2.data)
#println(lmix1.data === lmix2.data)
#println(lmix1[tstp,1],lmix1[tstp,2])
#println(lmix2[1],lmix2[2])
#memcpy!(lmix2, lmix1, tstp)
#println(lmix1.data == lmix2.data)
#println(lmix1.data === lmix2.data)
#println(lmix1[tstp,1] === lmix2[1])
#println(lmix1[tstp,2] === lmix2[2])
#println(lmix1[tstp,1],lmix1[tstp,2])
#println(lmix2[1],lmix2[2])
#

#-------------------------------
# For CnLessM and CnLessV
#-------------------------------

#
#less = CnLessM(C)
#less = CnLessM(C.ntime, x)
#less = CnLessM(C, x)
#println(less[1,10] == less[2,10])
#println(less[1,10] === less[2,10])
#println(less[1,10])
#println(less[2,10])
#less[1,10] = 2.0im
#println(less[1,10])
#println(less[2,10])
#

#
#less = CnLessV(tstp, C.ndim1)
#less = CnLessV(tstp, x)
#println(less[10] == less[8])
#println(less[10] === less[8])
#println(less[10])
#println(less[8])
#less[10] = 2.0im
#println(less[10])
#println(less[8])
#

#
#less1 = CnLessM(C)
#less2 = CnLessM(C.ntime, x)
#less1[1,10] = -1.7 + 2.0im
#less1[2,10] = 2.13 - 0.69im
#println(less1.data == less2.data)
#println(less1.data === less2.data)
#println(less1[1,10],less1[2,10])
#println(less2[1,10],less2[2,10])
#memcpy!(less1, less2)
#println(less1.data == less2.data)
#println(less1.data === less2.data)
#println(less1[1,10] === less2[1,10])
#println(less1[2,10] === less2[2,10])
#println(less1[1,10],less1[2,10])
#println(less2[1,10],less2[2,10])
#

#
#less1 = CnLessM(C)
#less2 = CnLessV(tstp, x)
#less1[1,tstp] = -1.7 + 2.0im
#less1[2,tstp] = 2.13 - 0.69im
#println(less1.data == less2.data)
#println(less1.data === less2.data)
#println(less1[1,tstp],less1[2,tstp])
#println(less2[1],less2[2])
#memcpy!(less2, less1)
#println(less1.data == less2.data)
#println(less1.data === less2.data)
#println(less1[1,tstp] === less2[1])
#println(less1[2,tstp] === less2[2])
#println(less1[1,tstp],less1[2,tstp])
#println(less2[1],less2[2])
#


include("../src/KadanoffBaym.jl")

using .KadanoffBaym

C = Cn(5.0, 10.0)
x = zeros(C64, C.ndim1, C.ndim2)
tstp = 20

#
#cfm = CnFunM(C)
#cfm.mat[10] = 2.0im
#cfm.ret[tstp, 2] = -3.13 + 2.7im
#cfm.lmix[tstp, 32] = 999.1 - 888.9im
#cfm.less[12, tstp] = -0.008 + 1.111im
#
#cfv = cfm[tstp]
#println("check cfv.mat")
#@show typeof(cfv.mat)
#println(cfv.mat[10], cfv.mat[9], cfv.mat[8])
#
#println("check cfv.ret")
#@show typeof(cfv.ret)
#println(cfv.ret[1], cfv.ret[2], cfv.ret[3])
#
#println("check cfv.lmix")
#@show typeof(cfv.lmix)
#println(cfv.lmix[31], cfv.lmix[32], cfv.lmix[33])
#
#println("check cfv.less")
#@show typeof(cfv.less)
#println(cfv.less[11], cfv.less[12], cfv.less[13])
#

cfm = CnFunM(C)
cfv = CnFunV(C, tstp)
cfv.mat[10] = 2.0im
cfv.ret[8] = -1.0+2.0im
cfv.lmix[80] = 0.33 - 0.45im
cfv.less[9] = 4.5 + 0.23im

cfm[tstp] = cfv
println(cfm.mat[10], cfm.mat[9], cfm.mat[11])
println(cfm.ret[tstp,8], cfm.ret[tstp,9], cfm.ret[tstp,7])
println(cfm.lmix[tstp,80], cfm.lmix[tstp,79], cfm.lmix[tstp,81])
println(cfm.less[9,tstp], cfm.less[8,tstp], cfm.less[10,tstp])
