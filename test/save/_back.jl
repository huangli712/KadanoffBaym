
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
