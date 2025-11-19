
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

