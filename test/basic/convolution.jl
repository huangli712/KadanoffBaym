haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using KadanoffBaym

# Parameters
ntime = 15
ntau = 21
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
C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
A = ℱ(C, BOSE)
B = ℱ(C, BOSE)
AB = ℱ(C, BOSE)

# H₀
eps_a = fill(zero(C64), ndim1, ndim2)
eps_b = fill(zero(C64), ndim1, ndim2)
eps_a[1,1] = wa
eps_b[1,1] = wb

# Generate A and B
init_green!(A, eps_a, mu, beta, h)
init_green!(B, eps_b, mu, beta, h)

I = Integrator(k)

#for m = 1:ntau
#    conv_mat_mat_1(m, AB.mat, A.mat, B.mat, I, A.sign)
#    #conv_mat_mat_1p(m, AB.mat, A.mat, B.mat, I)
#    #conv_mat_mat_2(m, AB.mat, A.mat, B.mat, I, A.sign)
#    #conv_mat_mat_2p(m, AB.mat, A.mat, B.mat, I)
#    @show m, AB.mat[m]
#end

#conv_mat(AB.mat, A.mat, B.mat, I, beta, A.sign)
#for m = 1:ntau
#    @show m, AB.mat[m]
#end

#for t = 1:ntime
#    conv_ret(t, AB.ret, A.ret, A.ret, B.ret, B.ret, I, h)
#    for m = 1:t
#        @show t, m, AB.ret[t,m]
#    end
#    println()
#end

#for n = 1:ntime
#    conv_ret_lmix(n, AB.lmix, A.ret, A.ret, B.lmix, B.lmix, I, h)
#    for m = 1:ntau
#        @show n, m, AB.lmix[n,m]
#    end
#    println()
#end

#for n = 1:ntime
#    conv_lmix_mat(n, AB.lmix, A.lmix, B.mat, I, beta, A.sign)
#    for m = 1:ntau
#        @show n, m, AB.lmix[n,m]
#    end
#    println()
#end

#for n = 1:ntime
#    conv_ret_lmix(n, AB.lmix, A.ret, A.ret, B.lmix, B.lmix, I, h)
#    conv_lmix_mat(n, AB.lmix, A.lmix, B.mat, I, beta, A.sign)
#    for m = 1:ntau
#        @show n, m, AB.lmix[n,m]
#    end
#    println()
#end

#for n = 1:ntime
#    conv_lmix_mat(n, AB.lmix, A.lmix, B.mat, I, beta, A.sign)
#    conv_ret_lmix(n, AB.lmix, A.ret, A.ret, B.lmix, B.lmix, I, h)
#    for m = 1:ntau
#        @show n, m, AB.lmix[n,m]
#    end
#    println()
#end

#for n = 1:ntime
#    conv_ret_less(n, AB.less, A.ret, B.less, I, h)
#    println()
#end

#for n = 1:ntime
#    conv_less_adv(n, AB.less, A.less, B.ret, I, h)
#    println()
#end

for n = 1:ntime
    conv_lmix_rmix(n, AB.less, A.lmix, A.lmix, B.lmix, B.lmix, I, beta, AB.sign)
    for m = 1:n
        @show m, n, AB.less[m,n]
    end
    println()
end
