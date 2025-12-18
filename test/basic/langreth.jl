haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using KadanoffBaym

# Parameters
ntime = 8
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

for m = 1:ntau
    #conv_mat_mat_1(m, AB.mat, A.mat, B.mat, I, A.sign)
    conv_mat_mat_1(m, AB.mat, A.mat, B.mat, I, A.sign)
    #conv_mat_mat_1p(m, AB.mat, A.mat, B.mat, I)
    @show m, AB.mat[m]
end

#conv_mat(AB.mat, A.mat, B.mat, I, beta, A.sign)
#@show AB.mat

#for t = 1:ntime
#    conv_tstp_ret(t, AB.ret, A.ret, A.ret, B.ret, B.ret, I, h)
#end
