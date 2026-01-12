#
# t_convolution.jl
#
# To test the convolution operations for contour-ordered Green's functions.
#

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 15
    ntau = 21
    ndim1 = 1
    ndim2 = 1
    tmax = 0.08
    beta = 5.0
    #
    δt = 0.01
    μ = 0.0
    order = 5
    ϵ = 1.0e-6
    wa = 1.123
    wb = 0.345
    #
    C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    G₁ = ℱ(C, BOSE)
    G₂ = ℱ(C, BOSE)
    G₃ = ℱ(C, BOSE)
    G₄ = ℱ(C, BOSE)
    #
    H₁ = fill(zero(C64), ndim1, ndim2)
    H₂ = fill(zero(C64), ndim1, ndim2)
    H₁[1,1] = wa
    H₂[1,1] = wb
    #
    init_green!(G₁, H₁, μ, beta, δt)
    init_green!(G₂, H₂, μ, beta, δt)
    #
    I = Integrator(order)
    #
    fac = 1.0 / (wa - wb)
    G₃ = deepcopy(G₁)
    for tstp = 0:ntime
        incr!(G₃, G₂, tstp, -1.0)
        smul!(G₃, tstp, fac)
    end
    #
    convolution(G₄, G₁, G₂, order, beta, δt)
    #for m = 1:ntau
    #    @show m, G₃.mat[m], G₄.mat[m]
    #end
    #
    err = 0.0
    err1 = 0.0
    for tstp = 0:ntime
        err1 = distance(G₄, G₃, tstp)
        err = err + err1
        @show tstp, err, err1
    end
    #@show err
    @test err < 1.0e-4
end
