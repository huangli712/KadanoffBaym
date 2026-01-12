#
# t_convolution.jl
#
# To test the convolution operations for contour-ordered Green's functions.
#

#=
@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 15
    ntau = 101
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
    # Prepare exact solution
    fac = 1.0 / (wa - wb)
    G₃ = deepcopy(G₁)
    for tstp = 0:ntime
        incr!(G₃, G₂, tstp, -1.0)
        smul!(G₃, tstp, fac)
    end
    #
    # Evaluate convolution
    convolution(G₄, G₁, G₂, order, beta, δt)
    #
    # Compute final error
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄, G₃, tstp)
    end
    @test err < ϵ
end
=#

#=
@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 15
    ntau = 101
    ndim1 = 1
    ndim2 = 1
    tmax = 2.0
    beta = 0.1
    #
    δt = 0.01
    μ = 0.0
    order = 5
    ϵ = 1.0e-6
    wa = 1.123
    wb = 0.345
    #
    for n = 1:7
        ntime = 2^(n-1) * 10 + 1
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
        init_green!(G₁, H₁, μ, beta, C.dt)
        init_green!(G₂, H₂, μ, beta, C.dt)
        #
        I = Integrator(order)
        #
        # Prepare exact solution
        fac = 1.0 / (wa - wb)
        G₃ = deepcopy(G₁)
        for tstp = 0:ntime
            incr!(G₃, G₂, tstp, -1.0)
            smul!(G₃, tstp, fac)
        end
        #
        # Evaluate convolution
        convolution(G₄, G₁, G₂, order, beta, C.dt)
        #
        # Compute final error
        err = 0.0
        for tstp = 0:ntime
            err = err + distance(G₄, G₃, tstp)
        end
        if n ≥ 4
            @test err < ϵ
        end
    end
end
=#

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 15
    ntau = 501
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
    G₁ = ℱ(C, FERMI)
    G₂ = ℱ(C, FERMI)
    G₃ = ℱ(C, FERMI)
    G₄ = ℱ(C, FERMI)
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
    # Prepare exact solution
    fac = 1.0 / (wa - wb)
    G₃ = deepcopy(G₁)
    for tstp = 0:ntime
        incr!(G₃, G₂, tstp, -1.0)
        smul!(G₃, tstp, fac)
    end
    #
    # Evaluate convolution
    convolution(G₄, G₁, G₂, order, beta, δt)
    #
    # Compute final error
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄, G₃, tstp)
        @show tstp, err
    end
    @test err < ϵ
end
