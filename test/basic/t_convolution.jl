#
# t_convolution.jl
#
# To test the convolution operations for contour-ordered Green's functions.
#

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

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 15
    ntau = 501
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
    end
    @test err < ϵ
end

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 11
    ntau = 401
    ndim1 = 2
    ndim2 = 2
    tmax = 0.2
    beta = 0.1
    #
    δt = 0.02
    μ = 0.0
    order = 5
    ϵ = 1.0e-6
    #
    C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    CS = Cn(ntime, ntau, 1, 1, tmax, beta)
    #
    G₁ = ℱ(C, BOSE)
    G₂ = ℱ(C, BOSE)
    G₃ = ℱ(C, BOSE)
    G₄ = ℱ(C, BOSE)
    #
    G₁_₁₁ = ℱ(CS, BOSE)
    G₁_₁₂ = ℱ(CS, BOSE)
    G₁_₂₁ = ℱ(CS, BOSE)
    G₁_₂₂ = ℱ(CS, BOSE)
    #
    G₂_₁₁ = ℱ(CS, BOSE)
    G₂_₁₂ = ℱ(CS, BOSE)
    G₂_₂₁ = ℱ(CS, BOSE)
    G₂_₂₂ = ℱ(CS, BOSE)
    #
    G₃_₁₁ = ℱ(CS, BOSE)
    G₃_₁₂ = ℱ(CS, BOSE)
    G₃_₂₁ = ℱ(CS, BOSE)
    G₃_₂₂ = ℱ(CS, BOSE)
    G₃ₜₘₚ = ℱ(CS, BOSE)
    #
    G₄_₁₁ = ℱ(CS, BOSE)
    G₄_₁₂ = ℱ(CS, BOSE)
    G₄_₂₁ = ℱ(CS, BOSE)
    G₄_₂₂ = ℱ(CS, BOSE)
    #
    H₁ = fill(zero(C64), ndim1, ndim2)
    H₂ = fill(zero(C64), ndim1, ndim2)
    H₁[1,1] = 1.123
    H₁[1,2] = 0.1
    H₁[2,1] = 0.1
    H₁[2,2] = 0.567
    H₂[1,1] = 0.345
    H₂[1,2] = 0.2
    H₂[2,1] = 0.2
    H₂[2,2] = 0.876
    #
    init_green!(G₁, H₁, μ, beta, δt)
    init_green!(G₂, H₂, μ, beta, δt)
    #
    cz11 = CopyZone(1,1)
    cz12 = CopyZone(1,2)
    cz21 = CopyZone(2,1)
    cz22 = CopyZone(2,2)
    czd  = CopyZone(1,1)
    elemcpy!(cz11, G₁, czd, G₁_₁₁)
    elemcpy!(cz12, G₁, czd, G₁_₁₂)
    elemcpy!(cz21, G₁, czd, G₁_₂₁)
    elemcpy!(cz22, G₁, czd, G₁_₂₂)
    elemcpy!(cz11, G₂, czd, G₂_₁₁)
    elemcpy!(cz12, G₂, czd, G₂_₁₂)
    elemcpy!(cz21, G₂, czd, G₂_₂₁)
    elemcpy!(cz22, G₂, czd, G₂_₂₂)
    #
    I = Integrator(order)
    #
    convolution(G₃_₁₁, G₁_₁₁, G₂_₁₁, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₁ , order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₁₁, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₁₂, G₁_₁₁, G₂_₁₂, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₂, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₁₂, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₂₁, G₁_₂₁, G₂_₁₁, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₁, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₂₁, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₂₂, G₁_₂₁, G₂_₁₂, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₂, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₂₂, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    elemcpy!(czd, G₃_₁₁, cz11, G₃)
    elemcpy!(czd, G₃_₁₂, cz12, G₃)
    elemcpy!(czd, G₃_₂₁, cz21, G₃)
    elemcpy!(czd, G₃_₂₂, cz22, G₃)
    #
    convolution(G₄, G₁, G₂, order, beta, δt)
    elemcpy!(cz11, G₄, czd, G₄_₁₁)
    elemcpy!(cz12, G₄, czd, G₄_₁₂)
    elemcpy!(cz21, G₄, czd, G₄_₂₁)
    elemcpy!(cz22, G₄, czd, G₄_₂₂)
    #
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄, G₃, tstp)
    end
    @test err < ϵ
    #
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄_₁₁, G₃_₁₁, tstp)
        err = err + distance(G₄_₁₂, G₃_₁₂, tstp)
        err = err + distance(G₄_₂₁, G₃_₂₁, tstp)
        err = err + distance(G₄_₂₂, G₃_₂₂, tstp)
    end
    @test err < ϵ
end

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 11
    ntau = 401
    ndim1 = 2
    ndim2 = 2
    tmax = 2.0
    beta = 0.1
    #
    δt = 0.02
    μ = 0.0
    order = 5
    ϵ = 1.0e-6
    #
    for i = 1:4
        ntime = 2^(i-1) * 10 + 1
        C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
        CS = Cn(ntime, ntau, 1, 1, tmax, beta)
        #
        G₁ = ℱ(C, BOSE)
        G₂ = ℱ(C, BOSE)
        G₃ = ℱ(C, BOSE)
        G₄ = ℱ(C, BOSE)
        #
        G₁_₁₁ = ℱ(CS, BOSE)
        G₁_₁₂ = ℱ(CS, BOSE)
        G₁_₂₁ = ℱ(CS, BOSE)
        G₁_₂₂ = ℱ(CS, BOSE)
        #
        G₂_₁₁ = ℱ(CS, BOSE)
        G₂_₁₂ = ℱ(CS, BOSE)
        G₂_₂₁ = ℱ(CS, BOSE)
        G₂_₂₂ = ℱ(CS, BOSE)
        #
        G₃_₁₁ = ℱ(CS, BOSE)
        G₃_₁₂ = ℱ(CS, BOSE)
        G₃_₂₁ = ℱ(CS, BOSE)
        G₃_₂₂ = ℱ(CS, BOSE)
        G₃ₜₘₚ = ℱ(CS, BOSE)
        #
        G₄_₁₁ = ℱ(CS, BOSE)
        G₄_₁₂ = ℱ(CS, BOSE)
        G₄_₂₁ = ℱ(CS, BOSE)
        G₄_₂₂ = ℱ(CS, BOSE)
        #
        H₁ = fill(zero(C64), ndim1, ndim2)
        H₂ = fill(zero(C64), ndim1, ndim2)
        H₁[1,1] = 1.123
        H₁[1,2] = 0.1
        H₁[2,1] = 0.1
        H₁[2,2] = 0.567
        H₂[1,1] = 0.345
        H₂[1,2] = 0.2
        H₂[2,1] = 0.2
        H₂[2,2] = 0.876
        #
        init_green!(G₁, H₁, μ, beta, C.dt)
        init_green!(G₂, H₂, μ, beta, C.dt)
        #
        cz11 = CopyZone(1,1)
        cz12 = CopyZone(1,2)
        cz21 = CopyZone(2,1)
        cz22 = CopyZone(2,2)
        czd  = CopyZone(1,1)
        elemcpy!(cz11, G₁, czd, G₁_₁₁)
        elemcpy!(cz12, G₁, czd, G₁_₁₂)
        elemcpy!(cz21, G₁, czd, G₁_₂₁)
        elemcpy!(cz22, G₁, czd, G₁_₂₂)
        elemcpy!(cz11, G₂, czd, G₂_₁₁)
        elemcpy!(cz12, G₂, czd, G₂_₁₂)
        elemcpy!(cz21, G₂, czd, G₂_₂₁)
        elemcpy!(cz22, G₂, czd, G₂_₂₂)
        #
        I = Integrator(order)
        #
        convolution(G₃_₁₁, G₁_₁₁, G₂_₁₁, order, beta, CS.dt)
        convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₁, order, beta, CS.dt)
        for tstp = 0:ntime
            incr!(G₃_₁₁, G₃ₜₘₚ, tstp, 1.0)
        end
        zeros!(G₃ₜₘₚ)
        #
        convolution(G₃_₁₂, G₁_₁₁, G₂_₁₂, order, beta, CS.dt)
        convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₂, order, beta, CS.dt)
        for tstp = 0:ntime
            incr!(G₃_₁₂, G₃ₜₘₚ, tstp, 1.0)
        end
        zeros!(G₃ₜₘₚ)
        #
        convolution(G₃_₂₁, G₁_₂₁, G₂_₁₁, order, beta, CS.dt)
        convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₁, order, beta, CS.dt)
        for tstp = 0:ntime
            incr!(G₃_₂₁, G₃ₜₘₚ, tstp, 1.0)
        end
        zeros!(G₃ₜₘₚ)
        #
        convolution(G₃_₂₂, G₁_₂₁, G₂_₁₂, order, beta, CS.dt)
        convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₂, order, beta, CS.dt)
        for tstp = 0:ntime
            incr!(G₃_₂₂, G₃ₜₘₚ, tstp, 1.0)
        end
        zeros!(G₃ₜₘₚ)
        #
        elemcpy!(czd, G₃_₁₁, cz11, G₃)
        elemcpy!(czd, G₃_₁₂, cz12, G₃)
        elemcpy!(czd, G₃_₂₁, cz21, G₃)
        elemcpy!(czd, G₃_₂₂, cz22, G₃)
        #
        convolution(G₄, G₁, G₂, order, beta, C.dt)
        elemcpy!(cz11, G₄, czd, G₄_₁₁)
        elemcpy!(cz12, G₄, czd, G₄_₁₂)
        elemcpy!(cz21, G₄, czd, G₄_₂₁)
        elemcpy!(cz22, G₄, czd, G₄_₂₂)
        #
        err = 0.0
        for tstp = 0:ntime
            err = err + distance(G₄, G₃, tstp)
        end
        @test err < ϵ
        #
        err = 0.0
        for tstp = 0:ntime
            err = err + distance(G₄_₁₁, G₃_₁₁, tstp)
            err = err + distance(G₄_₁₂, G₃_₁₂, tstp)
            err = err + distance(G₄_₂₁, G₃_₂₁, tstp)
            err = err + distance(G₄_₂₂, G₃_₂₂, tstp)
        end
        @test err < ϵ
    end
end

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 11
    ntau = 401
    ndim1 = 2
    ndim2 = 2
    tmax = 0.2
    beta = 0.1
    #
    δt = 0.02
    μ = 0.0
    order = 5
    ϵ = 1.0e-6
    #
    C = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    CS = Cn(ntime, ntau, 1, 1, tmax, beta)
    #
    G₁ = ℱ(C, FERMI)
    G₂ = ℱ(C, FERMI)
    G₃ = ℱ(C, FERMI)
    G₄ = ℱ(C, FERMI)
    #
    G₁_₁₁ = ℱ(CS, FERMI)
    G₁_₁₂ = ℱ(CS, FERMI)
    G₁_₂₁ = ℱ(CS, FERMI)
    G₁_₂₂ = ℱ(CS, FERMI)
    #
    G₂_₁₁ = ℱ(CS, FERMI)
    G₂_₁₂ = ℱ(CS, FERMI)
    G₂_₂₁ = ℱ(CS, FERMI)
    G₂_₂₂ = ℱ(CS, FERMI)
    #
    G₃_₁₁ = ℱ(CS, FERMI)
    G₃_₁₂ = ℱ(CS, FERMI)
    G₃_₂₁ = ℱ(CS, FERMI)
    G₃_₂₂ = ℱ(CS, FERMI)
    G₃ₜₘₚ = ℱ(CS, FERMI)
    #
    G₄_₁₁ = ℱ(CS, FERMI)
    G₄_₁₂ = ℱ(CS, FERMI)
    G₄_₂₁ = ℱ(CS, FERMI)
    G₄_₂₂ = ℱ(CS, FERMI)
    #
    H₁ = fill(zero(C64), ndim1, ndim2)
    H₂ = fill(zero(C64), ndim1, ndim2)
    H₁[1,1] = 1.123
    H₁[1,2] = 0.1
    H₁[2,1] = 0.1
    H₁[2,2] = 0.567
    H₂[1,1] = 0.345
    H₂[1,2] = 0.2
    H₂[2,1] = 0.2
    H₂[2,2] = 0.876
    #
    init_green!(G₁, H₁, μ, beta, δt)
    init_green!(G₂, H₂, μ, beta, δt)
    #
    cz11 = CopyZone(1,1)
    cz12 = CopyZone(1,2)
    cz21 = CopyZone(2,1)
    cz22 = CopyZone(2,2)
    czd  = CopyZone(1,1)
    elemcpy!(cz11, G₁, czd, G₁_₁₁)
    elemcpy!(cz12, G₁, czd, G₁_₁₂)
    elemcpy!(cz21, G₁, czd, G₁_₂₁)
    elemcpy!(cz22, G₁, czd, G₁_₂₂)
    elemcpy!(cz11, G₂, czd, G₂_₁₁)
    elemcpy!(cz12, G₂, czd, G₂_₁₂)
    elemcpy!(cz21, G₂, czd, G₂_₂₁)
    elemcpy!(cz22, G₂, czd, G₂_₂₂)
    #
    I = Integrator(order)
    #
    convolution(G₃_₁₁, G₁_₁₁, G₂_₁₁, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₁ , order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₁₁, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₁₂, G₁_₁₁, G₂_₁₂, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₁₂, G₂_₂₂, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₁₂, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₂₁, G₁_₂₁, G₂_₁₁, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₁, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₂₁, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    convolution(G₃_₂₂, G₁_₂₁, G₂_₁₂, order, beta, δt)
    convolution(G₃ₜₘₚ, G₁_₂₂, G₂_₂₂, order, beta, δt)
    for tstp = 0:ntime
        incr!(G₃_₂₂, G₃ₜₘₚ, tstp, 1.0)
    end
    zeros!(G₃ₜₘₚ)
    #
    elemcpy!(czd, G₃_₁₁, cz11, G₃)
    elemcpy!(czd, G₃_₁₂, cz12, G₃)
    elemcpy!(czd, G₃_₂₁, cz21, G₃)
    elemcpy!(czd, G₃_₂₂, cz22, G₃)
    #
    convolution(G₄, G₁, G₂, order, beta, δt)
    elemcpy!(cz11, G₄, czd, G₄_₁₁)
    elemcpy!(cz12, G₄, czd, G₄_₁₂)
    elemcpy!(cz21, G₄, czd, G₄_₂₁)
    elemcpy!(cz22, G₄, czd, G₄_₂₂)
    #
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄, G₃, tstp)
    end
    @test err < ϵ
    #
    err = 0.0
    for tstp = 0:ntime
        err = err + distance(G₄_₁₁, G₃_₁₁, tstp)
        err = err + distance(G₄_₁₂, G₃_₁₂, tstp)
        err = err + distance(G₄_₂₁, G₃_₂₁, tstp)
        err = err + distance(G₄_₂₂, G₃_₂₂, tstp)
    end
    @test err < ϵ
end
