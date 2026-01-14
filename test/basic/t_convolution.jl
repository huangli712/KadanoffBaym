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

#=
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
=#

@testset verbose = true "KadanoffBaym: convolution.jl" begin
    ntime = 11
    ntau = 21
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
    #for m = 1:ntau
    #    @show m, G₂.mat[m]
    #end
    #
    #for t = 1:ntime
    #    for m = 1:t
    #        @show t, m, G₂.ret[t,m]
    #    end
    #end
    #
    #for n = 1:ntime
    #    for m = 1:ntau
    #        @show n, m, G₂.lmix[n,m]
    #    end
    #end
    #
    #for n = 1:ntime
    #    for m = 1:n
    #        @show n, m, G₂.less[m,n]
    #    end
    #end
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
#=
    for m = 1:ntau
        @show m, G₁.mat[m][1,1] - G₁_₁₁.mat[m][1,1]
    end
    for m = 1:ntau
        @show m, G₁.mat[m][1,2] - G₁_₁₂.mat[m][1,1]
    end
    for m = 1:ntau
        @show m, G₁.mat[m][2,1] - G₁_₂₁.mat[m][1,1]
    end
    for m = 1:ntau
        @show m, G₁.mat[m][2,2] - G₁_₂₂.mat[m][1,1]
    end

    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.ret[n,m][1,1] - G₁_₁₁.ret[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.ret[n,m][1,2] - G₁_₁₂.ret[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.ret[n,m][2,1] - G₁_₂₁.ret[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.ret[n,m][2,2] - G₁_₂₂.ret[n,m][1,1]
        end
    end

    for n = 1:ntime
        for m = 1:ntau
            @show n, m, G₁.lmix[n,m][1,1] - G₁_₁₁.lmix[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:ntau
            @show n, m, G₁.lmix[n,m][1,2] - G₁_₁₂.lmix[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:ntau
            @show n, m, G₁.lmix[n,m][2,1] - G₁_₂₁.lmix[n,m][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:ntau
            @show n, m, G₁.lmix[n,m][2,2] - G₁_₂₂.lmix[n,m][1,1]
        end
    end

    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.less[m,n][1,1] - G₁_₁₁.less[m,n][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.less[m,n][1,2] - G₁_₁₂.less[m,n][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.less[m,n][2,1] - G₁_₂₁.less[m,n][1,1]
        end
    end
    for n = 1:ntime
        for m = 1:n
            @show n, m, G₁.less[m,n][2,2] - G₁_₂₂.less[m,n][1,1]
        end
    end
=#
    
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

    #for m = 1:ntau
    #    @show m, G₃_₂₂.mat[m]
    #end
    #
    #for n = 1:ntime
    #    for m = 1:n
    #        @show n, m, G₃_₁₂.ret[n,m]
    #    end
    #end
    #
    #for n = 1:ntime
    #    for m = 1:ntau
    #        @show n, m, G₃_₂₁.lmix[n,m]
    #    end
    #end
    #
    #for n = 1:ntime
    #    for m = 1:n
    #        @show n, m, G₃_₂₂.less[m,n]
    #    end
    #end

end
