haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset verbose = true "KadanoffBaym: traits.jl" begin
    ntime = 101
    ntau = 51
    ndim1 = 2
    ndim2 = 2
    tmax = 1.0
    beta = 10.0
    dt = 0.01
    mu = 0.0 
    ϵ = 1e-6; ϵ₁ = -0.4; ϵ₂ = 0.6; ϵ₃ = 0.435; ϵ₄ = 0.5676
    λ₁ = 0.1; λ₂ = 0.1566
    wr = 0.3
    wz = 1.0 - 0.3im
    #
    C = Cn(ntime, ntau, ndim1, ndim1, tmax, beta)
    G₁ = ℱ(C, FERMI)
    G₂ = ℱ(C, FERMI)
    G₃ = ℱ(C, FERMI)
    G₄ = ℱ(C, FERMI)

    @testset "Polynomial Interpolation Weights" begin
    end
end
