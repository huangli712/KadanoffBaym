haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset "Cn Struct: Constructor" begin
    ntime = 201
    ntau = 1001
    ndim1 = 2
    ndim2 = 2
    tmax = 5.0
    beta = 4.0

    C₁ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    C₂ = Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
    @test  C₁ == C₂
end
