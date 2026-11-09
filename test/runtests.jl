#
# runtests.jl
#
# Unified test entry of KadanoffBaym:
#
#   KB_TEST_PROFILE=quick|release julia --startup-file=no --project=. test/runtests.jl
#
# The default profile is `quick`. Any other value is rejected.
#

using Test
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "common", "profile.jl"))

const KB_PROFILE = kb_test_profile()
const KB_BUDGET = Dict("quick" => 180.0, "release" => 900.0)

println("KB_TEST_PROFILE = ", KB_PROFILE)

t_common = 0.0
t_basic = 0.0
t_nessi = 0.0
t_total = @elapsed begin
    @testset verbose = true "KadanoffBaym" begin
        global t_common = @elapsed include(joinpath(@__DIR__, "common", "order_gate.jl"))
        global t_basic = @elapsed include(joinpath(@__DIR__, "basic", "runtests.jl"))
        global t_nessi = @elapsed include(joinpath(@__DIR__, "nessi", "runtests.jl"))
    end
end

@printf("Wall time: common %.1f s, basic %.1f s, NESSi %.1f s, total %.1f s (budget %.0f s)\n",
        t_common, t_basic, t_nessi, t_total, KB_BUDGET[KB_PROFILE])
@testset "KadanoffBaym: time budget ($KB_PROFILE)" begin
    @test t_total ≤ KB_BUDGET[KB_PROFILE]
end
