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
#
# Time budgets (seconds), see the cumulative budget ledger in
# .omo/plans/00-roadmap.md. Only the cumulative totals and the sub-budget
# of the latest phase are checked here.
#
#   P2: quick sub-budget 7.0 min, cumulative quick 10 min, release 30 min.
#   P3: quick sub-budget 1.0 min, cumulative quick 11 min, release 45 min.
#   P2 and P3 contain no FFT and no full model run (not applicable).
#
const KB_BUDGET = Dict("quick" => 660.0, "release" => 2700.0)
const KB_P2_BUDGET = Dict("quick" => 420.0)
const KB_P3_BUDGET = Dict("quick" => 60.0)

println("KB_TEST_PROFILE = ", KB_PROFILE)

t_common = 0.0
t_basic = 0.0
t_nessi = 0.0
t_p2_basic = 0.0
t_p2_nessi = 0.0
t_p3_basic = 0.0
t_p3_nessi = 0.0
t_total = @elapsed begin
    @testset verbose = true "KadanoffBaym" begin
        global t_common = @elapsed include(joinpath(@__DIR__, "common", "order_gate.jl"))
        global t_basic = @elapsed include(joinpath(@__DIR__, "basic", "runtests.jl"))
        global t_nessi = @elapsed include(joinpath(@__DIR__, "nessi", "runtests.jl"))
    end
end

t_p2 = t_p2_basic + t_p2_nessi
@printf("Wall time: common %.1f s, basic %.1f s, NESSi %.1f s, total %.1f s (budget %.0f s)\n",
        t_common, t_basic, t_nessi, t_total, KB_BUDGET[KB_PROFILE])
@printf("Wall time of phase 2: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p2_basic, t_p2_nessi, t_p2,
        haskey(KB_P2_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P2_BUDGET[KB_PROFILE]) : "")
t_p3 = t_p3_basic + t_p3_nessi
@printf("Wall time of phase 3: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p3_basic, t_p3_nessi, t_p3,
        haskey(KB_P3_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P3_BUDGET[KB_PROFILE]) : "")
@testset "KadanoffBaym: time budget ($KB_PROFILE)" begin
    @test t_total ≤ KB_BUDGET[KB_PROFILE]
    haskey(KB_P2_BUDGET, KB_PROFILE) && @test t_p2 ≤ KB_P2_BUDGET[KB_PROFILE]
    haskey(KB_P3_BUDGET, KB_PROFILE) && @test t_p3 ≤ KB_P3_BUDGET[KB_PROFILE]
end
