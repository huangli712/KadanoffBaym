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
#   P4: quick sub-budget 1.0 min, cumulative quick 12 min, release 60 min.
#   P6: quick sub-budget 0.5 min. P5 is not merged yet, so the cumulative
#       totals are those of P4 plus the P6 increments (quick 0.5 min,
#       release 15 min): quick 12.5 min, release 75 min.
#   P7: quick sub-budget 1.0 min. P5 is still not merged, the cumulative
#       totals are those of P6 plus the P7 increments (quick 1.0 min,
#       release 30 min): quick 13.5 min, release 105 min.
#   P8: quick sub-budget 1.0 min (planned 0.5 min, raised to the measured
#       value, user decision). The cumulative totals of P7 still hold.
#   P5: quick sub-budget 1.0 min, merged after P6..P8; the cumulative
#       totals are those of P8 plus the P5 increments (quick 1.0 min,
#       release 30 min): quick 14.5 min, release 135 min.
#   P2, P3, P4, and P6 contain no FFT and no full model run (not applicable).
#
const KB_BUDGET = Dict("quick" => 870.0, "release" => 8100.0)
const KB_P2_BUDGET = Dict("quick" => 420.0)
const KB_P3_BUDGET = Dict("quick" => 60.0)
const KB_P4_BUDGET = Dict("quick" => 60.0)
const KB_P6_BUDGET = Dict("quick" => 30.0)
const KB_P7_BUDGET = Dict("quick" => 60.0)
# P8 sub-budget: raised from the planned 0.5 min to the measured value
# (41.7 s on 2026-11-10) plus a margin for the machine load (user
# decision, 2026-11-10)
const KB_P8_BUDGET = Dict("quick" => 60.0)
const KB_P5_BUDGET = Dict("quick" => 60.0)

println("KB_TEST_PROFILE = ", KB_PROFILE)

t_common = 0.0
t_basic = 0.0
t_nessi = 0.0
t_p2_basic = 0.0
t_p2_nessi = 0.0
t_p3_basic = 0.0
t_p3_nessi = 0.0
t_p4_basic = 0.0
t_p4_nessi = 0.0
t_p6_basic = 0.0
t_p6_nessi = 0.0
t_p7_basic = 0.0
t_p7_nessi = 0.0
t_p8_basic = 0.0
t_p8_nessi = 0.0
t_p5_basic = 0.0
t_p5_nessi = 0.0
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
t_p4 = t_p4_basic + t_p4_nessi
@printf("Wall time of phase 4: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p4_basic, t_p4_nessi, t_p4,
        haskey(KB_P4_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P4_BUDGET[KB_PROFILE]) : "")
t_p6 = t_p6_basic + t_p6_nessi
@printf("Wall time of phase 6: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p6_basic, t_p6_nessi, t_p6,
        haskey(KB_P6_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P6_BUDGET[KB_PROFILE]) : "")
t_p7 = t_p7_basic + t_p7_nessi
@printf("Wall time of phase 7: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p7_basic, t_p7_nessi, t_p7,
        haskey(KB_P7_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P7_BUDGET[KB_PROFILE]) : "")
t_p8 = t_p8_basic + t_p8_nessi
@printf("Wall time of phase 8: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p8_basic, t_p8_nessi, t_p8,
        haskey(KB_P8_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P8_BUDGET[KB_PROFILE]) : "")
t_p5 = t_p5_basic + t_p5_nessi
@printf("Wall time of phase 5: basic %.1f s, NESSi %.1f s, total %.1f s%s\n",
        t_p5_basic, t_p5_nessi, t_p5,
        haskey(KB_P5_BUDGET, KB_PROFILE) ? @sprintf(" (budget %.0f s)", KB_P5_BUDGET[KB_PROFILE]) : "")
@testset "KadanoffBaym: time budget ($KB_PROFILE)" begin
    @test t_total ≤ KB_BUDGET[KB_PROFILE]
    haskey(KB_P2_BUDGET, KB_PROFILE) && @test t_p2 ≤ KB_P2_BUDGET[KB_PROFILE]
    haskey(KB_P3_BUDGET, KB_PROFILE) && @test t_p3 ≤ KB_P3_BUDGET[KB_PROFILE]
    haskey(KB_P4_BUDGET, KB_PROFILE) && @test t_p4 ≤ KB_P4_BUDGET[KB_PROFILE]
    haskey(KB_P6_BUDGET, KB_PROFILE) && @test t_p6 ≤ KB_P6_BUDGET[KB_PROFILE]
    haskey(KB_P7_BUDGET, KB_PROFILE) && @test t_p7 ≤ KB_P7_BUDGET[KB_PROFILE]
    haskey(KB_P8_BUDGET, KB_PROFILE) && @test t_p8 ≤ KB_P8_BUDGET[KB_PROFILE]
    haskey(KB_P5_BUDGET, KB_PROFILE) && @test t_p5 ≤ KB_P5_BUDGET[KB_PROFILE]
end
