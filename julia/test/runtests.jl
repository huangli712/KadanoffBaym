#
# runtests.jl
#
# Unified test entry of KadanoffBaym:
#
#   KB_TEST_PROFILE=quick|release julia --startup-file=no --project=. test/runtests.jl
#
# The default profile is `quick`. Any other value is rejected.
#
# The suite contains the function-level tests only. The example-level
# validations (whole model runs against the NESSi reference data) moved
# to tutor/validate.jl (user decision, 2026-11-11): test/ must not
# depend on tutor/ and vice versa.
#

haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

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
# 2026-11-11: the model-level suites (P5, the P7 example comparisons, and
# the example-level parts of P8) moved to tutor/validate.jl. The caps
# below are upper bounds of the previous (larger) suite, so they remain
# valid; the P5 sub-budget row is dropped.
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
t_total = @elapsed begin
    @testset verbose = true "KadanoffBaym" begin
        global t_common = @elapsed include(joinpath(@__DIR__, "common", "order_gate.jl"))
        #
        # Basic tests (formerly test/basic/runtests.jl)
        #
        global t_basic = @elapsed begin
            include("t_structs.jl")
            include("t_properties.jl")
            include("t_indexing.jl")
            include("t_inout.jl")
            include("t_algebra.jl")
            include("t_traits.jl")
            include("t_convolution.jl")

            # Phase 2 tests, the wall time is reported below
            global t_p2_basic = @elapsed begin
                include("t_utilities.jl")
                include("t_differentiation.jl")
                include("t_gf_tools.jl")
                include("t_convolution_utilities.jl")
                include("t_response.jl")
            end

            # Phase 3 tests, the wall time is reported below
            global t_p3_basic = @elapsed begin
                include("t_linalg.jl")
                include("t_dyson_mat.jl")
                include("t_dyson.jl")
                include("t_equilibrium.jl")
                include("t_green_single_pole.jl")
                include("t_dyson_convergence.jl")
            end

            # Phase 4 tests, the wall time is reported below
            global t_p4_basic = @elapsed begin
                include("t_vie.jl")
                include("t_bubble.jl")
                include("t_observables.jl")
            end

            # Phase 6 tests, the wall time is reported below
            global t_p6_basic = @elapsed begin
                include("t_pseudo_structs.jl")
                include("t_pseudo_convolution.jl")
                include("t_pseudo_dyson.jl")
                include("t_pseudo_vie2.jl")
            end

            # Phase 7 tests, the wall time is reported below
            global t_p7_basic = @elapsed begin
                include("t_fft.jl")
                include("t_fourier.jl")
                include("t_matsubara_fourier.jl")
                include("t_matsubara_solvers_fourier.jl")
                include("t_ness.jl")
                include("t_ness_utils.jl")
                include("t_spectral.jl")
            end

            # Phase 8 tests, the wall time is reported below
            global t_p8_basic = @elapsed begin
                include("t_moving.jl")
                include("t_moving_algebra.jl")
                include("t_moving_bubble.jl")
                include("t_dyson_moving.jl")
                include("t_vie2_moving.jl")
                include("t_threads.jl")
            end

            include("t_base.jl")

            include("s_config.jl")
            include("s_weights.jl")
        end
        #
        # Point-wise comparison with NESSi reference data (formerly
        # test/nessi/runtests.jl). The data files in data/ are generated
        # by .nessi/kbref/generate.sh; this script consumes them only and
        # never runs the C++ generators.
        #
        global t_nessi = @elapsed begin
            @testset verbose = true "KadanoffBaym: NESSi reference" begin
                include(joinpath(@__DIR__, "common", "common.jl"))
                # The schema test is created by phase 2
                global t_p2_nessi = @elapsed include("t_kbref_schema.jl")
                include("t_weights.jl")
                include("t_convolution_reference.jl")
                # Phase 2 data are small, so both profiles consume all P2 cases
                global t_p2_nessi += @elapsed include("t_phase2_utilities.jl")
                # Phase 3: Dyson equation and equilibrium Green's functions
                global t_p3_nessi = @elapsed begin
                    include("t_phase3_dyson.jl")
                    include("t_phase3_equilibrium.jl")
                end
                # Phase 4: VIE2, bubble diagrams, and density matrix
                global t_p4_nessi = @elapsed include("t_phase4.jl")
                # Phase 6: pseudo-particle Green's functions
                global t_p6_nessi = @elapsed include("t_pseudo.jl")
                # Phase 7: Fourier transforms and steady-state Green's functions
                global t_p7_nessi = @elapsed include("t_phase7_reference.jl")
                # Phase 8: moving (truncated) solvers on closed-form inputs
                global t_p8_nessi = @elapsed include("t_moving_solvers.jl")
            end
        end
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
@testset "KadanoffBaym: time budget ($KB_PROFILE)" begin
    @test t_total ≤ KB_BUDGET[KB_PROFILE]
    haskey(KB_P2_BUDGET, KB_PROFILE) && @test t_p2 ≤ KB_P2_BUDGET[KB_PROFILE]
    haskey(KB_P3_BUDGET, KB_PROFILE) && @test t_p3 ≤ KB_P3_BUDGET[KB_PROFILE]
    haskey(KB_P4_BUDGET, KB_PROFILE) && @test t_p4 ≤ KB_P4_BUDGET[KB_PROFILE]
    haskey(KB_P6_BUDGET, KB_PROFILE) && @test t_p6 ≤ KB_P6_BUDGET[KB_PROFILE]
    haskey(KB_P7_BUDGET, KB_PROFILE) && @test t_p7 ≤ KB_P7_BUDGET[KB_PROFILE]
    haskey(KB_P8_BUDGET, KB_PROFILE) && @test t_p8 ≤ KB_P8_BUDGET[KB_PROFILE]
end
