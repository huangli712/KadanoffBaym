#
# t_p5_quick_profiles.jl
#
# Frozen quick-profile parameters of the Phase 5 examples (plan 3.6,
# P5-W6-T11). The NESSi reference drivers .nessi/kbref/*_text.cpp use the
# same values, the quick reference data in test/nessi/data/phase5 were
# generated with them; any change requires regenerating the data.
#

using Test
using TOML

const P5Q_TUTOR = joinpath(@__DIR__, "..", "..", "tutor")
isdefined(Main, :load_example_config) ||
    include(joinpath(P5Q_TUTOR, "common", "example_config.jl"))

# file => (nt, ntau, k, mats_max_iter, bootstrap_max_iter, corrector_steps)
const P5Q_FROZEN = [
    "hubbard_chain/hubbard_chain.toml" => (10, 40, 3, 100, 50, 3),
    "fkm_bethe_quench/fkm_bethe_quench.toml" => (10, 50, 3, 100, 100, 3),
    "holstein_bethe/Holstein_bethe_Migdal.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_bethe/Holstein_bethe_uMig.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_bethe/Holstein_bethe_Nambu_Migdal.toml" => (6, 32, 3, 300, 60, 3),
    "holstein_bethe/Holstein_bethe_Nambu_uMig.toml" => (6, 32, 3, 300, 60, 3),
    "holstein_impurity/Holstein_impurity_singlebath_Migdal.toml" => (8, 40, 3, 100, 50, 3),
    "holstein_impurity/Holstein_impurity_singlebath_uMig.toml" => (8, 40, 3, 100, 50, 3),
]

@testset verbose = true "Phase 5: frozen quick profiles" begin
    for (f, v) in P5Q_FROZEN
        @testset "$f" begin
            path = joinpath(P5Q_TUTOR, f)
            raw = TOML.parsefile(path)
            cfg = load_example_config(path; profile = "quick",
                                      model_required = Tuple(keys(raw["model"])),
                                      drives = Tuple(keys(get(raw, "drive", Dict()))))
            c, s = contour_params(cfg), solver_params(cfg)
            @test (c.ntime - 1, c.ntau_points - 1, c.k) == v[1:3]
            @test (s.mats_max_iter, s.bootstrap_max_iter, s.corrector_steps) == v[4:6]
            # At least one propagation step after the start-up
            @test c.ntime - 1 ≥ c.k + 2
        end
    end
end
