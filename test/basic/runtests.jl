haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

include("t_structs.jl")
include("t_properties.jl")
include("t_indexing.jl")
include("t_inout.jl")
include("t_algebra.jl")
include("t_traits.jl")
include("t_convolution.jl")

# Phase 2 tests, the wall time is reported by test/runtests.jl
t_p2_basic = @elapsed begin
    include("t_utilities.jl")
    include("t_differentiation.jl")
    include("t_gf_tools.jl")
    include("t_convolution_utilities.jl")
    include("t_response.jl")
end

# Phase 3 tests, the wall time is reported by test/runtests.jl
t_p3_basic = @elapsed begin
    include("t_linalg.jl")
    include("t_dyson_mat.jl")
    include("t_dyson.jl")
    include("t_equilibrium.jl")
    include("t_green_single_pole_XX.jl")
    include("t_dyson_convergence.jl")
end

# Phase 4 tests, the wall time is reported by test/runtests.jl
t_p4_basic = @elapsed begin
    include("t_vie.jl")
    include("t_bubble.jl")
    include("t_observables.jl")
end

# Phase 6 tests, the wall time is reported by test/runtests.jl
t_p6_basic = @elapsed begin
    include("t_pseudo_structs.jl")
    include("t_pseudo_convolution.jl")
    include("t_pseudodyson.jl")
    include("t_pseudo_vie2.jl")
end

# Phase 7 tests, the wall time is reported by test/runtests.jl
t_p7_basic = @elapsed begin
    include("t_fft.jl")
    include("t_fourier.jl")
    include("t_matsubara_fourier.jl")
    include("t_matsubara_solvers_fourier.jl")
    include("t_ness.jl")
    include("t_ness_utils.jl")
    include("t_spectral.jl")
    include("t_ness_examples.jl")
end

# Phase 8 tests, the wall time is reported by test/runtests.jl
t_p8_basic = @elapsed begin
    include("t_moving.jl")
    include("t_moving_algebra.jl")
    include("t_moving_bubble.jl")
    include("t_dyson_moving.jl")
    include("t_vie2_moving.jl")
    include("t_threads.jl")
    include("t_trunc_bethe.jl")
    include("t_gw.jl")
end

# Phase 5 tests, the wall time is reported by test/runtests.jl
t_p5_basic = @elapsed begin
    include("t_p5_examples.jl")
    include("t_p5_hubbard.jl")
    include("t_p5_holstein.jl")
end

include("t_base.jl")

include("s_config.jl")
include("s_weights.jl")
