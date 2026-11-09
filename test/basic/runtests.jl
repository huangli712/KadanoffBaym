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

include("t_base.jl")

include("s_config.jl")
include("s_weights.jl")
