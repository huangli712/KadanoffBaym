haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

#include("t_structs.jl")
#include("t_properties.jl")
#include("t_indexing.jl")
#include("t_inout.jl")
#include("t_algebra.jl")
#include("t_traits.jl")
include("t_convolution.jl")

#include("t_base.jl")

#include("s_config.jl")
#include("s_weights.jl")
