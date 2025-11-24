haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

#include("t_structs.jl")
#include("properties.jl")
#include("indexing.jl")
#include("t_inout.jl")

#include("traits.jl")

#include("s_config.jl")
include("s_weights.jl")
