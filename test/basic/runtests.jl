haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

#include("T_structs.jl")
#include("properties.jl")
#include("indexing.jl")
#include("T_inout.jl")
#include("weights.jl")
#include("traits.jl")

include("S_config.jl")
